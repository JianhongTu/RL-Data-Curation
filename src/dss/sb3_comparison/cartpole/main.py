import numpy as np
import torch
import gymnasium as gym
from pathlib import Path
import matplotlib.pyplot as plt

from stable_baselines3 import PPO as SB3PPO
from stable_baselines3.common.env_util import make_vec_env

from policies.ppov1 import PPOv1, PPOv1Config
from compare_ppo_versions import SingleEnvWrapper


def evaluate_sb3(model, env_id, episodes=20, seed=42):
    env = gym.make(env_id)
    returns = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        total = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            done = terminated or truncated
        returns.append(total)
    env.close()
    return float(np.mean(returns)), float(np.std(returns))


def evaluate_ppov1(agent: PPOv1, env_id, device, episodes=20, seed=42):
    env = gym.make(env_id)
    returns = []
    agent.agent.eval()
    with torch.no_grad():
        for ep in range(episodes):
            obs, _ = env.reset(seed=seed + ep)
            done = False
            total = 0.0
            while not done:
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                logits = agent.agent.actor(obs_tensor)
                action = torch.argmax(logits, dim=-1).item()
                obs, reward, terminated, truncated, _ = env.step(action)
                total += reward
                done = terminated or truncated
            returns.append(total)
    agent.agent.train()
    env.close()
    return float(np.mean(returns)), float(np.std(returns))


def plot_rewards(summary, filename):
    names = list(summary.keys())
    means = [summary[name]["mean_return"] for name in names]
    stds = [summary[name]["std_return"] for name in names]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(names, means, yerr=stds, capsize=8, color=["#1f77b4", "#ff7f0e"])
    ax.set_ylabel("Average Episodic Return")
    ax.set_title("CartPole-v1: SB3 PPO vs PPOv1")
    ax.set_ylim(0, max(means) + max(stds) + 20)
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, mean + 2, f"{mean:.1f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(filename, dpi=200)
    plt.close(fig)


def main():
    env_id = "CartPole-v1"
    total_timesteps = 100_000
    here = Path(__file__).resolve().parent
    models_dir = here / "model"
    models_dir.mkdir(exist_ok=True)

    sb3_path = models_dir / "sb3_cartpole.zip"
    ppov1_path = models_dir / "ppov1_cartpole"

    # === Train or load SB3 PPO ===
    if sb3_path.exists():
        print("Loading SB3 PPO from disk...")
        sb3_model = SB3PPO.load(sb3_path)
    else:
        print("Training SB3 PPO from scratch...")
        sb3_env = make_vec_env(env_id, n_envs=4)
        sb3_model = SB3PPO(
            policy="MlpPolicy",
            env=sb3_env,
            learning_rate=2.5e-4,
            n_steps=128,
            batch_size=256,
            n_epochs=4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
        )
        sb3_model.learn(total_timesteps=total_timesteps, progress_bar=True)
        sb3_model.save(sb3_path)
        sb3_env.close()

    # === Train or load PPOv1 ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if ppov1_path.exists():
        print("Loading PPOv1 from disk...")
        env_wrapper = SingleEnvWrapper(gym.make(env_id))
        config_v1 = PPOv1Config()
        ppov1 = PPOv1(envs=env_wrapper, config=config_v1, device=device)
        ppov1.load(ppov1_path)
    else:
        print("Training PPOv1 from scratch...")
        env_wrapper = SingleEnvWrapper(gym.make(env_id))
        config_v1 = PPOv1Config(
            learning_rate=3e-4,
            num_steps=512,
            anneal_lr=True,
            gae=False,
            gamma=0.99,
            gae_lambda=0.0,
            update_epochs=4,
            minibatch_size=64,
            clip_coef=0.2,
            norm_adv=True,
            clip_vloss=True,
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
            target_kl=0.03,
        )
        ppov1 = PPOv1(envs=env_wrapper, config=config_v1, device=device)

        # Initialize rollout state
        next_obs, _ = env_wrapper.reset()
        next_obs = torch.tensor(next_obs, device=device, dtype=torch.float32)
        next_term = torch.zeros(env_wrapper.num_envs, device=device)
        next_trunc = torch.zeros(env_wrapper.num_envs, device=device)
        global_step = 0

        steps_per_update = config_v1.num_steps * env_wrapper.num_envs
        num_updates = total_timesteps // steps_per_update

        for update in range(1, num_updates + 1):
            next_obs, next_term, next_trunc, _, global_step = ppov1.rollout(
                next_obs, next_term, next_trunc, global_step
            )
            adv, ret = ppov1.compute_advantages_and_returns(next_obs, next_term, next_trunc)
            metrics = ppov1.update(adv, ret)
            if update % 10 == 0 or update == 1:
                print(
                    f"Update {update}/{num_updates} | "
                    f"EV={metrics['explained_variance']:.3f} | "
                    f"KL={metrics['approx_kl']:.5f}"
                )

        ppov1.save(ppov1_path)

    # === Evaluate both models ===
    sb3_mean, sb3_std = evaluate_sb3(sb3_model, env_id)
    ppov1_mean, ppov1_std = evaluate_ppov1(ppov1, env_id, device)

    print(f"SB3 PPO average return: {sb3_mean:.2f} ± {sb3_std:.2f}")
    print(f"PPOv1 average return: {ppov1_mean:.2f} ± {ppov1_std:.2f}")

    summary = {
        "SB3 PPO": {"mean_return": sb3_mean, "std_return": sb3_std},
        "PPOv1": {"mean_return": ppov1_mean, "std_return": ppov1_std},
    }
    plot_rewards(summary, here / "cartpole_plot.png")


if __name__ == "__main__":
    main()

