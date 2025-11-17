import numpy as np
from pathlib import Path
import wandb
from wandb.integration.sb3 import WandbCallback
from envs.diversity_selection_env import DiversitySelectionEnv
from .embeddings import build_fashion_mnist_embeddings
from stable_baselines3 import PPO
from ..eval import evaluate, evaluate_ppov1
from ..plot import plot_results
from policies.ppov1 import PPOv1, PPOv1Config
from compare_ppo_versions import SingleEnvWrapper
import argparse
import torch

def main(debug: bool = False):
    X_embed, y = build_fashion_mnist_embeddings()
    norms = np.linalg.norm(X_embed, axis=1, keepdims=True) + 1e-8
    X_unit = (X_embed / norms).astype(np.float32)
    TOTAL_TIMESTEPS = 100000 if not debug else 10000

    here = Path(__file__).resolve().parent
    models_dir = here / "model"
    models_dir.mkdir(exist_ok=True)

    max_path = models_dir / "sb3_ppo_max.zip"
    min_path = models_dir / "sb3_ppo_min.zip"

    custom_max_path = models_dir / "ppov1_max"
    custom_min_path = models_dir / "ppov1_min"

    N = len(X_embed)
    M_train = int(0.2 * N)
    env_max = DiversitySelectionEnv(X_embed=X_unit, d_metric="trace", M=M_train, seed=42)
    env_min = DiversitySelectionEnv(X_embed=X_unit, M=M_train, d_metric="trace", seed=42, maximize=False)

    # Paper baseline: single environment per agent
    env_max_v1 = SingleEnvWrapper(
        DiversitySelectionEnv(
            X_embed=X_unit,
            d_metric="trace",
            M=M_train,
            seed=42,
            maximize=True,
        )
    )
    env_min_v1 = SingleEnvWrapper(
        DiversitySelectionEnv(
            X_embed=X_unit,
            d_metric="trace",
            M=M_train,
            seed=42,
            maximize=False,
        )
    )

    if max_path.exists() and min_path.exists():
        print("Found existing models, loading from disk...")
        model_max = PPO.load(max_path, env=env_max)
        model_min = PPO.load(min_path, env=env_min)
    else:
        print("No saved models found, training from scratch...")
        run = None
        if not debug:
            # W&B run (skip in fast debug mode)
            run = wandb.init(
                project="DRL Models",
                config={
                    "dataset": "fashion_mnist",
                    "algo": "SB3-PPO",
                    "total_timesteps": TOTAL_TIMESTEPS,
                },
            )

        model_max = PPO(
            policy="MlpPolicy",
            env=env_max,
            learning_rate=3e-4,
            batch_size=64,
            n_steps=2048,
            n_epochs=10,
            gamma=1.0,
            gae_lambda=0.0,
        )

        model_min = PPO(
            policy="MlpPolicy",
            env=env_min,
            learning_rate=3e-4,
            batch_size=64,
            n_steps=2048,
            n_epochs=10,
            gamma=1.0,
            gae_lambda=0.0,
        )

        wandb_cb = WandbCallback(
            gradient_save_freq=0,
            verbose=2,
        )

        model_max.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=True, callback=wandb_cb if run else None)
        model_min.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=True, callback=wandb_cb if run else None)

        model_max.save(max_path)
        model_min.save(min_path)

        if run is not None:
            run.finish()

    cartpole_like_config = PPOv1Config(
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
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=0.03,
    )

    if custom_max_path.exists() and custom_min_path.exists():
        print("Found existing PPOv1 models, loading from disk...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config_v1 = cartpole_like_config
        ppov1_max = PPOv1(envs=env_max_v1, config=config_v1, device=device)
        ppov1_min = PPOv1(envs=env_min_v1, config=config_v1, device=device)
        ppov1_max.load(custom_max_path)
        ppov1_min.load(custom_min_path)
    else:
        print("No saved models found, training from scratch...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config_v1 = cartpole_like_config  # reuse CartPole-proven config

        # Create PPOv1 agents
        ppov1_max = PPOv1(envs=env_max_v1, config=config_v1, device=device)
        ppov1_min = PPOv1(envs=env_min_v1, config=config_v1, device=device)

        # Initial obs/flags for each env
        next_obs_max, info_max = env_max_v1.reset()
        next_obs_max = torch.tensor(next_obs_max, device=device, dtype=torch.float32)
        next_term_max = torch.zeros(env_max_v1.num_envs, device=device)
        next_trunc_max = torch.zeros(env_max_v1.num_envs, device=device)

        next_obs_min, info_min = env_min_v1.reset()
        next_obs_min = torch.tensor(next_obs_min, device=device, dtype=torch.float32)
        next_term_min = torch.zeros(env_min_v1.num_envs, device=device)
        next_trunc_min = torch.zeros(env_min_v1.num_envs, device=device)

        # How many PPO updates to reach TOTAL_TIMESTEPS?
        steps_per_update = config_v1.num_steps * env_max_v1.num_envs
        num_updates = max(1, TOTAL_TIMESTEPS // steps_per_update)

        global_step_max = 0
        global_step_min = 0

        last_max_episode = None
        last_min_episode = None

        def extract_final_episode(info_bucket):
            if not info_bucket:
                return None
            if isinstance(info_bucket, dict):
                return info_bucket
            if isinstance(info_bucket, (list, tuple)):
                for entry in reversed(info_bucket):
                    if isinstance(entry, dict) and entry:
                        return entry
            return None

        log_interval = 1 if debug else 10

        for update in range(1, num_updates + 1):
            # Optional: anneal LR
            ppov1_max.anneal_learning_rate(update, num_updates)
            ppov1_min.anneal_learning_rate(update, num_updates)

            # --- MAX agent rollout + update ---
            next_obs_max, next_term_max, next_trunc_max, info_max_rollout, global_step_max = ppov1_max.rollout(
                next_obs_max,
                next_term_max,
                next_trunc_max,
                global_step_max,
            )
            adv_max, ret_max = ppov1_max.compute_advantages_and_returns(
                next_obs_max, next_term_max, next_trunc_max
            )
            metrics_max = ppov1_max.update(adv_max, ret_max)

            # --- MIN agent rollout + update ---
            next_obs_min, next_term_min, next_trunc_min, info_min_rollout, global_step_min = ppov1_min.rollout(
                next_obs_min,
                next_term_min,
                next_trunc_min,
                global_step_min,
            )
            adv_min, ret_min = ppov1_min.compute_advantages_and_returns(
                next_obs_min, next_term_min, next_trunc_min
            )
            metrics_min = ppov1_min.update(adv_min, ret_min)

            final = extract_final_episode(info_max_rollout)
            if final:
                last_max_episode = {
                    "reward": float(final.get("cum_reward", 0.0)),
                    "diversity": float(final.get("d_metric", 0.0)),
                    "selected": int(final.get("num_selected", 0)),
                    "length": int(final.get("episode_len", 0)),
                    "truncated": bool(final.get("truncated", False)),
                }
            final = extract_final_episode(info_min_rollout)
            if final:
                last_min_episode = {
                    "reward": float(final.get("cum_reward", 0.0)),
                    "diversity": float(final.get("d_metric", 0.0)),
                    "selected": int(final.get("num_selected", 0)),
                    "length": int(final.get("episode_len", 0)),
                    "truncated": bool(final.get("truncated", False)),
                }

            # (Optional) print basic progress
            if update % log_interval == 0 or update == 1 or update == num_updates:
                max_stats_str = (
                    f"R={last_max_episode['reward']:.3f}, "
                    f"D={last_max_episode['diversity']:.3f}, "
                    f"N={last_max_episode['selected']}, "
                    f"T={last_max_episode['truncated']}, "
                    f"L={last_max_episode['length']}"
                    if last_max_episode
                    else "R=NA, D=NA"
                )
                min_stats_str = (
                    f"R={last_min_episode['reward']:.3f}, "
                    f"D={last_min_episode['diversity']:.3f}, "
                    f"N={last_min_episode['selected']}, "
                    f"T={last_min_episode['truncated']}, "
                    f"L={last_min_episode['length']}"
                    if last_min_episode
                    else "R=NA, D=NA"
                )
                print(
                    f"Update {update}/{num_updates} | "
                    f"Max EV={metrics_max['explained_variance']:.3f} "
                    f"(Inc={metrics_max['action_mean']:.2f}; {max_stats_str}) | "
                    f"Min EV={metrics_min['explained_variance']:.3f} "
                    f"(Inc={metrics_min['action_mean']:.2f}; {min_stats_str})"
                )

        # Save PPOv1 models
        ppov1_max.save(custom_max_path)
        ppov1_min.save(custom_min_path)

    size_percents = [2, 5, 10, 15, 20, 25, 30, 40, 50]
    if debug:
        size_percents = [2, 10, 30]

    with torch.no_grad():
        X_sample = torch.tensor(X_unit[:10], dtype=torch.float32, device=device)
        probs = ppov1_max.agent.get_action_probs(X_sample)  # shape [10]
    print("PPOv1 Max sample probs:", probs[:10].cpu().numpy())


    results_max, results_min, results_rand = evaluate(size_percents, N, X_unit, model_max, model_min)
    results_max_ppov1, results_min_ppov1 = evaluate_ppov1(size_percents, N, X_unit, ppov1_max, ppov1_min)
    plot_results(size_percents, results_max, results_min, results_rand, "fashion_mnist_plot.png", results_max_ppov1, results_min_ppov1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fashion-MNIST PPO comparison")
    parser.add_argument(
        "--fast-debug",
        action="store_true",
        help="Use smaller settings for quick iteration (shorter training, fewer eval points, no W&B)",
    )
    args = parser.parse_args()
    main(debug=args.fast_debug)