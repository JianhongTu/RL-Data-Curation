import numpy as np
from pathlib import Path
import wandb
from wandb.integration.sb3 import WandbCallback
from envs.diversity_selection_env import DiversitySelectionEnv
from .embeddings import build_alpaca_embeddings
from stable_baselines3 import PPO
from ..eval import evaluate, evaluate_ppov1
from ..plot import plot_results
from policies.ppov1 import PPOv1, PPOv1Config
from compare_ppo_versions import SingleEnvWrapper
import argparse
import torch

def main(debug: bool = False):
    print("=" * 60)
    print("Starting Alpaca PPO Training")
    print("=" * 60)
    
    print("\n[1/6] Loading Alpaca embeddings...")
    X_unit, y = build_alpaca_embeddings()
    print(f"✓ Loaded {len(X_unit)} samples with embedding dim {X_unit.shape[1]}")
    
    TOTAL_TIMESTEPS = 100000 if not debug else 10000
    if debug:
        print("⚠️  Running in FAST DEBUG mode (reduced timesteps)")
    
    print(f"Total timesteps: {TOTAL_TIMESTEPS}")

    here = Path(__file__).resolve().parent
    models_dir = here / "model"
    models_dir.mkdir(exist_ok=True)

    max_path = models_dir / "sb3_ppo_max.zip"
    min_path = models_dir / "sb3_ppo_min.zip"

    custom_max_path = models_dir / "ppov1_max"
    custom_min_path = models_dir / "ppov1_min"

    print("\n[2/6] Setting up environments...")
    N = len(X_unit)
    M_train = int(0.2 * N)
    print(f"✓ N={N}, M_train={M_train}, metric=cosine distance")
    
    env_max = DiversitySelectionEnv(X_embed=X_unit, d_metric="cos", M=M_train, seed=42)
    env_min = DiversitySelectionEnv(X_embed=X_unit, M=M_train, d_metric="cos", seed=42, maximize=False)
    env_max_v1 = SingleEnvWrapper(DiversitySelectionEnv(
        X_embed=X_unit,
        d_metric="cos",
        M=M_train,
        seed=42,          # or 43, doesn't matter much
    ))

    env_min_v1 = SingleEnvWrapper(DiversitySelectionEnv(
        X_embed=X_unit,
        d_metric="cos",
        M=M_train,
        seed=42,
        maximize=False,
    ))

    print("\n[3/6] Training SB3 PPO models...")
    if max_path.exists() and min_path.exists():
        print("✓ Found existing SB3 models, loading from disk...")
        model_max = PPO.load(max_path, env=env_max)
        model_min = PPO.load(min_path, env=env_min)
    else:
        print("No saved SB3 models found, training from scratch...")
        # W&B run (skip in fast debug mode)
        run = None
        if not debug:
            run = wandb.init(
                project="DRL Models",
                config={
                    "dataset": "alpaca",
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

        wandb_cb = None
        if run is not None:
            wandb_cb = WandbCallback(
                gradient_save_freq=0,
                verbose=2,
            )

        print("Training MAX agent...")
        model_max.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=True, callback=wandb_cb)
        print("Training MIN agent...")
        model_min.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=True, callback=wandb_cb)

        model_max.save(max_path)
        model_min.save(min_path)
        print(f"✓ Saved SB3 models to {models_dir}")

        if run is not None:
            run.finish()

    print("\n[4/6] Training PPOv1 (paper-aligned) models...")
    
    paper_aligned_config = PPOv1Config(
        learning_rate=3e-4,
        num_steps=2048,
        anneal_lr=False,
        gae=False,
        gamma=1.0,
        gae_lambda=0.0,
        update_epochs=10,
        minibatch_size=64,
        clip_coef=0.2,
        norm_adv=True,
        clip_vloss=False,
        ent_coef=0.01,
        vf_coef=0.0,
        max_grad_norm=0.5,
        target_kl=None,
        reward_norm=True,
        use_value_function=False,
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    config_v1 = paper_aligned_config
    
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

    def train_or_load_ppov1(env, path, label):
        if path.exists():
            print(f"✓ Found existing PPOv1 {label} model, loading from disk...")
            agent = PPOv1(envs=env, config=config_v1, device=device)
            agent.load(path)
            return agent

        print(f"No saved PPOv1 {label} model, training from scratch...")
        agent = PPOv1(envs=env, config=config_v1, device=device)

        next_obs, _ = env.reset()
        next_obs = torch.tensor(next_obs, device=device, dtype=torch.float32)
        next_term = torch.zeros(env.num_envs, device=device)
        next_trunc = torch.zeros(env.num_envs, device=device)

        steps_per_update = config_v1.num_steps * env.num_envs
        num_updates = max(1, TOTAL_TIMESTEPS // steps_per_update)
        global_step = 0
        last_episode = None
        log_interval = 1 if debug else 10

        for update in range(1, num_updates + 1):
            agent.anneal_learning_rate(update, num_updates)
            next_obs, next_term, next_trunc, info_rollout, global_step = agent.rollout(
                next_obs,
                next_term,
                next_trunc,
                global_step,
            )
            adv, ret = agent.compute_advantages_and_returns(
                next_obs, next_term, next_trunc
            )
            metrics = agent.update(adv, ret)

            final = extract_final_episode(info_rollout)
            if final:
                last_episode = {
                    "reward": float(final.get("cum_reward", 0.0)),
                    "diversity": float(final.get("d_metric", 0.0)),
                    "selected": int(final.get("num_selected", 0)),
                    "length": int(final.get("episode_len", 0)),
                    "truncated": bool(final.get("truncated", False)),
                }

            if update % log_interval == 0 or update == 1 or update == num_updates:
                stats_str = (
                    f"R={last_episode['reward']:.3f}, "
                    f"D={last_episode['diversity']:.3f}, "
                    f"N={last_episode['selected']}, "
                    f"T={last_episode['truncated']}, "
                    f"L={last_episode['length']}"
                    if last_episode
                    else "R=NA, D=NA"
                )
                print(
                    f"[{label}] Update {update}/{num_updates} | "
                    f"ExplainedVar={metrics['explained_variance']:.3f} "
                    f"(Inc={metrics['action_mean']:.2f}; {stats_str})"
                )

        agent.save(path)
        print(f"✓ Saved PPOv1 {label} model to {path}")
        return agent

    ppov1_max = train_or_load_ppov1(env_max_v1, custom_max_path, "MAX")
    ppov1_min = train_or_load_ppov1(env_min_v1, custom_min_path, "MIN")
    
    if False:  # Old code block - keeping for reference
        if custom_max_path.exists() and custom_min_path.exists():
            print("Found existing PPOv1 models, loading from disk...")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            config_v1 = PPOv1Config()
            ppov1_max = PPOv1(envs=env_max_v1, config=config_v1, device=device)
            ppov1_min = PPOv1(envs=env_min_v1, config=config_v1, device=device)
            ppov1_max.load(custom_max_path)
            ppov1_min.load(custom_min_path)
        else:
            print("No saved models found, training from scratch...")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            config_v1 = PPOv1Config()  # uses the paper-aligned defaults

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
            num_updates = TOTAL_TIMESTEPS // steps_per_update

            global_step_max = 0
            global_step_min = 0

            for update in range(1, num_updates + 1):
                # Optional: anneal LR
                ppov1_max.anneal_learning_rate(update, num_updates)
                ppov1_min.anneal_learning_rate(update, num_updates)

                # --- MAX agent rollout + update ---
                next_obs_max, next_term_max, next_trunc_max, _, global_step_max = ppov1_max.rollout(
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
                next_obs_min, next_term_min, next_trunc_min, _, global_step_min = ppov1_min.rollout(
                    next_obs_min,
                    next_term_min,
                    next_trunc_min,
                    global_step_min,
                )
                adv_min, ret_min = ppov1_min.compute_advantages_and_returns(
                    next_obs_min, next_term_min, next_trunc_min
                )
                metrics_min = ppov1_min.update(adv_min, ret_min)

                # (Optional) print basic progress
                if update % 10 == 0 or update == 1:
                    print(
                        f"Update {update}/{num_updates} | "
                        f"Max EV={metrics_max['explained_variance']:.3f} | "
                        f"Min EV={metrics_min['explained_variance']:.3f}"
                    )

            # Save PPOv1 models
            ppov1_max.save(custom_max_path)
            ppov1_min.save(custom_min_path)

    print("\n[5/6] Evaluating models...")
    size_percents = [2, 5, 10, 15, 20, 25, 30, 40, 50]
    if debug:
        size_percents = [2, 10, 30]
        print(f"⚠️  Using reduced eval points: {size_percents}")

    with torch.no_grad():
        X_sample = torch.tensor(X_unit[:10], dtype=torch.float32, device=device)
        probs = ppov1_max.agent.get_action_probs(X_sample)  # shape [10]
    print("PPOv1 Max sample probs:", probs[:10].cpu().numpy())

    results_max, results_min, results_rand = evaluate(size_percents, N, X_unit, model_max, model_min)
    results_max_ppov1, results_min_ppov1 = evaluate_ppov1(size_percents, N, X_unit, ppov1_max, ppov1_min)
    
    print("\n[6/6] Generating plot...")
    plot_results(size_percents, results_max, results_min, results_rand, "alpaca_plot.png", results_max_ppov1, results_min_ppov1)
    print(f"✓ Plot saved to alpaca_plot.png")
    
    print("\n" + "=" * 60)
    print("✓ Alpaca PPO Training Complete!")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Alpaca PPO comparison")
    parser.add_argument(
        "--fast-debug",
        action="store_true",
        help="Use smaller settings for quick iteration (shorter training, fewer eval points, no W&B)",
    )
    args = parser.parse_args()
    main(debug=args.fast_debug)