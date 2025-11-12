import numpy as np
from pathlib import Path
import wandb
from wandb.integration.sb3 import WandbCallback
from envs.diversity_selection_env import DiversitySelectionEnv
from .embeddings import build_fashion_mnist_embeddings
from stable_baselines3 import PPO
from ..eval import evaluate
from ..plot import plot_results

def main():
    X_embed, y = build_fashion_mnist_embeddings()
    norms = np.linalg.norm(X_embed, axis=1, keepdims=True) + 1e-8
    X_unit = (X_embed / norms).astype(np.float32)
    TOTAL_TIMESTEPS = 100000

    here = Path(__file__).resolve().parent
    models_dir = here / "model"
    models_dir.mkdir(exist_ok=True)

    max_path = models_dir / "sb3_ppo_max"
    min_path = models_dir / "sb3_ppo_min"

    N = len(X_embed)
    M_train = int(0.2 * N)
    env_max = DiversitySelectionEnv(X_embed=X_unit, d_metric="trace", M=M_train, seed=42)
    env_min = DiversitySelectionEnv(X_embed=X_unit, M=M_train, d_metric="trace", seed=42, maximize=False)
    
    if max_path.exists() and min_path.exists():
        print("Found existing models, loading from disk...")
        model_max = PPO.load(max_path, env=env_max)
        model_min = PPO.load(min_path, env=env_min)
    else:
        print("No saved models found, training from scratch...")
        # W&B run (only when training)
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

        model_max.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=True, callback=wandb_cb)
        model_min.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=True, callback=wandb_cb)

        model_max.save(max_path)
        model_min.save(min_path)

        run.finish()

    size_percents = [2, 5, 10, 15, 20, 25, 30, 40, 50]

    results_max, results_min, results_rand = evaluate(size_percents, N, X_unit, model_max, model_min)

    plot_results(size_percents, results_max, results_min, results_rand, "fashion_mnist_plot.png")

if __name__ == "__main__":
    main()