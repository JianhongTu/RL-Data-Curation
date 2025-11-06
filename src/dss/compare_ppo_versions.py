"""
Compare PPO vs PPOv1 (Paper-Aligned)
=====================================
This script compares the standard PPO (main2.py) with the paper-aligned PPOv1 (main3.py)
on the same diversity selection task.
"""

import numpy as np
import torch
import time
from envs import DiversitySelectionEnv, OnlineCovTrace
from policies import PPO, PPOConfig, PPOv1, PPOv1Config


def create_synthetic_embeddings(N, d, seed=42):
    """Create synthetic embeddings for testing"""
    rng = np.random.default_rng(seed)
    K = max(3, d // 4)
    cluster_centers = rng.normal(0, 3, size=(K, d)).astype(np.float32)
    embeddings = []
    for i in range(N):
        cluster_id = i % K
        center = cluster_centers[cluster_id]
        noise = rng.normal(0, 0.5, size=d).astype(np.float32)
        embeddings.append(center + noise)
    return np.array(embeddings, dtype=np.float32)


class SingleEnvWrapper:
    """Wrapper to make single env look like vectorized env"""
    def __init__(self, env):
        self.env = env
        self.num_envs = 1
        self.single_observation_space = env.observation_space
        self.single_action_space = env.action_space
        self.episodes_completed = 0
    
    def reset(self):
        obs, info = self.env.reset()
        return np.expand_dims(obs, 0), [info]
    
    def step(self, actions):
        action = actions[0] if hasattr(actions, '__len__') else actions
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        if terminated or truncated:
            self.episodes_completed += 1
            final_info = info.copy() if info else {}
            obs, reset_info = self.env.reset()
            info = {"final_info": [final_info]}
        
        return (
            np.expand_dims(obs, 0),
            np.array([reward]),
            np.array([terminated]),
            np.array([truncated]),
            info
        )
    
    def close(self):
        pass


def train_and_evaluate(ppo_instance, env_wrapped, num_updates, device, name="PPO"):
    """Train and evaluate a PPO instance"""
    print(f"\n{'='*60}")
    print(f"Training {name}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    # Initialize
    obs1, _ = env_wrapped.reset()
    next_obs = torch.Tensor(obs1).to(device)
    next_terminated = torch.zeros(1).to(device)
    next_truncated = torch.zeros(1).to(device)
    global_step = 0
    
    episode_rewards = []
    episode_diversities = []
    
    for update in range(1, num_updates + 1):
        # Anneal LR
        ppo_instance.anneal_learning_rate(update, num_updates)
        
        # Rollout
        next_obs, next_terminated, next_truncated, episode_info, global_step = ppo_instance.rollout(
            next_obs, next_terminated, next_truncated, global_step
        )
        
        # Log episodes
        if episode_info and "final_info" in episode_info:
            final_info = episode_info["final_info"][0]
            if "cum_reward" in final_info:
                episode_rewards.append(final_info["cum_reward"])
                episode_diversities.append(final_info["final_trace_unbiased"])
        
        # Compute advantages
        advantages, returns = ppo_instance.compute_advantages_and_returns(
            next_obs, next_terminated, next_truncated
        )
        
        # Update
        metrics = ppo_instance.update(advantages, returns)
        
        if update % 5 == 0:
            avg_reward = np.mean(episode_rewards[-10:]) if episode_rewards else 0
            print(f"Update {update:3d}/{num_updates} - "
                  f"Avg Reward (last 10): {avg_reward:8.4f}, "
                  f"Episodes: {len(episode_rewards)}")
    
    training_time = time.time() - start_time
    
    results = {
        "name": name,
        "training_time": training_time,
        "episode_rewards": episode_rewards,
        "episode_diversities": episode_diversities,
        "episodes_completed": len(episode_rewards),
    }
    
    return results


def compare_inference_speed(ppo_standard, ppo_v1, X_embed, M, device):
    """Compare inference speed between PPO versions"""
    print(f"\n{'='*60}")
    print("Comparing Inference Methods")
    print(f"{'='*60}")
    
    # PPOv1 Score+Rank inference
    start = time.time()
    selected_v1, scores_v1 = ppo_v1.score_and_rank_inference(X_embed, M)
    time_v1 = time.time() - start
    
    # Compute diversity of selected subset
    tracker = OnlineCovTrace(d=X_embed.shape[1])
    for idx in selected_v1:
        tracker.add(X_embed[idx])
    div_v1 = tracker.trace_cov_unbiased
    
    print(f"\nPPOv1 Score+Rank Inference:")
    print(f"  Time: {time_v1:.3f}s")
    print(f"  Selected: {len(selected_v1)} samples")
    print(f"  Final diversity: {div_v1:.4f}")
    
    return {
        "ppov1_inference_time": time_v1,
        "ppov1_diversity": div_v1,
    }


def main():
    """Main comparison"""
    print("\n" + "="*60)
    print("PPO vs PPOv1 (Paper-Aligned) Comparison")
    print("="*60)
    
    # Setup
    N, d, M = 100, 16, 10
    seed = 42
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\nConfiguration:")
    print(f"  Dataset size (N): {N}")
    print(f"  Embedding dim (d): {d}")
    print(f"  Target subset (M): {M}")
    print(f"  Device: {device}")
    
    # Create shared embeddings
    np.random.seed(seed)
    torch.manual_seed(seed)
    X_embed = create_synthetic_embeddings(N, d, seed)
    
    # Standard PPO (main2.py defaults)
    print(f"\n{'='*60}")
    print("Standard PPO Configuration:")
    print(f"{'='*60}")
    env1 = DiversitySelectionEnv(X_embed, M, seed)
    env1_wrapped = SingleEnvWrapper(env1)
    
    ppo_config = PPOConfig(
        learning_rate=3e-4,
        num_steps=256,
        gamma=0.99,
        gae_lambda=0.95,
        update_epochs=4,
        minibatch_size=64,
    )
    print(f"  gamma: {ppo_config.gamma}")
    print(f"  gae_lambda: {ppo_config.gae_lambda}")
    print(f"  num_steps: {ppo_config.num_steps}")
    print(f"  update_epochs: {ppo_config.update_epochs}")
    print(f"  network: [64, 64]")
    
    ppo_standard = PPO(env1_wrapped, ppo_config, device)
    
    # Paper-aligned PPOv1 (main3.py)
    print(f"\n{'='*60}")
    print("PPOv1 (Paper-Aligned) Configuration:")
    print(f"{'='*60}")
    env2 = DiversitySelectionEnv(X_embed, M, seed)
    env2_wrapped = SingleEnvWrapper(env2)
    
    ppov1_config = PPOv1Config(
        learning_rate=3e-4,
        num_steps=2048,
        gamma=1.0,
        gae_lambda=0.0,
        update_epochs=10,
        minibatch_size=64,
    )
    print(f"  gamma: {ppov1_config.gamma}")
    print(f"  gae_lambda: {ppov1_config.gae_lambda}")
    print(f"  num_steps: {ppov1_config.num_steps}")
    print(f"  update_epochs: {ppov1_config.update_epochs}")
    print(f"  network: [256, 256]")
    
    ppo_v1 = PPOv1(env2_wrapped, ppov1_config, device)
    
    # Train both (small number of updates for quick comparison)
    num_updates = 10
    
    results_standard = train_and_evaluate(
        ppo_standard, env1_wrapped, num_updates, device, "Standard PPO"
    )
    
    results_v1 = train_and_evaluate(
        ppo_v1, env2_wrapped, num_updates, device, "PPOv1 (Paper-Aligned)"
    )
    
    # Compare inference
    inference_results = compare_inference_speed(ppo_standard, ppo_v1, X_embed, M, device)
    
    # Print comparison
    print(f"\n{'='*60}")
    print("Training Comparison")
    print(f"{'='*60}")
    print(f"\n{'Standard PPO':<30} {'PPOv1 (Paper)':<30}")
    print("-" * 60)
    print(f"{'Training Time:':<30} {results_standard['training_time']:.2f}s {' '*15} {results_v1['training_time']:.2f}s")
    print(f"{'Episodes Completed:':<30} {results_standard['episodes_completed']:<15} {results_v1['episodes_completed']}")
    
    if results_standard['episode_rewards'] and results_v1['episode_rewards']:
        print(f"{'Avg Episode Reward:':<30} {np.mean(results_standard['episode_rewards']):.4f}{' '*10} {np.mean(results_v1['episode_rewards']):.4f}")
        print(f"{'Avg Final Diversity:':<30} {np.mean(results_standard['episode_diversities']):.4f}{' '*10} {np.mean(results_v1['episode_diversities']):.4f}")
    
    print(f"\n{'='*60}")
    print("Key Differences")
    print(f"{'='*60}")
    print(f"{'Parameter':<25} {'Standard PPO':<20} {'PPOv1 (Paper)':<20}")
    print("-" * 65)
    print(f"{'Discount (γ)':<25} {ppo_config.gamma:<20} {ppov1_config.gamma:<20}")
    print(f"{'GAE Lambda (λ)':<25} {ppo_config.gae_lambda:<20} {ppov1_config.gae_lambda:<20}")
    print(f"{'Rollout Steps':<25} {ppo_config.num_steps:<20} {ppov1_config.num_steps:<20}")
    print(f"{'Update Epochs':<25} {ppo_config.update_epochs:<20} {ppov1_config.update_epochs:<20}")
    print(f"{'Network Size':<25} {'[64, 64]':<20} {'[256, 256]':<20}")
    
    print(f"\n{'='*60}")
    print("✅ Comparison Complete!")
    print(f"{'='*60}")
    print("\nKey Takeaways:")
    print("1. PPOv1 uses γ=1.0 (no discounting) vs standard γ=0.99")
    print("2. PPOv1 uses λ=0.0 (no GAE) vs standard λ=0.95")
    print("3. PPOv1 uses larger network [256,256] for better representations")
    print("4. PPOv1 has longer rollouts (2048) for better sample efficiency")
    print("5. PPOv1 includes Score+Rank inference for fast deployment")
    
    print(f"\nFor full training, run:")
    print(f"  Standard PPO: uv run python main2.py --total-timesteps 50000")
    print(f"  PPOv1 (Paper): uv run python main3.py --total-timesteps 50000")
    print()
    
    # Cleanup
    env1_wrapped.close()
    env2_wrapped.close()


if __name__ == "__main__":
    main()

