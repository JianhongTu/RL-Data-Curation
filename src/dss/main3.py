"""
main3.py - Paper-Aligned PPO Training for Diversity Selection
=============================================================
Uses PPOv1 (paper-aligned) with DiversitySelectionEnv.
Supports both synthetic embeddings and real LLM embeddings.

Key differences from main2.py:
- Uses PPOv1 with gamma=1.0, gae_lambda=0.0
- Larger network architecture [256, 256]
- More update epochs (10 vs 4)
- Longer rollouts (2048 vs 256)
- Supports loading real LLM embeddings
- Implements Score+Rank inference
"""

import argparse
import os
from distutils.util import strtobool
from torch.utils.tensorboard import SummaryWriter
import time
import torch
import numpy as np
import random

from envs import DiversitySelectionEnv
from policies import PPOv1, PPOv1Config


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    
    # Experiment settings
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
                        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True,
                        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True,
                        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False,
                        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="rl-datacuration-v1",
                        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
                        help="the entity (team) of wandb's project")
    
    # Data settings
    parser.add_argument("--embeddings-path", type=str, default=None,
                        help="path to .npy file containing LLM embeddings (N, d)")
    parser.add_argument("--dataset-size", type=int, default=100,
                        help="size of synthetic dataset (N) if no embeddings provided")
    parser.add_argument("--embedding-dim", type=int, default=16,
                        help="embedding dimension (d) for synthetic data")
    parser.add_argument("--target-subset-size", type=int, default=10,
                        help="target subset size (M)")
    
    # PPO v1 algorithm settings (Paper-aligned)
    parser.add_argument("--total-timesteps", type=int, default=50000,
                        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                        help="the learning rate of the optimizer")
    parser.add_argument("--num-steps", type=int, default=2048,
                        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True,
                        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=1.0,
                        help="the discount factor gamma (Paper: 1.0)")
    parser.add_argument("--gae-lambda", type=float, default=0.0,
                        help="the lambda for GAE (Paper: 0.0 for no GAE)")
    parser.add_argument("--update-epochs", type=int, default=10,
                        help="the K epochs to update the policy (Paper: 10)")
    parser.add_argument("--minibatch-size", type=int, default=64,
                        help="the mini-batch size (Paper: 64)")
    parser.add_argument("--clip-coef", type=float, default=0.2,
                        help="the surrogate clipping coefficient")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True,
                        help="Toggles advantages normalization")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True,
                        help="Toggles whether or not to use a clipped loss for the value function")
    parser.add_argument("--ent-coef", type=float, default=0.01,
                        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
                        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
                        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
                        help="the target KL divergence threshold")
    
    # Model saving & inference
    parser.add_argument("--save-model", type=lambda x: bool(strtobool(x)), default=True,
                        help="whether to save model into the `runs/{run_name}` folder")
    parser.add_argument("--save-interval", type=int, default=10,
                        help="save model every n updates")
    parser.add_argument("--test-inference", type=lambda x: bool(strtobool(x)), default=True,
                        help="test score+rank inference after training")
    
    args = parser.parse_args()
    args.batch_size = args.num_steps
    
    return args


def setup_experiment(args):
    """Setup seeding, device, and logging"""
    # Create run name
    run_name = f"diversity-v1-{args.exp_name}-{args.seed}-{int(time.time())}"
    
    # Setup wandb tracking
    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            name=run_name,
            config=vars(args),
            monitor_gym=True,
        )
    
    # Setup tensorboard writer
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    
    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    
    return run_name, writer, device


def create_synthetic_embeddings(N, d, seed=42):
    """
    Create synthetic embeddings for testing.
    Uses clustered data to make the diversity selection problem interesting.
    """
    rng = np.random.default_rng(seed)
    
    # Create K clusters
    K = max(3, d // 4)
    cluster_centers = rng.normal(0, 3, size=(K, d)).astype(np.float32)
    
    # Assign each point to a cluster and add noise
    embeddings = []
    for i in range(N):
        cluster_id = i % K
        center = cluster_centers[cluster_id]
        noise = rng.normal(0, 0.5, size=d).astype(np.float32)
        embedding = center + noise
        embeddings.append(embedding)
    
    return np.array(embeddings, dtype=np.float32)


def load_or_create_embeddings(args):
    """Load LLM embeddings or create synthetic ones"""
    if args.embeddings_path and os.path.exists(args.embeddings_path):
        print(f"\n📂 Loading embeddings from: {args.embeddings_path}")
        X_embed = np.load(args.embeddings_path).astype(np.float32)
        print(f"  ✓ Loaded embeddings: shape {X_embed.shape}")
        return X_embed
    else:
        if args.embeddings_path:
            print(f"\n⚠ Embeddings file not found: {args.embeddings_path}")
        print(f"\n🔧 Creating synthetic embeddings...")
        print(f"  Dataset size (N): {args.dataset_size}")
        print(f"  Embedding dim (d): {args.embedding_dim}")
        X_embed = create_synthetic_embeddings(
            N=args.dataset_size,
            d=args.embedding_dim,
            seed=args.seed
        )
        return X_embed


class SingleEnvWrapper:
    """Wrapper to make single env look like vectorized env"""
    def __init__(self, env):
        self.env = env
        self.num_envs = 1
        self.single_observation_space = env.observation_space
        self.single_action_space = env.action_space
    
    def reset(self):
        obs, info = self.env.reset()
        return np.expand_dims(obs, 0), [info]
    
    def step(self, actions):
        action = actions[0] if hasattr(actions, '__len__') else actions
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Auto-reset on episode end (like VectorEnv does)
        if terminated or truncated:
            # Save final info for logging
            final_info = info.copy() if info else {}
            # Reset for next episode
            obs, reset_info = self.env.reset()
            # Return the info in "final_info" key like Gymnasium's vector envs
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


def main():
    """Main training loop"""
    # Parse arguments
    args = parse_args()
    
    # Setup experiment
    run_name, writer, device = setup_experiment(args)
    print(f"Device: {device}")
    
    # Print paper alignment info
    print("\n" + "="*60)
    print("PPO v1: Paper-Aligned Configuration")
    print("="*60)
    print(f"Discount Factor (γ):     {args.gamma} (Paper: 1.0)")
    print(f"GAE Lambda (λ):          {args.gae_lambda} (Paper: 0.0)")
    print(f"Learning Rate:           {args.learning_rate} (Paper: 3e-4)")
    print(f"Network Architecture:    [256, 256] (Paper: [256, 256])")
    print(f"Update Epochs:           {args.update_epochs} (Paper: 10)")
    print(f"Rollout Steps (n_steps): {args.num_steps} (Paper: 2048)")
    print(f"Minibatch Size:          {args.minibatch_size} (Paper: 64)")
    print("="*60 + "\n")
    
    # Load or create embeddings
    X_embed = load_or_create_embeddings(args)
    N, d = X_embed.shape
    
    print(f"  Target subset (M): {args.target_subset_size}")
    print(f"  Selection ratio: {args.target_subset_size/N:.1%}\n")
    
    # Create environment
    env = DiversitySelectionEnv(
        X_embed=X_embed,
        M=args.target_subset_size,
        seed=args.seed
    )
    
    print(f"Environment created:")
    print(f"  Observation space: {env.observation_space.shape}")
    print(f"  Action space: {env.action_space.n}")
    
    env_wrapped = SingleEnvWrapper(env)
    
    # Create PPO v1 config (Paper-aligned)
    ppo_config = PPOv1Config(
        learning_rate=args.learning_rate,
        num_steps=args.num_steps,
        anneal_lr=args.anneal_lr,
        gae=False,  # Paper: λ=0 means no GAE
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        clip_coef=args.clip_coef,
        norm_adv=args.norm_adv,
        clip_vloss=args.clip_vloss,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        target_kl=args.target_kl,
    )
    
    # Create PPO v1 trainer
    ppo = PPOv1(env_wrapped, ppo_config, device)
    print("\nAgent architecture (Paper-aligned [256, 256]):")
    print(ppo.agent)
    
    # Training loop
    global_step = 0
    start_time = time.time()
    
    # Initialize environment
    obs1, info = env_wrapped.reset()
    next_obs = torch.Tensor(obs1).to(device)
    next_terminated = torch.zeros(1).to(device)
    next_truncated = torch.zeros(1).to(device)
    
    num_updates = args.total_timesteps // args.batch_size
    
    print(f"\n{'='*60}")
    print(f"Starting training for {num_updates} updates ({args.total_timesteps} timesteps)")
    print(f"{'='*60}\n")
    
    # Track episode statistics
    episode_rewards = []
    episode_diversities = []
    
    for update in range(1, num_updates + 1):
        # Anneal learning rate
        current_lr = ppo.anneal_learning_rate(update, num_updates)
        
        # Collect rollout
        next_obs, next_terminated, next_truncated, episode_info, global_step = ppo.rollout(
            next_obs, next_terminated, next_truncated, global_step
        )
        
        # Log episode info (check for final_info like Gymnasium's vector envs)
        if episode_info and "final_info" in episode_info:
            final_info = episode_info["final_info"][0]
            if "cum_reward" in final_info:
                cum_reward = final_info["cum_reward"]
                final_diversity = final_info["final_trace_unbiased"]
                num_selected = final_info["num_selected"]
                
                episode_rewards.append(cum_reward)
                episode_diversities.append(final_diversity)
                
                print(f"Update {update:3d}/{num_updates} - "
                      f"Reward: {cum_reward:8.4f}, "
                      f"Diversity: {final_diversity:8.4f}, "
                      f"Selected: {num_selected:2d}/{args.target_subset_size}")
                
                writer.add_scalar("episode/cumulative_reward", cum_reward, global_step)
                writer.add_scalar("episode/final_diversity", final_diversity, global_step)
                writer.add_scalar("episode/num_selected", num_selected, global_step)
        
        # Compute advantages and returns
        advantages, returns = ppo.compute_advantages_and_returns(
            next_obs, next_terminated, next_truncated
        )
        
        # Update policy
        metrics = ppo.update(advantages, returns)
        
        # Log metrics
        if update % 5 == 0:
            writer.add_scalar("charts/learning_rate", current_lr, global_step)
            writer.add_scalar("losses/value_loss", metrics["value_loss"], global_step)
            writer.add_scalar("losses/policy_loss", metrics["policy_loss"], global_step)
            writer.add_scalar("losses/entropy", metrics["entropy"], global_step)
            writer.add_scalar("losses/approx_kl", metrics["approx_kl"], global_step)
            writer.add_scalar("losses/explained_variance", metrics["explained_variance"], global_step)
            
            # Calculate and log SPS
            sps = int(global_step / (time.time() - start_time))
            writer.add_scalar("charts/SPS", sps, global_step)
        
        # Save model periodically
        if args.save_model and update % args.save_interval == 0:
            model_path = f"runs/{run_name}/ppov1_update_{update}.pt"
            os.makedirs(f"runs/{run_name}", exist_ok=True)
            ppo.save(model_path)
    
    # Save final model
    if args.save_model:
        model_path = f"runs/{run_name}/ppov1_final.pt"
        os.makedirs(f"runs/{run_name}", exist_ok=True)
        ppo.save(model_path)
        print(f"\n✓ Training complete! Final model saved to {model_path}")
    
    # Print summary statistics
    if episode_rewards:
        print(f"\n{'='*60}")
        print(f"Training Summary:")
        print(f"{'='*60}")
        print(f"Total episodes: {len(episode_rewards)}")
        print(f"Average cumulative reward: {np.mean(episode_rewards):.4f} ± {np.std(episode_rewards):.4f}")
        print(f"Average final diversity: {np.mean(episode_diversities):.4f} ± {np.std(episode_diversities):.4f}")
        print(f"Best cumulative reward: {np.max(episode_rewards):.4f}")
        print(f"Best diversity achieved: {np.max(episode_diversities):.4f}")
        print(f"{'='*60}")
    
    # Test Score+Rank inference (Paper's method)
    if args.test_inference:
        print(f"\n{'='*60}")
        print("Testing Score+Rank Inference (Paper's Method)")
        print(f"{'='*60}")
        
        inference_start = time.time()
        selected_indices, scores = ppo.score_and_rank_inference(
            X_embed, 
            k=args.target_subset_size
        )
        inference_time = time.time() - inference_start
        
        print(f"✓ Inference completed in {inference_time:.3f}s")
        print(f"✓ Selected {len(selected_indices)} samples")
        print(f"✓ Score range: [{scores.min():.4f}, {scores.max():.4f}]")
        print(f"✓ Selected indices (first 10): {selected_indices[:10].tolist()}")
        
        # Compute final diversity of selected subset
        from envs import OnlineCovTrace
        tracker = OnlineCovTrace(d=d)
        for idx in selected_indices:
            tracker.add(X_embed[idx])
        final_div = tracker.trace_cov_unbiased
        
        print(f"✓ Final diversity of selected subset: {final_div:.4f}")
        print(f"{'='*60}")
        
        # Save inference results
        results_path = f"runs/{run_name}/inference_results.npz"
        np.savez(
            results_path,
            selected_indices=selected_indices,
            scores=scores,
            final_diversity=final_div,
            inference_time=inference_time
        )
        print(f"✓ Inference results saved to {results_path}\n")
    
    # Cleanup
    env_wrapped.close()
    writer.close()
    
    print(f"Total training time: {time.time() - start_time:.2f}s")
    print(f"Final SPS: {int(global_step / (time.time() - start_time))}")
    
    print(f"\n{'='*60}")
    print("Experiment Complete!")
    print(f"{'='*60}")
    print(f"Run name: {run_name}")
    print(f"Model saved: runs/{run_name}/ppov1_final.pt")
    print(f"TensorBoard logs: runs/{run_name}")
    print(f"\nView results: tensorboard --logdir=runs")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

