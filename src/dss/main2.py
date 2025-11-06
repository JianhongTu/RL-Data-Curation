import argparse
import os
from distutils.util import strtobool
from torch.utils.tensorboard import SummaryWriter
import time
import torch
import numpy as np
import random

from envs import DiversitySelectionEnv
from policies import PPO, PPOConfig


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
    parser.add_argument("--wandb-project-name", type=str, default="rl-datacuration",
                        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
                        help="the entity (team) of wandb's project")
    
    # Diversity environment settings
    parser.add_argument("--dataset-size", type=int, default=100,
                        help="size of the dataset (N)")
    parser.add_argument("--embedding-dim", type=int, default=16,
                        help="embedding dimension (d)")
    parser.add_argument("--target-subset-size", type=int, default=10,
                        help="target subset size (M)")
    
    # PPO algorithm settings
    parser.add_argument("--total-timesteps", type=int, default=50000,
                        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                        help="the learning rate of the optimizer")
    parser.add_argument("--num-steps", type=int, default=256,
                        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True,
                        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True,
                        help="Use GAE for advantage computation")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="the lambda for the general advantage estimation")
    parser.add_argument("--update-epochs", type=int, default=4,
                        help="the K epochs to update the policy")
    parser.add_argument("--minibatch-size", type=int, default=64,
                        help="the mini-batch size")
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
    
    # Model saving
    parser.add_argument("--save-model", type=lambda x: bool(strtobool(x)), default=True,
                        help="whether to save model into the `runs/{run_name}` folder")
    parser.add_argument("--save-interval", type=int, default=10,
                        help="save model every n updates")
    
    args = parser.parse_args()
    args.batch_size = args.num_steps
    
    return args


def setup_experiment(args):
    """Setup seeding, device, and logging"""
    # Create run name
    run_name = f"diversity-{args.exp_name}-{args.seed}-{int(time.time())}"
    
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


def main():
    """Main training loop"""
    # Parse arguments
    args = parse_args()
    print("Arguments:", args)
    
    # Setup experiment
    run_name, writer, device = setup_experiment(args)
    print(f"Device: {device}")
    
    # Create synthetic embeddings
    print(f"\nCreating synthetic dataset...")
    print(f"  Dataset size (N): {args.dataset_size}")
    print(f"  Embedding dim (d): {args.embedding_dim}")
    print(f"  Target subset (M): {args.target_subset_size}")
    
    X_embed = create_synthetic_embeddings(
        N=args.dataset_size,
        d=args.embedding_dim,
        seed=args.seed
    )
    
    # Create single environment (no vectorization for this task)
    env = DiversitySelectionEnv(
        X_embed=X_embed,
        M=args.target_subset_size,
        seed=args.seed
    )
    
    print(f"\nEnvironment created:")
    print(f"  Observation space: {env.observation_space.shape}")
    print(f"  Action space: {env.action_space.n}")
    
    # Wrap in a dummy "vectorized" wrapper for PPO compatibility
    # (PPO expects vectorized envs, but we only use 1)
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
    
    env_wrapped = SingleEnvWrapper(env)
    
    # Create PPO config
    ppo_config = PPOConfig(
        learning_rate=args.learning_rate,
        num_steps=args.num_steps,
        anneal_lr=args.anneal_lr,
        gae=args.gae,
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
    
    # Create PPO trainer
    ppo = PPO(env_wrapped, ppo_config, device)
    print("\nAgent architecture:")
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
    
    print(f"\nStarting training for {num_updates} updates ({args.total_timesteps} timesteps)")
    
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
                
                print(f"Update {update}/{num_updates} - "
                      f"Reward: {cum_reward:.4f}, "
                      f"Diversity: {final_diversity:.4f}, "
                      f"Selected: {num_selected}/{args.target_subset_size}")
                
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
        if update % 10 == 0:
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
            model_path = f"runs/{run_name}/ppo_update_{update}.pt"
            os.makedirs(f"runs/{run_name}", exist_ok=True)
            ppo.save(model_path)
    
    # Save final model
    if args.save_model:
        model_path = f"runs/{run_name}/ppo_final.pt"
        os.makedirs(f"runs/{run_name}", exist_ok=True)
        ppo.save(model_path)
        print(f"\nTraining complete! Final model saved to {model_path}")
    
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
    
    # Cleanup
    env_wrapped.close()
    writer.close()
    
    print(f"\nTotal training time: {time.time() - start_time:.2f}s")
    print(f"Final SPS: {int(global_step / (time.time() - start_time))}")


if __name__ == "__main__":
    main()

