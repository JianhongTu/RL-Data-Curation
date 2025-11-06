import argparse
import os
from distutils.util import strtobool
from torch.utils.tensorboard import SummaryWriter
import time
import torch
import numpy as np
import random

from envs import EnvManager, EnvConfig
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
    
    # Environment settings
    parser.add_argument("--env-id", type=str, default="CartPole-v1",
                        help="the id of the environment")
    parser.add_argument("--num-envs", type=int, default=4,
                        help="the number of parallel game environments")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=True,
                        help="whether to capture videos of the agent performances")
    
    # PPO algorithm settings
    parser.add_argument("--total-timesteps", type=int, default=25000,
                        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
                        help="the learning rate of the optimizer")
    parser.add_argument("--num-steps", type=int, default=128,
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
    parser.add_argument("--minibatch-size", type=int, default=4,
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
    args.batch_size = args.num_steps * args.num_envs
    
    return args


def setup_experiment(args):
    """Setup seeding, device, and logging"""
    # Create run name
    run_name = f"{args.env_id}-{args.exp_name}-{args.seed}-{int(time.time())}"
    
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


def main():
    """Main training loop"""
    # Parse arguments
    args = parse_args()
    print("Arguments:", args)
    
    # Setup experiment
    run_name, writer, device = setup_experiment(args)
    print(f"Device: {device}")
    
    # Create environment
    env_config = EnvConfig(
        env_id=args.env_id,
        seed=args.seed,
        num_envs=args.num_envs,
        capture_video=args.capture_video,
        run_name=run_name,
    )
    env_manager = EnvManager(env_config)
    envs = env_manager.create_envs()
    
    print(f"Action space: {envs.single_action_space.n}")
    print(f"Observation space: {envs.single_observation_space.shape}")
    
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
    ppo = PPO(envs, ppo_config, device)
    print("Agent architecture:")
    print(ppo.agent)
    
    # Training loop
    global_step = 0
    start_time = time.time()
    
    # Initialize environment
    obs1, info = envs.reset()
    next_obs = torch.Tensor(obs1).to(device)
    next_terminated = torch.zeros(args.num_envs).to(device)
    next_truncated = torch.zeros(args.num_envs).to(device)
    
    num_updates = args.total_timesteps // args.batch_size
    
    print(f"\nStarting training for {num_updates} updates ({args.total_timesteps} timesteps)")
    
    for update in range(1, num_updates + 1):
        # Anneal learning rate
        current_lr = ppo.anneal_learning_rate(update, num_updates)
        
        # Collect rollout
        next_obs, next_terminated, next_truncated, episode_info, global_step = ppo.rollout(
            next_obs, next_terminated, next_truncated, global_step
        )
        
        # Log episode info
        if episode_info and "episode" in episode_info.keys():
            print(f"global_step={global_step}, episodic_return={episode_info['episode']['r']}")
            for i in range(args.num_envs):
                writer.add_scalar("charts/episodic_return", episode_info["episode"]["r"][i], global_step)
                writer.add_scalar("charts/episodic_length", episode_info["episode"]["l"][i], global_step)
        
        # Compute advantages and returns
        advantages, returns = ppo.compute_advantages_and_returns(
            next_obs, next_terminated, next_truncated
        )
        
        # Update policy
        metrics = ppo.update(advantages, returns)
        
        # Log metrics
        writer.add_scalar("charts/learning_rate", current_lr, global_step)
        writer.add_scalar("losses/value_loss", metrics["value_loss"], global_step)
        writer.add_scalar("losses/policy_loss", metrics["policy_loss"], global_step)
        writer.add_scalar("losses/entropy", metrics["entropy"], global_step)
        writer.add_scalar("losses/old_approx_kl", metrics["old_approx_kl"], global_step)
        writer.add_scalar("losses/approx_kl", metrics["approx_kl"], global_step)
        writer.add_scalar("losses/clipfrac", metrics["clipfrac"], global_step)
        writer.add_scalar("losses/explained_variance", metrics["explained_variance"], global_step)
        
        # Calculate and log SPS (steps per second)
        sps = int(global_step / (time.time() - start_time))
        print(f"Update {update}/{num_updates} - SPS: {sps}")
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
    
    # Cleanup
    env_manager.close()
    writer.close()
    
    print(f"\nTotal training time: {time.time() - start_time:.2f}s")
    print(f"Final SPS: {int(global_step / (time.time() - start_time))}")


if __name__ == "__main__":
    main()

