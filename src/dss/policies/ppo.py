from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from typing import Optional
import gymnasium as gym


@dataclass
class PPOConfig:
    """Configuration for PPO algorithm"""
    learning_rate: float = 2.5e-4
    num_steps: int = 128
    anneal_lr: bool = True
    gae: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    update_epochs: int = 4
    minibatch_size: int = 4
    clip_coef: float = 0.2
    norm_adv: bool = True
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = None
    

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Initialize layers with orthogonal initialization"""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    """PPO Agent with actor-critic architecture"""
    
    def __init__(self, envs):
        super().__init__()
        obs_shape = np.array(envs.single_observation_space.shape).prod()
        n_actions = envs.single_action_space.n
        
        # Critic network (value function)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_shape, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        
        # Actor network (policy)
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_shape, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, n_actions), std=0.01),
        )
    
    def get_value(self, x):
        """Get value estimate from critic"""
        return self.critic(x)
    
    def get_action(self, x):
        """Get action logits from actor"""
        return self.actor(x)
    
    def get_action_and_value(self, x, action=None):
        """
        Get action, log probability, entropy, and value.
        If action is None, sample a new action.
        """
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.get_value(x)


class PPO:
    """PPO algorithm implementation"""
    
    def __init__(self, envs, config: PPOConfig, device):
        self.envs = envs
        self.config = config
        self.device = device
        
        # Calculate batch size
        self.num_envs = envs.num_envs
        self.batch_size = config.num_steps * self.num_envs
        
        # Create agent and optimizer
        self.agent = Agent(envs).to(device)
        self.optimizer = optim.Adam(
            self.agent.parameters(), 
            lr=config.learning_rate, 
            eps=1e-5
        )
        
        # Storage setup
        obs_shape = envs.single_observation_space.shape
        self.obs = torch.zeros(config.num_steps, self.num_envs, *obs_shape).to(device)
        self.actions = torch.zeros((config.num_steps, self.num_envs)).to(device)
        self.logprobs = torch.zeros((config.num_steps, self.num_envs)).to(device)
        self.rewards = torch.zeros((config.num_steps, self.num_envs)).to(device)
        self.terminateds = torch.zeros((config.num_steps, self.num_envs)).to(device)
        self.truncateds = torch.zeros((config.num_steps, self.num_envs)).to(device)
        self.values = torch.zeros((config.num_steps, self.num_envs)).to(device)
        
    def rollout(self, next_obs, next_terminated, next_truncated, global_step):
        """
        Collect rollout data from the environment.
        
        Returns:
            Updated next_obs, next_terminated, next_truncated, episode_info, new_global_step
        """
        episode_info = {}
        
        for step in range(self.config.num_steps):
            global_step += self.num_envs
            self.obs[step] = next_obs
            self.terminateds[step] = next_terminated
            self.truncateds[step] = next_truncated
            
            # Get action from policy
            with torch.no_grad():
                action, logprob, _, value = self.agent.get_action_and_value(next_obs)
                self.values[step] = value.flatten()
            self.actions[step] = action
            self.logprobs[step] = logprob
            
            # Step the environment
            next_obs, reward, terminated, truncated, info = self.envs.step(action.cpu().numpy())
            self.rewards[step] = torch.tensor(reward).to(self.device).view(-1)
            next_obs = torch.Tensor(next_obs).to(self.device)
            next_terminated = torch.Tensor(terminated).to(self.device)
            next_truncated = torch.Tensor(truncated).to(self.device)
            
            # Log episode info if available
            if "episode" in info.keys():
                episode_info = info
        
        return next_obs, next_terminated, next_truncated, episode_info, global_step
    
    def compute_advantages_and_returns(self, next_obs, next_terminated, next_truncated):
        """
        Compute advantages and returns using GAE or simple returns.
        
        Returns:
            advantages, returns
        """
        with torch.no_grad():
            next_value = self.agent.get_value(next_obs).reshape(1, -1)
            
            if self.config.gae:
                # Generalized Advantage Estimation
                advantages = torch.zeros_like(self.rewards).to(self.device)
                lastgaelam = 0
                for t in reversed(range(self.config.num_steps)):
                    if t == self.config.num_steps - 1:
                        nextnonterminal = 1.0 - torch.logical_or(
                            next_terminated, next_truncated
                        ).float()
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - torch.logical_or(
                            self.terminateds[t + 1], self.truncateds[t + 1]
                        ).float()
                        nextvalues = self.values[t + 1]
                    delta = self.rewards[t] + self.config.gamma * nextvalues * nextnonterminal - self.values[t]
                    advantages[t] = lastgaelam = delta + self.config.gamma * self.config.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + self.values
            else:
                # Simple returns
                returns = torch.zeros_like(self.rewards).to(self.device)
                for t in reversed(range(self.config.num_steps)):
                    if t == self.config.num_steps - 1:
                        nextnonterminal = 1.0 - torch.logical_or(
                            next_terminated, next_truncated
                        ).float()
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - torch.logical_or(
                            self.terminateds[t + 1], self.truncateds[t + 1]
                        ).float()
                        next_return = returns[t + 1]
                    returns[t] = self.rewards[t] + self.config.gamma * nextnonterminal * next_return
                advantages = returns - self.values
        
        return advantages, returns
    
    def update(self, advantages, returns):
        """
        Update the policy and value networks.
        
        Returns:
            Dictionary with training metrics
        """
        # Flatten the batch
        b_obs = self.obs.reshape((-1,) + self.envs.single_observation_space.shape)
        b_logprobs = self.logprobs.reshape(-1)
        b_actions = self.actions.reshape((-1,) + self.envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = self.values.reshape(-1)
        
        # Optimization
        b_inds = np.arange(self.batch_size)
        clipfracs = []
        
        for epoch in range(self.config.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, self.batch_size, self.config.minibatch_size):
                end = start + self.config.minibatch_size
                mb_inds = b_inds[start:end]
                
                _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(
                    b_obs[mb_inds], b_actions.long()[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                
                with torch.no_grad():
                    # Calculate approx KL divergence
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > self.config.clip_coef).float().mean().item()]
                
                mb_advantages = b_advantages[mb_inds]
                if self.config.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                
                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - self.config.clip_coef, 1 + self.config.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                # Value loss
                newvalue = newvalue.view(-1)
                if self.config.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -self.config.clip_coef,
                        self.config.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                
                entropy_loss = entropy.mean()
                loss = pg_loss - self.config.ent_coef * entropy_loss + v_loss * self.config.vf_coef
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
            
            # Early stopping based on KL divergence
            if self.config.target_kl is not None:
                if approx_kl > self.config.target_kl:
                    break
        
        # Calculate explained variance
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        
        return {
            "value_loss": v_loss.item(),
            "policy_loss": pg_loss.item(),
            "entropy": entropy_loss.item(),
            "old_approx_kl": old_approx_kl.item(),
            "approx_kl": approx_kl.item(),
            "clipfrac": np.mean(clipfracs),
            "explained_variance": explained_var,
        }
    
    def anneal_learning_rate(self, update, num_updates):
        """Anneal the learning rate linearly"""
        if self.config.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * self.config.learning_rate
            self.optimizer.param_groups[0]["lr"] = lrnow
            return lrnow
        return self.config.learning_rate
    
    def save(self, path):
        """Save the agent's state dict"""
        torch.save({
            'agent_state_dict': self.agent.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path):
        """Load the agent's state dict"""
        checkpoint = torch.load(path, weights_only=False)
        self.agent.load_state_dict(checkpoint['agent_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model loaded from {path}")

