import gymnasium as gym
from gymnasium import spaces
import numpy as np
from dataclasses import dataclass


@dataclass
class DiversityEnvConfig:
    """Configuration for DiversitySelectionEnv"""
    M: int = 10  # Target subset size
    seed: int = 42


class OnlineCovTrace:
    """
    Online computation of covariance matrix trace for diversity tracking.
    Uses Welford's online algorithm for numerical stability.
    """
    
    def __init__(self, d: int):
        """
        Initialize online covariance tracker.
        
        Args:
            d: Embedding dimension
        """
        self.emb_dim = d
        self.t = 0
        self.mean = np.zeros(self.emb_dim, dtype=np.float32)
        self.cov_mat = np.zeros((self.emb_dim, self.emb_dim), dtype=np.float32)
    
    def add(self, x: np.ndarray):
        """
        Add a new sample to the online covariance computation.
        
        Args:
            x: Sample vector of shape (d,)
        """
        x = x.astype(np.float32)
        self.t += 1
        delta = x - self.mean
        self.mean += delta / self.t
        self.cov_mat += (np.outer(delta, delta) - self.cov_mat) / self.t
    
    @property
    def trace_cov_pop(self) -> float:
        """Population covariance trace (biased estimator)"""
        return float(np.trace(self.cov_mat)) if self.t >= 2 else 0.0
    
    @property
    def trace_cov_unbiased(self) -> float:
        """Unbiased covariance trace estimator"""
        if self.t <= 1:
            return 0.0
        return float(np.trace(self.cov_mat) * self.t / (self.t - 1))
    
    def reset(self):
        """Reset the tracker to initial state"""
        self.t = 0
        self.mean = np.zeros(self.emb_dim, dtype=np.float32)
        self.cov_mat = np.zeros((self.emb_dim, self.emb_dim), dtype=np.float32)


class DiversitySelectionEnv(gym.Env):
    """
    Gymnasium environment for diversity-based data selection.
    
    The agent sequentially observes data points (embeddings) and decides whether
    to include them in a selected subset. The goal is to maximize the diversity
    (trace of covariance matrix) of the selected subset.
    
    Action space: Discrete(2)
        - 0: Exclude current candidate
        - 1: Include current candidate
    
    Observation space: Box(-inf, inf, shape=(d,))
        - Current candidate embedding vector
    
    Reward: Marginal diversity gain when including a candidate
    
    Uses OnlineCovTrace for efficient online covariance computation.
    """
    
    def __init__(self, X_embed: np.ndarray, M: int, seed: int | None = None):
        """
        Initialize the diversity selection environment.
        
        Args:
            X_embed: Embedding matrix of shape (N, d) where N is dataset size
                     and d is embedding dimension
            M: Target subset size (number of items to select)
            seed: Random seed for reproducibility
        """
        super().__init__()
        assert X_embed.ndim == 2, "X_embed must be 2D array (N, d)"
        
        self.X = X_embed.astype(np.float32)
        self.N, self.d = self.X.shape
        self.M = int(M)
        self.rng = np.random.default_rng(seed)
        
        # Define action and observation spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.d,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(2)
        
        # Episode state variables
        self.cum_reward = 0.0
        self.cur_pos = None              # Current position in pool array
        self.pool = None                 # Remaining indices to pick from
        self.selected = None             # Indices of selected data points
        self.div_tracker = None          # Online diversity tracker
        self.cur_idx = None              # Current candidate index
    
    def reset(self, seed: int | None = None, options=None):
        """
        Reset the environment to initial state.
        
        Returns:
            obs: First candidate observation
            info: Empty dict
        """
        super().reset(seed=seed)
        
        # Reset episode state
        self.pool = np.arange(self.N)
        self.selected = []
        self.div_tracker = OnlineCovTrace(d=self.d)
        self.cur_idx = None
        self.cur_pos = None
        self.cum_reward = 0.0
        
        # Sample first candidate
        obs = self._sample_next()
        if obs is None:
            obs = np.zeros(self.d, dtype=np.float32)
        
        return obs.astype(np.float32), {}
    
    def _sample_next(self):
        """
        Sample next candidate from the pool.
        
        Returns:
            Observation vector or None if pool is empty
        """
        if self.pool.size == 0:
            return None
        
        # Randomly sample from remaining pool
        self.cur_pos = int(self.rng.integers(0, self.pool.size))
        self.cur_idx = int(self.pool[self.cur_pos])
        
        return self.X[self.cur_idx]
    
    def step(self, action: int):
        """
        Execute one step in the environment.
        
        Args:
            action: 0 = exclude, 1 = include current candidate
            
        Returns:
            obs: Next observation
            reward: Marginal diversity gain (if included) or 0.0 (if excluded)
            terminated: True if M items selected or pool exhausted
            truncated: Always False
            info: Dict with episode info (only on termination)
        """
        reward = 0.0
        
        if action == 1:  # Include current candidate
            # Compute marginal diversity gain
            prev_diversity = self.div_tracker.trace_cov_unbiased
            self.div_tracker.add(self.X[self.cur_idx])
            self.selected.append(self.cur_idx)
            reward = self.div_tracker.trace_cov_unbiased - prev_diversity
            self.cum_reward += float(reward)
        
        # Remove from pool (swap with last and pop) - we've seen this item
        if self.pool.size > 0 and self.cur_pos < self.pool.size:
            last = self.pool[-1]
            self.pool[self.cur_pos] = last
            self.pool = self.pool[:-1]
        
        # Check termination: M items selected or pool exhausted
        terminated = (len(self.selected) >= self.M) or (self.pool.size == 0)
        
        if terminated:
            # Return zero observation on termination
            obs = np.zeros(self.d, dtype=np.float32)
            info = {
                "selected_idx": np.array(self.selected, dtype=np.int32),
                "final_trace_pop": self.div_tracker.trace_cov_pop,
                "final_trace_unbiased": self.div_tracker.trace_cov_unbiased,
                "cum_reward": self.cum_reward,
                "num_selected": len(self.selected),
            }
            return obs, float(reward), True, False, info
        
        # Sample next candidate
        obs = self._sample_next()
        if obs is None:
            obs = np.zeros(self.d, dtype=np.float32)
        
        return obs.astype(np.float32), float(reward), False, False, {}
