from dataclasses import dataclass
import gymnasium as gym


@dataclass
class EnvConfig:
    """Configuration for environment setup"""
    env_id: str = "CartPole-v1"
    seed: int = 1
    num_envs: int = 4
    capture_video: bool = True
    run_name: str = "experiment"


class EnvManager:
    """Manages the creation and setup of vectorized environments"""
    
    def __init__(self, config: EnvConfig):
        self.config = config
        self.envs = None
        
    def make_env(self, idx: int):
        """
        Creates a single environment with proper wrappers.
        
        Args:
            idx: Index of the environment (used for seeding)
            
        Returns:
            A function that creates the environment
        """
        def thunk():
            env = gym.make(self.config.env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordEpisodeStatistics(env)
            
            # Only capture video for the first environment and at specific intervals
            if self.config.capture_video and idx == 0:
                env = gym.wrappers.RecordVideo(
                    env, 
                    f"videos/{self.config.run_name}", 
                    episode_trigger=lambda episode_id: episode_id % 100 == 0
                )
            
            # Set seeds for reproducibility
            env.reset(seed=self.config.seed + idx)
            env.action_space.seed(self.config.seed + idx)
            env.observation_space.seed(self.config.seed + idx)
            
            return env
        return thunk
    
    def create_envs(self):
        """
        Creates vectorized environments.
        
        Returns:
            A SyncVectorEnv instance with all environments
        """
        self.envs = gym.vector.SyncVectorEnv([
            self.make_env(i) for i in range(self.config.num_envs)
        ])
        
        # Verify we have discrete action spaces
        assert isinstance(
            self.envs.single_action_space, gym.spaces.Discrete
        ), "Only discrete action spaces are supported"
        
        return self.envs
    
    def get_envs(self):
        """Returns the created environments"""
        if self.envs is None:
            return self.create_envs()
        return self.envs
    
    def close(self):
        """Closes the environments"""
        if self.envs is not None:
            self.envs.close()

