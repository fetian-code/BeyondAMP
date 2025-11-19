import torch
from .vecenv_wrapper import RslRlVecEnvWrapper


class WorldModelEnvWrapper(RslRlVecEnvWrapper):
    def __init__(self, env, clip_actions = None):
        super().__init__(env, clip_actions)
        self.rewards_shape = self.unwrapped.reward_manager._step_reward.shape[-1]
    
    def get_observations(self) -> tuple[torch.Tensor, dict]:
        """Returns the current observations of the environment."""
        if hasattr(self.unwrapped, "observation_manager"):
            obs_dict = self.unwrapped.observation_manager.compute()
        else:
            obs_dict = self.unwrapped._get_observations()
        return obs_dict["policy"], {"observations": obs_dict}
    
    def get_amp_observations(self) -> tuple[torch.Tensor, dict]:
        """Returns the current observations of the environment."""
        if hasattr(self.unwrapped, "observation_manager"):
            obs_dict = self.unwrapped.observation_manager.compute()
        else:
            obs_dict = self.unwrapped._get_observations()
        return obs_dict["amp"]
    