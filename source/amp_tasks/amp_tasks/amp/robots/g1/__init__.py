import gymnasium as gym

from . import rsl_rl_ppo_cfg, flat_env_cfg

##
# Register Gym environments.
##

gym.register(
    id="PPOG1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1FlatEnvCfg,
        "rsl_rl_cfg_entry_point": rsl_rl_ppo_cfg.G1FlatPPORunnerCfg,
    },
)

gym.register(
    id="AMPG1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1FlatEnvCfg,
        "rsl_rl_cfg_entry_point": rsl_rl_ppo_cfg.G1FlatAMPRunnerCfg,
    },
)
