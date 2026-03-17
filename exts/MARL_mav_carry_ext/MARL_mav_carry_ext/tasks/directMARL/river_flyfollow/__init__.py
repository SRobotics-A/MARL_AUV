"""
River Fly-follow MARL environment.
"""

import gymnasium as gym

from . import agents


gym.register(
    id="Isaac-flyfollow-marl-river-v0",
    entry_point=f"{__name__}.marl_riverflyfollow_env:MARLRiverFlyFollowEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.marl_riverflyfollow_env_cfg:MARLRiverFlyFollowEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_mappo_cfg.yaml",
    },
)
