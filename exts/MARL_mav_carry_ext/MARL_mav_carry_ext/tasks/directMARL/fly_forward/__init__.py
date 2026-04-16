"""Single-drone fly-forward task (DDPG)."""

import gymnasium as gym

from . import agents


gym.register(
    id="Isaac-fly-forward-v0",
    entry_point=f"{__name__}.fly_forward_env:FlyForwardEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fly_forward_env_cfg:FlyForwardEnvCfg",
        "skrl_ddpg_cfg_entry_point": f"{agents.__name__}:skrl_ddpg_cfg.yaml",
    },
)
