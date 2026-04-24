"""Single-drone fly-forward task (PPO default, DDPG preserved)."""

import gymnasium as gym

from . import agents


gym.register(
    id="Isaac-fly-forward-v0",
    entry_point=f"{__name__}.fly_forward_env:FlyForwardEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fly_forward_env_cfg:FlyForwardEnvCfg",
        # PPO 的 yaml 在 train.py 中以 "skrl_cfg_entry_point" 读取（非 _ppo_）
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "skrl_ddpg_cfg_entry_point": f"{agents.__name__}:skrl_ddpg_cfg.yaml",
    },
)
