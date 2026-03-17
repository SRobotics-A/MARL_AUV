# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Fly-follow MARL environment.
"""

import gymnasium as gym

from . import agents


gym.register(
    id="Isaac-flyfollow-marl-v0",
    entry_point=f"{__name__}.marl_flyfollow_env:MARLFlyFollowEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.marl_flyfollow_env_cfg:MARLFlyFollowEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_mappo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-flyfollow-marl-river-v0",
    entry_point=f"{__name__}.marl_flyfollow_env:MARLFlyFollowEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.marl_flyfollow_env_cfg:MARLFlyFollowRiverEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_mappo_cfg.yaml",
    },
)
