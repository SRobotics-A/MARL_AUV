# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import math

from MARL_mav_carry_ext.assets import FALCON_CFG  # 使用独立Falcon无人机

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectMARLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass


@configclass
class EventCfg:
    """Events for the hovering task.

    Resetting states on resets, disturbances, etc.
    """

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")


@configclass
class MARLMoveEnvCfg(DirectMARLEnvCfg):
    # control mode
    control_mode = "ACCBR"  # ACCBR or geometric
    # 动作空间参数
    # 增大最大速度，确保能追上物块(0.2m/s)，并有足够余量进行机动
    lin_vel_max = 3.0  # m/s
    ang_vel_max = 3.0  # rad/s
    # env
    decimation = 3
    episode_length_s = 60
    # Added action limits for scaling
    lin_acc_max = 5.0  # m/s^2
    # PD velocity controller gains
    vel_Kp = 3.0  # Proportional gain
    vel_Kd = 0.5  # Derivative gain

    # history of observations
    partial_obs = True  # if only local observations are used
    history_len = 3

    possible_agents = ["falcon1", "falcon2", "falcon3"]
    num_drones = len(possible_agents)

    # === 移动物块配置 ===
    num_targets = 4  # 4个移动物块
    target_colors = ["red", "yellow", "green", "blue"]
    target_values = [4.0, 3.0, 2.0, 1.0]  # 对应的价值
    target_velocity = 0.3  # 物块移动速度 (m/s)
    target_size = (0.5, 0.5, 0.5)  # 物块尺寸 (m)
    capture_distance = 1.0  # 捕获距离阈值 (m)
    sustained_follow_duration = 3.0  # 持续跟随秒数
    # 物块初始位置（地面一侧，y方向分开）
    # Spread: Targets wider apart
    target_spawn_x_range = (-6.0, -2.0)
    # Spread: Wider Y spread for dispersion
    target_spawn_y_positions = [
        6.0,
        2.0,
        -2.0,
        -6.0,
    ]
    target_spawn_z = 0.25  # z高度固定在地面上方

    # 无人机初始位置（地图另一侧）
    # Spread: Drones further apart
    drone_spawn_x_range = (-8.0, -6.0)
    drone_spawn_y_range = (-3.0, 3.0)
    drone_spawn_z_range = (1.5, 2.5)

    # === Reward Weights (Exponential Decay Style, matches hover/hover_flycart) ===
    # 所有正奖励乘 step_dt，指数衰减上界为 1.0

    # Distance Reward: w * exp(-dist * scale) * step_dt
    dist_reward_weight = 1.5
    dist_reward_scale = 0.5  # 衰减速率减小，扩大吸引范围

    # Success Reward: 持续跟随满 5 秒触发终止时的奖励
    success_reward_weight = 0.0

    # Tracking Reward: w * exp(-track_dist * scale) * step_dt
    tracking_reward_weight = 1.0
    tracking_reward_scale = 1.0

    # Action Smoothness: w * exp(-||Δaction||²) * step_dt
    action_smoothness_weight = 1.0

    # Body Rate Penalty: w * exp(-||body_rates||) * step_dt — 新增
    body_rate_penalty_weight = 2.0

    # Time Penalty: fixed penalty per step to encourage speed
    time_penalty = 0.0

    # Velocity Penalty: w * exp(-||vel||) * step_dt — 新增：抑制高速
    velocity_penalty_weight = 0.3

    # Force Penalty: w * exp(-max_thrust) * step_dt — 新增
    force_penalty_weight = 0.5

    # Upright Penalty: w * (z_dot - 1) * step_dt — 防止翻滚
    upright_penalty_weight = 2.0
    upright_expect_dir = (0.0, 0.0, 1.0)

    # Altitude Reward: w * exp(-|z - desired|) * step_dt
    height_reward_weight = 2.0  # Increased from 0.5 to 2.0
    desired_height = 2.5
    # Height Penalty (New: Strict constraint)
    height_penalty_weight = 1.0
    height_penalty_threshold = 0.5  # Meters

    # Penalties — 固定惩罚，不乘 step_dt
    drone_out_of_bounds_penalty = 1.0
    crash_penalty_scale = 1.0
    collision_penalty_scale = 1.0
    illegal_contact_penalty = 1.0
    fly_low_penalty = 1.0

    # action和observation配置
    if control_mode == "geometric":
        action_dim_geo = 12
        action_spaces = {
            "falcon1": action_dim_geo,
            "falcon2": action_dim_geo,
            "falcon3": action_dim_geo,
        }
        obs_dim_geo = 86  # 修正：实际是86维
        observation_spaces = {
            "falcon1": obs_dim_geo,
            "falcon2": obs_dim_geo,
            "falcon3": obs_dim_geo,
        }
        state_space = 86  # 全局状态86维
    elif control_mode == "ACCBR":
        action_dim_accbr = 6  # Changed from 5 to 6 to unlock Yaw Rate control
        action_spaces = {
            "falcon1": action_dim_accbr,
            "falcon2": action_dim_accbr,
            "falcon3": action_dim_accbr,
        }
        if partial_obs:
            obs_dim_accbr = 49 * history_len
        else:
            obs_dim_accbr = 49
        observation_spaces = {
            "falcon1": obs_dim_accbr,
            "falcon2": obs_dim_accbr,
            "falcon3": obs_dim_accbr,
        }
        state_space = 86  # Critic still uses full state

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=0.0033333333333333335,
        render_interval=decimation,
        gravity=(0.0, 0.0, -9.8066),
    )

    # === 独立的无人机配置（每个Agent一个Articulation） ===
    # 基础位置偏移用于三角编队初始化
    robot_0: ArticulationCfg = FALCON_CFG.replace(
        prim_path="/World/envs/env_.*/Falcon1"
    )
    robot_0.spawn.activate_contact_sensors = True
    robot_0.init_state.pos = (0.0, 2.0, 2.5)

    robot_1: ArticulationCfg = FALCON_CFG.replace(
        prim_path="/World/envs/env_.*/Falcon2"
    )
    robot_1.spawn.activate_contact_sensors = True
    robot_1.init_state.pos = (0.0, 0.0, 2.5)

    robot_2: ArticulationCfg = FALCON_CFG.replace(
        prim_path="/World/envs/env_.*/Falcon3"
    )
    robot_2.spawn.activate_contact_sensors = True
    robot_2.init_state.pos = (0.0, -2.0, 2.5)

    # === 独立的接触传感器（每个Agent一个） ===
    contact_forces_0: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Falcon1/.*",
        update_period=0.0,
        history_length=3,
        debug_vis=False,
    )
    contact_forces_1: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Falcon2/.*",
        update_period=0.0,
        history_length=3,
        debug_vis=False,
    )
    contact_forces_2: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Falcon3/.*",
        update_period=0.0,
        history_length=3,
        debug_vis=False,
    )

    # falcon CoM names (falcon.usd通常只有一个link: base_link)
    falcon_names = "base_link"
    falcon_rotor_names = "rotor_.*"

    # 终止条件阈值
    drone_collision_threshold = 0.6
    bounding_box_threshold = 12.0  # 缩小边界以避免多环境干扰 (spacing=20.0)

    # contact sensor
    contact_sensor_threshold = 1.0  # 增加接触阈值，避免过度敏感

    # low level control
    low_level_decimation: int = 1  # Reduced to 1 for maximum stability
    max_thrust_pp = 6.25  # N

    # 物块颜色映射（用于可视化）
    target_color_rgb = {
        "red": (1.0, 0.0, 0.0),
        "yellow": (1.0, 1.0, 0.0),
        "green": (0.0, 1.0, 0.0),
        "blue": (0.0, 0.0, 1.0),
    }

    # debug visualization
    debug_vis: bool = False
    if debug_vis:
        marker_cfg_goal = FRAME_MARKER_CFG.copy()
        marker_cfg_goal.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg_goal.prim_path = "/Visuals/Command/goal_pose"

        marker_cfg_body = FRAME_MARKER_CFG.copy()
        marker_cfg_body.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg_body.prim_path = "/Visuals/Command/body_pose"

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1, env_spacing=25.0, replicate_physics=True
    )

    events = EventCfg()
