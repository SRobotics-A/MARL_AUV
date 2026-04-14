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
    # Run39: 速度上限贴近小车速度(0.3m/s)，速度比从5:1降到2.7:1，减少超调
    lin_vel_max = 0.8   # m/s（Run39: 1.5→0.8，减少超调，小车0.3m/s，速度比2.7:1）
    ang_vel_max = 2.0  # rad/s（Run31: 3.0→2.0，配合姿态稳定）
    # env
    decimation = 3
    episode_length_s = 60
    # Added action limits for scaling
    lin_acc_max = 1.5  # m/s^2（Run39: 3.0→1.5，配合低速上限，减少激进加速）
    # PD velocity controller gains
    vel_Kp = 2.0  # Proportional gain（Run39: 3.0→2.0，减少速度误差激进响应）
    vel_Kd = 0.3  # Derivative gain（Run39: 0.5→0.3，配合较小Kp）

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
    target_bounce_x_min = -8.0  # 折返边界：最小 x（相对 env_origin）
    target_bounce_x_max = 8.0  # 折返边界：最大 x（相对 env_origin）
    target_size = (0.5, 0.5, 0.5)  # 物块尺寸 (m)
    capture_distance = 3.0  # 捕获距离阈值 (m，XY平面)，扩大适配NovaCarter 3x实际尺寸
    sustained_follow_duration = 0.5  # 持续跟随秒数（需连续满足，条件断掉时清零）
    # 注：目标初始位置由 USD 文件决定（_reset_targets 从 USD 读取），
    # target_spawn_* 参数已删除，避免与实现不一致导致调参误判。

    # NovaCarter小车USD路径
    nova_carter_usd_path: str = (
        "/media/xtj/1CC8D044C8D01DB8/RL-download/isaac-sim/v5.1.0/Assets/Isaac/5.1/"
        "Isaac/Robots/NVIDIA/NovaCarter/Variants/nova_carter_sim_optimized.usd"
    )
    nova_carter_scale: tuple = (3.0, 3.0, 3.0)  # 与move_flyfollow.usda中一致

    # 无人机初始位置（地图另一侧）
    # Spread: Drones further apart
    drone_spawn_x_range = (-9.0, -9.0)  # Run39: 固定x=-9（调试用，消除随机起点干扰）
    drone_spawn_y_range = (0.0, 0.0)    # Run39: 固定y=0（调试用，三架无人机每次从同一圆心出发）
    drone_spawn_z_range = (2.0, 3.0)  # 抬高避开 NovaCarter 3x 缩放后车顶(≈1.6m)

    # === Reward Weights (Exponential Decay Style, matches hover/hover_flycart) ===
    # 所有正奖励乘 step_dt，指数衰减上界为 1.0

    # Distance Reward: w * exp(-dist * scale) * step_dt
    dist_reward_weight = 3.5  # Run35: 2.5→3.5，增强正向梯度
    dist_reward_scale = 0.2  # Run40: 1.5→0.2，scale越小衰减越慢，10m时exp(-2)=0.135（Run39的1.5是错误方向：exp(-15)≈0）

    # Progress Reward: w * Σ(dist_decrease * target_value) * step_dt per step
    # Run30: 设为0，baseline无此项，与姿态约束冲突导致乱飞
    progress_reward_weight = 0.0  # Run41: 1.0→0，3D距离计算导致爬高时dist_progress为负

    # Success Reward: 持续跟随满 0.5 秒触发终止时的奖励（稀疏终止奖励）
    success_reward_weight = 5.0  # Run41: 0→5.0，接通 done 后需要非零奖励信号

    # Tracking Reward: w * exp(-track_dist * scale) * step_dt
    tracking_reward_weight = 2.0  # Run35: 1.5→2.0
    tracking_reward_scale = 1.0

    # Action Smoothness: w * exp(-||Δaction||²) * step_dt
    action_smoothness_weight = 1.0  # baseline值

    # Body Rate Penalty: w * exp(-||body_rates||) * step_dt — 新增
    body_rate_penalty_weight = 0.5  # Run33: 2.0→0.5（Run21验证值，2.0与追踪冲突）

    # Time Penalty: fixed penalty per step to encourage speed
    time_penalty = 0.0

    # Velocity Penalty: w * exp(-||vel||) * step_dt — 新增：抑制高速
    velocity_penalty_weight = 0.0  # Run39: 0.3→0，分析证明0.3仍然抑制水平靠近（梯度149:1）

    # Force Penalty: w * exp(-max_thrust) * step_dt — 新增
    force_penalty_weight = 0.5

    # Upright Penalty: w * (z_dot - 1) * step_dt — 防止翻滚
    upright_penalty_weight = 0.5  # Run33: 2.0→0.5（Run21三次验证，gradient过载根因）
    upright_penalty_threshold = 0.766  # Run33: cos(40°)，恢复baseline宽松值
    upright_expect_dir = (0.0, 0.0, 1.0)

    # Altitude Reward: w * exp(-|z - desired|) * step_dt
    # Run38: 0→0，高度吸引子是无人机悬停局部最优的根因，关闭后靠 fly_low+height_penalty 约束高度
    height_reward_weight = 0.0
    desired_height = 2.5
    # Height Penalty (New: Strict constraint)
    height_penalty_weight = 2.0  # Run41: 0.5→2.0，加强上方约束（配合fly_high终止）
    height_penalty_threshold = 1.0  # Run36: 2.0→1.0，z>3.5m即触发（对称fly_low软惩罚）

    # Penalties — 固定惩罚，不乘 step_dt
    drone_out_of_bounds_penalty = 1.0
    crash_penalty_scale = 1.0
    collision_penalty_scale = 2.0
    illegal_contact_penalty = 0.05
    fly_low_penalty = 50.0  # Run41: 8.0→50.0（配合新增* step_dt，维持物理量级：50*0.01=0.5/step）
    fly_high_termination_z = 5.5  # Run41: 新增，z>5.5m终止episode，防止高飞局部最优

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
            obs_dim_accbr = 57 * history_len  # Run29: +4(assigned_target_onehot)，53→57
        else:
            obs_dim_accbr = 57  # Run29: +4(assigned_target_onehot)
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

    # === 无人机 Articulation 配置（spawn=None 模式，从 USD 绑定）===
    # prim_path 在 _setup_scene 中会被 resolve_agent_prim_path 动态覆盖
    robot_cfg: ArticulationCfg = FALCON_CFG.replace(
        prim_path="/World/envs/env_.*/falcon1"
    )
    robot_cfg.spawn.activate_contact_sensors = True

    # === 接触传感器（prim_path 在 _setup_scene 中动态填充）===
    contact_forces: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/falcon1/.*",
        update_period=0.0,
        history_length=3,
        debug_vis=False,
    )

    # falcon CoM names (falcon.usd通常只有一个link: base_link)
    falcon_names = "base_link"
    falcon_rotor_names = "rotor_.*"

    # 终止条件阈值
    drone_collision_threshold = 0.6
    bounding_box_threshold = 24.0  # Run23：放宽，给(-10,-8) spawn足够生存空间（间隙14m）
    boundary_soft_threshold = 18.0  # Run23：配合 bbox=24m，软惩罚在18m触发
    boundary_soft_penalty_weight = 2.0  # 加强：使软惩罚大于追踪收益

    # contact sensor
    contact_sensor_threshold = 50.0  # Run17：消除灾难性脉冲（中段 6.5% 数据点 <-50/ep）

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
