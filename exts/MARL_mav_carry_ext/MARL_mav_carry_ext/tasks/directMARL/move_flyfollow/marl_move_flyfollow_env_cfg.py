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
    """事件配置：用于 episode 重置时恢复场景默认状态。"""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")


@configclass
class MARLMoveEnvCfg(DirectMARLEnvCfg):
    """move_flyfollow 任务的完整配置。

    任务描述：3架 Falcon 无人机协作追捕 4个沿 x 轴匀速运动的彩色物块。
    成功条件：同时跟随至少3个目标（距离 < capture_distance）持续 sustained_follow_duration 秒。
    """

    # ===== 控制模式 =====
    # "ACCBR"：速度指令 → PD → 加速度 → 几何控制器 → INDI → 电机（推荐，更稳定）
    # "geometric"：直接输出位置/速度/加速度/jerk 设定点（实验性）
    control_mode = "ACCBR"

    # ===== 动作空间参数 =====
    # lin_vel_max 需大于 target_velocity（0.3），否则无人机永远追不上目标
    lin_vel_max = 3.0    # 最大线速度指令（m/s），policy 输出 [-1,1] × lin_vel_max
    ang_vel_max = 3.0    # 最大角速率指令（rad/s）
    lin_acc_max = 5.0    # PD 控制器加速度上限（m/s²），防止指令过大导致仿真不稳

    # PD 速度控制器增益（ACCBR 模式）
    # vel_error → acc = Kp * error + Kd * d(error)/dt
    vel_Kp = 3.0   # 比例增益
    vel_Kd = 0.5   # 微分增益（防抖，但可能放大噪声）

    # ===== 仿真时间设置 =====
    decimation = 3          # 高级控制频率 = physics_freq / decimation = 300/3 = 100 Hz
    episode_length_s = 60   # 每个 episode 最大时长（秒）

    # ===== 历史观测配置 =====
    partial_obs = True    # True：仅使用局部观测（分散执行）；False：全局观测
    history_len = 3       # 历史帧数；obs 维度 = obs_dim_per_step × history_len

    # ===== 智能体配置 =====
    possible_agents = ["falcon1", "falcon2", "falcon3"]
    num_drones = len(possible_agents)  # 3架无人机

    # ===== 移动目标配置 =====
    num_targets = 4           # 目标物块数量（比无人机多1个，鼓励协作分工）
    target_colors = ["red", "yellow", "green", "blue"]
    # 目标价值（高价值目标优先级更高，体现在距离奖励的加权上）
    target_values = [4.0, 3.0, 2.0, 1.0]
    target_velocity = 0.3     # 物块沿 x 轴正方向的匀速（m/s）
    target_size = (0.5, 0.5, 0.5)    # 物块尺寸（m），用于可视化和碰撞形状
    capture_distance = 1.0    # 捕获/跟随判定距离（m）：无人机 xy 距离 < 此值视为跟随
    sustained_follow_duration = 3.0  # 成功终止需持续跟随的时间（秒）

    # 目标初始位置范围
    target_spawn_x_range = (-6.0, -2.0)  # x：随机，制造追捕距离差异
    # y：固定分层，使4个目标在 y 方向等间距分布（鼓励无人机分散覆盖）
    target_spawn_y_positions = [
        6.0,    # 目标0（红，价值4）
        2.0,    # 目标1（黄，价值3）
        -2.0,   # 目标2（绿，价值2）
        -6.0,   # 目标3（蓝，价值1）
    ]
    target_spawn_z = 0.25   # z：地面高度，物块贴地滑动

    # 无人机初始位置范围（在目标左侧，给追捕留出初始距离）
    drone_spawn_x_range = (-8.0, -6.0)  # x：比目标更靠左
    drone_spawn_y_range = (-3.0, 3.0)   # y：居中分布
    drone_spawn_z_range = (1.5, 2.5)    # z：空中悬停高度

    # ===== 奖励权重（指数衰减风格）=====
    # 设计原则：正奖励均乘 step_dt，使其单位为"奖励/秒"，量纲统一

    # 距离奖励：w * exp(-dist × scale) × value × dt
    # scale 越小，奖励梯度覆盖范围越广（吸引盆地更宽）
    dist_reward_weight = 1.5
    dist_reward_scale = 0.5   # 较小值：远距离也有梯度，避免无人机卡在局部最优

    # 成功奖励（当前已移除，设为0）
    success_reward_weight = 0.0

    # 跟随奖励：仅在捕获状态下（dist < capture_distance）给予
    # 与距离奖励叠加，增强进入捕获区后的保持动机
    tracking_reward_weight = 1.0
    tracking_reward_scale = 1.0

    # 动作平滑度：exp(-||Δaction||²)，鼓励连续平滑的控制输出
    action_smoothness_weight = 1.0

    # 机体角速率：exp(-||ω||)，抑制过激角运动（大角速率 = 不稳定飞行）
    body_rate_penalty_weight = 2.0

    # 时间惩罚（已禁用）：可用于鼓励快速完成任务
    time_penalty = 0.0

    # 速度惩罚：exp(-||v||)，抑制高速飞行（影响精确跟随）
    velocity_penalty_weight = 0.3

    # 推力惩罚：exp(-max_thrust_normalized)，抑制高能耗悬停
    force_penalty_weight = 0.5

    # 竖直姿态：w * (R_zz - 1.0) × dt，R_zz = 机体 z 轴与世界 z 轴夹角余弦
    # 完全竖直时 R_zz=1（奖励=0），翻滚时 R_zz=-1（奖励=-2×w×dt）
    upright_penalty_weight = 2.0
    upright_expect_dir = (0.0, 0.0, 1.0)  # 期望机体上方向 = 世界 z 轴

    # 高度奖励：鼓励维持在 desired_height 附近飞行
    height_reward_weight = 2.0
    desired_height = 2.5   # 期望飞行高度（m），需保持在目标上方以便俯视追踪

    # 高度超限惩罚（线性）：|z - desired| > threshold 时额外惩罚
    height_penalty_weight = 1.0
    height_penalty_threshold = 0.5   # 容忍偏差（m）

    # 安全惩罚（固定值，不乘 dt）
    drone_out_of_bounds_penalty = 1.0  # 单次出界惩罚
    crash_penalty_scale = 1.0          # 坠机惩罚（未使用，保留接口）
    collision_penalty_scale = 1.0      # 无人机碰撞惩罚（per collision pair）
    illegal_contact_penalty = 1.0      # 非法接触惩罚
    fly_low_penalty = 1.0              # 飞低惩罚

    # ===== 动作/观测空间（在 cfg 类内直接计算）=====
    if control_mode == "geometric":
        action_dim_geo = 12  # pos(3) + vel(3) + acc(3) + jerk(3)
        action_spaces = {
            "falcon1": action_dim_geo,
            "falcon2": action_dim_geo,
            "falcon3": action_dim_geo,
        }
        obs_dim_geo = 86
        observation_spaces = {
            "falcon1": obs_dim_geo,
            "falcon2": obs_dim_geo,
            "falcon3": obs_dim_geo,
        }
        state_space = 86  # Critic 全局状态 86 维
    elif control_mode == "ACCBR":
        # 6维动作：vel_cmd(3) + body_rates(3)
        action_dim_accbr = 6
        action_spaces = {
            "falcon1": action_dim_accbr,
            "falcon2": action_dim_accbr,
            "falcon3": action_dim_accbr,
        }
        # 单步观测 49 维：pos(3)+vel(3)+rot_mat(9)+other_drones(6)+targets(20)+dist(4)+closest(4)
        # 历史拼接后：49 × history_len = 49 × 3 = 147 维
        if partial_obs:
            obs_dim_accbr = 49 * history_len
        else:
            obs_dim_accbr = 49
        observation_spaces = {
            "falcon1": obs_dim_accbr,
            "falcon2": obs_dim_accbr,
            "falcon3": obs_dim_accbr,
        }
        # Critic 仍然使用全局状态（86维），与观测维度无关
        # 86 = pos(9)+rot(27)+vel(9)+ang_vel(9)+t_pos(12)+t_vel(12)+captured(4)+values(4)
        state_space = 86

    # ===== 仿真配置 =====
    # dt = 1/300 ≈ 3.33ms（物理步长）；decimation=3 → 高级控制步长 = 10ms（100Hz）
    sim: SimulationCfg = SimulationCfg(
        dt=0.0033333333333333335,
        render_interval=decimation,
        gravity=(0.0, 0.0, -9.8066),
    )

    # ===== 无人机 Articulation 配置（独立对象，避免多智能体索引混乱）=====
    # init_state.pos 仅作为占位符，实际位置在 _reset_idx 中随机化
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

    # ===== 接触传感器（每架无人机独立，检测与地面/障碍物的接触）=====
    # update_period=0.0：每物理步更新；history_length=3：保留3帧历史
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

    # 刚体名称（用于 find_bodies 查询索引）
    falcon_names = "base_link"       # 主体链接名
    falcon_rotor_names = "rotor_.*"  # 4个旋翼链接名（正则匹配）

    # ===== 终止条件阈值 =====
    drone_collision_threshold = 0.6   # 两无人机中心距离 < 0.6m 触发碰撞终止
    bounding_box_threshold = 12.0     # 超出 ±12m 范围终止；env_spacing=25m，留 1m 间距

    # 接触传感器触发阈值（N）：过小易误触发（如旋转时的气动力），过大漏检碰撞
    contact_sensor_threshold = 1.0

    # ===== 低级控制参数 =====
    low_level_decimation: int = 1   # 内环控制抽取率（1 = 每物理步执行一次，最高频率）
    max_thrust_pp = 6.25            # 单个旋翼最大推力（N）；Falcon 悬停约需 4×3.0N

    # ===== 目标颜色映射（RGB 浮点）=====
    target_color_rgb = {
        "red":    (1.0, 0.0, 0.0),   # 高价值目标（4分）
        "yellow": (1.0, 1.0, 0.0),   # 次高价值（3分）
        "green":  (0.0, 1.0, 0.0),   # 中等价值（2分）
        "blue":   (0.0, 0.0, 1.0),   # 低价值（1分）
    }

    # ===== Debug 可视化 =====
    debug_vis: bool = False
    if debug_vis:
        marker_cfg_goal = FRAME_MARKER_CFG.copy()
        marker_cfg_goal.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg_goal.prim_path = "/Visuals/Command/goal_pose"

        marker_cfg_body = FRAME_MARKER_CFG.copy()
        marker_cfg_body.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg_body.prim_path = "/Visuals/Command/body_pose"

    # ===== 场景配置 =====
    # env_spacing=25.0：相邻环境间距 25m，远大于 bounding_box_threshold(12m)，防止跨 env 干扰
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1, env_spacing=25.0, replicate_physics=True
    )

    events = EventCfg()
