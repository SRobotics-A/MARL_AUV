# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from typing import Sequence

# 导入单机 Falcon 机器人配置
from MARL_mav_carry_ext.assets import FALCON_CFG

# 导入IsaacLab相关模块
import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectMARLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

@configclass
class EventCfg:
    """跟随任务的事件配置类
    
    定义环境重置时的各种事件，包括状态重置、扰动等
    """

    # 重置整个场景到默认状态
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # 重置机器人的基础状态（位置和姿态）
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            # 位置范围配置
            "pose_range": {
                "x": (-1.0, 1.0),      # x坐标范围：-1到1米（相对起始位置）
                "y": (-1.0, 1.0),      # y坐标范围：-1到1米
                "z": (1.5, 2.5),       # z坐标范围：1.5到2.5米（起始高度）
                "roll": (-0.0, 0.0),    # 滚转角：保持水平
                "pitch": (-0.0, 0.0),   # 俯仰角：保持水平
                "yaw": (-math.pi, math.pi),  # 偏航角：任意朝向
            },
            # 速度范围配置（初始速度为0）
            "velocity_range": {
                "x": (-0.0, 0.0),      # x方向速度：0（静止）
                "y": (-0.0, 0.0),      # y方向速度：0（静止）
                "z": (-0.0, 0.0),      # z方向速度：0（静止）
                "roll": (-0.0, 0.0),   # 滚转角速度：0
                "pitch": (-0.0, 0.0),  # 俯仰角速度：0
                "yaw": (-0.0, 0.0),    # 偏航角速度：0
            },
        },
    )

    # 应用外部力和力矩（重置时执行）
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),  # 应用于所有机器人部件
            "force_range": (0.0, 0.0),    # 力的范围：0（无外力）
            "torque_range": (-0.0, 0.0),  # 力矩范围：0（无外力矩）
        },
    )


@configclass
class MARLRiverFlyFollowEnvCfg(DirectMARLEnvCfg):
    """多智能体 River 场景跟随环境配置类"""
    
    # 控制模式配置（与 flycrane 悬停任务保持一致）
    control_mode = "ACCBR"  # 控制模式：ACCBR（加速度+角速度）或 geometric（几何控制）

    # 环境基本参数
    decimation = 3              # 决策降频因子（物理步数/决策步数）
    episode_length_s = 60       # 回合长度（秒）

    # 观测历史配置
    history_len = 3             # 观测历史长度（参考 move：history_len=3，提供速度/加速度隐式信息）

    # 多智能体配置
    possible_agents = ["falcon", "falcon_01", "falcon_02"]  # 默认三架Falcon无人机
    num_drones = len(possible_agents)                   # 无人机数量

    # 目标小车参数配置
    num_targets = 4                                     # 目标小车数量
    target_values: Sequence[float] = (4.0, 3.0, 2.0, 1.0)  # 各目标的价值（红黄绿蓝）
    target_start_x = 0                                  # 目标起始x坐标
    target_end_x = 50.0                                 # 目标结束x坐标
    target_y_positions: Sequence[float] = (15.0, 5.0, -5.0, -15.0)  # 各目标的y坐标分布
    target_speed = 0.8                                  # 目标移动速度（m/s）
    tracking_distance_xy = 3.0                          # tracking reward 激活的 XY 距离阈值（从2.0扩至3.0：zone中心外移，holding提前介入）
    success_distance_xy = 6.0                          # success_mask 判定的 XY 距离阈值（从5.0小幅放宽至6.0）
    track_distance_xy = 3.5                            # 保留旧参数供兼容（success_distance_xy 优先）
    tracking_bonus_distance_xy = 2.0                   # 保留旧参数供兼容（tracking_distance_xy 优先）
    success_hold_time = 0.3                            # 持续满足 success 条件的秒数门槛（从0.5降至0.3：降低绝对持续时长要求）
    success_timer_decay_rate: float = 0.5              # 离开 success zone 时 timer 衰减倍率（从2.0降至0.5：在区50%占比即可净增长）

    success_velocity_tolerance = 2.0                   # 进入 success zone 的严格纵向速度门槛（enter 条件，硬阈值）
    success_distance_xy_hold: float = 8.0              # 进入后维持 timer 的宽松距离门槛（hold 软化距离边界中心）
    success_velocity_tolerance_hold: float = 4.0       # hold 纵向速度 Gaussian sigma：vel=0时factor=1，vel=sigma时factor≈0.37
    success_hold_dist_sharpness: float = 0.3           # hold 距离 sigmoid 陡度（1/m）：0.5→约2m过渡带，越大越陡
    success_proximity_weight: float = 3.0              # 成功区每步密集奖励权重：dist+vel+height 同时达标时发放，对齐 success_mask
    success_bonus_weight: float = 500.0                # 成功终止一次性奖励权重：val=1→5.0，val=4→20.0（×step_dt后）

    distance_reward_sigma = 10                              # 从28降至10：更强的近程梯度（5m内明显提升）
    height_reward_sigma = 1.0
    action_smoothness_sigma = 0.25
    collision_soft_margin = 0.5
    boundary_soft_margin = 1.0
    altitude_soft_margin = 1.0                         # 低高度软惩罚区间：z < min_altitude+1.0=2.0m 时开始罚
    altitude_upper_soft_threshold = 4.0                # 高度软上限前移：z > 4.0m 就开始持续受罚
    high_altitude_soft_margin = 1.0                    # 超高软惩罚归一化区间（4.0~5.0m 迅速拉高代价）
    high_altitude_penalty_weight = 1.2                 # 超高软惩罚权重加强，避免高空平台成为便宜解
    alive_reward_weight = 0.05

    # 奖励塑形参数
    distance_reward_weight = 2.0                        # 距离奖励权重（放大近场梯度）
    tracking_reward_weight = 2.5                        # 进入区奖励权重（entry component，进入即有）
    tracking_zone_sharpness: float = 1.5               # sigmoid 软边界陡度（1/m）：从4.0软化至1.5，梯度覆盖5m范围而非0.25m
    tracking_vel_match_sigma: float = 1.5               # 速度质量高斯核宽度（m/s），仅影响 entry 的速度加成
    tracking_vel_quality_alpha: float = 0.5             # 速度质量加成权重（entry 额外最多 +50%）
    tracking_hold_weight: float = 10.0                 # 持续保持奖励权重（从5.0翻倍至10.0：圈内停留成为最高价值行为）
    tracking_hold_ramp_time: float = 1.0               # ramp 达满值所需时间（从3.0降至1.0s：100步内即可体验满额 holding 正反馈）
    tracking_stable_timer_decay_rate: float = 3.0      # 离圈时 timer 每秒衰减倍率（替代硬归零，防止短暂出圈蒸发所有积累）
    dist_progress_weight: float = 2.0                   # 距离进度奖励权重（势函数 shaping，激励主动追近）
    velocity_follow_weight = 0.8                        # 速度跟随主权重
    velocity_follow_progress_weight = 0.3               # 向前进度奖励系数
    velocity_follow_overspeed_weight = 0.05             # 超速惩罚系数（从0.2降至0.05：允许积极追近）
    velocity_follow_sigma = 1.5                         # 速度匹配高斯核宽度（从0.8放宽至1.5：降低早期训练的匹配难度）
    velocity_follow_overspeed_margin = 1.0              # 超速容忍裕量（从2.0降至1.0：避免无人机5m/s乱飞）
    action_smoothness_weight = 0.5                      # 动作平滑性奖励权重
    body_rate_penalty_weight = 0.02                     # 机体角速率惩罚权重（降至辅助约束，不主导负项）
    velocity_penalty_weight = 0.05                      # 速度惩罚权重（从0.2降至0.05：仅作防失控约束，不主导行为塑形）
    velocity_penalty_xy_safe = 4.0                      # XY 安全速度阈值（从2.0升至4.0 m/s：正常追踪机动完全不触发）
    velocity_penalty_z_safe = 1.0                       # Z 安全速度阈值收紧：持续向上冲高更早进入惩罚区
    velocity_penalty_z_scale = 0.25                     # Z 超速惩罚相对权重
    force_penalty_weight = 0.2                          # 推力惩罚权重
    # 不要求定高悬停，只需保持在安全高度带内（不超高/不超低）
    # 高度控制完全交给软边界惩罚（safety_penalty 中的 low/high_altitude_penalty）和终止条件（fly_low/fly_high）
    height_reward_weight = 0.0                          # 关闭"靠近 desired_height"的高斯奖励（不需要定高）
    height_error_penalty_weight = 0.0                   # 关闭高度误差惩罚（不需要贴 2m 飞）
    height_error_above_extra_weight = 0.0               # 关闭超高额外惩罚（由 high_altitude_penalty 统一处理）
    height_error_quadratic_weight = 0.0                 # 关闭高度二次惩罚
    height_hold_deadband = 0.10                         # 保留参数（已不生效）
    vertical_direction_penalty_weight = 0.0             # 关闭方向性垂直速度惩罚（不需要定高）
    upward_vz_penalty_weight: float = 0.30             # 上升速度惩罚基础权重：持续爬升会更明显吃亏
    upward_vz_penalty_altitude_start: float = 3.0      # 高于该高度后，上升速度惩罚开始随高度继续增强
    upward_vz_penalty_altitude_scale: float = 1.0      # 高度增强系数：每高于起点1m，惩罚系数额外增加1.0
    upward_acc_z_alt_lo: float = 2.5                   # 正向 az 限幅起点高度（m）：低于此值 cap=1.0 不限制
    upward_acc_z_alt_hi: float = 3.5                   # 正向 az 限幅饱和高度（m）：高于此值 cap=az_min_scale
    upward_acc_z_min_scale: float = 0.02               # 饱和高度处正向 az 最小缩放比例（几乎禁止继续上升）
    safety_penalty_weight = 1.0                         # 安全惩罚权重

    # === move 对齐参数 ===
    # 目标捕获距离（实时可撤销，与 move.capture_distance 含义一致）
    capture_distance = 3.0
    # 持续跟随成功门槛（≥3个目标同时被捕获持续此时长则 episode 成功终止，与 move 逻辑对齐）
    sustained_follow_duration = 3.0
    # 距离奖励衰减速率（move=0.5；river 场景更大故取更小值）
    dist_reward_scale = 0.2
    # 追踪区奖励衰减速率（仅在捕获区内有效，move 同参数）
    tracking_reward_scale = 0.3
    # 竖直保持奖励权重（exp-decay，与 move.upright_penalty_weight=2.0 对齐）
    upright_penalty_weight = 2.0
    # 安全硬惩罚（不乘 step_dt，与 move 对齐）
    collision_penalty_scale = 1.0
    drone_out_of_bounds_penalty = 1.0
    fly_low_penalty = 1.0
    illegal_contact_penalty = 1.0
    contact_sensor_threshold = 1.0  # N，低于此值忽略（与 move 一致）

    # ACCBR 速度指令 + PD 控制参数（与 move.ACCBR 对齐）
    lin_vel_max: float = 3.0    # 速度指令最大值（m/s），action[:, :3] 缩放到 ±lin_vel_max
    ang_vel_max: float = 3.0    # 角速度指令最大值（rad/s）
    lin_acc_max: float = 5.0    # 加速度饱和限幅（m/s²）
    vel_Kp: float = 3.0         # 速度 PD 比例增益
    vel_Kd: float = 0.5         # 速度 PD 微分增益

    # 高度与碰撞相关
    desired_height = 2.0                                # 期望高度
    drone_collision_threshold = 1.0                     # 无人机碰撞阈值

    # 终止条件参数
    min_altitude = 1.0                                 # 最小飞行高度（防撞地）
    max_altitude = 7.0                                  # 最大飞行高度（略放宽，减少误终止）
    bounding_box_threshold = 150.0                      # 边界框阈值（防飞出区域）

    # reset 位姿策略：
    # - "event_randomized": 保留 EventCfg.reset_base 的随机化（依赖事件对多机体正确生效）
    # - "usd_fixed": 每次回到 USD 初始位姿（评估/复现实验）
    # - "usd_perturbed": 以 USD 位姿为中心加入小扰动（多机体训练推荐）
    reset_pose_mode: str = "usd_perturbed"
    usd_reset_position_noise_xy: float = 0.5            # usd_perturbed 时，xy 扰动半径（均匀采样）
    usd_reset_position_noise_z: float = 0.2             # usd_perturbed 时，z 扰动范围（均匀采样）
    usd_reset_yaw_noise: float = math.pi                # usd_perturbed 时，yaw 扰动范围（弧度）

    # 动作/观测空间尺寸（在 __post_init__ 中根据数量动态计算）
    obs_dim_per_step = 0
    action_spaces = {}
    observation_spaces = {}

    state_space = -1  # 占位符；__post_init__ 会根据 _get_states() 实际维度重新计算并覆盖此值

    # 仿真配置
    sim: SimulationCfg = SimulationCfg(
        dt=0.0033333333333333335,    # 仿真时间步长（约3ms）
        render_interval=decimation,   # 渲染间隔
        gravity=(0.0, 0.0, -9.8066), # 重力加速度（m/s²）
    )

    # 机器人配置（单机 Falcon）
    robot_cfg: ArticulationCfg = FALCON_CFG.replace(prim_path="/World/envs/env_.*/falcon")
    robot_cfg.spawn.activate_contact_sensors = True  # 激活接触传感器

    # 接触力传感器配置
    contact_forces = ContactSensorCfg(
        prim_path="/World/envs/env_.*/falcon.*/.*",  # 传感器路径
        update_period=0.0,                           # 更新周期
        history_length=3,                            # 历史长度
        debug_vis=False                              # 调试可视化
    )
    # 底层控制参数
    low_level_decimation: int = 1                   # 底层控制降频因子
    max_thrust_pp = 6.25                            # 单个推进器最大推力（牛顿）

    # 场景配置
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=64,          # 环境数量
        env_spacing=8.0,     # 环境间距
        replicate_physics=True  # 复制物理属性
    )

    # 事件配置（重置等）
    events = EventCfg()

    # 调试可视化配置
    debug_vis: bool = True   # 调试可视化开关
    
    # 兼容 MARLHoverEnv 的可视化接口：即使 debug_vis=False 也要有字段
    marker_cfg_goal = FRAME_MARKER_CFG.copy()
    marker_cfg_goal.markers["frame"].scale = (0.1, 0.1, 0.1)  # 标记缩放
    marker_cfg_goal.prim_path = "/World/envs/env_0/Visuals/FlyFollow/goal_pose"  # 目标姿态路径

    marker_cfg_body = FRAME_MARKER_CFG.copy()
    marker_cfg_body.markers["frame"].scale = (0.1, 0.1, 0.1)  # 标记缩放
    marker_cfg_body.prim_path = "/World/envs/env_0/Visuals/FlyFollow/body_pose"  # 机体姿态路径

    def __post_init__(self):
        self.num_drones = len(self.possible_agents)
        # Keep this in the same order as _get_observations()
        self.obs_dim_per_step = (
            self.num_drones                                # agent one-hot
            + 3                                            # own position xyz（含高度 z）
            + 9                                            # own rotation matrix
            + 3                                            # own linear velocity xyz（含垂直速度 vz）
            + 3                                            # own angular velocity
            + (self.num_targets * 2)                      # own relative target xy
            + self.num_targets                            # own target distances
            + ((self.num_drones - 1) * 2)                 # relative xy to other drones
            + ((self.num_drones - 1) * self.num_targets)  # other drones to target distances
            + (self.num_targets * 2)                      # target velocities xy
            + (self.num_targets * self.num_drones)        # closest drone one-hot for each target
            + self.num_targets                            # target values
            + self.num_targets                            # assigned target one-hot
        )

        if self.control_mode == "geometric":
            action_dim = 12
        else:
            # 6维 ACCBR：[ax, ay, az, p, q, r]，同 move 保持一致，开放偏航率控制
            action_dim = 6

        obs_dim = self.obs_dim_per_step * self.history_len
        self.action_spaces = {agent: action_dim for agent in self.possible_agents}
        self.observation_spaces = {agent: obs_dim for agent in self.possible_agents}

        # 全局状态维度（Critic 输入，用于 CTDE）：与 _get_states() 实际输出对齐
        # drone_pos(3*D) + rot_mat(9*D) + lin_vel(3*D) + ang_vel(3*D)
        # + target_pos(3*T) + target_vel(3*T) + target_claimed(T) + target_values(T)
        # + assigned_target_one_hot(D*T)
        self.state_space = (
            self.num_drones * 3           # drone positions
            + self.num_drones * 9         # rotation matrices
            + self.num_drones * 3         # linear velocities
            + self.num_drones * 3         # angular velocities
            + self.num_targets * 3        # target positions
            + self.num_targets * 3        # target velocities
            + self.num_targets            # target claimed (bool)
            + self.num_targets            # target values
            + self.num_drones * self.num_targets  # assigned target one-hot
        )
