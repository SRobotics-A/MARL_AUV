# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from pathlib import Path
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


# 资产根目录（与本机 Isaac Sim 资产路径一致）
_ASSET_ROOT = Path(
    "/media/xtj/1CC8D044C8D01DB8/RL-download/isaac-sim/v5.1.0/Assets/Isaac/5.1/Isaac"
)

# 目标小车与场地 USD 模型路径
_NOVA_CARTER_USD = _ASSET_ROOT / "Robots/NVIDIA/NovaCarter/Variants/nova_carter_sim_optimized.usd"  # 目标小车模型
_JETRACER_TRACK_USD = _ASSET_ROOT / "Environments/Jetracer/jetracer_track_solid.usd"              # 赛道模型


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
class MARLFlyFollowEnvCfg(DirectMARLEnvCfg):
    """多智能体跟随环境配置类"""
    
    # 控制模式配置（与 flycrane 悬停任务保持一致）
    control_mode = "ACCBR"  # 控制模式：ACCBR（加速度+角速度）或 geometric（几何控制）

    # 环境基本参数
    decimation = 3              # 决策降频因子（物理步数/决策步数）
    episode_length_s = 60       # 回合长度（秒）

    # 观测历史配置
    history_len = 1             # 观测历史长度（跟随任务中通常不需要历史信息）

    # 多智能体配置
    possible_agents = ["falcon", "falcon_01", "falcon_02"]  # 默认三架Falcon无人机
    num_drones = len(possible_agents)                   # 无人机数量

    # 目标小车参数配置
    num_targets = 4                                     # 目标小车数量
    target_values: Sequence[float] = (4.0, 3.0, 2.0, 1.0)  # 各目标的价值（红黄绿蓝）
    target_colors: Sequence[tuple[float, float, float]] = (
        (1.0, 0.0, 0.0),  # 红色 - 最高价值目标
        (1.0, 1.0, 0.0),  # 黄色 - 次高价值目标
        (0.0, 1.0, 0.0),  # 绿色 - 中等价值目标
        (0.0, 0.4, 1.0),  # 蓝色 - 最低价值目标
    )
    target_start_x = 0                                  # 目标起始x坐标
    target_end_x = 50.0                                 # 目标结束x坐标
    target_y_positions: Sequence[float] = (15.0, 5.0, -5.0, -15.0)  # 各目标的y坐标分布
    target_speed = 0.8                                  # 目标移动速度（m/s）
    track_distance_xy = 2.5                             # 成功跟踪的XY平面距离阈值

    distance_reward_sigma = 28
    tracking_reward_sigma = 18
    height_reward_sigma = 1.0
    action_smoothness_sigma = 0.25
    collision_soft_margin = 0.5
    boundary_soft_margin = 1.0
    altitude_soft_margin = 1.0
    altitude_upper_soft_threshold = 2.8                # 高度软上限，超过后开始惩罚
    high_altitude_soft_margin = 0.6                    # 高度超上限惩罚归一化范围
    high_altitude_penalty_weight = 0.6                 # 超高飞行惩罚权重（上调）
    alive_reward_weight = 0.05

    # 奖励塑形参数
    distance_reward_weight = 1.0                        # 距离奖励权重
    tracking_reward_weight = 1.0                        # 追踪奖励权重
    velocity_follow_weight = 0.8                        # 速度跟随主权重
    velocity_follow_progress_weight = 0.3               # 向前进度奖励系数
    velocity_follow_overspeed_weight = 0.2              # 超速惩罚系数（速度跟随项内部）
    velocity_follow_sigma = 0.8                         # 速度匹配高斯核宽度
    velocity_follow_overspeed_margin = 0.3              # 超速容忍裕量（m/s）
    action_smoothness_weight = 0.5                      # 动作平滑性奖励权重
    body_rate_penalty_weight = 0.2                      # 机体角速率惩罚权重
    velocity_penalty_weight = 0.2                       # 速度惩罚权重
    force_penalty_weight = 0.2                          # 推力惩罚权重
    height_reward_weight = 0.5                          # 高度奖励权重
    height_error_penalty_weight = 0.35                  # 对称高度误差惩罚权重 |z-desired_height|
    height_error_above_extra_weight = 0.35              # 高于目标高度时的额外惩罚权重
    height_error_quadratic_weight = 0.08                # 高度误差二次惩罚权重（拉回2m）
    height_hold_deadband = 0.10                         # 高度保持死区（米），死区内不罚
    vertical_direction_penalty_weight = 0.25            # 反方向垂直速度惩罚（高处继续上升/低处继续下降）
    safety_penalty_weight = 1.0                         # 安全惩罚权重

    # 高度与碰撞相关
    desired_height = 2.0                                # 期望高度
    drone_collision_threshold = 0.6                     # 无人机碰撞阈值

    # 终止条件参数
    min_altitude = 0.2                                  # 最小飞行高度（防撞地）
    max_altitude = 6.0                                  # 最大飞行高度（超出即终止）
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

    state_space = -1  # 状态空间维度（跟随任务中不使用全局状态）

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
    sensor_cfg = SceneEntityCfg("contact_forces", body_names=".*")  # 传感器实体配置
    contact_sensor_threshold = 0.1                  # 接触传感器阈值

    # 机体部件名称配置
    falcon_names = "Falcon.*_base_link_inertia"     # 无人机质心链接名称
    falcon_rotor_names = "Falcon.*_rotor_.*"        # 旋翼名称
    payload_name = "load_odometry_sensor_link"      # 负载名称
    rope_name = "rope_.*_link"                      # 绳索名称

    # 底层控制参数
    low_level_decimation: int = 1                   # 底层控制降频因子
    max_thrust_pp = 6.25                            # 单个推进器最大推力（牛顿）

    # 可视化资产路径配置
    track_usd_path: str = str(_JETRACER_TRACK_USD)  # 赛道模型路径
    target_usd_path: str = str(_NOVA_CARTER_USD)    # 目标小车模型路径

    # 场景配置
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1,          # 环境数量
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
        self.obs_dim_per_step = (
            self.num_drones
            + 2
            + 9
            + 2
            + 3
            + (self.num_targets * 2)
            + self.num_targets
            + ((self.num_drones - 1) * 2)
            + ((self.num_drones - 1) * self.num_targets)
            + (self.num_targets * 2)
            + (self.num_targets * self.num_drones)
            + self.num_targets
        )

        if self.control_mode == "geometric":
            action_dim = 12
        else:
            action_dim = 5

        obs_dim = self.obs_dim_per_step * self.history_len
        self.action_spaces = {agent: action_dim for agent in self.possible_agents}
        self.observation_spaces = {agent: obs_dim for agent in self.possible_agents}
