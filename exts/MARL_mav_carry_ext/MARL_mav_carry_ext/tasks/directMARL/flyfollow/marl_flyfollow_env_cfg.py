# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

# 导入飞 crane 机器人配置
from MARL_mav_carry_ext.assets import FLYCRANE_CFG

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
                "roll": (0.0, 0.0),    # 滚转角：保持水平
                "pitch": (0.0, 0.0),   # 俯仰角：保持水平
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
    episode_length_s = 20       # 回合长度（秒）

    # 观测历史配置
    history_len = 1             # 观测历史长度（跟随任务中通常不需要历史信息）

    # 多智能体配置
    possible_agents = ["falcon1", "falcon2", "falcon3"]  # 三架Falcon无人机
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
    target_start_x = -20.0                              # 目标起始x坐标
    target_end_x = -12.0                                # 目标结束x坐标
    target_y_positions: Sequence[float] = (-24.5, -22.5, -20.0, -17.5)  # 各目标的y坐标分布
    target_speed = 0.35                                 # 目标移动速度（m/s）
    track_distance_xy = 0.6                             # 成功跟踪的XY平面距离阈值

    # 奖励塑形参数
    distance_penalty_weight = 0.12                      # 距离惩罚权重
    mean_distance_weight = 0.35                         # 平均距离权重
    min_distance_weight = 1.0                           # 最短距离权重
    proximity_reward_weight = 0.25                      # 接近奖励权重
    proximity_sigma = 1.0                               # 接近奖励的衰减系数

    # 终止条件参数
    min_altitude = 0.2                                  # 最小飞行高度（防撞地）
    bounding_box_threshold = 10.0                       # 边界框阈值（防飞出区域）

    # 动作/观测空间尺寸计算
    obs_dim_per_step = (
        num_drones  # one-hot智能体标识（3维）
        + 3  # 无人机位置（3维）
        + 3  # 无人机速度（3维）
        + (num_targets * 2)  # 到各目标的相对XY位置（8维）
        + num_targets  # 到各目标的距离（4维）
        + ((num_drones - 1) * num_targets)  # 其他无人机到目标的距离（8维）
        + num_targets  # 目标价值（4维）
    )  # 总计：3+3+3+8+4+8+4 = 33维（单步观测）
    
    # 根据控制模式设置动作空间和观测空间
    if control_mode == "geometric":
        action_dim_geo = 12     # 几何控制的动作维度
        action_spaces = {"falcon1": action_dim_geo, "falcon2": action_dim_geo, "falcon3": action_dim_geo}
        obs_dim_geo = obs_dim_per_step * history_len  # 考虑历史长度
        observation_spaces = {"falcon1": obs_dim_geo, "falcon2": obs_dim_geo, "falcon3": obs_dim_geo}
    elif control_mode == "ACCBR":
        action_dim_accbr = 5    # ACCBR控制的动作维度（3个线加速度 + 2个角速度）
        action_spaces = {"falcon1": action_dim_accbr, "falcon2": action_dim_accbr, "falcon3": action_dim_accbr}
        obs_dim_accbr = obs_dim_per_step * history_len
        observation_spaces = {"falcon1": obs_dim_accbr, "falcon2": obs_dim_accbr, "falcon3": obs_dim_accbr}

    state_space = -1  # 状态空间维度（跟随任务中不使用全局状态）

    # 仿真配置
    sim: SimulationCfg = SimulationCfg(
        dt=0.0033333333333333335,    # 仿真时间步长（约3ms）
        render_interval=decimation,   # 渲染间隔
        gravity=(0.0, 0.0, -9.8066), # 重力加速度（m/s²）
    )

    # 机器人配置（飞行吊挂系统）
    robot_cfg: ArticulationCfg = FLYCRANE_CFG.replace(prim_path="/World/envs/env_.*/flycrane")
    robot_cfg.spawn.activate_contact_sensors = True  # 激活接触传感器

    # 接触力传感器配置
    contact_forces = ContactSensorCfg(
        prim_path="/World/envs/env_.*/flycrane/.*",  # 传感器路径
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
    debug_vis: bool = False  # 调试可视化开关
    
    # 兼容 MARLHoverEnv 的可视化接口：即使 debug_vis=False 也要有字段
    marker_cfg_goal = FRAME_MARKER_CFG.copy()
    marker_cfg_goal.markers["frame"].scale = (0.1, 0.1, 0.1)  # 标记缩放
    marker_cfg_goal.prim_path = "/Visuals/Command/goal_pose"  # 目标姿态路径

    marker_cfg_body = FRAME_MARKER_CFG.copy()
    marker_cfg_body.markers["frame"].scale = (0.1, 0.1, 0.1)  # 标记缩放
    marker_cfg_body.prim_path = "/Visuals/Command/body_pose"  # 机体姿态路径