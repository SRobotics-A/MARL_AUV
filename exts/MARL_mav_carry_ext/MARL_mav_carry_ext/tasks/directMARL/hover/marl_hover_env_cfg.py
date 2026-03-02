# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import math

# 导入飞 crane 机器人配置
from MARL_mav_carry_ext.assets import FLYCRANE_CFG

# 导入MDP相关的工具函数
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
    """悬停任务的事件配置类
    
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
                "x": (-1.0, 1.0),      # x坐标范围：-1到1米
                "y": (-1.0, 1.0),      # y坐标范围：-1到1米
                "z": (0.5, 1.5),       # z坐标范围：0.5到1.5米（高度）
                "roll": (-0, 0),       # 滚转角范围：0度（保持水平）
                "pitch": (-0, 0),      # 俯仰角范围：0度（保持水平）
                "yaw": (-math.pi, math.pi),  # 偏航角范围：-π到π弧度（任意朝向）
            },
            # 速度范围配置
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

    # 随机化负载质量（在启动时执行）
    randomize_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["load_link"]),  # 指定要随机化的部件
            "mass_distribution_params": (1.4, 1.4),  # 质量随机化范围
            "operation": "abs",                      # 绝对值操作
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
class MARLHoverEnvCfg(DirectMARLEnvCfg):
    """多智能体悬停环境配置类"""
    
    # 控制模式配置
    control_mode = "ACCBR"  # 控制模式：ACCBR（加速度+角速度）或 geometric（几何控制）
    
    # 环境基本参数
    decimation = 3              # 决策降频因子
    episode_length_s = 20       # 回合长度（秒）

    # 观测历史配置
    partial_obs = True          # 是否使用局部观测
    history_len = 3             # 观测历史长度

    # 多智能体配置
    possible_agents = ["falcon1", "falcon2", "falcon3"]  # 可能的智能体列表
    num_drones = len(possible_agents)                   # 无人机数量
    
    # 根据不同控制模式设置动作空间和观测空间
    if control_mode == "geometric":
        action_dim_geo = 12     # 几何控制的动作维度
        action_spaces = {"falcon1": action_dim_geo, "falcon2": action_dim_geo, "falcon3": action_dim_geo}
        obs_dim_geo = 87        # 几何控制的观测维度
        observation_spaces = {"falcon1": obs_dim_geo, "falcon2": obs_dim_geo, "falcon3": obs_dim_geo}
        state_space = 84        # 全局状态空间维度
    elif control_mode == "ACCBR":
        action_dim_accbr = 5    # ACCBR控制的动作维度（3个线加速度 + 2个角速度）
        action_spaces = {"falcon1": action_dim_accbr, "falcon2": action_dim_accbr, "falcon3": action_dim_accbr}
        if partial_obs:
            obs_dim_accbr = 45 * history_len  # 局部观测维度 × 历史长度
        else:
            obs_dim_accbr = 87                # 全局观测维度
        observation_spaces = {"falcon1": obs_dim_accbr, "falcon2": obs_dim_accbr, "falcon3": obs_dim_accbr}
        state_space = 84

    # 仿真配置
    sim: SimulationCfg = SimulationCfg(
        dt=0.0033333333333333335,    # 仿真时间步长（约3ms）
        render_interval=decimation,   # 渲染间隔
        gravity=(0.0, 0.0, -9.8066), # 重力加速度（m/s²）
    )
    
    # 机器人配置
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

    # 安全限制参数
    cable_angle_limits_drone = 0.0                  # 无人机端电缆角度限制（余弦值）
    cable_angle_limits_payload = -math.sqrt(2) / 2  # 负载端电缆角度限制（余弦值）
    cable_collision_threshold = 0.2                 # 电缆碰撞阈值
    cable_collision_num_points = 10                 # 电缆碰撞检测点数
    drone_collision_threshold = 0.6                 # 无人机碰撞阈值
    bounding_box_threshold = 5.0                    # 边界框阈值
    goal_achieved_range = 0.3                       # 目标达成距离范围
    goal_achieved_ori_range = 0.4                   # 目标达成姿态范围

    # 底层控制参数
    low_level_decimation: int = 1                   # 底层控制降频因子
    max_thrust_pp = 6.25                            # 单个推进器最大推力（牛顿）

    # 奖励权重配置
    pos_track_weight = 1.5                          # 位置跟踪奖励权重
    ori_track_weight = 1.5                          # 姿态跟踪奖励权重
    action_smoothness_weight = 0.5                  # 动作平滑性奖励权重
    body_rate_penalty_weight = 0.5                  # 机体角速度惩罚权重
    force_penalty_weight = 0.5                      # 力惩罚权重
    downwash_rew_weight = 0.5                       # 下洗流奖励权重

    # 目标范围配置
    goal_range = {
        "pos_x": (-1.0, 1.0),                       # x坐标目标范围
        "pos_y": (-1.0, 1.0),                       # y坐标目标范围
        "pos_z": (0.5, 1.5),                        # z坐标目标范围
        "roll": (-math.pi / 4, math.pi / 4),        # 滚转角目标范围
        "pitch": (-math.pi / 4, math.pi / 4),       # 俯仰角目标范围
        "yaw": (-math.pi, math.pi),                 # 偏航角目标范围
    }
    range_curriculum_steps = 7500                   # 课程学习步数

    make_quat_unique_command = False                # 是否使四元数唯一化

    # 调试可视化配置
    debug_vis: bool = True
    if debug_vis:
        # 目标标记配置
        marker_cfg_goal = FRAME_MARKER_CFG.copy()
        marker_cfg_goal.markers["frame"].scale = (0.1, 0.1, 0.1)  # 标记缩放
        marker_cfg_goal.prim_path = "/Visuals/Command/goal_pose"  # 目标姿态路径

        # 机体标记配置
        marker_cfg_body = FRAME_MARKER_CFG.copy()
        marker_cfg_body.markers["frame"].scale = (0.1, 0.1, 0.1)  # 标记缩放
        marker_cfg_body.prim_path = "/Visuals/Command/body_pose"  # 机体姿态路径

    # 场景配置
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1,          # 环境数量
        env_spacing=8.0,     # 环境间距
        replicate_physics=True  # 复制物理属性
    )

    # 事件配置
    events = EventCfg()
