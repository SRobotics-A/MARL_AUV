"""单无人机 fly-forward 任务的环境配置（PPO）。

任务：从出生点出发，在保持低空稳定飞行的同时向前推进。
      当前阶段课程：先学习低空存活和短距离前向飞行。
      飞行高度必须保持在 4m 以下。
场景：fly_forward.usda（Rivermark 室外场景 + 单架 Falcon 无人机）。
"""

from __future__ import annotations

from pathlib import Path

from gymnasium.spaces import Box

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

import isaaclab.sim as sim_utils

from MARL_mav_carry_ext.assets import FALCON_CFG


@configclass
class FlyForwardEnvCfg(DirectRLEnvCfg):
    """fly-forward 单无人机 PPO 任务配置（低空、短距离、前向飞行课程）。"""

    # ── Control ──────────────────────────────────────────────────────────────
    # 控制模式：ACCBR 表示动作直接给速度 / 角速度指令，z 方向由高度保持逻辑自动处理。
    control_mode: str = "ACCBR"  # 5 维动作：[vx, vy, roll_rate, pitch_rate, yaw_rate]
    lin_vel_max: float = 3.0     # x/y 平面速度指令上限，单位 m/s
    lin_vel_z_up_max: float = 0.08    # z 方向向上速度上限，保守限制爬升速度
    lin_vel_z_down_max: float = 1.5   # z 方向向下速度上限，保留从过高高度恢复的下降能力
    ang_vel_max: float = 0.5     # 姿态角速度指令上限，单位 rad/s
    lin_acc_max: float = 2.0     # x/y 平面加速度上限，单位 m/s²
    lin_acc_z_up_max: float = 0.1     # z 方向向上加速度上限，抑制高度超调
    lin_acc_z_down_max: float = 2.0   # z 方向向下加速度上限，允许更强下降修正
    vel_Kp: float = 2.0          # 速度控制比例增益
    vel_Kd: float = 0.3          # 速度控制微分增益

    # ── Episode ───────────────────────────────────────────────────────────────
    decimation: int = 3          # 环境每执行一次 action，对应的物理仿真步数
    episode_length_s: float = 120.0 # 单个 episode 的最长持续时间，单位秒

    # ── Spaces ────────────────────────────────────────────────────────────────
    # 观测 obs：pos(3) + lin_vel(3) + rot_mat(9) + ang_vel(3) + goal_rel(3) = 21
    # 注意：必须用有界 Box，否则 SKRL random_act 从 Uniform(-inf,+inf) 采样得到 NaN
    action_space: Box = Box(low=-1.0, high=1.0, shape=(5,)) # 归一化动作空间，5 维连续动作
    observation_space: int = 21 # 单智能体观测维度
    state_space: int = 0        # 未使用集中式 state，置 0

    # ── Goal ──────────────────────────────────────────────────────────────────
    # 阶段课程目标：先要求无人机从出生点完成短距离前向推进。
    goal_x: float = 8.0         # 目标点 x 坐标
    goal_y: float = 0.0         # 目标点 y 坐标
    goal_z: float = 2.0         # 目标高度
    goal_tolerance: float = 2.0 # 短距离阶段的目标 shaping 半径，单位 m
    success_speed_tolerance: float = 0.8 # 成功时的速度上限，第一阶段先宽松限制高速撞线
    height_hold_kp: float = 1.5 # 高度保持比例增益
    height_hold_damping: float = 0.4 # 高度保持阻尼系数

    # ── Spawn ─────────────────────────────────────────────────────────────────
    spawn_x: float = -8.0        # 固定起点 x（与 USDA 一致）
    spawn_y: float = 3.0         # 固定起点 y（与 USDA 一致）
    spawn_z: float = 2.0         # 固定起点 z（与 USDA 一致）
    spawn_x_noise: float = 0.5   # 随机扰动（训练鲁棒性）
    spawn_y_noise: float = 0.5   # y 方向随机扰动（训练鲁棒性）

    # ── Altitude limits ───────────────────────────────────────────────────────
    fly_high_z: float = 3.0     # 飞得过高的终止高度阈值
    fly_high_guard_z: float = 2.0 # 过高保护开始介入的高度阈值
    fly_high_guard_descent_acc: float = 3.5 # 过高保护触发时额外施加的下降加速度
    fly_low_z: float = 0.3      # 飞得过低的终止高度阈值
    out_of_bounds_radius: float = 35.0 # 水平越界半径；短距离目标下保持较紧范围

    # ── Normalisation ─────────────────────────────────────────────────────────
    norm_pos_scale: float = 50.0  # x/y 位置归一化尺度，适配前向飞行课程
    norm_goal_xy_scale: float = 8.0 # 目标相对 x/y 归一化尺度，当前 8m 课程下初始 goal_rel_x 约为 1
    norm_z_scale: float = 5.0     # z 位置独立归一化，避免高度信号过小
    norm_vel_scale: float = 5.0   # 速度归一化尺度

    # ── Reward weights ────────────────────────────────────────────────────────
    progress_reward_weight: float = 5.0     # 真实接近目标的进度奖励权重
    forward_vel_reward_weight: float = 2.5  # 兼容旧配置，当前 reward 不使用
    dist_reward_weight: float = 1.0         # 距离目标奖励权重
    dist_reward_scale: float = 0.25         # 距离奖励缩放系数；8m 初始距离下约 exp(-2)=0.135
    height_reward_weight: float = 0.8       # 兼容旧配置，当前 reward 不使用
    altitude_band_reward_weight: float = 0.2 # 安全高度带内存活正奖励
    height_penalty_weight: float = 3.0      # 偏离目标高度的惩罚权重
    fly_high_guard_penalty_weight: float = 20.0 # 过高保护区域惩罚权重
    near_goal_radius: float = 3.0           # 目标附近减速区域半径，需大于 success 半径以提前减速
    slow_near_goal_reward_weight: float = 1.0 # 目标附近低速奖励权重
    speed_reward_scale: float = 1.0         # 低速奖励中的速度衰减系数
    speed_penalty_weight: float = 0.02      # 全局速度惩罚权重
    upright_penalty_weight: float = 0.5     # 姿态偏离竖直 / 水平稳定状态的惩罚权重
    action_smoothness_weight: float = 0.01  # 动作平滑惩罚权重
    success_forward_x: float = 8.0          # 判定前向成功所需达到的 x 坐标
    success_reward: float = 20.0            # 成功到达目标后的终止奖励
    fly_high_penalty: float = 20.0          # 飞得过高时的终止惩罚
    fly_low_penalty: float = 20.0           # 飞得过低时的终止惩罚
    out_of_bounds_penalty: float = 20.0     # 水平越界时的终止惩罚

    # ── Simulation ────────────────────────────────────────────────────────────
    # 仿真配置：控制物理时间步、渲染间隔和重力。
    sim: SimulationCfg = SimulationCfg(
        dt=0.0033333333333333335, # 物理仿真时间步，约 300Hz
        render_interval=decimation, # 渲染间隔与 decimation 保持一致
        gravity=(0.0, 0.0, -9.8066), # 标准重力加速度
    )

    # ── Scene USD（fly_forward.usda 包含 Rivermark + Falcon）─────────────────
    # 路径在 _setup_scene 中通过 Path(__file__) 动态解析，无需在此硬编码

    # ── Robot cfg（_setup_scene 中设 spawn=None，prim 由 USDA 定义）─────────────
    robot_cfg: ArticulationCfg = FALCON_CFG.replace(
        prim_path="/World/envs/env_.*/falcon",  # 占位，_setup_scene 中动态替换
    )

    # 场景配置：设置并行环境数量、环境间距和物理复制策略。
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=16, env_spacing=220.0, replicate_physics=True # 保持足够 env 间距，避免并行环境互相干扰
    )

    # ── Collision contact threshold ───────────────────────────────────────────
    contact_sensor_threshold: float = 50.0 # 接触力阈值，超过后可认为发生明显碰撞

    # ── Low-level control ─────────────────────────────────────────────────────
    low_level_decimation: int = 1 # 底层控制执行频率相对仿真的降采样倍数
    max_thrust_pp: float = 6.25  # 单个旋翼最大推力，单位 N
