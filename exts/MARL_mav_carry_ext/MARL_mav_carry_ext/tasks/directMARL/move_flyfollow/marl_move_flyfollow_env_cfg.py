# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from MARL_mav_carry_ext.assets import FALCON_CFG

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectMARLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
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
    # prim 名称需与 move_flyfollow.usda 中的 Falcon prim 名称一致
    possible_agents = ["falcon", "falcon_01", "falcon_02"]
    num_drones = len(possible_agents)  # 3架无人机

    # ===== 移动目标配置 =====
    num_targets = 4           # 目标小车数量（move_flyfollow.usda 中有4辆 NovaCarter）
    # 目标价值（高价值目标优先级更高，体现在距离奖励的加权上）
    target_values = [4.0, 3.0, 2.0, 1.0]
    target_velocity = 0.3     # 小车沿 x 轴正方向的匀速（m/s）
    capture_distance = 2.0    # 捕获/跟随判定距离（m）：NovaCarter 车身较大，适当放宽
    sustained_follow_duration = 3.0  # 成功终止需持续跟随的时间（秒）
    # 目标小车边界：x 超过此值触发 targets_out_of_bounds 终止
    target_end_x = 30.0       # 小车跑出场景边界的 x 坐标（与场景 bounding_box 匹配）

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

    # ===== 动作/观测空间维度（在 __post_init__ 中动态填充到 action_spaces/observation_spaces）=====
    # ACCBR 模式：vel_cmd(3) + body_rates(3) = 6维动作
    # geometric 模式：pos(3)+vel(3)+acc(3)+jerk(3) = 12维动作
    action_dim = 6
    # 单步观测 49 维：pos(3)+vel(3)+rot_mat(9)+other_drones(6)+targets(20)+dist(4)+closest(4)
    obs_dim_per_step = 49
    # 全局 state 86 维：pos(9)+rot(27)+vel(9)+ang_vel(9)+t_pos(12)+t_vel(12)+captured(4)+values(4)
    state_space = 86

    # 占位，__post_init__ 中按 possible_agents 动态填充
    action_spaces: dict = {}
    observation_spaces: dict = {}

    # ===== 仿真配置 =====
    # dt = 1/300 ≈ 3.33ms（物理步长）；decimation=3 → 高级控制步长 = 10ms（100Hz）
    sim: SimulationCfg = SimulationCfg(
        dt=0.0033333333333333335,
        render_interval=decimation,
        gravity=(0.0, 0.0, -9.8066),
    )

    # ===== 无人机 Articulation 配置 =====
    # prim_path 与 move_flyfollow.usda 中第一架 Falcon（名为 "falcon"）对应
    # spawn=None 模式下此处 prim_path 仅为模板，_setup_scene 会动态覆盖
    robot_cfg: ArticulationCfg = FALCON_CFG.replace(
        prim_path="/World/envs/env_.*/falcon"
    )
    robot_cfg.spawn.activate_contact_sensors = True

    # ===== 接触传感器（单一配置，_setup_scene 中按 agent 动态实例化）=====
    # prim_path 使用 falcon.* 通配符，匹配 falcon / falcon_01 / falcon_02
    contact_forces: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/falcon.*/.*",
        update_period=0.0,
        history_length=3,
        debug_vis=False,
    )

    # ===== 终止条件阈值 =====
    drone_collision_threshold = 0.6   # 两无人机中心距离 < 0.6m 触发碰撞终止
    bounding_box_threshold = 12.0     # 超出 ±12m 范围终止

    # 接触传感器触发阈值（N）
    contact_sensor_threshold = 1.0

    # ===== 低级控制参数 =====
    low_level_decimation: int = 1   # 内环控制抽取率（1 = 每物理步执行一次）
    max_thrust_pp = 6.25            # 单个旋翼最大推力（N）

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
    # env_spacing=25.0：相邻环境间距 25m，防止跨 env 干扰
    # replicate_physics=True：物理属性从 env_0 复制到其余环境（需 USD 加载）
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1, env_spacing=25.0, replicate_physics=True
    )

    def __post_init__(self):
        """根据 possible_agents 动态填充动作/观测空间字典。"""
        super().__post_init__()
        obs_dim = self.obs_dim_per_step * self.history_len if self.partial_obs else self.obs_dim_per_step
        self.action_spaces = {agent: self.action_dim for agent in self.possible_agents}
        self.observation_spaces = {agent: obs_dim for agent in self.possible_agents}
