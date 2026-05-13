"""单无人机 fly-forward 任务的环境配置（PPO）。

任务：从出生点出发，在保持低空稳定飞行的同时向前推进。
      当前阶段课程：先学习低空存活和短距离前向飞行。
      当前高飞终止阈值由 fly_high_z 控制，过高保护从 fly_high_guard_z 开始介入。
场景：fly_forward.usda（Rivermark 室外场景 + 单架 Falcon 无人机）。

配置组织方式：
    Control: 高层动作到期望速度 / 期望加速度的映射和限幅。
    Episode/Spaces: DirectRLEnv 的 episode 长度、action_space、observation_space。
    Goal/Spawn/Altitude: 当前课程目标、出生点随机化和低空安全边界。
    Normalisation: 观测归一化尺度，影响 PPO 能否看到足够强的目标信号。
    Reward weights: reward shaping 的所有权重，只定义数值，不实现公式。
    Simulation/Scene/Robot: Isaac Lab 仿真、USD 场景和 Falcon articulation 配置。

注意：
    当前课程故意不是最终长航程任务，而是“短距离、低速、低空、一维前飞”的第一阶段。
    如果后续把 goal_x 加到 20m/50m/200m，需要同步检查 norm_goal_xy_scale、dist_reward_scale、
    near_goal_radius、速度上限和终止半径，否则 reward 尺度会再次失衡。
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
    """fly-forward 单无人机 PPO 任务配置（低空、短距离、前向飞行课程）。

    这个类只保存参数，不包含训练逻辑。实际使用位置：
        - fly_forward_env.py::_pre_physics_step 读取控制限幅和高度保持参数。
        - fly_forward_env.py::_get_observations 读取观测归一化参数。
        - fly_forward_env.py::_get_rewards 读取 reward 权重。
        - fly_forward_env.py::_get_dones 读取成功 / 失败阈值。
    """

    # ── Control ──────────────────────────────────────────────────────────────
    # 控制模式：ACCBR 表示兼容“加速度 + body rate”控制接口。
    # 但当前课程为了降低探索维度，只使用 action[0] -> 非负 vx_cmd。
    # 动作映射为 a0=-1 -> 0m/s，a0=0 -> 半速前进，a0=1 -> 最大前进速度，避免初始策略开局悬停。
    # z 方向由高度保持逻辑自动处理，body_rates/yaw_rate 暂时固定为 0。
    control_mode: str = "ACCBR"  # 兼容 5 维动作；当前课程只使用 action[0] 生成非负 vx_cmd
    lin_vel_max: float = 3.0     # 兼容旧配置，当前 x/y 分别使用 lin_vel_x_max / lin_vel_y_max
    lin_vel_x_max: float = 0.6   # x 方向速度指令上限，单位 m/s；10m 课程稳定后小幅提速
    lin_vel_y_max: float = 0.0   # y 方向速度指令上限，单位 m/s
    lin_vel_z_up_max: float = 0.4     # z 方向向上速度上限，禁用 acc_load 后需要足够恢复高度
    lin_vel_z_down_max: float = 1.5   # z 方向向下速度上限，保留从过高高度恢复的下降能力
    ang_vel_max: float = 0.5     # 姿态角速度指令上限，单位 rad/s
    lin_acc_max: float = 0.5     # x/y 平面加速度上限，单位 m/s²；比 2m 快，但避免 20m 版本过冲
    lin_acc_z_up_max: float = 1.0     # z 方向向上加速度上限，给高度保持足够上升余量
    lin_acc_z_down_max: float = 3.0   # z 方向向下加速度上限，允许更强下降修正
    vel_Kp: float = 2.0          # 速度控制比例增益
    vel_Kd: float = 0.3          # 速度控制微分增益
    simple_vx_kp: float = 0.2    # SIMPLE_CONTROL 下 desired_vx -> acc_x 的比例增益
    simple_acc_x_max: float = 0.05 # SIMPLE_CONTROL 下 x 方向加速度限幅，单位 m/s²
    drone_mass: float = 0.6017   # Falcon 质量估计，direct-force 诊断模式用于计算悬停推力

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
    # 阶段课程目标：在低空稳定前飞的基础上，把目标距离扩展到 10m。
    # reset 时实际目标为 spawn_pos + (goal_x, goal_y, 0)，z 使用 env_origin + goal_z。
    goal_x: float = 10.0        # 目标点相对出生点的 x 偏移，单位 m
    goal_y: float = 0.0         # 目标点相对出生点的 y 偏移，单位 m
    goal_z: float = 2.0         # 目标局部高度，单位 m
    goal_tolerance: float = 1.3 # 10m 课程的目标半径，先保留一定容差，稳定后再收紧
    success_speed_tolerance: float = 0.5 # 成功时的速度上限，要求接近目标时完成减速
    height_hold_kp: float = 2.0625 # 高度保持比例增益
    height_hold_damping: float = 0.4 # 高度保持阻尼系数

    # ── Spawn ─────────────────────────────────────────────────────────────────
    spawn_x: float = -8.0        # 固定起点 x（与 USDA 一致）
    spawn_y: float = 3.0         # 固定起点 y（与 USDA 一致）
    spawn_z: float = 2.0         # 固定起点 z（与 USDA 一致）
    spawn_x_noise: float = 0.5   # 随机扰动（训练鲁棒性）
    spawn_y_noise: float = 0.5   # y 方向随机扰动（训练鲁棒性）

    # ── Altitude limits ───────────────────────────────────────────────────────
    fly_high_z: float = 5.0     # 飞得过高的终止高度阈值
    fly_high_guard_z: float = 4.5 # 接近高度上限时才开始轻度保护，不作为低空奖励目标
    fly_high_guard_descent_acc: float = 3.5 # 过高保护触发时额外施加的下降加速度
    fly_low_z: float = 1.0      # 飞得过低的终止高度阈值
    out_of_bounds_radius: float = 35.0 # 水平越界半径；短距离目标下保持较紧范围

    # ── Normalisation ─────────────────────────────────────────────────────────
    norm_pos_scale: float = 50.0  # x/y 位置归一化尺度，适配前向飞行课程
    norm_goal_xy_scale: float = 10.0 # 目标相对 x/y 归一化尺度，10m 课程下初始 goal_rel_x 约为 1
    norm_z_scale: float = 5.0     # z 位置独立归一化，避免高度信号过小
    norm_vel_scale: float = 5.0   # 速度归一化尺度

    # ── Reward weights ────────────────────────────────────────────────────────
    # reward 公式在 fly_forward_env.py::_get_rewards 中实现。
    # 这里的权重按“移动并接近目标才有奖励”设计；高度只作为安全边界，不再给低空存活奖励。
    move_reward_min_speed: float = 0.05     # 低于该速度时不认为无人机真正动起来
    progress_reward_weight: float = 8.0     # 真实接近目标的进度奖励权重
    forward_progress_reward_weight: float = 8.0 # 向目标方向移动的进度奖励权重
    forward_speed_reward_weight: float = 2.0 # 目标前向速度奖励权重；实际 reward 中会乘 step_dt
    target_forward_speed: float = 0.45      # 期望前向速度，单位 m/s
    forward_speed_sigma: float = 0.12       # 前向速度奖励高斯宽度；10m 中速课程使用中等宽度
    forward_speed_min_vx: float = 0.1       # 前向速度奖励的最小 vx 门槛，避免悬停拿速度奖励
    forward_vel_reward_weight: float = 2.5  # 兼容旧配置，当前 reward 不使用
    dist_reward_weight: float = 0.5         # 距离目标奖励权重；仅在移动并接近目标时生效
    dist_reward_scale: float = 0.25         # 距离奖励缩放系数；10m 初始距离下约 exp(-2.5)=0.08
    height_reward_weight: float = 0.8       # 兼容旧配置，当前 reward 不使用
    altitude_band_reward_weight: float = 0.0 # 高度范围内不再给存活奖励
    height_penalty_weight: float = 0.0      # 高度不再按 goal_z 做密集惩罚，只保留上下界约束
    fly_high_guard_penalty_weight: float = 5.0 # 接近高度上限时的轻度保护惩罚
    near_goal_radius: float = 3.0           # 目标附近减速区域半径，10m 课程下提前进入减速阶段
    near_goal_reward_radius: float = 1.8    # 低速接近奖励的触发半径，需小于减速半径，避免远处悬停刷分
    slow_near_goal_reward_weight: float = 0.0 # 不再给近目标低速密集奖励，减速只由 success 速度条件约束
    speed_reward_scale: float = 1.0         # 低速奖励中的速度衰减系数
    speed_penalty_weight: float = 0.005     # 全局速度惩罚权重，避免压制正常前飞
    speed_soft_limit: float = 1.2           # 速度软限制阈值，超过后额外惩罚
    speed_limit_penalty_weight: float = 0.0 # 超过速度软限制后的二次惩罚权重
    hover_still_speed_threshold: float = 0.08 # 远离目标时，低于该速度视为“基本悬停”
    hover_still_penalty_weight: float = 0.3 # 远离目标且基本不动时的惩罚，专门用于打掉悬停局部最优
    time_penalty_weight: float = 0.05       # 每步时间惩罚权重，鼓励更早完成任务；实际 reward 中会乘 step_dt
    action_magnitude_weight: float = 0.001  # 动作幅度惩罚权重，只作用于当前使用的 action[0]
    upright_penalty_weight: float = 0.5     # 姿态偏离竖直 / 水平稳定状态的惩罚权重
    action_smoothness_weight: float = 0.002 # 动作平滑惩罚权重，避免过强约束前向动作
    success_forward_x: float = 8.0          # 判定成功前至少需要完成的前向位移，单位 m
    success_reward: float = 2000.0          # 成功到达目标后的保底终止奖励，需高于超时悬停累计 dense reward
    success_early_bonus: float = 2000.0     # 越早成功额外奖励越高：bonus * 剩余 episode 比例
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
