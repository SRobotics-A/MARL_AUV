"""Configuration for the single-drone fly-forward task (DDPG).

Task: Fly 200m in the +x direction from (-8, 3, 2).
      Altitude must stay below 4m.
Scene: fly_forward.usda (Rivermark outdoor + single Falcon drone).
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
    """Configuration for the fly-forward single-drone DDPG task."""

    # ── Control ──────────────────────────────────────────────────────────────
    control_mode: str = "ACCBR"  # 6-dim: [vx,vy,vz,roll_rate,pitch_rate,yaw_rate]
    lin_vel_max: float = 3.0     # m/s, x/y command limit
    lin_vel_z_up_max: float = 0.25    # m/s, conservative upward command limit
    lin_vel_z_down_max: float = 1.5  # m/s, keep enough authority to recover from fly_high
    ang_vel_max: float = 0.5     # rad/s
    lin_acc_max: float = 2.0     # m/s², x/y acceleration limit
    lin_acc_z_up_max: float = 0.3    # m/s², conservative upward acceleration limit
    lin_acc_z_down_max: float = 2.0  # m/s², allow stronger descent correction
    vel_Kp: float = 2.0
    vel_Kd: float = 0.3

    # ── Episode ───────────────────────────────────────────────────────────────
    decimation: int = 3
    episode_length_s: float = 120.0

    # ── Spaces ────────────────────────────────────────────────────────────────
    # obs: pos(3) + lin_vel(3) + rot_mat(9) + ang_vel(3) + goal_rel(3) = 21
    # 注意：必须用有界 Box，否则 SKRL random_act 从 Uniform(-inf,+inf) 采样得到 NaN
    action_space: Box = Box(low=-1.0, high=1.0, shape=(6,))
    observation_space: int = 21
    state_space: int = 0

    # ── Goal ──────────────────────────────────────────────────────────────────
    goal_x: float = 20.0
    goal_y: float = 0.0
    goal_z: float = 2.0
    goal_tolerance: float = 10.0  # success radius (m)

    # ── Spawn ─────────────────────────────────────────────────────────────────
    spawn_x: float = -8.0        # 固定起点 x（与 USDA 一致）
    spawn_y: float = 3.0         # 固定起点 y（与 USDA 一致）
    spawn_z: float = 2.0         # 固定起点 z（与 USDA 一致）
    spawn_x_noise: float = 0.5   # 随机扰动（训练鲁棒性）
    spawn_y_noise: float = 0.5

    # ── Altitude limits ───────────────────────────────────────────────────────
    fly_high_z: float = 4.0
    fly_high_guard_z: float = 3.0
    fly_high_guard_descent_acc: float = 2.0
    fly_low_z: float = 0.3
    out_of_bounds_y: float = 25.0
    out_of_bounds_x_min: float = -15.0

    # ── Normalisation ─────────────────────────────────────────────────────────
    norm_pos_scale: float = 250.0   # for x/y (goal distance)
    norm_z_scale: float = 5.0       # for z position — 独立缩放，避免 z/250 ≈ 0.008 不可见
    norm_vel_scale: float = 5.0

    # ── Reward weights ────────────────────────────────────────────────────────
    progress_reward_weight: float = 5.0
    dist_reward_weight: float = 2.0
    dist_reward_scale: float = 0.01
    height_reward_weight: float = 0.8
    height_penalty_weight: float = 5.0
    fly_high_guard_penalty_weight: float = 8.0
    upright_penalty_weight: float = 3.0
    action_smoothness_weight: float = 0.5
    success_reward: float = 200.0
    fly_high_penalty: float = 30.0
    fly_low_penalty: float = 10.0
    out_of_bounds_penalty: float = 10.0
    # ── 单步惩罚硬上限（防止关闭 fly_high 终止时二次项量级爆炸）─────────────────
    # z=4m 时 height_pen≈-0.2/step、guard_pen≈-0.08/step；clip 在 -5/-3 确保
    # 梯度在正常区完整保留，异常高度时 episode 累计 < ~96k（远低于 1e6）
    height_penalty_clip: float = 5.0       # 单步下界：height_pen >= -clip
    fly_high_guard_penalty_clip: float = 3.0  # 单步下界：guard_pen >= -clip

    # ── Simulation ────────────────────────────────────────────────────────────
    sim: SimulationCfg = SimulationCfg(
        dt=0.0033333333333333335,
        render_interval=decimation,
        gravity=(0.0, 0.0, -9.8066),
    )

    # ── Scene USD（fly_forward.usda 包含 Rivermark + Falcon）─────────────────
    # 路径在 _setup_scene 中通过 Path(__file__) 动态解析，无需在此硬编码

    # ── Robot cfg（_setup_scene 中设 spawn=None，prim 由 USDA 定义）─────────────
    robot_cfg: ArticulationCfg = FALCON_CFG.replace(
        prim_path="/World/envs/env_.*/falcon",  # 占位，_setup_scene 中动态替换
    )

    # scene — large env_spacing in x so 200m flights don't cross into other envs
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=16, env_spacing=220.0, replicate_physics=True
    )

    # ── Collision contact threshold ───────────────────────────────────────────
    contact_sensor_threshold: float = 50.0

    # ── Low-level control ─────────────────────────────────────────────────────
    low_level_decimation: int = 1
    max_thrust_pp: float = 6.25  # N per rotor
