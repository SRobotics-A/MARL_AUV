"""Configuration for the single-drone fly-forward task (DDPG).

Task: Fly 200m in the +x direction from the start position.
      Altitude must stay below 4m.
"""

from __future__ import annotations

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

import isaaclab.sim as sim_utils

from MARL_mav_carry_ext.assets import FALCON_CFG


@configclass
class FlyForwardEnvCfg(DirectRLEnvCfg):
    """Configuration for the fly-forward single-drone DDPG task."""

    # ── Control ──────────────────────────────────────────────────────────────
    control_mode: str = "ACCBR"  # 6-dim: [vx,vy,vz,roll_rate,pitch_rate,yaw_rate]
    lin_vel_max: float = 3.0     # m/s — faster than move task (need to cover 200m)
    ang_vel_max: float = 1.2     # rad/s
    lin_acc_max: float = 3.0     # m/s² clamp on commanded acc
    vel_Kp: float = 2.0          # velocity PD — proportional gain
    vel_Kd: float = 0.3          # velocity PD — derivative gain

    # ── Episode ───────────────────────────────────────────────────────────────
    decimation: int = 3
    episode_length_s: float = 120.0  # 200m / 3m/s = 67s min; give 120s budget

    # ── Spaces ────────────────────────────────────────────────────────────────
    # obs: pos(3) + lin_vel(3) + rot_mat(9) + ang_vel(3) + goal_rel(3) = 21
    action_space: int = 6
    observation_space: int = 21
    state_space: int = 0

    # ── Goal ──────────────────────────────────────────────────────────────────
    goal_x: float = 200.0        # +x travel distance (m)
    goal_y: float = 0.0
    goal_z: float = 2.0          # desired cruise altitude (m)
    goal_tolerance: float = 10.0  # success radius (m)

    # ── Spawn ─────────────────────────────────────────────────────────────────
    spawn_x_range: tuple = (-1.0, 1.0)   # small random offset at reset
    spawn_y_range: tuple = (-1.0, 1.0)
    spawn_z: float = 2.0

    # ── Altitude limits ───────────────────────────────────────────────────────
    fly_high_z: float = 4.0    # terminate if z > 4m
    fly_low_z: float = 0.3     # terminate if z < 0.3m
    out_of_bounds_y: float = 20.0  # terminate if |y| > 20m
    out_of_bounds_x_min: float = -5.0  # terminate if x < -5m (moved backward)

    # ── Normalisation ─────────────────────────────────────────────────────────
    norm_pos_scale: float = 250.0
    norm_vel_scale: float = 5.0

    # ── Reward weights ────────────────────────────────────────────────────────
    # Progress: reward for moving in +x direction each step
    progress_reward_weight: float = 5.0

    # Distance to goal (exponential decay)
    dist_reward_weight: float = 2.0
    dist_reward_scale: float = 0.01   # exp(-dist * scale); scale=0.01 → exp(-2)@200m

    # Height anchor: exp(-|z - goal_z|) * step_dt
    height_reward_weight: float = 0.5

    # Upright: -(1 - z_body_z) * step_dt
    upright_penalty_weight: float = 3.0

    # Action smoothness: exp(-||Δa||²) * step_dt
    action_smoothness_weight: float = 0.5

    # Success: sparse reward on reaching goal
    success_reward: float = 200.0

    # Crash penalties (fixed, not × step_dt)
    fly_high_penalty: float = 10.0
    fly_low_penalty: float = 10.0
    out_of_bounds_penalty: float = 10.0

    # ── Simulation ────────────────────────────────────────────────────────────
    sim: SimulationCfg = SimulationCfg(
        dt=0.0033333333333333335,  # 300 Hz physics
        render_interval=decimation,
        gravity=(0.0, 0.0, -9.8066),
    )

    # ground plane
    terrain: TerrainImporterCfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # robot: single Falcon, spawned by Isaac Lab into each env
    robot: ArticulationCfg = FALCON_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
    ).replace(
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 2.0),   # start at (0, 0, 2) relative to env origin
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={".*": 0.0},
            joint_vel={
                "Falcon_rotor_0_joint": 0.0,
                "Falcon_rotor_1_joint": 0.0,
                "Falcon_rotor_2_joint": 0.0,
                "Falcon_rotor_3_joint": 0.0,
            },
        )
    )

    # scene — large env_spacing in x so 200m flights don't cross into other envs
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=16, env_spacing=220.0, replicate_physics=True
    )

    # ── Scene USD ─────────────────────────────────────────────────────────────
    # Rivermark 室外场景 USD（仅在 env_0 生成，供可视化使用）
    scene_usd_path: str = (
        "/media/xtj/1CC8D044C8D01DB8/RL-download/isaac-sim/v5.1.0/Assets/Isaac/5.1/"
        "Isaac/Environments/Outdoor/Rivermark/rivermark.usd"
    )

    # ── Low-level control ─────────────────────────────────────────────────────
    low_level_decimation: int = 1
    max_thrust_pp: float = 6.25  # N per rotor
