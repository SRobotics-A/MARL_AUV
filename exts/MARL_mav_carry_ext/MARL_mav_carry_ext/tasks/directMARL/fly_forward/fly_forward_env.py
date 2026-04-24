"""Single-drone fly-forward environment (DirectRLEnv, PPO).

Task: Fly to a goal sampled on a 50m-radius circle around the spawn point.
      A fresh azimuth is drawn uniformly in [0, 2π) every reset.
Control: ACCBR (velocity command + body-rate command, 6-dim continuous action).
"""

from __future__ import annotations

import csv
import os
import torch
from pathlib import Path

from MARL_mav_carry_ext.controllers import GeometricController, IndiController
from MARL_mav_carry_ext.controllers.motor_model import RotorMotor

import isaaclab.sim as sim_utils
import isaacsim.core.utils.prims as prim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import matrix_from_quat
from pxr import UsdPhysics

from .fly_forward_env_cfg import FlyForwardEnvCfg


class FlyForwardEnv(DirectRLEnv):
    """Single-drone forward-flight environment for DDPG training.

    Observation (21-dim):
        drone_pos_norm(3) | drone_lin_vel_norm(3) | rot_matrix(9) |
        drone_ang_vel_norm(3) | goal_rel_norm(3)

    Action (6-dim):
        [vx_cmd, vy_cmd, vz_cmd, roll_rate, pitch_rate, yaw_rate] in [-1, 1].
        vx/vy use lin_vel_max, vz uses conservative asymmetric up/down limits.
    """

    cfg: FlyForwardEnvCfg

    def __init__(self, cfg: FlyForwardEnvCfg, render_mode: str | None = None, **kwargs):
        # super().__init__ calls _setup_scene, making env_origins available afterwards
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)

        # ── Body/rotor index cache ────────────────────────────────────────────
        self._falcon_body_idx = self._robot.find_bodies(".*base_link")[0]  # list[int]
        self._rotor_idx = self._robot.find_bodies(".*rotor_.*")[0]         # list[int]

        # ── Physics buffers ───────────────────────────────────────────────────
        # _forces: (N, 4, 3) — per-rotor thrust in body z
        self._forces = torch.zeros(self.num_envs, 4, 3, device=self.device)
        # _moments: (N, 1, 3) — reaction torque on base_link
        self._moments = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._prev_vel_error = torch.zeros(self.num_envs, 3, device=self.device)
        self._prev_actions = torch.zeros(self.num_envs, 6, device=self.device)
        self._drone_prev_acc = torch.zeros(self.num_envs, 3, device=self.device)
        self._ll_counter: int = 0

        # ── Controllers ───────────────────────────────────────────────────────
        # hover omega = sqrt(m*g / (4*k_tau)) ≈ 972 rad/s for Falcon
        # 原来用 1355 rad/s，是悬停的 1.94×，reset 后产生 +9 m/s² 瞬时向上加速度
        _hover_omega = 972.0
        self._geo_ctrl = GeometricController(self.num_envs, "ACCBR")
        self._indi_ctrl = IndiController(self.num_envs)
        self._motor_model = RotorMotor(
            self.num_envs,
            torch.full((self.num_envs, 4), _hover_omega, device=self.device),
        )

        # ── Diagnostic logger (FLY_FORWARD_DIAG=1 to enable) ─────────────────
        self._diag_enabled = os.environ.get("FLY_FORWARD_DIAG", "0") == "1"
        self._diag_random = os.environ.get("FLY_FORWARD_RANDOM", "0") == "1"
        if self._diag_enabled:
            self._diag_path = os.environ.get("FLY_FORWARD_DIAG_PATH", "/tmp/fly_forward_diag.csv")
            self._diag_file = open(self._diag_path, "w", newline="")
            self._diag_writer = csv.writer(self._diag_file)
            mode = "random" if self._diag_random else "policy"
            self._diag_writer.writerow(
                ["mode", "episode", "policy_step",
                 "x", "z", "vx", "vz",
                 "a0", "a1", "a2",
                 "cmd_acc_x", "cmd_acc_y", "cmd_acc_z"]
            )
            self._diag_episode: int = 0
            self._diag_step: int = 0
            self._diag_prev_done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            print(f"[FlyForwardEnv] Diagnostic logger → {self._diag_path}  mode={mode}")
        self.sampling_time = self.sim.get_physics_dt() * self.cfg.low_level_decimation

        # ── Goal world position (assigned per-env at reset) ───────────────────
        self._goal_w = torch.zeros(self.num_envs, 3, device=self.device)

        # ── Spawn world position (圆心；_reset_idx 写入，OOB 检查使用) ─────────
        self._spawn_w = torch.zeros(self.num_envs, 3, device=self.device)

        # ── Progress tracking: horizontal distance to goal at last step ──────
        self._prev_dist_xy = torch.zeros(self.num_envs, device=self.device)

        # ── Termination flags ─────────────────────────────────────────────────
        self._fly_high = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._fly_low = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._out_of_bounds = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # ── Episode sum logging ───────────────────────────────────────────────
        self._episode_sums = {
            k: torch.zeros(self.num_envs, device=self.device)
            for k in [
                "progress_reward",
                "dist_reward",
                "height_reward",
                "height_penalty",
                "fly_high_guard_penalty",
                "upright_penalty",
                "action_smoothness",
                "success_reward",
                "crash_penalty",
            ]
        }

    # ── Scene setup ──────────────────────────────────────────────────────────

    def _setup_scene(self):
        """从 fly_forward.usda 加载场景（对齐 move 任务范式）。

        USD 内含：Rivermark 室外环境 + 1 架 Falcon 无人机。
        步骤：
          1. 加载地面平面（物理碰撞）
          2. spawn_from_usd → clone_environments
          3. resolve falcon prim → 绑定 Articulation（spawn=None）
          4. 补充环境光照
        """
        # ── 1. 地面平面 ──────────────────────────────────────────────────────
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # ── 2. 加载 fly_forward USD 场景 ──────────────────────────────────────
        scene_usd_path = (
            Path(__file__).resolve().parents[3]
            / "assets/data/AMR/fly_forward/fly_forward.usda"
        )
        scene_cfg = sim_utils.UsdFileCfg(usd_path=str(scene_usd_path))
        sim_utils.spawn_from_usd(prim_path="/World/envs/env_0/World", cfg=scene_cfg)

        # ── 3. 克隆到所有并行 env ─────────────────────────────────────────────
        self.scene.clone_environments(copy_from_source=False)

        # ── 4. 确定 env_0 的实际根路径 ───────────────────────────────────────
        env_root_base = "/World/envs/env_0"
        env_root = env_root_base
        if prim_utils.is_prim_path_valid(f"{env_root_base}/World"):
            env_root = f"{env_root_base}/World"

        def resolve_agent_prim_path(agent_name: str) -> str:
            """定位 env_0 中指定 agent 的 prim 路径（与 move 任务逻辑一致）。"""
            roots = [env_root]
            if prim_utils.is_prim_path_valid(f"{env_root}/World"):
                roots.append(f"{env_root}/World")

            for root in roots:
                for candidate in [
                    f"{root}/{agent_name}/Robot",
                    f"{root}/{agent_name}/Falcon",
                    f"{root}/{agent_name}",
                ]:
                    if prim_utils.is_prim_path_valid(candidate):
                        return candidate

                # instanceable prim 处理
                outer_path = f"{root}/{agent_name}"
                if prim_utils.is_prim_path_valid(outer_path):
                    import omni.usd
                    _stage = omni.usd.get_context().get_stage()
                    _prim = _stage.GetPrimAtPath(outer_path)
                    if _prim.IsValid() and _prim.IsInstance():
                        proto = _prim.GetPrototype()
                        if proto:
                            for child in proto.GetAllChildren():
                                if child.HasAPI(UsdPhysics.ArticulationRootAPI):
                                    return f"{outer_path}/{child.GetName()}"

                prims = sim_utils.get_all_matching_child_prims(
                    root, predicate=lambda p: p.GetName() == agent_name
                )
                if prims:
                    return prims[0].GetPath().pathString
                prims = sim_utils.get_all_matching_child_prims(
                    root, predicate=lambda p: agent_name in p.GetName()
                )
                if prims:
                    return prims[0].GetPath().pathString

            prim = sim_utils.find_first_matching_prim(f"{env_root_base}.*/{agent_name}(/.*)?")
            if prim is not None:
                return prim.GetPath().pathString
            raise RuntimeError(f"Could not resolve prim path for agent '{agent_name}' under {env_root_base}.")

        # ── 5. 绑定 Falcon Articulation（spawn=None）─────────────────────────
        env0_prim = resolve_agent_prim_path("falcon")
        env_prim_pattern = env0_prim.replace(env_root_base, "/World/envs/env_.*", 1)

        robot_cfg = self.cfg.robot_cfg.replace(prim_path=env_prim_pattern)
        robot_cfg.spawn = None
        self._robot = Articulation(robot_cfg)
        self.scene.articulations["robot"] = self._robot

        # ── 6. 补充环境光照 ───────────────────────────────────────────────────
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ── Action pipeline ──────────────────────────────────────────────────────

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Parse 6-dim ACCBR action → velocity/body-rate setpoints."""
        if self._diag_random:
            actions = torch.rand_like(actions) * 2.0 - 1.0
        self._actions = actions.clone()

        current_pos = self._robot.data.root_pos_w - self.scene.env_origins
        z = current_pos[:, 2]
        high_guard = torch.clamp(
            (z - self.cfg.fly_high_guard_z) / (self.cfg.fly_high_z - self.cfg.fly_high_guard_z),
            0.0,
            1.0,
        )
        upward_scale = 1.0 - high_guard

        desired_vel = torch.empty_like(actions[:, :3])
        desired_vel[:, :2] = actions[:, :2] * self.cfg.lin_vel_max
        z_action = actions[:, 2]
        desired_vel[:, 2] = torch.where(
            z_action >= 0.0,
            z_action * self.cfg.lin_vel_z_up_max * upward_scale,
            z_action * self.cfg.lin_vel_z_down_max,
        )
        current_vel = self._robot.data.root_lin_vel_w
        vel_error = desired_vel - current_vel
        d_error = (vel_error - self._prev_vel_error) / self.step_dt
        self._prev_vel_error = vel_error.clone()

        commanded_acc = self.cfg.vel_Kp * vel_error + self.cfg.vel_Kd * d_error
        commanded_acc[:, :2] = torch.clamp(
            commanded_acc[:, :2], -self.cfg.lin_acc_max, self.cfg.lin_acc_max
        )
        max_up_acc = self.cfg.lin_acc_z_up_max * upward_scale
        commanded_acc[:, 2] = torch.clamp_min(commanded_acc[:, 2], -self.cfg.lin_acc_z_down_max)
        commanded_acc[:, 2] = torch.minimum(commanded_acc[:, 2], max_up_acc)
        descent_cap = torch.where(
            high_guard > 0.0,
            -self.cfg.fly_high_guard_descent_acc * high_guard,
            max_up_acc,
        )
        commanded_acc[:, 2] = torch.minimum(commanded_acc[:, 2], descent_cap)

        self._setpoint = {
            "lin_acc": commanded_acc,
            "body_rates": actions[:, 3:6] * self.cfg.ang_vel_max,
            "yaw": torch.zeros(self.num_envs, 1, device=self.device),
            "yaw_rate": actions[:, 5:6] * self.cfg.ang_vel_max,
            "yaw_acc": torch.zeros(self.num_envs, 1, device=self.device),
        }

        if self._diag_enabled:
            self._diag_step += 1
            root = self._robot.data.root_state_w
            pos_l = root[0, :3] - self.scene.env_origins[0]
            vel = root[0, 7:10]
            a = actions[0]
            ca = commanded_acc[0]
            mode = "random" if self._diag_random else "policy"
            self._diag_writer.writerow([
                mode, self._diag_episode, self._diag_step,
                f"{pos_l[0].item():.4f}", f"{pos_l[2].item():.4f}",
                f"{vel[0].item():.4f}", f"{vel[2].item():.4f}",
                f"{a[0].item():.4f}", f"{a[1].item():.4f}", f"{a[2].item():.4f}",
                f"{ca[0].item():.4f}", f"{ca[1].item():.4f}", f"{ca[2].item():.4f}",
            ])

    def _apply_action(self) -> None:
        if self._ll_counter % self.cfg.low_level_decimation == 0:
            root = self._robot.data.root_state_w
            drone_pos = root[:, :3] - self.scene.env_origins
            drone_quat = root[:, 3:7]
            drone_lin_vel = root[:, 7:10]
            drone_ang_vel = root[:, 10:13]
            body_acc = self._robot.data.body_acc_w
            drone_lin_acc = body_acc[:, 0, :3]
            drone_ang_acc = body_acc[:, 0, 3:6]
            jerk = (drone_lin_acc - self._drone_prev_acc) / self.physics_dt
            self._drone_prev_acc = drone_lin_acc.clone()

            drone_states = {
                "pos": drone_pos,
                "quat": drone_quat,
                "lin_vel": drone_lin_vel,
                "ang_vel": drone_ang_vel,
                "lin_acc": drone_lin_acc,
                "ang_acc": drone_ang_acc,
                "jerk": jerk,
            }

            alpha_cmd, acc_load, acc_cmd, _ = self._geo_ctrl.getCommand(
                drone_states, self._forces, self._setpoint
            )

            target_rpm = self._indi_ctrl.getCommand(
                drone_states, self._forces, alpha_cmd, acc_cmd, acc_load
            )
            thrusts, moments = self._motor_model.get_motor_thrusts_moments(
                target_rpm, self.sampling_time
            )

            forces = torch.clamp(thrusts, min=0.0, max=self.cfg.max_thrust_pp)
            self._forces[..., 2] = forces
            self._moments[:, 0, 2] = torch.clamp(moments, -1.0, 1.0).sum(-1)
            self._ll_counter = 0
        self._ll_counter += 1

        # Apply torques on base_link (idx 0)
        self._robot.set_external_force_and_torque(
            forces=torch.zeros(self.num_envs, 1, 3, device=self.device),
            torques=self._moments,
            body_ids=torch.zeros(1, dtype=torch.int, device=self.device),
        )

        # Apply rotor thrusts
        self._robot.set_external_force_and_torque(
            forces=self._forces,
            torques=torch.zeros_like(self._forces),
            body_ids=self._rotor_idx,
        )

    # ── Observations ─────────────────────────────────────────────────────────

    def _get_observations(self) -> dict:
        root = self._robot.data.root_state_w
        pos_local = root[:, :3] - self.scene.env_origins   # (N,3)
        lin_vel = root[:, 7:10]
        ang_vel = root[:, 10:13]
        rot_mat = matrix_from_quat(root[:, 3:7]).view(self.num_envs, 9)
        goal_rel = self._goal_w - root[:, :3]              # (N,3)

        # z 用独立缩放 norm_z_scale (5m) 使策略能感知 2-4m 高度变化
        # x/y 用 norm_pos_scale (250m) 以适配 200m 任务范围
        pos_norm = torch.stack(
            [pos_local[:, 0] / self.cfg.norm_pos_scale,
             pos_local[:, 1] / self.cfg.norm_pos_scale,
             pos_local[:, 2] / self.cfg.norm_z_scale],
            dim=-1,
        )
        goal_rel_norm = torch.stack(
            [goal_rel[:, 0] / self.cfg.norm_pos_scale,
             goal_rel[:, 1] / self.cfg.norm_pos_scale,
             goal_rel[:, 2] / self.cfg.norm_z_scale],
            dim=-1,
        )

        obs = torch.cat(
            [
                pos_norm,
                lin_vel / self.cfg.norm_vel_scale,
                rot_mat,
                ang_vel / self.cfg.ang_vel_max,
                goal_rel_norm,
            ],
            dim=-1,
        )  # (N, 21)
        return {"policy": obs}

    # ── Rewards ───────────────────────────────────────────────────────────────

    def _get_rewards(self) -> torch.Tensor:
        root = self._robot.data.root_state_w
        pos_w = root[:, :3]
        pos_local = pos_w - self.scene.env_origins
        rot_mat = matrix_from_quat(root[:, 3:7])
        z_body_z = rot_mat[:, 2, 2]   # cos(tilt)
        step_dt = self.step_dt

        # 1. Progress toward goal (dense shaping, direction-agnostic)
        dist_xy = torch.norm((pos_w - self._goal_w)[:, :2], dim=-1)
        delta_dist = self._prev_dist_xy - dist_xy   # positive when approaching
        progress_rew = delta_dist * self.cfg.progress_reward_weight * step_dt
        self._prev_dist_xy = dist_xy.clone()

        # 2. Distance to goal (exponential decay)
        dist = torch.norm(pos_w - self._goal_w, dim=-1)
        dist_rew = self.cfg.dist_reward_weight * torch.exp(
            -dist * self.cfg.dist_reward_scale
        ) * step_dt

        # 3. Height anchor and soft penalty above target altitude
        z = pos_local[:, 2]
        height_error = z - self.cfg.goal_z
        height_rew = self.cfg.height_reward_weight * torch.exp(
            -torch.abs(height_error)
        ) * step_dt
        height_pen = -self.cfg.height_penalty_weight * torch.relu(height_error).square() * step_dt
        fly_high_guard_pen = (
            -self.cfg.fly_high_guard_penalty_weight
            * torch.relu(z - self.cfg.fly_high_guard_z).square()
            * step_dt
        )

        # 4. Upright penalty (negative when tilted)
        upright_pen = self.cfg.upright_penalty_weight * (z_body_z - 1.0) * step_dt

        # 5. Action smoothness penalty
        delta_a = self._actions - self._prev_actions
        smooth_rew = -self.cfg.action_smoothness_weight * delta_a.square().sum(-1) * step_dt
        self._prev_actions = self._actions.clone()

        # 6. Success bonus (sparse)
        success_rew = self._success.float() * self.cfg.success_reward

        # 7. Termination penalties (fixed, not × dt)
        crash_pen = -(
            self._fly_high.float() * self.cfg.fly_high_penalty
            + self._fly_low.float() * self.cfg.fly_low_penalty
            + self._out_of_bounds.float() * self.cfg.out_of_bounds_penalty
        )

        total = (
            progress_rew + dist_rew + height_rew + height_pen + fly_high_guard_pen + upright_pen
            + smooth_rew + success_rew + crash_pen
        )

        self._episode_sums["progress_reward"] += progress_rew
        self._episode_sums["dist_reward"] += dist_rew
        self._episode_sums["height_reward"] += height_rew
        self._episode_sums["height_penalty"] += height_pen
        self._episode_sums["fly_high_guard_penalty"] += fly_high_guard_pen
        self._episode_sums["upright_penalty"] += upright_pen
        self._episode_sums["action_smoothness"] += smooth_rew
        self._episode_sums["success_reward"] += success_rew
        self._episode_sums["crash_penalty"] += crash_pen

        return total

    # ── Terminations ──────────────────────────────────────────────────────────

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        root = self._robot.data.root_state_w
        pos_local = root[:, :3] - self.scene.env_origins
        z = pos_local[:, 2]
        x = pos_local[:, 0]
        y = pos_local[:, 1]

        self._fly_high = z > self.cfg.fly_high_z
        self._fly_low = z < self.cfg.fly_low_z
        # OOB：偏离 spawn 中心（水平面）超过 out_of_bounds_radius
        dist_from_spawn_xy = torch.norm((root[:, :3] - self._spawn_w)[:, :2], dim=-1)
        self._out_of_bounds = dist_from_spawn_xy > self.cfg.out_of_bounds_radius
        self._success = (
            torch.norm(root[:, :3] - self._goal_w, dim=-1) < self.cfg.goal_tolerance
        )

        terminated = self._fly_high | self._fly_low | self._out_of_bounds | self._success
        timed_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, timed_out

    # ── Reset ─────────────────────────────────────────────────────────────────

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        # Logging
        extras = {}
        for key, buf in self._episode_sums.items():
            extras[f"Episode_Reward/{key}"] = buf[env_ids].mean().item()
            buf[env_ids] = 0.0
        self.extras["log"] = extras
        self.extras["log"]["Episode_Termination/fly_high"] = int(self._fly_high[env_ids].sum())
        self.extras["log"]["Episode_Termination/fly_low"] = int(self._fly_low[env_ids].sum())
        self.extras["log"]["Episode_Termination/out_of_bounds"] = int(self._out_of_bounds[env_ids].sum())
        self.extras["log"]["Episode_Termination/success"] = int(self._success[env_ids].sum())

        # Reset robot
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        n = len(env_ids)
        # 固定起点 (spawn_x, spawn_y, spawn_z) + 小随机扰动（训练鲁棒性）
        x_noise = (torch.rand(n, device=self.device) * 2.0 - 1.0) * self.cfg.spawn_x_noise
        y_noise = (torch.rand(n, device=self.device) * 2.0 - 1.0) * self.cfg.spawn_y_noise
        spawn_pos = self.scene.env_origins[env_ids].clone()
        spawn_pos[:, 0] += self.cfg.spawn_x + x_noise
        spawn_pos[:, 1] += self.cfg.spawn_y + y_noise
        spawn_pos[:, 2] = self.cfg.spawn_z

        # ── 随机目标：以 nominal spawn 为圆心，goal_radius 为半径，方向均匀采样 ──
        angles = torch.rand(n, device=self.device) * (2.0 * torch.pi)
        self._goal_w[env_ids, 0] = (
            self.scene.env_origins[env_ids, 0]
            + self.cfg.spawn_x
            + self.cfg.goal_radius * torch.cos(angles)
        )
        self._goal_w[env_ids, 1] = (
            self.scene.env_origins[env_ids, 1]
            + self.cfg.spawn_y
            + self.cfg.goal_radius * torch.sin(angles)
        )
        self._goal_w[env_ids, 2] = self.scene.env_origins[env_ids, 2] + self.cfg.goal_z

        # 记录 spawn 世界坐标（OOB 使用；圆心与实际落点一致）
        self._spawn_w[env_ids] = spawn_pos

        root_state = self._robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] = spawn_pos
        root_state[:, 3] = 1.0
        root_state[:, 4:7] = 0.0
        root_state[:, 7:] = 0.0  # zero velocity

        self._robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(
            self._robot.data.default_joint_pos[env_ids],
            self._robot.data.default_joint_vel[env_ids],
            None,
            env_ids,
        )

        # Reset control buffers
        self._prev_vel_error[env_ids] = 0.0
        self._prev_actions[env_ids] = 0.0
        self._drone_prev_acc[env_ids] = 0.0
        self._forces[env_ids] = 0.0
        self._moments[env_ids] = 0.0
        # 重置电机转速到悬停值，避免跨 episode 积累导致初始过推力
        self._motor_model.reset(env_ids)

        if self._diag_enabled and 0 in env_ids:
            self._diag_episode += 1
            self._diag_step = 0

        # 初始化水平距离缓存（progress reward 的基准）
        self._prev_dist_xy[env_ids] = torch.norm(
            (spawn_pos - self._goal_w[env_ids])[:, :2], dim=-1
        )
