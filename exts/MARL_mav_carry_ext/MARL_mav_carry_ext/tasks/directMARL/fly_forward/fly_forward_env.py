"""Single-drone fly-forward environment (DirectRLEnv, DDPG).

Task: Fly 200m in the +x direction. Altitude must stay below 4m.
Control: ACCBR (velocity command + body-rate command, 6-dim continuous action).
"""

from __future__ import annotations

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

    Action (6-dim, scaled by lin/ang vel max):
        [vx_cmd, vy_cmd, vz_cmd, roll_rate, pitch_rate, yaw_rate] ∈ [-1, 1]
    """

    cfg: FlyForwardEnvCfg

    def __init__(self, cfg: FlyForwardEnvCfg, render_mode: str | None = None, **kwargs):
        # super().__init__ calls _setup_scene, making env_origins available afterwards
        print("[fly_forward] __init__: calling super().__init__ (this triggers _setup_scene + sim.reset) ...")
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)
        print("[fly_forward] __init__: super().__init__ DONE")

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
        self._geo_ctrl = GeometricController(self.num_envs, "ACCBR")
        self._indi_ctrl = IndiController(self.num_envs)
        self._motor_model = RotorMotor(
            self.num_envs,
            torch.full((self.num_envs, 4), 1355.0, device=self.device),
        )
        self.sampling_time = self.sim.get_physics_dt() * self.cfg.low_level_decimation

        # ── Goal world position ───────────────────────────────────────────────
        goal_offset = torch.tensor(
            [cfg.goal_x, cfg.goal_y, cfg.goal_z], device=self.device
        )
        self._goal_w = self.scene.env_origins + goal_offset  # (N, 3)

        # ── Progress tracking ─────────────────────────────────────────────────
        self._prev_x = torch.zeros(self.num_envs, device=self.device)

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
        import time as _time

        # ── 1. 地面平面 ──────────────────────────────────────────────────────
        print("[fly_forward] step 1: spawn_ground_plane ...")
        _t0 = _time.time()
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        print(f"[fly_forward] step 1 done ({_time.time() - _t0:.2f}s)")

        # ── 2. 加载 fly_forward USD 场景 ──────────────────────────────────────
        scene_usd_path = (
            Path(__file__).resolve().parents[3]
            / "assets/data/AMR/fly_forward/fly_forward.usda"
        )
        print(f"[fly_forward] step 2: spawn_from_usd({scene_usd_path}) ...")
        _t0 = _time.time()
        scene_cfg = sim_utils.UsdFileCfg(usd_path=str(scene_usd_path))
        sim_utils.spawn_from_usd(prim_path="/World/envs/env_0/World", cfg=scene_cfg)
        print(f"[fly_forward] step 2 done ({_time.time() - _t0:.2f}s)")

        # ── 3. 克隆到所有并行 env ─────────────────────────────────────────────
        print(f"[fly_forward] step 3: clone_environments (num_envs={self.scene.cfg.num_envs}) ...")
        _t0 = _time.time()
        self.scene.clone_environments(copy_from_source=False)
        print(f"[fly_forward] step 3 done ({_time.time() - _t0:.2f}s)")

        # ── 4. 确定 env_0 的实际根路径 ───────────────────────────────────────
        env_root_base = "/World/envs/env_0"
        env_root = env_root_base
        if prim_utils.is_prim_path_valid(f"{env_root_base}/World"):
            env_root = f"{env_root_base}/World"
        print(f"[fly_forward] step 4: env_root = {env_root}")

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
                    print(f"[fly_forward]   checking: {candidate} -> {prim_utils.is_prim_path_valid(candidate)}")
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
        print("[fly_forward] step 5: resolve falcon prim ...")
        env0_prim = resolve_agent_prim_path("falcon")
        env_prim_pattern = env0_prim.replace(env_root_base, "/World/envs/env_.*", 1)
        print(f"[fly_forward] step 5: env0_prim={env0_prim}, pattern={env_prim_pattern}")

        _t0 = _time.time()
        robot_cfg = self.cfg.robot_cfg.replace(prim_path=env_prim_pattern)
        robot_cfg.spawn = None
        print(f"[fly_forward] step 5: creating Articulation(spawn=None, prim_path={env_prim_pattern}) ...")
        self._robot = Articulation(robot_cfg)
        self.scene.articulations["robot"] = self._robot
        print(f"[fly_forward] step 5 done ({_time.time() - _t0:.2f}s)")

        # ── 6. 补充环境光照 ───────────────────────────────────────────────────
        print("[fly_forward] step 6: light ...")
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        print("[fly_forward] _setup_scene COMPLETE")

    # ── Action pipeline ──────────────────────────────────────────────────────

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Parse 6-dim ACCBR action → velocity/body-rate setpoints."""
        self._actions = actions.clone()

        desired_vel = actions[:, :3] * self.cfg.lin_vel_max
        current_vel = self._robot.data.root_lin_vel_w
        vel_error = desired_vel - current_vel
        d_error = (vel_error - self._prev_vel_error) / self.step_dt
        self._prev_vel_error = vel_error.clone()

        commanded_acc = self.cfg.vel_Kp * vel_error + self.cfg.vel_Kd * d_error
        commanded_acc = torch.clamp(commanded_acc, -self.cfg.lin_acc_max, self.cfg.lin_acc_max)

        self._setpoint = {
            "lin_acc": commanded_acc,
            "body_rates": actions[:, 3:6] * self.cfg.ang_vel_max,
            "yaw": torch.zeros(self.num_envs, 1, device=self.device),
            "yaw_rate": actions[:, 5:6] * self.cfg.ang_vel_max,
            "yaw_acc": torch.zeros(self.num_envs, 1, device=self.device),
        }

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

        obs = torch.cat(
            [
                pos_local / self.cfg.norm_pos_scale,
                lin_vel / self.cfg.norm_vel_scale,
                rot_mat,
                ang_vel / self.cfg.ang_vel_max,
                goal_rel / self.cfg.norm_pos_scale,
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

        # 1. X-direction progress (dense shaping)
        delta_x = pos_local[:, 0] - self._prev_x
        progress_rew = delta_x * self.cfg.progress_reward_weight
        self._prev_x = pos_local[:, 0].clone()

        # 2. Distance to goal (exponential decay)
        dist = torch.norm(pos_w - self._goal_w, dim=-1)
        dist_rew = self.cfg.dist_reward_weight * torch.exp(
            -dist * self.cfg.dist_reward_scale
        ) * step_dt

        # 3. Height anchor
        z = pos_local[:, 2]
        height_rew = self.cfg.height_reward_weight * torch.exp(
            -torch.abs(z - self.cfg.goal_z)
        ) * step_dt

        # 4. Upright penalty (negative when tilted)
        upright_pen = self.cfg.upright_penalty_weight * (z_body_z - 1.0) * step_dt

        # 5. Action smoothness
        delta_a = self._actions - self._prev_actions
        smooth_rew = self.cfg.action_smoothness_weight * torch.exp(
            -(delta_a ** 2).sum(-1)
        ) * step_dt
        self._prev_actions = self._actions.clone()

        # 6. Success bonus (sparse)
        success_rew = self._success.float() * self.cfg.success_reward

        # 7. Crash penalty (fixed, not × dt)
        crash = (self._fly_high | self._fly_low | self._out_of_bounds).float()
        crash_pen = -crash * self.cfg.fly_high_penalty

        total = (
            progress_rew + dist_rew + height_rew + upright_pen
            + smooth_rew + success_rew + crash_pen
        )

        self._episode_sums["progress_reward"] += progress_rew
        self._episode_sums["dist_reward"] += dist_rew
        self._episode_sums["height_reward"] += height_rew
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
        self._out_of_bounds = (
            (torch.abs(y) > self.cfg.out_of_bounds_y)
            | (x < self.cfg.out_of_bounds_x_min)
        )
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

        # Update goal for these envs (in case env_origins changed, though they don't)
        goal_offset = torch.tensor(
            [self.cfg.goal_x, self.cfg.goal_y, self.cfg.goal_z], device=self.device
        )
        self._goal_w[env_ids] = self.scene.env_origins[env_ids] + goal_offset

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

        # Reset prev_x to actual spawn local-x
        self._prev_x[env_ids] = self.cfg.spawn_x + x_noise
