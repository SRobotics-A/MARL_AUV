# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
from collections.abc import Sequence
from pathlib import Path

from MARL_mav_carry_ext.controllers import GeometricController, IndiController
from MARL_mav_carry_ext.controllers.motor_model import RotorMotor
from MARL_mav_carry_ext.tasks.managerbased.mdp_llc.utils import (
    get_drone_pdist,
    get_drone_rpos,
)

import isaaclab.sim as sim_utils
import isaacsim.core.utils.prims as prim_utils
from isaacsim.core.prims import XFormPrim
from isaaclab.assets import Articulation
from isaaclab.envs import DirectMARLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.sim import schemas as sim_schemas
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import CircularBuffer
from isaaclab.utils.math import (
    euler_xyz_from_quat,
    matrix_from_quat,
    quat_rotate,
)
from pxr import UsdPhysics

from .marl_move_env_cfg import MARLMoveEnvCfg


class MARLMoveEnv(DirectMARLEnv):
    cfg: MARLMoveEnvCfg

    def __init__(self, cfg: MARLMoveEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._num_drones = 3
        self._control_mode = cfg.control_mode

        # body indices (same structure for all 3 drones, just query from robot_0)
        self._falcon_idx = torch.tensor(
            self.robots[0].find_bodies(".*base_link")[0], device=self.device
        )
        self._falcon_rotor_idx = torch.tensor(
            self.robots[0].find_bodies(".*rotor_.*")[0], device=self.device
        )

        # observation buffers
        self._observation_buffers = {}
        for agent in self.cfg.possible_agents:
            self._observation_buffers[agent] = CircularBuffer(
                cfg.history_len, self.num_envs, device=self.device
            )

        # action buffers - per drone forces/moments
        # _forces: (num_envs, 4, 3) per drone — 4 rotors, 3D force
        self._forces = [
            torch.zeros(self.num_envs, 4, 3, device=self.device)
            for _ in range(self._num_drones)
        ]
        self._moments = torch.zeros(
            self.num_envs, self._num_drones, 3, device=self.device
        )

        self._setpoints = {}
        self.prev_actions = {}
        for agent in self.cfg.possible_agents:
            self._setpoints[agent] = {}
            if self._control_mode == "geometric":
                self.prev_actions[agent] = torch.zeros(
                    self.num_envs, 12, device=self.device
                )
            elif self._control_mode == "ACCBR":
                self.prev_actions[agent] = torch.zeros(
                    self.num_envs, 6, device=self.device
                )

        self.drone_positions = torch.zeros(
            self.num_envs, self._num_drones, 3, device=self.device
        )
        self.drone_orientations = torch.zeros(
            self.num_envs, self._num_drones, 4, device=self.device
        )
        self.drone_orientations[..., 0] = 1.0
        self.drone_rot_matrices = torch.zeros(
            self.num_envs, self._num_drones, 3, 3, device=self.device
        )
        self.drone_linear_velocities = torch.zeros(
            self.num_envs, self._num_drones, 3, device=self.device
        )
        self.drone_angular_velocities = torch.zeros(
            self.num_envs, self._num_drones, 3, device=self.device
        )
        self.drone_linear_accelerations = torch.zeros(
            self.num_envs, self._num_drones, 3, device=self.device
        )
        self.drone_angular_accelerations = torch.zeros(
            self.num_envs, self._num_drones, 3, device=self.device
        )
        self._drone_jerk = torch.zeros(
            self.num_envs, self._num_drones, 3, device=self.device
        )
        self._drone_prev_acc = torch.zeros(
            self.num_envs, self._num_drones, 3, device=self.device
        )
        # PD velocity controller: previous velocity error
        self._prev_vel_error = torch.zeros(
            self.num_envs, self._num_drones, 3, device=self.device
        )

        # outer loop controller
        self.geo_controllers = {}
        for i in range(self._num_drones):
            self.geo_controllers[i] = GeometricController(
                self.num_envs, self._control_mode
            )
        self._ll_counter = 0
        self._constant_yaw = torch.zeros([self.num_envs, 1], device=self.device)
        self._zeros = torch.zeros([self.num_envs, 3], device=self.device)

        # inner loop controller
        self._indi_controllers = {}
        for i in range(self._num_drones):
            self._indi_controllers[i] = IndiController(self.num_envs)

        # motor model
        self.motor_models = {}
        initial_rpms = [
            torch.tensor([[1355.0, 1355.0, 1355.0, 1355.0]], device=self.device).repeat(
                self.num_envs, 1
            )
            for _ in range(self._num_drones)
        ]
        for i in range(self._num_drones):
            self.motor_models[i] = RotorMotor(self.num_envs, initial_rpms[i])
        self.sampling_time = self.sim.get_physics_dt() * self.cfg.low_level_decimation

        # 移动物块状态缓冲区
        self.num_targets = cfg.num_targets
        self.target_positions = torch.zeros(
            self.num_envs, self.num_targets, 3, device=self.device
        )
        self.target_velocities = torch.zeros(
            self.num_envs, self.num_targets, 3, device=self.device
        )
        # 折返方向：+1 向 +x 运动，-1 向 -x 运动
        self.target_directions = torch.ones(
            self.num_envs, self.num_targets, device=self.device
        )
        self.target_captured = torch.zeros(
            self.num_envs, self.num_targets, dtype=torch.bool, device=self.device
        )
        # 预创建四元数张量，避免每帧重复创建
        self._target_quat = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]], device=self.device
        ).repeat(self.num_envs, 1)
        self.target_values = (
            torch.tensor(cfg.target_values, device=self.device)
            .unsqueeze(0)
            .repeat(self.num_envs, 1)
        )

        # 记录哪个无人机捕获了哪个物块 (-1 表示未被捕获)
        self.target_captured_by = torch.full(
            (self.num_envs, self.num_targets), -1, dtype=torch.long, device=self.device
        )

        # 奖励日志记录（动态补充，此处仅预创建）
        self._episode_sums = {
            key: torch.zeros(
                self.num_envs,
                dtype=torch.float,
                device=self.device,
            )
            for key in [
                "distance_reward",
                "success_reward",
                "tracking_reward",
                "action_smoothness",
                "body_rate_penalty",
                "velocity_penalty",
                "force_penalty",
                "height_reward",
                "collision_penalty",
                "drone_out",
                "fly_low",
                "illegal_contact",
                "time_penalty",
                "upright_penalty",
                "height_penalty",
                "boundary_soft",
            ]
        }

        # metrics
        self.metrics = {}
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["orientation_error"] = torch.zeros(
            self.num_envs, device=self.device
        )

        # 终止条件缓冲区
        self.falcon_fly_low = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.falcon_fly_high = torch.zeros(  # Run41: 新增高飞终止缓冲
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.illegal_contact = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.drone_collision = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.body_pos_outside = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.all_targets_captured = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self._sustained_follow_timer = torch.zeros(self.num_envs, device=self.device)
        # Target assignment buffer: drone_assigned_target[env, drone] = target_idx (fixed per episode)
        self.drone_assigned_target = torch.zeros(
            self.num_envs, self._num_drones, dtype=torch.long, device=self.device
        )
        # Progress reward buffer: (N, D) — per-drone distance to assigned target
        # -1.0 = first-step sentinel (skip progress on reset step)
        self._prev_min_dists = torch.full(
            (self.num_envs, self.cfg.num_drones), fill_value=-1.0, device=self.device
        )
        self.targets_out_of_bounds = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.time_out = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        # 归一化配置
        self._norm_pos_scale = self.cfg.bounding_box_threshold
        self._norm_vel_scale = 5.0
        self._norm_dist_scale = self.cfg.bounding_box_threshold * 2

        # debug vis
        self.set_debug_vis(cfg.debug_vis)

    def _setup_scene(self):
        """从 move.usda 加载完整场景（对齐 move_flyfollow 范式）。

        USD 内含：Rivermark 室外环境 + 3架 Falcon 无人机（falcon1/2/3）+ 4辆 NovaCarter 目标小车。
        步骤：
          1. 加载地面平面（物理碰撞）
          2. spawn_from_usd → clone_environments（USD 场景复制到所有并行 env）
          3. resolve_agent_prim_path 定位各 Falcon prim → 绑定 Articulation（spawn=None）
          4. 激活接触传感器 API
          5. 绑定 NovaCarter 小车为 XFormPrim（运动学，直接写位置）
        """
        # ── 1. 地面平面 ──────────────────────────────────────────────────────
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # ── 2. 加载 move USD 场景 ─────────────────────────────────────────────
        scene_usd_path = (
            Path(__file__).resolve().parents[3]
            / "assets/data/AMR/move/move.usda"
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
            """定位 env_0 中指定 agent 的带 ArticulationRootAPI 的 prim 路径。"""
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

        # ── 5. 绑定 Falcon 无人机 Articulation（spawn=None）─────────────────
        self.robots = []
        self.contact_sensors = []
        self._usd_root_state_rel = torch.zeros(
            len(self.cfg.possible_agents), 7, device=self.device
        )
        for i, agent in enumerate(self.cfg.possible_agents):
            env0_prim = resolve_agent_prim_path(agent)
            env_prim_pattern = env0_prim.replace(env_root_base, "/World/envs/env_.*", 1)

            robot_cfg = self.cfg.robot_cfg.replace(prim_path=env_prim_pattern)
            robot_cfg.spawn = None
            robot = Articulation(robot_cfg)
            self.robots.append(robot)
            self.scene.articulations[f"robot_{i}"] = robot

            # 激活接触传感器 API
            for env_id in range(self.num_envs):
                env_prim = env0_prim.replace("/env_0", f"/env_{env_id}", 1)
                sim_schemas.activate_contact_sensors(env_prim, threshold=self.cfg.contact_sensor_threshold)

            contact_cfg = self.cfg.contact_forces.replace(
                prim_path=f"{env_prim_pattern}/.*"
            )
            contact = ContactSensor(contact_cfg)
            self.contact_sensors.append(contact)
            self.scene.sensors[f"contact_forces_{i}"] = contact

            # 缓存 USD 初始位姿（相对 env_origin），reset 时恢复
            try:
                xfm = XFormPrim(env0_prim)
                pos, ori = xfm.get_world_poses()
                self._usd_root_state_rel[i, :3] = pos[0] - self.scene.env_origins[0]
                self._usd_root_state_rel[i, 3:7] = ori[0]
            except Exception as exc:
                print(f"[move] Failed to read USD pose for {env0_prim}: {exc}")

        if self.robots:
            self.scene.articulations["robot"] = self.robots[0]

        # ── 6. 补充环境光照 ───────────────────────────────────────────────────
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # ── 7. 绑定 NovaCarter 小车为 XFormPrim ──────────────────────────────
        # 小车不参与物理仿真，每步通过 set_world_poses 直接驱动位置
        self._target_prims: list[XFormPrim] = []
        self._usd_target_positions_rel = torch.zeros(self.cfg.num_targets, 3, device=self.device)
        self._usd_target_orientations = torch.zeros(self.cfg.num_targets, 4, device=self.device)
        target_prim_names = [
            "nova_carter_sim_optimized",
            "nova_carter_sim_optimized_01",
            "nova_carter_sim_optimized_02",
            "nova_carter_sim_optimized_03",
        ]
        for idx, name in enumerate(target_prim_names[: self.cfg.num_targets]):
            prim_path = f"{env_root}/{name}"
            xfm = XFormPrim(prim_path)
            self._target_prims.append(xfm)
            try:
                pos, ori = xfm.get_world_poses()
                self._usd_target_positions_rel[idx] = pos[0] - self.scene.env_origins[0]
                self._usd_target_orientations[idx] = ori[0]
            except Exception as exc:
                print(f"[move] Failed to read USD target pose for {prim_path}: {exc}")

        # ── 禁用 NovaCarter 碰撞几何体 ─────────────────────────────────────────
        # XFormPrim 运动学驱动不参与物理，但 USD 内嵌 CollisionAPI 仍会触发 ContactSensor
        # 遍历所有 env 下的 nova_carter prim 树，将所有 CollisionAPI 禁用
        import omni.usd
        from pxr import Usd
        _stage = omni.usd.get_context().get_stage()
        for env_id in range(self.num_envs):
            env_base = f"/World/envs/env_{env_id}"
            env_r = f"{env_base}/World" if prim_utils.is_prim_path_valid(f"{env_base}/World") else env_base
            for name in target_prim_names[: self.cfg.num_targets]:
                root_prim = _stage.GetPrimAtPath(f"{env_r}/{name}")
                if not root_prim.IsValid():
                    continue
                for desc in Usd.PrimRange(root_prim):
                    col_api = UsdPhysics.CollisionAPI(desc)
                    if col_api:
                        col_api.GetCollisionEnabledAttr().Set(False)
        print(f"[move] Disabled collision geometry on NovaCarter prims across {self.num_envs} envs.")

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        for agent in self.cfg.possible_agents:
            self.prev_actions[agent][:] = self.actions[agent]
            self.actions[agent][:] = actions[agent]

        for drone, action in actions.items():
            if self._control_mode == "geometric":
                self._setpoints[drone]["pos"] = (
                    action[:, :3] * self.cfg.bounding_box_threshold
                )
                self._setpoints[drone]["lin_vel"] = (
                    action[:, 3:6] * self.cfg.lin_vel_max
                )
                self._setpoints[drone]["lin_acc"] = (
                    action[:, 6:9] * self.cfg.lin_acc_max
                )
                self._setpoints[drone]["jerk"] = action[:, 9:12] * 20.0
            elif self._control_mode == "ACCBR":
                # Velocity command + PD controller
                drone_idx = self.cfg.possible_agents.index(drone)
                desired_vel = action[:, :3] * self.cfg.lin_vel_max
                current_vel = self.drone_linear_velocities[:, drone_idx]
                vel_error = desired_vel - current_vel
                d_error = (
                    vel_error - self._prev_vel_error[:, drone_idx]
                ) / self.step_dt
                self._prev_vel_error[:, drone_idx] = vel_error
                commanded_acc = self.cfg.vel_Kp * vel_error + self.cfg.vel_Kd * d_error
                commanded_acc = torch.clamp(
                    commanded_acc,
                    -self.cfg.lin_acc_max,
                    self.cfg.lin_acc_max,
                )
                self._setpoints[drone]["lin_acc"] = commanded_acc
                self._setpoints[drone]["body_rates"] = (
                    action[:, 3:6] * self.cfg.ang_vel_max
                )

            self._setpoints[drone]["yaw"] = self._constant_yaw
            self._setpoints[drone]["yaw_rate"] = (
                action[:, 5].unsqueeze(-1) * self.cfg.ang_vel_max
                if self._control_mode == "ACCBR"
                else self._constant_yaw
            )
            self._setpoints[drone]["yaw_acc"] = self._constant_yaw

    def _apply_action(self) -> None:
        if self._ll_counter % self.cfg.low_level_decimation == 0:
            all_thrusts = []
            all_moments = []

            # ===== 从每个独立 robot 读取状态 (无需索引映射!) =====
            for i, robot in enumerate(self.robots):
                # root_state_w shape: (num_envs, 13) — 每个robot独立!
                root_state = robot.data.root_state_w  # (N, 13)

                drone_pos = root_state[:, :3] - self.scene.env_origins  # 局部坐标
                drone_quat = root_state[:, 3:7]
                drone_lin_vel = root_state[:, 7:10]
                drone_ang_vel = root_state[:, 10:13]

                # body_acc_w: (N, num_bodies, 6)
                # 对于独立Falcon, base_link idx 通常是 0
                body_acc = robot.data.body_acc_w
                drone_lin_acc = body_acc[:, 0, 0:3]
                drone_ang_acc = body_acc[:, 0, 3:6]

                self.drone_positions[:, i] = drone_pos
                self.drone_orientations[:, i] = drone_quat
                self.drone_linear_velocities[:, i] = drone_lin_vel
                self.drone_angular_velocities[:, i] = drone_ang_vel
                self.drone_linear_accelerations[:, i] = drone_lin_acc
                self.drone_angular_accelerations[:, i] = drone_ang_acc

            # ===== 计算每架无人机的控制力 =====
            for i in range(self._num_drones):
                drone_states: dict = {}
                drone_states["pos"] = self.drone_positions[:, i]
                drone_states["quat"] = self.drone_orientations[:, i]
                drone_states["lin_vel"] = self.drone_linear_velocities[:, i]
                drone_states["ang_vel"] = self.drone_angular_velocities[:, i]
                drone_states["lin_acc"] = self.drone_linear_accelerations[:, i]
                drone_states["ang_acc"] = self.drone_angular_accelerations[:, i]
                # calculate current jerk
                self._drone_jerk[:, i] = (
                    drone_states["lin_acc"] - self._drone_prev_acc[:, i]
                ) / (self.physics_dt)
                drone_states["jerk"] = self._drone_jerk[:, i]
                self._drone_prev_acc[:, i] = drone_states["lin_acc"]

                agent_name = self.cfg.possible_agents[i]

                alpha_cmd, acc_load, acc_cmd, q_cmd = self.geo_controllers[
                    i
                ].getCommand(
                    drone_states,
                    self._forces[i],  # (N, 4, 3)
                    self._setpoints[agent_name],
                )

                target_rpm = self._indi_controllers[i].getCommand(
                    drone_states,
                    self._forces[i],
                    alpha_cmd,
                    acc_cmd,
                    acc_load,
                )

                thrusts, moments = self.motor_models[i].get_motor_thrusts_moments(
                    target_rpm, self.sampling_time
                )
                all_thrusts.append(thrusts)
                all_moments.append(moments)

            # ===== 限制力 =====
            for i in range(self._num_drones):
                forces_i = torch.clamp(
                    all_thrusts[i], min=0.0, max=self.cfg.max_thrust_pp
                )
                self._forces[i][..., 2] = forces_i

                moments_i = torch.clamp(all_moments[i], min=-1.0, max=1.0)
                self._moments[:, i, 2] = moments_i.sum(-1)

            self._ll_counter = 0
        self._ll_counter += 1

        # ===== 对每个robot独立施加力/扭矩 (不再互相覆盖!) =====
        for i, robot in enumerate(self.robots):
            # base_link 扭矩
            robot.set_external_force_and_torque(
                forces=torch.zeros((self.num_envs, 1, 3), device=self.device),
                torques=self._moments[:, i].unsqueeze(1),  # (N, 1, 3)
                body_ids=torch.zeros(1, dtype=torch.int, device=self.device),
            )

            # rotor 推力
            robot.set_external_force_and_torque(
                forces=self._forces[i],  # (N, 4, 3)
                torques=torch.zeros_like(self._forces[i]),
                body_ids=self._falcon_rotor_idx,
            )

        # ===== 更新移动物块位置（折返轨迹）=====
        dt = self.physics_dt * self.cfg.decimation

        # 检查是否到达折返边界，翻转方向
        x_pos = self.target_positions[:, :, 0]
        hit_max = x_pos >= self.cfg.target_bounce_x_max
        hit_min = x_pos <= self.cfg.target_bounce_x_min
        self.target_directions = torch.where(hit_max, -torch.ones_like(self.target_directions), self.target_directions)
        self.target_directions = torch.where(hit_min, torch.ones_like(self.target_directions), self.target_directions)

        self.target_velocities[:, :, 0] = self.target_directions * self.cfg.target_velocity
        self.target_velocities[:, :, 1] = 0.0
        self.target_velocities[:, :, 2] = 0.0

        self.target_positions += self.target_velocities * dt

        # 仅更新 env_0 的可视化（XFormPrim 只有 env_0 下有 prim）
        positions_world = self.target_positions + self.scene.env_origins.unsqueeze(1)
        for i, prim in enumerate(self._target_prims):
            pos = positions_world[0:1, i, :]
            ori = self._usd_target_orientations[i].unsqueeze(0)
            prim.set_world_poses(positions=pos, orientations=ori)

    def _normalize_observation(self, obs: torch.Tensor) -> torch.Tensor:
        """No manual normalization — relying on skrl's RunningStandardScaler."""
        return obs

    def _get_observations(self) -> dict[str, torch.Tensor]:
        """Gets observations (Simplified Reduced State)."""
        # ===== 从每个独立 robot 读取状态 =====
        for i, robot in enumerate(self.robots):
            root_state = robot.data.root_state_w  # (N, 13)
            self.drone_positions[:, i] = root_state[:, :3] - self.scene.env_origins
            self.drone_orientations[:, i] = root_state[:, 3:7]
            self.drone_linear_velocities[:, i] = root_state[:, 7:10]
            self.drone_angular_velocities[:, i] = root_state[:, 10:13]

        self.drone_rot_matrices[:] = matrix_from_quat(self.drone_orientations)

        # Calculate distances (xy plane)
        drone_to_target_distances = torch.zeros(
            self.num_envs, self._num_drones, self.num_targets, device=self.device
        )
        for i in range(self._num_drones):
            for j in range(self.num_targets):
                dist = torch.norm(
                    self.drone_positions[:, i, :2] - self.target_positions[:, j, :2],
                    dim=-1,
                )
                drone_to_target_distances[:, i, j] = dist

        closest_drone_to_target = torch.argmin(
            drone_to_target_distances, dim=1
        )  # (N, T)

        observations = {}
        for drone_idx, agent_name in enumerate(self.cfg.possible_agents):
            # 1. Self State (15)
            obs_self = torch.cat(
                [
                    self.drone_positions[:, drone_idx],  # 3
                    self.drone_linear_velocities[:, drone_idx],  # 3
                    self.drone_rot_matrices[:, drone_idx].view(self.num_envs, -1),  # 9
                ],
                dim=-1,
            )

            # 2. Other Drones (Rel Pos)
            other_drones_states = []
            for other_idx in range(self._num_drones):
                if other_idx != drone_idx:
                    rel_pos = (
                        self.drone_positions[:, other_idx]
                        - self.drone_positions[:, drone_idx]
                    )
                    other_drones_states.append(rel_pos)

            if len(other_drones_states) > 0:
                obs_other_drones = torch.cat(other_drones_states, dim=-1)
            else:
                obs_other_drones = torch.zeros(
                    self.num_envs, (self._num_drones - 1) * 3, device=self.device
                )

            # 3. Targets (Rel Pos + State)
            # Zero out z component: targets are at z=0.25m (ground), drone at z=2.5m.
            # Including z=-2.25 causes policy to dive toward targets to minimize 3D distance,
            # even though capture and distance rewards are purely XY-plane.
            target_rel_pos = self.target_positions - self.drone_positions[
                :, drone_idx
            ].unsqueeze(1)
            target_rel_pos = target_rel_pos.clone()
            target_rel_pos[:, :, 2] = 0.0  # XY-only: prevent policy from diving toward ground targets

            obs_targets = torch.cat(
                [
                    target_rel_pos.view(self.num_envs, -1),  # 12
                    self.target_captured.float().view(self.num_envs, -1),  # 4
                    self.target_values,  # 4
                    self.target_velocities[:, :, 0],  # 4 — x方向速度（±0.3），用于预判目标运动方向
                ],
                dim=-1,
            )

            # 4. Extra Info
            obs_distances = drone_to_target_distances[:, drone_idx]  # 4
            is_closest = (closest_drone_to_target == drone_idx).float()  # 4

            # 5. Assigned Target (fixed per episode, one-hot) — stable coordination signal
            assigned_idx = self.drone_assigned_target[:, drone_idx]  # (N,)
            assigned_onehot = torch.zeros(self.num_envs, self.cfg.num_targets, device=self.device)
            assigned_onehot.scatter_(1, assigned_idx.unsqueeze(1), 1.0)  # (N, 4)

            # Combine
            obs_t = torch.cat(
                [
                    obs_self,
                    obs_other_drones,
                    obs_targets,
                    obs_distances,
                    is_closest,
                    assigned_onehot,  # 4 — 告知无人机本 episode 负责哪个目标
                ],
                dim=-1,
            )

            # Apply buffer if partial_obs is True
            if self.cfg.partial_obs:
                self._observation_buffers[agent_name].append(obs_t)
                observations[agent_name] = self._observation_buffers[
                    agent_name
                ].buffer.reshape(self.num_envs, -1)
            else:
                observations[agent_name] = obs_t

        return observations

    def _get_states(self) -> torch.Tensor:
        """Global state (Critic Input)."""
        states = torch.cat(
            (
                self.drone_positions.view(self.num_envs, -1),  # 9
                self.drone_rot_matrices.view(self.num_envs, -1),  # 27
                self.drone_linear_velocities.view(self.num_envs, -1),  # 9
                self.drone_angular_velocities.view(self.num_envs, -1),  # 9
                self.target_positions.view(self.num_envs, -1),  # 12
                self.target_velocities.view(self.num_envs, -1),  # 12
                self.target_captured.float().view(self.num_envs, -1),  # 4
                self.target_values,  # 4
            ),
            dim=-1,
        )
        return states

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        """Exponential-decay rewards (hover/hover_flycart style)."""
        rewards = {}
        step_dt = self.step_dt

        # --- 1. Distance / Tracking Reward (per assigned target, no cross-interference) ---
        # Run37: Each drone only earns reward for its own assigned target.
        # Previous: min over all drones → drone A benefits from drone B approaching B's target,
        # causing path interference. Now each drone has an exclusive reward lane.
        d_pos = self.drone_positions[:, :, :2].unsqueeze(2)  # (N,D,1,2)
        t_pos = self.target_positions[:, :, :2].unsqueeze(1)  # (N,1,T,2)
        dist_matrix = torch.norm(d_pos - t_pos, dim=-1)  # (N, D, T)

        # Per-drone distance to its assigned target
        env_idx = torch.arange(self.num_envs, device=self.device)
        assigned_dists = torch.stack([
            dist_matrix[env_idx, drone_idx, self.drone_assigned_target[:, drone_idx]]
            for drone_idx in range(self._num_drones)
        ], dim=1)  # (N, D) — distance of each drone to its own assigned target

        # Distance reward: sum of per-drone contributions
        assigned_values = torch.stack([
            self.target_values[env_idx, self.drone_assigned_target[:, drone_idx]]
            for drone_idx in range(self._num_drones)
        ], dim=1)  # (N, D)
        dist_reward = (torch.exp(-assigned_dists * self.cfg.dist_reward_scale) * assigned_values).sum(dim=-1)
        rewards["distance_reward"] = self.cfg.dist_reward_weight * dist_reward * step_dt

        # --- 1b. Progress Reward (per assigned target) ---
        valid_prev = self._prev_min_dists >= 0  # (N, D)
        dist_progress = torch.where(
            valid_prev,
            (self._prev_min_dists - assigned_dists).clamp(-0.1, 0.1),
            torch.zeros_like(assigned_dists),
        )
        progress_reward = (dist_progress * assigned_values).sum(dim=-1)  # (N,)
        rewards["dist_progress"] = self.cfg.progress_reward_weight * progress_reward * step_dt
        self._prev_min_dists = assigned_dists.clone()

        # --- Update Capture State (global, for success condition) ---
        # Keep global min_dists for sustained_follow_timer (requires any drone near any target)
        min_dists, closest_drone_indices = torch.min(dist_matrix, dim=1)  # (N, T)
        is_captured_now = min_dists < self.cfg.capture_distance
        self.target_captured = is_captured_now
        self.target_captured_by = torch.where(
            is_captured_now,
            closest_drone_indices,
            self.target_captured_by,
        )

        # 持续跟随计时：至少3个不同物块各自被至少一架无人机跟随
        # Run41 fix: 条件断掉时清零 timer，确保真正连续的持续跟随
        target_min_dist = dist_matrix.min(dim=1)[0]  # (N, T)
        target_followed = target_min_dist < self.cfg.capture_distance  # (N, T)
        num_targets_followed = target_followed.sum(dim=-1)  # (N,)
        enough_targets_followed = num_targets_followed >= 3
        self._sustained_follow_timer = torch.where(
            enough_targets_followed,
            self._sustained_follow_timer + step_dt,
            torch.zeros_like(self._sustained_follow_timer),  # 清零而非保持，确保连续计时
        )
        self.all_targets_captured = (
            self._sustained_follow_timer >= self.cfg.sustained_follow_duration
        )

        # Success Reward
        rewards["success_reward"] = (
            self.cfg.success_reward_weight * self.all_targets_captured.float()
        )

        # Tracking Reward: per-drone, only for its assigned target
        is_assigned_captured = assigned_dists < self.cfg.capture_distance  # (N, D)
        tracking_reward = (
            is_assigned_captured.float()
            * torch.exp(-assigned_dists * self.cfg.tracking_reward_scale)
        ).sum(dim=-1)
        rewards["tracking_reward"] = (
            self.cfg.tracking_reward_weight * tracking_reward * step_dt
        )

        # --- 2. Action Smoothness (exp-decay, positive) ---
        current_actions = torch.cat(
            [self.actions[a] for a in self.cfg.possible_agents],
            dim=-1,
        )
        prev_actions = torch.cat(
            [self.prev_actions[a] for a in self.cfg.possible_agents],
            dim=-1,
        )
        diff_action = ((current_actions - prev_actions).abs()) / self._num_drones
        rewards["action_smoothness"] = (
            self.cfg.action_smoothness_weight
            * torch.exp(-torch.norm(diff_action, dim=-1).square())
            * step_dt
        )

        # --- 3. Body Rate Penalty (exp-decay, positive) ---
        commanded_body_rates = torch.cat(
            [self.actions[a][:, 3:] for a in self.cfg.possible_agents],
            dim=-1,
        )
        body_rate_norm = torch.norm(commanded_body_rates / self._num_drones, dim=-1)
        rewards["body_rate_penalty"] = (
            self.cfg.body_rate_penalty_weight * torch.exp(-body_rate_norm) * step_dt
        )

        # --- 4. Velocity Penalty (exp-decay, positive) ---
        vel_norms = torch.norm(self.drone_linear_velocities, dim=-1)  # (N, D)
        avg_vel = vel_norms.mean(dim=-1)  # (N,)
        rewards["velocity_penalty"] = (
            self.cfg.velocity_penalty_weight * torch.exp(-avg_vel) * step_dt
        )

        # --- 5. Force Penalty (exp-decay, positive) ---
        all_forces = torch.stack([f[..., 2] for f in self._forces], dim=1)  # (N, D, 4)
        normalized_forces = all_forces / self.cfg.max_thrust_pp
        effort_max = torch.max(normalized_forces.view(self.num_envs, -1), dim=-1)[0]
        rewards["force_penalty"] = (
            self.cfg.force_penalty_weight * torch.exp(-effort_max) * step_dt
        )

        # --- 6. Altitude Reward (exp-decay, positive) ---
        # Fix: use per-drone mean absolute error instead of combined L2 norm.
        # L2 norm across drones masked single-drone dives (one drone at z=1.5m
        # with others at 2.5m gave norm=1.0 < threshold=1.5, no penalty triggered).
        height_error_per_drone = (self.drone_positions[..., 2] - self.cfg.desired_height).abs()  # (N, D)
        height_error = height_error_per_drone.mean(dim=-1)  # (N,)
        rewards["height_reward"] = (
            self.cfg.height_reward_weight * torch.exp(-height_error) * step_dt
        )

        # Height Penalty (New: Strict constraint for deviating > threshold)
        # Linear penalty: weight * max(0, error - threshold), applied per-drone then summed
        excess_height = (height_error_per_drone - self.cfg.height_penalty_threshold).clamp(min=0.0)  # (N, D)
        rewards["height_penalty"] = (
            -self.cfg.height_penalty_weight * excess_height.sum(dim=-1) * step_dt
        )

        # --- 6.5 Upright Penalty ---
        # Restore baseline continuous gradient: reward = w * (z_dot - 1) * dt
        # z_axis_body=1.0 when upright (penalty=0), decreases as drone tilts.
        # This gives gradient at ALL tilt angles (0°→90°), not just beyond a threshold.
        # Previous threshold-based approach had zero gradient below 40°, allowing
        # persistent 30-35° tilt with no correction signal.
        z_axis_body = self.drone_rot_matrices[:, :, 2, 2]  # (N, D), 1.0=upright, 0.0=90°tilt
        # Run34 fix: use mean(-1) not sum(-1) to avoid 3x amplification with 3 drones
        rewards["upright_penalty"] = (
            self.cfg.upright_penalty_weight * (z_axis_body - 1.0).mean(-1) * step_dt
        )

        # ====== Safety Penalties (kept unchanged) ======

        # --- 7. Collision Penalty ---
        dd = torch.cdist(self.drone_positions, self.drone_positions, p=2)
        eye_mask = (
            torch.eye(self._num_drones, device=self.device)
            .bool()
            .unsqueeze(0)
            .expand(self.num_envs, -1, -1)
        )
        dd.masked_fill_(eye_mask, float("inf"))
        num_collisions = (
            torch.sum(dd < self.cfg.drone_collision_threshold, dim=(1, 2)) / 2.0
        )
        rewards["collision_penalty"] = (
            -num_collisions * self.cfg.collision_penalty_scale
        )

        # --- 8. Out of Bounds Penalty ---
        out_of_bounds = (
            (self.drone_positions.abs() > self.cfg.bounding_box_threshold)
            .any(dim=-1)
            .any(dim=-1)
        )
        rewards["drone_out"] = (
            -out_of_bounds.float() * self.cfg.drone_out_of_bounds_penalty
        )

        # --- 8b. Soft Boundary Penalty（引导无人机在硬边界前减速）---
        xy_dist = self.drone_positions[:, :, :2].abs()  # (N, D, 2)
        soft_excess = (xy_dist - self.cfg.boundary_soft_threshold).clamp(min=0.0)  # (N, D, 2)
        soft_penalty = soft_excess.sum(dim=(1, 2))  # (N,)
        rewards["boundary_soft"] = -self.cfg.boundary_soft_penalty_weight * soft_penalty * step_dt

        # --- 9. Fly Low Penalty (Run41: add * step_dt to normalize scale) ---
        # Run40 analysis: fly_low WITHOUT step_dt had per-step strength=8.0,
        # while height_penalty WITH step_dt had per-step strength=0.005 → 1600:1 ratio.
        # Drones rationally climbed to z>>3.5m to eliminate fly_low (save 8.0/step),
        # since height_penalty in 1.5-3.5m range is zero anyway.
        # Fix: add * step_dt, compensate by raising penalty 8.0→50.0 to maintain similar
        # physical magnitude (50*dt=50*0.01=0.5 per step; vs old 8.0 per step — still ~16x
        # stronger than height_penalty to maintain asymmetry, but no longer 1600x).
        fly_low_deficit = (1.5 - self.drone_positions[:, :, 2]).clamp(min=0.0)  # (N, D)
        rewards["fly_low"] = -fly_low_deficit.sum(dim=-1) * self.cfg.fly_low_penalty * step_dt

        # --- 10. Illegal Contact Penalty (Run23: disabled) ---
        # NovaCarter CollisionAPI was never fully disabled; contact readings are noise.
        # Threshold was raised 1N→50N chasing noise without fixing root cause.
        # Completely disabled to remove noisy gradient signal.
        rewards["illegal_contact"] = torch.zeros(self.num_envs, device=self.device)

        # --- 11. Time Penalty (Removed) ---
        # rewards["time_penalty"] = ...

        # --- Total Reward (shared) + per-agent upright bonus ---
        # Run36: Add small individual upright penalty per agent so the unstable drone
        # gets its own corrective gradient instead of being averaged with stable drones.
        total_reward = sum(rewards.values())

        agent_rewards = {}
        for drone_idx, agent_name in enumerate(self.cfg.possible_agents):
            z_body_i = self.drone_rot_matrices[:, drone_idx, 2, 2]  # (N,)
            individual_upright = (
                self.cfg.upright_penalty_weight * (z_body_i - 1.0) * step_dt
            )
            agent_rewards[agent_name] = total_reward + individual_upright

        # Logging
        for key, val in rewards.items():
            if key not in self._episode_sums:
                self._episode_sums[key] = torch.zeros_like(val)
            self._episode_sums[key] += val

        return agent_rewards

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """终止条件"""
        # 无人机飞太低（Run19 fix: 0.1→0.5m，关闭0.9m无约束俯冲区）
        self.falcon_fly_low = (self.drone_positions[:, :, 2] < 0.5).any(dim=-1)

        # 无人机飞太高（Run41: 新增 fly_high 终止，z>5.5m 即终止）
        # 修复高飞局部最优：之前无 fly_high 终止，drone 可爬至 bbox=24m 才结束 episode
        self.falcon_fly_high = (self.drone_positions[:, :, 2] > self.cfg.fly_high_termination_z).any(dim=-1)

        # 非法接触（per-drone contact sensor）
        self.illegal_contact = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        for contact_sensor in self.contact_sensors:
            net_forces = contact_sensor.data.net_forces_w_history
            max_force = torch.max(torch.norm(net_forces, dim=-1), dim=1)[0]
            has_contact = (max_force > self.cfg.contact_sensor_threshold).any(dim=1)
            self.illegal_contact = self.illegal_contact | has_contact

        # 无人机碰撞
        rpos = get_drone_rpos(self.drone_positions)
        pdist = get_drone_pdist(rpos)
        separation = pdist.min(dim=-1).values.min(dim=-1).values
        self.drone_collision = separation < self.cfg.drone_collision_threshold

        # 边界检查
        is_outside = (self.drone_positions.abs() > self.cfg.bounding_box_threshold).any(
            dim=-1
        )

        # 豁免条件：捕获目标的无人机出界不惩罚
        drone_has_captured = torch.zeros(
            (self.num_envs, self._num_drones), dtype=torch.bool, device=self.device
        )
        for d in range(self._num_drones):
            drone_has_captured[:, d] = (
                (self.target_captured_by == d) & self.target_captured
            ).any(dim=-1)

        self.body_pos_outside = (is_outside & (~drone_has_captured)).any(dim=-1)

        # 持续跟随终止已在 _get_rewards 中更新

        # 物块超出边界
        self.targets_out_of_bounds = (
            (self.target_positions[:, :, 0] > self.cfg.bounding_box_threshold)
            | (self.target_positions[:, :, 0] < -self.cfg.bounding_box_threshold)
            | (self.target_positions[:, :, 1] > self.cfg.bounding_box_threshold)
            | (self.target_positions[:, :, 1] < -self.cfg.bounding_box_threshold)
            | (self.target_positions[:, :, 2] < -0.5)  # NovaCarter 在地面 z≈0，只在意外入地时触发
        ).any(dim=-1)

        # 组合终止条件
        terminations = (
            self.falcon_fly_low
            | self.falcon_fly_high  # Run41: 新增高飞终止
            # illegal_contact 仅作惩罚，不终止 episode（NovaCarter CollisionAPI 禁用失败的临时规避）
            | self.drone_collision
            | self.body_pos_outside
            | self.targets_out_of_bounds
            | self.all_targets_captured  # Run41: success 触发终止（连续跟随3目标≥0.5s）
        )

        # Debug logging
        reset_indices = torch.nonzero(terminations).flatten()
        if len(reset_indices) > 0:
            for i in reset_indices[:5]:
                idx = i.item()
                reasons = []
                if self.falcon_fly_low[idx]:
                    reasons.append(
                        f"Fly Low (z={self.drone_positions[idx, :, 2].min():.2f})"
                    )
                if self.falcon_fly_high[idx]:
                    reasons.append(
                        f"Fly High (z={self.drone_positions[idx, :, 2].max():.2f})"
                    )
                if self.illegal_contact[idx]:
                    reasons.append("Illegal Contact")
                if self.drone_collision[idx]:
                    reasons.append("Drone Collision")
                if self.body_pos_outside[idx]:
                    max_pos = self.drone_positions[idx].abs().max().item()
                    reasons.append(
                        f"Outside Boundary (max_pos={max_pos:.1f} > {self.cfg.bounding_box_threshold})"
                    )
                if self.targets_out_of_bounds[idx]:
                    reasons.append("Target Outside")
                if self.all_targets_captured[idx]:
                    reasons.append(f"SUCCESS (timer={self._sustained_follow_timer[idx]:.2f}s)")
                    # Check currently captured targets
                    current_captures = self.target_captured[idx]  # (T,) bool
                    if current_captures.any():
                        current_captured_by = self.target_captured_by[idx][
                            current_captures
                        ]
                        unique_drones = torch.unique(current_captured_by)
                        # Filter out -1 just in case, though is_captured check should prevent it
                        unique_drones = unique_drones[unique_drones != -1]
                        if len(unique_drones) >= 3:
                            reasons.append("All Captured")
                if reasons:
                    print(f"[Reset] Env {idx}: {', '.join(reasons)}")

        self.time_out = self.episode_length_buf >= self.max_episode_length - 1
        timed_outs = self.time_out
        terminated = {agent: terminations for agent in self.cfg.possible_agents}
        time_outs = {agent: timed_outs for agent in self.cfg.possible_agents}
        return terminated, time_outs

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        super()._reset_idx(env_ids)
        self._reset_targets(env_ids)
        self._prev_vel_error[env_ids] = 0.0
        self._sustained_follow_timer[env_ids] = 0.0
        self._prev_min_dists[env_ids] = -1.0  # invalidate for first-step skip

        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device)

        # ===== 对每个robot独立重置 (无需索引映射!) =====
        # Calculate Formation Center
        center_x = torch.empty(len(env_ids), device=self.device).uniform_(
            *self.cfg.drone_spawn_x_range
        )
        center_y = torch.empty(len(env_ids), device=self.device).uniform_(
            *self.cfg.drone_spawn_y_range
        )

        radius = 2.0
        # Run41 fix: 使用 cfg 的 drone_spawn_z_range 随机采样高度（之前硬编码 2.5 导致配置无效）
        center_z = torch.empty(len(env_ids), device=self.device).uniform_(
            *self.cfg.drone_spawn_z_range
        )
        phases = torch.tensor(
            [0.0, 2.0 * 3.14159 / 3.0, 4.0 * 3.14159 / 3.0], device=self.device
        )

        for i, robot in enumerate(self.robots):
            robot.reset(env_ids)

            origins = self.scene.env_origins[env_ids]

            offsets = torch.zeros((len(env_ids), 3), device=self.device)
            offsets[:, 0] = center_x + radius * torch.cos(phases[i])
            offsets[:, 1] = center_y + radius * torch.sin(phases[i])
            offsets[:, 2] = center_z  # 使用配置中的随机高度范围

            new_positions = origins + offsets

            default_root_state = robot.data.default_root_state[env_ids]
            new_quats = default_root_state[:, 3:7]
            new_poses = torch.cat([new_positions, new_quats], dim=-1)

            robot.write_root_pose_to_sim(new_poses, env_ids=env_ids)
            robot.write_root_velocity_to_sim(
                torch.zeros_like(default_root_state[:, 7:]), env_ids=env_ids
            )

        # ===== Target Assignment (fixed for entire episode) =====
        # Assign each drone to a target by y-rank: highest-y drone → highest-y target (highest value).
        # drone spawn y: center_y + radius * sin(phase[i]), radius=2.0
        radius = 2.0
        drone_spawn_y = center_y.unsqueeze(1) + radius * torch.sin(phases).unsqueeze(0)  # (B, D)
        # drone_y_rank[b, r] = drone_idx with r-th highest y in batch b
        drone_y_rank = drone_spawn_y.argsort(dim=1, descending=True)  # (B, D)
        # assignment[b, drone_idx] = target_idx (0=highest-value/y, 1, 2; target 3 unassigned)
        target_indices = torch.arange(self._num_drones, device=self.device).unsqueeze(0).expand(
            len(env_ids), -1
        )
        assignment = torch.zeros(len(env_ids), self._num_drones, dtype=torch.long, device=self.device)
        assignment.scatter_(1, drone_y_rank, target_indices)
        self.drone_assigned_target[env_ids] = assignment

        for agent in self.cfg.possible_agents:
            self._observation_buffers[agent].reset(env_ids)

        # log reward components
        if "log" not in self.extras:
            self.extras["log"] = dict()
        if "crash" not in self.extras["log"]:
            self.extras["log"]["Episode_Termination/crash"] = 0
        if "out_of_bounds" not in self.extras["log"]:
            self.extras["log"]["Episode_Termination/out_of_bounds"] = 0

        self.extras["log"]["Episode_Termination/drones_collide"] = torch.count_nonzero(
            self.drone_collision[env_ids]
        ).item()
        self.extras["log"]["Episode_Termination/bounding_box"] = torch.count_nonzero(
            self.body_pos_outside[env_ids]
        ).item()
        self.extras["log"]["Episode_Termination/falcon_fly_low"] = torch.count_nonzero(
            self.falcon_fly_low[env_ids]
        ).item()
        self.extras["log"]["Episode_Termination/falcon_fly_high"] = torch.count_nonzero(
            self.falcon_fly_high[env_ids]
        ).item()
        self.extras["log"]["Episode_Termination/crash"] = torch.count_nonzero(
            self.falcon_fly_low[env_ids] | self.illegal_contact[env_ids]
        ).item()
        self.extras["log"]["Episode_Termination/out_of_bounds"] = torch.count_nonzero(
            self.targets_out_of_bounds[env_ids]
        ).item()

        # Log Episode Rewards
        for key, value in self._episode_sums.items():
            if len(env_ids) > 0:
                mean_val = value[env_ids].mean().item()
                self.extras["log"][f"Episode_Reward/{key}"] = mean_val
            value[env_ids] = 0.0

        self.extras["log"]["Episode_Termination/illegal_contact"] = torch.count_nonzero(
            self.illegal_contact[env_ids]
        ).item()
        self.extras["log"]["Episode_Termination/all_targets_captured"] = (
            torch.count_nonzero(self.all_targets_captured[env_ids]).item()
        )
        self.extras["log"]["Episode_Termination/targets_out_of_bounds"] = (
            torch.count_nonzero(self.targets_out_of_bounds[env_ids]).item()
        )
        self.extras["log"]["Episode_Termination/time_out"] = torch.count_nonzero(
            self.time_out[env_ids]
        ).item()

        # reset the action history
        for agent in self.cfg.possible_agents:
            self.prev_actions[agent][env_ids] = 0.0
            self.actions[agent][env_ids] = 0.0

        self.all_targets_captured[env_ids] = False

    def _reset_targets(self, env_ids):
        """重置目标小车位置：使用 USD 中缓存的初始位姿作为起始点。"""
        if env_ids.numel() == 0:
            return

        # 以 USD 初始位置为基准（相对坐标），广播到所有被重置的 env
        rel = self._usd_target_positions_rel.unsqueeze(0).repeat(env_ids.numel(), 1, 1)
        self.target_positions[env_ids] = rel

        # 重置速度、方向和捕获状态
        self.target_velocities[env_ids] = 0.0
        self.target_directions[env_ids] = 1.0  # 重置为 +x 方向
        self.target_captured[env_ids] = False
        self.target_captured_by[env_ids] = -1

        # 更新 env_0 可视化
        if 0 in env_ids.tolist():
            positions_world = self.target_positions[0:1] + self.scene.env_origins[0:1].unsqueeze(1)
            for i, prim in enumerate(self._target_prims):
                pos = positions_world[:, i, :]
                ori = self._usd_target_orientations[i].unsqueeze(0)
                prim.set_world_poses(positions=pos, orientations=ori)

    def _set_debug_vis_impl(self, debug_vis: bool):
        pass

    def _debug_vis_callback(self, event):
        pass


@torch.jit.script
def scale(x, lower, upper):
    return 0.5 * (x + 1.0) * (upper - lower) + lower


@torch.jit.script
def unscale(x, lower, upper):
    return (2.0 * x - upper - lower) / (upper - lower)
