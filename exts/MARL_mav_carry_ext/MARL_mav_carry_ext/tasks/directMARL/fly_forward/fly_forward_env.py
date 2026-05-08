"""单无人机 fly-forward 环境实现（DirectRLEnv, PPO）。

任务：从出生点出发，在保持低空稳定飞行的同时向前推进。
      当前阶段课程：先学习低空存活和短距离前向飞行。
控制：ACCBR（平面速度指令 + 机体系角速度指令，5 维连续动作）。
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
    """用于 PPO 训练的单无人机低空短距离前向飞行环境。

    观测（21 维）：
        drone_pos_norm(3) | drone_lin_vel_norm(3) | rot_matrix(9) |
        drone_ang_vel_norm(3) | goal_rel_norm(3)

    动作（5 维）：
        [vx_cmd, vy_cmd, roll_rate, pitch_rate, yaw_rate]，取值范围 [-1, 1]。
        策略学习 vx/vy 和机体系角速度，z 方向由高度保持逻辑自动维持在 goal_z 附近。
    """

    cfg: FlyForwardEnvCfg

    def __init__(self, cfg: FlyForwardEnvCfg, render_mode: str | None = None, **kwargs):
        # super().__init__ 会调用 _setup_scene，调用结束后 scene.env_origins 才可用。
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)

        # ── Body/rotor index cache ────────────────────────────────────────────
        # 缓存 base_link 和 4 个 rotor 的 body id，后续施加外力 / 外力矩时会频繁使用。
        self._falcon_body_idx = self._robot.find_bodies(".*base_link")[0]  # list[int]，机体主体索引
        self._rotor_idx = self._robot.find_bodies(".*rotor_.*")[0]         # list[int]，旋翼索引

        # ── Physics buffers ───────────────────────────────────────────────────
        # _forces: (N, 4, 3) —— 每个环境、每个旋翼的推力向量，推力沿机体 z 方向写入。
        self._forces = torch.zeros(self.num_envs, 4, 3, device=self.device)
        # _moments: (N, 1, 3) —— 施加在 base_link 上的反扭矩。
        self._moments = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._prev_vel_error = torch.zeros(self.num_envs, 3, device=self.device) # 速度误差历史，用于 D 项
        self._prev_actions = torch.zeros(self.num_envs, 5, device=self.device)   # 上一步动作，用于动作平滑惩罚
        self._drone_prev_acc = torch.zeros(self.num_envs, 3, device=self.device) # 上一步线加速度，用于计算 jerk
        self._ll_counter: int = 0 # 底层控制降采样计数器

        # ── Controllers ───────────────────────────────────────────────────────
        # Falcon 的悬停转速 hover omega = sqrt(m*g / (4*k_tau)) ≈ 972 rad/s。
        # 原来用 1355 rad/s，是悬停的 1.94×，reset 后产生 +9 m/s² 瞬时向上加速度
        _hover_omega = 972.0
        self._geo_ctrl = GeometricController(self.num_envs, "ACCBR") # 几何控制器：将期望加速度 / 角速度转为中间控制量
        self._indi_ctrl = IndiController(self.num_envs)              # INDI 控制器：根据当前状态计算目标转速
        self._motor_model = RotorMotor(
            self.num_envs,
            torch.full((self.num_envs, 4), _hover_omega, device=self.device),
        )

        # ── Diagnostic logger (FLY_FORWARD_DIAG=1 to enable) ─────────────────
        # 诊断日志：设置 FLY_FORWARD_DIAG=1 后，将 env_0 的关键状态写入 CSV 便于离线排查。
        self._diag_enabled = os.environ.get("FLY_FORWARD_DIAG", "0") == "1"
        # 设置 FLY_FORWARD_ZERO=1 或 FLY_FORWARD_ZERO_ACTION=1 时忽略策略动作，改用全 0 动作。
        self._diag_zero = os.environ.get("FLY_FORWARD_ZERO", "0") == "1"
        self._diag_zero_action = self._diag_zero or os.environ.get("FLY_FORWARD_ZERO_ACTION", "0") == "1"
        # 设置 FLY_FORWARD_RANDOM=1 时忽略策略动作，改用随机动作做控制链 smoke test。
        self._diag_random = os.environ.get("FLY_FORWARD_RANDOM", "0") == "1"
        if self._diag_enabled:
            self._diag_path = os.environ.get("FLY_FORWARD_DIAG_PATH", "/tmp/fly_forward_diag.csv")
            self._diag_file = open(self._diag_path, "w", newline="")
            self._diag_writer = csv.writer(self._diag_file)
            mode = "zero" if self._diag_zero_action else "random" if self._diag_random else "policy"
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
        self.sampling_time = self.sim.get_physics_dt() * self.cfg.low_level_decimation # 电机模型采样周期

        # ── Goal world position (assigned per-env at reset; fixed 20m forward target) ─────
        # 每个 env 的目标世界坐标，在 reset 时根据实际出生点写入。
        self._goal_w = torch.zeros(self.num_envs, 3, device=self.device)

        # ── Spawn world position (_reset_idx 写入，OOB 检查使用) ───────────────────
        # 每个 env 的出生点世界坐标，用于越界判断和前向进度计算。
        self._spawn_w = torch.zeros(self.num_envs, 3, device=self.device)

        # ── Progress tracking: previous full distance to goal ──────────────────
        # 上一步到目标的距离，用于计算真正的“向目标靠近”进度奖励。
        self._prev_dist = torch.zeros(self.num_envs, device=self.device)

        # ── Termination flags ─────────────────────────────────────────────────
        # 终止标志会在 _get_dones 中更新，并在奖励 / 日志中复用。
        self._fly_high = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._fly_low = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._out_of_bounds = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # ── Episode sum logging ───────────────────────────────────────────────
        # 记录每个 episode 内各奖励分量的累计值，reset 时汇总到 self.extras["log"]。
        self._episode_sums = {
            k: torch.zeros(self.num_envs, device=self.device)
            for k in [
                "progress_reward",
                "dist_reward",
                "altitude_band_reward",
                "height_penalty",
                "fly_high_guard_penalty",
                "slow_near_goal_reward",
                "speed_penalty",
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
        """解析 5 维 ACCBR 动作，并叠加自动高度保持。"""
        if self._diag_zero_action:
            # 零动作 smoke test：隔离策略输出，只检查高度保持和底层控制是否会自行爬升。
            actions = torch.zeros_like(actions)
        elif self._diag_random:
            # 诊断模式下使用随机动作覆盖策略动作，便于排除 policy 输出对控制链的影响。
            actions = torch.rand_like(actions) * 2.0 - 1.0
        self._actions = actions.clone() # 保存当前动作，用于奖励中的动作平滑项

        # 当前局部位置用于高度保护；root_pos_w 是世界坐标，需要减去 env 原点。
        current_pos = self._robot.data.root_pos_w - self.scene.env_origins
        z = current_pos[:, 2]
        # 高度保护门控：z 从 fly_high_guard_z 接近 fly_high_z 时，upward_scale 从 1 降到 0。
        high_guard = torch.clamp(
            (z - self.cfg.fly_high_guard_z) / (self.cfg.fly_high_z - self.cfg.fly_high_guard_z),
            0.0,
            1.0,
        )
        upward_scale = 1.0 - high_guard

        # 将归一化动作映射为期望速度。x/y 来自策略，z 由高度保持 PD 计算。
        desired_vel = torch.zeros(self.num_envs, 3, device=self.device)
        desired_vel[:, :2] = actions[:, :2] * self.cfg.lin_vel_max
        current_vel = self._robot.data.root_lin_vel_w
        height_vel_cmd = self.cfg.height_hold_kp * (self.cfg.goal_z - z) - self.cfg.height_hold_damping * current_vel[:, 2]
        desired_vel[:, 2] = torch.clamp_min(height_vel_cmd, -self.cfg.lin_vel_z_down_max)
        desired_vel[:, 2] = torch.minimum(desired_vel[:, 2], self.cfg.lin_vel_z_up_max * upward_scale)
        # 速度误差 PD：P 项跟踪期望速度，D 项抑制速度误差变化过快。
        vel_error = desired_vel - current_vel
        d_error = (vel_error - self._prev_vel_error) / self.step_dt
        self._prev_vel_error = vel_error.clone()

        # 将速度误差转为期望加速度，并分别约束水平 / 垂直方向的加速度范围。
        commanded_acc = self.cfg.vel_Kp * vel_error + self.cfg.vel_Kd * d_error
        commanded_acc[:, :2] = torch.clamp(
            commanded_acc[:, :2], -self.cfg.lin_acc_max, self.cfg.lin_acc_max
        )
        max_up_acc = self.cfg.lin_acc_z_up_max * upward_scale
        commanded_acc[:, 2] = torch.clamp_min(commanded_acc[:, 2], -self.cfg.lin_acc_z_down_max)
        commanded_acc[:, 2] = torch.minimum(commanded_acc[:, 2], max_up_acc)
        # 进入高度保护区后，额外限制 z 方向加速度，强制策略更偏向下降而不是继续爬升。
        descent_cap = torch.where(
            high_guard > 0.0,
            -self.cfg.fly_high_guard_descent_acc * high_guard,
            max_up_acc,
        )
        commanded_acc[:, 2] = torch.minimum(commanded_acc[:, 2], descent_cap)
        guard_descent_floor = -self.cfg.fly_high_guard_descent_acc * (0.5 + 0.5 * high_guard)
        commanded_acc[:, 2] = torch.where(
            z > self.cfg.fly_high_guard_z,
            torch.minimum(commanded_acc[:, 2], guard_descent_floor),
            commanded_acc[:, 2],
        )

        # setpoint 是几何控制器的输入：线加速度、机体系角速度、yaw/yaw_rate/yaw_acc。
        self._setpoint = {
            "lin_acc": commanded_acc,
            "body_rates": actions[:, 2:5] * self.cfg.ang_vel_max,
            "yaw": torch.zeros(self.num_envs, 1, device=self.device),
            "yaw_rate": actions[:, 4:5] * self.cfg.ang_vel_max,
            "yaw_acc": torch.zeros(self.num_envs, 1, device=self.device),
        }

        if self._diag_enabled:
            # 只记录 env_0，降低 CSV 体积；主要关注位置、速度、动作和命令加速度。
            self._diag_step += 1
            root = self._robot.data.root_state_w
            pos_l = root[0, :3] - self.scene.env_origins[0]
            vel = root[0, 7:10]
            a = actions[0]
            ca = commanded_acc[0]
            mode = "zero" if self._diag_zero_action else "random" if self._diag_random else "policy"
            self._diag_writer.writerow([
                mode, self._diag_episode, self._diag_step,
                f"{pos_l[0].item():.4f}", f"{pos_l[2].item():.4f}",
                f"{vel[0].item():.4f}", f"{vel[2].item():.4f}",
                f"{a[0].item():.4f}", f"{a[1].item():.4f}", f"{a[2].item():.4f}",
                f"{ca[0].item():.4f}", f"{ca[1].item():.4f}", f"{ca[2].item():.4f}",
            ])

    def _apply_action(self) -> None:
        """将高层 setpoint 通过几何控制、INDI 控制和电机模型转换为外力 / 外力矩。"""
        if self._diag_zero_action:
            root = self._robot.data.root_state_w
            pos_local = root[:, :3] - self.scene.env_origins
            z = pos_local[:, 2]
            vz = root[:, 9]

            desired_acc_z = (
                self.cfg.height_hold_kp * (self.cfg.goal_z - z)
                - self.cfg.height_hold_damping * vz
            )
            desired_acc_z = torch.clamp(
                desired_acc_z,
                -self.cfg.lin_acc_z_down_max,
                self.cfg.lin_acc_z_up_max,
            )

            collective_thrust = self._geo_ctrl.falcon_mass * (9.8066 + desired_acc_z)
            collective_thrust = torch.clamp(collective_thrust, 0.0, 4.0 * self.cfg.max_thrust_pp)
            rotor_thrust = collective_thrust / 4.0

            self._forces.zero_()
            self._forces[..., 2] = rotor_thrust.unsqueeze(-1).expand(-1, 4)
            self._moments.zero_()

            self._robot.set_external_force_and_torque(
                forces=torch.zeros(self.num_envs, 1, 3, device=self.device),
                torques=self._moments,
                body_ids=self._falcon_body_idx,
            )
            self._robot.set_external_force_and_torque(
                forces=self._forces,
                torques=torch.zeros_like(self._forces),
                body_ids=self._rotor_idx,
            )
            return

        if self._ll_counter % self.cfg.low_level_decimation == 0:
            # 从 Isaac Lab articulation buffer 中读取当前无人机状态。
            root = self._robot.data.root_state_w
            drone_pos = root[:, :3] - self.scene.env_origins
            drone_quat = root[:, 3:7]
            drone_lin_vel = root[:, 7:10]
            drone_ang_vel = root[:, 10:13]
            body_acc = self._robot.data.body_acc_w
            base_id = self._falcon_body_idx[0]
            drone_lin_acc = body_acc[:, base_id, :3]
            drone_ang_acc = body_acc[:, base_id, 3:6]
            jerk = (drone_lin_acc - self._drone_prev_acc) / self.physics_dt
            self._drone_prev_acc = drone_lin_acc.clone()

            # 控制器统一使用 drone_states 字典，保持和其它任务 / 控制器接口一致。
            drone_states = {
                "pos": drone_pos,
                "quat": drone_quat,
                "lin_vel": drone_lin_vel,
                "ang_vel": drone_ang_vel,
                "lin_acc": drone_lin_acc,
                "ang_acc": drone_ang_acc,
                "jerk": jerk,
            }

            # 几何控制器先把期望加速度 / 角速度转换为姿态相关的中间命令。
            alpha_cmd, acc_load, acc_cmd, _ = self._geo_ctrl.getCommand(
                drone_states, self._forces, self._setpoint
            )

            # INDI 控制器根据当前状态和中间命令计算目标电机转速。
            target_rpm = self._indi_ctrl.getCommand(
                drone_states, self._forces, alpha_cmd, acc_cmd, acc_load
            )
            # 电机模型将目标转速转换为每个旋翼的推力和反扭矩。
            thrusts, moments = self._motor_model.get_motor_thrusts_moments(
                target_rpm, self.sampling_time
            )

            # 推力写入每个 rotor 的 z 方向；yaw 反扭矩汇总后施加到 base_link。
            forces = torch.clamp(thrusts, min=0.0, max=self.cfg.max_thrust_pp)
            self._forces[..., 2] = forces
            self._moments[:, 0, 2] = torch.clamp(moments, -1.0, 1.0).sum(-1)
            self._ll_counter = 0
        self._ll_counter += 1

        # 在 base_link 上施加反扭矩。
        self._robot.set_external_force_and_torque(
            forces=torch.zeros(self.num_envs, 1, 3, device=self.device),
            torques=self._moments,
            body_ids=self._falcon_body_idx,
        )

        # 在 4 个 rotor 上施加旋翼推力。
        self._robot.set_external_force_and_torque(
            forces=self._forces,
            torques=torch.zeros_like(self._forces),
            body_ids=self._rotor_idx,
        )

    # ── Observations ─────────────────────────────────────────────────────────

    def _get_observations(self) -> dict:
        """构造策略网络输入观测。"""
        root = self._robot.data.root_state_w
        pos_local = root[:, :3] - self.scene.env_origins   # (N, 3)，局部位置
        lin_vel = root[:, 7:10]                            # (N, 3)，世界系线速度
        ang_vel = root[:, 10:13]                           # (N, 3)，世界系角速度
        rot_mat = matrix_from_quat(root[:, 3:7]).view(self.num_envs, 9) # 姿态四元数转 3x3 旋转矩阵并展平
        goal_rel = self._goal_w - root[:, :3]              # (N, 3)，目标相对当前位置的世界系向量

        # 绝对 x/y 位置和目标相对 x/y 分开缩放，避免短距离目标信号过小。
        pos_norm = torch.stack(
            [pos_local[:, 0] / self.cfg.norm_pos_scale,
             pos_local[:, 1] / self.cfg.norm_pos_scale,
             pos_local[:, 2] / self.cfg.norm_z_scale],
            dim=-1,
        )
        goal_rel_norm = torch.stack(
            [goal_rel[:, 0] / self.cfg.norm_goal_xy_scale,
             goal_rel[:, 1] / self.cfg.norm_goal_xy_scale,
             goal_rel[:, 2] / self.cfg.norm_z_scale],
            dim=-1,
        )

        # 拼接后维度为 21，对应 cfg.observation_space。
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
        """计算每个并行环境当前 step 的奖励。"""
        pos_w = self._robot.data.root_pos_w
        pos_l = pos_w - self.scene.env_origins
        vel_w = self._robot.data.root_lin_vel_w
        rot_mat = matrix_from_quat(self._robot.data.root_state_w[:, 3:7])
        z_body_z = rot_mat[:, 2, 2]

        z = pos_l[:, 2]
        height_error = z - self.cfg.goal_z

        dist = torch.norm(pos_w - self._goal_w, dim=-1)
        speed = torch.norm(vel_w, dim=-1)

        # 1. 真正的到目标进度
        delta_dist = self._prev_dist - dist
        progress_rew = self.cfg.progress_reward_weight * torch.clamp(delta_dist, -0.2, 0.2)
        self._prev_dist = dist.clone()

        # 2. 距离目标越近越好
        dist_rew = self.cfg.dist_reward_weight * torch.exp(-self.cfg.dist_reward_scale * dist)

        # 3. 安全高度带内存活正奖励，让早期策略明确知道低空稳定是有效行为。
        in_altitude_band = (z > self.cfg.fly_low_z + 0.2) & (z < self.cfg.fly_high_guard_z)
        altitude_band_rew = self.cfg.altitude_band_reward_weight * in_altitude_band.float()

        # 4. 高度约束，明确惩罚偏离目标高度
        height_pen = -self.cfg.height_penalty_weight * height_error.square()

        # 5. 过高强惩罚
        fly_high_guard_pen = (
            -self.cfg.fly_high_guard_penalty_weight
            * torch.relu(z - self.cfg.fly_high_guard_z).square()
        )

        # 6. 目标附近必须慢下来
        near_goal = dist < self.cfg.near_goal_radius
        slow_near_goal_rew = (
            self.cfg.slow_near_goal_reward_weight
            * torch.exp(-self.cfg.speed_reward_scale * speed)
            * near_goal.float()
        )

        # 7. 全局速度惩罚，避免高速冲过目标
        speed_pen = -self.cfg.speed_penalty_weight * speed.square()

        # 8. 姿态稳定
        upright_pen = self.cfg.upright_penalty_weight * (z_body_z - 1.0)

        # 9. 动作平滑
        delta_a = self._actions - self._prev_actions
        smooth_pen = -self.cfg.action_smoothness_weight * delta_a.square().sum(-1)
        self._prev_actions = self._actions.clone()

        # 10. 成功奖励
        success_rew = self._success.float() * self.cfg.success_reward

        # 11. 失败惩罚
        crash_pen = -(
            self._fly_high.float() * self.cfg.fly_high_penalty
            + self._fly_low.float() * self.cfg.fly_low_penalty
            + self._out_of_bounds.float() * self.cfg.out_of_bounds_penalty
        )

        total = (
            progress_rew
            + dist_rew
            + altitude_band_rew
            + height_pen
            + fly_high_guard_pen
            + slow_near_goal_rew
            + speed_pen
            + upright_pen
            + smooth_pen
            + success_rew
            + crash_pen
        )

        # 累计各奖励分量，reset 时写入训练日志，方便观察 reward shaping 是否按预期工作。
        self._episode_sums["progress_reward"] += progress_rew
        self._episode_sums["dist_reward"] += dist_rew
        self._episode_sums["altitude_band_reward"] += altitude_band_rew
        self._episode_sums["height_penalty"] += height_pen
        self._episode_sums["fly_high_guard_penalty"] += fly_high_guard_pen
        self._episode_sums["slow_near_goal_reward"] += slow_near_goal_rew
        self._episode_sums["speed_penalty"] += speed_pen
        self._episode_sums["upright_penalty"] += upright_pen
        self._episode_sums["action_smoothness"] += smooth_pen
        self._episode_sums["success_reward"] += success_rew
        self._episode_sums["crash_penalty"] += crash_pen

        self.extras.setdefault("log", {}).update(
            {
                "Episode Reward/progress": progress_rew.mean().item(),
                "Episode Reward/dist": dist_rew.mean().item(),
                "Episode Reward/altitude_band": altitude_band_rew.mean().item(),
                "Episode Reward/height_pen": height_pen.mean().item(),
                "Episode Reward/fly_high_guard_pen": fly_high_guard_pen.mean().item(),
                "Episode Reward/slow_near_goal": slow_near_goal_rew.mean().item(),
                "Episode Reward/speed_pen": speed_pen.mean().item(),
                "Episode Reward/upright": upright_pen.mean().item(),
                "Episode Reward/smooth": smooth_pen.mean().item(),
                "Episode Reward/success": success_rew.mean().item(),
                "Episode Reward/crash": crash_pen.mean().item(),
                "Metrics/z_mean": z.mean().item(),
                "Metrics/z_max": z.max().item(),
                "Metrics/dist_mean": dist.mean().item(),
                "Metrics/speed_mean": speed.mean().item(),
            }
        )

        return total

    # ── Terminations ──────────────────────────────────────────────────────────

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """计算终止条件和超时条件。"""
        root = self._robot.data.root_state_w
        pos_local = root[:, :3] - self.scene.env_origins
        z = pos_local[:, 2]

        # 高度过高、过低直接终止。
        self._fly_high = z > self.cfg.fly_high_z
        self._fly_low = z < self.cfg.fly_low_z
        # OOB：相对出生点的水平漂移超过课程设定边界。
        dist_from_spawn_xy = torch.norm((root[:, :3] - self._spawn_w)[:, :2], dim=-1)
        self._out_of_bounds = dist_from_spawn_xy > self.cfg.out_of_bounds_radius
        # 成功条件与 reward 的目标距离保持一致，同时要求低速、低空和姿态稳定。
        dist_to_goal = torch.norm(root[:, :3] - self._goal_w, dim=-1)
        speed = torch.norm(root[:, 7:10], dim=-1)
        reached_goal = dist_to_goal < self.cfg.goal_tolerance
        slow_enough = speed < self.cfg.success_speed_tolerance
        low_enough = z < self.cfg.fly_high_guard_z
        stable_upright = matrix_from_quat(root[:, 3:7])[:, 2, 2] > 0.9
        self._success = reached_goal & slow_enough & low_enough & stable_upright

        terminated = self._fly_high | self._fly_low | self._out_of_bounds | self._success
        timed_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, timed_out

    # ── Reset ─────────────────────────────────────────────────────────────────

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """重置指定并行环境，并初始化目标、出生点、控制缓存和日志。"""
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        # 记录刚结束 episode 的奖励分量和终止原因，然后清空对应 env 的累计值。
        extras = {}
        for key, buf in self._episode_sums.items():
            extras[f"Episode_Reward/{key}"] = buf[env_ids].mean().item()
            buf[env_ids] = 0.0
        self.extras["log"] = extras
        self.extras["log"]["Episode_Termination/fly_high"] = int(self._fly_high[env_ids].sum())
        self.extras["log"]["Episode_Termination/fly_low"] = int(self._fly_low[env_ids].sum())
        self.extras["log"]["Episode_Termination/out_of_bounds"] = int(self._out_of_bounds[env_ids].sum())
        self.extras["log"]["Episode_Termination/success"] = int(self._success[env_ids].sum())

        # 重置 articulation 和 DirectRLEnv 内部 buffer。
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        n = len(env_ids)
        # 固定出生点 (spawn_x, spawn_y, spawn_z)，叠加少量 x/y 随机扰动提升鲁棒性。
        x_noise = (torch.rand(n, device=self.device) * 2.0 - 1.0) * self.cfg.spawn_x_noise
        y_noise = (torch.rand(n, device=self.device) * 2.0 - 1.0) * self.cfg.spawn_y_noise
        spawn_pos = self.scene.env_origins[env_ids].clone()
        spawn_pos[:, 0] += self.cfg.spawn_x + x_noise
        spawn_pos[:, 1] += self.cfg.spawn_y + y_noise
        spawn_pos[:, 2] = self.cfg.spawn_z

        # 固定目标：放在实际出生点前方，用来隔离“低空向前飞”这个子任务。
        self._goal_w[env_ids, 0] = spawn_pos[:, 0] + self.cfg.goal_x
        self._goal_w[env_ids, 1] = spawn_pos[:, 1] + self.cfg.goal_y
        self._goal_w[env_ids, 2] = self.scene.env_origins[env_ids, 2] + self.cfg.goal_z

        # 记录出生点世界坐标，供 OOB 判断和前向进度奖励使用。
        self._spawn_w[env_ids] = spawn_pos

        # 将无人机放回出生点，并清零姿态以外的速度状态。
        root_state = self._robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] = spawn_pos
        root_state[:, 3] = 1.0
        root_state[:, 4:7] = 0.0
        root_state[:, 7:] = 0.0  # 清零速度

        # 写入 root pose、root velocity 和关节状态到仿真。
        self._robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(
            self._robot.data.default_joint_pos[env_ids],
            self._robot.data.default_joint_vel[env_ids],
            None,
            env_ids,
        )

        # 重置控制相关缓存，避免跨 episode 残留误差、动作、推力或力矩。
        self._prev_vel_error[env_ids] = 0.0
        self._prev_actions[env_ids] = 0.0
        self._drone_prev_acc[env_ids] = 0.0
        self._forces[env_ids] = 0.0
        self._moments[env_ids] = 0.0
        # 重置电机转速到悬停值，避免跨 episode 积累导致初始过推力
        self._motor_model.reset(env_ids)

        if self._diag_enabled and 0 in env_ids:
            # 诊断日志按 env_0 的 episode 计数，便于 CSV 后处理。
            self._diag_episode += 1
            self._diag_step = 0

        # 初始化目标距离缓存（progress reward 的基准）
        self._prev_dist[env_ids] = torch.norm(
            spawn_pos - self._goal_w[env_ids], dim=-1
        )
