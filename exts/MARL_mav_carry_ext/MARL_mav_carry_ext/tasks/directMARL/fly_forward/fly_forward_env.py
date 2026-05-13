"""单无人机 fly-forward 环境实现（DirectRLEnv, PPO）。

任务：从出生点出发，在保持低空稳定飞行的同时向前推进。
      当前阶段课程：先学习低空存活和短距离前向飞行。
控制：ACCBR 兼容 5 维动作空间；当前课程阶段只使用 action[0] 生成非负 vx_cmd。

整体数据流：
    policy action
        -> _pre_physics_step(): 动作限幅、诊断动作覆盖、生成期望速度和期望加速度
        -> _apply_action(): 几何控制器 / INDI / 电机模型把期望加速度转换为旋翼推力
        -> Isaac Lab physics step
        -> _get_observations(): 组装下一步策略观测
        -> _get_rewards() / _get_dones(): 用当前物理状态计算奖励和终止条件

当前训练重点：
    1. 先把任务收缩成“一维低速前飞”：只训练 action[0] -> vx_cmd。
    2. z 方向不交给策略直接控制，而是由高度保持和高度保护逻辑维持在 goal_z 附近。
    3. reward 同时约束前向进度、低空高度带、目标附近减速、速度和姿态稳定。
    4. 诊断环境变量用于隔离策略、控制器和电机模型问题，避免把底层控制问题误判成 PPO 问题。
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

    当前课程阶段动作：
        action[0] -> 非负 vx_cmd，-1 对应 0m/s，0 对应半速前进，1 对应最大前进速度
        action[1:5] 暂时忽略
        vy/body_rates/yaw_rate 固定为 0，z 方向由高度保持逻辑自动维持在 goal_z 附近。

    主要状态缓存：
        _goal_w: 每个并行环境的目标世界坐标，reset 时由 spawn_pos + cfg.goal_* 得到。
        _spawn_w: 每个并行环境的出生点世界坐标，用于越界判断。
        _forces/_moments: 施加到 rotor/base_link 的外力和外力矩，是最终物理输入。
        _prev_dist/_prev_x: reward 中进度项的上一时刻参考值。
        _episode_sums: 每个 episode 的奖励分量累计，用于 TensorBoard 日志。
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
        _hover_omega = 980.0
        self._geo_ctrl = GeometricController(self.num_envs, "ACCBR") # 几何控制器：将期望加速度 / 角速度转为中间控制量
        self._geo_ctrl.disable_acc_load = True # Isaac body_acc_w 语义与 acc_load 估计不匹配，fly_forward 默认禁用
        self._indi_ctrl = IndiController(self.num_envs)              # INDI 控制器：根据当前状态计算目标转速
        self._motor_model = RotorMotor(
            self.num_envs,
            torch.full((self.num_envs, 4), _hover_omega, device=self.device),
        )

        # ── Diagnostic logger (FLY_FORWARD_DIAG=1 to enable) ─────────────────
        # 诊断日志：设置 FLY_FORWARD_DIAG=1 后，将 env_0 的关键状态写入 CSV 便于离线排查。
        # 这些模式互相独立，用来判断问题来自 policy、速度 PD、几何控制器、INDI 还是电机模型。
        self._diag_enabled = os.environ.get("FLY_FORWARD_DIAG", "0") == "1"
        # 设置 FLY_FORWARD_ZERO_ACTION=1 时忽略策略动作，并强制 action[0]=-1，
        # 使 desired_vx=0，真正测试“零前向速度命令”下的正常控制链。
        self._diag_zero_action = os.environ.get("FLY_FORWARD_ZERO_ACTION", "0") == "1"
        # 设置 FLY_FORWARD_DIRECT_HEIGHT=1 时绕过正常控制链，直接施加手写高度保持推力。
        self._diag_direct_height = os.environ.get("FLY_FORWARD_DIRECT_HEIGHT", "0") == "1"
        # 设置 FLY_FORWARD_TEST_X_FORCE=1 时，绕过策略和几何控制器，只保留高度保持，
        # 并在世界系对 base_link 施加固定 x 向水平力，验证“固定水平力 -> 前飞响应”。
        self._diag_test_x_force = os.environ.get("FLY_FORWARD_TEST_X_FORCE", "0") == "1"
        self._diag_test_fx = float(os.environ.get("FLY_FORWARD_TEST_FX", "0.0"))
        # 设置 FLY_FORWARD_CONST_A0=<value> 时固定 action[0]，用于测试给定 vx_cmd 下的物理响应。
        self._diag_const_a0 = os.environ.get("FLY_FORWARD_CONST_A0", None)
        # 设置 FLY_FORWARD_RANDOM=1 时忽略策略动作，改用随机动作做控制链 smoke test。
        self._diag_random = os.environ.get("FLY_FORWARD_RANDOM", "0") == "1"
        if self._diag_enabled:
            self._diag_path = os.environ.get("FLY_FORWARD_DIAG_PATH", "/tmp/fly_forward_diag.csv")
            self._diag_file = open(self._diag_path, "w", newline="")
            self._diag_writer = csv.writer(self._diag_file)
            mode = (
                "test_x_force"
                if self._diag_test_x_force
                else "direct_height"
                if self._diag_direct_height
                else "zero"
                if self._diag_zero_action
                else "random"
                if self._diag_random
                else "policy"
            )
            self._diag_writer.writerow(
                ["phase", "episode", "policy_step",
                 "a0", "a1", "a2", "a3", "a4",
                 "desired_vx", "desired_vy",
                 "cmd_acc_x", "cmd_acc_y", "cmd_acc_z",
                 "vx", "vy", "vz", "z", "dist",
                 "thrust0", "thrust1", "thrust2", "thrust3",
                 "total_thrust", "hover_ratio",
                 "target_rpm0", "target_rpm1", "target_rpm2", "target_rpm3",
                 "acc_cmd_z", "acc_cmd_norm", "collective_thrust_des",
                 "ll_counter"]
            )
            self._diag_episode: int = 0
            self._diag_step: int = 0
            self._diag_prev_done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            print(f"[FlyForwardEnv] Diagnostic logger → {self._diag_path}  mode={mode}")
        self.sampling_time = self.sim.get_physics_dt() * self.cfg.low_level_decimation # 电机模型采样周期

        # ── Goal world position (assigned per-env at reset) ───────────────────
        # 每个 env 的目标世界坐标，在 reset 时根据实际出生点写入。
        self._goal_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._last_desired_vel = torch.zeros(self.num_envs, 3, device=self.device)
        self._last_target_rpm = torch.zeros(self.num_envs, 4, device=self.device)
        self._last_acc_cmd = torch.zeros(self.num_envs, 3, device=self.device)
        self._last_collective_thrust_des = torch.zeros(self.num_envs, device=self.device)

        # ── Spawn world position (_reset_idx 写入，OOB 检查使用) ───────────────────
        # 每个 env 的出生点世界坐标，用于越界判断和前向进度计算。
        self._spawn_w = torch.zeros(self.num_envs, 3, device=self.device)

        # ── Progress tracking: previous full distance to goal ──────────────────
        # 上一步到目标的距离，用于计算真正的“向目标靠近”进度奖励。
        self._prev_dist = torch.zeros(self.num_envs, device=self.device)
        self._prev_x = torch.zeros(self.num_envs, device=self.device)

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
                "forward_progress_reward",
                "forward_speed_reward",
                "dist_reward",
                "altitude_band_reward",
                "height_penalty",
                "fly_high_guard_penalty",
                "slow_near_goal_reward",
                "speed_penalty",
                "speed_limit_penalty",
                "hover_still_penalty",
                "upright_penalty",
                "action_smoothness",
                "action_magnitude",
                "success_reward",
                "success_early_bonus",
                "crash_penalty",
            ]
        }

        self.metrics = {
            "position_error": torch.zeros(self.num_envs, device=self.device),
            "height_error": torch.zeros(self.num_envs, device=self.device),
            "distance_to_goal": torch.zeros(self.num_envs, device=self.device),
            "speed": torch.zeros(self.num_envs, device=self.device),
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
        """解析策略动作并生成几何控制器需要的 setpoint。

        这一层是“高层课程逻辑”，不直接施加力：
            1. 先把策略动作限制到 [-1, 1]，避免网络输出异常值进入控制器。
            2. 诊断模式可以覆盖动作，例如 zero-action、const-a0、random-action。
            3. 当前课程只使用 action[0]，并把 [-1, 1] 线性映射为非负 x 方向期望速度。
            4. y 方向和机体系角速度暂时固定为 0，降低早期探索难度。
            5. z 方向由高度保持 PD 自动生成期望速度，再转换成期望加速度。

        输出 self._setpoint，供 _apply_action() 中的 GeometricController 使用。
        """
        actions = torch.clamp(actions, -1.0, 1.0)
        if self._diag_const_a0 is not None:
            actions = torch.zeros_like(actions)
            actions[:, 0] = float(self._diag_const_a0)
        elif self._diag_zero_action:
            # 零速度 smoke test：隔离策略输出，并强制 desired_vx=0，
            # 只检查高度保持和底层控制是否会自行爬升或自行前移。
            actions = torch.zeros_like(actions)
            actions[:, 0] = -1.0
        elif self._diag_random:
            # 诊断模式下使用随机动作覆盖策略动作，便于排除 policy 输出对控制链的影响。
            actions = torch.rand_like(actions) * 2.0 - 1.0
        self._actions = actions.clone() # 保存当前动作，用于奖励中的动作平滑项

        # 当前局部位置用于高度保护；root_pos_w 是世界坐标，需要减去 env 原点。
        root_pos_w = self._robot.data.root_pos_w
        current_pos = root_pos_w - self.scene.env_origins
        z = current_pos[:, 2]
        # 高度保护门控：z 从 fly_high_guard_z 接近 fly_high_z 时，upward_scale 从 1 降到 0。
        high_guard = torch.clamp(
            (z - self.cfg.fly_high_guard_z) / (self.cfg.fly_high_z - self.cfg.fly_high_guard_z),
            0.0,
            1.0,
        )
        upward_scale = 1.0 - high_guard

        # 将归一化动作映射为期望速度。
        # 当前阶段只有 x 方向来自策略；y 固定为 0；z 由高度保持 PD 计算。
        # 注意不要用 clamp(a0, 0, 1)：初始策略均值约为 0，会直接映射成 desired_vx=0 并导致开局悬停。
        # 线性映射后，a0=-1 -> 0m/s，a0=0 -> 0.5*lin_vel_x_max，a0=1 -> lin_vel_x_max。
        desired_vel = torch.zeros(self.num_envs, 3, device=self.device)
        desired_vel[:, 0] = 0.5 * (actions[:, 0] + 1.0) * self.cfg.lin_vel_x_max
        # 接近目标时主动降低 x 方向速度上限，避免 10m/20m 课程里高速冲过目标却无法满足 success_speed。
        dist_to_goal = torch.norm(root_pos_w - self._goal_w, dim=-1)
        near_goal_speed_scale = torch.clamp(dist_to_goal / self.cfg.near_goal_radius, 0.0, 1.0)
        desired_vel[:, 0] *= near_goal_speed_scale
        desired_vel[:, 1] = 0.0
        current_vel = self._robot.data.root_lin_vel_w
        height_vel_cmd = self.cfg.height_hold_kp * (self.cfg.goal_z - z) - self.cfg.height_hold_damping * current_vel[:, 2]
        desired_vel[:, 2] = torch.clamp_min(height_vel_cmd, -self.cfg.lin_vel_z_down_max)
        desired_vel[:, 2] = torch.minimum(desired_vel[:, 2], self.cfg.lin_vel_z_up_max * upward_scale)
        self._last_desired_vel = desired_vel.clone()
        # 速度误差 PD：P 项跟踪期望速度，D 项抑制速度误差变化过快。
        # 这一步把“目标速度”变成“目标加速度”，后续几何控制器只接收加速度 setpoint。
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
        # body_rates/yaw 均置零，表示当前课程暂不让 PPO 直接学习姿态控制。
        self._setpoint = {
            "lin_acc": commanded_acc,
            "body_rates": torch.zeros(self.num_envs, 3, device=self.device),
            "yaw": torch.zeros(self.num_envs, 1, device=self.device),
            "yaw_rate": torch.zeros(self.num_envs, 1, device=self.device),
            "yaw_acc": torch.zeros(self.num_envs, 1, device=self.device),
        }

        if self._diag_enabled:
            # 只计数；实际施力诊断在 _apply_action() 写入，避免 thrust 滞后一行。
            self._diag_step += 1

    def _write_apply_diag(self) -> None:
        """把“实际施力后”的诊断量写入 CSV。

        这里读取 self._forces，而不是中间变量 thrusts，目的是确认日志和真实施加到 Isaac
        articulation 的外力同源。核心指标是 hover_ratio：
            hover_ratio = applied_total_thrust / true_hover_total

        如果 zero-action 下 hover_ratio 长期明显大于 1，说明底层控制链本身会过推力；
        如果 hover_ratio 约等于 1 但 z 仍发散，则需要继续检查姿态 / 坐标系 / 质量参数。
        """
        if not self._diag_enabled:
            return

        root = self._robot.data.root_state_w
        pos_l = root[0, :3] - self.scene.env_origins[0]
        vel = root[0, 7:10]
        dist = torch.norm(root[0, :3] - self._goal_w[0])

        applied_total = self._forces[0, :, 2].sum()
        true_hover_total = 1.025 * self.cfg.drone_mass * 9.81
        hover_ratio_applied = applied_total / true_hover_total
        lin_acc = self._setpoint["lin_acc"][0] if hasattr(self, "_setpoint") else torch.zeros(3, device=self.device)
        desired_vel = self._last_desired_vel[0] if hasattr(self, "_last_desired_vel") else torch.zeros(3, device=self.device)
        action = self._actions[0] if hasattr(self, "_actions") else torch.zeros(5, device=self.device)
        target_rpm = self._last_target_rpm[0]
        acc_cmd = self._last_acc_cmd[0]
        collective_thrust_des = self._last_collective_thrust_des[0]

        self._diag_writer.writerow([
            "apply",
            self._diag_episode,
            self._diag_step,
            f"{action[0].item():.4f}",
            f"{action[1].item():.4f}",
            f"{action[2].item():.4f}",
            f"{action[3].item():.4f}",
            f"{action[4].item():.4f}",
            f"{desired_vel[0].item():.4f}",
            f"{desired_vel[1].item():.4f}",
            f"{lin_acc[0].item():.4f}",
            f"{lin_acc[1].item():.4f}",
            f"{lin_acc[2].item():.4f}",
            f"{vel[0].item():.4f}",
            f"{vel[1].item():.4f}",
            f"{vel[2].item():.4f}",
            f"{pos_l[2].item():.4f}",
            f"{dist.item():.4f}",
            f"{self._forces[0, 0, 2].item():.6f}",
            f"{self._forces[0, 1, 2].item():.6f}",
            f"{self._forces[0, 2, 2].item():.6f}",
            f"{self._forces[0, 3, 2].item():.6f}",
            f"{applied_total.item():.6f}",
            f"{hover_ratio_applied.item():.4f}",
            f"{target_rpm[0].item():.4f}",
            f"{target_rpm[1].item():.4f}",
            f"{target_rpm[2].item():.4f}",
            f"{target_rpm[3].item():.4f}",
            f"{acc_cmd[2].item():.4f}",
            f"{torch.norm(acc_cmd).item():.4f}",
            f"{collective_thrust_des.item():.6f}",
            str(self._ll_counter),
        ])

    def _apply_action(self) -> None:
        """将高层 setpoint 转换为 Isaac Lab 的外力 / 外力矩。

        正常控制链：
            self._setpoint["lin_acc"]
                -> GeometricController.getCommand()
                -> IndiController.getCommand()
                -> RotorMotor.get_motor_thrusts_moments()
                -> set_external_force_and_torque()

        诊断 / 旁路模式：
            FLY_FORWARD_DIRECT_FORCE=1  直接给定每个旋翼推力，检查物理施力和质量参数。
            FLY_FORWARD_TEST_X_FORCE=1  保留高度保持，并在世界系对 base_link 施加固定 x 向水平力。
            FLY_FORWARD_SIMPLE_CONTROL=1  用简化版“高度推力 + x 方向速度跟踪力”训练，绕过几何控制器和 INDI。
            FLY_FORWARD_DIRECT_HEIGHT=1  手写高度保持推力分支，用于和正常控制链对照。
        """
        if os.environ.get("FLY_FORWARD_DIRECT_FORCE", "0") == "1":
            root = self._robot.data.root_state_w
            mass = self.cfg.drone_mass
            f_hover = mass * 9.81 / 4.0
            scale = float(os.environ.get("FLY_FORWARD_FORCE_SCALE", "1.0"))

            self._forces.zero_()
            self._moments.zero_()
            self._forces[..., 2] = scale * f_hover

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
            self._write_apply_diag()
            return

        if self._diag_test_x_force:
            root = self._robot.data.root_state_w
            pos_l = root[:, :3] - self.scene.env_origins
            vel = root[:, 7:10]

            z = pos_l[:, 2]
            vz = vel[:, 2]

            # 纯水平力诊断：不使用策略，不使用 desired_vx，只验证固定世界系 x 向力是否能产生稳定前飞。
            # 注意 Isaac Lab 的 articulation 外力缓冲区共享 is_global 标志，因此这里把 base force
            # 和 rotor thrust 都统一写成世界系，避免“前一个调用设 global，后一个调用又改回 local”。
            acc_z = (
                self.cfg.height_hold_kp * (self.cfg.goal_z - z)
                - self.cfg.height_hold_damping * vz
            )
            acc_z = torch.clamp(acc_z, -self.cfg.lin_acc_z_down_max, self.cfg.lin_acc_z_up_max)

            mass = self._geo_ctrl.falcon_mass
            collective_thrust = mass * (9.8066 + acc_z)
            collective_thrust = torch.clamp(collective_thrust, 0.0, 4.0 * self.cfg.max_thrust_pp)
            base_forces = torch.zeros(self.num_envs, 1, 3, device=self.device)
            base_forces[:, 0, 0] = self._diag_test_fx

            self._forces.zero_()
            self._moments.zero_()
            self._forces[..., 2] = (collective_thrust / 4.0).unsqueeze(-1).expand(-1, 4)
            self._last_target_rpm.zero_()
            self._last_acc_cmd.zero_()
            self._last_acc_cmd[:, 0] = self._diag_test_fx / mass
            self._last_acc_cmd[:, 2] = 9.8066 + acc_z
            self._last_collective_thrust_des = collective_thrust.clone()

            self._robot.set_external_force_and_torque(
                forces=base_forces,
                torques=torch.zeros_like(base_forces),
                body_ids=self._falcon_body_idx,
                is_global=True,
            )
            self._robot.set_external_force_and_torque(
                forces=self._forces,
                torques=torch.zeros_like(self._forces),
                body_ids=self._rotor_idx,
                is_global=True,
            )
            self._write_apply_diag()
            return

        if os.environ.get("FLY_FORWARD_SIMPLE_CONTROL", "0") == "1":
            root = self._robot.data.root_state_w
            pos_l = root[:, :3] - self.scene.env_origins
            vel = root[:, 7:10]

            z = pos_l[:, 2]
            vx = vel[:, 0]
            vz = vel[:, 2]

            # 简化训练分支：
            #   1. 继续用高度 PD 生成总推力，保证低空稳定。
            #   2. 在世界系 base_link 上直接施加 x 方向速度跟踪力，让前飞命令转成水平加速度。
            # 这样可以绕过 Geometric/INDI/Motor 链，但保留“策略想前飞 -> 物理上确实会前飞”的最小闭环。
            acc_z = (
                self.cfg.height_hold_kp * (self.cfg.goal_z - z)
                - self.cfg.height_hold_damping * vz
            )
            acc_z = torch.clamp(acc_z, -self.cfg.lin_acc_z_down_max, self.cfg.lin_acc_z_up_max)
            acc_x = self.cfg.simple_vx_kp * (self._last_desired_vel[:, 0] - vx)
            acc_x = torch.clamp(acc_x, -self.cfg.simple_acc_x_max, self.cfg.simple_acc_x_max)
            self._setpoint["lin_acc"][:, 0] = acc_x

            mass = self._geo_ctrl.falcon_mass
            collective_thrust = mass * (9.8066 + acc_z)
            collective_thrust = torch.clamp(collective_thrust, 0.0, 4.0 * self.cfg.max_thrust_pp)
            base_forces = torch.zeros(self.num_envs, 1, 3, device=self.device)
            base_forces[:, 0, 0] = mass * acc_x

            self._forces.zero_()
            self._moments.zero_()
            self._forces[..., 2] = (collective_thrust / 4.0).unsqueeze(-1).expand(-1, 4)
            self._last_target_rpm.zero_()
            self._last_acc_cmd.zero_()
            self._last_acc_cmd[:, 0] = acc_x
            self._last_acc_cmd[:, 2] = 9.8066 + acc_z
            self._last_collective_thrust_des = collective_thrust.clone()

            self._robot.set_external_force_and_torque(
                forces=base_forces,
                torques=torch.zeros_like(base_forces),
                body_ids=self._falcon_body_idx,
                is_global=True,
            )
            self._robot.set_external_force_and_torque(
                forces=self._forces,
                torques=torch.zeros_like(self._forces),
                body_ids=self._rotor_idx,
                is_global=True,
            )
            self._write_apply_diag()
            return

        if self._diag_direct_height:
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
            self._write_apply_diag()
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
            self._last_target_rpm = target_rpm.clone()
            self._last_acc_cmd = acc_cmd.clone()
            self._last_collective_thrust_des = self._geo_ctrl.falcon_mass * torch.norm(acc_cmd, dim=1)
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
        self._write_apply_diag()

    # ── Observations ─────────────────────────────────────────────────────────

    def _update_metrics(self) -> None:
        """更新 play/plotting 工具会读取的轻量级指标。

        metrics 不参与训练反传，只用于外部绘图和运行时观察。这里保持和 reward 使用的局部高度
        口径一致：高度误差使用 pos_w - env_origin 后的 local z。
        """
        root = self._robot.data.root_state_w
        pos_w = root[:, :3]
        pos_l = pos_w - self.scene.env_origins
        goal_rel = self._goal_w - pos_w
        self.metrics["position_error"] = torch.norm(goal_rel, dim=-1)
        self.metrics["height_error"] = torch.abs(pos_l[:, 2] - self.cfg.goal_z)
        self.metrics["distance_to_goal"] = torch.norm(goal_rel, dim=-1)
        self.metrics["speed"] = torch.norm(root[:, 7:10], dim=-1)

    def _get_observations(self) -> dict:
        """构造策略网络输入观测。

        观测设计原则：
            1. 姿态直接给 3x3 旋转矩阵，避免四元数符号二义性。
            2. 绝对位置和目标相对位置同时提供，让策略既知道自己在哪里，也知道目标在哪。
            3. x/y 目标相对位置使用 norm_goal_xy_scale 单独缩放，避免 2m/3m 短课程下信号过小。
            4. 所有量纲尽量压到接近 [-1, 1]，减少 PPO 早期优化难度。
        """
        root = self._robot.data.root_state_w
        pos_local = root[:, :3] - self.scene.env_origins   # (N, 3)，局部位置
        lin_vel = root[:, 7:10]                            # (N, 3)，世界系线速度
        ang_vel = root[:, 10:13]                           # (N, 3)，世界系角速度
        rot_mat = matrix_from_quat(root[:, 3:7]).view(self.num_envs, 9) # 姿态四元数转 3x3 旋转矩阵并展平
        goal_rel = self._goal_w - root[:, :3]              # (N, 3)，目标相对当前位置的世界系向量
        self._update_metrics()

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

    def _compute_task_terms(self) -> dict[str, torch.Tensor]:
        """计算 reward 和 done 共用的当前任务状态。

        这个 helper 的目的不是省代码，而是保证 reward 和 done 使用同一时刻、同一口径的状态：
            - 高度统一使用 local z，而不是 world z。
            - success 同时要求到达目标、速度足够低、高度在上下限内、姿态稳定。
            - fly_high/fly_low/out_of_bounds 直接基于当前 step 计算，避免使用上一 step 的 flag。
        """
        root = self._robot.data.root_state_w
        pos_w = root[:, :3]
        pos_l = pos_w - self.scene.env_origins
        vel_w = root[:, 7:10]
        z = pos_l[:, 2]
        dist = torch.norm(pos_w - self._goal_w, dim=-1)
        speed = torch.norm(vel_w, dim=-1)
        forward_progress = pos_w[:, 0] - self._spawn_w[:, 0]

        fly_high = z > self.cfg.fly_high_z
        fly_low = z < self.cfg.fly_low_z
        out_of_bounds = torch.norm((pos_w - self._spawn_w)[:, :2], dim=-1) > self.cfg.out_of_bounds_radius

        rot_mat = matrix_from_quat(root[:, 3:7])
        z_body_z = rot_mat[:, 2, 2]
        stable_upright = z_body_z > 0.9
        within_altitude_limits = (z > self.cfg.fly_low_z) & (z < self.cfg.fly_high_z)
        success = (
            (dist < self.cfg.goal_tolerance)
            & (forward_progress > self.cfg.success_forward_x)
            & (speed < self.cfg.success_speed_tolerance)
            & within_altitude_limits
            & stable_upright
        )

        return {
            "pos_w": pos_w,
            "pos_l": pos_l,
            "vel_w": vel_w,
            "z": z,
            "dist": dist,
            "speed": speed,
            "forward_progress": forward_progress,
            "z_body_z": z_body_z,
            "fly_high": fly_high,
            "fly_low": fly_low,
            "out_of_bounds": out_of_bounds,
            "success": success,
        }

    def _get_rewards(self) -> torch.Tensor:
        """计算每个并行环境当前 step 的奖励。

        当前 reward 是“短距离低空前飞课程”的 shaping，而不是最终 200m 长航程版本：
            - progress_rew: 到目标距离变小才给奖励，防止远离目标。
            - forward_progress_rew: 只有向目标方向前进时才给奖励。
            - forward_speed_rew: 只有移动并接近目标时，奖励接近 target_forward_speed 的前向速度。
            - height/fly_high/fly_low: 高度只作为上下界约束，不再给低空存活奖励。

        注意 _prev_dist、_prev_x、_prev_actions 会在这里更新，因此它们必须在 reset 时重新初始化。
        持续存活型 dense reward 需要乘 step_dt，否则 PPO 会偏向“悬停到超时刷累计奖励”。
        """
        terms = self._compute_task_terms()
        pos_w = terms["pos_w"]
        pos_l = terms["pos_l"]
        vel_w = terms["vel_w"]
        z = terms["z"]
        dist = terms["dist"]
        speed = terms["speed"]
        forward_progress = terms["forward_progress"]
        z_body_z = terms["z_body_z"]
        cmd_acc = self._setpoint["lin_acc"]
        base_id = self._falcon_body_idx[0]
        body_acc = self._robot.data.body_acc_w[:, base_id, :3]

        # 1. 真正的到目标进度：必须动起来且距离变小，静止或远离目标不给正奖励。
        delta_dist = self._prev_dist - dist
        moving = speed > self.cfg.move_reward_min_speed
        moving_to_goal = moving & (delta_dist > 0.0)
        progress_rew = (
            self.cfg.progress_reward_weight
            * torch.clamp(delta_dist, 0.0, 0.2)
            * moving.float()
        )
        self._prev_dist = dist.clone()

        # 1b. 目标方向前进奖励：只奖励朝目标方向移动，不奖励原地存活。
        delta_x = pos_l[:, 0] - self._prev_x
        self._prev_x = pos_l[:, 0].clone()

        # 2. 距离目标越近越好，但必须处于“移动并接近目标”的状态，避免悬停刷距离 dense reward。
        dist_rew = (
            self.cfg.dist_reward_weight
            * torch.exp(-self.cfg.dist_reward_scale * dist)
            * moving_to_goal.float()
            * self.step_dt
        )

        # 3. 高度只作为上下界安全约束，范围内不再给正奖励。
        within_altitude_limits = (z > self.cfg.fly_low_z) & (z < self.cfg.fly_high_z)
        altitude_safe = within_altitude_limits.float()
        altitude_band_rew = torch.zeros_like(dist)
        goal_rel_x = self._goal_w[:, 0] - pos_w[:, 0]
        before_goal = goal_rel_x > 0.0
        forward_progress_rew = (
            self.cfg.forward_progress_reward_weight
            * torch.clamp(delta_x, 0.0, 0.05)
            * altitude_safe
            * moving.float()
            * before_goal.float()
        )
        vx = vel_w[:, 0]
        approach_phase = dist > self.cfg.goal_tolerance
        forward_speed_rew = (
            self.cfg.forward_speed_reward_weight
            * torch.exp(-((vx - self.cfg.target_forward_speed) ** 2) / self.cfg.forward_speed_sigma)
            * altitude_safe
            * approach_phase.float()
            * before_goal.float()
            * moving_to_goal.float()
            * self.step_dt
        )

        # 4. 不再惩罚偏离 goal_z，高度由控制器保持，reward 只关心越界。
        height_pen = torch.zeros_like(dist)

        # 5. 接近高度上限时轻度惩罚，真正越界由 done + crash_pen 处理。
        fly_high_guard_pen = (
            -self.cfg.fly_high_guard_penalty_weight
            * torch.relu(z - self.cfg.fly_high_guard_z).square()
        )

        # 6. 目标附近必须慢下来。低速奖励只在“已完成大部分前向位移且真正接近目标”时生效，
        # 避免策略在离目标还远时靠悬停刷累计奖励。
        near_goal = dist < self.cfg.near_goal_reward_radius
        final_approach = forward_progress > (self.cfg.success_forward_x - self.cfg.goal_tolerance)
        slow_near_goal_rew = (
            self.cfg.slow_near_goal_reward_weight
            * torch.exp(-self.cfg.speed_reward_scale * speed)
            * final_approach.float()
            * near_goal.float()
            * self.step_dt
        )

        # 7. 全局速度惩罚，避免高速冲过目标
        speed_pen = -self.cfg.speed_penalty_weight * speed.square()
        speed_limit_pen = (
            -self.cfg.speed_limit_penalty_weight
            * torch.relu(speed - self.cfg.speed_soft_limit).square()
        )
        hovering_far = (
            (speed < self.cfg.hover_still_speed_threshold)
            & (forward_progress < self.cfg.success_forward_x - self.cfg.goal_tolerance)
            & (dist > self.cfg.near_goal_reward_radius)
        )
        hover_still_pen = -self.cfg.hover_still_penalty_weight * hovering_far.float() * self.step_dt

        # 8. 姿态稳定
        upright_pen = self.cfg.upright_penalty_weight * (z_body_z - 1.0)

        # 9. 动作平滑
        delta_a = self._actions - self._prev_actions
        smooth_pen = -self.cfg.action_smoothness_weight * delta_a.square().sum(-1)
        action_mag_pen = -self.cfg.action_magnitude_weight * self._actions[:, 0].square()
        self._prev_actions = self._actions.clone()

        # 10. 成功奖励：保底成功奖励 + 按剩余时间递减的早到奖励。
        # 这样“快速到达并稳定停下”会比“慢慢挪到目标”拿到更高回报。
        success_mask = terms["success"].float()
        elapsed_fraction = self.episode_length_buf.float() / max(float(self.max_episode_length - 1), 1.0)
        remaining_fraction = torch.clamp(1.0 - elapsed_fraction, 0.0, 1.0)
        success_base_rew = success_mask * self.cfg.success_reward
        success_early_bonus = success_mask * self.cfg.success_early_bonus * remaining_fraction
        success_rew = success_base_rew + success_early_bonus

        # 11. 失败惩罚
        crash_pen = -(
            terms["fly_high"].float() * self.cfg.fly_high_penalty
            + terms["fly_low"].float() * self.cfg.fly_low_penalty
            + terms["out_of_bounds"].float() * self.cfg.out_of_bounds_penalty
        )

        total = (
            progress_rew
            + forward_progress_rew
            + forward_speed_rew
            + dist_rew
            + altitude_band_rew
            + height_pen
            + fly_high_guard_pen
            + slow_near_goal_rew
            + speed_pen
            + speed_limit_pen
            + hover_still_pen
            + upright_pen
            + smooth_pen
            + action_mag_pen
            + success_rew
            + crash_pen
        )

        # 累计各奖励分量，reset 时写入训练日志，方便观察 reward shaping 是否按预期工作。
        self._episode_sums["progress_reward"] += progress_rew
        self._episode_sums["forward_progress_reward"] += forward_progress_rew
        self._episode_sums["forward_speed_reward"] += forward_speed_rew
        self._episode_sums["dist_reward"] += dist_rew
        self._episode_sums["altitude_band_reward"] += altitude_band_rew
        self._episode_sums["height_penalty"] += height_pen
        self._episode_sums["fly_high_guard_penalty"] += fly_high_guard_pen
        self._episode_sums["slow_near_goal_reward"] += slow_near_goal_rew
        self._episode_sums["speed_penalty"] += speed_pen
        self._episode_sums["speed_limit_penalty"] += speed_limit_pen
        self._episode_sums["hover_still_penalty"] += hover_still_pen
        self._episode_sums["upright_penalty"] += upright_pen
        self._episode_sums["action_smoothness"] += smooth_pen
        self._episode_sums["action_magnitude"] += action_mag_pen
        self._episode_sums["success_reward"] += success_rew
        self._episode_sums["success_early_bonus"] += success_early_bonus
        self._episode_sums["crash_penalty"] += crash_pen

        self.extras.setdefault("log", {}).update(
            {
                "Episode Reward/progress": progress_rew.mean().item(),
                "Episode Reward/forward_progress": forward_progress_rew.mean().item(),
                "Episode Reward/forward_speed": forward_speed_rew.mean().item(),
                "Episode Reward/dist": dist_rew.mean().item(),
                "Episode Reward/altitude_band": altitude_band_rew.mean().item(),
                "Episode Reward/height_pen": height_pen.mean().item(),
                "Episode Reward/fly_high_guard_pen": fly_high_guard_pen.mean().item(),
                "Episode Reward/slow_near_goal": slow_near_goal_rew.mean().item(),
                "Episode Reward/speed_pen": speed_pen.mean().item(),
                "Episode Reward/speed_limit_pen": speed_limit_pen.mean().item(),
                "Episode Reward/hover_still_pen": hover_still_pen.mean().item(),
                "Episode Reward/upright": upright_pen.mean().item(),
                "Episode Reward/smooth": smooth_pen.mean().item(),
                "Episode Reward/action_magnitude": action_mag_pen.mean().item(),
                "Episode Reward/success": success_rew.mean().item(),
                "Episode Reward/success_early_bonus": success_early_bonus.mean().item(),
                "Episode Reward/crash": crash_pen.mean().item(),
                "Metrics/z_mean": z.mean().item(),
                "Metrics/z_max": z.max().item(),
                "Metrics/dist_mean": dist.mean().item(),
                "Metrics/speed_mean": speed.mean().item(),
                "Metrics/forward_progress_mean": forward_progress.mean().item(),
                "Metrics/cmd_acc_x_mean": cmd_acc[:, 0].mean().item(),
                "Metrics/cmd_acc_y_mean": cmd_acc[:, 1].mean().item(),
                "Metrics/cmd_acc_z_mean": cmd_acc[:, 2].mean().item(),
                "Metrics/body_acc_x_mean": body_acc[:, 0].mean().item(),
                "Metrics/body_acc_y_mean": body_acc[:, 1].mean().item(),
                "Metrics/body_acc_z_mean": body_acc[:, 2].mean().item(),
                "Metrics/cmd_acc_x_abs_mean": cmd_acc[:, 0].abs().mean().item(),
                "Metrics/body_acc_x_abs_mean": body_acc[:, 0].abs().mean().item(),
                "Metrics/success_remaining_fraction": remaining_fraction.mean().item(),
            }
        )

        return total

    # ── Terminations ──────────────────────────────────────────────────────────

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """计算终止条件和超时条件。

        terminated 表示任务语义上的结束：
            - fly_high: 局部高度超过 fly_high_z。
            - fly_low: 局部高度低于 fly_low_z。
            - out_of_bounds: 水平距离出生点过远。
            - success: 到达目标且速度 / 高度 / 姿态满足约束。

        timed_out 只表示 episode 达到最大长度，不代表成功或失败。
        """
        terms = self._compute_task_terms()
        self._fly_high = terms["fly_high"]
        self._fly_low = terms["fly_low"]
        self._out_of_bounds = terms["out_of_bounds"]
        self._success = terms["success"]

        terminated = self._fly_high | self._fly_low | self._out_of_bounds | self._success
        timed_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, timed_out

    # ── Reset ─────────────────────────────────────────────────────────────────

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """重置指定并行环境，并初始化目标、出生点、控制缓存和日志。

        reset 做四类事情：
            1. 把上一个 episode 的奖励累计和终止原因写入 self.extras["log"]。
            2. 随机化出生点 x/y，并基于实际出生点生成目标点。
            3. 把无人机 pose、速度、关节状态写回仿真。
            4. 清空控制器、电机、reward 进度和诊断缓存，避免跨 episode 污染。
        """
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

        # 固定目标：放在实际出生点前方。
        # 这样 spawn noise 不会改变“相对目标距离”，训练目标始终是向前飞 cfg.goal_x 米。
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
        self._prev_x[env_ids] = spawn_pos[:, 0]
        self._last_desired_vel[env_ids] = 0.0
        self._last_target_rpm[env_ids] = 0.0
        self._last_acc_cmd[env_ids] = 0.0
        self._last_collective_thrust_des[env_ids] = 0.0
