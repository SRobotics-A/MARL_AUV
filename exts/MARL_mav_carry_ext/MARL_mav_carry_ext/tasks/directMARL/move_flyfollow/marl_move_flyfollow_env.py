# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
from collections.abc import Sequence

# 外环几何控制器（计算期望姿态/加速度）和内环 INDI 控制器（计算转子转速）
from MARL_mav_carry_ext.controllers import GeometricController, IndiController
# 电机模型：将目标转速转换为物理推力和力矩
from MARL_mav_carry_ext.controllers.motor_model import RotorMotor
# 多无人机相对位置/距离工具函数
from MARL_mav_carry_ext.tasks.managerbased.mdp_llc.utils import (
    get_drone_pdist,   # 计算无人机两两间距离矩阵
    get_drone_rpos,    # 计算无人机两两间相对位置
)

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectMARLEnv          # IsaacLab 多智能体环境基类
from isaaclab.sensors import ContactSensor
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import CircularBuffer          # 环形缓冲区，用于存储历史观测
from isaaclab.utils.math import (
    euler_xyz_from_quat,   # 四元数 → 欧拉角
    matrix_from_quat,      # 四元数 → 旋转矩阵（用于观测中的姿态表示）
    quat_rotate,           # 向量的四元数旋转
)

from .marl_move_flyfollow_env_cfg import MARLMoveEnvCfg


class MARLMoveEnv(DirectMARLEnv):
    """move_flyfollow 任务环境：3架 Falcon 无人机追捕 4个匀速移动物块。

    核心设计：
    - 目标中心制（target-centric）奖励：距离奖励按最近无人机到各目标的距离计算
    - 可撤销捕获（revocable capture）：每帧实时更新，无人机飞远则状态失效
    - 计时器暂停：计时器在条件不满足时暂停（不重置），避免因短暂失跟而从零计时
    - 集中训练分散执行（CTDE）：Critic 使用全局 state，Actor 只用局部 obs
    - 控制层次：policy → velocity cmd → PD → geometric控制器 → INDI → 电机模型
    """

    cfg: MARLMoveEnvCfg

    def __init__(self, cfg: MARLMoveEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._num_drones = 3
        self._control_mode = cfg.control_mode

        # ===== 机体索引：从 robot_0 获取（所有无人机结构相同）=====
        # base_link：无人机主体；rotor_.*：4个旋翼
        self._falcon_idx = torch.tensor(
            self.robots[0].find_bodies(".*base_link")[0], device=self.device
        )
        self._falcon_rotor_idx = torch.tensor(
            self.robots[0].find_bodies(".*rotor_.*")[0], device=self.device
        )

        # ===== 历史观测缓冲区（CircularBuffer）=====
        # 每个智能体独立维护 history_len 步的观测历史，拼接后输入 Actor
        self._observation_buffers = {}
        for agent in self.cfg.possible_agents:
            self._observation_buffers[agent] = CircularBuffer(
                cfg.history_len, self.num_envs, device=self.device
            )

        # ===== 控制力/力矩缓冲区 =====
        # _forces[i]: shape (num_envs, 4, 3) — 第 i 架无人机的 4 个旋翼推力（仅 z 分量有效）
        # _moments: shape (num_envs, num_drones, 3) — 各无人机机体力矩（仅 z 分量，即偏航力矩）
        self._forces = [
            torch.zeros(self.num_envs, 4, 3, device=self.device)
            for _ in range(self._num_drones)
        ]
        self._moments = torch.zeros(
            self.num_envs, self._num_drones, 3, device=self.device
        )

        # ===== 控制设定点（每步由 _pre_physics_step 填充）=====
        # _setpoints[agent]["lin_acc"/"body_rates"/"yaw"] 等，传给外环控制器
        self._setpoints = {}
        # 上一步动作（用于动作平滑度奖励计算）
        self.prev_actions = {}
        for agent in self.cfg.possible_agents:
            self._setpoints[agent] = {}
            if self._control_mode == "geometric":
                self.prev_actions[agent] = torch.zeros(
                    self.num_envs, 12, device=self.device
                )
            elif self._control_mode == "ACCBR":
                # ACCBR 模式：前3维速度指令 + 后3维机体角速率指令
                self.prev_actions[agent] = torch.zeros(
                    self.num_envs, 6, device=self.device
                )

        # ===== 无人机状态缓冲区（每帧从仿真读取）=====
        # 所有张量形状：(num_envs, num_drones, ...) 即 (E, D, ...)
        self.drone_positions = torch.zeros(
            self.num_envs, self._num_drones, 3, device=self.device
        )
        self.drone_orientations = torch.zeros(
            self.num_envs, self._num_drones, 4, device=self.device
        )
        self.drone_orientations[..., 0] = 1.0  # 初始化为单位四元数 [w=1, x=0, y=0, z=0]
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
        # jerk（加加速度）：用于几何控制器的前馈项
        self._drone_jerk = torch.zeros(
            self.num_envs, self._num_drones, 3, device=self.device
        )
        self._drone_prev_acc = torch.zeros(
            self.num_envs, self._num_drones, 3, device=self.device
        )
        # PD 速度控制器：上一步的速度误差，用于微分项计算（防抖）
        self._prev_vel_error = torch.zeros(
            self.num_envs, self._num_drones, 3, device=self.device
        )

        # ===== 三层控制器（每架无人机独立实例化）=====
        # 外环（几何控制器）：将速度/加速度设定点 → 期望姿态四元数 q_cmd 和期望加速度 acc_cmd
        self.geo_controllers = {}
        for i in range(self._num_drones):
            self.geo_controllers[i] = GeometricController(
                self.num_envs, self._control_mode
            )
        self._ll_counter = 0
        # 固定偏航角参考（0 rad），让无人机保持朝向 x 轴正方向
        self._constant_yaw = torch.zeros([self.num_envs, 1], device=self.device)
        self._zeros = torch.zeros([self.num_envs, 3], device=self.device)

        # 内环（INDI 控制器）：将期望角加速度 α_cmd 和线性加速度 acc_cmd → 目标旋翼转速
        self._indi_controllers = {}
        for i in range(self._num_drones):
            self._indi_controllers[i] = IndiController(self.num_envs)

        # 电机模型：将目标 RPM → 实际推力和力矩（考虑电机动态响应）
        self.motor_models = {}
        initial_rpms = [
            # 悬停转速约 1355 RPM（对应 Falcon 的悬停工作点）
            torch.tensor([[1355.0, 1355.0, 1355.0, 1355.0]], device=self.device).repeat(
                self.num_envs, 1
            )
            for _ in range(self._num_drones)
        ]
        for i in range(self._num_drones):
            self.motor_models[i] = RotorMotor(self.num_envs, initial_rpms[i])
        # 内环采样周期：物理 dt × 低级控制抽取率
        self.sampling_time = self.sim.get_physics_dt() * self.cfg.low_level_decimation

        # ===== 移动物块（目标）状态缓冲区 =====
        # shape: (num_envs, num_targets, 3)  — 4个目标的位置（局部坐标）
        self.num_targets = cfg.num_targets
        self.target_positions = torch.zeros(
            self.num_envs, self.num_targets, 3, device=self.device
        )
        # 速度：x 方向匀速运动（cfg.target_velocity），y/z 为 0
        self.target_velocities = torch.zeros(
            self.num_envs, self.num_targets, 3, device=self.device
        )
        # 捕获状态：bool (E, T)，每帧实时更新；若无人机飞远则自动置 False（可撤销捕获）
        self.target_captured = torch.zeros(
            self.num_envs, self.num_targets, dtype=torch.bool, device=self.device
        )
        # 预创建单位四元数，写入仿真位置时使用，避免每帧重新分配内存
        self._target_quat = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]], device=self.device
        ).repeat(self.num_envs, 1)
        # 目标价值（固定不变）：红4分、黄3分、绿2分、蓝1分
        # shape: (num_envs, num_targets)，用于距离奖励加权
        self.target_values = (
            torch.tensor(cfg.target_values, device=self.device)
            .unsqueeze(0)
            .repeat(self.num_envs, 1)
        )

        # 记录哪个无人机捕获了哪个物块 (-1 表示未被捕获)
        # 注意：该记录只增不减（用于边界豁免判断）；实时捕获状态以 target_captured 为准
        self.target_captured_by = torch.full(
            (self.num_envs, self.num_targets), -1, dtype=torch.long, device=self.device
        )

        # ===== 奖励分项累计（每个 episode 结束时输出 log）=====
        # 每个 key 对应一个奖励分量，shape: (num_envs,)
        self._episode_sums = {
            key: torch.zeros(
                self.num_envs,
                dtype=torch.float,
                device=self.device,
            )
            for key in [
                "distance_reward",    # 距离奖励：exp(-dist) × 目标价值
                "success_reward",     # 成功奖励（已移除，保留 key 供日志兼容）
                "tracking_reward",    # 跟随奖励：仅在捕获状态下给予
                "action_smoothness",  # 动作平滑度奖励：抑制动作抖动
                "body_rate_penalty",  # 机体角速率惩罚：抑制过激机动
                "velocity_penalty",   # 速度惩罚：抑制高速飞行
                "force_penalty",      # 推力惩罚：抑制过大旋翼推力
                "height_reward",      # 高度奖励：鼓励维持期望高度
                "collision_penalty",  # 无人机碰撞惩罚
                "drone_out",          # 出界惩罚
                "fly_low",            # 飞太低惩罚
                "illegal_contact",    # 非法接触惩罚（碰地面等）
                "time_penalty",       # 时间惩罚（已移除）
                "upright_penalty",    # 姿态惩罚：防止翻滚
                "height_penalty",     # 高度超限惩罚（线性）
            ]
        }

        # 性能指标（当前未用于奖励，仅供 debug）
        self.metrics = {}
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["orientation_error"] = torch.zeros(
            self.num_envs, device=self.device
        )

        # ===== 终止条件缓冲区（bool, shape: (num_envs,)）=====
        # 各条件独立记录，便于分项统计原因
        self.falcon_fly_low = torch.zeros(
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
        # all_targets_captured：持续跟随计时满 sustained_follow_duration 秒（成功终止）
        self.all_targets_captured = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        # 持续跟随计时器：条件不满足时暂停（不重置），避免短暂失跟后从零开始
        self._sustained_follow_timer = torch.zeros(self.num_envs, device=self.device)
        # targets_out_of_bounds：目标物块超出边界
        self.targets_out_of_bounds = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.time_out = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        # ===== 观测归一化参数（目前依赖 RunningStandardScaler，仅备用）=====
        self._norm_pos_scale = self.cfg.bounding_box_threshold
        self._norm_vel_scale = 5.0
        self._norm_dist_scale = self.cfg.bounding_box_threshold * 2

        # debug vis
        self.set_debug_vis(cfg.debug_vis)

    def _setup_scene(self):
        """构建仿真场景：3个独立无人机 Articulation + 4个运动物块 RigidObject。

        采用 IsaacLab-HARL 模式：每架无人机是独立的 Articulation 对象，
        避免多智能体合并为单个 Articulation 带来的索引复杂度。
        """
        # ===== 创建3个独立的Articulation对象（每架无人机独立）=====
        self.robots = []
        self.contact_sensors = []

        for i in range(3):
            robot_cfg = getattr(self.cfg, f"robot_{i}")
            # Articulation 构造函数会通过 scene replication 调用 spawn 函数
            robot = Articulation(robot_cfg)
            self.robots.append(robot)
            self.scene.articulations[f"robot_{i}"] = robot

            # 每架无人机对应独立的接触传感器（用于检测非法碰撞）
            contact_cfg = getattr(self.cfg, f"contact_forces_{i}")
            contact = ContactSensor(contact_cfg)
            self.contact_sensors.append(contact)
            self.scene.sensors[f"contact_forces_{i}"] = contact

        # ===== 创建4个移动物块（目标）=====
        # 物块设为 kinematic（运动学刚体），由代码直接控制位置，不参与物理碰撞求解
        from isaaclab.assets import RigidObjectCfg

        self.targets = []
        for i, color_name in enumerate(self.cfg.target_colors):
            color_rgb = self.cfg.target_color_rgb[color_name]
            target_cfg = RigidObjectCfg(
                prim_path=f"/World/envs/env_.*/target_{i}",
                spawn=sim_utils.CuboidCfg(
                    size=self.cfg.target_size,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        kinematic_enabled=True,   # 运动学模式：位置由代码写入，忽略碰撞力
                        disable_gravity=True,     # 禁用重力，保持地面高度匀速滑动
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=color_rgb   # 颜色区分：红/黄/绿/蓝
                    ),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.0, 0.0, self.cfg.target_spawn_z),
                    rot=(1.0, 0.0, 0.0, 0.0),
                ),
            )
            target = RigidObject(target_cfg)
            self.scene.rigid_objects[f"target_{i}"] = target
            self.targets.append(target)

        # 地面平面（防止无人机无限下落）
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # 复制并行环境（每个 env 独立一份场景）
        self.scene.clone_environments(copy_from_source=False)
        # 添加场景照明
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        """将 policy 输出的动作解码为控制器设定点。

        ACCBR 模式动作向量（6维）：
          - action[:, 0:3]：归一化速度指令（×lin_vel_max → 期望速度 m/s）
          - action[:, 3:6]：归一化机体角速率指令（×ang_vel_max → rad/s）

        速度→加速度转换（PD 控制器）：
          commanded_acc = Kp * vel_err + Kd * (vel_err - prev_vel_err) / dt
          作用：将速度跟踪误差转换为加速度前馈，再由外环几何控制器处理
        """
        # 保存上一步动作，用于动作平滑度奖励计算
        for agent in self.cfg.possible_agents:
            self.prev_actions[agent][:] = self.actions[agent]
            self.actions[agent][:] = actions[agent]

        for drone, action in actions.items():
            if self._control_mode == "geometric":
                # 几何模式：policy 直接输出位置/速度/加速度/jerk 设定点
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
                # ACCBR 模式：policy 输出期望速度（归一化），通过 PD 控制器转换为加速度
                drone_idx = self.cfg.possible_agents.index(drone)
                desired_vel = action[:, :3] * self.cfg.lin_vel_max   # 还原为 m/s
                current_vel = self.drone_linear_velocities[:, drone_idx]
                vel_error = desired_vel - current_vel
                # 微分项：(当前误差 - 上步误差) / dt，近似速度误差的时间导数
                d_error = (
                    vel_error - self._prev_vel_error[:, drone_idx]
                ) / self.step_dt
                self._prev_vel_error[:, drone_idx] = vel_error
                # PD 控制律：Kp=3.0, Kd=0.5（见 cfg）
                commanded_acc = self.cfg.vel_Kp * vel_error + self.cfg.vel_Kd * d_error
                # 限幅：防止加速度指令过大导致仿真不稳
                commanded_acc = torch.clamp(
                    commanded_acc,
                    -self.cfg.lin_acc_max,
                    self.cfg.lin_acc_max,
                )
                self._setpoints[drone]["lin_acc"] = commanded_acc
                # 机体角速率（偏航/俯仰/滚转）直接从 action 的后3维取得
                self._setpoints[drone]["body_rates"] = (
                    action[:, 3:6] * self.cfg.ang_vel_max
                )

            # 固定偏航参考为 0，无人机不主动旋转；ACCBR 模式下用 action[5] 作为偏航角速率
            self._setpoints[drone]["yaw"] = self._constant_yaw
            self._setpoints[drone]["yaw_rate"] = (
                action[:, 5].unsqueeze(-1) * self.cfg.ang_vel_max
                if self._control_mode == "ACCBR"
                else self._constant_yaw
            )
            self._setpoints[drone]["yaw_acc"] = self._constant_yaw

    def _apply_action(self) -> None:
        """执行三层控制器链，将设定点转换为实际推力并写入仿真。

        控制层次（每个低级控制周期执行一次）：
          1. 几何控制器（外环）：设定点 → 期望姿态 q_cmd + 期望加速度 acc_cmd
          2. INDI 控制器（内环）：q_cmd + acc_cmd → 目标旋翼转速
          3. 电机模型：目标转速 → 实际推力 thrusts + 力矩 moments

        低级控制抽取：low_level_decimation=1 时每物理步执行一次；
        若 >1 则高级控制步内复用上一步的控制输出。
        """
        if self._ll_counter % self.cfg.low_level_decimation == 0:
            all_thrusts = []
            all_moments = []

            # ===== 从每个独立 robot 读取当前状态 =====
            # root_state_w shape: (num_envs, 13) = [pos(3), quat(4), lin_vel(3), ang_vel(3)]
            for i, robot in enumerate(self.robots):
                root_state = robot.data.root_state_w  # (N, 13)

                drone_pos = root_state[:, :3] - self.scene.env_origins  # 转为局部坐标系
                drone_quat = root_state[:, 3:7]
                drone_lin_vel = root_state[:, 7:10]
                drone_ang_vel = root_state[:, 10:13]

                # body_acc_w: (N, num_bodies, 6)，前3维线加速度，后3维角加速度
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

            # ===== 依次通过三层控制器计算每架无人机的推力和力矩 =====
            for i in range(self._num_drones):
                # 组装当前无人机状态字典（传给控制器）
                drone_states: dict = {}
                drone_states["pos"] = self.drone_positions[:, i]
                drone_states["quat"] = self.drone_orientations[:, i]
                drone_states["lin_vel"] = self.drone_linear_velocities[:, i]
                drone_states["ang_vel"] = self.drone_angular_velocities[:, i]
                drone_states["lin_acc"] = self.drone_linear_accelerations[:, i]
                drone_states["ang_acc"] = self.drone_angular_accelerations[:, i]
                # jerk（线性加加速度）= (当前加速度 - 上步加速度) / 物理 dt
                self._drone_jerk[:, i] = (
                    drone_states["lin_acc"] - self._drone_prev_acc[:, i]
                ) / (self.physics_dt)
                drone_states["jerk"] = self._drone_jerk[:, i]
                self._drone_prev_acc[:, i] = drone_states["lin_acc"]

                agent_name = self.cfg.possible_agents[i]

                # 第1层：几何控制器（外环）
                # 输出：alpha_cmd（角加速度指令）、acc_load（负载加速度）、acc_cmd（期望线加速度）、q_cmd（期望姿态）
                alpha_cmd, acc_load, acc_cmd, q_cmd = self.geo_controllers[
                    i
                ].getCommand(
                    drone_states,
                    self._forces[i],  # 上一步的推力，用于前馈计算
                    self._setpoints[agent_name],
                )

                # 第2层：INDI 控制器（内环）
                # 输出：目标旋翼转速 target_rpm (N, 4)
                target_rpm = self._indi_controllers[i].getCommand(
                    drone_states,
                    self._forces[i],
                    alpha_cmd,
                    acc_cmd,
                    acc_load,
                )

                # 第3层：电机模型
                # 输出：thrusts (N, 4)，moments (N, 4)
                thrusts, moments = self.motor_models[i].get_motor_thrusts_moments(
                    target_rpm, self.sampling_time
                )
                all_thrusts.append(thrusts)
                all_moments.append(moments)

            # ===== 限幅：防止推力/力矩超出物理极限 =====
            for i in range(self._num_drones):
                # 推力限幅：[0, max_thrust_pp]，仅保留 z 分量（垂直推力）
                forces_i = torch.clamp(
                    all_thrusts[i], min=0.0, max=self.cfg.max_thrust_pp
                )
                self._forces[i][..., 2] = forces_i

                # 力矩限幅：[-1, 1] N·m，取 4 旋翼之和作为偏航力矩
                moments_i = torch.clamp(all_moments[i], min=-1.0, max=1.0)
                self._moments[:, i, 2] = moments_i.sum(-1)

            self._ll_counter = 0
        self._ll_counter += 1

        # ===== 将推力/力矩写入仿真（每架无人机独立写，互不干扰）=====
        for i, robot in enumerate(self.robots):
            # 机体力矩作用于 base_link
            robot.set_external_force_and_torque(
                forces=torch.zeros((self.num_envs, 1, 3), device=self.device),
                torques=self._moments[:, i].unsqueeze(1),  # (N, 1, 3)
                body_ids=torch.zeros(1, dtype=torch.int, device=self.device),
            )

            # 旋翼推力作用于 4 个 rotor 刚体（仅 z 方向推力）
            robot.set_external_force_and_torque(
                forces=self._forces[i],  # (N, 4, 3)
                torques=torch.zeros_like(self._forces[i]),
                body_ids=self._falcon_rotor_idx,
            )

        # ===== 更新目标物块位置（匀速直线运动，x 方向正向）=====
        # 使用高级控制时间步长（decimation × physics_dt）推进位置
        dt = self.physics_dt * self.cfg.decimation
        self.target_velocities[:, :, 0] = self.cfg.target_velocity   # x 方向速度
        self.target_velocities[:, :, 1] = 0.0
        self.target_velocities[:, :, 2] = 0.0

        self.target_positions += self.target_velocities * dt   # 位置积分

        # 将位置写回仿真（需加 env_origins 转换到世界坐标）
        for i, target in enumerate(self.targets):
            target_poses = torch.cat(
                [
                    self.target_positions[:, i] + self.scene.env_origins,
                    self._target_quat,  # 固定朝向，不旋转
                ],
                dim=-1,
            )
            target.write_root_pose_to_sim(target_poses)

    def _normalize_observation(self, obs: torch.Tensor) -> torch.Tensor:
        """No manual normalization — relying on skrl's RunningStandardScaler."""
        return obs

    def _get_observations(self) -> dict[str, torch.Tensor]:
        """构建每个智能体的局部观测向量。

        观测结构（每步 49 维，history_len=3 时输入 147 维）：
          1. 自身状态（15维）：位置(3) + 线速度(3) + 旋转矩阵(9)
          2. 其他无人机相对位置（6维）：(num_drones-1) × 3
          3. 目标信息（20维）：相对位置(12) + 捕获状态(4) + 价值(4)
          4. 距离信息（4维）：到各目标的水平距离
          5. 最近标志（4维）：当前无人机是否是各目标最近的无人机（one-hot 风格）

        注：观测使用 CircularBuffer 拼接历史帧，提供时序信息（替代 RNN）
        """
        # ===== 刷新无人机状态缓冲区（每步从仿真读取）=====
        for i, robot in enumerate(self.robots):
            root_state = robot.data.root_state_w  # (N, 13)
            self.drone_positions[:, i] = root_state[:, :3] - self.scene.env_origins
            self.drone_orientations[:, i] = root_state[:, 3:7]
            self.drone_linear_velocities[:, i] = root_state[:, 7:10]
            self.drone_angular_velocities[:, i] = root_state[:, 10:13]

        # 旋转矩阵（由四元数转换）：作为姿态表示避免四元数符号歧义
        self.drone_rot_matrices[:] = matrix_from_quat(self.drone_orientations)

        # ===== 计算所有无人机到所有目标的水平（XY）距离矩阵 =====
        # shape: (num_envs, num_drones, num_targets)
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

        # 对每个目标，找到距离最近的无人机索引，shape: (num_envs, num_targets)
        closest_drone_to_target = torch.argmin(
            drone_to_target_distances, dim=1
        )  # (N, T)

        observations = {}
        for drone_idx, agent_name in enumerate(self.cfg.possible_agents):
            # --- 1. 自身状态（15维）---
            # 位置(3) + 线速度(3) + 旋转矩阵展平(9)
            obs_self = torch.cat(
                [
                    self.drone_positions[:, drone_idx],  # 3
                    self.drone_linear_velocities[:, drone_idx],  # 3
                    self.drone_rot_matrices[:, drone_idx].view(self.num_envs, -1),  # 9
                ],
                dim=-1,
            )

            # --- 2. 其他无人机相对位置（6维 = 2×3）---
            # 使用相对位置而非绝对位置，提高平移不变性
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

            # --- 3. 目标信息（20维）---
            # 相对位置(4×3=12) + 捕获状态 bool→float(4) + 目标价值(4)
            target_rel_pos = self.target_positions - self.drone_positions[
                :, drone_idx
            ].unsqueeze(1)   # (N, T, 3)

            obs_targets = torch.cat(
                [
                    target_rel_pos.view(self.num_envs, -1),         # 12
                    self.target_captured.float().view(self.num_envs, -1),  # 4
                    self.target_values,                               # 4
                ],
                dim=-1,
            )

            # --- 4. 距离信息（4维）+ 最近标志（4维）---
            # obs_distances：该无人机到各目标的标量距离
            # is_closest：该无人机是否是对应目标的最近无人机（协作任务分工信号）
            obs_distances = drone_to_target_distances[:, drone_idx]  # 4
            is_closest = (closest_drone_to_target == drone_idx).float()  # 4

            # --- 拼接完整观测（49维）---
            obs_t = torch.cat(
                [
                    obs_self,         # 15
                    obs_other_drones, # 6
                    obs_targets,      # 20
                    obs_distances,    # 4
                    is_closest,       # 4
                ],
                dim=-1,
            )

            # --- 历史拼接（partial_obs=True 时启用）---
            # CircularBuffer.append → buffer.reshape 得到 (N, history_len × obs_dim)
            if self.cfg.partial_obs:
                self._observation_buffers[agent_name].append(obs_t)
                observations[agent_name] = self._observation_buffers[
                    agent_name
                ].buffer.reshape(self.num_envs, -1)
            else:
                observations[agent_name] = obs_t

        return observations

    def _get_states(self) -> torch.Tensor:
        """构建 Critic 使用的全局状态（CTDE：集中训练时 Critic 可观测全部信息）。

        全局状态（86维）：
          - 无人机绝对位置（9）：所有无人机的 xyz 坐标
          - 无人机旋转矩阵（27）：所有无人机的 3×3 旋转矩阵展平
          - 无人机线速度（9）：所有无人机的 vx, vy, vz
          - 无人机角速度（9）：所有无人机的 ωx, ωy, ωz
          - 目标绝对位置（12）：4个目标的 xyz 坐标
          - 目标速度（12）：4个目标的速度向量（x 方向匀速）
          - 目标捕获状态（4）：bool → float
          - 目标价值（4）：固定权重

        注：Critic 用全局状态，可准确评估多智能体协作效果；
            Actor 仅用局部观测，实现分散执行。
        """
        states = torch.cat(
            (
                self.drone_positions.view(self.num_envs, -1),           # 9
                self.drone_rot_matrices.view(self.num_envs, -1),         # 27
                self.drone_linear_velocities.view(self.num_envs, -1),    # 9
                self.drone_angular_velocities.view(self.num_envs, -1),   # 9
                self.target_positions.view(self.num_envs, -1),           # 12
                self.target_velocities.view(self.num_envs, -1),          # 12
                self.target_captured.float().view(self.num_envs, -1),    # 4
                self.target_values,                                       # 4
            ),
            dim=-1,
        )
        return states

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        """计算每步奖励（指数衰减风格，与 hover/hover_flycart 统一）。

        设计原则：
        - 所有正奖励乘以 step_dt，使其为每秒奖励，与 episode 时长解耦
        - 指数衰减函数 exp(-x) 在 x→0 时最大化奖励，避免稀疏性
        - 共享奖励：所有智能体共享同一个 total_reward，促进协作
        - 安全惩罚：不乘 step_dt，避免违规持续累积

        奖励分量总览：
          正奖励：距离(1.5) + 跟随(1.0) + 动作平滑(1.0) + 角速率(2.0)
                  + 速度(0.3) + 推力(0.5) + 高度(2.0) + 竖直姿态(2.0)
          负奖励：碰撞 + 出界 + 飞低 + 非法接触 + 高度超限
        """
        rewards = {}
        step_dt = self.step_dt

        # --- 1. 距离奖励（目标中心制，按价值加权）---
        # 目标中心制：对每个目标，找距离最近的无人机；鼓励无人机覆盖所有目标
        d_pos = self.drone_positions[:, :, :2].unsqueeze(2)  # (N,D,1,2)
        t_pos = self.target_positions[:, :, :2].unsqueeze(1)  # (N,1,T,2)
        dist_matrix = torch.norm(d_pos - t_pos, dim=-1)  # (N, D, T)
        # closest_drone_indices: 各目标最近的无人机索引，shape (N, T)
        min_dists, closest_drone_indices = torch.min(dist_matrix, dim=1)  # (N, T)

        # exp(-dist × scale) × value：距离越近奖励越大；高价值目标奖励更多
        dist_per_target = torch.exp(-min_dists * self.cfg.dist_reward_scale)  # (N, T)
        dist_reward = torch.sum(dist_per_target * self.target_values, dim=-1)
        rewards["distance_reward"] = self.cfg.dist_reward_weight * dist_reward * step_dt

        # --- 更新实时捕获状态（可撤销，每帧刷新）---
        # is_captured_now: 各目标是否当前被某架无人机覆盖（距离 < capture_distance）
        is_captured_now = min_dists < self.cfg.capture_distance
        self.target_captured = is_captured_now  # 注意：此处会覆盖上步状态
        # target_captured_by：只增不减，记录最后一次捕获该目标的无人机（用于边界豁免）
        self.target_captured_by = torch.where(
            is_captured_now,
            closest_drone_indices,
            self.target_captured_by,
        )

        # --- 持续跟随计时（成功终止条件）---
        # 要求至少3个目标同时被跟随（鼓励协作覆盖）
        target_min_dist = dist_matrix.min(dim=1)[0]  # (N, T)：各目标到最近无人机的距离
        target_followed = target_min_dist < self.cfg.capture_distance  # (N, T)
        num_targets_followed = target_followed.sum(dim=-1)  # (N,)
        enough_targets_followed = num_targets_followed >= 3
        # 计时器"暂停"策略：条件不满足时维持现值（不重置），避免短暂失跟后从零计时
        self._sustained_follow_timer = torch.where(
            enough_targets_followed,
            self._sustained_follow_timer + step_dt,
            self._sustained_follow_timer,
        )
        self.all_targets_captured = (
            self._sustained_follow_timer >= self.cfg.sustained_follow_duration
        )

        # --- 跟随奖励（仅在捕获状态下给予，防止奖励倒置）---
        # 修复说明：旧逻辑 exp(-dist) 在无目标时 dist=0，反而给出最高奖励
        # 新逻辑：is_captured × exp(-dist)，未捕获时贡献为 0
        tracking_reward = (
            is_captured_now.float()
            * torch.exp(-min_dists * self.cfg.tracking_reward_scale)
        ).sum(dim=-1)
        rewards["tracking_reward"] = (
            self.cfg.tracking_reward_weight * tracking_reward * step_dt
        )

        # --- 2. 动作平滑度奖励（抑制震荡动作）---
        # 计算相邻两步动作差值的范数，差值越小奖励越高
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

        # --- 3. 机体角速率惩罚（抑制激烈机动，防止翻滚）---
        # 取所有无人机角速率指令的平均范数
        commanded_body_rates = torch.cat(
            [self.actions[a][:, 3:] for a in self.cfg.possible_agents],
            dim=-1,
        )
        body_rate_norm = torch.norm(commanded_body_rates / self._num_drones, dim=-1)
        rewards["body_rate_penalty"] = (
            self.cfg.body_rate_penalty_weight * torch.exp(-body_rate_norm) * step_dt
        )

        # --- 4. 线速度惩罚（抑制高速飞行，提高跟随精度）---
        vel_norms = torch.norm(self.drone_linear_velocities, dim=-1)  # (N, D)
        avg_vel = vel_norms.mean(dim=-1)  # (N,)：所有无人机平均速度
        rewards["velocity_penalty"] = (
            self.cfg.velocity_penalty_weight * torch.exp(-avg_vel) * step_dt
        )

        # --- 5. 推力惩罚（抑制高推力悬停，节省能量）---
        # 取所有旋翼中最大推力（归一化），exp(-max_effort) 鼓励低能耗飞行
        all_forces = torch.stack([f[..., 2] for f in self._forces], dim=1)  # (N, D, 4)
        normalized_forces = all_forces / self.cfg.max_thrust_pp
        effort_max = torch.max(normalized_forces.view(self.num_envs, -1), dim=-1)[0]
        rewards["force_penalty"] = (
            self.cfg.force_penalty_weight * torch.exp(-effort_max) * step_dt
        )

        # --- 6. 高度奖励（鼓励维持期望高度 desired_height）---
        height_error = torch.norm(
            self.drone_positions[..., 2] - self.cfg.desired_height,
            dim=-1,
        )
        rewards["height_reward"] = (
            self.cfg.height_reward_weight * torch.exp(-height_error) * step_dt
        )

        # 高度超限惩罚（线性）：高度偏差超过 threshold 后额外惩罚
        excess_height = (height_error - self.cfg.height_penalty_threshold).clamp(
            min=0.0
        )
        rewards["height_penalty"] = (
            -self.cfg.height_penalty_weight * excess_height * step_dt
        )

        # --- 6.5 竖直姿态惩罚（防止翻滚/大倾角）---
        # R[i, :, 2, 2] = 机体 z 轴与世界 z 轴的点积：完全竖直时为 1，翻转时为 -1
        # (z - 1.0) ∈ [-2, 0]，乘以正权重后为负奖励，惩罚倾斜
        z_axis_body = self.drone_rot_matrices[:, :, 2, 2]  # (N, D)
        rewards["upright_penalty"] = (
            self.cfg.upright_penalty_weight * (z_axis_body - 1.0).sum(-1) * step_dt
        )

        # ====== 安全惩罚（不乘 step_dt，每步固定触发）======

        # --- 7. 无人机碰撞惩罚 ---
        # cdist 计算两两距离，对角线置 inf 排除自身
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
        )  # 除以 2 避免双重计数
        rewards["collision_penalty"] = (
            -num_collisions * self.cfg.collision_penalty_scale
        )

        # --- 8. 无人机出界惩罚 ---
        out_of_bounds = (
            (self.drone_positions.abs() > self.cfg.bounding_box_threshold)
            .any(dim=-1)
            .any(dim=-1)
        )
        rewards["drone_out"] = (
            -out_of_bounds.float() * self.cfg.drone_out_of_bounds_penalty
        )

        # --- 9. 飞行高度过低惩罚（z < 0.1m）---
        fly_low = (self.drone_positions[:, :, 2] < 0.1).any(dim=-1)
        rewards["fly_low"] = -fly_low.float() * self.cfg.fly_low_penalty

        # --- 10. 非法接触惩罚（接触传感器读取）---
        # 遍历所有无人机的接触传感器，检测是否有接触力超过阈值
        illegal_any = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for cs in self.contact_sensors:
            net_f = cs.data.net_forces_w_history  # (N, history, bodies, 3)
            max_f = torch.max(torch.norm(net_f, dim=-1), dim=1)[0]
            has_contact = (max_f > self.cfg.contact_sensor_threshold).any(dim=1)
            illegal_any = illegal_any | has_contact
        rewards["illegal_contact"] = (
            -illegal_any.float() * self.cfg.illegal_contact_penalty
        )

        # --- 总奖励（共享：所有智能体收到相同奖励，促进协作）---
        total_reward = sum(rewards.values())

        # 累计到 episode sums（用于 tensorboard 记录每 episode 奖励分量均值）
        for key, val in rewards.items():
            if key not in self._episode_sums:
                self._episode_sums[key] = torch.zeros_like(val)
            self._episode_sums[key] += val

        return {agent: total_reward for agent in self.cfg.possible_agents}

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """检测终止条件，返回 (terminated, time_outs) 两个字典。

        终止条件（terminated = 提前结束，不做 bootstrap）：
          1. falcon_fly_low：任一无人机 z < 0.1m（接地/坠落）
          2. illegal_contact：任一无人机接触传感器超过阈值（碰到地面/障碍物）
          3. drone_collision：任意两架无人机距离 < collision_threshold
          4. body_pos_outside：无人机超出边界（但捕获目标的无人机豁免）
          5. targets_out_of_bounds：目标物块超出任意边界（x/y/z）

        time_outs（时间到，做 bootstrap）：
          - episode_length_buf >= max_episode_length - 1

        注意：all_targets_captured（成功终止）在 _get_rewards 中更新，
        但此处不作为 terminated 返回，以避免影响 GAE/bootstrap 计算。
        """
        # --- 条件1：飞行高度过低（z < 0.1m）---
        self.falcon_fly_low = (self.drone_positions[:, :, 2] < 0.1).any(dim=-1)

        # --- 条件2：非法接触（逐个接触传感器检查）---
        self.illegal_contact = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        for contact_sensor in self.contact_sensors:
            net_forces = contact_sensor.data.net_forces_w_history
            max_force = torch.max(torch.norm(net_forces, dim=-1), dim=1)[0]
            has_contact = (max_force > self.cfg.contact_sensor_threshold).any(dim=1)
            self.illegal_contact = self.illegal_contact | has_contact

        # --- 条件3：无人机互相碰撞 ---
        # 计算两两间最小距离，比 cdist 更高效（使用 get_drone_rpos/pdist 工具函数）
        rpos = get_drone_rpos(self.drone_positions)
        pdist = get_drone_pdist(rpos)
        separation = pdist.min(dim=-1).values.min(dim=-1).values
        self.drone_collision = separation < self.cfg.drone_collision_threshold

        # --- 条件4：无人机出边界（含豁免逻辑）---
        is_outside = (self.drone_positions.abs() > self.cfg.bounding_box_threshold).any(
            dim=-1
        )  # (N, D)

        # 豁免：正在追捕目标（has_captured & is_captured_now）的无人机出界不终止
        # 场景：无人机追出边界追目标，不应被惩罚
        drone_has_captured = torch.zeros(
            (self.num_envs, self._num_drones), dtype=torch.bool, device=self.device
        )
        for d in range(self._num_drones):
            # 该无人机曾捕获过的目标中，当前仍处于捕获状态的任一目标
            drone_has_captured[:, d] = (
                (self.target_captured_by == d) & self.target_captured
            ).any(dim=-1)

        # 只有"出界 AND 未捕获目标"的无人机才触发终止
        self.body_pos_outside = (is_outside & (~drone_has_captured)).any(dim=-1)

        # --- 条件5：目标物块超出边界（x 方向运动最终会超出）---
        # 当所有目标都跑出右边界时，episode 自然结束
        self.targets_out_of_bounds = (
            (self.target_positions[:, :, 0] > self.cfg.bounding_box_threshold)
            | (self.target_positions[:, :, 0] < -self.cfg.bounding_box_threshold)
            | (self.target_positions[:, :, 1] > self.cfg.bounding_box_threshold)
            | (self.target_positions[:, :, 1] < -self.cfg.bounding_box_threshold)
            | (self.target_positions[:, :, 2] < 0.05)
        ).any(dim=-1)

        # --- 组合所有终止条件（任一触发即终止）---
        terminations = (
            self.falcon_fly_low
            | self.illegal_contact
            | self.drone_collision
            | self.body_pos_outside
            | self.targets_out_of_bounds
        )

        # --- Debug 打印（仅输出前5个重置 env 的原因，避免日志爆炸）---
        reset_indices = torch.nonzero(terminations).flatten()
        if len(reset_indices) > 0:
            for i in reset_indices[:5]:
                idx = i.item()
                reasons = []
                if self.falcon_fly_low[idx]:
                    reasons.append(
                        f"Fly Low (z={self.drone_positions[idx, :, 2].min():.2f})"
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
                    current_captures = self.target_captured[idx]
                    if current_captures.any():
                        current_captured_by = self.target_captured_by[idx][
                            current_captures
                        ]
                        unique_drones = torch.unique(current_captured_by)
                        unique_drones = unique_drones[unique_drones != -1]
                        if len(unique_drones) >= 3:
                            reasons.append("All Captured")
                if reasons:
                    print(f"[Reset] Env {idx}: {', '.join(reasons)}")

        # --- 超时终止（做 GAE bootstrap）---
        self.time_out = self.episode_length_buf >= self.max_episode_length - 1
        timed_outs = self.time_out
        # 所有智能体共享同一终止信号（协作任务）
        terminated = {agent: terminations for agent in self.cfg.possible_agents}
        time_outs = {agent: timed_outs for agent in self.cfg.possible_agents}
        return terminated, time_outs

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None):
        """重置指定环境的所有状态（在 episode 终止后由基类调用）。

        重置流程：
          1. 调用基类 _reset_idx（处理时间步计数等）
          2. 重置目标物块位置（_reset_targets）
          3. 清空 PD 控制器历史误差（防止重置时微分项爆炸）
          4. 清空持续跟随计时器
          5. 随机化无人机初始位置（三角编队，加随机偏移）
          6. 重置历史观测缓冲区（防止新 episode 混入旧帧）
          7. 记录本 episode 的奖励和终止统计到 extras["log"]
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        super()._reset_idx(env_ids)
        self._reset_targets(env_ids)
        # 清空速度误差历史，防止重置后微分项残留导致初始动作异常
        self._prev_vel_error[env_ids] = 0.0
        # 清空持续跟随计时器（新 episode 从 0 开始）
        self._sustained_follow_timer[env_ids] = 0.0

        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device)

        # ===== 随机化无人机初始位置（三角编队）=====
        # 随机采样编队中心（x: drone_spawn_x_range, y: drone_spawn_y_range）
        center_x = torch.empty(len(env_ids), device=self.device).uniform_(
            *self.cfg.drone_spawn_x_range
        )
        center_y = torch.empty(len(env_ids), device=self.device).uniform_(
            *self.cfg.drone_spawn_y_range
        )

        radius = 2.0   # 三角编队半径（m）
        height = 2.5   # 初始高度（m）
        # 三角编队相位：0°, 120°, 240°
        phases = torch.tensor(
            [0.0, 2.0 * 3.14159 / 3.0, 4.0 * 3.14159 / 3.0], device=self.device
        )

        for i, robot in enumerate(self.robots):
            robot.reset(env_ids)  # 重置关节等内部状态

            origins = self.scene.env_origins[env_ids]  # 各 env 的世界坐标原点

            # 计算初始位置偏移（三角形顶点 + 中心偏移）
            offsets = torch.zeros((len(env_ids), 3), device=self.device)
            offsets[:, 0] = center_x + radius * torch.cos(phases[i])
            offsets[:, 1] = center_y + radius * torch.sin(phases[i])
            offsets[:, 2] = height

            new_positions = origins + offsets

            default_root_state = robot.data.default_root_state[env_ids]
            new_quats = default_root_state[:, 3:7]  # 保持默认朝向
            new_poses = torch.cat([new_positions, new_quats], dim=-1)

            robot.write_root_pose_to_sim(new_poses, env_ids=env_ids)
            # 初始速度归零（防止继承上一 episode 的速度）
            robot.write_root_velocity_to_sim(
                torch.zeros_like(default_root_state[:, 7:]), env_ids=env_ids
            )

        # 清空历史观测缓冲区（防止新 episode 混入旧 episode 帧）
        for agent in self.cfg.possible_agents:
            self._observation_buffers[agent].reset(env_ids)

        # ===== 记录 episode 统计信息到 extras["log"]（供 tensorboard 读取）=====
        if "log" not in self.extras:
            self.extras["log"] = dict()
        if "crash" not in self.extras["log"]:
            self.extras["log"]["Episode_Termination/crash"] = 0
        if "out_of_bounds" not in self.extras["log"]:
            self.extras["log"]["Episode_Termination/out_of_bounds"] = 0

        # 各终止原因的计数（在重置的 env 中统计）
        self.extras["log"]["Episode_Termination/drones_collide"] = torch.count_nonzero(
            self.drone_collision[env_ids]
        ).item()
        self.extras["log"]["Episode_Termination/bounding_box"] = torch.count_nonzero(
            self.body_pos_outside[env_ids]
        ).item()
        self.extras["log"]["Episode_Termination/falcon_fly_low"] = torch.count_nonzero(
            self.falcon_fly_low[env_ids]
        ).item()
        self.extras["log"]["Episode_Termination/crash"] = torch.count_nonzero(
            self.falcon_fly_low[env_ids] | self.illegal_contact[env_ids]
        ).item()
        self.extras["log"]["Episode_Termination/out_of_bounds"] = torch.count_nonzero(
            self.targets_out_of_bounds[env_ids]
        ).item()

        # 各奖励分量在本 episode 的平均值（用于 tensorboard 曲线监控）
        for key, value in self._episode_sums.items():
            if len(env_ids) > 0:
                mean_val = value[env_ids].mean().item()
                self.extras["log"][f"Episode_Reward/{key}"] = mean_val
            value[env_ids] = 0.0  # 清零，开始新 episode 统计

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

        # 清空动作历史（防止上一 episode 的动作影响平滑度计算）
        for agent in self.cfg.possible_agents:
            self.prev_actions[agent][env_ids] = 0.0
            self.actions[agent][env_ids] = 0.0

        # 重置成功标志
        self.all_targets_captured[env_ids] = False

    def _reset_targets(self, env_ids):
        """重置目标物块的位置、速度和捕获状态。

        目标初始化策略：
        - x 坐标：在 target_spawn_x_range 内随机采样（制造追捕任务的初始距离差异）
        - y 坐标：固定在 target_spawn_y_positions（4个目标等间距排列，初始分散）
        - z 坐标：固定在 target_spawn_z（地面高度，目标在地面滑动）
        - 速度：初始为 0（由 _apply_action 在第一步赋予 x 方向速度）
        """
        # 计算展平索引：(num_env_reset × num_targets,)
        target_indices_flat = (
            env_ids.view(-1, 1) * self.num_targets
            + torch.arange(self.num_targets, device=self.device).view(1, -1)
        ).flatten()

        flat_pos = self.target_positions.view(-1, 3)
        flat_vel = self.target_velocities.view(-1, 3)

        # 重置捕获状态（新 episode 无人机尚未接近任何目标）
        self.target_captured[env_ids] = False
        self.target_captured_by[env_ids] = -1  # -1 表示未被任何无人机捕获

        # 逐目标随机化 x 坐标，y/z 固定
        for i in range(self.num_targets):
            current_indices = env_ids * self.num_targets + i

            r = torch.empty(len(env_ids), device=self.device)
            val_x = r.uniform_(*self.cfg.target_spawn_x_range)  # x: 随机
            val_y = self.cfg.target_spawn_y_positions[i]         # y: 固定分层
            val_z = self.cfg.target_spawn_z                      # z: 地面高度

            flat_pos[current_indices, 0] = val_x
            flat_pos[current_indices, 1] = val_y
            flat_pos[current_indices, 2] = val_z

        # 初始速度归零（第一个物理步骤后会被赋予 target_velocity）
        flat_vel[target_indices_flat] = 0.0

    def _set_debug_vis_impl(self, debug_vis: bool):
        """设置 debug 可视化（当前未实现，预留接口）。"""
        pass

    def _debug_vis_callback(self, event):
        """debug 可视化回调（当前未实现，预留接口）。"""
        pass


@torch.jit.script
def scale(x, lower, upper):
    """将 [-1, 1] 的归一化值还原到 [lower, upper]。"""
    return 0.5 * (x + 1.0) * (upper - lower) + lower


@torch.jit.script
def unscale(x, lower, upper):
    """将 [lower, upper] 的值归一化到 [-1, 1]。"""
    return (2.0 * x - upper - lower) / (upper - lower)
