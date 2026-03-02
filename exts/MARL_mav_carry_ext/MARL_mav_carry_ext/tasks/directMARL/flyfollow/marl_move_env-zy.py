# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
此版本为周元在2026年2月24日提供的版本。

多智能体移动环境 - 三架无人机协同追踪移动目标

该环境实现了三架无人机同时追踪四个移动目标的任务。每架无人机需要：
1. 保持合理的位置和姿态
2. 追踪分配给自己的移动目标
3. 避免与其他无人机碰撞
4. 在限定区域内完成任务

主要特点：
- 支持几何控制器和ACCBR控制器两种控制模式
- 实现了实时的目标捕获检测和撤销机制
- 包含丰富的奖励函数设计
- 支持部分观测和全局状态两种观测模式
"""

from __future__ import annotations

import torch
from collections.abc import Sequence

# 导入自定义控制器模块
from MARL_mav_carry_ext.controllers import GeometricController, IndiController
from MARL_mav_carry_ext.controllers.motor_model import RotorMotor
# 导入辅助函数
from MARL_mav_carry_ext.tasks.managerbased.mdp_llc.utils import (
    get_drone_pdist,
    get_drone_rpos,
)

# 导入Isaac Lab相关模块
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectMARLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import CircularBuffer
from isaaclab.utils.math import (
    euler_xyz_from_quat,
    matrix_from_quat,
    quat_rotate,
)

# 导入环境配置
from .marl_move_env_cfg import MARLMoveEnvCfg


class MARLMoveEnv(DirectMARLEnv):
    """
    多智能体移动追踪环境类
    
    继承自DirectMARLEnv，实现了三架无人机协同追踪移动目标的核心逻辑。
    每个无人机作为一个独立的智能体，有自己的观测空间和动作空间。
    """
    
    cfg: MARLMoveEnvCfg  # 环境配置对象

    def __init__(self, cfg: MARLMoveEnvCfg, render_mode: str | None = None, **kwargs):
        """
        初始化环境
        
        Args:
            cfg: 环境配置对象
            render_mode: 渲染模式
            **kwargs: 其他参数
        """
        super().__init__(cfg, render_mode, **kwargs)

        # 无人机数量固定为3架
        self._num_drones = 3
        # 控制模式：'geometric' 或 'ACCBR'
        self._control_mode = cfg.control_mode

        # 获取无人机机体和旋翼的body索引（查询robot_0即可，因为结构相同）
        self._falcon_idx = torch.tensor(
            self.robots[0].find_bodies(".*base_link")[0], device=self.device
        )
        self._falcon_rotor_idx = torch.tensor(
            self.robots[0].find_bodies(".*rotor_.*")[0], device=self.device
        )

        # 观测缓冲区 - 用于存储历史观测值（支持部分观测模式）
        self._observation_buffers = {}
        for agent in self.cfg.possible_agents:
            self._observation_buffers[agent] = CircularBuffer(
                cfg.history_len, self.num_envs, device=self.device
            )

        # 动作缓冲区 - 存储每架无人机的推力和力矩
        # _forces: (num_envs, 4, 3) 每架无人机有4个旋翼，每个旋翼产生3维推力
        self._forces = [
            torch.zeros(self.num_envs, 4, 3, device=self.device)
            for _ in range(self._num_drones)
        ]
        # _moments: (num_envs, 3, 3) 三架无人机的三轴力矩
        self._moments = torch.zeros(
            self.num_envs, self._num_drones, 3, device=self.device
        )

        # 控制设定点和上一时刻动作存储
        self._setpoints = {}
        self.prev_actions = {}
        for agent in self.cfg.possible_agents:
            self._setpoints[agent] = {}
            # 根据控制模式初始化不同的动作维度
            if self._control_mode == "geometric":
                self.prev_actions[agent] = torch.zeros(
                    self.num_envs, 12, device=self.device
                )
            elif self._control_mode == "ACCBR":
                self.prev_actions[agent] = torch.zeros(
                    self.num_envs, 6, device=self.device
                )

        # 无人机状态变量初始化
        self.drone_positions = torch.zeros(self.num_envs, self._num_drones, 3, device=self.device)
        self.drone_orientations = torch.zeros(self.num_envs, self._num_drones, 4, device=self.device)
        self.drone_orientations[..., 0] = 1.0  # 初始化为单位四元数
        self.drone_rot_matrices = torch.zeros(self.num_envs, self._num_drones, 3, 3, device=self.device)
        self.drone_linear_velocities = torch.zeros(self.num_envs, self._num_drones, 3, device=self.device)
        self.drone_angular_velocities = torch.zeros(self.num_envs, self._num_drones, 3, device=self.device)
        self.drone_linear_accelerations = torch.zeros(self.num_envs, self._num_drones, 3, device=self.device)
        self.drone_angular_accelerations = torch.zeros(self.num_envs, self._num_drones, 3, device=self.device)
        self._drone_jerk = torch.zeros(self.num_envs, self._num_drones, 3, device=self.device)
        self._drone_prev_acc = torch.zeros(self.num_envs, self._num_drones, 3, device=self.device)
        # PD速度控制器：上一时刻的速度误差
        self._prev_vel_error = torch.zeros(self.num_envs, self._num_drones, 3, device=self.device)

        # 外环控制器（几何控制器）
        self.geo_controllers = {}
        for i in range(self._num_drones):
            self.geo_controllers[i] = GeometricController(self.num_envs, self._control_mode)
        self._ll_counter = 0                                                      # 低层控制计数器
        self._constant_yaw = torch.zeros([self.num_envs, 1], device=self.device)  # 恒定偏航角
        self._zeros = torch.zeros([self.num_envs, 3], device=self.device)         # 零向量

        # 内环控制器（INDI增量式控制器）
        self._indi_controllers = {}
        for i in range(self._num_drones):
            self._indi_controllers[i] = IndiController(self.num_envs)

        # 电机模型
        self.motor_models = {}
        # 初始化转速：每架无人机4个旋翼，初始转速1355 RPM
        initial_rpms = [
            torch.tensor([[1355.0, 1355.0, 1355.0, 1355.0]], device=self.device).repeat(
                self.num_envs, 1
            )
            for _ in range(self._num_drones)
        ]
        for i in range(self._num_drones):
            self.motor_models[i] = RotorMotor(self.num_envs, initial_rpms[i])
        # 采样时间 = 物理时间步长 × 低层控制降频因子
        self.sampling_time = self.sim.get_physics_dt() * self.cfg.low_level_decimation

        # 移动物块（目标）状态缓冲区
        self.num_targets = cfg.num_targets  # 目标数量（通常为4个）
        self.target_positions = torch.zeros(self.num_envs, self.num_targets, 3, device=self.device)
        self.target_velocities = torch.zeros(self.num_envs, self.num_targets, 3, device=self.device)
        # 目标捕获状态：True表示被某架无人机捕获
        self.target_captured = torch.zeros(self.num_envs, self.num_targets, dtype=torch.bool, device=self.device)
        # 预创建四元数张量，避免每帧重复创建
        self._target_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device).repeat(self.num_envs, 1)
        # 目标价值权重
        self.target_values = (
            torch.tensor(cfg.target_values, device=self.device)
            .unsqueeze(0)
            .repeat(self.num_envs, 1)
        )

        # 记录哪个无人机捕获了哪个物块 (-1 表示未被捕获)
        self.target_captured_by = torch.full((self.num_envs, self.num_targets), -1, dtype=torch.long, device=self.device)

        # 奖励日志记录（动态补充，此处仅预创建）
        self._episode_sums = {
            key: torch.zeros(
                self.num_envs,
                dtype=torch.float,
                device=self.device,
            )
            for key in [
                "distance_reward",      # 距离奖励
                "success_reward",       # 成功奖励
                "tracking_reward",      # 追踪奖励
                "action_smoothness",    # 动作平滑性奖励
                "body_rate_penalty",    # 机体角速率惩罚
                "velocity_penalty",     # 速度惩罚
                "force_penalty",        # 推力惩罚
                "height_reward",        # 高度奖励
                "collision_penalty",    # 碰撞惩罚
                "drone_out",           # 无人机越界惩罚
                "fly_low",             # 飞行过低惩罚
                "illegal_contact",     # 非法接触惩罚
                "time_penalty",        # 时间惩罚
                "upright_penalty",     # 姿态 upright 惩罚
                "height_penalty",      # 高度偏差惩罚
            ]
        }

        # 性能指标
        self.metrics = {}
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["orientation_error"] = torch.zeros(
            self.num_envs, device=self.device
        )

        # 终止条件缓冲区
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
        # 所有目标都被捕获的标志
        self.all_targets_captured = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        # 持续跟随计时器
        self._sustained_follow_timer = torch.zeros(self.num_envs, device=self.device)
        self.targets_out_of_bounds = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.time_out = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        # 归一化配置参数
        self._norm_pos_scale = self.cfg.bounding_box_threshold
        self._norm_vel_scale = 5.0
        self._norm_dist_scale = self.cfg.bounding_box_threshold * 2

        # 调试可视化设置
        self.set_debug_vis(cfg.debug_vis)

    def _setup_scene(self):
        """
        设置场景：创建3个独立的无人机Articulation和4个移动目标
        
        采用IsaacLab-HARL模式，每个无人机都是独立的Articulation对象，
        避免了传统方法中需要复杂索引映射的问题。
        """
        
        # ===== 创建3个独立的Articulation对象 =====
        self.robots = []
        self.contact_sensors = []

        for i in range(3):
            robot_cfg = getattr(self.cfg, f"robot_{i}")

            # 禁用载荷/绳索（如果USD文件中存在）
            # spawn函数将通过Articulation构造函数调用场景复制
            robot = Articulation(robot_cfg)
            self.robots.append(robot)
            self.scene.articulations[f"robot_{i}"] = robot

            contact_cfg = getattr(self.cfg, f"contact_forces_{i}")
            contact = ContactSensor(contact_cfg)
            self.contact_sensors.append(contact)
            self.scene.sensors[f"contact_forces_{i}"] = contact

        # ===== 创建4个移动物块（目标） =====
        from isaaclab.assets import RigidObjectCfg

        self.targets = []
        for i, color_name in enumerate(self.cfg.target_colors):
            color_rgb = self.cfg.target_color_rgb[color_name]
            target_cfg = RigidObjectCfg(
                prim_path=f"/World/envs/env_.*/target_{i}",
                spawn=sim_utils.CuboidCfg(
                    size=self.cfg.target_size,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        kinematic_enabled=True,    # 启用运动学控制
                        disable_gravity=True,      # 禁用重力
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=color_rgb
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

        # 添加地面平面
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # 克隆和复制环境
        self.scene.clone_environments(copy_from_source=False)
        # 添加灯光
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        """
        物理步前处理：解析并应用智能体动作
        
        Args:
            actions: 智能体动作字典，键为agent名称，值为动作张量
        """
        # 保存上一时刻动作并更新当前动作
        for agent in self.cfg.possible_agents:
            self.prev_actions[agent][:] = self.actions[agent]
            self.actions[agent][:] = actions[agent]

        # 解析每个无人机的动作
        for drone, action in actions.items():
            if self._control_mode == "geometric":
                # 几何控制模式：直接指定位置、速度、加速度、加加速度
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
                # ACCBR控制模式：速度指令 + PD控制器
                drone_idx = self.cfg.possible_agents.index(drone)
                # 计算期望速度（归一化到最大线速度）
                desired_vel = action[:, :3] * self.cfg.lin_vel_max
                current_vel = self.drone_linear_velocities[:, drone_idx]
                # 计算速度误差及其导数
                vel_error = desired_vel - current_vel
                d_error = (
                    vel_error - self._prev_vel_error[:, drone_idx]
                ) / self.step_dt
                self._prev_vel_error[:, drone_idx] = vel_error
                # PD控制器计算指令加速度
                commanded_acc = self.cfg.vel_Kp * vel_error + self.cfg.vel_Kd * d_error
                commanded_acc = torch.clamp(
                    commanded_acc,
                    -self.cfg.lin_acc_max,
                    self.cfg.lin_acc_max,
                )
                self._setpoints[drone]["lin_acc"] = commanded_acc
                # 角速度指令
                self._setpoints[drone]["body_rates"] = (
                    action[:, 3:6] * self.cfg.ang_vel_max
                )

            # 偏航角相关设定点（目前保持恒定）
            self._setpoints[drone]["yaw"] = self._constant_yaw
            self._setpoints[drone]["yaw_rate"] = (
                action[:, 5].unsqueeze(-1) * self.cfg.ang_vel_max
                if self._control_mode == "ACCBR"
                else self._constant_yaw
            )
            self._setpoints[drone]["yaw_acc"] = self._constant_yaw

    def _apply_action(self) -> None:
        """
        应用控制动作：计算并施加推力和力矩到无人机
        
        实现了完整的控制链路：
        1. 读取无人机当前状态
        2. 外环几何控制器计算期望力和力矩
        3. 内环INDI控制器计算目标转速
        4. 电机模型计算实际推力和力矩
        5. 施加到物理引擎
        """
        if self._ll_counter % self.cfg.low_level_decimation == 0:
            all_thrusts = []
            all_moments = []

            # ===== 从每个独立 robot 读取状态 (无需索引映射!) =====
            for i, robot in enumerate(self.robots):
                # root_state_w shape: (num_envs, 13) — 每个robot独立!
                root_state = robot.data.root_state_w  # (N, 13)

                # 提取位置、姿态、线速度、角速度（转换到局部坐标系）
                drone_pos = root_state[:, :3] - self.scene.env_origins  # 局部坐标
                drone_quat = root_state[:, 3:7]
                drone_lin_vel = root_state[:, 7:10]
                drone_ang_vel = root_state[:, 10:13]

                # 提取机体加速度（body_acc_w: (N, num_bodies, 6)）
                # 对于独立Falcon, base_link idx 通常是 0
                body_acc = robot.data.body_acc_w
                drone_lin_acc = body_acc[:, 0, 0:3]
                drone_ang_acc = body_acc[:, 0, 3:6]

                # 更新状态变量
                self.drone_positions[:, i] = drone_pos
                self.drone_orientations[:, i] = drone_quat
                self.drone_linear_velocities[:, i] = drone_lin_vel
                self.drone_angular_velocities[:, i] = drone_ang_vel
                self.drone_linear_accelerations[:, i] = drone_lin_acc
                self.drone_angular_accelerations[:, i] = drone_ang_acc

            # ===== 计算每架无人机的控制力 =====
            for i in range(self._num_drones):
                # 构造无人机状态字典
                drone_states: dict = {}
                drone_states["pos"] = self.drone_positions[:, i]
                drone_states["quat"] = self.drone_orientations[:, i]
                drone_states["lin_vel"] = self.drone_linear_velocities[:, i]
                drone_states["ang_vel"] = self.drone_angular_velocities[:, i]
                drone_states["lin_acc"] = self.drone_linear_accelerations[:, i]
                drone_states["ang_acc"] = self.drone_angular_accelerations[:, i]
                
                # 计算加加速度（jerk）
                self._drone_jerk[:, i] = (
                    drone_states["lin_acc"] - self._drone_prev_acc[:, i]
                ) / (self.physics_dt)
                drone_states["jerk"] = self._drone_jerk[:, i]
                self._drone_prev_acc[:, i] = drone_states["lin_acc"]

                agent_name = self.cfg.possible_agents[i]

                # 外环几何控制器计算指令
                alpha_cmd, acc_load, acc_cmd, q_cmd = self.geo_controllers[
                    i
                ].getCommand(
                    drone_states,
                    self._forces[i],  # (N, 4, 3)
                    self._setpoints[agent_name],
                )

                # 内环INDI控制器计算目标转速
                target_rpm = self._indi_controllers[i].getCommand(
                    drone_states,
                    self._forces[i],
                    alpha_cmd,
                    acc_cmd,
                    acc_load,
                )

                # 电机模型计算推力和力矩
                thrusts, moments = self.motor_models[i].get_motor_thrusts_moments(
                    target_rpm, self.sampling_time
                )
                all_thrusts.append(thrusts)
                all_moments.append(moments)

            # ===== 限制推力和力矩 =====
            for i in range(self._num_drones):
                # 推力限制：最小0，最大max_thrust_pp
                forces_i = torch.clamp(
                    all_thrusts[i], min=0.0, max=self.cfg.max_thrust_pp
                )
                self._forces[i][..., 2] = forces_i

                # 力矩限制：±1.0
                moments_i = torch.clamp(all_moments[i], min=-1.0, max=1.0)
                self._moments[:, i, 2] = moments_i.sum(-1)  # Z轴力矩求和

            self._ll_counter = 0
        self._ll_counter += 1

        # ===== 对每个robot独立施加力/扭矩 (不再互相覆盖!) =====
        for i, robot in enumerate(self.robots):
            # 对机体施加力矩（绕质心的旋转力）
            robot.set_external_force_and_torque(
                forces=torch.zeros((self.num_envs, 1, 3), device=self.device),
                torques=self._moments[:, i].unsqueeze(1),  # (N, 1, 3)
                body_ids=torch.zeros(1, dtype=torch.int, device=self.device),  # base_link
            )

            # 对旋翼施加推力
            robot.set_external_force_and_torque(
                forces=self._forces[i],  # (N, 4, 3)
                torques=torch.zeros_like(self._forces[i]),
                body_ids=self._falcon_rotor_idx,  # 旋翼body索引
            )

        # ===== 更新移动物块位置 =====
        dt = self.physics_dt * self.cfg.decimation
        # 设置目标速度：X方向恒定速度，Y/Z方向静止
        self.target_velocities[:, :, 0] = self.cfg.target_velocity
        self.target_velocities[:, :, 1] = 0.0
        self.target_velocities[:, :, 2] = 0.0

        # 更新目标位置（积分速度得到位置）
        self.target_positions += self.target_velocities * dt

        # 将新位置写入仿真器
        for i, target in enumerate(self.targets):
            target_poses = torch.cat(
                [
                    self.target_positions[:, i] + self.scene.env_origins,
                    self._target_quat,
                ],
                dim=-1,
            )
            target.write_root_pose_to_sim(target_poses)

    def _normalize_observation(self, obs: torch.Tensor) -> torch.Tensor:
        """
        观测归一化函数
        
        注意：这里不进行手动归一化，而是依赖skrl的RunningStandardScaler
        """
        return obs

    def _get_observations(self) -> dict[str, torch.Tensor]:
        """
        获取观测值（简化约简状态）
        
        返回每个无人机的局部观测，包括：
        1. 自身状态（位置、速度、姿态矩阵）
        2. 其他无人机相对位置
        3. 目标相对位置和状态
        4. 距离信息和最近无人机标识
        """
        # ===== 从每个独立 robot 读取状态 =====
        for i, robot in enumerate(self.robots):
            root_state = robot.data.root_state_w  # (N, 13)
            self.drone_positions[:, i] = root_state[:, :3] - self.scene.env_origins
            self.drone_orientations[:, i] = root_state[:, 3:7]
            self.drone_linear_velocities[:, i] = root_state[:, 7:10]
            self.drone_angular_velocities[:, i] = root_state[:, 10:13]

        # 计算旋转矩阵
        self.drone_rot_matrices[:] = matrix_from_quat(self.drone_orientations)

        # 计算无人机到目标的距离矩阵（仅考虑XY平面）
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

        # 找到每个目标最近的无人机索引
        closest_drone_to_target = torch.argmin(
            drone_to_target_distances, dim=1
        )  # (N, T)

        observations = {}
        for drone_idx, agent_name in enumerate(self.cfg.possible_agents):
            # 1. 自身状态 (15维)
            obs_self = torch.cat(
                [
                    self.drone_positions[:, drone_idx],  # 位置 (3)
                    self.drone_linear_velocities[:, drone_idx],  # 线速度 (3)
                    self.drone_rot_matrices[:, drone_idx].view(self.num_envs, -1),  # 旋转矩阵 (9)
                ],
                dim=-1,
            )

            # 2. 其他无人机状态（相对位置）
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

            # 3. 目标状态（相对位置 + 捕获状态 + 价值）
            target_rel_pos = self.target_positions - self.drone_positions[
                :, drone_idx
            ].unsqueeze(1)

            obs_targets = torch.cat(
                [
                    target_rel_pos.view(self.num_envs, -1),  # 相对位置 (12)
                    self.target_captured.float().view(self.num_envs, -1),  # 捕获状态 (4)
                    self.target_values,  # 目标价值 (4)
                ],
                dim=-1,
            )

            # 4. 额外信息
            obs_distances = drone_to_target_distances[:, drone_idx]  # 到各目标距离 (4)
            is_closest = (closest_drone_to_target == drone_idx).float()  # 是否最近 (4)

            # 组合所有观测
            obs_t = torch.cat(
                [
                    obs_self,
                    obs_other_drones,
                    obs_targets,
                    obs_distances,
                    is_closest,
                ],
                dim=-1,
            )

            # 如果启用部分观测，则使用缓冲区存储历史观测
            if self.cfg.partial_obs:
                self._observation_buffers[agent_name].append(obs_t)
                observations[agent_name] = self._observation_buffers[
                    agent_name
                ].buffer.reshape(self.num_envs, -1)
            else:
                observations[agent_name] = obs_t

        return observations

    def _get_states(self) -> torch.Tensor:
        """
        获取全局状态（评论家输入）
        
        包含所有无人机和目标的完整状态信息，用于集中式训练。
        """
        states = torch.cat(
            (
                self.drone_positions.view(self.num_envs, -1),  # 无人机位置 (9)
                self.drone_rot_matrices.view(self.num_envs, -1),  # 旋转矩阵 (27)
                self.drone_linear_velocities.view(self.num_envs, -1),  # 线速度 (9)
                self.drone_angular_velocities.view(self.num_envs, -1),  # 角速度 (9)
                self.target_positions.view(self.num_envs, -1),  # 目标位置 (12)
                self.target_velocities.view(self.num_envs, -1),  # 目标速度 (12)
                self.target_captured.float().view(self.num_envs, -1),  # 捕获状态 (4)
                self.target_values,  # 目标价值 (4)
            ),
            dim=-1,
        )
        return states

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        """
        计算奖励函数（指数衰减奖励，类似hover/hover_flycart风格）
        
        包含多个奖励项：
        1. 距离奖励：鼓励接近目标
        2. 追踪奖励：奖励成功追踪目标
        3. 动作平滑性奖励：鼓励动作变化平缓
        4. 机体角速率惩罚：避免过度旋转
        5. 速度惩罚：避免速度过大
        6. 推力惩罚：避免推力过大
        7. 高度奖励：维持合适高度
        8. 安全惩罚：避免碰撞和越界
        """
        rewards = {}
        step_dt = self.step_dt

        # --- 1. 距离奖励 (XY平面, 指数衰减) ---
        # 计算所有无人机到所有目标的距离矩阵
        d_pos = self.drone_positions[:, :, :2].unsqueeze(2)  # (N,D,1,2)
        t_pos = self.target_positions[:, :, :2].unsqueeze(1)  # (N,1,T,2)
        dist_matrix = torch.norm(d_pos - t_pos, dim=-1)  # (N, D, T)
        min_dists, closest_drone_indices = torch.min(dist_matrix, dim=1)  # (N, T)

        # 加权指数衰减距离奖励
        dist_per_target = torch.exp(-min_dists * self.cfg.dist_reward_scale)  # (N, T)
        dist_reward = torch.sum(dist_per_target * self.target_values, dim=-1)
        rewards["distance_reward"] = self.cfg.dist_reward_weight * dist_reward * step_dt

        # --- 更新捕获状态 (实时, 可撤销) ---
        is_captured_now = min_dists < self.cfg.capture_distance
        self.target_captured = is_captured_now  # 实时状态
        # 记录哪个无人机捕获了哪个目标
        self.target_captured_by = torch.where(
            is_captured_now,
            closest_drone_indices,
            self.target_captured_by,
        )

        # 持续跟随计时：至少3个不同物块各自被至少一架无人机跟随
        target_min_dist = dist_matrix.min(dim=1)[
            0
        ]  # (N, T) — 每个物块到最近无人机的距离
        target_followed = target_min_dist < self.cfg.capture_distance  # (N, T)
        num_targets_followed = target_followed.sum(dim=-1)  # (N,)
        enough_targets_followed = num_targets_followed >= 3  # 要求至少3个不同物块被跟随
        self._sustained_follow_timer = torch.where(
            enough_targets_followed,
            self._sustained_follow_timer + step_dt,
            self._sustained_follow_timer,  # 修复：暂停计时器而不是重置
        )
        self.all_targets_captured = (
            self._sustained_follow_timer >= self.cfg.sustained_follow_duration
        )

        # 成功奖励（已移除）
        # rewards["success_reward"] = ...

        # 追踪奖励（修复：仅在捕获时奖励，防止奖励反转）
        # 旧的错误逻辑：exp(-dist) 当dist=0时无目标 -> 最大奖励什么都没做
        # 新逻辑：is_captured * exp(-dist) -> 未捕获时奖励为0
        tracking_reward = (
            is_captured_now.float()
            * torch.exp(-min_dists * self.cfg.tracking_reward_scale)
        ).sum(dim=-1)
        rewards["tracking_reward"] = (
            self.cfg.tracking_reward_weight * tracking_reward * step_dt
        )

        # --- 2. 动作平滑性奖励 (指数衰减, 正向) ---
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

        # --- 3. 机体角速率惩罚 (指数衰减, 正向) ---
        commanded_body_rates = torch.cat(
            [self.actions[a][:, 3:] for a in self.cfg.possible_agents],
            dim=-1,
        )
        body_rate_norm = torch.norm(commanded_body_rates / self._num_drones, dim=-1)
        rewards["body_rate_penalty"] = (
            self.cfg.body_rate_penalty_weight * torch.exp(-body_rate_norm) * step_dt
        )

        # --- 4. 速度惩罚 (指数衰减, 正向) ---
        vel_norms = torch.norm(self.drone_linear_velocities, dim=-1)  # (N, D)
        avg_vel = vel_norms.mean(dim=-1)  # (N,)
        rewards["velocity_penalty"] = (
            self.cfg.velocity_penalty_weight * torch.exp(-avg_vel) * step_dt
        )

        # --- 5. 推力惩罚 (指数衰减, 正向) ---
        all_forces = torch.stack([f[..., 2] for f in self._forces], dim=1)  # (N, D, 4)
        normalized_forces = all_forces / self.cfg.max_thrust_pp
        effort_max = torch.max(normalized_forces.view(self.num_envs, -1), dim=-1)[0]
        rewards["force_penalty"] = (
            self.cfg.force_penalty_weight * torch.exp(-effort_max) * step_dt
        )

        # --- 6. 高度奖励 (指数衰减, 正向) ---
        height_error = torch.norm(
            self.drone_positions[..., 2] - self.cfg.desired_height,
            dim=-1,
        )
        rewards["height_reward"] = (
            self.cfg.height_reward_weight * torch.exp(-height_error) * step_dt
        )

        # 高度惩罚（新增：严格约束偏离 > 阈值的情况）
        # 线性惩罚：weight * max(0, error - threshold)
        excess_height = (height_error - self.cfg.height_penalty_threshold).clamp(
            min=0.0
        )
        rewards["height_penalty"] = (
            -self.cfg.height_penalty_weight * excess_height * step_dt
        )

        # --- 6.5 Upright惩罚（新增：防止翻滚）---
        # 惩罚机体Z轴偏离世界Z轴(0,0,1)的情况
        # R[..., 2, 2] 是机体Z轴与世界Z轴的点积
        z_axis_body = self.drone_rot_matrices[:, :, 2, 2]  # (N, D)
        # (z - 1.0) 在倾斜时为负值。乘以权重（正值）-> 负奖励
        rewards["upright_penalty"] = (
            self.cfg.upright_penalty_weight * (z_axis_body - 1.0).sum(-1) * step_dt
        )

        # ====== 安全惩罚（保持不变）======

        # --- 7. 碰撞惩罚 ---
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

        # --- 8. 越界惩罚 ---
        out_of_bounds = (
            (self.drone_positions.abs() > self.cfg.bounding_box_threshold)
            .any(dim=-1)
            .any(dim=-1)
        )
        rewards["drone_out"] = (
            -out_of_bounds.float() * self.cfg.drone_out_of_bounds_penalty
        )

        # --- 9. 飞行过低惩罚 ---
        fly_low = (self.drone_positions[:, :, 2] < 0.1).any(dim=-1)
        rewards["fly_low"] = -fly_low.float() * self.cfg.fly_low_penalty

        # --- 10. 非法接触惩罚 ---
        illegal_any = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for cs in self.contact_sensors:
            net_f = cs.data.net_forces_w_history
            max_f = torch.max(torch.norm(net_f, dim=-1), dim=1)[0]
            has_contact = (max_f > self.cfg.contact_sensor_threshold).any(dim=1)
            illegal_any = illegal_any | has_contact
        rewards["illegal_contact"] = (
            -illegal_any.float() * self.cfg.illegal_contact_penalty
        )

        # --- 11. 时间惩罚（已移除）---
        # rewards["time_penalty"] = ...

        # --- 总奖励 ---
        total_reward = sum(rewards.values())

        # 日志记录
        for key, val in rewards.items():
            if key not in self._episode_sums:
                self._episode_sums[key] = torch.zeros_like(val)
            self._episode_sums[key] += val

        return {agent: total_reward for agent in self.cfg.possible_agents}

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """
        计算终止条件
        
        返回两个字典：
        1. terminations: 立即终止条件（失败）
        2. time_outs: 时间超时终止
        """
        # 无人机飞太低
        self.falcon_fly_low = (self.drone_positions[:, :, 2] < 0.1).any(dim=-1)

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
            | (self.target_positions[:, :, 2] < 0.05)
        ).any(dim=-1)

        # 组合终止条件
        terminations = (
            self.falcon_fly_low
            | self.illegal_contact
            | self.drone_collision
            | self.body_pos_outside
            | self.targets_out_of_bounds
        )

        # 调试日志
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
                    # 检查当前捕获的目标
                    current_captures = self.target_captured[idx]  # (T,) bool
                    if current_captures.any():
                        current_captured_by = self.target_captured_by[idx][
                            current_captures
                        ]
                        unique_drones = torch.unique(current_captured_by)
                        # 过滤掉-1，虽然is_captured检查应该已经防止了
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
        """
        重置指定环境索引的状态
        
        Args:
            env_ids: 需要重置的环境索引列表
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        super()._reset_idx(env_ids)
        self._reset_targets(env_ids)
        self._prev_vel_error[env_ids] = 0.0
        self._sustained_follow_timer[env_ids] = 0.0

        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device)

        # ===== 对每个robot独立重置 (无需索引映射!) =====
        # 计算编队中心位置
        center_x = torch.empty(len(env_ids), device=self.device).uniform_(
            *self.cfg.drone_spawn_x_range
        )
        center_y = torch.empty(len(env_ids), device=self.device).uniform_(
            *self.cfg.drone_spawn_y_range
        )

        # 三角形编队配置
        radius = 2.0
        height = 2.5
        phases = torch.tensor(
            [0.0, 2.0 * 3.14159 / 3.0, 4.0 * 3.14159 / 3.0], device=self.device
        )

        for i, robot in enumerate(self.robots):
            robot.reset(env_ids)

            origins = self.scene.env_origins[env_ids]

            # 计算偏移位置
            offsets = torch.zeros((len(env_ids), 3), device=self.device)
            offsets[:, 0] = center_x + radius * torch.cos(phases[i])
            offsets[:, 1] = center_y + radius * torch.sin(phases[i])
            offsets[:, 2] = height

            new_positions = origins + offsets

            # 获取默认根状态并设置新位姿
            default_root_state = robot.data.default_root_state[env_ids]
            new_quats = default_root_state[:, 3:7]
            new_poses = torch.cat([new_positions, new_quats], dim=-1)

            robot.write_root_pose_to_sim(new_poses, env_ids=env_ids)
            robot.write_root_velocity_to_sim(
                torch.zeros_like(default_root_state[:, 7:]), env_ids=env_ids
            )

        # 重置观测缓冲区
        for agent in self.cfg.possible_agents:
            self._observation_buffers[agent].reset(env_ids)

        # 记录奖励组件
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
        self.extras["log"]["Episode_Termination/crash"] = torch.count_nonzero(
            self.falcon_fly_low[env_ids] | self.illegal_contact[env_ids]
        ).item()
        self.extras["log"]["Episode_Termination/out_of_bounds"] = torch.count_nonzero(
            self.targets_out_of_bounds[env_ids]
        ).item()

        # 记录回合奖励
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

        # 重置动作历史
        for agent in self.cfg.possible_agents:
            self.prev_actions[agent][env_ids] = 0.0
            self.actions[agent][env_ids] = 0.0

        self.all_targets_captured[env_ids] = False

    def _reset_targets(self, env_ids):
        """
        重置移动物块的位置和状态
        
        Args:
            env_ids: 需要重置的环境索引
        """
        # 计算展平的索引
        target_indices_flat = (
            env_ids.view(-1, 1) * self.num_targets
            + torch.arange(self.num_targets, device=self.device).view(1, -1)
        ).flatten()

        # 获取展平的位置和速度张量
        flat_pos = self.target_positions.view(-1, 3)
        flat_vel = self.target_velocities.view(-1, 3)

        # 重置捕获状态
        self.target_captured[env_ids] = False
        self.target_captured_by[env_ids] = -1

        # 重置位置
        for i in range(self.num_targets):
            current_indices = env_ids * self.num_targets + i

            # 随机生成X坐标，固定Y坐标，固定Z坐标
            r = torch.empty(len(env_ids), device=self.device)
            val_x = r.uniform_(*self.cfg.target_spawn_x_range)
            val_y = self.cfg.target_spawn_y_positions[i]
            val_z = self.cfg.target_spawn_z

            flat_pos[current_indices, 0] = val_x
            flat_pos[current_indices, 1] = val_y
            flat_pos[current_indices, 2] = val_z

        # 重置速度
        flat_vel[target_indices_flat] = 0.0

    def _set_debug_vis_impl(self, debug_vis: bool):
        """
        设置调试可视化实现（当前为空实现）
        """
        pass

    def _debug_vis_callback(self, event):
        """
        调试可视化回调函数（当前为空实现）
        """
        pass


@torch.jit.script
def scale(x, lower, upper):
    """
    线性缩放函数：将[-1,1]范围映射到[lower, upper]
    
    Args:
        x: 输入值 (范围 [-1, 1])
        lower: 下界
        upper: 上界
    
    Returns:
        缩放后的值
    """
    return 0.5 * (x + 1.0) * (upper - lower) + lower


@torch.jit.script
def unscale(x, lower, upper):
    """
    线性反缩放函数：将[lower, upper]范围映射回[-1, 1]
    
    Args:
        x: 输入值 (范围 [lower, upper])
        lower: 下界
        upper: 上界
    
    Returns:
        反缩放后的值
    """
    return (2.0 * x - upper - lower) / (upper - lower)