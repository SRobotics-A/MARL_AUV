# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import copy
import math
import numpy as np
import torch
from collections.abc import Sequence

# 导入控制器模块
from MARL_mav_carry_ext.controllers import GeometricController, IndiController
from MARL_mav_carry_ext.controllers.motor_model import RotorMotor
# 导入低层控制工具函数
from MARL_mav_carry_ext.tasks.managerbased.mdp_llc.utils import get_drone_pdist, get_drone_rpos

# 导入IsaacLab相关模块
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectMARLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.sensors import ContactSensor
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import CircularBuffer, DelayBuffer
from isaaclab.utils.math import (
    compute_pose_error,
    euler_xyz_from_quat,
    matrix_from_quat,
    quat_error_magnitude,
    quat_from_angle_axis,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    quat_apply,
    quat_unique,
    sample_uniform,
)

# 导入环境配置
from .marl_hover_env_cfg import MARLHoverEnvCfg


class MARLHoverEnv(DirectMARLEnv):
    """多智能体悬停环境类
    
    实现三架Falcon无人机协同控制悬挂负载在指定位置悬停的任务
    支持ACCBR和几何控制两种控制模式
    """
    
    cfg: MARLHoverEnvCfg

    def __init__(self, cfg: MARLHoverEnvCfg, render_mode: str | None = None, **kwargs):
        """初始化环境
        
        Args:
            cfg: 环境配置对象
            render_mode: 渲染模式
            **kwargs: 其他参数
        """
        super().__init__(cfg, render_mode, **kwargs)

        # 获取机体索引
        self._falcon_idx = self.robot.find_bodies(cfg.falcon_names)[0]        # 无人机索引
        self._falcon_rotor_idx = self.robot.find_bodies(cfg.falcon_rotor_names)[0]  # 旋翼索引
        self._payload_idx = self.robot.find_bodies(cfg.payload_name)[0]       # 负载索引
        self._rope_idx = self.robot.find_bodies(cfg.rope_name)[0]             # 绳索索引

        # 配置参数
        self._num_drones = len(self._falcon_idx)    # 无人机数量
        self._control_mode = cfg.control_mode       # 控制模式

        # 观测缓冲区：用于存储历史观测数据
        self._observation_buffers = {}
        for agent in self.cfg.possible_agents:
            self._observation_buffers[agent] = CircularBuffer(cfg.history_len, self.num_envs, device=self.device)

        # 动作缓冲区和力矩计算相关
        self._forces = torch.zeros(self.num_envs, len(self._falcon_rotor_idx), 3, device=self.device)  # 推力
        self._moments = torch.zeros(self.num_envs, len(self._falcon_idx), 3, device=self.device)       # 力矩
        
        # 设定点和前一时刻动作存储
        self._setpoints = {}
        self.prev_actions = {}
        for agent in self.cfg.possible_agents:
            self._setpoints[agent] = {}
            if self._control_mode == "geometric":
                self.prev_actions[agent] = torch.zeros(self.num_envs, 12, device=self.device)  # 几何控制12维动作
            elif self._control_mode == "ACCBR":
                self.prev_actions[agent] = torch.zeros(self.num_envs, 5, device=self.device)   # ACCBR控制5维动作

        # 无人机状态变量初始化
        self.drone_positions = torch.zeros(self.num_envs, self._num_drones, 3, device=self.device)           # 位置
        self.drone_orientations = torch.zeros(self.num_envs, self._num_drones, 4, device=self.device)        # 四元数姿态
        self.drone_orientations[..., 0] = 1.0  # 初始化为单位四元数
        self.drone_rot_matrices = torch.zeros(self.num_envs, self._num_drones, 3, 3, device=self.device)     # 旋转矩阵
        self.drone_linear_velocities = torch.zeros(self.num_envs, self._num_drones, 3, device=self.device)   # 线速度
        self.drone_angular_velocities = torch.zeros(self.num_envs, self._num_drones, 3, device=self.device)  # 角速度
        self.drone_linear_accelerations = torch.zeros(self.num_envs, self._num_drones, 3, device=self.device) # 线加速度
        self.drone_angular_accelerations = torch.zeros(self.num_envs, self._num_drones, 3, device=self.device) # 角加速度
        self._drone_jerk = torch.zeros(self.num_envs, self._num_drones, 3, device=self.device)               # 加加速度
        self._drone_prev_acc = torch.zeros(self.num_envs, self._num_drones, 3, device=self.device)           # 前一时刻加速度

        # 外环控制器（几何控制器）
        self.geo_controllers = {}
        for i in range(self._num_drones):
            self.geo_controllers[i] = GeometricController(self.num_envs, self._control_mode)
        self._ll_counter = 0                    # 低层控制计数器
        self._constant_yaw = torch.zeros([self.num_envs, 1], device=self.device)  # 恒定偏航角
        self._zeros = torch.zeros([self.num_envs, 3], device=self.device)         # 零向量

        # 内环控制器（INDI增量式非线性动态反演控制器）
        self._indi_controllers = {}
        for i in range(self._num_drones):
            self._indi_controllers[i] = IndiController(self.num_envs)

        # 电机模型
        self.motor_models = {}
        # 实验获得的初始转速
        initial_rpms = [
            torch.tensor([[1441.5819, 1351.1626, 1341.0111, 1428.5597]], device=self.device).repeat(self.num_envs, 1),
            torch.tensor([[1377.9199, 1451.8428, 1408.9022, 1329.2014]], device=self.device).repeat(self.num_envs, 1),
            torch.tensor([[1281.3964, 1293.0708, 1361.7539, 1347.2434]], device=self.device).repeat(self.num_envs, 1),
        ]
        for i in range(self._num_drones):
            self.motor_models[i] = RotorMotor(self.num_envs, initial_rpms[i])
        self.sampling_time = self.sim.get_physics_dt() * self.cfg.low_level_decimation  # 采样时间

        # 负载状态缓冲区
        self.load_position = torch.zeros(self.num_envs, 3, device=self.device)        # 负载位置
        self.load_orientation = torch.zeros(self.num_envs, 4, device=self.device)     # 负载姿态
        self.load_orientation[:, 0] = 1.0  # 初始化为单位四元数
        self.current_load_matrix = torch.zeros(self.num_envs, 3, 3, device=self.device)  # 负载旋转矩阵
        self.load_vel = torch.zeros(self.num_envs, 3, device=self.device)             # 负载线速度
        self.load_ang_vel = torch.zeros(self.num_envs, 3, device=self.device)         # 负载角速度
        self.load_length_x = torch.tensor([[0.275, 0, 0]] * self.num_envs, device=self.device)  # 负载尺寸x方向
        self.load_length_y = torch.tensor([[0, 0.275, 0]] * self.num_envs, device=self.device)  # 负载尺寸y方向

        # 目标指令相关
        self.pose_command_w = torch.zeros(self.num_envs, 7, device=self.device)       # 位姿指令（位置+四元数）
        self.pose_command_w[:, 3] = 1.0  # 初始化四元数实部为1
        self.goal_pos_error = torch.zeros(self.num_envs, 3, device=self.device)       # 位置误差
        self.difference_matrix = torch.zeros(self.num_envs, 3, 3, device=self.device) # 姿态差矩阵
        self.goal_dist_counter = torch.zeros(self.num_envs, device=self.device)       # 目标距离计数器

        # 奖励累计统计
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "pos_reward",           # 位置奖励
                "ori_reward",           # 姿态奖励
                "action_smoothness",    # 动作平滑性奖励
                "body_rate_penalty",    # 机体角速度惩罚
                "force_penalty",        # 推力惩罚
                "downwash_reward",      # 下洗流奖励
            ]
        }

        # 性能指标
        self.metrics = {}
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)      # 位置误差
        self.metrics["orientation_error"] = torch.zeros(self.num_envs, device=self.device)   # 姿态误差

        # 终止条件缓冲区
        self.falcon_fly_low = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)      # 无人机飞得太低
        self.payload_fly_low = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)     # 负载飞得太低
        self.illegal_contact = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)     # 非法接触
        self.angle_limit_drone = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)   # 无人机端角度限制
        self.angle_limit_load = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)    # 负载端角度限制
        self.cable_collision = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)     # 电缆碰撞
        self.drone_collision = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)     # 无人机碰撞
        self.body_pos_outside = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)    # 超出边界
        self.time_out = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)            # 时间超时

        # 调试可视化设置
        self.set_debug_vis(cfg.debug_vis)

    def _setup_scene(self):
        """设置仿真场景"""
        self.robot = Articulation(self.cfg.robot_cfg)           # 创建机器人实体
        contact_sensors = ContactSensor(self.cfg.contact_forces) # 创建接触传感器
        # 添加地面平面
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # 克隆和复制环境
        self.scene.clone_environments(copy_from_source=False)
        # 将实体添加到场景中
        self.scene.articulations["robot"] = self.robot
        self.scene.sensors["contact_forces"] = contact_sensors
        # 添加光源
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        """物理步骤前的处理：解析和缓存动作
        
        Args:
            actions: 各智能体的动作字典
        """
        for agent in self.cfg.possible_agents:
            # 保存前一时刻动作用于平滑性奖励计算
            self.prev_actions[agent][:] = self.actions[agent]
            self.actions[agent][:] = actions[agent]

        # 解析各无人机的动作指令
        for drone, action in actions.items():
            if self._control_mode == "geometric":
                # 几何控制模式：12维动作 [位置(3) + 线速度(3) + 线加速度(3) + 加加速度(3)]
                self._setpoints[drone]["pos"] = action[:, :3]
                self._setpoints[drone]["lin_vel"] = action[:, 3:6]
                self._setpoints[drone]["lin_acc"] = action[:, 6:9]
                self._setpoints[drone]["jerk"] = action[:, 9:12]
            elif self._control_mode == "ACCBR":
                # ACCBR控制模式：5维动作 [线加速度(3) + 角速度xy(2)]
                self._setpoints[drone]["lin_acc"] = action[:, :3]
                self._setpoints[drone]["body_rates"] = torch.cat((action[:, 3:], self._constant_yaw), dim=-1)

            # 设置恒定的偏航角相关指令
            self._setpoints[drone]["yaw"] = self._constant_yaw
            self._setpoints[drone]["yaw_rate"] = self._constant_yaw
            self._setpoints[drone]["yaw_acc"] = self._constant_yaw

    def _apply_action(self) -> None:
        """应用动作：将RL动作转换为物理控制输入"""
        if self._ll_counter % self.cfg.low_level_decimation == 0:
            all_thrusts = []    # 存储所有推力
            all_moments = []    # 存储所有力矩

            # 获取当前无人机状态
            drone_positions = self.robot.data.body_com_state_w[
                :, self._falcon_idx, :3
            ] - self.scene.env_origins.unsqueeze(1)  # 相对于环境原点的位置
            drone_orientations = self.robot.data.body_com_state_w[:, self._falcon_idx, 3:7]      # 姿态四元数
            drone_linear_velocities = self.robot.data.body_com_state_w[:, self._falcon_idx, 7:10] # 线速度
            drone_angular_velocities = self.robot.data.body_com_state_w[:, self._falcon_idx, 10:13] # 角速度
            drone_linear_accelerations = self.robot.data.body_acc_w[:, self._falcon_idx, :3]     # 线加速度
            drone_angular_accelerations = self.robot.data.body_acc_w[:, self._falcon_idx, 3:6]   # 角加速度

            # 更新状态变量（注释掉的噪声添加代码可用于增加鲁棒性）
            self.drone_positions[:] = drone_positions
            self.drone_orientations[:] = drone_orientations
            self.drone_linear_velocities[:] = drone_linear_velocities
            self.drone_angular_velocities[:] = drone_angular_velocities
            self.drone_linear_accelerations[:] = drone_linear_accelerations
            self.drone_angular_accelerations[:] = drone_angular_accelerations

            # 对每架无人机进行控制计算
            for i in range(self._num_drones):
                # 构造无人机状态字典
                drone_states: dict = {}
                drone_states["pos"] = self.drone_positions[:, i]
                drone_states["quat"] = self.drone_orientations[:, i]
                drone_states["lin_vel"] = self.drone_linear_velocities[:, i]
                drone_states["ang_vel"] = self.drone_angular_velocities[:, i]
                drone_states["lin_acc"] = self.drone_linear_accelerations[:, i]
                drone_states["ang_acc"] = self.drone_angular_accelerations[:, i]
                
                # 计算加加速度（ jerk ）
                self._drone_jerk[:, i] = (drone_states["lin_acc"] - self._drone_prev_acc[:, i]) / (self.step_dt)
                drone_states["jerk"] = self._drone_jerk[:, i]
                self._drone_prev_acc[:, i] = drone_states["lin_acc"]

                # 外环控制器计算期望的力和力矩
                alpha_cmd, acc_load, acc_cmd, q_cmd = self.geo_controllers[i].getCommand(
                    drone_states, self._forces[:, i * 4 : i * 4 + 4], self._setpoints[f"falcon{i+1}"]
                )

                # 内环INDI控制器计算目标转速
                target_rpm = self._indi_controllers[i].getCommand(
                    drone_states, self._forces[:, i * 4 : i * 4 + 4], alpha_cmd, acc_cmd, acc_load
                )

                # 电机模型计算实际推力和力矩
                thrusts, moments = self.motor_models[i].get_motor_thrusts_moments(target_rpm, self.sampling_time)
                all_thrusts.append(thrusts)
                all_moments.append(moments)

            # 合并所有无人机的推力和力矩
            forces = torch.cat(all_thrusts, dim=-1)
            torques = torch.cat(all_moments, dim=-1)
            self._forces[..., 2] = forces                    # Z方向推力
            self._moments[..., 2] = torques.view(self.num_envs, self._num_drones, 4).sum(-1)  # 合力矩
            self._ll_counter = 0  # 重置计数器
            
        self._ll_counter += 1

        # 将计算得到的力和力矩应用到物理系统
        self.robot.set_external_force_and_torque(
            torch.zeros_like(self._moments), self._moments, body_ids=self._falcon_idx  # 对机体施加力矩
        )
        self.robot.set_external_force_and_torque(
            self._forces, torch.zeros_like(self._forces), body_ids=self._falcon_rotor_idx  # 对旋翼施加推力
        )

    def _get_observations(self) -> dict[str, torch.Tensor]:
        """获取观测值：构建各智能体的局部观测空间"""
        
        # 更新负载状态
        self.load_position[:] = (
            self.robot.data.body_com_state_w[:, self._payload_idx, :3].squeeze(1) - self.scene.env_origins
        )
        self.current_load_matrix[:] = matrix_from_quat(self.load_orientation)
        self.load_vel[:] = self.robot.data.body_com_state_w[:, self._payload_idx, 7:10].squeeze(1)
        self.load_ang_vel[:] = self.robot.data.body_com_state_w[:, self._payload_idx, 10:13].squeeze(1)

        # 更新无人机状态
        self.drone_positions[:] = self.robot.data.body_com_state_w[
            :, self._falcon_idx, :3
        ] - self.scene.env_origins.unsqueeze(1)
        self.drone_orientations[:] = self.robot.data.body_com_state_w[:, self._falcon_idx, 3:7]
        self.drone_rot_matrices[:] = matrix_from_quat(self.drone_orientations)
        self.drone_linear_velocities[:] = self.robot.data.body_com_state_w[:, self._falcon_idx, 7:10]
        self.drone_angular_velocities[:] = self.robot.data.body_com_state_w[:, self._falcon_idx, 10:13]

        # 计算目标误差
        self.goal_pos_error[:] = self.pose_command_w[:, :3] - self.load_position
        goal_load_matrix = matrix_from_quat(self.pose_command_w[:, 3:7])
        self.difference_matrix[:] = torch.matmul(goal_load_matrix, self.current_load_matrix.transpose(1, 2))

        # 根据是否使用局部观测构建不同的观测空间
        if self.cfg.partial_obs:
            # 局部观测模式：每架无人机只能看到自己的信息和其他共享信息
            obs_falcon1_t = torch.cat(
                (
                    self.load_position,                             # 负载位置
                    self.current_load_matrix.view(self.num_envs, -1), # 负载姿态矩阵
                    # 本机信息（one-hot编码标识）
                    torch.tensor([[1, 0, 0]] * self.num_envs, device=self.device),
                    self.drone_positions[:, 0].view(self.num_envs, -1),      # 本机位置
                    self.drone_rot_matrices[:, 0].view(self.num_envs, -1),   # 本机姿态
                    self.drone_linear_velocities[:, 0].view(self.num_envs, -1), # 本机线速度
                    self.drone_angular_velocities[:, 0].view(self.num_envs, -1), # 本机角速度
                    self.goal_pos_error,                            # 目标位置误差
                    self.difference_matrix.view(self.num_envs, -1), # 目标姿态误差
                ),
                dim=-1,
            )
            self._observation_buffers["falcon1"].append(obs_falcon1_t)

            # 重复类似过程构建其他两架无人机的观测
            obs_falcon2_t = torch.cat(
                (
                    self.load_position,
                    self.current_load_matrix.view(self.num_envs, -1),
                    torch.tensor([[0, 1, 0]] * self.num_envs, device=self.device),  # 标识第二架无人机
                    self.drone_positions[:, 1].view(self.num_envs, -1),
                    self.drone_rot_matrices[:, 1].view(self.num_envs, -1),
                    self.drone_linear_velocities[:, 1].view(self.num_envs, -1),
                    self.drone_angular_velocities[:, 1].view(self.num_envs, -1),
                    self.goal_pos_error,
                    self.difference_matrix.view(self.num_envs, -1),
                    # self.all_action_histories.reshape(self.num_envs, -1),
                ),
                dim=-1,
            )

            self._observation_buffers["falcon2"].append(obs_falcon2_t)

            obs_falcon3_t = torch.cat(
                (
                    self.load_position,
                    self.current_load_matrix.view(self.num_envs, -1),
                    torch.tensor([[0, 0, 1]] * self.num_envs, device=self.device),  # 标识第三架无人机
                    self.drone_positions[:, 2].view(self.num_envs, -1),
                    self.drone_rot_matrices[:, 2].view(self.num_envs, -1),
                    self.drone_linear_velocities[:, 2].view(self.num_envs, -1),
                    self.drone_angular_velocities[:, 2].view(self.num_envs, -1),
                    self.goal_pos_error,
                    self.difference_matrix.view(self.num_envs, -1),
                    # self.all_action_histories.reshape(self.num_envs, -1),
                ),
                dim=-1,
            )

            self._observation_buffers["falcon3"].append(obs_falcon3_t)

            # 从缓冲区获取历史观测数据
            obs_falcon1 = self._observation_buffers["falcon1"].buffer.reshape(self.num_envs, -1)
            obs_falcon2 = self._observation_buffers["falcon2"].buffer.reshape(self.num_envs, -1)
            obs_falcon3 = self._observation_buffers["falcon3"].buffer.reshape(self.num_envs, -1)
        else:
            # 全局观测模式：包含所有无人机的信息
            obs_falcon1 = torch.cat(
                (
                    self.load_position,
                    self.current_load_matrix.view(self.num_envs, -1),
                    self.load_vel,                                  # 负载速度
                    self.load_ang_vel,                              # 负载角速度
                    torch.tensor([[1, 0, 0]] * self.num_envs, device=self.device),  # one-hot编码
                    self.drone_positions.view(self.num_envs, -1),   # 所有无人机位置
                    self.drone_rot_matrices.view(self.num_envs, -1), # 所有无人机姿态
                    self.drone_linear_velocities.view(self.num_envs, -1), # 所有无人机线速度
                    self.drone_angular_velocities.view(self.num_envs, -1), # 所有无人机角速度
                    self.goal_pos_error,
                    self.difference_matrix.view(self.num_envs, -1),
                    # self.all_action_histories.reshape(self.num_envs, -1),
                ),
                dim=-1,
            )
            # 为其他无人机构建类似的全局观测...

        observations = {
            "falcon1": obs_falcon1,
            "falcon2": obs_falcon2,
            "falcon3": obs_falcon3,
        }
        return observations

    def _get_states(self) -> torch.Tensor:
        """获取全局状态：包含所有信息的完整状态表示"""
        states = torch.cat(
            (
                # 负载状态
                self.load_position,
                self.current_load_matrix.view(self.num_envs, -1),
                self.load_vel,
                self.load_ang_vel,
                # 所有无人机状态
                self.drone_positions.view(self.num_envs, -1),
                self.drone_rot_matrices.view(self.num_envs, -1),
                self.drone_linear_velocities.view(self.num_envs, -1),
                self.drone_angular_velocities.view(self.num_envs, -1),
                # 目标信息
                self.goal_pos_error,
                self.difference_matrix.view(self.num_envs, -1),
                # self.all_action_histories.reshape(self.num_envs, -1),
            ),
            dim=-1,
        )

        return states

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        """计算奖励函数"""
        # 位置跟踪奖励：指数衰减形式
        goal_pos_error_norm = torch.norm(self.pose_command_w[:, :3] - self.load_position, dim=-1)
        reward_distance_scale = 1.5
        reward_position = (
            self.cfg.pos_track_weight * torch.exp(-goal_pos_error_norm * reward_distance_scale) * self.step_dt
        )

        # 姿态跟踪奖励
        orientation_error = quat_error_magnitude(self.pose_command_w[:, 3:7], self.load_orientation)
        reward_distance_scale = 1.5
        reward_orientation = (
            self.cfg.ori_track_weight * torch.exp(-orientation_error * reward_distance_scale) * self.step_dt
        )

        # 动作平滑性奖励：惩罚动作变化过大的情况
        current_actions = torch.cat([self.actions[agent] for agent in self.cfg.possible_agents], dim=-1)
        action_prev = torch.cat([self.prev_actions[agent] for agent in self.cfg.possible_agents], dim=-1)
        diff_action = ((current_actions - action_prev).abs()) / self._num_drones
        reward_action_smoothness = (
            self.cfg.action_smoothness_weight * torch.exp(-torch.norm(diff_action, dim=-1).square()) * self.step_dt
        )

        # 机体角速度惩罚：避免过快的旋转
        commanded_body_rates = torch.cat([self.actions[agent][:, 3:] for agent in self.cfg.possible_agents], dim=-1)
        body_rate_penalty = torch.norm(commanded_body_rates / self._num_drones, dim=-1)
        reward_body_rate_penalty = self.cfg.body_rate_penalty_weight * torch.exp(-body_rate_penalty) * self.step_dt

        # 推力惩罚：鼓励节能控制
        normalized_forces = self._forces[..., 2] / self.cfg.max_thrust_pp
        effort_sum = torch.max(normalized_forces, dim=-1)[0]
        reward_effort = self.cfg.force_penalty_weight * torch.exp(-effort_sum) * self.step_dt

        # 下洗流奖励：鼓励无人机避开彼此的下洗气流影响
        reward_downwash = self.cfg.downwash_rew_weight * self._downwash_reward() * self.step_dt

        # 构建奖励字典
        rewards = {
            "pos_reward": reward_position,
            "ori_reward": reward_orientation,
            "action_smoothness": reward_action_smoothness,
            "body_rate_penalty": reward_body_rate_penalty,
            "force_penalty": reward_effort,
            "downwash_reward": reward_downwash,
        }

        # 累计奖励用于日志记录
        for key, value in rewards.items():
            self._episode_sums[key] += value

        # 共享奖励：所有智能体获得相同的总奖励
        shared_rewards = (
            reward_position
            + reward_orientation
            + reward_action_smoothness
            + reward_body_rate_penalty
            + reward_effort
            + reward_downwash
        )

        return {agent: shared_rewards for agent in self.cfg.possible_agents}

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """判断终止条件：检查各种安全和任务完成条件"""
        
        # 更新负载位置
        self.load_position[:] = (
            self.robot.data.body_com_state_w[:, self._payload_idx, :3].squeeze(1) - self.scene.env_origins
        )

        # 高度限制检查：防止撞击地面
        self.falcon_fly_low = (self.drone_positions[:, :, 2] < 0.1).any(dim=-1)  # 任意无人机太低
        self.payload_fly_low = self.load_position[:, 2] < 0.1                    # 负载太低

        # 非法接触检测：通过接触传感器检测异常碰撞
        contact_sensor = self.scene.sensors[self.cfg.sensor_cfg.name]
        net_contact_forces = contact_sensor.data.net_forces_w_history
        self.illegal_contact = torch.any(
            torch.max(torch.norm(net_contact_forces[:, :, self.cfg.sensor_cfg.body_ids], dim=-1), dim=1)[0]
            > self.cfg.contact_sensor_threshold,
            dim=1,
        )

        # 电缆角度限制检查：防止电缆过度倾斜导致脱落
        rope_orientations_world = self.robot.data.body_com_state_w[:, self._rope_idx, 3:7].view(-1, 4)
        drone_orientation_world = self.drone_orientations.view(-1, 4)
        drone_orientation_inv = quat_inv(drone_orientation_world)
        rope_orientations_drones = quat_mul(
            drone_orientation_inv, rope_orientations_world
        )  # 电缆相对于无人机的姿态
        roll_drone, pitch_drone, _ = euler_xyz_from_quat(rope_orientations_drones)
        mapped_angle_drone = torch.stack((torch.cos(roll_drone), torch.cos(pitch_drone)), dim=1)
        self.angle_limit_drone = (
            (mapped_angle_drone < self.cfg.cable_angle_limits_drone).any(dim=1).view(-1, self._num_drones).any(dim=1)
        )

        # 负载端角度限制检查
        self.load_orientation[:] = self.robot.data.body_com_state_w[:, self._payload_idx, 3:7].squeeze(1)
        payload_orientation_world = self.load_orientation.repeat(1, self._num_drones, 1).view(-1, 4)
        payload_orientation_inv = quat_inv(payload_orientation_world)
        rope_orientations_payload = quat_mul(
            payload_orientation_inv, rope_orientations_world
        )  # 电缆相对于负载的姿态
        roll_load, pitch_load, _ = euler_xyz_from_quat(rope_orientations_payload)
        mapped_angle_load = torch.stack((torch.cos(roll_load), torch.cos(pitch_load)), dim=1)
        self.angle_limit_load = (
            (mapped_angle_load < self.cfg.cable_angle_limits_payload).any(dim=1).view(-1, self._num_drones).any(dim=1)
        )

        # 电缆碰撞检测
        self.cable_collision = self._cable_collision(
            self.cfg.cable_collision_threshold, self.cfg.cable_collision_num_points
        )

        # 无人机间碰撞检测
        rpos = get_drone_rpos(self.drone_positions)
        pdist = get_drone_pdist(rpos)
        separation = pdist.min(dim=-1).values.min(dim=-1).values  # 群体中最小的无人机间距
        self.drone_collision = separation < self.cfg.drone_collision_threshold

        # 边界检查：防止飞出限定区域
        self.body_pos_outside = (self.drone_positions.abs() > self.cfg.bounding_box_threshold).any(dim=-1).any(dim=-1)

        # 更新性能指标
        self._update_metrics()

        # 综合终止条件
        terminations = (
            self.falcon_fly_low
            | self.payload_fly_low
            | self.illegal_contact
            | self.angle_limit_drone
            | self.angle_limit_load
            | self.cable_collision
            | self.drone_collision
            | self.body_pos_outside
        )
        self.time_out = self.episode_length_buf >= self.max_episode_length - 1  # 时间超时

        timed_outs = self.time_out

        # 所有智能体共享相同的终止信号
        terminated = {agent: terminations for agent in self.cfg.possible_agents}
        time_outs = {agent: timed_outs for agent in self.cfg.possible_agents}

        return terminated, time_outs

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None):
        """重置指定环境的索引"""
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
            
        # 重置机器人和刚体属性
        super()._reset_idx(env_ids)
        self._reset_target_pose(env_ids)  # 重置目标位姿

        # 重置缓冲区
        for agent in self.cfg.possible_agents:
            # self.setpoint_delay_buffers[agent].reset(env_ids)
            self._observation_buffers[agent].reset(env_ids)

        # 日志记录：统计各类终止原因
        if "log" not in self.extras:
            self.extras["log"] = dict()
            
        self.extras["log"]["Episode_Termination/angle_drones_cable"] = torch.count_nonzero(
            self.angle_limit_drone[env_ids]
        ).item()
        self.extras["log"]["Episode_Termination/angle_load_cable"] = torch.count_nonzero(
            self.angle_limit_load[env_ids]
        ).item()
        self.extras["log"]["Episode_Termination/cables_collide"] = torch.count_nonzero(
            self.cable_collision[env_ids]
        ).item()
        self.extras["log"]["Episode_Termination/drones_collide"] = torch.count_nonzero(
            self.drone_collision[env_ids]
        ).item()
        self.extras["log"]["Episode_Termination/bounding_box"] = torch.count_nonzero(
            self.body_pos_outside[env_ids]
        ).item()
        self.extras["log"]["Episode_Termination/falcon_fly_low"] = torch.count_nonzero(
            self.falcon_fly_low[env_ids]
        ).item()
        self.extras["log"]["Episode_Termination/payload_fly_low"] = torch.count_nonzero(
            self.payload_fly_low[env_ids]
        ).item()
        self.extras["log"]["Episode_Termination/illegal_contact"] = torch.count_nonzero(
            self.illegal_contact[env_ids]
        ).item()
        self.extras["log"]["Episode_Termination/time_out"] = torch.count_nonzero(self.time_out[env_ids]).item()

        # 记录奖励成分的平均值
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            self.extras["log"]["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0

        # 记录性能指标
        for metric_name, metric_value in self.metrics.items():
            self.extras["log"][f"Metrics/pose_command/{metric_name}"] = metric_value.mean()

        # 重置动作历史
        for agent in self.cfg.possible_agents:
            self.prev_actions[agent][env_ids] = 0.0
            self.actions[agent][env_ids] = 0.0

        # if self.common_step_counter > self.cfg.range_curriculum_steps:
        #     self.cfg.goal_range ={
        #     "pos_x": (-2.0, 2.0),
        #     "pos_y": (-2.0, 2.0),
        #     "pos_z": (0.5, 2.5),
        #     "roll": (-math.pi/4, math.pi/4),
        #     "pitch": (-math.pi/4, math.pi/4),
        #     "yaw": (-math.pi, math.pi),
        # }

    def _reset_target_pose(self, env_ids):
        """重置目标位姿：为指定环境生成新的目标位置和姿态"""
        r = torch.empty(len(env_ids), device=self.device)
        
        # 随机生成目标位置
        self.pose_command_w[env_ids, 0] = r.uniform_(*self.cfg.goal_range["pos_x"])
        self.pose_command_w[env_ids, 1] = r.uniform_(*self.cfg.goal_range["pos_y"])
        self.pose_command_w[env_ids, 2] = r.uniform_(*self.cfg.goal_range["pos_z"])
        
        # 随机生成目标姿态（欧拉角）
        euler_angles = torch.zeros_like(self.pose_command_w[env_ids, :3])
        euler_angles[:, 0].uniform_(*self.cfg.goal_range["roll"])
        euler_angles[:, 1].uniform_(*self.cfg.goal_range["pitch"])
        euler_angles[:, 2].uniform_(*self.cfg.goal_range["yaw"])
        quat = quat_from_euler_xyz(euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2])
        
        # 确保四元数实部为正（唯一化表示）
        self.pose_command_w[env_ids, 3:] = quat_unique(quat) if self.cfg.make_quat_unique_command else quat

    def _update_metrics(self):
        """更新性能指标：计算当前位置和姿态误差"""
        pos_error, rot_error = compute_pose_error(
            self.pose_command_w[:, :3],
            self.pose_command_w[:, 3:],
            self.load_position,
            self.load_orientation,
        )
        self.metrics["position_error"] = torch.norm(pos_error, dim=-1)
        self.metrics["orientation_error"] = torch.norm(rot_error, dim=-1)

    def _cable_collision(
        self,
        threshold: float = 0.0,
        num_points: int = 5,
    ) -> torch.Tensor:
        """检测电缆间碰撞
        
        通过在电缆上采样多个点并计算点间最小距离来检测碰撞
        """
        # 计算电缆底部位置
        cable_bottom_pos_env = self.robot.data.body_com_state_w[
            :, self._rope_idx, :3
        ] - self.scene.env_origins.unsqueeze(1)
        cable_directions = self.drone_positions - cable_bottom_pos_env  # 电缆方向向量

        # 在电缆上创建线性插值点
        linspace_points = torch.linspace(0, 1, num_points, device=self.device).view(1, 1, num_points, 1)
        cable_points = cable_bottom_pos_env.unsqueeze(2) + linspace_points * cable_directions.unsqueeze(2)

        # 展平点以便计算距离
        cable_points_flat = cable_points.view(self.num_envs, -1, 3)

        # 计算点对间距离
        cable_points_a = cable_points_flat.unsqueeze(2)
        cable_points_b = cable_points_flat.unsqueeze(1)
        pairwise_diff = cable_points_a - cable_points_b
        pairwise_distances = torch.norm(pairwise_diff, dim=-1)

        # 创建掩码忽略同一条电缆上的点和自距离
        num_cables = cable_bottom_pos_env.shape[1]
        points_per_cable = num_points
        cable_indices = torch.arange(num_cables, device=self.device).repeat_interleave(points_per_cable)
        same_cable_mask = cable_indices.unsqueeze(0) == cable_indices.unsqueeze(1)
        same_cable_mask = same_cable_mask.unsqueeze(0).expand(self.num_envs, -1, -1)

        # 应用掩码：将被忽略的距离设为大值
        pairwise_distances[same_cable_mask] = 1000.0

        # 找到每个环境中点间的最小距离
        min_distances, _ = torch.min(pairwise_distances.view(self.num_envs, -1), dim=-1)

        # 检查最小距离是否低于阈值
        is_cable_collision = min_distances < threshold
        return is_cable_collision

    def _downwash_reward(self):
        """计算下洗流奖励：鼓励无人机避开彼此的下洗气流影响"""
        # 负载平面方程
        x_len_payload_env = quat_apply(self.load_orientation, self.load_length_x)
        y_len_payload_env = quat_apply(self.load_orientation, self.load_length_y)
        edge_payload_x = self.load_position + x_len_payload_env
        edge_payload_y = self.load_position + y_len_payload_env
        plane_vec1 = edge_payload_x - self.load_position
        plane_vec2 = edge_payload_y - self.load_position
        normal = torch.linalg.cross(plane_vec1, plane_vec2)  # 平面法向量
        d = torch.sum(normal * self.load_position, dim=-1).unsqueeze(-1).unsqueeze(-1)

        # 每架无人机推力方向的直线方程
        thrust_directions = quat_apply(
            self.drone_orientations.view(-1, 4),
            torch.tensor([[0, 0, 1.0]] * self.num_envs * self._num_drones, device=self.device),
        ).view(self.num_envs, self._num_drones, 3)

        # 计算与平面的交点
        numerator = d - torch.sum(normal.unsqueeze(1) * self.drone_positions, dim=-1, keepdim=True)
        denominator = torch.sum(normal.unsqueeze(1) * thrust_directions, dim=-1, keepdim=True) + 1e-6
        t = numerator / denominator

        # 平面上的交点
        line_point_proj = self.drone_positions + t * thrust_directions

        # 计算交点到负载位置的距离
        line_dist = torch.norm(line_point_proj - self.load_position.unsqueeze(1), dim=-1)
        
        # 奖励：基于到负载的最小距离进行惩罚
        scaling_factor = 3
        reward_downwash = 1 - torch.exp(-torch.min(line_dist, dim=-1).values * scaling_factor)
        return reward_downwash

    def _set_debug_vis_impl(self, debug_vis: bool):
        """设置调试可视化"""
        if not hasattr(self, "goal_pose_visualizer"):
            # 创建目标位姿和机体位姿的可视化标记
            self.goal_pose_visualizer = VisualizationMarkers(self.cfg.marker_cfg_goal)
            self.body_pose_visualizer = VisualizationMarkers(self.cfg.marker_cfg_body)
            self.goal_pose_visualizer.set_visibility(True)
            self.body_pose_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer.set_visibility(False)
                self.body_pose_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        """调试可视化回调函数"""
        if not self.robot.is_initialized:
            return
        # 更新标记显示
        self.goal_pose_visualizer.visualize(
            self.pose_command_w[:, :3] + self.scene.env_origins, self.pose_command_w[:, 3:]
        )
        self.body_pose_visualizer.visualize(self.load_position + self.scene.env_origins, self.load_orientation)


# JIT编译的辅助函数
@torch.jit.script
def scale(x, lower, upper):
    """将[-1,1]范围的值缩放到[lower, upper]范围"""
    return 0.5 * (x + 1.0) * (upper - lower) + lower

@torch.jit.script
def unscale(x, lower, upper):
    """将[lower, upper]范围的值反缩放到[-1,1]范围"""
    return (2.0 * x - upper - lower) / (upper - lower)

@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    """随机化旋转：通过两个轴的旋转组合生成随机旋转"""
    return quat_mul(
        quat_from_angle_axis(rand0 * np.pi, x_unit_tensor), quat_from_angle_axis(rand1 * np.pi, y_unit_tensor)
    )