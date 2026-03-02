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
import isaacsim.core.utils.prims as prim_utils
from isaacsim.core.prims import XFormPrim
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

# 继承悬停环境作为基础
from MARL_mav_carry_ext.tasks.directMARL.hover.marl_hover_env import MARLHoverEnv

# 导入跟随环境配置
from .marl_flyfollow_env_cfg import MARLFlyFollowEnvCfg


class MARLFlyFollowEnv(DirectMARLEnv):
    """
    多智能体跟随环境类
    
    继承自DirectMARLEnv，在此基础上实现无人机跟随移动目标的任务
    每个目标具有不同的价值，无人机需要智能地选择跟随目标
    """
    
    cfg: MARLFlyFollowEnvCfg # 环境配置对象

    def __init__(self, cfg: MARLFlyFollowEnvCfg, render_mode: str | None = None, **kwargs):
        """初始化跟随环境
        
        Args:
            cfg: 环境配置对象
            render_mode: 渲染模式
            **kwargs: 其他参数
        """
        super().__init__(cfg, render_mode, **kwargs)

        # 目标相关配置
        self._num_targets = cfg.num_targets           # 目标数量
        self._target_values = torch.tensor(cfg.target_values, device=self.device)  # 各目标的价值

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
        self._drone_prev_vel_error = torch.zeros(self.num_envs, self._num_drones, 3, device=self.device)

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

        # 目标状态缓存（在环境坐标系中）
        self._target_positions = torch.zeros(self.num_envs, self._num_targets, 3, device=self.device)    # 位置
        self._target_velocities = torch.zeros(self.num_envs, self._num_targets, 3, device=self.device)  # 速度
        self._target_orientations = torch.zeros(self.num_envs, self._num_targets, 4, device=self.device) # 姿态
        self._target_orientations[..., 0] = 1.0  # 初始化为单位四元数
        self._target_claimed = torch.zeros(self.num_envs, self._num_targets, dtype=torch.bool, device=self.device)  # 是否已被认领
        self._target_assignment = torch.full(
            (self.num_envs, self._num_targets), -1, dtype=torch.long, device=self.device
        )  # 目标分配给哪个无人机

        # 目标沿x轴匀速移动
        self._target_velocities[..., 0] = cfg.target_speed

        # 奖励统计（用于日志记录）
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "target_reward",        # 目标奖励
                "distance_penalty",     # 距离惩罚
                "proximity_reward",     # 接近奖励
            ]
        }

        # 初始化目标位置
        self._reset_targets(torch.arange(self.num_envs, device=self.device))

    def _setup_scene(self):
        """设置场景：在悬停环境基础上添加跟随任务特有的元素"""
        # # 先创建flycrane基础场景（无人机、传感器、地面、灯光）
        # super()._setup_scene()

        # 轨道场景（仅用于可视化）
        track_cfg = sim_utils.UsdFileCfg(
            usd_path=self.cfg.track_usd_path,  # 轨道模型路径
            scale=(0.1, 1.0, 1.0),            # 缩放比例
        )
        sim_utils.spawn_from_usd(
            prim_path="/World/Track",
            cfg=track_cfg,
            translation=(-20.0, -21.0, 0.0),  # 位置
            orientation=(0.7071068, 0.0, 0.0, -0.7071068),  # 朝向（绕y轴旋转90度）
        )

        # 目标小车（仅用于可视化）：使用XFormPrim避免RigidBodyAPI依赖
        # 先创建父级prim，确保正则路径可解析
        prim_utils.create_prim("/World/envs/env_0/targets", "Xform")
        self._target_prims: list[XFormPrim] = []
        for i in range(self.cfg.num_targets):
            prim_path = f"/World/envs/env_0/targets/target_{i}"
            target_cfg = sim_utils.UsdFileCfg(usd_path=self.cfg.target_usd_path)  # 目标模型路径
            sim_utils.spawn_from_usd(prim_path=prim_path, cfg=target_cfg)
            self._target_prims.append(XFormPrim(prim_path))

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        """物理步骤前处理：先执行基础控制，再更新目标位置"""
        # 先执行父类动作处理与控制（无人机控制）
        super()._pre_physics_step(actions)
        # 再更新目标小车位置
        self._update_targets()

    def _get_observations(self) -> dict[str, torch.Tensor]:
        """获取观测值：构建各智能体的局部观测空间"""
        # 获取无人机状态（位置 + 速度）
        self.drone_positions[:] = (
            self.robot.data.body_com_state_w[:, self._falcon_idx, :3] - self.scene.env_origins.unsqueeze(1)
        )
        self.drone_linear_velocities[:] = self.robot.data.body_com_state_w[:, self._falcon_idx, 7:10]

        # 提取XY平面坐标用于2D跟随任务
        drone_pos_xy = self.drone_positions[:, :, :2]
        target_pos_xy = self._target_positions[:, :, :2]

        # 计算无人机到各目标的相对位置与距离（仅XY平面）
        rel_xy = target_pos_xy.unsqueeze(1) - drone_pos_xy.unsqueeze(2)  # (num_envs, num_drones, num_targets, 2)
        dist_xy = torch.norm(rel_xy, dim=-1)  # 各无人机到各目标的XY平面距离

        obs = {}
        for i, agent in enumerate(self.cfg.possible_agents):
            # one-hot标识当前智能体（用于区分不同无人机的观测）
            one_hot = torch.zeros(self.num_envs, self._num_drones, device=self.device)
            one_hot[:, i] = 1.0

            # 当前无人机到各目标的相对位置和距离
            own_rel_xy = rel_xy[:, i].reshape(self.num_envs, -1)  # 展平为(num_envs, num_targets*2)
            own_dist = dist_xy[:, i]  # (num_envs, num_targets)

            # 其他无人机到各目标的距离（用于协作信息）
            other_ids = [j for j in range(self._num_drones) if j != i]
            other_dist = dist_xy[:, other_ids].reshape(self.num_envs, -1)  # (num_envs, (num_drones-1)*num_targets)

            # 目标价值信息
            target_values = self._target_values.unsqueeze(0).repeat(self.num_envs, 1)  # (num_envs, num_targets)

            # 构建观测向量
            obs_t = torch.cat(
                (
                    one_hot,                    # 智能体标识
                    self.drone_positions[:, i], # 本机位置(3维)
                    self.drone_linear_velocities[:, i], # 本机速度(3维)
                    own_rel_xy,                 # 到各目标相对位置(2*num_targets维)
                    own_dist,                   # 到各目标距离(num_targets维)
                    other_dist,                 # 其他无人机到目标距离((num_drones-1)*num_targets维)
                    target_values,              # 目标价值(num_targets维)
                ),
                dim=-1,
            )

            # 存入观测缓冲区并返回展平后的观测
            self._observation_buffers[agent].append(obs_t)
            obs[agent] = self._observation_buffers[agent].buffer.reshape(self.num_envs, -1)

        return obs

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        """计算奖励函数：多目标优化的奖励设计"""
        # 计算奖励所需的无人机位置
        self.drone_positions[:] = (
            self.robot.data.body_com_state_w[:, self._falcon_idx, :3] - self.scene.env_origins.unsqueeze(1)
        )
        drone_pos_xy = self.drone_positions[:, :, :2]
        target_pos_xy = self._target_positions[:, :, :2]

        # 计算相对位置和距离
        rel_xy = target_pos_xy.unsqueeze(1) - drone_pos_xy.unsqueeze(2)
        dist_xy = torch.norm(rel_xy, dim=-1)  # (num_envs, num_drones, num_targets)

        # 距离相关统计
        mean_dist = dist_xy.mean(dim=-1)      # 平均距离 (num_envs, num_drones)
        min_dist = dist_xy.min(dim=-1).values # 最近距离 (num_envs, num_drones)

        # 距离惩罚：鼓励无人机靠近目标
        distance_penalty = self.cfg.distance_penalty_weight * (
            self.cfg.mean_distance_weight * mean_dist + self.cfg.min_distance_weight * min_dist
        )
        
        # 接近奖励：指数衰减形式，距离越近奖励越高
        proximity_reward = self.cfg.proximity_reward_weight * torch.exp(-min_dist / self.cfg.proximity_sigma)

        # 目标奖励：当无人机首次成功跟踪目标时获得相应价值奖励
        target_reward = torch.zeros_like(mean_dist)  # 初始化为0

        # 找到每个目标最近的无人机
        closest_dist, closest_drone = dist_xy.min(dim=1)  # (num_envs, num_targets)
        # 判断是否有无人机在跟踪距离内
        has_tracker = closest_dist <= self.cfg.track_distance_xy
        # 找到新被认领的目标（之前未被认领但现在被跟踪）
        newly_claimed = has_tracker & (~self._target_claimed)

        # 为新认领的目标分配奖励
        if newly_claimed.any():
            for t in range(self._num_targets):
                claim_envs = newly_claimed[:, t]
                if not claim_envs.any():
                    continue
                # 获取负责跟踪该目标的无人机ID
                drone_ids = closest_drone[claim_envs, t]
                # 给对应的无人机分配目标价值奖励
                target_reward[claim_envs, drone_ids] += self._target_values[t]
                # 标记目标已被认领并记录分配关系
                self._target_claimed[claim_envs, t] = True
                self._target_assignment[claim_envs, t] = drone_ids

        # 总奖励：目标奖励 + (接近奖励 - 距离惩罚)
        total_reward = target_reward + (proximity_reward - distance_penalty) * self.step_dt

        # 累计奖励用于日志记录
        # 记录每个环境的总目标奖励（对 3 个无人机求和）
        self._episode_sums["target_reward"] += target_reward.sum(dim=-1)
        # 记录每个环境的距离惩罚与接近奖励（对 3 个无人机求和）
        self._episode_sums["distance_penalty"] += distance_penalty.sum(dim=-1)
        self._episode_sums["proximity_reward"] += proximity_reward.sum(dim=-1)

        # 为每个智能体分配对应的奖励
        rewards = {}
        for i, agent in enumerate(self.cfg.possible_agents):
            rewards[agent] = total_reward[:, i]
        return rewards

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """判断终止条件"""
        # 获取无人机位置用于终止判断
        self.drone_positions[:] = (
            self.robot.data.body_com_state_w[:, self._falcon_idx, :3] - self.scene.env_origins.unsqueeze(1)
        )

        # 高度过低终止：防止撞击地面
        falcon_fly_low = (self.drone_positions[:, :, 2] < self.cfg.min_altitude).any(dim=-1)
        
        # 越界终止：防止飞出限定区域
        body_pos_outside = (self.drone_positions.abs() > self.cfg.bounding_box_threshold).any(dim=-1).any(dim=-1)

        # 时间超时
        self.time_out = self.episode_length_buf >= self.max_episode_length - 1

        # 综合终止条件
        terminations = falcon_fly_low | body_pos_outside
        timed_outs = self.time_out

        # 所有智能体共享相同终止信号
        terminated = {agent: terminations for agent in self.cfg.possible_agents}
        time_outs = {agent: timed_outs for agent in self.cfg.possible_agents}

        return terminated, time_outs

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None):
        """重置指定环境索引"""
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        # 重置articulation和rigid body属性
        from isaaclab.envs import DirectMARLEnv

        # 基类重置（包含场景和事件）
        DirectMARLEnv._reset_idx(self, env_ids)

        # 重置目标状态
        self._reset_targets(env_ids)

        # 重置观测缓冲区和动作历史
        for agent in self.cfg.possible_agents:
            self._observation_buffers[agent].reset(env_ids)
            self.prev_actions[agent][env_ids] = 0.0
            self.actions[agent][env_ids] = 0.0

        # 日志记录初始化
        if "log" not in self.extras:
            self.extras["log"] = dict()

        # 记录终止原因统计
        self.drone_positions[:] = (
            self.robot.data.body_com_state_w[:, self._falcon_idx, :3] - self.scene.env_origins.unsqueeze(1)
        )
        self.extras["log"]["Episode_Termination/falcon_fly_low"] = torch.count_nonzero(
            (self.drone_positions[:, :, 2] < self.cfg.min_altitude).any(dim=-1)[env_ids]
        ).item()
        self.extras["log"]["Episode_Termination/bounding_box"] = torch.count_nonzero(
            (self.drone_positions.abs() > self.cfg.bounding_box_threshold).any(dim=-1).any(dim=-1)[env_ids]
        ).item()
        self.extras["log"]["Episode_Termination/time_out"] = torch.count_nonzero(self.time_out[env_ids]).item()

        # 记录奖励成分平均值
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            self.extras["log"]["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0

    def _reset_targets(self, env_ids: torch.Tensor):
        """重置目标位置：为指定环境生成新的目标起始位置"""
        if env_ids.numel() == 0:
            return

        # 初始化目标位置（起点x坐标 + 固定的y坐标分布）
        target_y = torch.tensor(self.cfg.target_y_positions, device=self.device)  # 各目标的y坐标
        self._target_positions[env_ids, :, 0] = self.cfg.target_start_x  # 所有目标起始x坐标相同
        self._target_positions[env_ids, :, 1] = target_y.unsqueeze(0).repeat(env_ids.numel(), 1)  # y坐标分布
        self._target_positions[env_ids, :, 2] = 0.0  # z坐标为0（地面高度）

        # 重置目标状态
        self._target_claimed[env_ids] = False      # 未被认领
        self._target_assignment[env_ids] = -1      # 未分配

        # 将目标位置写入场景（XFormPrim可视化）
        # 注意：仅 env_0 有可视化 prim，因此只在 env_0 被重置时更新可视化
        if 0 in env_ids.tolist():
            positions_world = self._target_positions[0:1] + self.scene.env_origins[0:1].unsqueeze(1)
            for i, prim in enumerate(self._target_prims):
                pos = positions_world[:, i, :]
                ori = self._target_orientations[0:1, i, :]
                prim.set_world_poses(positions=pos, orientations=ori)

    def _update_targets(self):
        """更新目标位置：实现目标的匀速移动"""
        # 目标沿x轴匀速移动
        self._target_positions[..., 0] += self._target_velocities[..., 0] * self.step_dt
        
        # 边界处理：到达终点后回到起点
        overflow = self._target_positions[..., 0] > self.cfg.target_end_x
        if overflow.any():
            self._target_positions[..., 0] = torch.where(
                overflow,
                torch.full_like(self._target_positions[..., 0], self.cfg.target_start_x),
                self._target_positions[..., 0],
            )
            # 到达终点后回到起点，允许重新奖励
            self._target_claimed = torch.where(
                overflow, torch.zeros_like(self._target_claimed), self._target_claimed
            )
            self._target_assignment = torch.where(
                overflow, torch.full_like(self._target_assignment, -1), self._target_assignment
            )

        # 可视化更新（仅更新env_0环境以提高性能）
        positions_world = self._target_positions + self.scene.env_origins.unsqueeze(1)
        for i, prim in enumerate(self._target_prims):
            pos = positions_world[0:1, i, :]
            ori = self._target_orientations[0:1, i, :]
            # 为性能考虑仅更新第一个环境的可视化
            prim.set_world_poses(positions=pos, orientations=ori)


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
