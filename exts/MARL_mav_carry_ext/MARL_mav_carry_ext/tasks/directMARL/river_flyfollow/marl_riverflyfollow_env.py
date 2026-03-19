# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import torch
from collections.abc import Sequence
from pathlib import Path

# 导入控制器模块
from MARL_mav_carry_ext.controllers import GeometricController, IndiController
from MARL_mav_carry_ext.controllers.motor_model import RotorMotor
# 导入低层控制工具函数
# 导入IsaacLab相关模块
import isaaclab.sim as sim_utils
import isaacsim.core.utils.prims as prim_utils
from isaacsim.core.prims import XFormPrim
from isaaclab.assets import Articulation
from isaaclab.envs import DirectMARLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from pxr import UsdPhysics
from isaaclab.utils import CircularBuffer
from isaaclab.utils.math import (
    matrix_from_quat,
    quat_from_angle_axis,
    quat_mul,
)

# 导入跟随环境配置
from .marl_riverflyfollow_env_cfg import MARLRiverFlyFollowEnvCfg


class MARLRiverFlyFollowEnv(DirectMARLEnv):
    """
    多智能体 River 场景跟随环境类

    继承自DirectMARLEnv，在此基础上实现无人机跟随移动目标的任务
    每个目标具有不同的价值，无人机需要智能地选择跟随目标
    """

    cfg: MARLRiverFlyFollowEnvCfg  # 环境配置对象

    def __init__(self, cfg: MARLRiverFlyFollowEnvCfg, render_mode: str | None = None, **kwargs):
        """初始化跟随环境
        
        Args:
            cfg: 环境配置对象
            render_mode: 渲染模式
            **kwargs: 其他参数
        """
        super().__init__(cfg, render_mode, **kwargs)

        # 多无人机与控制模式配置
        self._num_drones = cfg.num_drones
        self._control_mode = cfg.control_mode

        # 目标相关配置
        self._num_targets = cfg.num_targets           # 目标数量
        self._target_values = torch.tensor(cfg.target_values, device=self.device)  # 各目标的价值

        # 每台无人机的机体与旋翼索引
        self._base_body_ids = []
        self._rotor_body_ids = []
        for robot in self.robots:
            self._base_body_ids.append(
                torch.tensor(robot.find_bodies(".*base_link_inertia")[0], device=self.device)
            )
            self._rotor_body_ids.append(
                torch.tensor(robot.find_bodies(".*rotor_.*")[0], device=self.device)
            )
        self._rotor_forces = [
            torch.zeros(self.num_envs, len(rotor_ids), 3, device=self.device)
            for rotor_ids in self._rotor_body_ids
        ]

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
            action_dim = self.cfg.action_spaces[agent]
            self.prev_actions[agent] = torch.zeros(self.num_envs, action_dim, device=self.device)

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
        for i in range(self.cfg.num_drones):
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
        self._target_velocities = torch.zeros(self.num_envs, self._num_targets, 3, device=self.device)   # 速度
        self._target_orientations = torch.zeros(self.num_envs, self._num_targets, 4, device=self.device) # 姿态
        self._target_orientations[..., 0] = 1.0                                                          # 初始化为单位四元数
        self._target_claimed = torch.zeros(self.num_envs, self._num_targets, dtype=torch.bool, device=self.device)  # 是否已被认领
        self._target_assignment = torch.full(
            (self.num_envs, self._num_targets), -1, dtype=torch.long, device=self.device
        )  # 目标分配给哪个无人机

        # 目标沿x轴匀速移动
        self._target_velocities[..., 0] = cfg.target_speed

        # 势函数距离进度奖励：记录上一步各无人机到分配目标的距离
        self._prev_assigned_dist = torch.full(
            (self.num_envs, self._num_drones), 100.0, device=self.device
        )

        # 奖励统计（用于日志记录）
        self._episode_sums = {
            key: torch.zeros(
                self.num_envs,
                dtype=torch.float,
                device=self.device,
            )
            for key in [
                # ── 主奖励项 ──────────────────────────────────────────────
                "distance_reward",          # 高斯距离奖励：exp(-(d/σ)²)，σ=10，近场梯度
                "dist_progress_reward",     # 距离进度奖励（势函数 shaping）：每步缩短距离即得正值
                "tracking_reward",          # 追踪区奖励：entry（进入即有）+ holding（持续保持线性增长）
                "velocity_follow_reward",   # 速度跟随奖励：vel_match + 前向进度 - 超速惩罚
                "success_proximity_reward", # 成功区密集奖励：同时满足 dist+vel+height 三个 success 条件时每步发放
                "success_bonus",            # 成功终止奖励：_sustained_follow_timer 越过阈值时一次性大额奖励
                # ── 辅助约束项 ────────────────────────────────────────────
                "action_smoothness",        # 动作平滑奖励：抑制相邻帧动作突变，防抖
                "body_rate_penalty",        # 机体角速率惩罚：‖ω‖，辅助约束，防止过度翻滚
                "velocity_penalty",         # 速度惩罚：XY/Z 超出安全阈值后按比例扣分
                "force_penalty",            # 推力惩罚：总推力过大时扣分，节约能量
                # ── 高度相关项 ────────────────────────────────────────────
                "height_reward",            # 高度奖励：贴近 desired_height 的高斯奖励（当前已关闭，weight=0）
                "height_error_penalty",     # 高度误差惩罚（当前已关闭，weight=0）
                "vertical_direction_penalty", # 垂直方向速度惩罚（当前已关闭，weight=0）
                "upward_vz_penalty",        # 上升速度惩罚：仅 vz>0 时生效，抑制起步无意义上窜
                "high_altitude_penalty",    # 超高软惩罚：z > altitude_upper_soft_threshold 时线性增大
                # ── 综合安全项 ────────────────────────────────────────────
                "safety_penalty",           # 安全惩罚合计：碰撞软惩罚 + 边界软惩罚 + 低高度软惩罚 + 超高惩罚
            ]
        }

        # 性能指标
        self.metrics = {}
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["orientation_error"] = torch.zeros(
            self.num_envs, device=self.device
        )
        # 临时调试：奖励 batch mean 打印
        self._reward_debug_print_interval = 500
        self._reward_debug_counter = 0

        # ── 终止条件缓冲区 ────────────────────────────────────────────────────
        # 每个 bool 张量形状均为 (num_envs,)，在 _get_dones() 中更新，
        # 任一条件为 True 时该 env 的 episode 提前结束。
        self.falcon_fly_low = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )  # 无人机飞行高度低于 min_altitude，触发硬终止（撞地风险）
        self.illegal_contact = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )  # 接触传感器检测到非法碰撞（与障碍物或地面）
        self.drone_collision = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )  # 无人机间距离低于 drone_collision_threshold，相互碰撞
        self.body_pos_outside = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )  # 无人机飞出边界框（bounding_box_threshold），超出允许飞行区域
        # 所有目标都被捕获的标志（当前任务中暂不启用提前终止）
        self.all_targets_captured = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )  # 四个目标全部被持续跟随成功时置 True（当前由 sustained_success 触发终止）
        # # 持续跟随计时器
        # self._sustained_follow_timer = torch.zeros(self.num_envs, device=self.device)
        self.targets_out_of_bounds = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.time_out = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        # 归一化配置参数
        self._norm_pos_scale = self.cfg.bounding_box_threshold
        self._norm_vel_scale = 5.0
        self._norm_dist_scale = self.cfg.bounding_box_threshold * 2
        self._norm_ang_vel_scale = 10.0
        self._norm_target_value_scale = torch.clamp(self._target_values.max(), min=1.0)

        # 每个 episode 固定的 target assignment（每架无人机一个目标）
        self._assigned_target_idx = torch.zeros(
            (self.num_envs, self._num_drones), dtype=torch.long, device=self.device
        )

        # 持续跟随计时器（按 env、drone）
        self._sustained_follow_timer = torch.zeros(
            (self.num_envs, self._num_drones), dtype=torch.float, device=self.device
        )

        # 近距稳定跟随计时器（tracking_reward 持续时间加成用）
        # 条件：dist < tracking_distance_xy AND vel_err < 2*tracking_vel_match_sigma
        self._tracking_stable_timer = torch.zeros(
            (self.num_envs, self._num_drones), dtype=torch.float, device=self.device
        )

        # 初始化目标位置
        self._reset_targets(torch.arange(self.num_envs, device=self.device))

        # 调试可视化设置
        self.set_debug_vis(cfg.debug_vis)

    def _assign_targets_for_envs(self, env_ids: torch.Tensor) -> None:
        """为指定环境固定每架无人机的目标，保持一个 episode 内不变。

        采用贪心最近距离匹配：每轮找当前最近的（无人机, 目标）对完成配对，
        直到无人机或目标耗尽。目标数多于无人机数时，多余目标不分配。
        结果写入 _assigned_target_idx[env_id, drone_idx] = target_idx。
        """
        if env_ids.numel() == 0:
            return

        # 取当前帧无人机和目标的 XY 坐标，形状 (len(env_ids), num_drones/targets, 2)
        drone_pos_xy = self.drone_positions[env_ids, :, :2]
        target_pos_xy = self._target_positions[env_ids, :, :2]

        # 初始化：所有无人机默认指向 target 0，所有目标标记为未分配（-1）
        self._assigned_target_idx[env_ids] = 0
        self._target_assignment[env_ids] = -1

        for local_env_idx, env_id in enumerate(env_ids.tolist()):
            # dist[drone_idx, target_idx]：该环境内每对（无人机, 目标）的 XY 欧氏距离
            # unsqueeze 分别扩维以广播：drone (D,1,2)，target (1,T,2) → (D,T,2) → norm → (D,T)
            dist = torch.norm(
                target_pos_xy[local_env_idx].unsqueeze(0) - drone_pos_xy[local_env_idx].unsqueeze(1),
                dim=-1,
            )
            free_drones = set(range(self._num_drones))    # 尚未配对的无人机索引集合
            free_targets = set(range(self._num_targets))  # 尚未配对的目标索引集合

            # 贪心匹配：每轮在所有剩余（无人机, 目标）对中选最近的完成配对
            while free_drones and free_targets:
                best_pair = None
                best_dist = None
                for drone_idx in free_drones:
                    for target_idx in free_targets:
                        value = float(dist[drone_idx, target_idx].item())
                        if best_dist is None or value < best_dist:
                            best_dist = value
                            best_pair = (drone_idx, target_idx)

                drone_idx, target_idx = best_pair
                # 写入双向映射：无人机→目标，目标→无人机
                self._assigned_target_idx[env_id, drone_idx] = target_idx
                self._target_assignment[env_id, target_idx] = drone_idx
                free_drones.remove(drone_idx)
                free_targets.remove(target_idx)

    def _setup_scene(self):
        """设置场景：在悬停环境基础上添加跟随任务特有的元素"""
        # ── 1. 生成地面平面（物理碰撞用，不可见）────────────────────────────
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # ── 2. 加载 River 场景 USD ──────────────────────────────────────────
        # river_flyfollow.usda 内含：Rivermark 室外环境 + 4 辆 NovaCarter 目标小车 + 3 架 Falcon 无人机
        # 挂载到 env_0 子树下，之后由 clone_environments 自动复制到所有并行环境
        flyfollow_scene_path = (
            Path(__file__).resolve().parents[3] / "assets/data/AMR/river_flyfollow/river_flyfollow.usda"
        )
        scene_cfg = sim_utils.UsdFileCfg(usd_path=str(flyfollow_scene_path))
        sim_utils.spawn_from_usd(prim_path="/World/envs/env_0/World", cfg=scene_cfg)

        # ── 3. 克隆 env_0 到其他并行环境 ────────────────────────────────────
        # copy_from_source=False：基于已创建的 env_0 实例进行复制，而非重新从 USD 加载
        self.scene.clone_environments(copy_from_source=False)

        # ── 4. 解析 env_0 的实际 prim 根路径 ────────────────────────────────
        # 某些 USD 会在 env_0 下多嵌一层 "World" 节点，需要动态判断
        env_root_base = "/World/envs/env_0"
        if not prim_utils.is_prim_path_valid(env_root_base):
            raise RuntimeError(f"Expected environment root prim at {env_root_base}, but it does not exist.")
        env_root = env_root_base
        if prim_utils.is_prim_path_valid(f"{env_root_base}/World"):
            env_root = f"{env_root_base}/World"  # 存在 World 子节点时，以其为实际根

        def resolve_agent_prim_path(agent_name: str) -> str:
            """在 env_0 下定位指定 agent（如 "falcon"）真正带 ArticulationRootAPI 的 prim 路径。

            查找策略（按优先级）：
            1. 直接路径：<root>/<agent>/Robot、<root>/<agent>/Falcon、<root>/<agent>
            2. Instanceable prim（instanceable=true）：内层子 prim 对 is_prim_path_valid 不可见，
               通过 USD prototype 遍历，找到带 ArticulationRootAPI 的子节点，
               返回 instance proxy 路径（<outer_path>/<child_name>）
            3. 模糊名称匹配：get_all_matching_child_prims 按名称精确/包含匹配
            4. 全局正则搜索：find_first_matching_prim 在整个 env_0.* 子树内搜索
            """
            # 候选根路径：env_root 本身，以及可能存在的双层 World/env_0 嵌套
            roots = [env_root]
            if prim_utils.is_prim_path_valid(f"{env_root}/World"):
                roots.append(f"{env_root}/World")
            if prim_utils.is_prim_path_valid(f"{env_root}/env_0"):
                roots.append(f"{env_root}/env_0")

            for root in roots:
                # 策略 1：尝试常见的内层 prim 命名规范
                candidate_paths = [
                    f"{root}/{agent_name}/Robot",   # Isaac Lab 标准：外层 Xform + 内层 Robot
                    f"{root}/{agent_name}/Falcon",  # Falcon 资产特有命名
                    f"{root}/{agent_name}",         # 外层 Xform 本身即 ArticulationRoot
                ]
                for path in candidate_paths:
                    if prim_utils.is_prim_path_valid(path):
                        return path

                # 策略 2：instanceable prim 处理
                # Isaac Sim 对 payload/reference 引入的 Xform 可能自动加 instanceable=true，
                # 此时 prim 的子节点路径对 is_prim_path_valid 返回 False，
                # 需通过 GetPrototype() 获取 master prim，遍历其子节点找 ArticulationRootAPI
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
                                    # 返回 instance proxy 路径（而非 prototype 路径）
                                    return f"{outer_path}/{child.GetName()}"

                # 策略 3：模糊名称匹配（精确 → 包含）
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

            # 策略 4：全局正则搜索，覆盖所有并行 env 子树（兜底）
            prim = sim_utils.find_first_matching_prim(f"{env_root_base}.*/{agent_name}(/.*)?")
            if prim is None:
                prim = sim_utils.find_first_matching_prim(f"{env_root_base}.*/.*[Ff]alcon.*")
            if prim is not None:
                return prim.GetPath().pathString

            # 所有策略均失败：收集直接子节点名称以辅助调试
            child_names = []
            try:
                children = sim_utils.get_all_matching_child_prims(env_root_base, depth=1)
                child_names = [child.GetName() for child in children if child.GetName()]
            except Exception:
                child_names = []
            raise RuntimeError(
                f"Could not resolve prim path for agent {agent_name} under {env_root_base}. "
                f"Direct children: {child_names}"
            )

        # ── 5. 创建多台无人机 Articulation 对象 ─────────────────────────────
        # spawn=None：不重新生成 prim，而是绑定到 USD 中已有的 ArticulationRoot prim
        # robot_cfg.prim_path 使用 env_.* 通配符，Isaac Lab 会自动映射到所有并行环境
        self.robots = []
        self._usd_root_state_rel = torch.zeros(
            len(self.cfg.possible_agents), 7, device=self.device
        )  # 缓存每架无人机在 env_0 中的初始位姿（相对 env_origin），用于 reset 时恢复
        for i, agent in enumerate(self.cfg.possible_agents):
            # 定位 env_0 下该 agent 的真实 prim 路径
            env0_agent_prim = resolve_agent_prim_path(agent)

            # 将 env_0 路径替换为 env_.* 通配符，构造跨所有环境的 prim_path 模式
            if env0_agent_prim.startswith(env_root_base):
                env_agent_prim_pattern = env0_agent_prim.replace(
                    env_root_base, "/World/envs/env_.*", 1
                )
            elif "/env_0" in env0_agent_prim:
                env_agent_prim_pattern = env0_agent_prim.replace("/env_0", "/env_.*", 1)
            else:
                raise RuntimeError(
                    f"Resolved prim path {env0_agent_prim} does not include env_0; cannot build pattern."
                )

            robot_cfg = self.cfg.robot_cfg.replace(prim_path=env_agent_prim_pattern)
            robot_cfg.spawn = None  # 不重新生成，绑定已有 prim
            robot = Articulation(robot_cfg)
            self.robots.append(robot)
            self.scene.articulations[f"robot_{i}"] = robot  # 注册到 scene，确保物理更新

            # 缓存 USD 初始位姿（相对于 env_0 原点），reset 时用于 usd_fixed/usd_perturbed 模式
            try:
                prim = XFormPrim(env0_agent_prim)
                pos, ori = prim.get_world_poses()
                self._usd_root_state_rel[i, :3] = pos[0] - self.scene.env_origins[0]
                self._usd_root_state_rel[i, 3:7] = ori[0]
            except Exception as exc:
                print(f"[flyfollow] Failed to read USD pose for {env0_agent_prim}: {exc}")

        # scene.articulations["robot"] 需指向至少一个机器人，用于 contact sensor 等基础接口
        if self.robots:
            self.scene.articulations["robot"] = self.robots[0]

        # ── 6. 添加环境光照 ──────────────────────────────────────────────────
        # USD 内已有 DistantLight，此处再添加 DomeLight 补充环境亮度
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # ── 7. 绑定目标小车 XFormPrim 并缓存初始位姿 ────────────────────────
        # NovaCarter 小车不作为 Articulation 控制，仅作为可视化/位置参考对象（XFormPrim）
        # 每步通过 write_root_state_to_sim / set_world_poses 直接驱动其位置
        self._target_prims: list[XFormPrim] = []
        self._usd_target_positions_rel = torch.zeros(self.cfg.num_targets, 3, device=self.device)
        self._usd_target_orientations = torch.zeros(self.cfg.num_targets, 4, device=self.device)
        # USD 中小车 prim 命名约定：第一辆无后缀，其余依次加 _01/_02/_03
        target_names = [
            "nova_carter_sim_optimized",        # 目标 0（最高价值，红色）
            "nova_carter_sim_optimized_01",     # 目标 1（次高价值，黄色）
            "nova_carter_sim_optimized_02",     # 目标 2（中等价值，绿色）
            "nova_carter_sim_optimized_03",     # 目标 3（最低价值，蓝色）
        ]
        for name in target_names[: self.cfg.num_targets]:
            prim_path = f"{env_root}/{name}"
            prim = XFormPrim(prim_path)
            self._target_prims.append(prim)
            # 缓存 USD 初始位姿（相对 env_origin），reset 时作为起始点基准
            try:
                pos, ori = prim.get_world_poses()
                idx = len(self._target_prims) - 1
                self._usd_target_positions_rel[idx, :] = pos[0] - self.scene.env_origins[0]
                self._usd_target_orientations[idx, :] = ori[0]
            except Exception as exc:
                print(f"[flyfollow] Failed to read USD target pose for {prim_path}: {exc}")

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        """物理步骤前处理：解析动作并更新目标位置"""
        # 保存上一时刻动作并更新当前动作
        for agent in self.cfg.possible_agents:
            self.prev_actions[agent][:] = self.actions[agent]
            self.actions[agent][:] = actions[agent]

        # 解析每个无人机动作
        for drone, action in actions.items():
            if self._control_mode == "geometric":
                # 几何控制模式：12维动作 [pos, lin_vel, lin_acc, jerk]
                self._setpoints[drone]["pos"] = action[:, :3]
                self._setpoints[drone]["lin_vel"] = action[:, 3:6]
                self._setpoints[drone]["lin_acc"] = action[:, 6:9]
                self._setpoints[drone]["jerk"] = action[:, 9:12]
            elif self._control_mode == "ACCBR":
                # ACCBR模式：优先支持6维 [lin_acc(3) + body_rates(3)]
                lin_acc = action[:, :3]
                # 限制 az 正向分量，抑制起步上窜（geometric controller 已补重力，az>0 = 主动上升）
                az_scale = getattr(self.cfg, "upward_acc_z_scale", 1.0)
                if az_scale < 1.0:
                    az = lin_acc[:, 2:3]
                    lin_acc = torch.cat(
                        [lin_acc[:, :2], torch.where(az > 0, az * az_scale, az)], dim=-1
                    )
                self._setpoints[drone]["lin_acc"] = lin_acc
                if action.shape[-1] >= 6:
                    self._setpoints[drone]["body_rates"] = action[:, 3:6]
                else:
                    # 兼容5维：body_rates仅xy，z保持恒定
                    self._setpoints[drone]["body_rates"] = torch.cat(
                        (action[:, 3:], self._constant_yaw), dim=-1
                    )

            # 维持恒定偏航角设定
            self._setpoints[drone]["yaw"] = self._constant_yaw
            self._setpoints[drone]["yaw_rate"] = self._constant_yaw
            self._setpoints[drone]["yaw_acc"] = self._constant_yaw

        # 更新目标小车位置
        self._update_targets()

    def _apply_action(self) -> None:
        """应用控制动作：计算并施加推力和力矩到无人机"""
        if self._ll_counter % self.cfg.low_level_decimation == 0:
            # 读取无人机当前状态
            for i, robot in enumerate(self.robots):
                root_state = robot.data.root_state_w
                self.drone_positions[:, i] = root_state[:, :3] - self.scene.env_origins
                self.drone_orientations[:, i] = root_state[:, 3:7]
                self.drone_linear_velocities[:, i] = root_state[:, 7:10]
                self.drone_angular_velocities[:, i] = root_state[:, 10:13]

                body_acc = robot.data.body_acc_w
                self.drone_linear_accelerations[:, i] = body_acc[:, 0, :3]
                self.drone_angular_accelerations[:, i] = body_acc[:, 0, 3:6]

            # 计算每架无人机控制指令
            for i in range(self._num_drones):
                drone_states: dict = {}
                drone_states["pos"] = self.drone_positions[:, i]
                drone_states["quat"] = self.drone_orientations[:, i]
                drone_states["lin_vel"] = self.drone_linear_velocities[:, i]
                drone_states["ang_vel"] = self.drone_angular_velocities[:, i]
                drone_states["lin_acc"] = self.drone_linear_accelerations[:, i]
                drone_states["ang_acc"] = self.drone_angular_accelerations[:, i]

                # jerk
                self._drone_jerk[:, i] = (drone_states["lin_acc"] - self._drone_prev_acc[:, i]) / (self.step_dt)
                drone_states["jerk"] = self._drone_jerk[:, i]
                self._drone_prev_acc[:, i] = drone_states["lin_acc"]

                # 外环几何控制器
                agent = self.cfg.possible_agents[i]
                alpha_cmd, acc_load, acc_cmd, q_cmd = self.geo_controllers[i].getCommand(
                    drone_states, self._forces[i], self._setpoints[agent]
                )

                # 内环INDI控制器
                target_rpm = self._indi_controllers[i].getCommand(
                    drone_states, self._forces[i], alpha_cmd, acc_cmd, acc_load
                )

                # 电机模型
                thrusts, moments = self.motor_models[i].get_motor_thrusts_moments(target_rpm, self.sampling_time)
                self._forces[i][..., 2] = thrusts

                self._rotor_forces[i][:] = 0.0
                self._rotor_forces[i][..., 2] = thrusts

                self._moments[:, i, :] = 0.0
                self._moments[:, i, 2] = moments.sum(-1)

            self._ll_counter = 0

        # 每个物理步都向仿真施加当前推力/力矩（与可视化解耦）
        for i, robot in enumerate(self.robots):
            body_torque = self._moments[:, i : i + 1, :]
            robot.set_external_force_and_torque(
                torch.zeros_like(body_torque),
                body_torque,
                body_ids=self._base_body_ids[i],
            )
            robot.set_external_force_and_torque(
                self._rotor_forces[i],
                torch.zeros_like(self._rotor_forces[i]),
                body_ids=self._rotor_body_ids[i],
            )

        self._ll_counter += 1

    def _set_debug_vis_impl(self, debug_vis: bool):
        """设置调试可视化"""
        if not debug_vis:
            return

        # Create simple shape prims for visualization to avoid PointInstancer warnings
        if not hasattr(self, "_vis_root"):
            self._vis_root = "/World/envs/env_0/Visuals/FlyFollow"
        if not prim_utils.is_prim_path_valid(self._vis_root):
            prim_utils.create_prim(self._vis_root, "Xform")

        targets_root = f"{self._vis_root}/targets"
        drones_root = f"{self._vis_root}/drones"
        if not prim_utils.is_prim_path_valid(targets_root):
            prim_utils.create_prim(targets_root, "Xform")
        if not prim_utils.is_prim_path_valid(drones_root):
            prim_utils.create_prim(drones_root, "Xform")

        # Cache visualization prims
        if not hasattr(self, "_target_vis_prims"):
            self._target_vis_prims = []
            sphere_cfg = sim_utils.SphereCfg(radius=0.15)
            for i in range(self.cfg.num_targets):
                prim_path = f"{targets_root}/target_{i}"
                if not prim_utils.is_prim_path_valid(prim_path):
                    sim_utils.spawn_sphere(prim_path=prim_path, cfg=sphere_cfg)
                self._target_vis_prims.append(XFormPrim(prim_path))

        if not hasattr(self, "_drone_vis_prims"):
            self._drone_vis_prims = []
            cuboid_cfg = sim_utils.CuboidCfg(size=(0.25, 0.25, 0.06))
            for i in range(self._num_drones):
                prim_path = f"{drones_root}/drone_{i}"
                if not prim_utils.is_prim_path_valid(prim_path):
                    sim_utils.spawn_cuboid(prim_path=prim_path, cfg=cuboid_cfg)
                self._drone_vis_prims.append(XFormPrim(prim_path))

    def _debug_vis_callback(self, event):
        """调试可视化回调函数"""
        if not self.robots:
            return
        if not self.robots[0].is_initialized:
            return
        if not hasattr(self, "_target_vis_prims") or not hasattr(self, "_drone_vis_prims"):
            return

        # 仅显示 env_0 的目标与无人机
        target_pos = self._target_positions[0] + self.scene.env_origins[0]
        target_ori = self._target_orientations[0]
        for i, prim in enumerate(self._target_vis_prims):
            prim.set_world_poses(
                positions=target_pos[i : i + 1],
                orientations=target_ori[i : i + 1],
            )

        for i, robot in enumerate(self.robots):
            root_state = robot.data.root_state_w
            pos = root_state[0:1, :3]
            ori = root_state[0:1, 3:7]
            self._drone_vis_prims[i].set_world_poses(positions=pos, orientations=ori)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        """获取观测值：构建各智能体的局部观测空间"""
        # 获取无人机状态（位置 + 姿态 + 速度）
        for i, robot in enumerate(self.robots):
            root_state = robot.data.root_state_w
            self.drone_positions[:, i] = root_state[:, :3] - self.scene.env_origins
            self.drone_linear_velocities[:, i] = root_state[:, 7:10]
            self.drone_linear_velocities[:, i] = root_state[:, 7:10]
            self.drone_orientations[:, i] = root_state[:, 3:7]
            self.drone_linear_velocities[:, i] = root_state[:, 7:10]
            self.drone_angular_velocities[:, i] = root_state[:, 10:13]

        # 计算旋转矩阵
        self.drone_rot_matrices[:] = matrix_from_quat(self.drone_orientations)

        # 提取XY平面坐标用于2D跟随任务，以及Z高度和Z速度用于高度控制
        drone_pos_xy = self.drone_positions[:, :, :2]
        drone_pos_z = self.drone_positions[:, :, 2:3]              # (E, D, 1)，高度
        drone_lin_vel_xy = self.drone_linear_velocities[:, :, :2]
        drone_lin_vel_z = self.drone_linear_velocities[:, :, 2:3]  # (E, D, 1)，垂直速度
        target_pos_xy = self._target_positions[:, :, :2]
        target_vel_xy = self._target_velocities[:, :, :2]

        # 计算无人机到各目标的相对位置与距离（仅XY平面）
        rel_xy = target_pos_xy.unsqueeze(1) - drone_pos_xy.unsqueeze(2)  # (num_envs, num_drones, num_targets, 2)
        dist_xy = torch.norm(rel_xy, dim=-1)  # 各无人机到各目标的XY平面距离
        closest_drone = dist_xy.argmin(dim=1)  # (num_envs, num_targets)
        closest_drone_one_hot = torch.zeros(
            self.num_envs, self._num_targets, self._num_drones, device=self.device
        )
        closest_drone_one_hot.scatter_(2, closest_drone.unsqueeze(-1), 1.0)
        assigned_target_one_hot = torch.zeros(
            self.num_envs, self._num_drones, self._num_targets, device=self.device
        )
        assigned_target_one_hot.scatter_(2, self._assigned_target_idx.unsqueeze(-1), 1.0)

        # 统一归一化连续物理量，降低不同量纲对策略学习的干扰
        drone_pos_xy_norm = drone_pos_xy / self._norm_pos_scale
        drone_pos_z_norm = drone_pos_z / 5.0                       # z归一化：0~10m → 0~2.0
        drone_lin_vel_xy_norm = drone_lin_vel_xy / self._norm_vel_scale
        drone_lin_vel_z_norm = drone_lin_vel_z / self._norm_vel_scale
        drone_ang_vel_norm = self.drone_angular_velocities / self._norm_ang_vel_scale
        rel_xy_norm = rel_xy / self._norm_pos_scale
        dist_xy_norm = dist_xy / self._norm_dist_scale
        target_vel_xy_norm = target_vel_xy / self._norm_vel_scale
        target_values_norm = (
            self._target_values.unsqueeze(0).repeat(self.num_envs, 1) / self._norm_target_value_scale
        )

        observations = {}
        for drone_idx, agent in enumerate(self.cfg.possible_agents):
            # one-hot标识当前智能体（用于区分不同无人机的观测）
            one_hot = torch.zeros(self.num_envs, self._num_drones, device=self.device)
            one_hot[:, drone_idx] = 1.0

            # 当前无人机到各目标的相对位置和距离
            own_rel_xy = rel_xy_norm[:, drone_idx].reshape(self.num_envs, -1)  # 展平为(num_envs, num_targets*2)
            own_dist = dist_xy_norm[:, drone_idx]  # (num_envs, num_targets)

            # 其他无人机到各目标的距离（用于协作信息）
            other_ids = [j for j in range(self._num_drones) if j != drone_idx]
            other_rel_xy = (
                drone_pos_xy_norm[:, other_ids] - drone_pos_xy_norm[:, drone_idx].unsqueeze(1)
            ).reshape(self.num_envs, -1)
            other_dist = dist_xy_norm[:, other_ids].reshape(self.num_envs, -1)  # (num_envs, (num_drones-1)*num_targets)

            # 构建观测向量
            obs_t = torch.cat(
                (
                    one_hot,                    # 智能体标识
                    drone_pos_xy_norm[:, drone_idx],          # 本机位置XY(2维)
                    drone_pos_z_norm[:, drone_idx],           # 本机高度Z(1维)，用于高度控制
                    self.drone_rot_matrices[:, drone_idx].reshape(self.num_envs, -1),  # 本机姿态(9维)
                    drone_lin_vel_xy_norm[:, drone_idx],      # 本机速度XY(2维)
                    drone_lin_vel_z_norm[:, drone_idx],       # 本机垂直速度Vz(1维)，用于高度控制
                    drone_ang_vel_norm[:, drone_idx],  # 本机角速度(3维)
                    own_rel_xy,                 # 到各目标相对位置(2*num_targets维)
                    own_dist,                   # 到各目标距离(num_targets维)
                    other_rel_xy,               # 其他无人机相对位置XY(2*(num_drones-1)维)
                    other_dist,                 # 其他无人机到目标距离((num_drones-1)*num_targets维)
                    target_vel_xy_norm.reshape(self.num_envs, -1),  # 目标速度XY(2*num_targets维)
                    closest_drone_one_hot.reshape(self.num_envs, -1),  # 最近无人机one-hot(num_targets*num_drones维)
                    target_values_norm,         # 目标价值(num_targets维)
                    assigned_target_one_hot[:, drone_idx],  # 固定分配目标one-hot(num_targets维)
                ),
                dim=-1,
            )

            # 存入观测缓冲区并返回展平后的观测
            self._observation_buffers[agent].append(obs_t)
            observations[agent] = self._observation_buffers[agent].buffer.reshape(self.num_envs, -1)

        return observations

    def _get_states(self) -> torch.Tensor:
        """
        获取全局状态（评论家输入）
        
        包含所有无人机和目标的完整状态信息，用于集中式训练。
        """
        for i, robot in enumerate(self.robots):
            root_state = robot.data.root_state_w
            self.drone_positions[:, i] = root_state[:, :3] - self.scene.env_origins
            self.drone_orientations[:, i] = root_state[:, 3:7]
            self.drone_linear_velocities[:, i] = root_state[:, 7:10]
            self.drone_angular_velocities[:, i] = root_state[:, 10:13]
        self.drone_rot_matrices[:] = matrix_from_quat(self.drone_orientations)

        drone_positions_norm = self.drone_positions / self._norm_pos_scale
        drone_linear_velocities_norm = self.drone_linear_velocities / self._norm_vel_scale
        drone_angular_velocities_norm = self.drone_angular_velocities / self._norm_ang_vel_scale
        target_positions_norm = self._target_positions / self._norm_pos_scale
        target_velocities_norm = self._target_velocities / self._norm_vel_scale
        target_values_norm = self._target_values.unsqueeze(0).repeat(self.num_envs, 1) / self._norm_target_value_scale
        assigned_target_one_hot = torch.zeros(
            self.num_envs, self._num_drones, self._num_targets, device=self.device
        )
        assigned_target_one_hot.scatter_(2, self._assigned_target_idx.unsqueeze(-1), 1.0)

        states = torch.cat(
            (
                drone_positions_norm.view(self.num_envs, -1),  # 无人机位置 (9)
                self.drone_rot_matrices.view(self.num_envs, -1),  # 旋转矩阵 (27)
                drone_linear_velocities_norm.view(self.num_envs, -1),  # 线速度 (9)
                drone_angular_velocities_norm.view(self.num_envs, -1),  # 角速度 (9)
                target_positions_norm.view(self.num_envs, -1),  # 目标位置 (12)
                target_velocities_norm.view(self.num_envs, -1),  # 目标速度 (12)
                self._target_claimed.float().view(self.num_envs, -1),  # 捕获状态 (4)
                target_values_norm,  # 目标价值 (4)
                assigned_target_one_hot.view(self.num_envs, -1),  # 固定分配关系 (num_drones*num_targets)
            ),
            dim=-1,
        )
        return states

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        """更稳的多目标奖励设计：连续奖励 + 软约束 + step_dt 缩放"""
    
        # =========================
        # 0) 更新无人机状态
        # =========================
        for i, robot in enumerate(self.robots):
            root_state = robot.data.root_state_w
            self.drone_positions[:, i] = root_state[:, :3] - self.scene.env_origins
            self.drone_orientations[:, i] = root_state[:, 3:7]
            self.drone_linear_velocities[:, i] = root_state[:, 7:10]
            self.drone_angular_velocities[:, i] = root_state[:, 10:13]
    
        drone_pos = self.drone_positions                              # (E, D, 3)
        drone_pos_xy = drone_pos[:, :, :2]                           # (E, D, 2)
        drone_z = drone_pos[:, :, 2]                                 # (E, D)
        drone_vel = self.drone_linear_velocities                     # (E, D, 3)
        drone_vel_xy = drone_vel[:, :, :2]                           # (E, D, 2)
        drone_vel_z = drone_vel[:, :, 2]                             # (E, D)
        drone_ang_vel = self.drone_angular_velocities                # (E, D, 3)
    
        target_pos_xy = self._target_positions[:, :, :2]             # (E, T, 2)
    
        assigned_target_idx = self._assigned_target_idx
        gather_idx_xy = assigned_target_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 2)
        assigned_target_pos_xy = torch.gather(target_pos_xy.unsqueeze(1).expand(-1, self._num_drones, -1, -1), 2, gather_idx_xy).squeeze(2)
        assigned_target_vel_xy = torch.gather(
            self._target_velocities[:, :, :2].unsqueeze(1).expand(-1, self._num_drones, -1, -1),
            2,
            gather_idx_xy,
        ).squeeze(2)
        assigned_target_values = self._target_values[assigned_target_idx]

        # =========================
        # 1) 相对位置 / 距离
        # =========================
        rel_xy = target_pos_xy.unsqueeze(1) - drone_pos_xy.unsqueeze(2)   # (E, D, T, 2)
        dist_xy = torch.norm(rel_xy, dim=-1)                               # (E, D, T)
        assigned_rel_xy = assigned_target_pos_xy - drone_pos_xy
        assigned_dist = torch.norm(assigned_rel_xy, dim=-1)
    
        # 推荐在 cfg 里新增这些参数
        dist_sigma = getattr(self.cfg, "distance_reward_sigma", 2.0)
        tracking_distance_xy = self.cfg.tracking_distance_xy
        height_sigma = getattr(self.cfg, "height_reward_sigma", 1.0)
        smooth_sigma = getattr(self.cfg, "action_smoothness_sigma", 0.25)
        collision_margin = getattr(self.cfg, "collision_soft_margin", 0.5)
        boundary_soft_margin = getattr(self.cfg, "boundary_soft_margin", 1.0)
        altitude_soft_margin = getattr(self.cfg, "altitude_soft_margin", 1.0)
        altitude_upper_soft_threshold = getattr(self.cfg, "altitude_upper_soft_threshold", self.cfg.desired_height + 1.0)
        high_altitude_soft_margin = getattr(self.cfg, "high_altitude_soft_margin", 1.0)
        high_altitude_penalty_weight = getattr(self.cfg, "high_altitude_penalty_weight", 1.0)
        height_error_penalty_weight = getattr(self.cfg, "height_error_penalty_weight", 0.0)
        height_error_above_extra_weight = getattr(self.cfg, "height_error_above_extra_weight", 0.0)
        height_error_quadratic_weight = getattr(self.cfg, "height_error_quadratic_weight", 0.0)
        height_hold_deadband = getattr(self.cfg, "height_hold_deadband", 0.0)
        vertical_direction_penalty_weight = getattr(self.cfg, "vertical_direction_penalty_weight", 0.0)
        velocity_follow_sigma = getattr(self.cfg, "velocity_follow_sigma", 0.8)
        velocity_follow_progress_weight = getattr(self.cfg, "velocity_follow_progress_weight", 0.3)
        velocity_follow_overspeed_weight = getattr(self.cfg, "velocity_follow_overspeed_weight", 0.2)
        velocity_follow_overspeed_margin = getattr(self.cfg, "velocity_follow_overspeed_margin", 0.3)
        velocity_penalty_xy_safe = getattr(self.cfg, "velocity_penalty_xy_safe", 0.0)
        velocity_penalty_z_safe = getattr(self.cfg, "velocity_penalty_z_safe", 0.0)
        velocity_penalty_z_scale = getattr(self.cfg, "velocity_penalty_z_scale", 0.25)
    
        eps = 1e-6

        # 速度误差（提前计算，供 tracking_reward 和 velocity_follow_reward 共用）
        vel_err_xy = drone_vel_xy - assigned_target_vel_xy
        vel_err_norm = torch.norm(vel_err_xy, dim=-1)  # (E, D)

        # =========================
        # 2) 距离奖励（连续型，更宽）
        #    新版 exp(-(d/sigma)^2)，sigma=10 在 5m 以内梯度显著
        # =========================
        distance_reward = self.cfg.distance_reward_weight * torch.exp(
            - (assigned_dist / (dist_sigma + eps)) ** 2
        )

        # 2.5) 距离进度奖励（势函数 shaping）
        #    reward = w * (prev_dist - curr_dist) / step_dt，在 total 乘以 step_dt 后
        #    等价于：每步奖励 w * ∆dist（减少距离即得正奖励）
        #    这提供了直接的时序梯度，驱动无人机主动缩短与目标的距离
        # =========================
        dist_progress_weight = getattr(self.cfg, "dist_progress_weight", 0.0)
        dist_progress = self._prev_assigned_dist - assigned_dist   # (E, D)，正值=拉近
        dist_progress_reward = dist_progress_weight * torch.clamp(
            dist_progress / (self.step_dt + eps), min=-10.0, max=10.0
        )
        self._prev_assigned_dist = assigned_dist.detach().clone()
        # =========================

        # =========================
        # 3) 追踪奖励（进入 + 持续保持 独立两项）
        #
        #    进入奖励（entry）：进入区立即获得，幅度固定，速度质量是加成
        #    持续保持奖励（holding）：独立项，从 0 线性增长，稳定在区内 ramp_time 秒后达满值
        #
        #    tracking = entry_reward + holding_reward
        #      entry   = w_entry × v × in_zone × (1 + vel_q_alpha × vel_q)
        #      holding = w_hold  × v × in_zone × clamp(timer / ramp_time, 0, 1)
        #
        #    w_hold > w_entry：稳定贴近后 holding 超过 entry，成为主要激励来源
        #    → 策略会学到"进去容易，待在里面才赚大"
        # =========================
        tracking_zone_sharpness    = getattr(self.cfg, "tracking_zone_sharpness", 4.0)
        tracking_vel_sigma         = getattr(self.cfg, "tracking_vel_match_sigma", 1.5)
        tracking_vel_quality_alpha = getattr(self.cfg, "tracking_vel_quality_alpha", 0.5)
        tracking_hold_weight       = getattr(self.cfg, "tracking_hold_weight", 3.0)
        tracking_hold_ramp_time    = getattr(self.cfg, "tracking_hold_ramp_time", 3.0)

        # 软边界区内因子：sharpness=4 → 约 0.25m 内完成 0→1 过渡
        in_zone_factor = torch.sigmoid(
            (tracking_distance_xy - assigned_dist) * tracking_zone_sharpness
        )

        # 速度质量：连续权重，entry 加成 + holding 乘数
        vel_quality = torch.exp(-(vel_err_norm / (tracking_vel_sigma + eps)) ** 2)

        # 进入奖励：进入区即有，速度质量好时可多赚 tracking_vel_quality_alpha 倍
        entry_reward = self.cfg.tracking_reward_weight * assigned_target_values * in_zone_factor * (
            1.0 + tracking_vel_quality_alpha * vel_quality
        )

        # 持续保持奖励：ramp 线性增长 + vel_quality 乘数
        # → 只有"在圈内 + 速度匹配"才能积累最大 holding 奖励
        holding_ramp = torch.clamp(
            self._tracking_stable_timer / (tracking_hold_ramp_time + eps), max=1.0
        )
        holding_reward = (
            tracking_hold_weight * assigned_target_values * in_zone_factor * holding_ramp * vel_quality
        )

        tracking_reward = entry_reward + holding_reward

        # 更新计时器：
        #   在圈内（dist < tracking_distance_xy）→ 以固定 step_dt 速率增长，不再由 vel_quality 节流。
        #     vel_quality 仅作为 holding_reward 的幅度乘数，而非计时速率调节器。
        #     早期训练中速度匹配差时，timer 仍能正常积累，策略获得清晰的"停留即有收益"信号。
        #   离圈 → 以 decay_rate 倍速衰减（而非硬归零），防止轻微抖动蒸发全部积累。
        #     decay_rate=3.0 意味着出圈 1/3s 后 timer 才清零，给策略一定容错余地。
        in_zone_hard = assigned_dist < tracking_distance_xy
        timer_decay_rate = getattr(self.cfg, "tracking_stable_timer_decay_rate", 0.0)
        self._tracking_stable_timer = torch.where(
            in_zone_hard,
            self._tracking_stable_timer + self.step_dt,                                          # 固定速率增长
            torch.clamp(self._tracking_stable_timer - self.step_dt * timer_decay_rate, min=0.0), # 软衰减
        )

        # =========================
        # 3.1) 速度跟随奖励（与最近目标保持速度一致并向前推进）
        # =========================
        # vel_err_xy / vel_err_norm 已在上方提前计算
        vel_match_reward = torch.exp(- (vel_err_norm / (velocity_follow_sigma + eps)) ** 2)

        target_speed = torch.norm(assigned_target_vel_xy, dim=-1)  # (E, D)
        target_dir = assigned_target_vel_xy / (target_speed.unsqueeze(-1) + eps)
        progress_speed = torch.sum(drone_vel_xy * target_dir, dim=-1)
        progress_reward = torch.clamp(progress_speed / (target_speed + eps), min=0.0, max=1.5)

        drone_speed_xy = torch.norm(drone_vel_xy, dim=-1)
        overspeed = torch.clamp(
            drone_speed_xy - (target_speed + velocity_follow_overspeed_margin),
            min=0.0,
        ) / (target_speed + velocity_follow_overspeed_margin + eps)

        velocity_follow_reward = self.cfg.velocity_follow_weight * (
            vel_match_reward
            + velocity_follow_progress_weight * progress_reward
            - velocity_follow_overspeed_weight * overspeed
        )
    
        # =========================
        # 4) 动作平滑奖励
        #    用平方差比 abs().mean 更敏感、更平滑
        # =========================
        current_actions = torch.stack(
            [self.actions[agent] for agent in self.cfg.possible_agents], dim=1
        )  # (E, D, A)
    
        prev_actions = torch.stack(
            [self.prev_actions[agent] for agent in self.cfg.possible_agents], dim=1
        )  # (E, D, A)
    
        action_delta = current_actions - prev_actions
        action_delta_sq_mean = (action_delta ** 2).mean(dim=-1)  # (E, D)
    
        action_smoothness = self.cfg.action_smoothness_weight * torch.exp(
            - action_delta_sq_mean / (smooth_sigma ** 2 + eps)
        )
    
        # =========================
        # 5) 角速度惩罚
        #    保留线性范数，足够直观
        # =========================
        body_rate_penalty = self.cfg.body_rate_penalty_weight * torch.norm(
            drone_ang_vel, dim=-1
        )
    
        # =========================
        # 6) 速度惩罚
        #    改成阈值型：只惩罚超出安全速度的部分
        # =========================
        vel_xy_speed = torch.norm(drone_vel_xy, dim=-1)
        vel_z_speed = torch.abs(drone_vel_z)
        vel_xy_penalty = torch.clamp(vel_xy_speed - velocity_penalty_xy_safe, min=0.0)
        vel_z_penalty = torch.clamp(vel_z_speed - velocity_penalty_z_safe, min=0.0)

        velocity_penalty = self.cfg.velocity_penalty_weight * (
            vel_xy_penalty + velocity_penalty_z_scale * vel_z_penalty
        )
    
        # =========================
        # 7) 推力 / 控制 effort 惩罚
        #    不再用 max(thrust_z)，改为动作平方均值
        #    这通常比 max 更稳、更一致
        # =========================
        force_penalty = self.cfg.force_penalty_weight * (current_actions ** 2).mean(dim=-1)
    
        # =========================
        # 8) 高度奖励（连续且更宽）
        # =========================
        height_error_raw = torch.abs(drone_z - self.cfg.desired_height)
        height_error = torch.clamp(height_error_raw - height_hold_deadband, min=0.0)
        above_height_error = torch.clamp(drone_z - self.cfg.desired_height, min=0.0)
        height_reward = self.cfg.height_reward_weight * torch.exp(
            - (height_error_raw / (height_sigma + eps)) ** 2
        )
        height_error_penalty = (
            height_error_penalty_weight * height_error
            + height_error_above_extra_weight * above_height_error
            + height_error_quadratic_weight * (height_error ** 2)
        )
        # 方向性惩罚：高于目标还在上升、低于目标还在下降
        moving_away_vz = torch.where(
            drone_z >= self.cfg.desired_height,
            torch.clamp(drone_vel_z, min=0.0),
            torch.clamp(-drone_vel_z, min=0.0),
        )
        vertical_direction_penalty = vertical_direction_penalty_weight * moving_away_vz

        # 上升速度软惩罚：仅 vz > 0 时生效，抑制起步先往上窜
        upward_vz_penalty = getattr(self.cfg, "upward_vz_penalty_weight", 0.0) * torch.clamp(
            drone_vel_z, min=0.0
        )

        # =========================
        # 9) 安全软惩罚
        #    collision / 边界 / 低高度都改成“离危险越近罚越多”
        #    同时你原有的 terminate 条件可以继续保留在 _get_dones 中
        # =========================
    
        # 9.1 机间距软惩罚
        pos_xy = drone_pos_xy
        diff = pos_xy.unsqueeze(2) - pos_xy.unsqueeze(1)  # (E, D, D, 2)
        dist = torch.norm(diff, dim=-1)                   # (E, D, D)
    
        eye = torch.eye(self._num_drones, device=self.device, dtype=torch.bool)
        dist.masked_fill_(eye.unsqueeze(0), float("inf"))
        min_sep = dist.min(dim=-1).values  # (E, D)
    
        # 当 min_sep 小于 threshold + margin 时开始罚
        collision_soft_threshold = self.cfg.drone_collision_threshold + collision_margin
        collision_margin_violation = torch.clamp(
            collision_soft_threshold - min_sep,
            min=0.0
        )
        collision_penalty = collision_margin_violation / (collision_margin + eps)
    
        # 9.2 边界软惩罚
        # 假设 bounding_box_threshold 是标量；如果是向量也兼容广播
        bbox = self.cfg.bounding_box_threshold
        abs_pos = torch.abs(drone_pos)  # (E, D, 3)
    
        # 距离硬边界还有多少余量
        boundary_margin_left = bbox - abs_pos
    
        # 当余量小于 soft_margin 时开始罚；越靠近边界罚越多；越界后继续增大
        boundary_violation = torch.clamp(
            boundary_soft_margin - boundary_margin_left,
            min=0.0
        ) / (boundary_soft_margin + eps)
    
        boundary_penalty = boundary_violation.sum(dim=-1)  # (E, D)
    
        # 9.3 低高度软惩罚
        altitude_margin = drone_z - self.cfg.min_altitude
        low_altitude_penalty = torch.clamp(
            altitude_soft_margin - altitude_margin,
            min=0.0
        ) / (altitude_soft_margin + eps)

        # 9.4 超高软惩罚
        high_altitude_excess = torch.clamp(drone_z - altitude_upper_soft_threshold, min=0.0)
        high_altitude_penalty = high_altitude_penalty_weight * (
            high_altitude_excess / (high_altitude_soft_margin + eps)
        )

        # 合并安全软惩罚
        safety_penalty = self.cfg.safety_penalty_weight * (
            collision_penalty + boundary_penalty + low_altitude_penalty + high_altitude_penalty
        )
    
        # =========================
        # 10) 可选：存活奖励（非常建议）
        #    防止 reward 总体过负，也有助于稳定训练
        # =========================
        alive_reward_weight = getattr(self.cfg, "alive_reward_weight", 0.0)
        alive_reward = alive_reward_weight * torch.ones_like(assigned_dist)

        # =========================
        # 10.5) success proximity reward + success terminal bonus
        #
        #   success_proximity_reward：同时满足所有 success 条件时的每步密集奖励。
        #     success_mask = dist≤success_dist AND vel_err≤tolerance AND in_height_band
        #     直接对齐 _get_dones() 中的 success_mask，无人机在"成功区"内停留即得，
        #     与 tracking_reward（仅 dist 条件）互补，提供"速度+距离同时达标"的联合信号。
        #
        #   success_bonus：_sustained_follow_timer 上步已越过 success_hold_time 时发放。
        #     _get_rewards() 在 _get_dones() 之前调用，此时 timer 已是上步 _get_dones() 更新后的值。
        #     等价于：成功条件首次持续满足 → 本步发放一次性大额奖励 → 下一步 _get_dones() 触发 reset。
        #     让策略明确区分"成功终止"与"撞地/超时终止"，提供清晰的价值锚点。
        # =========================
        success_dist_threshold_rew = getattr(self.cfg, "success_distance_xy", self.cfg.track_distance_xy)
        in_height_band_rew = (drone_z >= self.cfg.min_altitude) & (drone_z <= self.cfg.max_altitude)
        # 速度条件与 _get_dones() 保持一致：只取纵向分量（沿目标前进方向）
        _target_spd_rew = torch.norm(assigned_target_vel_xy, dim=-1, keepdim=True).clamp(min=1e-6)
        _target_dir_rew = assigned_target_vel_xy / _target_spd_rew
        vel_err_longitudinal_rew = torch.abs(
            torch.sum((drone_vel_xy - assigned_target_vel_xy) * _target_dir_rew, dim=-1)
        )  # (E, D)
        success_mask_rew = (
            (assigned_dist <= success_dist_threshold_rew)
            & (vel_err_longitudinal_rew <= self.cfg.success_velocity_tolerance)
            & in_height_band_rew
        )  # (E, D)，per-drone success 条件，与 _get_dones() enter_mask 对齐

        success_proximity_weight = getattr(self.cfg, "success_proximity_weight", 0.0)
        success_proximity_reward = (
            success_proximity_weight * success_mask_rew.float() * assigned_target_values
        )

        success_bonus_weight = getattr(self.cfg, "success_bonus_weight", 0.0)
        success_just_triggered = self._sustained_follow_timer >= self.cfg.success_hold_time  # (E, D)
        success_bonus = success_bonus_weight * success_just_triggered.float() * assigned_target_values

        # =========================
        # 11) 汇总
        # =========================
        total_reward = (
            alive_reward
            + distance_reward
            + dist_progress_reward
            + tracking_reward
            + velocity_follow_reward
            + success_proximity_reward
            + success_bonus
            + action_smoothness
            + height_reward
            - body_rate_penalty
            - velocity_penalty
            - force_penalty
            - height_error_penalty
            - vertical_direction_penalty
            - upward_vz_penalty
            - safety_penalty
        ) * self.step_dt

        # 临时调试：每隔 N 步打印一次奖励项 batch mean
        self._reward_debug_counter += 1
        if self._reward_debug_counter % self._reward_debug_print_interval == 0:
            print(
                "[flyfollow][reward_mean] "
                f"total={total_reward.mean().item():.4f}, "
                f"dist={distance_reward.mean().item():.4f}, "
                f"progress={dist_progress_reward.mean().item():.4f}, "
                f"track={tracking_reward.mean().item():.4f}, "
                f"vel_follow={velocity_follow_reward.mean().item():.4f}, "
                f"smooth={action_smoothness.mean().item():.4f}, "
                f"high_alt_pen={high_altitude_penalty.mean().item():.4f}, "
                f"rate_pen={body_rate_penalty.mean().item():.4f}, "
                f"vel_pen={velocity_penalty.mean().item():.4f}, "
                f"force_pen={force_penalty.mean().item():.4f}, "
                f"safe_pen={safety_penalty.mean().item():.4f}, "
                f"alive={alive_reward.mean().item():.4f}"
            )
    
        # =========================
        # 12) 日志（按环境求和）
        # =========================
        self._episode_sums["distance_reward"] += distance_reward.sum(dim=-1)
        self._episode_sums["dist_progress_reward"] += dist_progress_reward.sum(dim=-1)
        self._episode_sums["tracking_reward"] += tracking_reward.sum(dim=-1)
        self._episode_sums["velocity_follow_reward"] += velocity_follow_reward.sum(dim=-1)
        self._episode_sums["action_smoothness"] += action_smoothness.sum(dim=-1)
        self._episode_sums["body_rate_penalty"] += body_rate_penalty.sum(dim=-1)
        self._episode_sums["velocity_penalty"] += velocity_penalty.sum(dim=-1)
        self._episode_sums["force_penalty"] += force_penalty.sum(dim=-1)
        self._episode_sums["height_reward"] += height_reward.sum(dim=-1)
        self._episode_sums["height_error_penalty"] += height_error_penalty.sum(dim=-1)
        self._episode_sums["vertical_direction_penalty"] += vertical_direction_penalty.sum(dim=-1)
        self._episode_sums["upward_vz_penalty"] += upward_vz_penalty.sum(dim=-1)
        self._episode_sums["high_altitude_penalty"] += high_altitude_penalty.sum(dim=-1)
        self._episode_sums["safety_penalty"] += safety_penalty.sum(dim=-1)
        self._episode_sums["success_proximity_reward"] += success_proximity_reward.sum(dim=-1)
        self._episode_sums["success_bonus"] += success_bonus.sum(dim=-1)

        if "alive_reward" in self._episode_sums:
            self._episode_sums["alive_reward"] += alive_reward.sum(dim=-1)

        rewards = {
            agent: total_reward[:, i]
            for i, agent in enumerate(self.cfg.possible_agents)
        }
        return rewards

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """判断终止条件"""
        # 获取无人机位置用于终止判断
        for i, robot in enumerate(self.robots):
            root_state = robot.data.root_state_w
            self.drone_positions[:, i] = root_state[:, :3] - self.scene.env_origins
            self.drone_linear_velocities[:, i] = root_state[:, 7:10]

        # 高度过低终止：防止撞击地面
        falcon_fly_low = (self.drone_positions[:, :, 2] < self.cfg.min_altitude).any(dim=-1)
        # 高度过高终止：防止长期高空逃逸
        falcon_fly_high = (self.drone_positions[:, :, 2] > self.cfg.max_altitude).any(dim=-1)
        
        # 越界终止：防止飞出限定区域
        body_pos_outside = (self.drone_positions.abs() > self.cfg.bounding_box_threshold).any(dim=-1).any(dim=-1)

        # 时间超时
        self.time_out = self.episode_length_buf >= self.max_episode_length - 1

        assigned_target_pos_xy = torch.gather(
            self._target_positions[:, :, :2].unsqueeze(1).expand(-1, self._num_drones, -1, -1),
            2,
            self._assigned_target_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 2),
        ).squeeze(2)
        assigned_target_vel_xy = torch.gather(
            self._target_velocities[:, :, :2].unsqueeze(1).expand(-1, self._num_drones, -1, -1),
            2,
            self._assigned_target_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 2),
        ).squeeze(2)
        assigned_dist_xy = torch.norm(self.drone_positions[:, :, :2] - assigned_target_pos_xy, dim=-1)
        # ── 速度误差：纵向分量（沿目标前进方向），忽略侧向漂移 ────────────────
        # vel_error_longitudinal = |dot(v_drone_xy - v_target_xy, target_dir)|
        # 与 _get_rewards() 的 success_mask_rew / vel_err_longitudinal_rew 口径完全一致
        _vel_diff_dones   = self.drone_linear_velocities[:, :, :2] - assigned_target_vel_xy  # (E, D, 2)
        _target_spd_dones = torch.norm(assigned_target_vel_xy, dim=-1, keepdim=True).clamp(min=1e-6)
        _target_dir_dones = assigned_target_vel_xy / _target_spd_dones                       # (E, D, 2)
        vel_error_longitudinal = torch.abs(
            torch.sum(_vel_diff_dones * _target_dir_dones, dim=-1)
        )  # (E, D) — 纵向速度误差，全局统一口径
        drone_z = self.drone_positions[:, :, 2]

        # ── 滞回成功判定（Hysteresis + Soft Hold） ───────────────────────────
        #
        # 分两层：
        #   enter_mask（硬阈值）：首次启动 timer 必须同时满足的严格条件
        #     dist ≤ d_enter  AND  vel_long ≤ v_enter  AND  in_height_band
        #
        #   hold_factor（软权重，[0,1]）：已进入后 timer 的增长/衰减速率乘数
        #     vel_factor  = exp(-(vel_long / v_hold_sigma)²)  — Gaussian，vel=0时1，vel=sigma时≈0.37
        #     dist_factor = sigmoid((d_hold - dist) * sharpness) — sigmoid，中心满值，边界过渡
        #     hold_factor = vel_factor × dist_factor × in_height_band
        #
        #   timer 更新（连续平滑）：
        #     已进入（timer>0）：delta = (hold_factor - decay_rate×(1-hold_factor)) × dt
        #       → hold_factor=1.0 → 满速增长
        #       → hold_factor=decay/(1+decay)≈0.33 → 中性（不增不减）
        #       → hold_factor=0.0 → 以 decay_rate 衰减
        #     未进入（timer=0）：enter_mask 才以固定速率启动，否则 timer 维持 0
        #
        # 效果：不再是单步硬开关，轻微超出 hold 区边界时 timer 缓慢衰减而非骤停，
        # 只有持续远离才会归零，归零后必须重新通过 enter_mask 才能再次启动。
        in_height_band = (drone_z >= self.cfg.min_altitude) & (drone_z <= self.cfg.max_altitude)
        success_dist_threshold = getattr(self.cfg, "success_distance_xy", self.cfg.track_distance_xy)
        hold_dist        = getattr(self.cfg, "success_distance_xy_hold", success_dist_threshold)
        hold_vel_sigma   = getattr(self.cfg, "success_velocity_tolerance_hold", self.cfg.success_velocity_tolerance)
        hold_dist_sharp  = getattr(self.cfg, "success_hold_dist_sharpness", 0.5)
        decay_rate       = getattr(self.cfg, "success_timer_decay_rate", 0.5)

        # enter_mask：硬阈值，保留严格进入门槛
        enter_mask = (
            (assigned_dist_xy      <= success_dist_threshold)
            & (vel_error_longitudinal <= self.cfg.success_velocity_tolerance)
            & in_height_band
        )  # (E, D)

        # hold_factor：连续软权重，已进入后控制 timer 增减速率
        hold_vel_factor  = torch.exp(
            -(vel_error_longitudinal / (hold_vel_sigma + 1e-6)) ** 2
        )  # Gaussian：vel=0→1.0，vel=sigma→0.37，vel=2σ→0.02
        hold_dist_factor = torch.sigmoid(
            (hold_dist - assigned_dist_xy) * hold_dist_sharp
        )  # sigmoid：dist远小于hold_dist→~1，超过hold_dist→快速降向0
        hold_factor = hold_vel_factor * hold_dist_factor * in_height_band.float()  # (E, D)

        already_entered = self._sustained_follow_timer > 0.0  # (E, D)

        # timer 增量：连续平滑，正值增长，负值衰减
        timer_delta = torch.where(
            already_entered,
            (hold_factor - decay_rate * (1.0 - hold_factor)) * self.step_dt,  # 软增减
            enter_mask.float() * self.step_dt,                                 # 硬启动
        )
        self._sustained_follow_timer = (self._sustained_follow_timer + timer_delta).clamp(min=0.0)

        # success_mask 仍用严格 enter 条件，与 _get_rewards() 的 success_mask_rew 对齐
        success_mask = enter_mask

        # hold_mask：用于日志，表示 hold_factor > 0.5（等效于软权重通过"中性点"）
        hold_mask = hold_factor > (decay_rate / (1.0 + decay_rate))  # 增减平衡点

        sustained_success = (self._sustained_follow_timer >= self.cfg.success_hold_time).any(dim=-1)

        # 综合终止条件
        terminations = falcon_fly_low | falcon_fly_high | body_pos_outside | sustained_success
        timed_outs = self.time_out

        # 调试日志（每步）- 记录到全局 log 以及每个 agent 的 extras 中
        fly_low_rate = falcon_fly_low.float().mean()
        fly_high_rate = falcon_fly_high.float().mean()
        out_of_bounds_rate = body_pos_outside.float().mean()
        sustained_success_rate = sustained_success.float().mean()
        time_out_rate = self.time_out.float().mean()
        any_rate = terminations.float().mean()
        min_height = self.drone_positions[:, :, 2].min(dim=-1).values.mean()
        max_height = self.drone_positions[:, :, 2].max(dim=-1).values.mean()
        # success 条件分解（所有速度指标均为纵向误差口径，与 _get_rewards() 对齐）
        success_dist_rate  = (assigned_dist_xy <= success_dist_threshold).float().mean()
        success_vel_rate   = (vel_error_longitudinal <= self.cfg.success_velocity_tolerance).float().mean()
        success_mask_rate  = enter_mask.float().mean()    # enter 严格条件同时满足比例
        hold_mask_rate     = hold_mask.float().mean()     # hold_factor > 中性点的比例
        hold_factor_mean   = hold_factor.mean()           # hold 软权重均值（0~1）
        in_success_rate    = already_entered.float().mean()  # timer>0（已进入）比例
        self.extras["log"] = {
            "Debug/Termination/fly_low_rate": fly_low_rate,
            "Debug/Termination/fly_high_rate": fly_high_rate,
            "Debug/Termination/out_of_bounds_rate": out_of_bounds_rate,
            "Debug/Termination/sustained_success_rate": sustained_success_rate,
            "Debug/Termination/time_out_rate": time_out_rate,
            "Debug/Termination/any_rate": any_rate,
            "Debug/Termination/min_height": min_height,
            "Debug/Termination/max_height": max_height,
            "Debug/Success/dist_rate": success_dist_rate,
            "Debug/Success/vel_rate_longitudinal": success_vel_rate,
            "Debug/Success/enter_mask_rate": success_mask_rate,
            "Debug/Success/hold_factor_mean": hold_factor_mean,
            "Debug/Success/hold_mask_rate": hold_mask_rate,
            "Debug/Success/in_success_rate": in_success_rate,
        }
        for agent in self.cfg.possible_agents:
            if "log" not in self.extras[agent]:
                self.extras[agent]["log"] = {}
            log = self.extras[agent]["log"]
            log["Debug/Termination/fly_low_rate"] = fly_low_rate
            log["Debug/Termination/fly_high_rate"] = fly_high_rate
            log["Debug/Termination/out_of_bounds_rate"] = out_of_bounds_rate
            log["Debug/Termination/sustained_success_rate"] = sustained_success_rate
            log["Debug/Termination/time_out_rate"] = time_out_rate
            log["Debug/Termination/any_rate"] = any_rate
            log["Debug/Termination/min_height"] = min_height
            log["Debug/Termination/max_height"] = max_height
            log["Debug/Success/dist_rate"] = success_dist_rate
            log["Debug/Success/vel_rate_longitudinal"] = success_vel_rate
            log["Debug/Success/enter_mask_rate"] = success_mask_rate
            log["Debug/Success/hold_factor_mean"] = hold_factor_mean
            log["Debug/Success/hold_mask_rate"] = hold_mask_rate
            log["Debug/Success/in_success_rate"] = in_success_rate

        # 所有智能体共享相同终止信号
        terminated = {agent: terminations for agent in self.cfg.possible_agents}
        time_outs = {agent: timed_outs for agent in self.cfg.possible_agents}

        return terminated, time_outs

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None):
        """重置指定环境索引"""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        # 基类重置（包含场景和事件）
        super()._reset_idx(env_ids)

        # 根据配置决定 reset 策略：
        # - event_randomized: 保留 EventCfg.reset_base 的随机化
        # - usd_fixed / usd_perturbed: 使用 USD 位姿（可选小扰动）
        reset_pose_mode = getattr(self.cfg, "reset_pose_mode", "event_randomized")
        if (
            reset_pose_mode in {"usd_fixed", "usd_perturbed"}
            and hasattr(self, "_usd_root_state_rel")
            and self._usd_root_state_rel is not None
        ):
            env_origins = self.scene.env_origins[env_ids]
            num_ids = env_ids.numel()
            zeros_vel = torch.zeros(num_ids, 6, device=self.device)
            position_noise = torch.zeros(num_ids, 3, device=self.device)
            yaw_noise = torch.zeros(num_ids, device=self.device)
            if reset_pose_mode == "usd_perturbed":
                noise_xy = float(getattr(self.cfg, "usd_reset_position_noise_xy", 0.0))
                noise_z = float(getattr(self.cfg, "usd_reset_position_noise_z", 0.0))
                yaw_range = float(getattr(self.cfg, "usd_reset_yaw_noise", 0.0))
                if noise_xy > 0.0:
                    position_noise[:, :2] = (2.0 * torch.rand(num_ids, 2, device=self.device) - 1.0) * noise_xy
                if noise_z > 0.0:
                    position_noise[:, 2] = (2.0 * torch.rand(num_ids, device=self.device) - 1.0) * noise_z
                if yaw_range > 0.0:
                    yaw_noise = (2.0 * torch.rand(num_ids, device=self.device) - 1.0) * yaw_range
            for i, robot in enumerate(self.robots):
                pos = (
                    self._usd_root_state_rel[i, :3].unsqueeze(0).repeat(num_ids, 1)
                    + env_origins
                    + position_noise
                )
                ori = self._usd_root_state_rel[i, 3:7].unsqueeze(0).repeat(num_ids, 1)
                if reset_pose_mode == "usd_perturbed" and yaw_noise.abs().max() > 0.0:
                    yaw_axis = torch.zeros(num_ids, 3, device=self.device)
                    yaw_axis[:, 2] = 1.0
                    yaw_quat = quat_from_angle_axis(yaw_noise, yaw_axis)
                    ori = quat_mul(yaw_quat, ori)
                root_state = torch.cat([pos, ori, zeros_vel], dim=-1)
                robot.write_root_state_to_sim(root_state, env_ids=env_ids)

        # 重置目标状态
        self._reset_targets(env_ids)

        # 在 reset 后基于当前初始几何关系固定 assignment
        for i, robot in enumerate(self.robots):
            root_state = robot.data.root_state_w
            self.drone_positions[:, i] = root_state[:, :3] - self.scene.env_origins
        self._assign_targets_for_envs(env_ids)
        self._sustained_follow_timer[env_ids] = 0.0
        self._tracking_stable_timer[env_ids] = 0.0  # 重置近距稳定计时器
        self._prev_assigned_dist[env_ids] = 100.0  # 重置进度奖励基准距离

        # 重置观测缓冲区和动作历史
        for agent in self.cfg.possible_agents:
            self._observation_buffers[agent].reset(env_ids)
            self.prev_actions[agent][env_ids] = 0.0
            self.actions[agent][env_ids] = 0.0

        # 日志记录初始化
        if "log" not in self.extras:
            self.extras["log"] = dict()
        for agent in self.cfg.possible_agents:
            if "log" not in self.extras[agent]:
                self.extras[agent]["log"] = dict()

        # 记录终止原因统计
        for i, robot in enumerate(self.robots):
            root_state = robot.data.root_state_w
            self.drone_positions[:, i] = root_state[:, :3] - self.scene.env_origins
        fly_low_count = torch.count_nonzero(
            (self.drone_positions[:, :, 2] < self.cfg.min_altitude).any(dim=-1)[env_ids]
        ).item()
        fly_high_count = torch.count_nonzero(
            (self.drone_positions[:, :, 2] > self.cfg.max_altitude).any(dim=-1)[env_ids]
        ).item()
        out_of_bounds_count = torch.count_nonzero(
            (self.drone_positions.abs() > self.cfg.bounding_box_threshold).any(dim=-1).any(dim=-1)[env_ids]
        ).item()
        time_out_count = torch.count_nonzero(self.time_out[env_ids]).item()
        self.extras["log"]["Episode_Termination/falcon_fly_low"] = fly_low_count
        self.extras["log"]["Episode_Termination/falcon_fly_high"] = fly_high_count
        self.extras["log"]["Episode_Termination/bounding_box"] = out_of_bounds_count
        success_count = torch.count_nonzero(
            (self._sustained_follow_timer >= self.cfg.success_hold_time).any(dim=-1)[env_ids]
        ).item()
        self.extras["log"]["Episode_Termination/sustained_success"] = success_count
        self.extras["log"]["Episode_Termination/time_out"] = time_out_count
        for agent in self.cfg.possible_agents:
            log = self.extras[agent]["log"]
            log["Episode_Termination/falcon_fly_low"] = fly_low_count
            log["Episode_Termination/falcon_fly_high"] = fly_high_count
            log["Episode_Termination/bounding_box"] = out_of_bounds_count
            log["Episode_Termination/sustained_success"] = success_count
            log["Episode_Termination/time_out"] = time_out_count

        # 记录奖励成分平均值
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            self.extras["log"]["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            for agent in self.cfg.possible_agents:
                self.extras[agent]["log"]["Episode_Reward/" + key] = (
                    episodic_sum_avg / self.max_episode_length_s
                )
            self._episode_sums[key][env_ids] = 0.0

    def _reset_targets(self, env_ids: torch.Tensor):
        """重置目标位置：为指定环境生成新的目标起始位置"""
        if env_ids.numel() == 0:
            return

        # 初始化目标位置：优先使用 USD 中定义的初始位置
        if hasattr(self, "_usd_target_positions_rel") and self._usd_target_positions_rel is not None:
            rel = self._usd_target_positions_rel.unsqueeze(0).repeat(env_ids.numel(), 1, 1)
            self._target_positions[env_ids] = rel
            self._target_orientations[env_ids] = self._usd_target_orientations.unsqueeze(0).repeat(
                env_ids.numel(), 1, 1
            )
        else:
            # 回退：使用配置参数中的起点
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
            start_x = self._usd_target_positions_rel[:, 0].unsqueeze(0).expand(self.num_envs, -1)
            self._target_positions[..., 0] = torch.where(
                overflow,
                start_x,
                self._target_positions[..., 0],
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
