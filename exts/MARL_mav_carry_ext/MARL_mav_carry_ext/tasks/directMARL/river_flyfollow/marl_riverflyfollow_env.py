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
from isaaclab.sensors import ContactSensor
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
        # 目标跟随状态：实时可撤销（与 move 对齐）
        # target_captured[e, t]：当前该目标是否被某架无人机跟随（距离 < capture_distance）
        self.target_captured = torch.zeros(
            self.num_envs, self._num_targets, dtype=torch.bool, device=self.device
        )
        # target_captured_by[e, t]：最近跟随该目标的无人机索引（-1=未被跟随）
        self.target_captured_by = torch.full(
            (self.num_envs, self._num_targets), -1, dtype=torch.long, device=self.device
        )
        self._target_assignment = torch.full(
            (self.num_envs, self._num_targets), -1, dtype=torch.long, device=self.device
        )  # 目标分配给哪个无人机（固定 per-episode，用于观测）

        # 目标沿x轴匀速移动
        self._target_velocities[..., 0] = cfg.target_speed

        # 奖励统计（与 move 对齐的简化版奖励项）
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                # ── 主奖励（与 move 结构对齐）─────────────────────────────
                "distance_reward",       # exp(-d/σ)² × target_value，最近无人机到各目标，move 同结构
                "tracking_reward",       # is_captured × exp(-d × scale) × value，move 同结构
                "velocity_follow_reward",# vel_match + progress - overspeed，river 特有（追随行为必需）
                # ── 辅助正奖励（exp-decay，与 move 对齐：低值越优） ────────
                "action_smoothness",     # exp(-‖Δa‖²/σ²)，move 同结构
                "body_rate_reward",      # exp(-‖ω‖)，move: body_rate_penalty_weight × exp，此处同
                "velocity_reward",       # exp(-‖v‖)，move: velocity_penalty_weight × exp，此处同
                "force_reward",          # exp(-max_thrust)，move: force_penalty_weight × exp，此处同
                # ── 高度与安全（river 特有保留项）────────────────────────
                "height_reward",         # exp(-(z-z_d)²/σ²)，move 同结构
                "upright_penalty",       # (z_body_dot - 1) × dt，move 同结构
                "safety_penalty",        # 超高 + 上升速度惩罚，river 任务高空问题保留
                # ── 硬约束固定惩罚（move 同结构，不乘 step_dt）────────────
                "collision_penalty",     # 无人机碰撞
                "drone_out",             # 越界
                "fly_low",               # 低飞
                "illegal_contact",       # 接触传感器：非法碰撞（障碍物/地面）
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
        # 持续跟随成功标志（≥3个目标同时 captured 达到 sustained_follow_duration 秒）
        self.all_targets_captured = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        # 持续跟随计时器（per-env，与 move 对齐：暂停而非归零）
        self._sustained_follow_timer = torch.zeros(self.num_envs, device=self.device)
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

        # ── 5b. 创建接触传感器（单传感器覆盖全部无人机子 prim，与 move 对齐）──────
        contact = ContactSensor(self.cfg.contact_forces)
        self.contact_sensors = [contact]
        self.scene.sensors["contact_forces"] = contact

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
                # 速度指令 + PD 控制器（与 move.ACCBR 对齐）
                # action[:, :3] → 归一化期望速度，缩放到 ±lin_vel_max
                # action[:, 3:6] → 归一化角速度，缩放到 ±ang_vel_max
                drone_idx_ap = self.cfg.possible_agents.index(drone)
                desired_vel = action[:, :3] * self.cfg.lin_vel_max
                current_vel = self.drone_linear_velocities[:, drone_idx_ap]
                vel_error = desired_vel - current_vel
                d_error = (vel_error - self._drone_prev_vel_error[:, drone_idx_ap]) / self.step_dt
                self._drone_prev_vel_error[:, drone_idx_ap] = vel_error
                commanded_acc = self.cfg.vel_Kp * vel_error + self.cfg.vel_Kd * d_error
                commanded_acc = torch.clamp(commanded_acc, -self.cfg.lin_acc_max, self.cfg.lin_acc_max)
                self._setpoints[drone]["lin_acc"] = commanded_acc
                self._setpoints[drone]["body_rates"] = action[:, 3:6] * self.cfg.ang_vel_max

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
                self.target_captured.float().view(self.num_envs, -1),  # 捕获状态 (4)
                target_values_norm,  # 目标价值 (4)
                assigned_target_one_hot.view(self.num_envs, -1),  # 固定分配关系 (num_drones*num_targets)
            ),
            dim=-1,
        )
        return states

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        """move 对齐的奖励设计：
        - 距离奖励 + 追踪区奖励（主信号，与 move 结构一致）
        - 速度跟随（river 任务必要辅助，保留）
        - 辅助正奖励：action_smoothness / body_rate / velocity / force（exp-decay，与 move 对齐）
        - 高度与安全（altitude、upright，与 move 对齐）
        - 硬约束固定惩罚：collision / out-of-bounds / fly-low（不乘 step_dt，与 move 对齐）
        - 奖励共享：所有智能体获得同一 per-env 奖励（与 move 对齐）
        """
        # =========================
        # 0) 更新无人机状态
        # =========================
        for i, robot in enumerate(self.robots):
            root_state = robot.data.root_state_w
            self.drone_positions[:, i] = root_state[:, :3] - self.scene.env_origins
            self.drone_orientations[:, i] = root_state[:, 3:7]
            self.drone_linear_velocities[:, i] = root_state[:, 7:10]
            self.drone_angular_velocities[:, i] = root_state[:, 10:13]

        drone_pos    = self.drone_positions                 # (E, D, 3)
        drone_pos_xy = drone_pos[:, :, :2]                 # (E, D, 2)
        drone_z      = drone_pos[:, :, 2]                  # (E, D)
        drone_vel    = self.drone_linear_velocities         # (E, D, 3)
        drone_vel_xy = drone_vel[:, :, :2]                 # (E, D, 2)
        drone_vel_z  = drone_vel[:, :, 2]                  # (E, D)
        drone_ang_vel = self.drone_angular_velocities       # (E, D, 3)
        target_pos_xy = self._target_positions[:, :, :2]   # (E, T, 2)

        step_dt = self.step_dt
        eps     = 1e-6

        # 固定分配（仅用于 velocity_follow_reward，观测已包含）
        assigned_target_idx = self._assigned_target_idx
        g_xy = assigned_target_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 2)
        assigned_target_vel_xy = torch.gather(
            self._target_velocities[:, :, :2].unsqueeze(1).expand(-1, self._num_drones, -1, -1),
            2, g_xy,
        ).squeeze(2)  # (E, D, 2)

        # =========================
        # 1) 距离矩阵（全对全，与 move 对齐）
        # =========================
        d_pos    = drone_pos_xy.unsqueeze(2)   # (E, D, 1, 2)
        t_pos    = target_pos_xy.unsqueeze(1)  # (E, 1, T, 2)
        dist_mat = torch.norm(d_pos - t_pos, dim=-1)  # (E, D, T)
        min_dists, closest_drone_idx = dist_mat.min(dim=1)  # (E, T)
    
        # =========================
        # 2) 距离奖励（exp-decay，全局最近分配，与 move 对齐）
        # =========================
        dist_sigma = self.cfg.distance_reward_sigma
        dist_per_target = torch.exp(-(min_dists / (dist_sigma + eps)) ** 2)  # (E, T)
        target_values = self._target_values  # (T,) broadcast to (E, T)
        distance_reward = self.cfg.distance_reward_weight * (dist_per_target * target_values).sum(dim=-1)  # (E,)

        # =========================
        # 3) 捕获状态（实时可撤销，与 move 对齐）
        # =========================
        is_captured_now = min_dists < self.cfg.capture_distance  # (E, T)
        self.target_captured = is_captured_now
        self.target_captured_by = torch.where(is_captured_now, closest_drone_idx, self.target_captured_by)
        # 只有 ≥3 个目标被捕获时才累积 timer（暂停不重置，与 move 对齐）
        enough_captured = is_captured_now.sum(dim=-1) >= 3  # (E,)
        self._sustained_follow_timer = torch.where(
            enough_captured,
            self._sustained_follow_timer + step_dt,
            self._sustained_follow_timer,
        )
        self.all_targets_captured = self._sustained_follow_timer >= self.cfg.sustained_follow_duration

        # =========================
        # 4) 追踪奖励（仅在捕获区内，与 move 对齐）
        # =========================
        tracking_per_target = (
            is_captured_now.float()
            * torch.exp(-min_dists * self.cfg.tracking_reward_scale)
            * target_values
        )  # (E, T)
        tracking_reward = self.cfg.tracking_reward_weight * tracking_per_target.sum(dim=-1)  # (E,)

        # =========================
        # 5) 速度跟随奖励（per-drone → per-env 均值，与 move 对齐）
        # =========================
        vel_err_xy = drone_vel_xy - assigned_target_vel_xy  # (E, D, 2)
        vel_err_norm = vel_err_xy.norm(dim=-1)              # (E, D)
        vel_sigma = self.cfg.velocity_follow_sigma
        vel_match = torch.exp(-(vel_err_norm / (vel_sigma + eps)) ** 2)

        target_speed = assigned_target_vel_xy.norm(dim=-1)  # (E, D)
        target_dir = assigned_target_vel_xy / (target_speed.unsqueeze(-1) + eps)
        progress_r = (drone_vel_xy * target_dir).sum(dim=-1) / (target_speed + eps)
        progress_r = progress_r.clamp(0.0, 1.5)

        drone_speed_xy = drone_vel_xy.norm(dim=-1)
        overspeed_margin = self.cfg.velocity_follow_overspeed_margin
        overspeed = (drone_speed_xy - (target_speed + overspeed_margin)).clamp(0.0) / (
            target_speed + overspeed_margin + eps
        )

        vel_follow_per_drone = self.cfg.velocity_follow_weight * (
            vel_match
            + self.cfg.velocity_follow_progress_weight * progress_r
            - self.cfg.velocity_follow_overspeed_weight * overspeed
        )  # (E, D)
        velocity_follow_reward = vel_follow_per_drone.mean(dim=-1)  # (E,)

        # =========================
        # 6) 动作平滑奖励（exp-decay 正奖励，与 move 对齐）
        # =========================
        current_actions = torch.stack(
            [self.actions[agent] for agent in self.cfg.possible_agents], dim=1
        )  # (E, D, A)
        prev_actions_t = torch.stack(
            [self.prev_actions[agent] for agent in self.cfg.possible_agents], dim=1
        )  # (E, D, A)
        smooth_sigma = self.cfg.action_smoothness_sigma
        action_delta_sq = ((current_actions - prev_actions_t) ** 2).mean(dim=-1)  # (E, D)
        action_smoothness_per_drone = self.cfg.action_smoothness_weight * torch.exp(
            -action_delta_sq / (smooth_sigma ** 2 + eps)
        )  # (E, D)
        action_smoothness = action_smoothness_per_drone.mean(dim=-1)  # (E,)

        # =========================
        # 7) 角速度奖励（exp-decay 正奖励，与 move 对齐）
        # =========================
        body_rate_reward_per_drone = self.cfg.body_rate_penalty_weight * torch.exp(
            -drone_ang_vel.norm(dim=-1)
        )  # (E, D)
        body_rate_reward = body_rate_reward_per_drone.mean(dim=-1)  # (E,)

        # =========================
        # 8) 速度奖励（exp-decay 正奖励，与 move 对齐）
        # =========================
        velocity_reward_per_drone = self.cfg.velocity_penalty_weight * torch.exp(
            -drone_vel.norm(dim=-1)
        )  # (E, D)
        velocity_reward = velocity_reward_per_drone.mean(dim=-1)  # (E,)

        # =========================
        # 9) 推力奖励（exp-decay 正奖励，与 move 对齐）
        # =========================
        all_forces = torch.stack([f[..., 2] for f in self._forces], dim=1)  # (E, D, 4)
        effort_max = (all_forces / (self.cfg.max_thrust_pp + eps)).view(self.num_envs, -1).max(dim=-1)[0]  # (E,)
        force_reward = self.cfg.force_penalty_weight * torch.exp(-effort_max)  # (E,)

        # =========================
        # 10) 高度奖励（exp-decay 正奖励，与 move 对齐）
        # =========================
        height_sigma = self.cfg.height_reward_sigma
        height_error_raw = (drone_z - self.cfg.desired_height).abs().mean(dim=-1)  # (E,)
        height_reward = self.cfg.height_reward_weight * torch.exp(
            -(height_error_raw / (height_sigma + eps)) ** 2
        )  # (E,)

        # =========================
        # 11) 竖直对齐奖励（与 move 对齐）
        # =========================
        self.drone_rot_matrices[:] = matrix_from_quat(self.drone_orientations)
        z_axis = self.drone_rot_matrices[:, :, 2, 2]  # (E, D)
        upright_penalty = self.cfg.upright_penalty_weight * (z_axis - 1.0).sum(dim=-1)  # (E,)

        # =========================
        # 12) 高空软惩罚 + 上升速度惩罚（river 特有，保留）
        # =========================
        alt_upper = self.cfg.altitude_upper_soft_threshold
        alt_margin = self.cfg.high_altitude_soft_margin
        alt_weight = self.cfg.high_altitude_penalty_weight
        high_alt_penalty = alt_weight * (drone_z - alt_upper).clamp(0.0) / (alt_margin + eps)  # (E, D)

        upward_vz = drone_vel_z.clamp(0.0)  # (E, D)
        vz_start = self.cfg.upward_vz_penalty_altitude_start
        vz_scale_cfg = self.cfg.upward_vz_penalty_altitude_scale
        alt_factor = 1.0 + (drone_z - vz_start).clamp(0.0) * vz_scale_cfg
        upward_vz_penalty = self.cfg.upward_vz_penalty_weight * upward_vz * alt_factor  # (E, D)

        safety_penalty = self.cfg.safety_penalty_weight * (
            high_alt_penalty + upward_vz_penalty
        ).sum(dim=-1)  # (E,)

        # =========================
        # 13) 硬约束固定惩罚（不乘 step_dt，与 move 对齐）
        # =========================
        # 机间碰撞
        dd = torch.cdist(drone_pos_xy, drone_pos_xy, p=2)  # (E, D, D)
        eye_mask = torch.eye(self._num_drones, device=self.device, dtype=torch.bool).unsqueeze(0)
        dd.masked_fill_(eye_mask, float("inf"))
        collision_penalty_r = (
            -(dd < self.cfg.drone_collision_threshold).sum(dim=(1, 2)) / 2.0
            * self.cfg.collision_penalty_scale
        )  # (E,)
        # 越界
        drone_out_r = (
            -(drone_pos.abs() > self.cfg.bounding_box_threshold).any(dim=-1).any(dim=-1).float()
            * self.cfg.drone_out_of_bounds_penalty
        )  # (E,)
        # 低空
        fly_low_r = (
            -(drone_z < self.cfg.min_altitude).any(dim=-1).float()
            * self.cfg.fly_low_penalty
        )  # (E,)
        # 非法接触（contact sensor，与 move 对齐：直接读当前步传感器数据）
        illegal_any = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for cs in self.contact_sensors:
            net_f = cs.data.net_forces_w_history
            max_f = torch.max(torch.norm(net_f, dim=-1), dim=1)[0]
            has_contact = (max_f > self.cfg.contact_sensor_threshold).any(dim=1)
            illegal_any = illegal_any | has_contact
        illegal_contact_r = -illegal_any.float() * self.cfg.illegal_contact_penalty  # (E,)

        # =========================
        # 14) 汇总（共享 per-env 奖励，与 move 对齐）
        # =========================
        total_reward = (
            distance_reward
            + tracking_reward
            + velocity_follow_reward
            + action_smoothness
            + body_rate_reward
            + velocity_reward
            + force_reward
            + height_reward
            + upright_penalty
            - safety_penalty
        ) * step_dt + (collision_penalty_r + drone_out_r + fly_low_r + illegal_contact_r)

        # =========================
        # 15) 日志
        # =========================
        self._episode_sums["distance_reward"] += distance_reward
        self._episode_sums["tracking_reward"] += tracking_reward
        self._episode_sums["velocity_follow_reward"] += velocity_follow_reward
        self._episode_sums["action_smoothness"] += action_smoothness
        self._episode_sums["body_rate_reward"] += body_rate_reward
        self._episode_sums["velocity_reward"] += velocity_reward
        self._episode_sums["force_reward"] += force_reward
        self._episode_sums["height_reward"] += height_reward
        self._episode_sums["upright_penalty"] += upright_penalty
        self._episode_sums["safety_penalty"] += safety_penalty
        self._episode_sums["collision_penalty"] += collision_penalty_r
        self._episode_sums["drone_out"] += drone_out_r
        self._episode_sums["fly_low"] += fly_low_r
        self._episode_sums["illegal_contact"] += illegal_contact_r

        return {agent: total_reward for agent in self.cfg.possible_agents}

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
        # 成功终止：_sustained_follow_timer 由 _get_rewards() 更新，此处仅判断
        sustained_success = self.all_targets_captured  # (E,), 已由 _get_rewards() 写入

        # 非法接触终止（contact sensor，与 move 对齐）
        self.illegal_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for cs in self.contact_sensors:
            net_f = cs.data.net_forces_w_history
            max_f = torch.max(torch.norm(net_f, dim=-1), dim=1)[0]
            has_contact = (max_f > self.cfg.contact_sensor_threshold).any(dim=1)
            self.illegal_contact = self.illegal_contact | has_contact

        # 无人机碰撞终止（与 move 对齐）
        drone_pos_xy_d = self.drone_positions[:, :, :2]  # (E, D, 2)
        dd = torch.cdist(drone_pos_xy_d, drone_pos_xy_d, p=2)  # (E, D, D)
        eye_mask = torch.eye(self._num_drones, device=self.device, dtype=torch.bool).unsqueeze(0)
        dd.masked_fill_(eye_mask, float("inf"))
        drone_collision = (dd < self.cfg.drone_collision_threshold).any(dim=-1).any(dim=-1)  # (E,)

        # 综合终止条件（与 move 对齐）
        terminations = falcon_fly_low | falcon_fly_high | body_pos_outside | drone_collision | self.illegal_contact | sustained_success
        timed_outs = self.time_out

        # 调试日志
        self.extras["log"] = {
            "Debug/Termination/fly_low_rate": falcon_fly_low.float().mean(),
            "Debug/Termination/fly_high_rate": falcon_fly_high.float().mean(),
            "Debug/Termination/out_of_bounds_rate": body_pos_outside.float().mean(),
            "Debug/Termination/collision_rate": drone_collision.float().mean(),
            "Debug/Termination/illegal_contact_rate": self.illegal_contact.float().mean(),
            "Debug/Termination/sustained_success_rate": sustained_success.float().mean(),
            "Debug/Termination/time_out_rate": self.time_out.float().mean(),
            "Debug/Termination/any_rate": terminations.float().mean(),
            "Debug/Termination/min_height": self.drone_positions[:, :, 2].min(dim=-1).values.mean(),
            "Debug/Termination/max_height": self.drone_positions[:, :, 2].max(dim=-1).values.mean(),
            "Debug/Success/captured_rate": self.target_captured.float().mean(),
            "Debug/Success/timer_mean": self._sustained_follow_timer.mean(),
        }
        for agent in self.cfg.possible_agents:
            if "log" not in self.extras[agent]:
                self.extras[agent]["log"] = {}
            self.extras[agent]["log"].update(self.extras["log"])

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
        self.target_captured[env_ids] = False
        self.target_captured_by[env_ids] = -1
        self.all_targets_captured[env_ids] = False
        self.illegal_contact[env_ids] = False
        self._drone_prev_vel_error[env_ids] = 0.0

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
            self._sustained_follow_timer[env_ids] >= self.cfg.sustained_follow_duration
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
        self.target_captured[env_ids] = False      # 未被捕获
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
