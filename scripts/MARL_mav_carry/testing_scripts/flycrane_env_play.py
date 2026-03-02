# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
FlyCrane环境基础测试脚本 - 三无人机吊装系统验证

该脚本用于测试FlyCrane三无人机吊装系统的基线功能。
主要功能：
1. 验证三无人机协同控制基础能力
2. 测试吊装系统的物理仿真准确性
3. 调试绳索和载荷的动力学行为
4. 为复杂吊装任务提供基础测试平台

适用于：
- FlyCrane系统基础功能验证
- 三无人机协同控制测试
- 吊装动力学仿真验证
"""

"""启动Isaac Sim模拟器"""

import argparse
import torch

from isaaclab.app import AppLauncher

# 添加命令行参数
parser = argparse.ArgumentParser(description="This script demonstrates how to simulate a quadcopter.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")  # 环境数量，默认为1

# 添加AppLauncher的命令行参数
AppLauncher.add_app_launcher_args(parser)
# 解析参数
args_cli = parser.parse_args()

# 启动Omniverse应用
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

from gymnasium.spaces import Box

# 导入悬停环境配置
from MARL_mav_carry_ext.tasks.managerbased.hover.hover_env_cfg import HoverEnvCfg

# 导入基于Manager的强化学习环境
from isaaclab.envs import ManagerBasedRLEnv


def main():
    """主函数"""
    # 创建环境配置
    env_cfg = HoverEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs  # 设置环境数量
    # 初始化强化学习环境
    env = ManagerBasedRLEnv(cfg=env_cfg)
    # 定义动作空间：连续动作空间，范围[-1.0, 1.0]，形状为(num_envs, 12)，数据类型为float32
    env.action_space = Box(-1.0, 1.0, shape=(env.scene.num_envs, 12), dtype="float32")
    
    # 获取机器人总质量
    robot_mass = env.scene["robot"].root_physx_view.get_masses().sum()
    # 获取重力加速度大小
    gravity = torch.tensor(env.sim.cfg.gravity, device=env.sim.device).norm()
    
    # 计算各部件质量
    falcon_mass = 0.6 + 0.0042 * 4 + 0.00002  # Falcon无人机质量
    rope_mass = 0.0033692587500000004 * 7 + 0.001 * 14  # 绳索总质量
    payload_mass = 1.4 + 0.00001 + 0.006  # 载荷质量
    
    # 计算左右两侧总质量（考虑载荷分配）
    mass_left_side = 2 * falcon_mass + 2 * rope_mass + 0.5 * payload_mass  # 左侧总质量（两架无人机+两段绳索+一半载荷）
    mass_right_side = falcon_mass + rope_mass + 0.5 * payload_mass  # 右侧总质量（一架无人机+一段绳索+一半载荷）
    
    # 模拟物理过程
    count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            # 环境重置逻辑
            if count % 500 == 0:
                count = 0
                env.reset()  # 重置环境
                print("-" * 80)
                print("[INFO]: Resetting environment...")  # 打印重置信息
            
            # 生成控制指令（目标力）
            waypoint = torch.zeros_like(env.action_manager.action)  # 创建与动作同形状的零张量
            # 左前无人机推力：左侧总重量的一半
            waypoint[:, 0] = mass_left_side * gravity / 2
            # 左后无人机推力：左侧总重量的一半
            waypoint[:, 4] = mass_left_side * gravity / 2
            # 右侧无人机推力：右侧总重量
            waypoint[:, 8] = mass_right_side * gravity
            # 可选：waypoint[:, 6] = 0.05  # 可能用于控制偏航或其他姿态
            
            # 执行环境步进
            obs, rew, terminated, truncated, info = env.step(waypoint * 1)
            # 打印当前杆件关节状态（已注释）
            # print("[Env 0]: Pole joint: ", obs["policy"][0][1].item())
            
            # 更新计数器
            count += 1

    # 关闭模拟器
    env.close()


if __name__ == "__main__":
    # 运行主函数
    main()
    # 关闭sim应用
    simulation_app.close()
