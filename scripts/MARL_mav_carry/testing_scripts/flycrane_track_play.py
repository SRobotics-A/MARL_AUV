# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
FlyCrane轨迹跟踪测试脚本 - 多智能体协同控制验证

该脚本用于测试FlyCrane系统的多智能体协同控制能力。
主要特点：
1. 实现多无人机系统的协同轨迹跟踪
2. 验证载荷搬运过程中的稳定性控制
3. 测试多智能体间的协调运动能力
4. 支持视频录制功能用于结果分析

适用于：
- 多智能体协同控制算法验证
- 载荷搬运任务性能测试
- 复杂系统稳定性评估
- 控制策略可视化分析
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import torch

from isaaclab.app import AppLauncher

# 添加命令行参数
parser = argparse.ArgumentParser(description="This script demonstrates how to simulate a quadcopter.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during execution.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")

# 添加AppLauncher的命令行参数
AppLauncher.add_app_launcher_args(parser)
# 解析参数
args_cli = parser.parse_args()
if args_cli.video:
    args_cli.enable_cameras = True

# 启动Omniverse应用
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""其余代码如下"""

import gymnasium as gym
import math
import matplotlib.pyplot as plt

from MARL_mav_carry_ext.tasks.managerbased.track import TrackEnvCfg

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.dict import print_dict


def main():
    """主程序"""
    # 创建环境配置
    env_cfg = TrackEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    # 设置强化学习环境，如果启用视频录制则使用rgb_array渲染模式
    env = ManagerBasedRLEnv(cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.video:
        # 视频录制参数配置
        video_kwargs = {
            "video_folder": "./videos",  # 视频存储文件夹
            "step_trigger": lambda step: step == 0,  # 视频录制触发条件（在第0步开始）
            "video_length": args_cli.video_length,  # 视频长度（步数）
            "disable_logger": True,  # 禁用记录器
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        # 包装环境以支持视频录制
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # 获取机器人总质量
    robot_mass = env.scene["robot"].root_physx_view.get_masses().sum()

    # 计算重力大小
    gravity = torch.tensor(env.sim.cfg.gravity, device=env.sim.device).norm()
    # 计算各部件质量
    falcon_mass = 0.6 + 0.0042 * 4 + 0.00002  # Falcon无人机质量
    rope_mass = 0.0033692587500000004 * 7 + 0.001 * 14  # 绳索质量
    payload_mass = 1.4 + 0.00001 + 0.006  # 载荷质量
    # 计算左右两侧总质量（包括半份载荷质量）
    mass_left_side = 2 * falcon_mass + 2 * rope_mass + 0.5 * payload_mass
    mass_right_side = falcon_mass + rope_mass + 0.5 * payload_mass

    # 定义伸展状态下的无人机目标位置
    # 包含三架无人机的(x, y, z)坐标
    stretch_position = torch.tensor(
        [
            [
                2.7,
                -0.5,
                2.5,  # 无人机1位置 [x, y, z]
                2.0,
                0.0,
                2.5,  # 无人机2位置 [x, y, z]
                2.7,
                0.5,
                2.5,  # 无人机3位置 [x, y, z]
            ]
        ],
        dtype=torch.float32,
    )

    # 定义垂直上升状态下的无人机目标位置， 包含三架无人机的(x, y, z)坐标
    straight_up_position = torch.tensor(
        [
            [
                0.27,
                0.22,
                2.141,  # 无人机1位置 [x, y, z]
                0.27,
                -0.22,
                2.141,  # 无人机2位置 [x, y, z]
                -0.27,
                0.0,
                2.141,  # 无人机3位置 [x, y, z]
            ]
        ],
        dtype=torch.float32,
    )
    count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            # 每500步重置一次环境和目标点
            if count % 500 == 0:
                # env.reset()
                print("-" * 80)
                print("[INFO]: Resetting environment...")
                # 创建与动作空间相同形状的零张量作为航路点
                waypoint = torch.zeros_like(env.action_manager.action)
                # 设置三架无人机的目标位置
                # 第一架无人机目标位置
                waypoint[:, :3] = stretch_position[:, :3]
                # 第二架无人机目标位置
                waypoint[:, 12:15] = stretch_position[:, 3:6]
                # 第三架无人机目标位置
                waypoint[:, 24:27] = stretch_position[:, 6:9]
                # 注：此处可选择使用straight_up_position作为备选目标
            # 执行环境步进，获取观测、奖励、终止状态等信息
            obs, rew, terminated, truncated, info = env.step(waypoint)
            # 可选：打印仿真时间戳
            # print("sim timestep: ", env.scene["robot"].data._sim_timestamp)
            # 可选：打印杆件当前角度信息
            # print("[Env 0]: Pole joint: ", obs["policy"][0][1].item())
            # 更新步数计数器
            count += 1

            # 如果启用了视频录制且达到指定长度，则停止
            if args_cli.video:
                if count == args_cli.video_length:
                    break

    # 关闭环境
    env.close()


if __name__ == "__main__":
    # 运行主函数
    main()
    # 关闭sim应用
    simulation_app.close()
