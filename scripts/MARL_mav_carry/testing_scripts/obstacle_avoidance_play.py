# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
障碍物规避测试脚本 - 多无人机协同导航验证

该脚本用于测试多无人机系统在障碍物环境中的协同导航和避障能力。
主要功能：
1. 创建包含多个静态障碍物的复杂测试环境
2. 验证多无人机系统的路径规划和避障算法有效性
3. 测试无人机编队在受限空间中的机动能力和协同控制
4. 分析避障过程中的系统稳定性和响应特性

适用于：
- 多无人机协同避障算法验证
- 编队飞行控制系统测试
- 复杂环境下的自主导航能力评估
- 强化学习训练结果的可视化验证
"""

"""启动Isaac Sim模拟器"""

import argparse
import os
import torch

from isaaclab.app import AppLauncher

# 添加命令行参数解析
parser = argparse.ArgumentParser(description="This script demonstrates how to simulate a quadcopter.")
# 环境数量：指定要创建的并行仿真环境数量
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# 视频录制：是否在执行过程中录制视频
parser.add_argument("--video", action="store_true", default=False, help="Record videos during execution.")
# 视频长度：录制视频的步数长度
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")

# 添加AppLauncher的命令行参数
AppLauncher.add_app_launcher_args(parser)
# 解析命令行参数
args_cli = parser.parse_args()
# 如果启用了视频录制，则自动启用摄像头
if args_cli.video:
    args_cli.enable_cameras = True

# 启动Omniverse应用程序
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""其余代码如下"""

import gymnasium as gym
import math
import matplotlib.pyplot as plt

from MARL_mav_carry_ext.tasks.managerbased.obstacle_avoidance import ObstacleEnvCfg

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.dict import print_dict


def main():
    """主函数 - 执行仿真流程"""
    # 创建环境配置
    env_cfg = ObstacleEnvCfg()
    # 设置环境数量
    env_cfg.scene.num_envs = args_cli.num_envs
    # 设置强化学习环境
    env = ManagerBasedRLEnv(cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    
    # 如果需要录制视频，则设置视频录制参数
    if args_cli.video:
        video_kwargs = {
            "video_folder": "./videos",  # 视频存储文件夹
            "step_trigger": lambda step: step == 0,  # 触发录制的步数条件
            "video_length": args_cli.video_length,  # 录制视频的长度（步数）
            "disable_logger": True,  # 禁用记录器
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        # 包装环境以支持视频录制
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # 定义伸展位置 - 无人机编队的目标位置配置
    # 包含三架无人机的三维坐标 (x, y, z)
    stretch_position = torch.tensor(
        [
            [
                -2.5,
                1.0,
                1.5,  # 无人机1
                -2.5,
                0.0,
                1.5,  # 无人机2
                -3.2,
                0.5,
                1.5,  # 无人机3
            ]
        ],
        dtype=torch.float32,
    )

    # 定义垂直上升位置 - 无人机垂直上升的目标位置配置
    # 包含三架无人机的三维坐标 (x, y, z)
    straight_up_position = torch.tensor(
        [
            [
                0.27,
                0.22,
                2.141,  # 无人机1
                0.27,
                -0.22,
                2.141,  # 无人机2
                -0.27,
                0.0,
                2.141,  # 无人机3
            ]
        ],
        dtype=torch.float32,
    )
    
    # 步数计数器
    count = 0
    
    # 主仿真循环 - 持续运行直到模拟器停止
    while simulation_app.is_running():
        with torch.inference_mode():
            # 重置逻辑 - 每50步重置一次环境状态
            if count % 50 == 0:
                # env.reset()  # 重置环境（当前被注释）
                print("-" * 80)
                print("[INFO]: Resetting environment...")
                # 创建与当前动作维度相同的零张量作为航路点
                waypoint = torch.zeros_like(env.action_manager.action)
                # 设置第一架无人机的目标位置
                waypoint[:, :3] = stretch_position[:, :3]
                # 设置第二架无人机的目标位置
                waypoint[:, 12:15] = stretch_position[:, 3:6]
                # 设置第三架无人机的目标位置
                waypoint[:, 24:27] = stretch_position[:, 6:9]
                # waypoint[1] = straight_up_position
            # step the environment
            obs, rew, terminated, truncated, info = env.step(waypoint)
            # print("sim timestep: ", env.scene["robot"].data._sim_timestamp)
            # print current orientation of pole
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
    # 关闭模拟器应用程序
    simulation_app.close()
