# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Falcon无人机低层控制器(LLC)测试脚本 - 控制器算法验证

该脚本专门用于测试和验证Falcon无人机的低层控制器。
主要功能：
1. 测试INDI（增量式）控制器性能
2. 验证电机模型和推力分配算法
3. 调试控制器响应特性和稳定性
4. 为上层控制提供可靠的底层执行

适用于：
- 低层控制器算法验证
- 推力分配系统测试
- 电机响应特性分析
"""

"""启动Isaac Sim模拟器"""

import argparse
import torch

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates how to simulate a quadcopter.")
parser.add_argument("--num_envs", type=int, default=1, help="要创建的环境数量")
parser.add_argument("--video", action="store_true", default=False, help="是否录制执行过程视频")
parser.add_argument("--video_length", type=int, default=200, help="录制视频的长度（步数） ")

# 添加AppLauncher命令行参数
AppLauncher.add_app_launcher_args(parser)
# 解析命令行参数
args_cli = parser.parse_args()
if args_cli.video:
    args_cli.enable_cameras = True  # 启用相机以支持视频录制

# 启动Omniverse应用程序
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""主程序逻辑开始"""

import csv
import gymnasium as gym
import matplotlib.pyplot as plt

from MARL_mav_carry_ext.tasks.single_falcon.track_ref import FalconEnv, FalconEnvCfg

from isaaclab.envs import DirectRLEnv
from isaaclab.utils.dict import print_dict
from isaaclab.utils.timer import Timer


def main():
    """主函数 - 执行环境测试逻辑"""
    # 创建环境配置
    env_cfg = FalconEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    # 设置强化学习环境
    env = FalconEnv(cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    
    # 设置强化学习环境
    if args_cli.video:
        video_kwargs = {
            "video_folder": "./falcon_videos",           # 视频保存文件夹
            "step_trigger": lambda step: step == 0,      # 触发录制的条件（第0步开始）
            "video_length": args_cli.video_length,       # 视频长度
            "disable_logger": True,                      # 禁用日志记录
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)  # 打印视频配置信息
        env = gym.wrappers.RecordVideo(env, **video_kwargs)  # 包装环境以支持视频录制

    count = 0  # 步数计数器

    # 主循环 - 持续运行直到模拟器关闭
    while simulation_app.is_running():
        with torch.inference_mode():  # 启用推理模式以提高性能
            # step the environment

            # 执行环境步进
            obs, rew, terminated, truncated, info = env.step(torch.tensor([0.0], device=env.device))
            if terminated.any() | truncated.any():   # 如果任一环境终止或截断，则打印重置信息
                print("-" * 80)
                print("[INFO]: Resetting environment...")
            # update counter

            # if args_cli.video:
            #     if count/2 == args_cli.video_length:
            # break

    # 关闭环境
    env.close()


if __name__ == "__main__":
    # 运行主函数
    main()
    # 关闭模拟器应用
    simulation_app.close()
