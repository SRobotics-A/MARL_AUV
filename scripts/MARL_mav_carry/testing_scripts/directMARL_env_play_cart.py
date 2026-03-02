# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
DirectMARL环境测试脚本 - 四无人机搬运小车环境演示

该脚本用于测试和演示DirectMARL架构下的四无人机搬运小车环境。
主要特点：
1. 支持四个无人机协同控制
2. 实现ACC-BR（加速度-角速率）控制模式
3. 测试无人机与地面小车的协同操作
4. 验证复杂多体系统的控制逻辑

适用于：
- 四无人机系统功能测试
- ACC-BR控制模式验证
- 多体协同搬运任务测试
"""

"""启动Isaac Sim模拟器"""

import argparse
import torch

from isaaclab.app import AppLauncher

# 添加命令行参数配置
parser = argparse.ArgumentParser(description="This script demonstrates how to simulate a quadcopter.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during execution.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")

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

# 导入自定义环境模块
from MARL_mav_carry_ext.tasks.directMARL.hover_flycart.marl_hover_flycart_env import MARLHoverFlycartEnv
from MARL_mav_carry_ext.tasks.directMARL.hover_flycart.marl_hover_flycart_env_cfg import MARLHoverFlycartEnvCfg

from isaaclab.envs import DirectMARLEnv
from isaaclab.utils.dict import print_dict


def main():
    """主程序逻辑开始."""
    # 创建环境配置
    env_cfg = MARLHoverFlycartEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs  # 设置环境数量
    # 设置强化学习环境
    env = MARLHoverFlycartEnv(cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # 如果启用了视频录制功能
    if args_cli.video:
        video_kwargs = {
            "video_folder": "./marl_videos",           # 视频保存文件夹
            "step_trigger": lambda step: step == 0,    # 触发录制的条件（第0步开始）
            "video_length": args_cli.video_length,     # 视频长度
            "disable_logger": True,                    # 禁用日志记录
        }
        print_dict(video_kwargs, nesting=4)  # 打印视频配置信息
        env = gym.wrappers.RecordVideo(env, **video_kwargs)  # 包装环境以支持视频录制

    count = 0  # 步数计数器

    # 测试参考位置配置（注释掉的伸展位置示例）
    # 无人机1位置 [x, y, z]
    # 无人机2位置 [x, y, z]
    # 无人机3位置 [x, y, z]
    stretch_position = torch.tensor(
        [
            [
                0.27,
                1.0867,
                1.7,    # 无人机1位置 [x, y, z]
                0.27,
                -1.0867,
                1.7,    # 无人机2位置 [x, y, z]
                -1.1367,
                0.0,
                1.7,    # 无人机3位置 [x, y, z]
            ]
        ],
        dtype=torch.float32,
    )

    # 初始化各无人机的动作张量
    falcon1_geo_tensor = torch.zeros((env.num_envs, 12), device=env.device)  # 几何控制指令
    falcon2_geo_tensor = torch.zeros((env.num_envs, 12), device=env.device)
    falcon3_geo_tensor = torch.zeros((env.num_envs, 12), device=env.device)
    falcon4_geo_tensor = torch.zeros((env.num_envs, 12), device=env.device)

    falcon1_accbr_tensor = torch.zeros((env.num_envs, 5), device=env.device)   # 加速度控制指令
    falcon2_accbr_tensor = torch.zeros((env.num_envs, 5), device=env.device)
    falcon3_accbr_tensor = torch.zeros((env.num_envs, 5), device=env.device)
    falcon4_accbr_tensor = torch.zeros((env.num_envs, 5), device=env.device)

    # 主循环 - 持续运行直到模拟器关闭
    while simulation_app.is_running():
        with torch.inference_mode():  # 启用推理模式以提高性能
            # step the environment
            if count % 500 == 0:
                env.reset()
            # falcon1_geo_tensor[:, 0:3] = stretch_position[:, 0:3]
            # falcon2_geo_tensor[:, 0:3] = stretch_position[:, 3:6]
            # falcon3_geo_tensor[:, 0:3] = stretch_position[:, 6:9]
            # action = {
            #     "falcon1": falcon1_geo_tensor,
            #     "falcon2": falcon2_geo_tensor,
            #     "falcon3": falcon3_geo_tensor,
            # }
            falcon1_accbr_tensor[:, 2] = 20
            falcon2_accbr_tensor[:, 2] = 20
            falcon3_accbr_tensor[:, 2] = 20
            falcon4_accbr_tensor[:, 2] = 0
            falcon1_accbr_tensor[:, 4] = 0
            falcon2_accbr_tensor[:, 4] = 0
            falcon3_accbr_tensor[:, 4] = 0
            falcon4_accbr_tensor[:, 4] = 0
            action = {
                "falcon1": falcon1_accbr_tensor,
                "falcon2": falcon2_accbr_tensor,
                "falcon3": falcon3_accbr_tensor,
                "falcon4": falcon4_accbr_tensor,
            }
            # 执行环境步进
            obs, rew, terminated, truncated, info = env.step(action)
            # 检查终止条件
            terminated = list(terminated.values())[0]  # 获取终止状态
            truncated = list(truncated.values())[0]    # 获取截断状态
            # 如果任一环境终止或截断，则打印重置信息
            if any(terminated) or any(truncated):
                print("-" * 80)
                print("[INFO]: Resetting environment...")
            # update counter

            # if args_cli.video:
            #     if count/2 == args_cli.video_length:
            # break
            count += 1 # 更新计数器

    # 关闭环境
    env.close()


if __name__ == "__main__":
    # 运行主函数
    main()
    # 关闭模拟器应用
    simulation_app.close()
