# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
DirectMARL障碍物规避环境测试脚本 - 多无人机避障任务演示

该脚本用于测试和演示DirectMARL架构下的障碍物规避环境。
主要功能：
1. 创建包含静态障碍物的三维环境
2. 测试无人机在复杂环境中的路径规划能力
3. 验证避障算法的有效性
4. 支持几何控制模式进行精确位置控制

适用于：
- 障碍物规避算法测试
- 路径规划算法验证
- 复杂环境下的多无人机协调
"""

"""启动Isaac Sim模拟器"""

import argparse
import torch

from isaaclab.app import AppLauncher

# 添加命令行参数配置
parser = argparse.ArgumentParser(description="DirectMARL障碍物规避环境测试脚本")
parser.add_argument("--num_envs", type=int, default=1, help="要创建的环境数量（用于并行测试）")
parser.add_argument("--video", action="store_true", default=False, help="是否录制执行过程视频")
parser.add_argument("--video_length", type=int, default=200, help="录制视频的长度（步数）")

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

# 导入障碍物规避环境模块
from MARL_mav_carry_ext.tasks.directMARL.obstacle_avoidance.marl_obstacle_env import MARLObstacleEnv
from MARL_mav_carry_ext.tasks.directMARL.obstacle_avoidance.marl_obstacle_env_cfg import MARLObstacleEnvCfg

from isaaclab.envs import DirectMARLEnv
from isaaclab.utils.dict import print_dict


def main():
    """主函数 - 执行障碍物规避环境测试逻辑"""
    # 创建环境配置
    env_cfg = MARLObstacleEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs  # 设置环境数量
    
    # 设置强化学习环境
    env = MARLObstacleEnv(cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    
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

    # 测试参考位置配置（三角形编队位置）
    stretch_position = torch.tensor([
        [0.27, 1.0867, 1.7,      # 无人机1位置 [x, y, z] - 右前方
         0.27, -1.0867, 1.7,     # 无人机2位置 [x, y, z] - 右后方
         -1.1367, 0.0, 1.7]      # 无人机3位置 [x, y, z] - 左侧
    ], dtype=torch.float32)

    # 初始化各无人机的几何控制张量
    # 每个无人机需要12维控制向量，前3维为位置控制
    falcon1_geo_tensor = torch.zeros((env.num_envs, 12), device=env.device)
    falcon2_geo_tensor = torch.zeros((env.num_envs, 12), device=env.device)
    falcon3_geo_tensor = torch.zeros((env.num_envs, 12), device=env.device)

    # 主循环 - 持续运行直到模拟器关闭
    while simulation_app.is_running():
        with torch.inference_mode():  # 启用推理模式以提高性能
            # 每500步重置一次环境（避免累积误差）
            if count % 500 == 0:
                env.reset()
            
            # 设置几何控制指令 - 将无人机移动到指定的伸展位置
            falcon1_geo_tensor[:, 0:3] = stretch_position[:, 0:3]  # 无人机1目标位置
            falcon2_geo_tensor[:, 0:3] = stretch_position[:, 3:6]  # 无人机2目标位置
            falcon3_geo_tensor[:, 0:3] = stretch_position[:, 6:9]  # 无人机3目标位置
            
            # 构造动作字典，包含所有无人机的控制指令
            action = {
                "falcon1": falcon1_geo_tensor,
                "falcon2": falcon2_geo_tensor,
                "falcon3": falcon3_geo_tensor,
            }
            
            # 执行环境步进，获取观测、奖励、终止状态等信息
            obs, rew, terminated, truncated, info = env.step(action)
            
            # 提取终止和截断状态（从字典中获取第一个值）
            terminated = list(terminated.values())[0]  # 获取终止状态
            truncated = list(truncated.values())[0]    # 获取截断状态
            
            # 如果任一环境终止或截断，则打印重置信息
            if any(terminated) or any(truncated):
                print("-" * 80)
                print("[INFO]: 环境重置中...")
            
            # 更新计数器
            count += 1

    # 关闭环境
    env.close()


if __name__ == "__main__":
    # 运行主函数
    main()
    # 关闭模拟器应用
    simulation_app.close()
