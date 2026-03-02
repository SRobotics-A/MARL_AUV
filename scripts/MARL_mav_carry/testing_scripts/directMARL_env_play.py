# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
DirectMARL环境测试脚本 - 多无人机悬停环境基础演示

该脚本用于测试和演示DirectMARL架构下的多无人机悬停环境。
主要功能包括：
1. 创建多环境实例进行并行测试
2. 实现基础的推力控制测试
3. 支持视频录制功能用于结果可视化
4. 验证环境的基本运行逻辑

适用于：
- DirectMARL环境功能验证
- 控制器基础测试
- 多环境并行运行测试
"""

"""启动Isaac Sim模拟器"""

import argparse
import torch

from isaaclab.app import AppLauncher

# 添加命令行参数配置
parser = argparse.ArgumentParser(description="DirectMARL多无人机悬停环境测试脚本")
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

# 导入自定义环境模块
from MARL_mav_carry_ext.tasks.directMARL.hover.marl_hover_env import MARLHoverEnv
from MARL_mav_carry_ext.tasks.directMARL.hover.marl_hover_env_cfg import MARLHoverEnvCfg

from isaaclab.envs import DirectMARLEnv
from isaaclab.utils.dict import print_dict


def main():
    """主函数 - 执行环境测试逻辑"""
    # 创建环境配置
    env_cfg = MARLHoverEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs  # 设置环境数量
    
    # 设置强化学习环境
    env = MARLHoverEnv(cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    
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
    stretch_position = torch.tensor(
        [
            [
                0.27,
                1.0867,
                1.7,       # 无人机1位置 [x, y, z]
                0.27,
                -1.0867,
                1.7,       # 无人机2位置 [x, y, z]
                -1.1367,
                0.0,
                1.7,       # 无人机3位置 [x, y, z]
            ]
        ],
        dtype=torch.float32,
    )

    # 初始化各无人机的动作张量
    falcon1_geo_tensor = torch.zeros((env.num_envs, 12), device=env.device)  # 几何控制指令
    falcon2_geo_tensor = torch.zeros((env.num_envs, 12), device=env.device)
    falcon3_geo_tensor = torch.zeros((env.num_envs, 12), device=env.device)

    falcon1_acc_tensor = torch.zeros((env.num_envs, 3), device=env.device)   # 加速度控制指令
    falcon2_acc_tensor = torch.zeros((env.num_envs, 3), device=env.device)
    falcon3_acc_tensor = torch.zeros((env.num_envs, 3), device=env.device)

    # 主循环 - 持续运行直到模拟器关闭
    while simulation_app.is_running():
        with torch.inference_mode():  # 启用推理模式以提高性能
            # 每500步重置一次环境（避免累积误差）
            if count % 500 == 0:
                env.reset()
            
            # 设置控制指令（当前使用简单的加速度控制）
            # X方向推力：20N，Z方向推力：5N（对抗重力）
            falcon1_acc_tensor[:, 0] = 20  # 无人机1 X方向推力
            falcon2_acc_tensor[:, 0] = 20  # 无人机2 X方向推力
            falcon3_acc_tensor[:, 0] = 20  # 无人机3 X方向推力
            
            falcon1_acc_tensor[:, 2] = 5   # 无人机1 Z方向推力
            falcon2_acc_tensor[:, 2] = 5   # 无人机2 Z方向推力
            falcon3_acc_tensor[:, 2] = 5   # 无人机3 Z方向推力
            
            # 构造动作字典
            action = {
                "falcon1": falcon1_acc_tensor,
                "falcon2": falcon2_acc_tensor,
                "falcon3": falcon3_acc_tensor,
            }
            
            # 执行环境步进
            obs, rew, terminated, truncated, info = env.step(action)
            
            # 检查终止条件
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
