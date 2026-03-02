# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
FlyFollow跟随环境测试脚本 - 移动目标跟踪验证

该脚本用于测试无人机对移动目标的跟随能力。
核心功能：
1. 实现无人机跟随动态移动目标
2. 验证视觉伺服和目标跟踪算法
3. 测试相对位置控制精度
4. 分析跟随过程中的稳定性表现

适用于：
- 移动目标跟随算法测试
- 视觉伺服控制验证
- 相对导航系统评估
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

import gymnasium as gym

# 导入自定义环境模块
from MARL_mav_carry_ext.tasks.directMARL.follow.marl_follow_env import MARLFollowEnv
from MARL_mav_carry_ext.tasks.directMARL.follow.marl_follow_env_cfg import MARLFollowEnvCfg
from isaaclab.utils.dict import print_dict


def main():
    """主函数 - 执行环境测试逻辑"""
    # 创建环境配置
    env_cfg = MARLFollowEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs  # 设置环境数量
    # 设置强化学习环境
    env = MARLFollowEnv(cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

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

    falcon1_geo_tensor = torch.zeros((env.num_envs, 12), device=env.device)
    falcon2_geo_tensor = torch.zeros((env.num_envs, 12), device=env.device)
    falcon3_geo_tensor = torch.zeros((env.num_envs, 12), device=env.device)
    count = 0  # 步数计数器

    while simulation_app.is_running():
        with torch.inference_mode():
            # 获取未包装的环境实例
            env_unwrapped = env.unwrapped
            
            # 初始化动作字典
            actions = {}
            
            # 从关节状态计算无人机位置
            # 获取无人机质心位置并减去环境原点得到相对位置
            drone_pos = (
                env_unwrapped.robot.data.body_com_state_w[:, env_unwrapped._falcon_idx, :3]
                - env_unwrapped.scene.env_origins.unsqueeze(1)
            )
            
            # 获取目标位置（只取x,y坐标）
            target_pos = env_unwrapped._target_positions[:, :, :2]

            # 为每个智能体计算动作
            for i, agent in enumerate(env_unwrapped.cfg.possible_agents):
                # 计算无人机到各个目标的相对位置向量
                rel = target_pos - drone_pos[:, i, :2].unsqueeze(1)
                
                # 计算距离
                dist = torch.norm(rel, dim=-1)
                
                # 找到最近的目标索引
                closest = torch.argmin(dist, dim=-1)
                
                # 获取指向最近目标的方向向量
                direction = rel[torch.arange(env_unwrapped.num_envs, device=env_unwrapped.device), closest]
                
                # 归一化方向向量（避免除零错误）
                direction = direction / (torch.norm(direction, dim=-1, keepdim=True) + 1e-6)

                # 初始化线性加速度（3维：x,y,z）
                lin_acc = torch.zeros((env_unwrapped.num_envs, 3), device=env_unwrapped.device)
                # 在x,y方向上施加推力，大小为2.0
                lin_acc[:, 0:2] = direction * 2.0
                
                # 初始化角速度（2维：绕x轴和y轴的旋转速率）
                body_rates = torch.zeros((env_unwrapped.num_envs, 2), device=env_unwrapped.device)
                
                # 将线性加速度和角速度合并为完整动作
                actions[agent] = torch.cat([lin_acc, body_rates], dim=-1)

            # 执行动作并获取新的状态
            obs, rewards, terminated, truncated, info = env.step(actions)
            
            # 检查是否需要重置环境（任一环境终止或截断）
            if any(list(terminated.values())[0]) or any(list(truncated.values())[0]):
                env.reset()

    # 关闭环境
    env.close()


if __name__ == "__main__":
    # 运行主函数
    main()
    # 关闭模拟器应用
    simulation_app.close()