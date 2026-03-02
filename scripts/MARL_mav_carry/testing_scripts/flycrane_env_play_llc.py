# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
FlyCrane低层控制器测试脚本 - 三无人机协同控制验证

该脚本用于测试FlyCrane系统的低层控制器性能。
主要功能：
1. 实现三无人机独立的低层控制
2. 验证几何控制器和ACCBR控制器在多机系统中的表现
3. 测试无人机编队控制精度
4. 分析多机协同的底层执行一致性

适用于：
- 多无人机低层控制算法验证
- 编队控制策略测试
- 多机协同控制精度分析
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import torch

from isaaclab.app import AppLauncher

# 添加命令行参数
parser = argparse.ArgumentParser(description="此脚本演示了如何模拟四旋翼飞行器")
parser.add_argument("--num_envs", type=int, default=1, help="要创建的环境数量（用于并行测试）")
parser.add_argument("--video", action="store_true", default=False, help="是否录制执行过程视频")
parser.add_argument("--video_length", type=int, default=200, help="录制视频的长度（步数）")
parser.add_argument("--control_mode", type=str, default="geometric", help="智能体的控制模式")

# 添加AppLauncher的命令行参数
AppLauncher.add_app_launcher_args(parser)
# 解析参数
args_cli = parser.parse_args()
if args_cli.video:
    args_cli.enable_cameras = True

# 启动Omniverse应用
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""主程序逻辑开始"""

import gymnasium as gym
import math
import matplotlib.pyplot as plt

# 导入自定义环境模块
from MARL_mav_carry_ext.plotting_tools import ManagerBasedPlotter
from MARL_mav_carry_ext.tasks.managerbased.hover_llc.hover_env_cfg import HoverEnvCfg_llc

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.dict import print_dict


def main():
    """主函数"""
    # 创建环境配置
    env_cfg = HoverEnvCfg_llc()
    env_cfg.scene.num_envs = args_cli.num_envs
    # 设置强化学习环境
    env = ManagerBasedRLEnv(cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # 创建绘图器，用于数据可视化
    plotter = ManagerBasedPlotter(env, command_name="pose_command", control_mode=args_cli.control_mode)
    
    # 如果需要录制视频
    if args_cli.video:
        video_kwargs = {
            "video_folder": "./marl_videos",           # 视频保存文件夹
            "step_trigger": lambda step: step == 0,    # 触发录制的条件（第0步开始）
            "video_length": args_cli.video_length,     # 视频长度
            "disable_logger": True,                    # 禁用日志记录
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)  # 打印视频配置信息
        env = gym.wrappers.RecordVideo(env, **video_kwargs)  # 包装环境以支持视频录制

    # 计算机器人总质量
    robot_mass = env.scene["robot"].root_physx_view.get_masses().sum()

    # 计算重力加速度大小
    gravity = torch.tensor(env.sim.cfg.gravity, device=env.sim.device).norm()
    
    # 计算各部分质量
    falcon_mass = 0.6 + 0.0042 * 4 + 0.00002  # Falcon无人机质量
    rope_mass = 0.0033692587500000004 * 7 + 0.001 * 14  # 绳索质量
    payload_mass = 1.4 + 0.00001 + 0.006  # 载荷质量
    
    # 计算两侧质量分布
    mass_left_side = 2 * falcon_mass + 2 * rope_mass + 0.5 * payload_mass  # 左侧质量
    mass_right_side = falcon_mass + rope_mass + 0.5 * payload_mass  # 右侧质量

    # 定义拉伸位置（三无人机编队的特定位置）
    stretch_position = torch.tensor(
        [
            [
                1.27,
                1.0867,
                1.7,   # 无人机1位置 [x, y, z]
                1.27,
                -1.0867,
                1.7,   # 无人机2位置 [x, y, z]
                -0.1367,
                0.0,
                1.7,   # 无人机3位置 [x, y, z]
            ]
        ],
        dtype=torch.float32,
    )

    # 定义垂直向上位置（三无人机编队的另一组特定位置）
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

    # 定义ACCBR参考值（用于ACCBR控制模式下的参考输入）
    ACC_BR_ref = torch.tensor(
        [
            [
                0.0,
                0.0,
                9.0,
                0.0,
                0.0,
                10.0,  # 无人机1
                0.0,
                0.0,
                8.0,
                0.0,
                0.0,
                10.0,  # 无人机2
                0.0,
                0.0,
                7.0,
                0.0,
                0.0,
                10.0,  # 无人机3
            ]
        ],
        dtype=torch.float32,
    )

    count = 0

    # 主循环：持续运行直到仿真应用关闭
    while simulation_app.is_running():
        with torch.inference_mode():
            # 获取Falcon无人机的位置信息
            falcon_pos = env.scene["robot"].data.body_com_state_w[:, [20, 27, 34], :3]
            
            # 每500步重置环境
            if count % 500 == 0:
                env.reset()
                print("-" * 80)
                print("[INFO]: Resetting environment...")
                
            # 创建零动作张量
            waypoint = torch.zeros_like(env.action_manager.action)
            
            # 几何控制模式
            if args_cli.control_mode == "geometric":
                waypoint[:, :3] = stretch_position[:, :3]  # 设置第一个无人机的目标位置
                waypoint[:, 12:15] = stretch_position[:, 3:6]  # 设置第二个无人机的目标位置
                waypoint[:, 24:27] = stretch_position[:, 6:9]  # 设置第三个无人机的目标位置

            # # 可选：使用当前位置作为目标位置
            # waypoint[:, :3] = falcon_pos[:, 0]
            # waypoint[:, 12:15] = falcon_pos[:, 1]
            # waypoint[:, 24:27] = falcon_pos[:, 2]

            # ACCBR控制模式
            if args_cli.control_mode == "ACCRBR":
                waypoint[:] = ACC_BR_ref  # 使用预定义的ACCBR参考值
                
            # 收集单环境数据用于绘图
            if env.num_envs == 1:
                plotter.collect_data()
                
            # 执行环境步骤
            obs, rew, terminated, truncated, info = env.step(waypoint)
            count += 1

            # 视频录制完成则退出
            if args_cli.video:
                if count == args_cli.video_length:
                    break

    # 关闭仿真环境
    env.close()

    # 单环境情况下显示或保存图表
    if args_cli.num_envs == 1:
        # if args_cli.save_plots:
        #     # 保存图表
        #     plot_path = os.path.join(log_dir, "plots", "play")
        #     plotter.plot(save=True, save_dir=plot_path)
        # else:
        # 显示图表
        plotter.plot(save=False)


if __name__ == "__main__":
    # 运行主函数
    main()
    # 关闭仿真应用
    simulation_app.close()
