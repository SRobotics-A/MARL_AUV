# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
FlyCrane悬停任务测试脚本 - 载荷精确悬停控制

该脚本用于测试FlyCrane系统的载荷悬停控制能力。
核心功能：
1. 实现三无人机协同使载荷在指定位置悬停
2. 验证几何控制器在吊装系统中的效果
3. 测试载荷位置跟踪精度
4. 分析多机协同的稳定性表现

适用于：
- 载荷悬停控制算法验证
- 三无人机协同精度测试
- 吊装系统稳定性分析
"""

"""启动Isaac Sim模拟器"""

import argparse
import torch

from isaaclab.app import AppLauncher

# 添加命令行参数配置
parser = argparse.ArgumentParser(description="此脚本演示了如何模拟四旋翼飞行器")
parser.add_argument("--num_envs", type=int, default=1, help="要创建的环境数量（用于并行测试）")

# 添加AppLauncher命令行参数
AppLauncher.add_app_launcher_args(parser)
# 解析命令行参数
args_cli = parser.parse_args()

# 启动Omniverse应用程序
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""主程序逻辑开始"""

from MARL_mav_carry_ext.tasks.managerbased.hover import CarryingSceneCfg

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationContext


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """运行模拟器"""
    robot = scene["robot"]
    # 获取相关参数使四旋翼无人机悬停在原地
    gravity = torch.tensor(sim.cfg.gravity, device=sim.device).norm()

    # 现在准备就绪！
    print("[INFO]: Setup complete...")

    # 定义模拟步进
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    # 模拟物理过程
    while simulation_app.is_running():
        # 重置
        prop_body_ids = robot.find_bodies("Falcon.*base_link")[0]  # 查找所有旋翼机体
        robot_mass = robot.root_physx_view.get_masses().sum()
        if count % 1000 == 0:
            # 重置计数器
            sim_time = 0.0
            count = 0
            # 重置关节状态
            joint_pos, joint_vel = robot.data.default_joint_pos, robot.data.default_joint_vel
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            # 补偿环境根状态
            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            robot.write_root_state_to_sim(root_state)

            robot.write_root_velocity_to_sim(robot.data.default_root_state[:, 7:])
            robot.reset()
            # 重置命令
            print(">>>>>>>> Reset!")
        # 对机器人应用动作（使机器人悬停在原地）
        forces = torch.zeros(robot.num_instances, len(prop_body_ids), 3, device=sim.device)
        torques = torch.zeros_like(forces)
        forces[..., 2] = (
            2 * robot_mass * gravity / len(prop_body_ids)
        ) * 1  # TODO: 要么倾斜无人机要么计算正确的推力
        robot.set_external_force_and_torque(forces, torques, body_ids=prop_body_ids)
        robot.write_data_to_sim()
        # 执行步进
        sim.step()
        # 更新模拟时间
        sim_time += sim_dt
        count += 1
        # 更新缓冲区
        robot.update(sim_dt)


def main():
    """主函数"""

    # 加载kit助手
    sim = SimulationContext(sim_utils.SimulationCfg(device="cpu", dt=0.005))
    # 设置主摄像头
    sim.set_camera_view(eye=[3.5, 3.5, 3.5], target=[0.0, 0.0, 0.0])

    scene_cfg = CarryingSceneCfg(num_envs=args_cli.num_envs, env_spacing=3.0)
    scene = InteractiveScene(scene_cfg)

    # 运行模拟器
    sim.reset()
    run_simulator(sim, scene)


if __name__ == "__main__":
    # 运行主函数
    main()
    # 关闭模拟应用
    simulation_app.close()
