# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
单Falcon无人机悬停测试脚本 - 基础飞行控制验证

该脚本用于测试Falcon无人机的基础悬停能力。
主要功能：
1. 测试双无人机的稳定悬停控制（当前配置）
2. 验证物理仿真环境的正确性
3. 调试外力施加和姿态控制
4. 为多无人机系统提供基础测试参考

注意：当前脚本直接通过施加外部力和力矩来实现悬停，
而不是使用完整的控制回路。主要用于验证平台基本功能。

适用于：
- 双无人机基础悬停测试
- 物理引擎参数验证
- 初始姿态设置调试
- 外部力施加机制验证
"""

"""启动Isaac Sim模拟器"""

import argparse
import torch

from isaaclab.app import AppLauncher

# 添加命令行参数配置
parser = argparse.ArgumentParser(description="单Falcon无人机悬停测试脚本")
# 添加AppLauncher命令行参数（包含图形界面、物理引擎等设置）
AppLauncher.add_app_launcher_args(parser)
# 解析命令行参数
args_cli = parser.parse_args()

# 启动模拟器应用
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""主程序逻辑"""

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

##
# 预定义配置
##
from MARL_mav_carry_ext.assets import FALCON_CFG  # isort:skip


def main():
    """主函数 - 执行双无人机悬停测试"""
    
    # 创建仿真上下文，设置仿真参数
    # device: 使用CPU进行仿真计算
    # dt: 仿真时间步长设为5ms（对应200Hz更新频率）
    sim = SimulationContext(sim_utils.SimulationCfg(device="cpu", dt=0.005))
    
    # 设置相机视角，便于观察仿真过程
    # eye: 相机位置坐标 [x, y, z]
    # target: 相机目标观察点 [x, y, z]
    sim.set_camera_view(eye=[3.5, 3.5, 3.5], target=[0.0, 0.0, 0.0])

    # 在场景中创建各种对象
    # 创建地面平面，提供视觉参考和物理碰撞
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    
    # 创建远距离光源，改善场景光照效果
    # intensity: 光照强度
    # color: 光源颜色（浅灰色）
    cfg = sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # 创建无人机机器人实例
    robot_cfg = FALCON_CFG
    
    # 在场景中生成两个Falcon无人机
    # Robot_1: 位于(1.5, 0.5, 0.42)位置
    # Robot_2: 位于(-1.5, 0.5, 0.42)位置
    robot_cfg.spawn.func("/World/Falcon/Robot_1", robot_cfg.spawn, translation=(1.5, 0.5, 0.42))
    robot_cfg.spawn.func("/World/Falcon/Robot_2", robot_cfg.spawn, translation=(-1.5, 0.5, 0.42))

    # 为机器人创建操作句柄，用于后续控制
    # prim_path使用正则表达式匹配两个机器人实例
    robot = Articulation(robot_cfg.replace(prim_path="/World/Falcon/Robot.*"))

    # 初始化仿真环境
    sim.reset()

    # 获取悬停所需的关键参数
    # 查找无人机主体链接的ID（用于施加外力）
    prop_body_ids = robot.find_bodies("Falcon_base_link")[0]
    
    # 获取无人机总质量（从物理引擎读取）
    robot_mass = robot.root_physx_view.get_masses().sum()
    print("falcon mass: ", robot_mass)
    
    # 获取重力加速度大小
    gravity = torch.tensor(sim.cfg.gravity, device=sim.device).norm()
    
    # 输出初始化完成信息
    print("[INFO]: Setup complete...")

    # 定义仿真循环参数
    sim_dt = sim.get_physics_dt()  # 获取物理仿真时间步长
    sim_time = 0.0  # 累计仿真时间
    count = 0  # 步数计数器
    
    # 获取默认根状态（位置、姿态）
    robot_root_state = robot.data.default_root_state[:, :7]
    
    # 设置初始位置偏移
    # Robot_1
    robot_root_state[0, :3] += torch.tensor([1.5, 0.5, 0.42])
    # 设置初始姿态（四元数表示）
    robot_root_state[0, 3:7] = torch.tensor([0.4645017, 0.1911519, 0.4645017, 0.7293403])
    # Robot_2  
    robot_root_state[1, :3] += torch.tensor([-1.5, 0.5, 0.42])
    
    # 主仿真循环
    while simulation_app.is_running():
        # 定期重置仿真（每200步）
        if count % 200 == 0:
            # 重置计数器
            sim_time = 0.0
            count = 0
            
            # 重置关节状态
            joint_pos, joint_vel = robot.data.default_joint_pos, robot.data.default_joint_vel
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            
            # 写入根节点位姿到仿真
            robot.write_root_pose_to_sim(robot_root_state)
            # 写入根节点速度到仿真
            robot.write_root_velocity_to_sim(robot.data.default_root_state[:, 7:])
            
            # 重置机器人内部状态
            robot.reset()
            
            # 输出重置信息
            print(">>>>>>>> Reset!")

        # 获取当前机器人状态（世界坐标系下的质心状态）
        robot_state = robot.data.body_com_state_w[0, prop_body_ids, :]
        
        # 打印关键状态信息用于调试
        print(f"robot position: {robot_state[0, :3]}")
        print(f"robot orientation: {robot_state[0, 3:7]}")
        print(f"robot linear velocity: {robot_state[0, 7:10]}")
        print(f"robot angular velocity: {robot_state[0, 10:]}")
        
        # 施加控制力和力矩使无人机悬停
        # 创建力张量：shape[num_instances, num_bodies, 3]
        forces = torch.zeros(robot.num_instances, len(prop_body_ids), 3, device=sim.device)
        # 创建力矩张量：与力张量形状相同
        torques = torch.zeros_like(forces)
        
        # 计算并施加升力以抵消重力
        # forces[..., 2]: 在Z轴方向施加向上的力
        # 力的大小 = 总重量 / (机体数量 * 实例数量)
        forces[..., 2] = robot_mass * gravity / (len(prop_body_ids) * robot.num_instances)
        
        # 施加Z轴方向的微小力矩（可能是为了测试旋转响应）
        torques[..., 2] = 1
        
        # 将外力和外力矩应用到机器人主体上
        robot.set_external_force_and_torque(forces, torques, body_ids=prop_body_ids)
        
        # 将所有写入的数据同步到仿真引擎
        robot.write_data_to_sim()
        
        # 执行一步仿真
        sim.step()
        
        # 更新累计仿真时间
        sim_time += sim_dt
        # 步数计数器递增
        count += 1
        
        # 更新机器人内部数据缓冲区
        robot.update(sim_dt)


if __name__ == "__main__":
    # 运行主函数
    main()
    # 关闭模拟器应用
    simulation_app.close()
