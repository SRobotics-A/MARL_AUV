"""Helper class to plot results of ManagerBasedRLEnv"""

"""DirectMARL环境结果绘图助手类

该模块提供了一个专门用于可视化DirectMARL环境训练结果的绘图工具。
主要功能包括：

核心能力：
1. **多维度数据收集**：自动收集无人机、负载的各种状态数据
2. **实时指标监控**：跟踪训练过程中的关键性能指标
3. **对比可视化**：同时显示参考值和实际值的对比曲线
4. **多图表展示**：自动分页显示大量数据曲线
5. **灵活保存**：支持多种格式的图表保存

适用场景：
- 训练过程监控和调试
- 控制器性能分析
- 算法效果验证
- 实验结果展示
"""

import math
import matplotlib.pyplot as plt
import os
import torch

from isaaclab.envs import DirectMARLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply


class DirectMARLPlotter:
    """
    DirectMARL环境数据收集和可视化工具类
    
    该类专门设计用于收集和绘制DirectMARL环境中多无人机系统的
    各种状态数据，包括位置、姿态、速度、加速度等关键信息。
    
    数据收集机制：
    - 自动识别环境中的负载和无人机实体
    - 实时采集物理状态和控制参考值
    - 支持几何控制器和INDI控制器的数据收集
    - 提供滤波前后数据的对比分析
    """

    def __init__(self, env: DirectMARLEnv, control_mode: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
        """
        初始化绘图器
        
        Args:
            env: DirectMARL环境实例
            control_mode: 控制模式 ("geometric" 或 "ACCBR")
            asset_cfg: 场景实体配置，默认为"robot"
        """
        # 环境相关属性
        self.control_mode = control_mode  # 控制模式
        self.env = env                    # 环境实例
        self.robot = env.scene[asset_cfg.name]  # 机器人实体
        # 负载ID（若环境不含负载则为None）
        self.load_id = None
        try:
            self.load_id = self.robot.find_bodies("load_odometry_sensor_link")[0]
        except Exception:
            self.load_id = None

        # 无人机索引（兼容不同命名）
        try:
            self.drone_idx = self.robot.find_bodies("Falcon.*base_link_inertia")[0]
        except Exception:
            self.drone_idx = self.robot.find_bodies("Falcon.*base_link")[0]
        self.sim_dt = env.sim.get_rendering_dt()  # 仿真时间步长

        # 数据缓冲区
        self.metrics: dict = {}           # 训练指标数据
        self.load_data: dict = {}         # 负载状态数据
        self.drone_data_by_id: dict = {}  # 按无人机ID分类的数据

    def collect_metrics(self):
        """
        收集环境的训练指标数据
        
        收集来自环境的性能指标，如奖励值、成功率等统计信息。
        第一次调用时初始化数据结构，后续调用追加新数据。
        """
        if not self.metrics:
            # 首次调用：初始化指标数据结构
            self.metrics = self.env.metrics.copy()
            for key in self.metrics:
                self.metrics[key] = self.metrics[key].tolist()
        else:
            # 后续调用：追加新的指标数据
            for key in self.metrics:
                self.metrics[key].append(self.env.metrics[key].item())

    def collect_load_data(self):
        """
        收集负载的相关数据
        
        收集负载的位置、姿态、速度、角速度等状态信息，
        以及对应的参考值，用于对比分析控制效果。
        """
        if self.load_id is None:
            return
        # 收集负载的实际状态数据
        load_pos = self.robot.data.body_com_state_w[:, self.load_id, :3].squeeze(1)[0]           # 位置
        load_orientation = self.robot.data.body_com_state_w[:, self.load_id, 3:7].squeeze(1)[0]  # 姿态四元数
        load_vel = self.robot.data.body_com_state_w[:, self.load_id, 7:10].squeeze(1)[0]         # 线速度
        load_ang_vel = self.robot.data.body_com_state_w[:, self.load_id, 10:].squeeze(1)[0]      # 角速度
        load_acc = self.robot.data.body_com_state_w[:, self.load_id, 10:].squeeze(1)[0]          # 加速度
        load_ang_acc = self.robot.data.body_com_state_w[:, self.load_id, 10:].squeeze(1)[0]      # 角加速度

        # 收集负载的参考值（目标值）
        load_pos_ref = self.env.pose_command_w[:, :3][0]        # 位置参考值
        load_orientation_ref = self.env.pose_command_w[..., 3:7][0]  # 姿态参考值

        # 将参考值和实际值拼接，便于在同一图中对比显示
        both_load_pos = torch.cat((load_pos_ref, load_pos), dim=-1)              # 位置对比
        both_load_orientation = torch.cat((load_orientation_ref, load_orientation), dim=-1)  # 姿态对比

        if not self.load_data:
            # 首次调用：初始化负载数据结构
            self.load_data = {
                "both_load_pos": both_load_pos.unsqueeze(0).tolist(),           # 位置对比数据
                "both_load_orientation": both_load_orientation.unsqueeze(0).tolist(),  # 姿态对比数据
                "load_vel": load_vel.unsqueeze(0).tolist(),                     # 负载速度
                "load_ang_vel": load_ang_vel.unsqueeze(0).tolist(),             # 负载角速度
                "load_acc": load_acc.unsqueeze(0).tolist(),                     # 负载加速度
                "load_ang_acc": load_ang_acc.unsqueeze(0).tolist(),             # 负载角加速度
            }
        else:
            # 后续调用：追加新的负载数据
            self.load_data["both_load_pos"].append(both_load_pos.tolist())
            self.load_data["both_load_orientation"].append(both_load_orientation.tolist())
            self.load_data["load_vel"].append(load_vel.tolist())
            self.load_data["load_ang_vel"].append(load_ang_vel.tolist())
            self.load_data["load_acc"].append(load_acc.tolist())
            self.load_data["load_ang_acc"].append(load_ang_acc.tolist())

    def collect_drone_data(self):
        """
        收集各无人机的详细状态数据
        
        针对每架无人机收集位置、姿态、速度、角速度、加速度等信息，
        同时收集控制器的参考值和滤波处理前后的数据，支持不同控制模式。
        """
        # 收集所有无人机的基本状态数据
        drone_pos = self.robot.data.body_com_state_w[:, self.drone_idx, :3][0]        # 位置
        drone_orientation = self.robot.data.body_com_state_w[:, self.drone_idx, 3:7][0]  # 姿态
        drone_vel = self.robot.data.body_com_state_w[:, self.drone_idx, 7:10][0]      # 线速度
        drone_ang_vel = self.robot.data.body_com_state_w[:, self.drone_idx, 10:][0]   # 角速度
        drone_BR = quat_apply(drone_orientation.unsqueeze(0), drone_ang_vel.unsqueeze(0))[0]  # 机体角速度
        drone_acc = self.robot.data.body_acc_w[:, self.drone_idx, :3][0]              # 加速度
        drone_ang_acc = self.robot.data.body_acc_w[:, self.drone_idx, 3:6][0]         # 角加速度
        drone_jerk = self.env._drone_jerk[0]                                          # 加加速度
        rotor_forces = self.env._forces[0][..., 2]                                    # 旋翼推力(3*4个)
        
        # 收集策略输出的参考值
        policy_refs = self.env.actions
        policy_ref = torch.cat([action[0] for drone_num, action in policy_refs.items()])
        action_space = policy_ref.shape[-1] / drone_pos.shape[0]  # 计算每架无人机的控制维度

        # 初始化无人机数据存储字典
        if not hasattr(self, "drone_data_by_id"):
            self.drone_data_by_id = {}

        # 遍历所有无人机收集数据
        for drone_num in range(drone_pos.shape[0]):
            # 获取该无人机的策略参考值
            ref_drone = policy_ref[drone_num * int(action_space) : (drone_num + 1) * int(action_space)]
            
            # 收集几何控制器的滤波数据
            filtered_acc = self.env.geo_controllers[drone_num].filtered_acc[0]        # 滤波后加速度
            filtered_rate = self.env.geo_controllers[drone_num].filtered_rate[0]      # 滤波后角速度
            unfiltered_thrusts_geo = self.env.geo_controllers[drone_num].unfiltered_thrusts[0]  # 未滤波推力
            filtered_thrusts_geo = self.env.geo_controllers[drone_num].filtered_thrusts[0]      # 滤波后推力
            acc_load = self.env.geo_controllers[drone_num].acc_load_debug[0]          # 负载加速度估计

            # 收集INDI控制器的滤波数据
            unfiltered_mot = self.env._indi_controllers[drone_num].unfiltered_mot[0]  # 未滤波力矩
            filtered_mot = self.env._indi_controllers[drone_num].filtered_mot[0]      # 滤波后力矩
            filtered_ang_acc = self.env._indi_controllers[drone_num].filtered_ang_acc[0]  # 滤波后角加速度

            if self.control_mode == "ACCBR":
                # ACCBR控制模式数据收集
                ref_acc = ref_drone[:3]  # 参考加速度
                ref_BR = torch.cat((ref_drone[3:], torch.zeros((1), device="cuda")), dim=-1)  # 参考机体角速度
                
                # 拼接参考值和实际值用于对比
                both_drone_acc = torch.cat((ref_acc, drone_acc[drone_num]), dim=-1)    # 加速度对比
                both_drone_BR = torch.cat((ref_BR, drone_BR[drone_num]), dim=-1)       # 机体角速度对比
                both_filter_acc = torch.cat((drone_acc[drone_num], filtered_acc), dim=-1)  # 加速度滤波对比
                both_filter_rate = torch.cat((drone_ang_vel[drone_num], filtered_rate), dim=-1)  # 角速度滤波对比
                both_filter_cthrust = torch.cat((unfiltered_thrusts_geo, filtered_thrusts_geo), dim=-1)  # 推力滤波对比
                both_filter_ang_acc = torch.cat((drone_ang_acc[drone_num], filtered_ang_acc), dim=-1)    # 角加速度滤波对比
                both_filter_mot = torch.cat((unfiltered_mot, filtered_mot), dim=-1)    # 力矩滤波对比

                # 初始化或追加该无人机的数据
                if drone_num not in self.drone_data_by_id:
                    self.drone_data_by_id[drone_num] = {
                        "drone_pos": drone_pos[drone_num].unsqueeze(0).tolist(),           # 无人机位置
                        "drone_orientation": drone_orientation[drone_num].unsqueeze(0).tolist(),  # 无人机姿态
                        "drone_vel": drone_vel[drone_num].unsqueeze(0).tolist(),          # 无人机速度
                        "drone_ang_vel": drone_ang_vel[drone_num].unsqueeze(0).tolist(),  # 无人机角速度
                        "both_filter_rate_geo": both_filter_rate.unsqueeze(0).tolist(),   # 角速度滤波对比
                        "both_drone_BR": both_drone_BR.unsqueeze(0).tolist(),             # 机体角速度对比
                        "both_drone_acc": both_drone_acc.unsqueeze(0).tolist(),           # 加速度对比
                        "both_filter_acc_geo": both_filter_acc.unsqueeze(0).tolist(),     # 加速度滤波对比
                        "both_filter_ang_acc_indi": both_filter_ang_acc.unsqueeze(0).tolist(),  # 角加速度滤波对比
                        "drone_jerk": drone_jerk[drone_num].unsqueeze(0).tolist(),        # 无人机加加速度
                        "both_filter_cthrust_geo": both_filter_cthrust.unsqueeze(0).tolist(),  # 推力滤波对比
                        "both_filter_mot_indi": both_filter_mot.unsqueeze(0).tolist(),    # 力矩滤波对比
                        "rotor_forces": rotor_forces[(drone_num * 4) : (drone_num * 4) + 4].unsqueeze(0).tolist(),  # 旋翼推力
                        "acc_load": acc_load.unsqueeze(0).tolist(),                       # 负载加速度
                    }
                else:
                    # 追加新数据
                    self.drone_data_by_id[drone_num]["drone_pos"].append(drone_pos[drone_num].tolist())
                    self.drone_data_by_id[drone_num]["drone_orientation"].append(drone_orientation[drone_num].tolist())
                    self.drone_data_by_id[drone_num]["drone_vel"].append(drone_vel[drone_num].tolist())
                    self.drone_data_by_id[drone_num]["drone_ang_vel"].append(drone_ang_vel[drone_num].tolist())
                    self.drone_data_by_id[drone_num]["both_filter_rate_geo"].append(both_filter_rate.tolist())
                    self.drone_data_by_id[drone_num]["both_drone_BR"].append(both_drone_BR.tolist())
                    self.drone_data_by_id[drone_num]["both_drone_acc"].append(both_drone_acc.tolist())
                    self.drone_data_by_id[drone_num]["both_filter_acc_geo"].append(both_filter_acc.tolist())
                    self.drone_data_by_id[drone_num]["both_filter_ang_acc_indi"].append(both_filter_ang_acc.tolist())
                    self.drone_data_by_id[drone_num]["drone_jerk"].append(drone_jerk[drone_num].tolist())
                    self.drone_data_by_id[drone_num]["both_filter_cthrust_geo"].append(both_filter_cthrust.tolist())
                    self.drone_data_by_id[drone_num]["both_filter_mot_indi"].append(both_filter_mot.tolist())
                    self.drone_data_by_id[drone_num]["rotor_forces"].append(
                        rotor_forces[(drone_num * 4) : (drone_num * 4) + 4].tolist()
                    )
                    self.drone_data_by_id[drone_num]["acc_load"].append(acc_load.tolist())

            elif self.control_mode == "geometric":
                # 几何控制模式数据收集
                ref_pos = ref_drone[:3]      # 参考位置
                ref_vel = ref_drone[3:6]     # 参考速度
                ref_acc = ref_drone[6:9]     # 参考加速度
                ref_jerk = ref_drone[9:12]   # 参考加加速度
                
                # 拼接参考值和实际值
                both_drone_pos = torch.cat((ref_pos, drone_pos[drone_num]), dim=-1)    # 位置对比
                both_drone_vel = torch.cat((ref_vel, drone_vel[drone_num]), dim=-1)    # 速度对比
                both_drone_acc = torch.cat((ref_acc, drone_acc[drone_num]), dim=-1)    # 加速度对比
                both_drone_jerk = torch.cat((ref_jerk, drone_jerk[drone_num]), dim=-1)  # 加加速度对比
                both_filter_acc = torch.cat((drone_acc[drone_num], filtered_acc), dim=-1)  # 加速度滤波对比
                both_filter_rate = torch.cat((drone_ang_vel[drone_num], filtered_rate), dim=-1)  # 角速度滤波对比
                both_filter_cthrust = torch.cat((unfiltered_thrusts_geo, filtered_thrusts_geo), dim=-1)  # 推力滤波对比
                both_filter_ang_acc = torch.cat((drone_ang_acc[drone_num], filtered_ang_acc), dim=-1)    # 角加速度滤波对比
                both_filter_mot = torch.cat((unfiltered_mot, filtered_mot), dim=-1)    # 力矩滤波对比

                # 初始化或追加该无人机的数据
                if drone_num not in self.drone_data_by_id:
                    self.drone_data_by_id[drone_num] = {
                        "both_drone_pos": both_drone_pos.unsqueeze(0).tolist(),           # 位置对比
                        "drone_orientation": drone_orientation[drone_num].unsqueeze(0).tolist(),  # 姿态
                        "both_drone_vel": both_drone_vel.unsqueeze(0).tolist(),           # 速度对比
                        "drone_ang_vel": drone_ang_vel[drone_num].unsqueeze(0).tolist(),  # 角速度
                        "both_filter_rate_geo": both_filter_rate.unsqueeze(0).tolist(),   # 角速度滤波对比
                        "both_drone_acc": both_drone_acc.unsqueeze(0).tolist(),           # 加速度对比
                        "both_filter_acc_geo": both_filter_acc.unsqueeze(0).tolist(),     # 加速度滤波对比
                        "both_filter_ang_acc_indi": both_filter_ang_acc.unsqueeze(0).tolist(),  # 角加速度滤波对比
                        "both_drone_jerk": both_drone_jerk.unsqueeze(0).tolist(),         # 加加速度对比
                        "both_filter_cthrust_geo": both_filter_cthrust.unsqueeze(0).tolist(),  # 推力滤波对比
                        "both_filter_mot_indi": both_filter_mot.unsqueeze(0).tolist(),    # 力矩滤波对比
                        "rotor_forces": rotor_forces[(drone_num * 4) : (drone_num * 4) + 4].unsqueeze(0).tolist(),  # 旋翼推力
                    }
                else:
                    # 追加新数据
                    self.drone_data_by_id[drone_num]["both_drone_pos"].append(both_drone_pos.tolist())
                    self.drone_data_by_id[drone_num]["drone_orientation"].append(drone_orientation[drone_num].tolist())
                    self.drone_data_by_id[drone_num]["both_drone_vel"].append(both_drone_vel.tolist())
                    self.drone_data_by_id[drone_num]["drone_ang_vel"].append(drone_ang_vel[drone_num].tolist())
                    self.drone_data_by_id[drone_num]["both_filter_rate_geo"].append(both_filter_rate.tolist())
                    self.drone_data_by_id[drone_num]["both_drone_acc"].append(both_drone_acc.tolist())
                    self.drone_data_by_id[drone_num]["both_filter_acc_geo"].append(both_filter_acc.tolist())
                    self.drone_data_by_id[drone_num]["both_filter_ang_acc_indi"].append(both_filter_ang_acc.tolist())
                    self.drone_data_by_id[drone_num]["both_drone_jerk"].append(both_drone_jerk.tolist())
                    self.drone_data_by_id[drone_num]["both_filter_cthrust_geo"].append(both_filter_cthrust.tolist())
                    self.drone_data_by_id[drone_num]["both_filter_mot_indi"].append(both_filter_mot.tolist())
                    self.drone_data_by_id[drone_num]["rotor_forces"].append(
                        rotor_forces[(drone_num * 4) : (drone_num * 4) + 4].tolist()
                    )

    def collect_data(self):
        """
        收集环境中的所有数据
        
        统一接口函数，依次调用各类数据收集方法，
        确保所有相关数据都被完整收集。
        """
        self.collect_metrics()    # 收集训练指标
        self.collect_load_data()  # 收集负载数据
        self.collect_drone_data() # 收集无人机数据

    def plot(self, save=False, save_dir="plots", file_format="png"):
        """
        绘制收集到的数据并可选择保存图表
        
        自动将所有收集到的数据整理成多个子图进行可视化显示，
        支持参考值与实际值的对比显示，以及滤波前后数据的对比。
        
        Args:
            save: 是否保存图表到文件
            save_dir: 保存目录路径
            file_format: 保存文件格式（png, pdf, svg等）
        """
        # 整合所有数据到一个字典中
        all_data = {
            **self.metrics,  # 训练指标数据
            **self.load_data,  # 负载数据
            **{
                f"{key} Drone {drone_num}": self.drone_data_by_id[drone_num][key]
                for drone_num in self.drone_data_by_id
                for key in self.drone_data_by_id[drone_num]
            },  # 各无人机数据
        }

        # 计算需要的子图数量
        num_plots = len(all_data)
        plots_per_figure = 6  # 每个图包含6个子图
        num_figures = math.ceil(num_plots / plots_per_figure)

        # 定义可重用的绘图逻辑
        def plot_entries(ax, time_data, data, colors, linestyle, labels=None):
            """
            在指定轴上绘制多条曲线
            
            Args:
                ax: matplotlib轴对象
                time_data: 时间数据
                data: 要绘制的数据
                colors: 颜色列表
                linestyle: 线型
                labels: 图例标签
            """
            for i, color in enumerate(colors):
                ax.plot(
                    time_data,
                    [entry[i] for entry in data],
                    linestyle=linestyle,
                    color=color,
                )
            if labels:
                ax.legend(labels)

        # 如果需要保存，确保存储目录存在
        if save:
            os.makedirs(save_dir, exist_ok=True)

        # 绘制所有数据
        keys = list(all_data.keys())
        time_data = [i * self.sim_dt for i in range(len(next(iter(all_data.values()))))]  # 生成时间轴

        # 分批绘制图表
        for fig_idx in range(num_figures):
            fig = plt.figure(figsize=(15, 10))  # 创建新图形
            start_idx = fig_idx * plots_per_figure
            end_idx = min(start_idx + plots_per_figure, num_plots)

            # 在当前图形中绘制子图
            for subplot_idx, key_idx in enumerate(range(start_idx, end_idx)):
                key = keys[key_idx]
                ax = plt.subplot(2, 3, subplot_idx + 1)  # 2行3列子图布局
                data = all_data[key]

                if "both" in key:
                    # 处理包含参考值和实际值的对比数据
                    if "orientation" in key or "mot" in key:
                        # 四维数据（如四元数、四个旋翼推力）
                        ref_data = [entry[:4] for entry in data]      # 参考值
                        actual_data = [entry[4:] for entry in data]   # 实际值
                        colors = ["red", "green", "blue", "purple"]   # 颜色方案
                        plot_entries(ax, time_data, ref_data, colors, linestyle="--")      # 参考值用虚线
                        plot_entries(ax, time_data, actual_data, colors, linestyle="-")   # 实际值用实线
                        if "orientation" in key:
                            ax.legend(["W_ref", "X_ref", "Y_ref", "Z_ref", "W", "X", "Y", "Z"])  # 四元数图例
                        else:
                            ax.legend(
                                ["F1", "F2", "F3", "F4", "F1_filtered", "F2_filtered", "F3_filtered", "F4_filtered"]
                            )  # 推力图例
                    else:
                        # 三维数据（如位置、速度、加速度）
                        ref_data = [entry[:3] for entry in data]      # 参考值(X,Y,Z)
                        actual_data = [entry[3:] for entry in data]   # 实际值(X,Y,Z)
                        colors = ["red", "green", "blue"]             # 颜色方案
                        plot_entries(ax, time_data, ref_data, colors, linestyle="--")      # 参考值用虚线
                        plot_entries(ax, time_data, actual_data, colors, linestyle="-")   # 实际值用实线
                        ax.legend(["X_ref", "Y_ref", "Z_ref", "X", "Y", "Z"])  # XYZ图例
                else:
                    # 处理单一数据类型
                    if "error" in key:
                        # 错误数据用红色显示
                        ax.plot(time_data, data, color="red")
                        ax.legend(["Norm error"])
                    else:
                        # 普通数据绘制
                        ax.plot(time_data, data)
                        if "orientation" in key:
                            ax.legend(["W", "X", "Y", "Z"])  # 姿态四元数
                        elif "rotor_forces" in key:
                            ax.legend(["Rotor 1", "Rotor 2", "Rotor 3", "Rotor 4"])  # 旋翼推力
                        else:
                            ax.legend(["X", "Y", "Z"])  # XYZ分量

                # 设置子图标题和标签
                ax.set_title(key)
                ax.set_xlabel("Time")
                ax.set_ylabel(key.split(" ")[0])  # 使用键名的第一部分作为Y轴标签

            # 如果需要保存，保存当前图形
            if save:
                save_path = os.path.join(save_dir, f"figure_{fig_idx + 1}.{file_format}")
                fig.savefig(save_path)
                print(f"Figure saved: {save_path}")

        plt.tight_layout()  # 调整布局
        plt.show()  # 显示图形
