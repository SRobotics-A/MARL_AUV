import os

import torch

from MARL_mav_carry_ext.controllers.utils import LowPassFilter

from isaaclab.utils.math import (
    euler_xyz_from_quat,
    matrix_from_quat,
    normalize,
    quat_from_matrix,
    quat_inv,
    quat_mul,
    quat_apply,
    quat_apply_inverse,
)


class GeometricController:
    """
    Falcon无人机几何控制器
    
    该控制器实现了基于几何控制理论的无人机姿态和位置控制算法。
    主要功能包括：
    1. 位置/速度/加速度控制（外环）
    2. 姿态控制（内环）  
    3. 角速度控制
    4. 低通滤波器用于传感器数据滤波
    5. 支持几何控制和ACCBR两种控制模式
    
    控制架构采用内外环级联控制结构：
    - 外环：根据位置误差计算期望加速度
    - 内环：根据期望加速度和当前姿态计算期望角速度和力矩
    """

    def __init__(self, num_envs: int, control_mode: str):
        """
        初始化几何控制器
        
        Args:
            num_envs: 并行环境数量
            control_mode: 控制模式 ('geometric' 或 'ACCBR')
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_envs = num_envs
        self.control_mode = control_mode
        self.disable_acc_load = os.environ.get("FLY_FORWARD_DISABLE_ACC_LOAD", "0") == "1"

        # 位置和速度误差限幅值，防止控制输入过大
        self.p_err_max_ = torch.full((self.num_envs, 3), torch.finfo(torch.float32).max, device=self.device)
        self.v_err_max_ = torch.full(
            (
                self.num_envs,
                3,
            ),
            torch.finfo(torch.float32).max,
            device=self.device,
        )

        # 绳索偏移量补偿（用于吊装系统）
        self.rope_offset = -0.03  # 绳索连接点相对于机体中心的Z轴偏移
        self.p_offset = torch.tensor([[0.0, 0.0, self.rope_offset]] * self.num_envs, device=self.device)
        self.integration_max = torch.tensor([0.0, 0.0, 0.0], device=self.device)

        # 仿真和无人机物理参数
        self.gravity = torch.tensor([[0.0, 0.0, -9.8066]] * self.num_envs, device=self.device)  # 重力加速度
        self.z_i = torch.tensor([[0.0, 0.0, 1.0]] * self.num_envs, device=self.device)  # 世界坐标系Z轴单位向量
        self.falcon_mass = 0.6017  # 无人机质量(kg)
        self._epsilon = torch.tensor(1e-6, device=self.device)  # 避免除零的小量

        # PID控制器参数
        # 加速度控制器增益（位置环）
        self.kp_acc = torch.tensor([4.0, 4.0, 9.0], device=self.device)  # 位置比例增益 [Kp_x, Kp_y, Kp_z]
        self.kd_acc = torch.tensor([4.0, 4.0, 6.0], device=self.device)  # 速度微分增益 [Kd_x, Kd_y, Kd_z]
        self.ki_acc = torch.tensor([0.0, 0.0, 0.0], device=self.device)  # 位置积分增益 [Ki_x, Ki_y, Ki_z]

        # 角速度控制器增益（姿态环）
        self.kp_rate = torch.tensor([25.0, 25.0, 8.0], device=self.device)  # 角速度比例增益 [Kp_roll, Kp_pitch, Kp_yaw]
        self.kp_att_xy = 150.0  # XY平面姿态控制增益
        self.kp_att_z = 5.0     # Z轴（偏航）姿态控制增益

        # TODO REMOVE - 已注释的旧参数配置
        # self.kappa = 0.022
        # self.beta = torch.deg2rad(torch.tensor([45], device=self.device))
        # self.l = 0.10606601717798213
        # self.G_1 = torch.tensor(
        #     [
        #         [1, 1, 1, 1],
        #         [
        #             self.l * torch.sin(self.beta),
        #             -self.l * torch.sin(self.beta),
        #             -self.l * torch.sin(self.beta),
        #             self.l * torch.sin(self.beta),
        #         ],
        #         [
        #             -self.l * torch.cos(self.beta),
        #             -self.l * torch.cos(self.beta),
        #             self.l * torch.cos(self.beta),
        #             self.l * torch.cos(self.beta),
        #         ],
        #         [self.kappa, -self.kappa, self.kappa, -self.kappa],
        #     ],
        #     device=self.device,
        # )
        # self.G_1_inv = torch.linalg.inv(self.G_1)
        # self.thrust_min_collective = 0.0
        # self.thrust_max_collective = 6.25 * 4  # [N]
        # self.inertia_mat = torch.diag(torch.tensor([0.00164, 0.00184, 0.0030], device=self.device))
        # self.thrust_min = 0.0
        # self.thrust_max = 6.25  # [N]
        # self.thrust_map = torch.tensor([1.562522e-06, 0.0, 0.0], device=self.device)

        # 低通滤波器配置
        self.filter_sampling_frequency = torch.full(
            (self.num_envs, 1), 300.0, device=self.device
        )  # 滤波器采样频率，与控制频率相同(Hz)
        self.filter_cutoff_frequency = torch.full(
            (self.num_envs, 1), 6.0, device=self.device
        )  # 加速度计滤波器截止频率(Hz)
        self.filter_cutoff_frequency_bodyrate = torch.full(
            (self.num_envs, 1), 20.0, device=self.device
        )  # 角速度控制滤波器截止频率(Hz)
        self.filter_init_value_acc = torch.full((self.num_envs, 3), 0.0, device=self.device)   # 加速度滤波器初始值
        self.filter_init_value_mot = torch.full((self.num_envs, 3), 0.0, device=self.device)   # 推力滤波器初始值
        self.filter_init_value_rate = torch.full((self.num_envs, 3), 0.0, device=self.device)  # 角速度滤波器初始值

        # 初始化三个低通滤波器
        self.filterAcc_ = LowPassFilter(
            self.filter_cutoff_frequency, self.filter_sampling_frequency, self.filter_init_value_acc
        )  # 加速度滤波器
        self.filterMot_ = LowPassFilter(
            self.filter_cutoff_frequency, self.filter_sampling_frequency, self.filter_init_value_mot
        )  # 推力滤波器
        self.filterRate_ = LowPassFilter(
            self.filter_cutoff_frequency_bodyrate, self.filter_sampling_frequency, self.filter_init_value_rate
        )  # 角速度滤波器

        # 调试模式开关和调试变量
        self.debug = True
        if self.debug:
            self.filtered_acc = torch.zeros((self.num_envs, 3), device=self.device)      # 滤波后的加速度
            self.filtered_rate = torch.zeros((self.num_envs, 3), device=self.device)     # 滤波后的角速度
            self.unfiltered_thrusts = torch.zeros((self.num_envs, 3), device=self.device) # 未滤波推力
            self.filtered_thrusts = torch.zeros((self.num_envs, 3), device=self.device)   # 滤波后推力
            self.acc_load_debug = torch.zeros((self.num_envs, 3), device=self.device)     # 负载加速度调试

    # function to overwrite parameters from yaml file
    # function to check if all parameters are valid

    def _move_to_device(self, device: torch.device | str) -> None:
        target_device = torch.device(device)
        if target_device == self.device:
            return

        self.device = target_device
        tensor_attrs = (
            "p_err_max_",
            "v_err_max_",
            "p_offset",
            "integration_max",
            "gravity",
            "z_i",
            "_epsilon",
            "kp_acc",
            "kd_acc",
            "ki_acc",
            "kp_rate",
            "filter_sampling_frequency",
            "filter_cutoff_frequency",
            "filter_cutoff_frequency_bodyrate",
            "filter_init_value_acc",
            "filter_init_value_mot",
            "filter_init_value_rate",
        )
        debug_tensor_attrs = (
            "filtered_acc",
            "filtered_rate",
            "unfiltered_thrusts",
            "filtered_thrusts",
            "acc_load_debug",
        )
        for attr in tensor_attrs:
            setattr(self, attr, getattr(self, attr).to(target_device))
        if self.debug:
            for attr in debug_tensor_attrs:
                setattr(self, attr, getattr(self, attr).to(target_device))

        self.filterAcc_.to(target_device)
        self.filterMot_.to(target_device)
        self.filterRate_.to(target_device)

    def getCommand(
        self,
        state: dict,
        actions: torch.tensor,
        setpoint: dict,
    ) -> torch.tensor:
        """
        计算无人机控制指令
        
        该函数实现了完整的几何控制算法，包括：
        1. 外环加速度控制（位置/速度控制）
        2. 负载加速度估计
        3. 内环姿态控制
        4. 角速度指令生成
        
        Args:
            state: 无人机当前状态字典，包含：
                  [pos, quat, lin_vel, ang_vel, lin_acc] - 位置、姿态、线速度、角速度、线加速度
            actions: 策略给出的动作，4个旋翼的推力值
            setpoint: 策略给出的设定点，包含：
                     [pos, lin_vel, lin_acc, quat, ang_vel, yaw, yaw_rate] - 位置、线速度、线加速度、姿态、角速度、偏航角、偏航角速度
            
        Returns:
            tuple: (alpha_b_des, acc_load, acc_cmd, q_cmd)
                   - alpha_b_des: 期望机体角加速度
                   - acc_load: 负载加速度估计
                   - acc_cmd: 期望加速度指令
                   - q_cmd: 期望姿态四元数
        """

        self._move_to_device(state["lin_acc"].device)

        # 计算总推力（所有旋翼推力之和）
        current_collective_thrust = actions.sum(1)  # sum over all propellors

        # 更新低通滤波器
        acc_filtered = self.filterAcc_.add(state["lin_acc"])        # 滤波加速度
        ang_vel_filtered = self.filterRate_.add(state["ang_vel"])   # 滤波角速度
        actions_filtered = self.filterMot_.add(current_collective_thrust)  # 滤波推力

        if self.debug:
            self.filtered_acc = acc_filtered
            self.filtered_rate = ang_vel_filtered
            self.unfiltered_thrusts = current_collective_thrust
            self.filtered_thrusts = actions_filtered

        # 加速度指令计算（外环控制）
        if self.control_mode == "geometric":
            # 几何控制模式：基于PID的位置控制
            # 补偿绳索偏移后的位置设定点
            p_ref_cg = setpoint["pos"] - quat_apply(state["quat"], self.p_offset)
            # 计算位置和速度误差并限幅
            pos_error = torch.clamp(p_ref_cg - state["pos"], -self.p_err_max_, self.p_err_max_)
            vel_error = torch.clamp(setpoint["lin_vel"] - state["lin_vel"], -self.v_err_max_, self.v_err_max_)
            # PID控制器计算期望加速度
            des_acc = self.kp_acc * pos_error + self.kd_acc * vel_error + setpoint["lin_acc"]

        elif self.control_mode == "ACCBR":
            # ACCBR模式：直接使用策略给出的加速度指令
            des_acc = setpoint["lin_acc"]

        # 负载加速度估计（在世界坐标系中）
        # 公式：acc_load = measured_acc - gravity - (thrust / mass) * body_z_axis
        if self.disable_acc_load or os.environ.get("FLY_FORWARD_DISABLE_ACC_LOAD", "0") == "1":
            acc_load = torch.zeros_like(state["lin_acc"])
        else:
            acc_load = (
                state["lin_acc"]
                - self.gravity
                - quat_apply(state["quat"], current_collective_thrust / self.falcon_mass)
            )

        if self.debug:
            self.acc_load_debug = acc_load

        # 计算总的加速度指令（考虑重力和负载影响）
        acc_cmd = des_acc - self.gravity - acc_load
        z_b_des = normalize(acc_cmd)  # 期望的新推力方向（机体Z轴）
        collective_thrust_des_magntiude = torch.norm(acc_cmd, dim=1, keepdim=True) * self.falcon_mass  # 期望总推力大小
        current_collective_thrust_magnitude = torch.norm(current_collective_thrust, dim=1, keepdim=True)  # 当前总推力大小

        # 姿态指令计算（内环控制）
        # 计算期望的姿态四元数
        setpoint_yaw = setpoint["yaw"]
        # 计算中间轴和新的期望机体坐标系
        x_intermediate_des = torch.cat(
            (torch.cos(setpoint_yaw), torch.sin(setpoint_yaw), torch.zeros_like(setpoint_yaw)), dim=1
        )
        # 计算期望的Y轴（通过Z轴和中间X轴的叉积）
        y_b_des = normalize(torch.linalg.cross(z_b_des, x_intermediate_des))  # / (
        # 计算期望的X轴（Y轴和Z轴的叉积）
        x_b_des = torch.linalg.cross(y_b_des, z_b_des)

        # 构造期望的旋转矩阵并转换为四元数
        des_rot_matrix = torch.stack([x_b_des, y_b_des, z_b_des], dim=2)
        q_cmd = quat_from_matrix(des_rot_matrix)
        
        # 角速度指令计算
        # 获取无人机当前的机体坐标轴
        if self.control_mode == "geometric":
            # 几何控制模式：基于微分平坦性理论计算期望角速度
            current_rot_matrix = matrix_from_quat(state["quat"])
            x_b = current_rot_matrix[..., 0]  # 当前X轴
            y_b = current_rot_matrix[..., 1]  # 当前Y轴
            z_b = current_rot_matrix[..., 2]  # 当前Z轴

            # 计算推力的时间导数
            T_dot = self.falcon_mass * torch.sum(setpoint["jerk"] * z_b, dim=-1, keepdim=True)
            # 计算Z轴的旋转导数
            h_omega = self.falcon_mass * setpoint["jerk"] - T_dot * z_b  # rotational derivative of z_b
            mask = (current_collective_thrust_magnitude > 0.01).squeeze()  # 避免除零
            h_omega[mask] /= current_collective_thrust_magnitude[mask]
            # 计算各轴的期望角速度分量
            omega_b_x = (-h_omega * y_b).sum(-1, keepdim=True)  # 滚转角速度
            omega_b_y = (h_omega * x_b).sum(-1, keepdim=True)   # 俯仰角速度
            omega_b_z = setpoint["yaw_rate"] * (self.z_i * z_b).sum(-1, keepdim=True)  # 偏航角速度
            omega_b_ref = torch.cat((omega_b_x, omega_b_y, omega_b_z), dim=-1)
            
        elif self.control_mode == "ACCBR":
            # ACCBR模式：直接使用策略给出的角速度指令
            omega_b_ref = setpoint["body_rates"]

        # 倾斜优先的姿态控制
        # 计算当前姿态与期望姿态的差异
        quat_diff = quat_mul(quat_inv(state["quat"]), q_cmd)
        q_e_w = quat_diff[..., 0].view(self.num_envs, 1)  # 实部
        q_e_x = quat_diff[..., 1].view(self.num_envs, 1)  # i虚部
        q_e_y = quat_diff[..., 2].view(self.num_envs, 1)  # j虚部
        q_e_z = quat_diff[..., 3].view(self.num_envs, 1)  # k虚部
        
        # 计算归一化因子（避免奇点）
        norm_factor = (2 / (torch.sqrt(q_e_w.square() + q_e_z.square())) + self._epsilon).view(self.num_envs, 1)
        zeros = torch.zeros((self.num_envs, 1), device=self.device)
        
        # 计算简化误差四元数（消除偏航自由度）
        q_e_red = norm_factor * torch.cat((q_e_w * q_e_x - q_e_y * q_e_z, q_e_w * q_e_y + q_e_x * q_e_z, zeros), dim=-1)
        q_e_yaw = norm_factor * torch.cat([zeros, zeros, q_e_z], dim=-1)
        
        # 将角速度转换到机体坐标系
        ang_vel_body = quat_apply(quat_inv(state["quat"]), state["ang_vel"])
        
        # 计算期望的机体角加速度（完整的姿态控制器）
        alpha_b_des = (
            self.kp_att_xy * q_e_red                           # XY姿态控制项
            + self.kp_att_z * torch.sign(q_e_w) * q_e_yaw      # Z轴姿态控制项
            + self.kp_rate * (omega_b_ref - ang_vel_body)      # 角速度控制项
        )

        # omega = quat_apply(quat_inv(state["quat"]), state["ang_vel"])  # 机体角速度 # 通常来自IMU
        # mu_ndi = torch.zeros((self.num_envs, 4), device=self.device)
        # collective_thrust_des_magntiude = torch.norm(acc_cmd, dim=1) * self.falcon_mass
        # mu_ndi[:, 0] = torch.clamp(collective_thrust_des_magntiude, self.thrust_min_collective, self.thrust_max_collective)

        # moments = self.inertia_mat.matmul(alpha_b_des.transpose(0, 1)).transpose(0, 1) + torch.linalg.cross(
        #     omega, self.inertia_mat.matmul(omega.transpose(0, 1)).transpose(0, 1)
        # ) - torch.linalg.cross(
        # self.p_offset, quat_apply(quat_inv(state["quat"]), acc_load * self.falcon_mass)) # M_load in body frame

        # mu_ndi[:, 1:] = moments
        # thrusts = self.G_1_inv.matmul(mu_ndi.transpose(0, 1))
        # thrusts = torch.clamp(thrusts, self.thrust_min, self.thrust_max)

        # rotor_speeds = torch.sqrt(thrusts / self.thrust_map[0]).transpose(0, 1)

        return alpha_b_des, acc_load, acc_cmd, q_cmd  # , rotor_speeds

    def reset(self, env_ids):
        """
        重置指定环境的滤波器状态
        
        Args:
            env_ids: 需要重置的环境索引
        """
        self.filterAcc_.reset(env_ids)   # 重置加速度滤波器
        self.filterMot_.reset(env_ids)   # 重置推力滤波器
        self.filterRate_.reset(env_ids)  # 重置角速度滤波器
        # pass
