import torch

from MARL_mav_carry_ext.controllers.utils import LowPassFilter

from isaaclab.utils.math import quat_inv, quat_apply


class IndiController:
    """
    INDI（Incremental Nonlinear Dynamic Inversion）增量式非线性动态反演控制器
    
    该控制器实现了INDI控制算法，主要用于将高层控制指令（角加速度、期望加速度）
    转换为具体的旋翼转速指令。INDI是一种先进的飞行器控制方法，具有以下特点：
    
    核心优势：
    1. **模型无关性**：不需要精确的系统动力学模型
    2. **增量式更新**：基于控制量的增量进行调节
    3. **快速响应**：对扰动和模型不确定性具有良好的鲁棒性
    4. **实时性好**：计算复杂度相对较低，适合实时控制
    
    控制原理：
    - 接收来自几何控制器的期望角加速度和期望推力
    - 通过推力分配矩阵将控制力矩分配给四个旋翼
    - 考虑电机动力学和惯性效应
    - 输出每个旋翼的目标转速
    """

    def __init__(self, num_envs: int):
        """
        初始化INDI控制器
        
        Args:
            num_envs: 并行环境数量
        """
        self.num_envs = num_envs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 推力分配矩阵参数定义
        self.kappa = 0.022  # 扭矩系数（推力差产生的扭矩比例）
        self.beta = torch.deg2rad(torch.tensor([45], device=self.device))  # 旋翼倾斜角度（弧度）
        self.l = 0.10606601717798213  # 旋翼臂长（米）
        
        # 推力分配矩阵G1：将4个旋翼推力转换为[总推力, 滚转力矩, 俯仰力矩, 偏航力矩]
        self.G_1 = torch.tensor(
            [
                [1, 1, 1, 1],  # 总推力：四个旋翼推力之和
                [
                    self.l * torch.sin(self.beta),      # 滚转力矩：右侧旋翼推力差
                    -self.l * torch.sin(self.beta),     # 左侧旋翼推力差
                    -self.l * torch.sin(self.beta),     # 左侧旋翼推力差
                    self.l * torch.sin(self.beta),      # 右侧旋翼推力差
                ],
                [
                    -self.l * torch.cos(self.beta),     # 俯仰力矩：前后旋翼推力差
                    -self.l * torch.cos(self.beta),     # 前侧旋翼推力差
                    self.l * torch.cos(self.beta),      # 后侧旋翼推力差
                    self.l * torch.cos(self.beta),      # 后侧旋翼推力差
                ],
                [self.kappa, -self.kappa, self.kappa, -self.kappa],  # 偏航力矩：对角旋翼推力差
            ],
            device=self.device,
        )
        self.G_1_inv = torch.linalg.inv(self.G_1)  # G1矩阵的逆矩阵，用于反向推力分配

        # 电机惯性参数
        self.motor_inertia_z = 9.3575e-6  # 电机转子绕Z轴转动惯量[kgm^2]
        # 电机惯性矩阵G2：考虑电机加速时产生的反作用扭矩
        self.G_2 = torch.zeros((4, 4), device=self.device)
        self.G_2[3, :] = torch.tensor(
            [self.motor_inertia_z, -self.motor_inertia_z, self.motor_inertia_z, -self.motor_inertia_z],
            device=self.device,
        )

        # 旋翼和推力限制参数
        self.motor_omega_min = 150.0      # 电机最小转速[rad/s]
        self.motor_omega_max = 2800.0     # 电机最大转速[rad/s]
        self.thrust_min = 0.0             # 单个旋翼最小推力[N]
        self.thrust_max = 6.25            # 单个旋翼最大推力[N]
        self.thrust_min_collective = 0.0  # 总推力最小值[N]
        self.thrust_max_collective = self.thrust_max * 4  # 总推力最大值[N]
        self.thrust_map = torch.tensor([1.562522e-06, 0.0, 0.0], device=self.device)  # 推力-转速映射系数

        # 无人机物理参数
        self.falcon_mass = 0.6017  # 无人机质量[kg]
        self.inertia_mat = torch.diag(torch.tensor([0.00164, 0.00184, 0.0030], device=self.device))  # 转动惯量矩阵
        self.rope_offset = -0.03   # 绳索偏移量（用于吊装系统）
        self.p_offset = torch.tensor([[0.0, 0.0, self.rope_offset]] * self.num_envs, device=self.device)

        # TODO REMOVE - 已注释的旧参数
        # self.kp = torch.tensor([100.0, 100.0, 10.0], device=self.device)

        # 低通滤波器配置
        self.filter_sampling_frequency = torch.full(
            (self.num_envs, 1), 300.0, device=self.device
        )  # 滤波器采样频率，与控制频率相同[Hz]
        self.filter_cutoff_frequency = torch.full(
            (self.num_envs, 1), 12.0, device=self.device
        )  # 加速度计滤波器截止频率[Hz]
        self.filter_init_value_mot = torch.full((self.num_envs, 4), 0.0, device=self.device)   # 推力滤波器初始值
        self.filter_init_value_rate = torch.full((self.num_envs, 3), 0.0, device=self.device)  # 角速度滤波器初始值

        # 初始化低通滤波器
        self.filterMot_ = LowPassFilter(
            self.filter_cutoff_frequency, self.filter_sampling_frequency, self.filter_init_value_mot
        )  # 推力滤波器
        self.filterRate_ = LowPassFilter(
            self.filter_cutoff_frequency, self.filter_sampling_frequency, self.filter_init_value_rate
        )  # 角速度滤波器

        # 调试模式和变量
        self.debug = True
        if self.debug:
            self.filtered_ang_acc = torch.zeros((self.num_envs, 3), device=self.device)  # 滤波后的角加速度
            self.unfiltered_mot = torch.zeros((self.num_envs, 4), device=self.device)    # 未滤波的推力
            self.filtered_mot = torch.zeros((self.num_envs, 4), device=self.device)      # 滤波后的推力

    def getCommand(
        self,
        state: dict,
        actions: torch.tensor,
        alpha_cmd: torch.tensor,
        acc_cmd: torch.tensor,
        acc_load: torch.tensor,
    ) -> torch.tensor:
        """
        计算INDI控制器输出的旋翼转速指令
        
        该函数实现了完整的INDI控制算法，主要包括：
        1. 状态滤波处理
        2. 角加速度估计
        3. 力矩计算和分配
        4. 电机动力学补偿
        5. 推力到转速的转换
        
        Args:
            state: 无人机当前状态字典，包含姿态、角速度、角加速度等信息
            actions: 当前旋翼推力（来自几何控制器）
            alpha_cmd: 期望机体角加速度（来自几何控制器）
            acc_cmd: 期望加速度指令（来自几何控制器）
            acc_load: 负载加速度估计（来自几何控制器）
            
        Returns:
            rotor_speeds: 四个旋翼的目标转速[rad/s]
        """
        # 计算总推力并进行滤波
        forces = actions.sum(-1)  # 当前总推力（四个旋翼推力之和）
        filtered_forces = self.filterMot_.add(forces)  # 滤波后的总推力
        self.filterRate_.add(state["ang_vel"])  # 更新角速度滤波器
        ang_acc_filtered = self.filterRate_.derivative()  # 通过滤波器导数估计角加速度

        if self.debug:
            self.filtered_ang_acc = ang_acc_filtered
            self.unfiltered_mot = forces
            self.filtered_mot = filtered_forces

        # 坐标变换：将角速度和角加速度从世界坐标系转换到机体坐标系
        omega = quat_apply(quat_inv(state["quat"]), state["ang_vel"])      # 机体角速度（通常来自IMU）
        omega_dot = quat_apply(
            quat_inv(state["quat"]), state["ang_acc"]
        )  # 机体角加速度（通常通过对滤波后的机体角速度求导获得）
        
        # 计算当前实际产生的力矩
        tau = torch.matmul(self.G_1, forces.transpose(0, 1)).transpose(0, 1)[:, 1:]  # 实际力矩命令[roll, pitch, yaw]

        # 构造控制输入向量mu：[总推力, 滚转力矩, 俯仰力矩, 偏航力矩]
        mu = torch.zeros((self.num_envs, 4), device=self.device)
        collective_thrust_des_magntiude = torch.norm(acc_cmd, dim=1) * self.falcon_mass  # 期望总推力大小
        mu[:, 0] = torch.clamp(collective_thrust_des_magntiude, self.thrust_min_collective, self.thrust_max_collective)

        # INDI核心控制算法
        mu_ndi = mu  # INDI控制输入

        # 计算期望力矩（考虑刚体动力学）
        moments = self.inertia_mat.matmul(alpha_cmd.transpose(0, 1)).transpose(0, 1) + torch.linalg.cross(
            omega, self.inertia_mat.matmul(omega.transpose(0, 1)).transpose(0, 1)
        )  # 刚体动力学项：I*α + ω×(I*ω)
        # 注释掉的负载力矩项：
        # - torch.linalg.cross(
        # self.p_offset, quat_apply(quat_inv(state["quat"]), acc_load * self.falcon_mass)) # 负载产生的力矩（机体坐标系）

        # 设置期望力矩
        mu_ndi[:, 1:] = moments

        # INDI控制律：考虑当前实际力矩和期望力矩的差异
        mu[:, 1:] = tau + self.inertia_mat.matmul((alpha_cmd - omega_dot).transpose(0, 1)).transpose(0, 1)
        # 控制律解释：
        # mu[:, 1:] = 当前实际力矩 + 惯性矩阵 * (期望角加速度 - 实际角加速度)
        # 这体现了INDI的增量式控制思想

        # 不进行偏航控制（保持偏航力矩为NDI计算值）
        mu[:, 3] = mu_ndi[:, 3]

        # 通过推力分配矩阵的逆矩阵计算各旋翼推力
        thrusts = self.G_1_inv.matmul(mu.transpose(0, 1))
        thrusts = torch.clamp(thrusts, self.thrust_min, self.thrust_max)  # 限制推力在合理范围内

        # 将推力转换为旋翼转速（使用推力-转速映射关系）
        rotor_speeds = torch.sqrt(thrusts / self.thrust_map[0]).transpose(0, 1)
        # 转速限制在电机允许范围内
        rotor_speeds = torch.clamp(rotor_speeds, self.motor_omega_min, self.motor_omega_max)

        return rotor_speeds

    # def getCommand(
    #     self,
    #     state: dict,
    #     actions: torch.tensor,
    #     setpoint: dict,
    # ) -> torch.tensor:
    #     forces = actions.sum(-1)
    #     # filtered_forces = self.filterMot_.add(forces)
    #     # self.filterRate_.add(state["ang_vel"])
    #     # ang_acc_filtered = self.filterRate_.derivative()

    #     # if self.debug:
    #     #     self.filtered_ang_acc = ang_acc_filtered
    #     #     self.unfiltered_mot = forces
    #     #     self.filtered_mot = filtered_forces

    #     omega = quat_apply(quat_inv(state["quat"]), state["ang_vel"])  # body rates # normally from IMU
    #     alpha_cmd = self.kp * (setpoint["body_rates"] - omega)
    #     acc_cmd = setpoint["cthrust"]
    #     omega_dot = quat_apply(
    #         quat_inv(state["quat"]), state["ang_acc"]
    #     )  # body accelerations # normally from derivative filtered body rate
    #     tau = torch.matmul(self.G_1, forces.transpose(0, 1)).transpose(0, 1)[:, 1:]  # torque commands
    #     mu = torch.zeros((self.num_envs, 4), device=self.device)
    #     # collective_thrust_des_magntiude = torch.norm(acc_cmd, dim=1) * self.falcon_mass
    #     collective_thrust_des_magntiude = acc_cmd * self.falcon_mass
    #     mu[:, 0] = torch.clamp(collective_thrust_des_magntiude, self.thrust_min_collective, self.thrust_max_collective)
    #     mu_ndi = mu

    #     moments = self.inertia_mat.matmul(alpha_cmd.transpose(0, 1)).transpose(0, 1) + torch.linalg.cross(
    #         omega, self.inertia_mat.matmul(omega.transpose(0, 1)).transpose(0, 1)
    #     )  # - torch.linalg.cross(
    #     # self.p_offset, quat_apply(quat_inv(state["quat"]), acc_load * self.falcon_mass)) # M_load in body frame

    #     mu_ndi[:, 1:] = moments
    #     mu[:, 1:] = tau + self.inertia_mat.matmul((alpha_cmd - omega_dot).transpose(0, 1)).transpose(0, 1)

    #     # without heading control
    #     mu[:, 3] = mu_ndi[:, 3]
    #     thrusts = self.G_1_inv.matmul(mu.transpose(0, 1))
    #     thrusts = torch.clamp(thrusts, self.thrust_min, self.thrust_max)

    #     rotor_speeds = torch.sqrt(thrusts / self.thrust_map[0]).transpose(0, 1)

    #     return rotor_speeds

    def reset(self, env_ids):
        """
        重置指定环境的滤波器状态
        
        Args:
            env_ids: 需要重置的环境索引
        """
        self.filterMot_.reset(env_ids)   # 重置推力滤波器
        self.filterRate_.reset(env_ids)  # 重置角速度滤波器
        # pass
