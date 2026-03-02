import torch


class RotorMotor:
    """
    旋翼电机模型类
    
    该类实现了四旋翼无人机电机的动态模型，包括：
    1. 电机转速动力学（一阶惯性环节）
    2. 推力计算（转速平方关系）
    3. 反扭矩计算（电机反作用力矩）
    4. 转速限制和饱和处理
    
    物理模型假设：
    - 推力与转速平方成正比：T = k_τ * ω²
    - 反扭矩与转速平方成正比：Q = k_Q * ω²
    - 电机动力学为一阶惯性环节：τ * dω/dt + ω = ω_cmd
    
    旋翼布局（从上往下看）：
        旋翼1(CW)    旋翼4(CCW)
              \        /
               \      /
                \    /
                 \  /
                  \/
                 /  \
                /    \
               /      \
              /        \
        旋翼2(CCW)   旋翼3(CW)
    CW: 顺时针旋转  CCW: 逆时针旋转
    """

    def __init__(self, num_envs: int, init_omega: torch.Tensor):
        """
        初始化电机模型
        
        Args:
            num_envs: 并行环境数量
            init_omega: 初始转速 [num_envs x 4] (每个环境4个旋翼)
        """
        # 电机动力学参数
        self.init_omega = init_omega           # 初始转速
        self.num_envs = num_envs               # 环境数量
        self.motor_omega_min = 150.0          # 电机最小转速 [rad/s]
        self.motor_omega_max = 2800.0         # 电机最大转速 [rad/s]
        self.tau_up = 0.033                   # 上升时间常数 [s]
        self.tau_down = 0.033                 # 下降时间常数 [s]
        self.motor_inertia = 9.3575e-6        # 电机转子转动惯量 [kg·m²]

        # 旋翼推力和扭矩映射参数
        self.thrust_map = torch.tensor([1.562522e-06, 0.0, 0.0])    # 推力系数 k_τ
        self.torque_map = torch.tensor([3.4375484e-08, 0.0, 0.0])   # 扭矩系数 k_Q

        # 旋翼旋转方向（从上往下看）
        # [1.0, -1.0, 1.0, -1.0] 对应 [CW, CCW, CW, CCW]
        self.direction = torch.tensor([1.0, -1.0, 1.0, -1.0], device="cuda")

        # 当前转速状态
        self.current_omega = init_omega       # 当前转速 [num_envs x 4]

    def get_motor_thrusts_moments(self, target_rates: torch.Tensor, sampling_time: float):
        """
        计算电机的推力和力矩输出
        
        该函数实现了完整的电机动力学模型：
        1. 一阶惯性环节模拟电机响应
        2. 推力计算（转速平方关系）
        3. 反扭矩计算（考虑旋转方向）
        
        Args:
            target_rates: 控制器给出的目标转速 [num_envs x 4]
            sampling_time: 采样时间间隔 [s]
            
        Returns:
            tuple: (thrusts, moments)
                - thrusts: 四个旋翼的推力 [num_envs x 4]
                - moments: 四个旋翼的反扭矩 [num_envs x 4]
        """
        # 判断转速变化方向以选择相应的时间常数
        # 如果目标转速大于当前转速，使用上升时间常数；否则使用下降时间常数
        tau = torch.where(target_rates > self.current_omega, self.tau_up, self.tau_down)
        alpha = torch.exp(-sampling_time / tau)  # 一阶惯性环节的衰减因子

        # 更新当前转速（一阶惯性环节差分方程）
        # ω[k+1] = α * ω[k] + (1-α) * ω_cmd[k]
        self.current_omega = alpha * self.current_omega + (1 - alpha) * target_rates

        # 计算四个旋翼的推力（推力与转速平方成正比）
        # T = k_τ * ω²
        thrusts = self.thrust_map[0] * self.current_omega**2

        # 计算四个旋翼的反扭矩（考虑旋转方向）
        # Q = k_Q * ω² * direction
        moments = self.torque_map[0] * self.current_omega**2 * self.direction

        # 转速限制检查（确保在电机允许范围内）
        # self.current_omega = torch.clamp(
        #     self.current_omega, 
        #     self.motor_omega_min, 
        #     self.motor_omega_max
        # )

        return thrusts, moments

    def reset(self, env_ids):
        """
        重置指定环境的电机状态
        
        Args:
            env_ids: 需要重置的环境索引
        """
        self.current_omega[env_ids] = self.init_omega[env_ids]  # 将指定环境的转速重置为初始值