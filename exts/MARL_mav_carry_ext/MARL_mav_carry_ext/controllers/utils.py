import math
import torch


class LowPassFilter:
    """
    二阶巴特沃斯低通滤波器实现
    
    该滤波器用于对传感器数据（如加速度、角速度、推力等）进行滤波处理，
    以减少高频噪声的影响，提高控制系统的稳定性。
    
    数学原理：
    采用双线性变换法设计的二阶巴特沃斯低通滤波器，具有以下特点：
    1. 平坦的通带响应
    2. 较陡的阻带衰减
    3. 良好的相位特性
    4. 数字实现稳定性好
    
    差分方程形式：
    y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]
    
    其中系数通过双线性变换从模拟原型滤波器获得。
    """

    def __init__(self, fc, fs, initial_value):
        """
        初始化低通滤波器
        
        Args:
            fc: 截止频率 (cutoff frequency) - envs x 1
            fs: 采样频率 (sampling frequency) - envs x 1  
            initial_value: 初始值 - envs x dim
        """
        self.sampling_freq = fs  # 采样频率 [envs x 1]
        self.num = self.init_num(fc, fs)  # 分子系数 [envs x 1 x 2]
        self.dem = self.init_dem(fc, fs)  # 分母系数 [envs x 1 x 2]
        self.initial_value = initial_value
        
        # 初始化输入和输出缓冲区 [envs x dim x 2]
        # 第二维度表示当前时刻(n)和前一时刻(n-1)
        self.input = initial_value.unsqueeze(2).repeat(1, 1, 2)   # 输入缓冲区
        self.output = initial_value.unsqueeze(2).repeat(1, 1, 2)  # 输出缓冲区

    def init_dem(self, fc, fs):
        """
        初始化滤波器分母系数（反馈系数）
        
        使用双线性变换法从模拟巴特沃斯滤波器转换而来。
        
        Args:
            fc: 截止频率
            fs: 采样频率
            
        Returns:
            dem: 分母系数矩阵 [envs x 1 x 2]
        """
        # 双线性变换预畸变
        K = torch.tan(math.pi * fc / fs)
        # 归一化多项式
        poly = K * K + math.sqrt(2.0) * K + 1.0
        
        # 计算分母系数 a1, a2
        dem = torch.zeros_like(fc).repeat(1, 2)
        dem[:, 0] = (2.0 * (K * K - 1.0) / poly).squeeze(1)      # a1系数
        dem[:, 1] = ((K * K - math.sqrt(2.0) * K + 1.0) / poly).squeeze(1)  # a2系数

        return dem.unsqueeze(1)

    def init_num(self, fc, fs):
        """
        初始化滤波器分子系数（前馈系数）
        
        Args:
            fc: 截止频率
            fs: 采样频率
            
        Returns:
            num: 分子系数矩阵 [envs x 1 x 2]
        """
        # 双线性变换预畸变
        K = torch.tan(math.pi * fc / fs)
        # 归一化多项式
        poly = K * K + math.sqrt(2.0) * K + 1.0
        
        # 计算分子系数 b0, b1, b2
        num = torch.zeros_like(fc).repeat(1, 2)
        num[:, 0] = (K * K / poly).squeeze(1)     # b0系数
        num[:, 1] = 2.0 * num[:, 0]               # b1系数 (b2 = b0 due to symmetry)

        return num.unsqueeze(1)

    def derivative(self):
        """
        计算滤波器输出的一阶导数（数值微分）
        
        通过一阶后向差分近似计算导数：
        dy/dt ≈ (y[n] - y[n-1]) / T
        
        Returns:
            derivative: 输出信号的导数 [envs x dim]
        """
        return self.sampling_freq * (self.output[:, :, 0] - self.output[:, :, 1])

    def add(self, sample):
        """
        向滤波器添加新样本并计算滤波输出
        
        实现二阶IIR滤波器的递推计算：
        y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]
        
        Args:
            sample: 新输入样本 [envs x dim]
            
        Returns:
            out: 当前滤波输出 [envs x dim]
        """
        # 更新输入缓冲区
        x2 = self.input[:, :, 1]              # x[n-2]
        self.input[:, :, 1] = self.input[:, :, 0]  # x[n-1] <- x[n]
        self.input[:, :, 0] = sample          # x[n] <- new_sample

        # 计算滤波输出（IIR滤波器差分方程）
        # out = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]
        out = self.num[:, :, 0] * x2 + (self.num * self.input - self.dem * self.output).sum(dim=2)
        
        # 更新输出缓冲区
        self.output[:, :, 1] = self.output[:, :, 0]  # y[n-1] <- y[n]
        self.output[:, :, 0] = out                   # y[n] <- new_output

        return out

    def valid(self):
        """
        检查滤波器参数和状态的有效性
        
        Returns:
            bool: 所有参数是否均为有限数值
        """
        return (
            torch.isfinite(self.sampling_freq).all()
            and torch.isfinite(self.dem).all()
            and torch.isfinite(self.num).all()
            and torch.isfinite(self.input).all()
            and torch.isfinite(self.output).all()
        )

    def reset(self, env_ids):
        """
        重置指定环境的滤波器状态
        
        Args:
            env_ids: 需要重置的环境索引
        """
        self.input[env_ids] = self.initial_value.unsqueeze(2).repeat(1, 1, 2)[env_ids]   # 重置输入缓冲区
        self.output[env_ids] = self.initial_value.unsqueeze(2).repeat(1, 1, 2)[env_ids]  # 重置输出缓冲区

    def __call__(self):
        """
        获取当前滤波输出
        
        Returns:
            output: 当前时刻的滤波输出 [envs x dim]
        """
        return self.output[:, :, 0]