# Reward 设计变更记录

本文档记录 fly-follow 多无人机强化学习任务中 **reward 设计的演进过程**。

当前 reward 共经历三个版本：

| 版本 | 目标                            |
| ---- | ------------------------------- |
| v1   | 原始 reward 设计                |
| v2   | 提升训练稳定性                  |
| v3   | 更专业的 fly-follow 任务 reward |

---

# v1 原始 Reward 设计

原始 reward 采用简单的加权和形式：

r = (distance_reward  
+ tracking_reward  
+ action_smoothness  
+ height_reward  
- body_rate_penalty  
- velocity_penalty  
- force_penalty  
- safety_penalty) * step_dt

各项含义如下：

| reward项          | 含义                   |
| ----------------- | ---------------------- |
| distance_reward   | 鼓励无人机靠近目标     |
| tracking_reward   | 进入跟踪半径时奖励     |
| action_smoothness | 控制动作平滑           |
| height_reward     | 保持目标高度           |
| body_rate_penalty | 惩罚角速度             |
| velocity_penalty  | 惩罚水平速度           |
| force_penalty     | 惩罚推力               |
| safety_penalty    | 碰撞 / 越界 / 过低惩罚 |

---

# v1 存在的问题

实验中发现以下问题：

## 1 tracking reward 过于稀疏

tracking reward 只有在：

distance < track_distance

时才触发。

问题：

- 大多数时间 reward = 0
- 学习信号很弱
- 收敛慢

---

## 2 distance reward 衰减过快

原始设计：

exp(-distance)

距离稍大时 reward 几乎为 0。

例如：

| distance | reward |
| -------- | ------ |
| 1        | 0.37   |
| 3        | 0.05   |
| 5        | 0.006  |

这会导致探索阶段几乎没有奖励信号。

---

## 3 height reward 过于敏感

原公式：

exp(-abs(height_error))

当高度误差稍大时 reward 几乎消失。

导致：

- 无法从高度偏差中恢复
- 学习不稳定

---

## 4 force penalty 不稳定

原始设计：

max(rotor_force)

问题：

- 只考虑单个旋翼最大推力
- 不反映整体控制 effort
- 梯度噪声较大

---

## 5 safety penalty 为离散惩罚

原始设计：

if violation:
    penalty = constant

问题：

- 没有提前预警
- 只有发生事故才惩罚

---

# v2 稳定性优化 Reward

目标：

提升训练稳定性和 reward 连续性。

---

## 1 tracking reward 连续化

原设计：

进入 tracking 半径才奖励

新设计：

tracking_reward = exp(-(distance/sigma)^2)

优势：

- reward 连续
- 梯度稳定
- 学习更快

---

## 2 distance reward 改为 Gaussian

原设计：

exp(-distance)

新设计：

exp(-(distance/sigma)^2)

优势：

- 衰减更慢
- 探索阶段信号更强

---

## 3 height reward 改为 Gaussian

原设计：

exp(-|height_error|)

新设计：

exp(-(height_error/sigma)^2)

优势：

- 更稳定
- 容错性更高

---

## 4 action smoothness 优化

原设计：

|a_t - a_{t-1}|

新设计：

(a_t - a_{t-1})^2

优势：

- 梯度更平滑
- 控制更稳定

---

## 5 force penalty 改为 action energy

原设计：

max rotor thrust

新设计：

mean(action^2)

优势：

- 控制 effort 更合理
- 更符合物理意义

---

## 6 safety penalty 改为 soft penalty

新增：

- 碰撞软边界
- 边界软约束
- 高度软约束

优势：

- 提前规避风险
- 学习更加稳定

---

# v2 训练结果

实验表明：

| 指标       | 结果        |
| ---------- | ----------- |
| 训练稳定性 | 显著提升    |
| 终止率     | 低          |
| 高度控制   | 稳定        |
| 动作平滑度 | 高          |
| 收敛速度   | ~100k steps |

---

# v3 专业 Fly-Follow Reward

v3 在 v2 基础上进一步优化。

目标：

实现 **真正的动态目标跟随任务**。

---

# 核心改进

## 1 速度跟踪 reward

新增：

velocity tracking reward

公式：

r_vel = exp(-||v_drone - v_target||^2)

意义：

鼓励无人机：

- 与目标保持相同运动方向
- 实现持续跟随

---

## 2 目标分配机制

原设计：

每个无人机追最近目标

问题：

- 多机可能追同一目标
- 有目标无人机未覆盖

新设计：

加入 assignment 机制：

- greedy assignment
- Hungarian matching

---

## 3 覆盖奖励

新增：

coverage reward

鼓励：

- 每个目标至少被一台无人机跟踪

---

## 4 速度惩罚优化

原设计：

penalty = velocity

新设计：

penalty = max(velocity - safe_speed, 0)

优势：

- 允许合理运动
- 只惩罚过快速度

---

## 5 安全约束优化

新增：

- collision penalty
- boundary penalty
- altitude penalty

全部采用 soft margin。

---

# Reward 设计演进总结

| 版本 | 特点                   |
| ---- | ---------------------- |
| v1   | 基础 reward            |
| v2   | 稳定性优化             |
| v3   | 专业 fly-follow reward |

---

# 推荐训练流程

建议使用 curriculum learning：

1. 静态目标 hover
2. 慢速移动目标
3. 多目标跟踪
4. 高速目标

---

# End