## [2026-03-13] 重构 tracking_reward 为加法结构，强化稳定保持

**修改文件：**
- `exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/flyfollow/marl_flyfollow_env.py`
- `exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/flyfollow/marl_flyfollow_env_cfg.py`
- `exts/MARL_mav_carry_ext/MARL_mav_carry_ext/assets/data/AMR/flyfollow/city.usd`

**修改原因：**
原三因子相乘结构（dist_factor × vel_match_factor × persistence_bonus）导致任何单项偏低都会让整体奖励跌近零，表现为偶发 spike 而非可持续的区内奖励抬升。重构为加法结构，使无人机进入 bonus 区后每步赚到明确保底收益，速度质量和持续时间作为叠加项而非乘法门控。

**主要变更：**
- `tracking_reward` 公式从 `w×v×dist_factor×vel_factor×persistence` 重构为 `w×v×in_zone_factor×(1 + vel_bonus + persistence_gain)`，消除乘法清零效应
- 新增 `in_zone_factor = sigmoid((bonus_dist - dist) × sharpness)`：sigmoid 软边界替代高斯 dist_factor，区内深处稳定趋近 1，每步保底基础收益清晰可见
- 速度质量和持续时间从乘法约束改为叠加项：即使速度不完全匹配，仅损失额外加成而不影响基础保底（最多各 +50%）
- 计时器激活条件简化为 `dist < bonus_dist`（不再叠加速度门槛），速度激励完全交给 vel_quality 叠加项独立处理
- cfg 新增 `tracking_zone_sharpness=4.0`（软边界陡度，约 0.25m 过渡区间）和 `tracking_vel_quality_alpha=0.5`（速度叠加项权重）

---

## [2026-03-13] 升级 tracking_reward 为持续稳定跟随奖励

**修改文件：**
- `exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/flyfollow/marl_flyfollow_env.py`
- `exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/flyfollow/marl_flyfollow_env_cfg.py`

**修改原因：**
原 tracking_reward 采用纯 binary mask（距离 < 1.5m 即给固定奖励），梯度不连续且无法区分"抖动进出"与"稳定贴近"两种截然不同的行为质量。升级为三因子相乘的持续奖励机制，使策略可感知"更近、更稳、持续更久"的方向梯度。

**主要变更：**
- 新增 `_tracking_stable_timer` 缓冲区（shape: `E×D`），记录各无人机连续满足近距+速度匹配条件的持续时间，并在 `_reset_idx` 中正确重置
- `tracking_reward` 从 `binary_mask × target_value` 升级为 `target_value × dist_factor × vel_match_factor × persistence_bonus` 三因子乘积
- `dist_factor = exp(-(dist/bonus_dist)²)`：在 bonus_distance_xy=1.5m 内有连续梯度，消除 1.5m 处断崖
- `vel_match_factor = exp(-(vel_err/sigma)²)`（sigma=1.5m/s）：速度不匹配时即使距离近也无法获得高奖励，抑制抖动进出行为
- `persistence_bonus = 1 + alpha × clamp(timer/T_ref, 0, 1)`（alpha=0.5, T_ref=2.0s）：稳定跟随 2 秒可获最多 +50% 额外加成，激励持续维持跟随状态
- 新增 cfg 参数：`tracking_vel_match_sigma=1.5`、`tracking_persistence_alpha=0.5`、`tracking_persistence_time=2.0`
- 将 `vel_err_xy/vel_err_norm` 的计算提前至 tracking_reward 之前，供 tracking 和 velocity_follow 两部分共用，消除重复计算

---

## [2026-03-12] 弱化速度惩罚为防失控约束

**修改文件：**
- `exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/flyfollow/marl_flyfollow_env_cfg.py`

**修改原因：**
速度惩罚过激导致贴近保持不稳定——惩罚权重过高且阈值过低，干扰了正常追踪机动。将 velocity_penalty 定位为"防失控的最后一道约束"而非行为塑形工具，行为塑形职责完全交给 velocity_follow_reward 和 dist_progress_reward。

**主要变更：**
- `velocity_penalty_weight` 从 0.2 降至 0.05，惩罚力度降低 4 倍
- `velocity_penalty_xy_safe` 从 2.0 升至 4.0 m/s，正常追踪机动（≤2.3 m/s）完全不触发
- `velocity_penalty_z_safe` 从 0.5 升至 2.0 m/s，允许高度调整时的正常爬升/下降

---

## [2026-03-12] 修复高度观测缺失 + 去除定高约束

**修改文件：**
- `exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/flyfollow/marl_flyfollow_env.py`
- `exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/flyfollow/marl_flyfollow_env_cfg.py`

**修改原因：**
训练分析发现无人机长期悬停在 3.7~4.7m（目标 2.0m），根因是观测向量缺少 z 位置和 vz 速度，导致高度控制无从学习。同时明确任务不要求定高，只需保持在安全高度带内，因此移除所有与"悬停在 2m"相关的奖励惩罚项，并将 success 判定中的高度精度条件改为高度带检查。

**主要变更：**
- 观测向量新增本机高度 z（归一化 /5.0）和垂直速度 vz，`obs_dim_per_step` +2
- 关闭所有定高相关奖励/惩罚：`height_reward_weight=0`、`height_error_penalty_weight=0`、`vertical_direction_penalty_weight=0`
- success 判定移除 `|z - desired_height| ≤ tolerance` 条件，改为 `min_altitude ≤ z ≤ max_altitude` 高度带检查
- 删除已失效的 `success_height_tolerance` 参数
- 高度软边界重新对齐硬终止边界：低惩罚区 z < 2.0m，高惩罚区 z > 5.5m（硬终止 7.0m），中间段完全自由

---

## [2026-03-12] 修复训练局部最优

**修改文件：**
- `exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/flyfollow/marl_flyfollow_env.py`
- `exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/flyfollow/marl_flyfollow_env_cfg.py`
- `.gitignore`（修正 `.claude` 路径）
- 删除废弃文件 `marl_move_env-zy.py`

**修改原因：**
训练稳定卡在"接近目标但进不了成功区、被高度/速度惩罚长期压制"的局部最优。根因是三个约束共同封死了追近行为：distance_reward_sigma=28 导致近程梯度几乎为零；overspeed_margin=0.3m/s 限制无人机只能以 0.3m/s 的速度优势追近；velocity_penalty_xy_safe=1.2m/s 与前者叠加后实际速度上限约 1.1m/s，严重不足。

**主要变更：**
- `distance_reward_sigma` 从 28 降至 10，d=5m→3m 的梯度增强约 10 倍，策略可感知靠近成功区的价值
- `tracking_reward_weight` 从 1.0 增至 2.5，增大进入 1.5m 成功圈的激励
- `velocity_follow_overspeed_margin` 从 0.3 增至 2.0 m/s，允许无人机以更高速冲入成功区
- `velocity_follow_overspeed_weight` 从 0.2 降至 0.05，大幅削减超速惩罚
- `velocity_penalty_xy_safe` 从 1.2 增至 2.5 m/s，彻底打开追近时的速度上限
- 新增 `dist_progress_reward`（势函数距离进度奖励）：每步奖励 `w × (prev_dist - curr_dist)`，提供直接的时序梯度信号驱动无人机主动缩短与目标的距离

---

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