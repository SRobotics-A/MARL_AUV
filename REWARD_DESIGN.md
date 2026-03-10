# Fly-Follow Reward 设计说明

本文档详细说明 fly-follow 多无人机强化学习任务的 reward 设计原理。

设计目标：

实现稳定、可扩展的 **多无人机目标跟随策略学习**。

---

# 1 任务目标

fly-follow 任务要求无人机：

1 保持稳定飞行  
2 跟随目标移动  
3 保持合理高度  
4 避免碰撞  
5 控制动作平滑  

---

# 2 Reward 总体结构

reward 采用如下结构：

r = r_task + r_stability - p_safety

展开为：

r =
r_pos
+ r_vel
+ r_height
+ r_smooth
+ r_cover
- p_rate
- p_vel
- p_force
- p_collision
- p_boundary
- p_low

最后乘以：

step_dt

以保证 reward 与仿真时间尺度一致。

---

# 3 任务奖励

## 3.1 位置跟踪奖励

目标：

鼓励无人机接近目标。

公式：

r_pos = exp(-(distance / σ_pos)^2)

其中：

distance = ||p_drone - p_target||

---

## 3.2 速度跟踪奖励

目标：

鼓励无人机与目标速度一致。

公式：

r_vel = exp(-(velocity_error / σ_vel)^2)

velocity_error = ||v_drone - v_target||

该项对于动态目标跟随任务非常关键。

---

## 3.3 高度奖励

目标：

保持无人机在指定高度附近。

公式：

r_height = exp(-(height_error / σ_h)^2)

height_error = |z - z_target|

---

## 3.4 覆盖奖励

用于多无人机多目标场景。

目标：

保证每个目标至少被一台无人机跟踪。

简单形式：

r_cover = coverage_weight

当某无人机为该目标最近无人机时给予奖励。

---

# 4 控制稳定性奖励

## 4.1 动作平滑奖励

目标：

减少控制抖动。

公式：

r_smooth = exp(-(Δaction^2) / σ_u^2)

其中：

Δaction = action_t - action_{t-1}

---

# 5 控制惩罚

## 5.1 角速度惩罚

公式：

p_rate = w_rate * ||ω||

目的：

避免剧烈姿态变化。

---

## 5.2 速度惩罚

公式：

p_vel = w_vel * max(||v_xy|| - v_safe, 0)

只有当速度超过安全阈值时才惩罚。

---

## 5.3 控制能量惩罚

公式：

p_force = w_force * ||action||^2

减少过度推力。

---

# 6 安全约束

## 6.1 碰撞惩罚

公式：

p_collision = max(d_safe - d_min, 0)

其中：

d_min 为无人机最小间距。

---

## 6.2 边界惩罚

公式：

p_boundary = max(boundary_margin - distance_to_boundary, 0)

避免无人机接近环境边界。

---

## 6.3 低高度惩罚

公式：

p_low = max(z_min + margin - z, 0)

防止无人机高度过低。

---

# 7 Reward Scaling

最终 reward 为：

r_total = r * step_dt

原因：

保持 reward 与时间尺度一致。

---

# 8 Reward 设计原则

本 reward 设计遵循以下原则：

1 连续 reward 信号  
2 任务目标优先  
3 控制稳定性约束  
4 安全约束提前介入  
5 多无人机协同  

---

# 9 推荐参数

推荐初始参数：

| 参数                     | 推荐值 |
| ------------------------ | ------ |
| pos_reward_weight        | 2.0    |
| vel_track_weight         | 1.5    |
| height_reward_weight     | 1.0    |
| action_smoothness_weight | 0.3    |
| coverage_reward_weight   | 0.5    |

惩罚：

| 参数                     | 推荐值 |
| ------------------------ | ------ |
| body_rate_penalty_weight | 0.03   |
| velocity_penalty_weight  | 0.05   |
| force_penalty_weight     | 0.02   |
| collision_penalty_weight | 0.8    |

---

# 10 Curriculum Training

建议采用分阶段训练：

阶段1  
静态目标 hover

阶段2  
慢速移动目标

阶段3  
多目标跟随

阶段4  
复杂轨迹

---

# 11 总结

该 reward 设计实现了：

- 连续 reward shaping
- 多无人机协同
- 稳定控制策略
- 动态目标跟随

适用于：

- 多无人机跟随任务
- swarm coordination
- reinforcement learning control

---

# End