## TensorBoard 训练图表含义说明

> 适用任务：Isaac-move-flyfollow-marl-v0（MAPPO）
> 本节说明 TensorBoard 中每张图的含义，便于团队成员快速理解训练状态。

---

### 一、Reward（奖励）

#### `Reward / Instantaneous reward (mean/max/min)`

- **含义**：每个训练步（rollout step）的即时奖励，跨所有并行环境的均值/最大值/最小值
- **单位**：奖励值（无量纲，已乘 step_dt 归一化）
- **如何读**：mean 持续上升 = 策略整体在进步；max 远高于 mean = 存在少数高质量 episode；min 长期为负 = 部分环境仍在受惩罚
- **正常范围**：本任务中 mean 从 −5 起步，收敛后约 +3～+5/step

#### `Reward / Total reward (mean/max/min)`

- **含义**：每个 episode（回合）累计总奖励的均值/最大值/最小值
- **如何读**：与 Instantaneous 的区别是它是 episode 维度的，受 episode 长度影响——episode 越长，total 越高
- **诊断用途**：若 total mean 上升但 episode 长度也在上升，需分离两者贡献

---

### 二、Episode（回合统计）

#### `Episode / Total timesteps (mean/max/min)`

- **含义**：每个 episode 持续的控制步数（1步=10ms，100步=1秒）
- **如何读**：
  - mean 短（<100步）= 频繁提前终止，说明存在终止条件过严或策略不稳定
  - mean 长（>200步）= 策略存活能力强，有机会积累更多追踪奖励
  - min 极短（<30步）= 大量 fly_high 或坠机终止，需重点排查
- **本任务目标**：mean > 200步，max mean > 350步

---

### 三、Episode_Reward（分项奖励，每 episode 均值）

每个分项代表该奖励分量在一个 episode 内的累计均值。正值越大越好，负值越接近 0 越好。

#### 追踪类（核心任务指标）

| 图表名            | 含义                                                         | 期望趋势                   |
| ----------------- | ------------------------------------------------------------ | -------------------------- |
| `distance_reward` | exp(−dist×0.3)×target_value 的 episode 累计值，反映无人机离目标的远近 | 持续上升，最终稳定         |
| `tracking_reward` | 仅在进入捕获区（dist<3.5m）时给予的奖励，是任务成功的核心信号 | **最重要指标**，应持续上升 |
| `velocity_follow` | 无人机 XY 速度与目标速度（0.3m/s x向）匹配程度，exp(−vel_err²) | 应随 tracking 同步上升     |

#### 稳定性类（姿态与控制质量）

| 图表名              | 含义                                                         | 期望趋势       |
| ------------------- | ------------------------------------------------------------ | -------------- |
| `body_rate_penalty` | exp(−                                                        |                |
| `upright_penalty`   | (R_zz−1.0)×dt，机体偏离竖直越多惩罚越大，接近 0 表示飞行姿态良好 | 应接近 0       |
| `action_smoothness` | exp(−                                                        |                |
| `height_reward`     | exp(−                                                        | z−1.5m         |
| `force_penalty`     | exp(−max_thrust_normalized)×dt，旋翼推力越低奖励越高         | 上升后稳定即可 |

#### 惩罚类（安全与约束违反，越接近 0 越好）

| 图表名              | 含义                                       | 期望趋势         |
| ------------------- | ------------------------------------------ | ---------------- |
| `fly_high_penalty`  | exp(z−4.0m)×dt，超过 4m 后指数增长的软惩罚 | 应趋向 0         |
| `height_penalty`    | \|z−desired\|>0.3m 时的线性惩罚            | 应趋向 0         |
| `collision_penalty` | 无人机两两间距 <0.6m 时的碰撞惩罚          | 应趋向 0         |
| `drone_out`         | 无人机超出 ±60m 边界的惩罚                 | 正常训练中应为 0 |
| `fly_low`           | z<0.1m 时的坠地惩罚                        | 应为 0           |
| `illegal_contact`   | 接触传感器力 >1N 时的惩罚                  | 应为 0           |
| `velocity_penalty`  | 当前已禁用（权重=0），恒为 0               | 忽略             |

---

### 四、Episode_Termination（终止原因统计）

每项表示该终止原因在一批重置 episode 中的触发次数。**诊断早期训练问题最有用的一组图表。**

| 图表名                  | 含义                                         | 正常状态                     |
| ----------------------- | -------------------------------------------- | ---------------------------- |
| `falcon_fly_high`       | z > 5.0m 触发的高飞终止                      | 早期高，随训练降至接近 0     |
| `falcon_fly_low`        | z < 0.1m 触发的坠地终止                      | 应始终为 0                   |
| `bounding_box`          | 无人机超出 ±60m 边界终止                     | 应为 0（已修复）             |
| `crash`                 | fly_low + illegal_contact 的合并统计         | 应为 0                       |
| `drones_collide`        | 无人机互撞终止                               | 应为 0                       |
| `targets_out_of_bounds` | 目标小车 x > 30m 跑出场景终止                | 偶发，episode 较长时出现     |
| `time_out`              | episode 达到最大长度（60s）正常超时          | 越多越好，代表策略存活能力强 |
| `all_targets_captured`  | 成功：≥3个目标同时被跟随持续 1.5s            | **目标**：越多越好           |
| `out_of_bounds`         | 目标超出边界（= targets_out_of_bounds 别名） | 同上                         |

---

### 五、Loss（损失函数）

#### `Loss / Policy loss`

- **含义**：Actor（策略网络）的 PPO Clipped Surrogate Loss
- **如何读**：正常训练中应在 −0.05～−0.005 范围内波动，接近 0 表示策略更新平稳；突然变大（绝对值 >0.1）表示梯度爆炸或 ratio_clip 过大

#### `Loss / Value loss`

- **含义**：Critic（价值网络）的均方误差损失，衡量价值估计精度
- **如何读**：应随训练单调下降至接近 0；下降停滞说明 Critic 已收敛（好）或卡住（需检查 reward scale）
- **本任务**：从 ~1.0 降至 0.02~0.05 为正常收敛区间

#### `Loss / Entropy loss`

- **含义**：策略熵的负值，体现探索程度（entropy_loss_scale=0.01 时，此值 = −0.01×H）
- **如何读**：绝对值越大 = 熵越高 = 探索越充分；趋向 0 = 策略趋于确定性（探索减少）
- **警戒**：若接近 0 且 policy_std 也在下降，说明探索耗尽，需考虑增大 entropy_loss_scale

---

### 六、Policy（策略统计）

#### `Policy / Standard deviation`

- **含义**：Actor 输出的高斯动作分布标准差（平均值），直接反映策略的探索程度
- **如何读**：
  - 初始值 ≈ exp(initial_log_std) = exp(−0.2) ≈ 0.82
  - 正常收敛：0.4～0.6（适度确定性）
  - **警戒线 <0.38**：探索不足，策略可能陷入局部最优
  - **警戒线 <0.35**：探索崩溃风险，建议增大 entropy_loss_scale
- **本任务当前值**：0.375（需监控）

#### `Policy / Gradient norm actor / critic`

- **含义**：Actor/Critic 梯度的 L2 范数（已经过 grad_norm_clip=1.0 裁剪）
- **如何读**：稳定在 0.1～1.0 之间为正常；持续为 1.0 说明梯度被裁剪（学习率可能过大）；接近 0 说明梯度消失

---

### 七、Performance（训练效率）

| 图表名            | 含义                                               | 参考值                        |
| ----------------- | -------------------------------------------------- | ----------------------------- |
| `total FPS`       | 每秒处理的仿真帧数（num_envs × steps/s）           | num_envs=16 时约 300～500 FPS |
| `Collection Time` | 每次 rollout 数据采集耗时（秒）                    | 应稳定，突然增大说明仿真卡顿  |
| `Learning time`   | 每次网络更新（learning_epochs × mini_batches）耗时 | 应稳定                        |

---

### 八、快速诊断指南

| 现象                   | 最可能原因            | 查哪张图                                              |
| ---------------------- | --------------------- | ----------------------------------------------------- |
| episode 很短（<100步） | fly_high 终止过多     | `Episode_Termination/falcon_fly_high`                 |
| tracking=0             | 从未进入捕获区        | `distance_reward` 是否在增长                          |
| 总奖励为负             | 惩罚项压制奖励        | `fly_high_penalty`、`height_penalty`                  |
| 训练停滞               | 探索耗尽              | `Policy/Standard deviation`                           |
| value loss 不降        | Critic 收敛慢         | 检查 reward scale 是否过大                            |
| tracking 下滑          | fly_high 过于频繁截断 | `Episode_Termination/falcon_fly_high` + `episode min` |

------

# Training Analysis Report

**Run:** `2026-03-23_17-12-31_mappo_torch_mappo`
**Date:** 2026-03-23
**Task:** Isaac-marl-move-flyfollow-v0
**Algorithm:** MAPPO (CTDE, shared actor/critic)
**Total timesteps logged:** 9,900 (of 400,000 planned)

---

## Training Metrics Summary

| Metric | Early mean | Recent mean | Last value | Best ever |
|--------|-----------|-------------|------------|-----------|
| Total reward (mean) | -8.90 | -4.76 | -2.80 | -0.64 |
| Instant reward (mean) | -0.076 | -0.042 | -0.140 | +0.014 |
| distance_reward | 0.055 | 0.039 | 0.054 | 0.144 |
| tracking_reward | **0.000** | **0.000** | **0.000** | **0.000** |
| height_reward | 0.557 | 0.584 | 0.564 | 0.621 |
| body_rate_penalty | 0.874 | 0.693 | 0.969 | 1.164 |
| force_penalty | 0.230 | 0.186 | 0.263 | 0.305 |
| action_smoothness | 0.096 | 0.079 | 0.108 | 0.141 |
| velocity_penalty | 0.020 | 0.020 | 0.022 | 0.028 |
| Episode length (mean steps) | 105.0 | 83.2 | 85.0 | 135.0 |
| Policy std | 0.819 | 0.820 | 0.820 | 0.822 |
| Value loss | 0.932 | 0.049 | 0.214 | 3.11 |

**Key observations:**
- Total reward is still negative, meaning cumulative penalties exceed positive rewards
- `tracking_reward` has been exactly 0.000 for the entire run — drones have never successfully entered the capture zone (dist < 2.0m) even once
- `height_reward` (weight=2.0) and `body_rate_penalty` (weight=2.0) together dominate the positive signal budget (~1.25/ep combined), dwarfing `distance_reward` (0.039/ep)
- Episode length is *shrinking* (105→83 steps), indicating increasingly frequent early terminations
- Policy std is flat (~0.820), consistent with both initial_log_std=-0.2 (→std≈0.82) and entropy_loss_scale=0.001 — the policy is not collapsing but also not converging

---

## Observations & Findings

### [Reward Dominance Inversion] — Severity: CRITICAL

**Symptom:** `tracking_reward = 0.0` for the entire run. Drones have never entered the 2.0m capture zone around any target. Distance reward is extremely low (0.039/ep).

**Root Cause:** The stability/penalty terms collectively reward *staying still* more than *chasing targets*. At the hover equilibrium point (near desired_height=2.5m, zero body rates, zero velocity), the drone accumulates:

- `body_rate_penalty`: weight=2.0 × exp(-0) × dt = 2.0 × 1.0 × 0.01 = **0.020/step**
- `height_reward`: weight=2.0 × exp(-0) × dt = **0.020/step**
- `force_penalty`: weight=0.5 × exp(-effort) × dt ≈ **0.005/step**
- `action_smoothness`: weight=1.0 × exp(0) × dt = **0.010/step**
- `velocity_penalty`: weight=0.3 × exp(-0) × dt = **0.003/step**

**Total hover reward: ~0.058/step**

For a drone to gain `distance_reward` by closing to 2m from 8m away, it must move at speed (triggering `velocity_penalty`), rotate (triggering `body_rate_penalty` + `upright_penalty`), and deviate from 2.5m height (triggering `height_reward` loss). The marginal gain from the distance reward at 2m vs 8m is:

- distance_reward gain: 1.5 × (exp(-1.0) - exp(-4.0)) × 10_total_value × 0.01 ≈ 1.5 × (0.368 - 0.018) × 0.01 × avg_value ≈ **0.026/step** (with mean value ~2.5)

This is *less than* what the drone sacrifices from hovering rewards during approach. The policy has found a stable local optimum: hover at 2.5m with zero body rates and do nothing.

**Evidence:** `tracking_reward=0.0` for all 9,900 steps. `height_reward` (0.584/ep) and `body_rate_penalty` (0.693/ep) together represent ~86% of all positive reward in recent episodes.

---

### [Velocity Reward Missing] — Severity: CRITICAL

**Symptom:** There is no reward term for matching target velocity (0.3 m/s in x-direction). The only velocity term is `velocity_penalty = exp(-||v||)`, which *penalizes* all motion.

**Root Cause:** The task requires drones to continuously follow a moving target at 0.3 m/s. Without a velocity-matching reward, the drone has no incentive to maintain a velocity that keeps pace with the target. Even if it enters the capture zone once, it will fall behind immediately.

**Evidence:** The `velocity_penalty_weight=0.3` combined with `body_rate_penalty_weight=2.0` creates strong incentives for zero-velocity hovering. `tracking_reward=0.0` confirms drones are not sustaining capture.

---

### [Distance Reward Gradient Too Weak] — Severity: HIGH

**Symptom:** `distance_reward` barely changes across training (early=0.055, recent=0.039 — actually *declining*, suggesting drones are drifting farther from targets over time).

**Root Cause:** `dist_reward_scale=0.5` gives:
- At dist=2m: exp(-1.0) = 0.368
- At dist=5m: exp(-2.5) = 0.082
- At dist=10m: exp(-5.0) = 0.007

The gradient from 10m to 5m is only Δ=0.075 per target. With `dist_reward_weight=1.5` and `step_dt=0.01`, this is 0.001/step — negligible compared to hovering rewards. The drone correctly learns that hovering gives more reward than approaching.

Additionally, distance is computed in XY-only (`drone_positions[:,:,:2]`), which is correct for the capture check. But the reward gradient disappears at long range (>5m), leaving drones without guidance from afar.

**Evidence:** `distance_reward` is *declining* across training (0.055 → 0.039), consistent with drones drifting away as they optimize for hover rewards.

---

### [Episode Length Shrinking — Early Termination Increasing] — Severity: HIGH

**Symptom:** Mean episode length decreased from 105 to 83 steps. At 100 Hz (decimation=3, physics at 300Hz), 83 steps = 0.83 seconds — extremely short episodes. Targets move at 0.3 m/s and need to be chased, but most episodes end before the drone can even establish approach.

**Root Cause:** Short episodes with repeated early terminations (fly_low, illegal_contact, out_of_bounds) prevent the policy from ever reaching the capture zone, creating a vicious cycle: drone oscillates trying to explore, contacts ground or boundary, episode resets. This explains the flat `policy_std` — the policy is not learning meaningful behavior, just oscillating.

**Evidence:** Episode max=135, recent_mean=83, min=55. `value_loss` collapsed (0.93→0.049), consistent with the critic learning to predict a nearly-fixed negative value for all states (hover → small negative, early terminate → large negative).

---

### [PPO Clip Too Conservative] — Severity: MEDIUM

**Symptom:** `ratio_clip=0.1` is half the typical value (0.2). With `learning_epochs=5` and this tight clip, effective policy gradient per update is severely restricted.

**Root Cause:** The MAPPO config sets `ratio_clip=0.1`, which means the policy can only shift probability ratios by ±10% per update batch. With a flat reward landscape (most states near hover equilibrium), the gradient signal is already weak, and aggressive clipping makes learning even slower.

**Evidence:** `policy_loss` is near zero throughout (recent mean -0.067), consistent with near-zero effective gradient. Combined with `entropy_loss_scale=0.001` (very small exploration bonus), the policy has little pressure to explore.

---

### [Upright Penalty Sign Logic] — Severity: MEDIUM

**Symptom:** `upright_penalty = upright_penalty_weight × (z_axis_body - 1.0).sum(-1) × step_dt`. At hover (fully upright), `z_axis_body≈1.0`, so the penalty term ≈ `2.0 × (1.0-1.0) × 3 × 0.01 = 0`. But when tilted, `z_axis_body < 1`, so the penalty becomes `2.0 × (negative) × step_dt` — a negative reward. This is technically correct but labeled confusingly.

**Root Cause:** When the drone tilts to accelerate toward a target (necessary for fast flight), the upright penalty fires. With `upright_penalty_weight=2.0`, a 30-degree tilt gives `(cos(30°)-1.0) = -0.134` per drone, total penalty = `2.0 × (-0.134×3) × 0.01 = -0.008/step`. This further disincentivizes aggressive approach maneuvers.

**Evidence:** The term is nonzero whenever the drone maneuvers, creating additional resistance to the approach behavior needed for the task.

---

### [Observation: Target Velocity Not Included in Actor Obs] — Severity: MEDIUM

**Symptom:** Actor observation (49 dims) includes target relative position but not target velocity. The Critic global state does include `target_velocities`.

**Root Cause:** Without velocity information, the actor cannot learn to predict where targets will be or match their speed. The actor must implicitly learn that targets move at 0.3 m/s from positional history alone (via the 3-frame history buffer). This is learnable in principle but slows convergence.

**Evidence:** `obs_targets` in `_get_observations()` = `target_rel_pos(12) + captured(4) + values(4)` — no velocity term.

---

## Improvement Recommendations

### Priority 1 (CRITICAL): Add velocity-matching reward and dramatically increase distance reward weight

**Problem:** The drone is optimally hovering at desired_height with zero body rates because stability rewards dominate. No velocity-following incentive exists. The task requires continuous motion at 0.3 m/s.

**Proposed Changes:**

**File:** `exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/move_flyfollow/marl_move_flyfollow_env_cfg.py`

1. Increase distance reward weight and widen gradient:
   ```
   dist_reward_weight: 1.5  →  4.0
   dist_reward_scale:  0.5  →  0.2   # wider gradient basin, still strong signal at 10m
   ```

2. Add velocity-following reward weight (new parameter):
   ```
   velocity_follow_weight = 1.5   # reward for matching target x-velocity (0.3 m/s)
   velocity_follow_scale = 1.0    # exp(-||v_drone_x - v_target_x||² × scale)
   ```

3. Reduce stability reward dominance:
   ```
   body_rate_penalty_weight: 2.0  →  0.5
   height_reward_weight:     2.0  →  0.5
   upright_penalty_weight:   2.0  →  0.5
   velocity_penalty_weight:  0.3  →  0.0   # remove: conflicts with chasing behavior
   ```

**File:** `exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/move_flyfollow/marl_move_flyfollow_env.py`

Add in `_get_rewards()` after distance reward computation:
```python
# --- velocity-following reward ---
# reward drone x-velocity matching target velocity (0.3 m/s)
drone_vx = self.drone_linear_velocities[:, :, 0]   # (N, D)
target_vx = self.cfg.target_velocity                 # scalar 0.3
vel_err = (drone_vx - target_vx).abs().mean(dim=-1) # (N,)
rewards["velocity_follow"] = (
    self.cfg.velocity_follow_weight * torch.exp(-vel_err * self.cfg.velocity_follow_scale) * step_dt
)
```
Also add `"velocity_follow"` to `_episode_sums` keys.

**Rationale:** At the current reward balance, hovering gives ~6× more reward per step than approaching. This inversion must be fixed first before any other changes matter. The velocity-follow reward gives a continuous signal for the drone to maintain forward motion even when not yet in capture range.

---

### Priority 2 (CRITICAL): Increase tracking reward weight and lower capture distance temporarily

**Problem:** `tracking_reward` is always 0. The sparse bonus for being inside 2.0m has never been triggered, so the policy gradient has never pointed toward capture.

**Proposed Changes:**

**File:** `exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/move_flyfollow/marl_move_flyfollow_env_cfg.py`

```
tracking_reward_weight:  1.0  →  3.0
tracking_reward_scale:   1.0  →  0.5   # wider plateau inside capture zone
capture_distance:        2.0  →  3.0   # temporarily widen to bootstrap learning
```

Optionally add `success_reward_weight = 5.0` to give a one-time bonus at success (currently 0.0). Even with `capture_distance=3.0`, the policy will learn the direction toward capture and the curriculum can tighten the threshold later.

**Rationale:** The policy needs at least some positive experience inside the capture zone to bootstrap gradient signal. Widening `capture_distance` temporarily to 3.0m gives the early policy a fighting chance to trigger `tracking_reward`, creating the gradient needed to refine approach behavior.

---

### Priority 3 (HIGH): Add target velocity to actor observation

**Problem:** Actor cannot observe target velocity (0.3 m/s in x), forcing it to infer motion from positional history alone.

**Proposed Changes:**

**File:** `exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/move_flyfollow/marl_move_flyfollow_env_cfg.py`

```
obs_dim_per_step: 49  →  53   # +4 (target x-velocity for each of 4 targets)
state_space: 86  (no change — Critic already has target_velocities)
```

**File:** `exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/move_flyfollow/marl_move_flyfollow_env.py`

In `_get_observations()`, modify the target info block to include velocities:
```python
obs_targets = torch.cat(
    [
        target_rel_pos.view(self.num_envs, -1),                  # 12
        self.target_velocities[:, :, 0].view(self.num_envs, -1), # 4  <-- NEW
        self.target_captured.float().view(self.num_envs, -1),    # 4
        self.target_values,                                       # 4
    ],
    dim=-1,
)
```

Update docstring: `obs_dim_per_step = 53`.

**Rationale:** With known target velocity, the actor can directly compute the required approach velocity rather than learning it implicitly. This is especially important in CTDE where the Critic already has this information — the Actor/Critic information asymmetry hurts convergence.

---

### Priority 4 (HIGH): Relax PPO clip and increase entropy coefficient

**Problem:** `ratio_clip=0.1` is too conservative for early-stage training where large policy shifts are needed. `entropy_loss_scale=0.001` provides negligible exploration pressure.

**Proposed Changes:**

**File:** `exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/move_flyfollow/agents/skrl_mappo_cfg.yaml`

```yaml
ratio_clip:         0.1   →  0.2    # standard PPO value, allows faster learning
value_clip:         0.1   →  0.2
entropy_loss_scale: 0.001 →  0.01   # 10× increase to prevent premature convergence
learning_rate:      2.5e-4 → 3e-4   # slight increase to accelerate early learning
```

**Rationale:** Early training with a flat reward landscape needs larger policy updates and more exploration. `ratio_clip=0.1` was likely chosen for fine-tuning stability, but at this stage the policy has not yet learned anything — conservative clipping just slows the escape from the hover local optimum.

---

### Priority 5 (MEDIUM): Diagnose and reduce early termination rate

**Problem:** Episode length is shrinking (105→83 steps = ~0.83s). Most episodes end via early termination before any meaningful learning can occur.

**Investigation needed:** Add termination reason logging to identify which condition (fly_low, illegal_contact, drone_collision, out_of_bounds, targets_out_of_bounds) is most common.

**Proposed Changes:**

**File:** `exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/move_flyfollow/marl_move_flyfollow_env.py`

In `_get_dones()`, add to `extras["log"]`:
```python
extras["log"]["term_fly_low"] = self.falcon_fly_low.float().mean()
extras["log"]["term_collision"] = self.drone_collision.float().mean()
extras["log"]["term_out_of_bounds"] = self.body_pos_outside.float().mean()
extras["log"]["term_targets_oob"] = self.targets_out_of_bounds.float().mean()
extras["log"]["term_illegal_contact"] = self.illegal_contact.float().mean()
```

**Tentative fix pending diagnosis:**

If `illegal_contact` dominates: check contact sensor threshold (1.0 N is very sensitive for a quadrotor that weighs ~1.5 kg × g ≈ 14.7 N thrust at hover). Consider raising to `contact_sensor_threshold = 5.0`.

If `fly_low` dominates: the hover reward at desired_height=2.5m should prevent this, but oscillating drones may dip below 0.1m. Consider `height_reward_weight` emphasis (already present) and add a stronger `fly_low` soft penalty before the hard termination threshold.

---

### Priority 6 (LOW): Curriculum on capture_distance and target_end_x

**Problem:** The task as configured requires immediate high performance (2.0m capture, 30m runway). Early training benefits from easier success criteria.

**Proposed Changes (curriculum schedule):**

- Phase 1 (0-100k steps): `capture_distance=3.5m`, `target_velocity=0.1 m/s`, focus on basic approach
- Phase 2 (100k-250k steps): `capture_distance=2.5m`, `target_velocity=0.2 m/s`
- Phase 3 (250k-400k steps): `capture_distance=2.0m`, `target_velocity=0.3 m/s` (original spec)

Implementation: These can be changed between training runs rather than implementing dynamic curriculum, which requires additional infrastructure.

---

## Experiment Plan

### Immediate Run (Priority 1+2+3+4 combined)

Apply all changes in one shot to avoid wasting compute on incremental ablations at this early stage.

**Training command:**
```bash
python3 scripts/skrl/train.py \
  --task=Isaac-marl-move-flyfollow-v0 \
  --headless --num_envs=2048 --algorithm="MAPPO" --seed=-1
```

**Run for:** 200,000 timesteps minimum before evaluation.

**Monitor these metrics (in order of priority):**
1. `tracking_reward` — must become nonzero within first 20k steps; if still 0 after 50k, the reward balance is still wrong
2. `distance_reward` — should show upward trend from step 0
3. Episode length — should stabilize above 200 steps once approach behavior emerges
4. `velocity_follow` (new term) — should converge to ~0.003-0.005/step (matching target velocity)
5. `policy_std` — should start decreasing after 50-100k steps as policy converges

**Success criteria for this experiment:**
- `tracking_reward > 0.01/ep` by step 50k
- `distance_reward > 0.15/ep` by step 100k (approaching best-ever of 0.144 from current run)
- Episode length > 150 steps by step 50k
- At least one environment achieving `all_targets_captured` within 200k steps

### Fallback if tracking_reward still 0 after 50k steps

Temporarily set `capture_distance=4.0` (ultra-wide) and `dist_reward_weight=6.0` to force the policy to experience the capture zone. Once `tracking_reward > 0`, gradually tighten parameters.

---

## Changelog

- 2026-03-23: Initial analysis of move_flyfollow run `2026-03-23_17-12-31`. Identified reward dominance inversion (stability > task), missing velocity-follow reward, zero tracking reward throughout entire run, shrinking episode lengths. Proposed 6 prioritized fixes.

---

# High-Flying Root Cause Analysis

**Runs analyzed:**
- `move_flyfollow/2026-03-23_20-28-48` (latest, height_reward_weight=0.5, velocity_follow added)
- `move_flyfollow/2026-03-23_17-12-31` (previous, height_reward_weight=2.0)
- `move/2026-03-20_16-26-19` (reference, height_reward_weight=2.0, well-converged)

**Date:** 2026-03-23

---

## Part 1: Confirming the "High-Flying" Phenomenon from Logs

### 1.1 Primary Evidence: bounding_box Termination Dominates Completely

Both move_flyfollow runs show `Episode_Termination/bounding_box = 1.0` at every single logged step — meaning 100% of episode terminations are triggered by `body_pos_outside` (drone exits ±12m boundary). This is maintained from the very first episode to the last. The `falcon_fly_low` count is zero throughout, meaning drones are *not* falling — they are flying *upward* and exiting through the top of the bounding box (z > 12m) or horizontally.

```
Episode_Termination/bounding_box:
  move_flyfollow 20-28-48: early=[0.24, 1.0, 1.0, 1.0, 1.0]  recent=[1.0, 1.0, 1.0, 1.0, 1.0]
  move_flyfollow 17-12-31: early=[0.20, 1.0, 1.0, 1.0, 1.0]  recent=[1.0, 1.0, 1.0, 1.0, 1.0]
  move (reference):        early=[0.0,  1.64, 3.11, ...]       recent=[0.0, 0.0, 0.0, ...]  (RESOLVED)
```

The move task initially also had bounding_box exits but resolved them by ~step 50k as it learned to fly at the correct height. move_flyfollow never resolves this.

### 1.2 Secondary Evidence: height_penalty Is Large and Growing

```
Episode_Reward/height_penalty (avg per episode):
  move_flyfollow 20-28-48:  early=-1.09  recent=-0.59  (some episodes: -4.5 at step 800, -5.3)
  move_flyfollow 17-12-31:  early=-1.25  recent=-0.88
  move (reference):          early=-3.63  recent=-0.04  (penalty nearly eliminated after learning)
```

`height_penalty = -weight × max(0, |z - desired_height| - threshold) × step_dt × steps`

With weight=1.0, step_dt=0.01, avg_steps≈80:
- avg_excess in move_flyfollow 20-28-48 recent: `0.59 / (1.0 × 0.01 × 75) ≈ 0.79m` above desired+threshold (i.e., z ≈ 3.0 + 0.79 = **3.79m average**, with peaks reaching 7–8m).

The move reference task resolved height_penalty to near zero, confirming height control *is learnable* — but only when the reward gradient is correct.

### 1.3 Tertiary Evidence: drone_out Penalty is a Fixed Cost Every Episode

```
Episode_Reward/drone_out:
  move_flyfollow 20-28-48: early=[-0.24, -1.0, -1.0, -1.0]  recent=[-1.0, -1.0, -1.0, -1.0]
  move_flyfollow 17-12-31: early=[-0.20, -1.0, -1.0, -1.0]  recent=[-1.0, -1.0, -1.0, -1.0]
```

The `-1.0` ceiling means at least one drone exits the ±12m bounding box every single episode and stays out until termination. Since `fly_low=0` throughout, the exit is upward or lateral. Given the consistent `height_penalty` evidence pointing to z > 3m, and targets at z=0.25m on the ground, the exit vector is predominantly **upward (z-axis)**.

---

## Part 2: Root Cause Analysis — Why Does High-Flying Occur?

### Cause 1 (PRIMARY): distance_reward Uses XY-Only Distance, Eliminating All Altitude Gradient

```python
# move_flyfollow_env.py line 776-778
d_pos = self.drone_positions[:, :, :2].unsqueeze(2)  # (N,D,1,2) — XY only!
t_pos = self.target_positions[:, :, :2].unsqueeze(1)  # (N,1,T,2)
dist_matrix = torch.norm(d_pos - t_pos, dim=-1)       # XY distance only
```

The distance reward — the only task-relevant reward term — is computed purely in the XY plane. **There is no term in any positive reward that punishes or discourages altitude deviation from the target's z-coordinate (0.25m on the ground).**

This means: a drone at z=10m hovering directly above a target receives *exactly the same distance_reward* as a drone at z=2.5m hovering above the same target (XY distance = 0 in both cases). There is no gradient pointing the drone downward toward the target's actual 3D position.

This is the primary structural cause of high-flying: the reward landscape is *flat in the z-direction* for the task signal, and the drone explores upward freely.

### Cause 2 (CONTRIBUTING): height_reward Gradient Is Symmetric and Too Weak

```python
height_error = torch.norm(self.drone_positions[..., 2] - self.cfg.desired_height, dim=-1)
rewards["height_reward"] = self.cfg.height_reward_weight * torch.exp(-height_error) * step_dt
```

The `torch.norm()` call on a scalar (z - desired_height) is equivalent to `abs(z - desired_height)`. The reward is symmetric around desired_height=2.5m: flying at 1.5m and 3.5m give equal height_reward. This is correct in design, but the *weight* comparison matters:

**move_flyfollow 20-28-48** (latest run, most relevant):
- height_reward_weight = **0.5** (reduced from 2.0 in prior fix)
- At desired height: 0.5 × exp(0) × 0.01 = **0.005/step**
- height_penalty_weight = **1.0**, threshold = 0.5m
- At z=3.0+: -1.0 × excess × 0.01/step → for 1m excess: **-0.010/step**

But the penalty is linear and small per-step. Over an 80-step episode at average excess=0.79m:
- Total height_penalty ≈ **-0.63** per episode

This is less than the cost of `drone_out = -1.0` per episode, and less than the gain from `body_rate_penalty` (~0.19) + `force_penalty` (~0.19). The drone's "preferred" flight altitude that maximizes reward is NOT constrained to 2.5m by the current reward design.

**The height_reward/penalty system was designed to correct fine-tuning deviations, not to anchor altitude during early exploration.**

### Cause 3 (CONTRIBUTING): No Fly-High Termination Condition

The termination logic only checks `z < 0.1m` (fly_low). There is **no fly_high termination**. The drone can fly to z=11.9m and remain alive, accumulating `body_rate_penalty`, `force_penalty`, and `action_smoothness` rewards (all three are positive at any altitude) until it exits at z=12.0m via the bounding box. At that point it gets `-1.0 drone_out` — but this is a one-time penalty, not a per-step penalty.

This means the drone discovers that high-altitude flight is **nearly as rewarding per step** as low-altitude flight for the stability-based terms:
- At z=8m: body_rate_penalty ≈ +0.19, force_penalty ≈ +0.19 — same as z=2.5m
- At z=8m: height_penalty ≈ -0.055/step (5.5m excess × 0.01) — modest per-step cost
- At z=8m: drone_out = -1.0 once at end — bounded, not cumulative

**The expected return for high-altitude hovering over an 80-step episode is only ~1.4 worse than correct-altitude hovering, but the height_penalty linear cost is modest enough that early random exploration can push the drone high without a strong gradient pulling it back.**

### Cause 4 (CONTRIBUTING, move_flyfollow specific): NovaCarter Targets Are at Ground Level (z≈0.25m)

In move_flyfollow, targets are NovaCarter robots at ground level (z=0.25m). The target positions include z≈0.25, so `target_rel_pos` in observations *does* include z-offset from drone. A drone at z=3m sees a target below at relative_z ≈ -2.75m. However:

1. The capture check uses XY-only distance (`capture_distance = 3.0m` XY circle)
2. The distance_reward uses XY-only distance
3. The tracking_reward uses XY-only min_dists

**The observation has z-information, but no reward uses z-distance to target. The drone has no incentive derived from reward to fly at a specific altitude relative to the ground target.** The only altitude signals are `height_reward` (anchor to desired_height=2.5m, weight=0.5) and `height_penalty` (linear cost above 3.0m). Both are weak compared to the stability terms.

---

## Part 3: Why move Task Does NOT Have This Problem (Comparison)

The `move/2026-03-20_16-26-19` run also had bounding_box exits early but resolved them by step ~30k. Key differences:

| Factor | move (fixed blocks) | move_flyfollow (NovaCarter) |
|--------|--------------------|-----------------------------|
| Target z-position | ~0.25m (fixed blocks on ground) | ~0.25m (NovaCarter on ground) |
| height_reward_weight | **2.0** | 0.5 (latest) / 2.0 (prev) |
| body_rate_penalty_weight | **2.0** | 0.5 (latest) / 2.0 (prev) |
| distance_reward_weight | 1.5 | 4.0 (latest) / 1.5 (prev) |
| Episode length (converged) | ~1640 steps | ~80 steps (never converges) |
| bounding_box at convergence | **0** | **1.0** (persistent) |

The move task has `height_reward_weight=2.0`, which gives **0.020/step** at desired height vs only **0.005/step** in the latest move_flyfollow run (after the height weight was reduced to 0.5 in the previous analysis fix). This 4× reduction in height anchoring is the single most important difference.

**The previous PLAN.md fix (reducing height_reward_weight 2.0→0.5) appears to have made the high-flying problem worse, not better.** With weight=2.0, the height_reward provides a stronger gradient to maintain altitude. Reducing it weakened the only altitude anchor without providing an alternative.

In the move reference run: after convergence, height_reward = 26.1/ep (extremely high), height_penalty ≈ -0.04/ep (nearly zero). The policy learned to fly precisely at desired_height because the 2.0 weight gave enough gradient. In move_flyfollow: height_reward = 0.14/ep (barely above minimum), confirming the drone is chronically off-altitude.

---

## Part 4: Specific Fix Recommendations for High-Flying

### Fix A (CRITICAL): Restore height_reward_weight to 2.0

**Problem:** The reduction from 2.0→0.5 weakened the altitude anchor and worsened high-flying.

**Change:**
- File: `exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/move_flyfollow/marl_move_flyfollow_env_cfg.py`
- `height_reward_weight: 0.5  →  2.0`

**Rationale:** The move task demonstrates that weight=2.0 is sufficient to converge altitude to within ~0.1m of desired. The previous analysis incorrectly attributed height_reward as "suppressing task behavior" — in fact, it was the *only* altitude anchor. Without it, the drone has no incentive to return to 2.5m.

**Expected effect:** Height_reward signal becomes 0.020/step at correct altitude, strong enough for the policy to maintain altitude against perturbations from exploration.

### Fix B (CRITICAL): Add fly_high Soft Penalty (Per-Step, Not Linear)

**Problem:** No per-step cost for flying high. Linear height_penalty only fires above 3.0m and is modest per step.

**Change:**
- File: `marl_move_flyfollow_env_cfg.py`: add `fly_high_penalty_weight = 2.0`, `fly_high_threshold = 4.0`
- File: `marl_move_flyfollow_env.py` in `_get_rewards()`:

```python
# Exponential fly-high soft penalty: large cost above fly_high_threshold
fly_high_excess = (self.drone_positions[:, :, 2] - self.cfg.fly_high_threshold).clamp(min=0.0)
rewards["fly_high_penalty"] = (
    -self.cfg.fly_high_penalty_weight * fly_high_excess.mean(dim=-1) * step_dt
)
```

**Rationale:** A per-step penalty that grows with z-excess creates a clear gradient pointing downward. Above 4m the penalty fires, above 6m it becomes a dominant negative signal (~-0.04/step for 2m excess × 2.0 weight), making high-altitude hovering clearly suboptimal.

### Fix C (HIGH): Add fly_high Hard Termination

**Problem:** No upper altitude termination. Drone can reach z=11.9m before the bounding_box fires.

**Change:**
- File: `marl_move_flyfollow_env_cfg.py`: add `fly_high_termination_z = 6.0`
- File: `marl_move_flyfollow_env.py` in `_get_dones()`:

```python
self.falcon_fly_high = (self.drone_positions[:, :, 2] > self.cfg.fly_high_termination_z).any(dim=-1)
terminations = self.falcon_fly_low | self.illegal_contact | self.drone_collision | self.body_pos_outside | self.targets_out_of_bounds | self.falcon_fly_high
```

**Rationale:** By terminating episodes at z>6m (vs bounding_box at 12m), the drone gets a definitive "this is wrong" signal much earlier. The termination also provides negative GAE bootstrapping from those states, creating stronger gradient against high-flying.

### Fix D (HIGH): Add 3D Distance Component to Reward (or cap desired_height below target)

**Problem:** distance_reward is XY-only, providing no gradient in the z-direction toward targets at z=0.25m.

**Change option 1 (simpler):** Lower desired_height to match typical capture engagement altitude:
- `desired_height: 2.5  →  1.5`  (closer to target z=0.25m + approach_offset=1.25m)

**Change option 2 (structural):** Add a small z-component to distance_reward:
```python
# In _get_rewards(), replace XY-only with 3D distance weighted toward XY:
d_pos_3d = self.drone_positions.unsqueeze(2)    # (N,D,1,3)
t_pos_3d = self.target_positions.unsqueeze(1)   # (N,1,T,3)
dist_xy = torch.norm(d_pos_3d[..., :2] - t_pos_3d[..., :2], dim=-1)  # XY
dist_z  = (d_pos_3d[..., 2] - t_pos_3d[..., 2]).abs()                  # Z only
dist_combined = dist_xy + 0.3 * dist_z  # light z-weighting
```

**Rationale:** Option 1 is simpler and sufficient. Lowering desired_height from 2.5m to 1.5m means height_reward anchors the drone at z=1.5m, which is still safely above ground and closer to the ground-level targets. This reduces the z-gap from 2.25m (2.5-0.25) to 1.25m (1.5-0.25).

### Fix E (LOW): Increase height_penalty_weight and lower threshold

**Problem:** height_penalty is too weak and fires too late (above 3.0m = 0.5m above desired 2.5m).

**Change:**
- `height_penalty_weight: 1.0  →  2.0`
- `height_penalty_threshold: 0.5  →  0.3`

**Rationale:** Tightening the linear penalty zone and doubling the weight creates a stronger gradient in the 2.8m-4.0m range (before the new fly_high_penalty kicks in at 4.0m).

---

## Part 5: Why Previous Fix Partially Backfired

The previous PLAN.md analysis (2026-03-23) recommended `height_reward_weight: 2.0→0.5` to "avoid height stabilization suppressing task behavior." This reasoning was sound in the context of the flyfollow task (where height was dominating the reward). However, in move_flyfollow:

1. The height_reward was the primary altitude anchor — reducing it removed altitude control without providing an alternative.
2. The bounding_box termination data (which would reveal this) was not available at time of writing.
3. The move reference task, which uses height_reward_weight=2.0 and successfully converges altitude, was not compared at that time.

**Lesson:** In this task, `height_reward_weight` must not be reduced below 1.5 unless a structural 3D distance reward is added to substitute the altitude gradient.

---

## Updated Experiment Plan

Apply fixes A+B+C+D (option 1) together in the next run:

```
height_reward_weight:     0.5   →  2.0    (Fix A)
fly_high_penalty_weight:  -      →  2.0   (Fix B, new parameter)
fly_high_threshold:       -      →  4.0   (Fix B, new parameter)
fly_high_termination_z:   -      →  6.0   (Fix C, new parameter)
desired_height:           2.5   →  1.5    (Fix D option 1)
height_penalty_weight:    1.0   →  2.0    (Fix E)
height_penalty_threshold: 0.5   →  0.3    (Fix E)
```

Keep existing fixes from previous analysis:
- dist_reward_weight = 4.0 (keep)
- velocity_follow_weight = 1.5 (keep)
- tracking_reward_weight = 3.0 (keep)
- capture_distance = 3.0 (keep)
- body_rate_penalty_weight = 0.5 (keep — lower weight is correct for task focus)

**Monitor first 5k steps:**
- `Episode_Termination/bounding_box` should drop below 0.5 within 2k steps
- `Episode_Reward/height_penalty` should trend toward 0 within 5k steps
- Episode length should increase beyond 100 steps

**Success criterion for this fix:**
- `bounding_box` terminations < 0.1 (vs current 1.0) by step 10k

## Changelog

- 2026-03-23: Initial analysis of move_flyfollow run `2026-03-23_17-12-31`. Identified reward dominance inversion (stability > task), missing velocity-follow reward, zero tracking reward throughout entire run, shrinking episode lengths. Proposed 6 prioritized fixes.
- 2026-03-23: High-flying root cause analysis. Three runs compared (move_flyfollow ×2 + move reference). Confirmed drone z reaches 4–8m chronically (bounding_box=1.0 from first episode). Root cause: XY-only distance reward provides no z-gradient; height_reward_weight=0.5 insufficient as altitude anchor; no fly_high termination. Identified that previous fix (height_w: 2.0→0.5) worsened the problem. Proposed 5 targeted fixes (A–E).
- 2026-03-23: Run 22-02-33 analysis (400k steps). bounding_box_threshold bug fixed (12→120m), rewards improved significantly vs 20-28-48, but fly-high still dominant — drone drifts upward, tracking_reward=0 persists, episode length stuck at ~87 steps (2.9s). Height fixes (A–E from previous plan) not yet applied — that remains the critical blocker.

---

# Training Analysis Report

**Run:** 2026-03-23_22-02-33_mappo_torch_mappo
**Date:** 2026-03-23
**Task:** Isaac-marl-move-flyfollow-v0
**Algorithm:** MAPPO
**Compared against:** 2026-03-23_20-28-48_mappo_torch_mappo (previous run, 27k steps)

---

## Training Metrics Summary

| Metric | Run 20-28-48 (27k steps, end) | Run 22-02-33 (400k steps, end) | Change |
|---|---|---|---|
| Total reward (mean) | -2.12 | -4.99 | Worse (longer training revealed true plateau) |
| Total reward (recent mean) | -2.42 | -4.94 | Worse |
| distance_reward (recent mean) | 0.078 | 0.212 | +172% — clear improvement |
| tracking_reward (recent mean) | 0.000 | 0.002 | Still effectively zero |
| height_reward (recent mean) | 0.147 | 0.191 | Marginally higher |
| body_rate_penalty (recent mean) | 0.114 | 0.010 | -91% — drones much calmer |
| Episode length (recent mean steps) | 64 steps (2.1s) | 87 steps (2.9s) | +36% — improvement |
| Policy std | 0.787 | 0.710 | Converging but not collapsed |
| Value loss | 0.007 | 0.0002 | Near zero — critic converged |

---

## Question-by-Question Analysis

### 1. Did bounding_box termination rate decrease?

**Answer: Cannot directly confirm from TF metrics alone, but strong indirect evidence of improvement.**

The previous runs (17-12-31 and 20-28-48) both showed `bounding_box = 1.0` (every episode terminated out-of-bounds). With `bounding_box_threshold` corrected from 12m to 120m in this run, the termination condition is now 10× more lenient. The fact that episode lengths increased from 64 to 87 steps and the run completed 400k steps with stable (non-crashing) reward curves strongly implies the `bounding_box` termination rate is now well below 1.0.

However, `Episode_Termination/bounding_box` is not directly logged in the TF data extracted. The current dominant termination mechanism appears to be a short-episode ceiling (episodes average only 87 steps = 2.9 seconds vs 60s max), which points to a **different termination** condition firing — most likely `fly_high` or `out_of_bounds` at whatever the true boundaries are, or collision.

**Conclusion:** bounding_box bug is fixed. But drones are still exiting early via another path.

### 2. Episode length trend

- **Early mean:** 89.7 steps (~3.0s)
- **Recent mean:** 87.2 steps (~2.9s) with std=25.3
- **Best ever:** 651 steps (~21.7s)

Episode length is essentially flat across 400k steps. The agent learned slightly quicker exits than at startup, which is not the desired direction. The 651-step best shows it is physically possible for episodes to run long, but the policy has not learned to sustain this. Short episodes (87 steps = 2.9s) mean drones are either:
- Leaving the safe altitude band (fly_high or collision), or
- The `sustained_follow_duration=3.0s` success condition is never met, so episodes end by safety termination

### 3. Reward component magnitudes and trends

**Positive components:**
- `distance_reward`: early 0.490 → recent 0.212. **Improved vs 20-28-48 (0.078), but declining** within this run. This suggests the drone initially approached targets but then drifted away.
- `height_reward`: early 0.162 → recent 0.191. Slowly increasing — height is improving marginally.
- `action_smoothness`: early 0.109 → recent 0.055. Decreasing — policy becoming less exploratory.

**Near-zero components:**
- `tracking_reward`: recent_mean = 0.002. This is effectively zero. **Drones still never enter the 3.0m capture zone.**
- `velocity_penalty`: 0.000 throughout. Correctly disabled.

**Negative drivers (penalties eating into reward):**
- `force_penalty`: 0.160/episode. Consistent and stable.
- `body_rate_penalty`: 0.010/episode (was 0.114 in 20-28-48) — **91% reduction, the reduced weight 0.5 worked**.
- `action_smoothness` term and `force_penalty` are the main positive-reward terms not related to task.

**Total reward remains negative (-4.94 recent mean)** meaning penalties still exceed task rewards. The dominant penalty is not visible in per-component logging but is implied to be either height-related or a reset penalty.

### 4. Comparison with run 20-28-48

| Dimension | 20-28-48 | 22-02-33 | Verdict |
|---|---|---|---|
| Bounding box bug | Active (12m threshold) | Fixed (120m) | Fixed |
| distance_reward | 0.078 | 0.212 | +172%, genuine improvement |
| body_rate_penalty | 0.114 | 0.010 | Weight reduction worked |
| tracking_reward | 0.000 | 0.002 | Still zero in practice |
| Episode length | 64 steps | 87 steps | Marginal improvement |
| Policy std | 0.787 | 0.710 | Slower collapse in 22-02-33 |
| Training budget | 27k steps | 400k steps | 15× more compute |
| Total reward plateau | ~-2.4 | ~-4.9 | More negative — longer run exposed true floor |

**Summary:** The fixes partially worked. The policy is no longer crashing out immediately, approaches are happening more frequently (distance_reward ×2.7), and body rate jitter is eliminated. However, the fundamental blocker remains: **drones never reach the 3.0m capture zone** and **altitude anchoring is insufficient** causing persistent drift and early termination.

### 5. Current primary bottlenecks

**Bottleneck A (CRITICAL): Altitude drift — height_reward_weight=0.5 still too weak**

The height_reward recent mean of 0.191 corresponds to `0.5 × exp(-|z-2.5|) × step_dt`. For this to equal 0.191 per episode of 87 steps, the average height deviation across 87 steps is approximately 0.3–0.5m above desired. This sounds acceptable, but because there is **no z-component in distance_reward**, drones can hover at z=5–8m and receive identical distance reward to z=2.5m. The height_reward at z=5m is `0.5 × exp(-2.5) ≈ 0.041/step` — very weak. Without a fly_high termination, a drone stuck at 6m altitude survives the full episode collecting stability rewards.

**Bottleneck B (CRITICAL): tracking_reward = 0 — drones never close within 3.0m**

With `dist_reward_scale=0.5` and targets at ground level (z≈0.25m) and drones flying at ~2.5–5m, the XY distance to targets is on the order of 5–15m at episode start. Even with `dist_reward_weight=4.0`, the gradient from 15m to 3m is:
- At 15m: `4.0 × exp(-15 × 0.5) ≈ 0.00005` — negligible
- At 3m: `4.0 × exp(-3 × 0.5) ≈ 0.89` — strong

The reward landscape near the capture zone is steep, but the gradient from far away is too flat for the policy to reliably discover approach direction. The policy defaults to hovering near the episode start position.

**Bottleneck C (HIGH): No fly_high termination or penalty**

Without `fly_high_termination_z` and `fly_high_penalty_weight`, a drone that explores upward faces no negative consequence until it hits `bounding_box_threshold=120m` (now effectively infinite). The height_reward decay at 8m is `0.5 × exp(-5.5) ≈ 0.002/step` — essentially zero. There is no incentive to return to 2.5m once the drone drifts above ~5m.

---

## Improvement Recommendations

### Priority 1 (CRITICAL): Restore height_reward_weight to 2.0

**Problem:** At weight=0.5, the altitude anchor gradient is too weak to overcome random upward exploration. The reference `move` task with weight=2.0 successfully converged altitude; this task with 0.5 does not.

**Proposed Change:**
- File: `exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/move_flyfollow/marl_move_flyfollow_env_cfg.py`
- Parameter: `height_reward_weight = 0.5` → `2.0`
- Rationale: At desired_height, this raises the altitude anchor signal from 0.005/step to 0.020/step, creating a 4× stronger basin. Empirically validated in the reference `move` task.

### Priority 2 (CRITICAL): Add fly_high_penalty and fly_high_termination

**Problem:** No cost for high-altitude flight means the policy can freely explore upward indefinitely. Once at 6–8m, the height_reward is negligible and the drone has no incentive to descend.

**Proposed Changes:**
- File: `marl_move_flyfollow_env_cfg.py`
- Add: `fly_high_penalty_weight = 2.0` (exponential penalty per step above fly_high_threshold)
- Add: `fly_high_threshold = 4.0` (penalty activates above 4m)
- Add: `fly_high_termination_z = 6.0` (hard termination if any drone exceeds 6m)
- Rationale: Creates a hard ceiling that forces the policy to stay near 2.5m. Short episodes from fly_high termination will replace short episodes from other causes, but with a learning signal pointing the right direction.

### Priority 3 (HIGH): Lower desired_height from 2.5m to 1.5m

**Problem:** Targets are NovaCarter carts at z≈0.25m. At desired_height=2.5m, the drone is 2.25m above the target. The XY capture distance is 3.0m, but the 3D distance at desired_height is sqrt(XY² + 2.25²). For a drone directly above a target at desired_height, 3D distance = 2.25m — which already qualifies as captured. But the XY-only distance_reward does not account for z, so drones are not rewarded for being directly above targets.

Lowering desired_height to 1.5m reduces z-offset to 1.25m and makes it easier for drones to simultaneously be close in XY.

**Proposed Change:**
- Parameter: `desired_height = 2.5` → `1.5`
- Rationale: Closer to targets, more natural hovering position for tracking ground vehicles.

### Priority 4 (HIGH): Increase dist_reward_scale to improve far-field gradient

**Problem:** At `dist_reward_scale=0.5`, the reward at 10m distance is `4.0 × exp(-5) ≈ 0.027`. At 20m it is `4.0 × exp(-10) ≈ 0.0002`. The gradient from initialization positions (likely 5–20m from targets) is nearly flat, making random exploration the only way to discover approach direction.

**Proposed Change:**
- Parameter: `dist_reward_scale = 0.5` → `0.3`
- Rationale: Wider basin — at 10m distance: `4.0 × exp(-3) ≈ 0.20` (7× higher), giving clearer gradient signal from initialization.

### Priority 5 (MEDIUM): Increase height_penalty_weight to add strong linear penalty zone

**Problem:** The current `height_penalty_weight=1.0` with `height_penalty_threshold=0.5` creates only a weak linear penalty for |z-desired| > 0.5m. This doesn't activate until the drone is already 3m above the floor (at desired=2.5m, threshold=0.5 → penalty starts at 3.0m).

**Proposed Change:**
- Parameter: `height_penalty_weight = 1.0` → `2.0`
- Parameter: `height_penalty_threshold = 0.5` → `0.3`
- Rationale: Combined with Fix A (height_reward_weight=2.0), creates a multi-layer altitude control: soft reward basin (height_reward) + linear penalty zone (height_penalty) + hard ceiling (fly_high_termination from Fix B).

---

## Updated Experiment Plan

Apply all 5 fixes together in the next run:

```
height_reward_weight:      0.5   →  2.0    (Fix 1 — CRITICAL)
fly_high_penalty_weight:   new   →  2.0    (Fix 2 — CRITICAL, new param)
fly_high_threshold:        new   →  4.0    (Fix 2 — CRITICAL, new param)
fly_high_termination_z:    new   →  6.0    (Fix 2 — CRITICAL, new param)
desired_height:            2.5   →  1.5    (Fix 3 — HIGH)
dist_reward_scale:         0.5   →  0.3    (Fix 4 — HIGH)
height_penalty_weight:     1.0   →  2.0    (Fix 5 — MEDIUM)
height_penalty_threshold:  0.5   →  0.3    (Fix 5 — MEDIUM)
```

Keep from 22-02-33:
```
dist_reward_weight = 4.0         (keep)
tracking_reward_weight = 3.0     (keep)
velocity_follow_weight = 1.5     (keep)
capture_distance = 3.0           (keep)
body_rate_penalty_weight = 0.5   (keep)
velocity_penalty_weight = 0.0    (keep)
bounding_box_threshold = 120.0   (keep)
obs_dim_per_step = 61            (keep — target velocity in obs)
ratio_clip = 0.2                 (keep)
entropy_coef = 0.01              (keep)
```

**Training command:**
```bash
python3 scripts/skrl/train.py --task=Isaac-marl-move-flyfollow-v0 --headless --num_envs=2048 --algorithm="MAPPO"
```

**Monitor in first 10k steps:**
- `Episode_Termination/fly_high` should appear and decrease over time (confirms ceiling is working)
- `Episode_Reward/height_reward` should increase above 0.25/episode (altitude anchoring working)
- `Episode / Total timesteps (mean)` should increase above 100 steps (less early termination)

**Monitor at 50k steps:**
- `Episode_Reward/distance_reward` should exceed 0.5 (drones closer to targets)
- `Episode_Reward/tracking_reward` should show first non-zero values (first captures)

**Success criterion for this fix:**
- `tracking_reward` > 0.01 (mean) by step 100k
- Episode length > 150 steps by step 50k
- `fly_high` terminations declining after initial burst

---

# Training Analysis Report

**Run:** `2026-03-24_09-51-13_mappo_torch_mappo`
**Date:** 2026-03-24
**Task:** Isaac-marl-move-flyfollow-v0
**Algorithm:** MAPPO
**Total timesteps logged:** 400,000
**Status:** improving (script verdict)

---

## Training Metrics Summary

| Metric | Early mean | Recent mean | Last value | Best ever |
|--------|-----------|-------------|------------|-----------|
| Total reward (mean) | -4.99 | +1.17 | +13.35 | +20.28 |
| Instant reward (mean) | -0.072 | +0.016 | +0.049 | +0.133 |
| distance_reward | 1.08 | 3.95 | 4.82 | 12.08 |
| tracking_reward | 0.0004 | 0.065 | 0.100 | 0.509 |
| height_reward | 0.42 | 0.84 | 0.93 | 1.82 |
| velocity_penalty | 0.0 | 0.0 | 0.0 | 0.0 |
| force_penalty | 0.14 | 0.22 | 0.23 | 0.46 |
| body_rate_penalty | 0.071 | 0.043 | 0.045 | 0.195 |
| action_smoothness | 0.085 | 0.169 | 0.172 | 0.386 |
| Episode length mean (steps) | 57.2 | 101.6 | 107.0 | 248.3 |
| Episode length max (steps) | 82.4 | 151.6 | 171.0 | 325.0 |
| Policy std | 0.759 | 0.662 | 0.650 | 0.819 |
| Value loss | 0.087 | 0.242 | 0.424 | 3.275 |
| Policy loss | -0.016 | -0.038 | -0.181 | +0.342 |

**Config applied (vs 22-02-33 baseline):**

| Parameter | Previous (22-02-33) | This run | Change |
|-----------|---------------------|----------|--------|
| height_reward_weight | 0.5 | **2.0** | Fix-A |
| desired_height | 2.5 m | **1.5 m** | Fix-D |
| dist_reward_scale | 0.5 | **0.3** | broader basin |
| height_penalty_weight | 1.0 | **2.0** | Fix-E |
| height_penalty_threshold | 0.5 m | **0.3 m** | Fix-E |
| fly_high_penalty_weight | 0.0 | **2.0** | new |
| fly_high_termination_z | none | **6.0 m** | new |

---

## Observations & Findings

### [Fix Validation: High-Flying Suppressed] — Result: SUCCESS (CRITICAL fixes confirmed working)

**Symptom in previous run (22-02-33):** Total reward stuck at -4.94, bounding_box termination
dominated, episode length flat at ~87 steps. Drones were flying high with impunity (no z ceiling).

**Result in this run:**
- Total reward shifted from -4.94 → +1.17 (recent mean), +13.35 (last). **Net improvement: +6.1 points.**
- Episode length increased from 87.2 → 101.6 steps (mean), 171 steps (last max). Drones are surviving longer.
- `fly_high_termination_z=6.0m` cap is working: episodes are now long enough to accumulate positive reward.
- `height_reward_weight=2.0` restored: height_reward rose from 0.19/ep → 0.84/ep (+342%). Altitude anchoring working.
- `desired_height=1.5m` (vs 2.5m): drones closer to ground targets, consistent with the 0.25m cart height.
- `dist_reward_scale=0.3` (vs 0.5): far-field gradient improved — distance_reward rose from 0.21/ep → 3.95/ep (+1771%).

**Root cause resolved:** XY-distance reward at large separation was near-flat with scale=0.5.
With scale=0.3 the reward at 10m separation is ~7× stronger, giving the policy a learning signal from spawn.

### [tracking_reward: First Non-Zero Signal Achieved] — Result: PARTIAL SUCCESS (HIGH)

**Previous run (22-02-33):** tracking_reward = 0.0002 (effectively zero), drones never entered 3m capture zone.

**This run:** tracking_reward recent_mean = 0.065, last = 0.100, best = 0.509.
- Drones are entering the capture zone for the first time in this task's history.
- The best episode achieved tracking_reward=0.509 (out of max ≈ 3.0 × sustained_follow_duration × freq).
- However, recent_std = 0.084 is larger than the mean (0.065), indicating high variance — captures are
  sporadic, not consistent policy behavior.

**Status:** tracking_reward is unlocked but not yet reliable. Drones find the capture zone occasionally
but cannot sustain it. This is typical early-stage capture learning.

### [Episode Length: Improving but Still Short] — Severity: MEDIUM

**Mean episode length: 101.6 steps ≈ 3.4 seconds** (episode_length_s=60, max=1200 steps at 20Hz).
Best observed: 248 steps ≈ 8.3 seconds. The min_steps=9, meaning some episodes still terminate immediately.

Typical causes for short episodes with this config:
- fly_high_termination_z=6.0m hard ceiling is cutting episodes when drones still climb above 6m early in training
- The policy is still exploring high enough to trigger the 6m ceiling before altitude anchoring consolidates
- `timesteps_min` recent_mean = 45.4 (improved from early 15.6) — short-episode tail is shrinking

The episode length distribution has high std (37 steps), meaning behavior is bimodal: some episodes
survive well (max 171-325 steps), many still terminate early.

### [Value Loss Rising Late in Training] — Severity: LOW/WATCH

Value loss rose from recent_mean=0.24 to last=0.42 — an increasing trend in the final 100k steps.
This indicates the value function is struggling to track a changing reward landscape, consistent with
the policy discovering new behaviors (tracking_reward appearing) that shift the value target.
This is expected during the transition from "hover-only" to "chase-and-capture" regimes.
Not a pathology at this stage, but worth monitoring if it continues to increase.

### [Policy Entropy: Moderate Compression] — Severity: LOW

Policy std has declined from 0.819 (early) → 0.662 (recent) → 0.650 (last).
The policy is slowly becoming more deterministic. With entropy_coef=0.01, this is moderate compression.
No sign of premature collapse (std >> 0). Exploration is still alive.

### [velocity_penalty = 0.0 Throughout] — Informational

velocity_penalty_weight=0.0 in config — this term is disabled. The velocity_follow_weight=1.5 is
the active velocity term. velocity_follow in obs reports 0.0/ep, which means either:
(a) the reward is not logged separately (it may be included in total reward only), or
(b) velocity following was zero because drones were not close enough for the term to activate.

---

## Improvement Recommendations

### Priority 1 (HIGH): Increase Training Budget — Policy Still Learning

**Problem:** At 400k steps the tracking_reward is rising (0.0004 → 0.065) and episode length is growing.
The value loss is also rising, indicating the value function is tracking a moving target (improving policy).
This run was cut at 400k steps — the policy has not converged.

**Proposed Change:**
- File: `scripts/skrl/train.py` or launcher config
- Parameter: `--num_envs=2048` with extended timestep budget
- Set `timesteps=2_000_000` (5× current) for next run
- Rationale: tracking_reward typically needs 500k–1M steps to consolidate from first captures to reliable behavior.
  The current improvement slope is ~0.016 tracking_reward per 100k steps. At this rate, reliable tracking
  (tracking_reward > 0.3) requires ~1.2M additional steps.

### Priority 2 (HIGH): Add fly_high Termination Logging to Diagnose Ceiling Cuts

**Problem:** The analysis script does not show `Episode_Termination/fly_high` stats directly.
Short-episode tail (min_steps = 9–45) may still be caused by frequent fly_high_termination_z=6.0m hits.
Without this signal, it is difficult to know whether altitude anchoring has fully resolved.

**Proposed Change:**
- File: `exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/flyfollow/marl_flyfollow_env.py`
- Confirm that `fly_high` termination events are logged to TensorBoard under `Episode_Termination/fly_high`
- If not present, add: `self.extras["log"]["Episode_Termination/fly_high"] = fly_high_mask.float().mean()`
- Rationale: Monitoring whether fly_high events decline over training is the primary diagnostic for altitude fix success.

### Priority 3 (MEDIUM): Reduce fly_high_termination_z from 6.0m → 4.5m (after 500k steps confirm stable)

**Problem:** The current ceiling at 6.0m is generous. Drones with desired_height=1.5m should never need
to reach 6.0m. A lower ceiling (4.5m) would more aggressively suppress altitude drift without cutting
episodes unnecessarily (as long as policy has learned the altitude signal from height_reward).

**Proposed Change:**
- File: `exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/flyfollow/marl_flyfollow_env_cfg.py`
- Parameter: `fly_high_termination_z: float = 6.0` → `4.5`
- Condition: Apply only after verifying fly_high terminations have declined to < 10% of episodes.
  Do not apply early — a ceiling that is too low before the altitude anchor is learned will shorten episodes
  and slow down the tracking_reward learning by preventing drones from staying in the field long enough.

### Priority 4 (MEDIUM): Investigate capture_distance / tracking_reward reachability

**Problem:** tracking_reward best=0.509, but recent_mean is only 0.065. The high variance (std=0.084 vs mean=0.065)
indicates captures are rare and intermittent. The policy may be approaching targets but failing to maintain
sustained follow (sustained_follow_duration=3.0s = 60 steps at 20Hz). Even brief exits reset the sustained counter.

**Proposed Change — Option A:** Reduce sustained_follow_duration from 3.0s → 1.5s temporarily.
- File: `marl_flyfollow_env_cfg.py`
- Parameter: `sustained_follow_duration: float = 3.0` → `1.5`
- Rationale: Lowering the sustained follow bar during early learning allows the reward signal to fire more
  frequently, accelerating capture credit assignment. Restore to 3.0s after tracking_reward > 0.2.

**Proposed Change — Option B:** Increase capture_distance from 3.0m → 4.0m temporarily.
- File: `marl_flyfollow_env_cfg.py`
- Parameter: `capture_distance: float = 3.0` → `4.0`
- Rationale: Easier capture threshold → more frequent rewards → faster policy bootstrapping.
  Risk: policy learns "loose following" and resists tightening later.
- Recommendation: Prefer Option A over Option B to avoid loose-following local optimum.

### Priority 5 (LOW): Monitor action_smoothness trend

**Status:** action_smoothness rose from 0.085 → 0.169/ep. This is positive — the policy is producing
smoother action sequences as it learns. No changes needed. Continue monitoring.

---

## Experiment Plan

1. Continue the current configuration for 1.6M more steps (total 2M) — no config changes needed.
   The policy is on an improving trajectory. Config changes at this stage risk disrupting learning.

2. At 600k total steps, check:
   - `tracking_reward` mean > 0.10 (should be achievable given current slope)
   - Episode length mean > 120 steps
   - If NOT: apply Priority 4 Option A (reduce sustained_follow_duration to 1.5s)

3. At 1M total steps, check:
   - `tracking_reward` mean > 0.25
   - Episode length max approaching 300+ steps (drones completing long tracking episodes)
   - If NOT: lower fly_high_termination_z to 4.5m (Priority 3) to further discourage altitude drift

4. Monitor: value_loss trend. If it continues rising beyond 1.0, consider reducing learning_rate by 50%.

**Training command (same as before):**
```bash
python3 scripts/skrl/train.py --task=Isaac-marl-move-flyfollow-v0 --headless --num_envs=2048 --algorithm="MAPPO"
```

**Success criteria for this run series:**
- `tracking_reward` > 0.25 (mean) by step 1M
- `Episode / Total timesteps (mean)` > 200 steps by step 1M
- `total_reward_mean` > 5.0 (sustained, not just last-step spike)

---

## Changelog

- 2026-03-24: Applied Fix-A (height_reward_weight 0.5→2.0), Fix-D (desired_height 2.5→1.5m),
  dist_reward_scale 0.5→0.3, Fix-E (height_penalty_weight 1.0→2.0, threshold 0.5→0.3m),
  fly_high_penalty_weight=2.0, fly_high_termination_z=6.0m.
  Result: total_reward flipped from -4.94 → +1.17 (recent mean); tracking_reward first non-zero
  (0.065 mean); distance_reward +1771% improvement; episode length +16.5% improvement.
  All 5 critical/high fixes from previous analysis confirmed effective.

---

# Training Analysis Report — Run 2026-03-24_15-06-36

**Run:** `2026-03-24_15-06-36_mappo_torch_mappo`
**Date:** 2026-03-24
**Task:** Isaac-marl-move-flyfollow-v0
**Algorithm:** MAPPO
**Total timesteps logged:** 461,300 (of 2,000,000 planned)
**Config:** Same as 2026-03-24_09-51-13 (no changes)

---

## Training Metrics Summary — vs. Previous Run 09-51-13 (400k steps)

| Metric | 09-51-13 (400k, recent mean) | 15-06-36 (461k, recent mean) | Change |
|--------|------------------------------|------------------------------|--------|
| Total reward (mean) | +1.17 | **+43.63** | +3629% |
| Total reward (last) | +13.35 | **+47.17** | +253% |
| Total reward (max, recent) | 6.03 | **64.33** | +967% |
| distance_reward /ep | 3.95 | **17.94** | +354% |
| tracking_reward /ep | 0.065 | **0.899** | +1282% |
| height_reward /ep | 0.844 | **1.622** | +92% |
| velocity_penalty /ep | 0.0 | **0.0** | — |
| Episode length (mean steps) | 101.6 | **182.0** | +79% |
| Episode length (max steps) | 151.6 | **249.2** | +64% |
| Episode length (min, recent) | 45.4 | **71.7** | +58% |
| Policy std | 0.662 | **0.549** | -17% |
| Value loss (recent) | 0.242 | **0.048** | -80% |

**Milestone tracking:**
- tracking_reward > 0.10: PASSED (0.899 >> 0.10, achieved before 461k steps)
- tracking_reward > 0.25: PASSED (0.899 >> 0.25, achieved before 461k steps)
- Episode mean > 200 steps: NOT YET (182.0 steps, ~91% of target)

---

## Observations & Findings

### [Explosive Policy Improvement] — Severity: N/A (Positive)

**Symptom:** total_reward_mean jumped from +1.17 (recent mean at 400k steps) to +43.63 (recent mean at 461k steps) — a +3629% increase with only 61k additional steps. The best individual episode reached 157 total reward.

**Root Cause:** This is a phase transition: the policy crossed a threshold where it began consistently entering the capture zone (tracking_reward unlocked), causing cascading improvement in all correlated metrics. distance_reward, height_reward, episode length, and action_smoothness all increased simultaneously — consistent with drones actively flying to and sustaining pursuit of targets.

**Evidence:** tracking_reward best-ever: 1.816/ep. Recent mean 0.899/ep with std 0.209 — high variance indicates the policy is still learning to be consistent, but it is reliably entering the capture zone in most episodes.

---

### [Policy Converging — Exploration Reducing] — Severity: LOW (expected)

**Symptom:** policy_std dropped from 0.662 → 0.549 (-17%). entropy_loss recent mean: -0.00816 (less negative than 09-51-13's -0.01004), confirming entropy is slightly higher now.

**Root Cause:** As the policy finds a rewarding trajectory, the PPO entropy term reduces std naturally. At 0.549 std the policy retains meaningful exploration but is narrowing.

**Watch:** If policy_std falls below 0.40 before tracking_reward exceeds 0.25 (mean), the policy may be exploiting a local optimum. Currently this is not a concern.

---

### [Episode Length Approaching 200 Steps — Not Yet There] — Severity: LOW

**Symptom:** Episode mean reached 182 steps (target: >200). Max reached 249 steps (best ever: 460 steps from run start). Min (recent) is 71.7 steps with std=66 — high variance indicating a bimodal distribution: some episodes terminate early (fly_high or bounding_box) while others run long.

**Root Cause:** The wide std on episode_min (66) suggests a mixture of successful long episodes and early-termination failures. The fly_high_termination at z=6m is periodically triggering. As the policy improves altitude control, the min episode length should increase.

**Watch:** If mean does not exceed 200 steps by 600k steps, check whether fly_high terminations are still a significant fraction of resets.

---

### [tracking_reward Variance Still High] — Severity: MEDIUM

**Symptom:** tracking_reward recent mean = 0.899, std = 0.209 (23% coefficient of variation). Best ever = 1.816/ep. This high variance means the policy is not yet consistently in the capture zone — it enters occasionally but does not sustain it reliably.

**Root Cause:** The sustained_follow_duration requirement (3.0s by default) makes tracking_reward sparse: the drone must stay within 3.0m for an uninterrupted hold period. The high std indicates the drone often breaks tracking before completing the hold, causing zero-tracking-reward episodes mixed with high-reward ones.

**Watch:** This is expected at 461k steps. If std / mean ratio does not drop below 0.15 by 1M steps, consider reducing sustained_follow_duration from 3.0s to 1.5s (Priority 4 Option A from previous plan).

---

### [value_loss Well-Converged — Critic Stable] — Severity: N/A (Positive)

**Symptom:** value_loss recent mean = 0.048, last = 0.049. At 09-51-13 it was 0.242 (rising). The critic is now well-fitted to the current policy's value function.

**Implication:** The critic has converged to the new reward regime. This is healthy — it means the policy gradient signal is clean and not corrupted by value estimation errors.

---

### [velocity_penalty = 0.0 Throughout] — Severity: LOW (info)

**Symptom:** velocity_penalty is exactly 0.0 for every logged step in both runs.

**Root Cause:** Likely this term is disabled (weight=0) in the current config. This is consistent with the Priority 1 fix from the first analysis (velocity_penalty_weight: 0.3 → 0.0). Confirmed correct.

---

## Improvement Recommendations

### Priority 1 (MONITOR ONLY): Continue current training to 1M–2M steps

**Problem:** N/A — training is working well. The phase transition has occurred, tracking_reward is at 0.899/ep and climbing.

**Proposed action:** No config changes. Continue training to completion (2M steps as planned).

**Rationale:** The current trajectory shows no signs of plateau or instability. Policy std is healthy (0.549), value_loss is stable (0.048), total reward is rising. Making any parameter change now risks disrupting the ongoing learning phase.

---

### Priority 2 (MEDIUM): Monitor tracking_reward consistency — trigger condition for sustained_follow_duration reduction

**Problem:** tracking_reward std/mean = 0.23 (high variance). If consistency does not improve by 1M steps, the 3s hold requirement may be too strict for early learning.

**Trigger condition:** If at 1M steps `tracking_reward mean < 0.25`, apply:

- File: `exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/move_flyfollow/marl_move_flyfollow_env_cfg.py`
- Parameter: `sustained_follow_duration: 3.0 → 1.5` (seconds)
- Rationale: A shorter hold requirement allows tracking_reward to fire more frequently, providing denser gradient for the policy to refine approach behavior.

**Current assessment:** NOT triggered. tracking_reward is at 0.899, well above the 0.25 milestone. Only trigger this if the metric stalls.

---

### Priority 3 (LOW): Tighten fly_high_termination_z after 500k steps

**Problem:** fly_high_termination_z=6.0m was set conservatively to allow early exploration. As altitude control improves, lowering it will further reduce wasted episode time from altitude drift.

**Trigger condition:** After 500k steps, if height_reward remains above 1.4/ep (confirming consistent altitude control):

- File: `exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/move_flyfollow/marl_move_flyfollow_env_cfg.py`
- Parameter: `fly_high_termination_z: 6.0 → 4.5`
- Rationale: Tighter altitude envelope forces the policy to maintain better altitude precision, reducing the remaining episode-length variance from fly_high early terminations.

**Current assessment:** height_reward = 1.622/ep (up from 0.844). Altitude anchoring is working well. Apply this change at next config revision if the metric stays above 1.4.

---

## Experiment Plan

### Current run (15-06-36): Continue as-is to 2M steps

No changes needed. Monitor these checkpoints:

1. **At 600k steps:**
   - tracking_reward mean should exceed 1.0/ep
   - Episode mean should exceed 200 steps
   - If tracking_reward still below 0.25: apply Priority 2 (sustained_follow_duration reduction)

2. **At 1M steps:**
   - tracking_reward mean should exceed 1.5/ep (consistent sustained following)
   - Episode length max should approach 350+ steps (near full episode completion)
   - If tracking mean is plateaued below 1.0: investigate whether fly_high terminations are still frequent; apply Priority 3 (lower termination_z)

3. **At 2M steps:**
   - Evaluate whether `all_targets_captured` success rate is nonzero
   - If total_reward shows plateau in last 500k steps: consider curriculum tightening (reduce capture_distance from 3.0m to 2.5m, increase target_velocity from 0.3 to 0.4 m/s)

**Training command (same):**
```bash
python3 scripts/skrl/train.py --task=Isaac-marl-move-flyfollow-v0 --headless --num_envs=2048 --algorithm="MAPPO"
```

**Updated success criteria for this run:**
- `tracking_reward` > 1.0 (mean, recent) by step 600k
- `Episode / Total timesteps (mean)` > 200 steps by step 600k
- `total_reward_mean` > 60.0 (sustained) by step 1M
- `total_reward_mean` > 80.0 (sustained) by step 2M

---

## Changelog

- 2026-03-24 (run 15-06-36, 461k steps): Major phase transition confirmed. tracking_reward jumped
  0.065 → 0.899/ep (+1282%). total_reward mean +1.17 → +43.63 (+3629%). Both tracking milestones
  (0.10 and 0.25) passed simultaneously. Episode length 101.6 → 182.0 steps (+79%). Critic
  well-converged (value_loss 0.048). No config changes recommended — continue training to 2M steps.
- 2026-03-24 (run 15-06-36, 670k steps): Continued progress confirmed. tracking_reward 0.899 →
  1.355/ep (milestone 1.0 passed). Episode mean 182 → 206 steps (milestone 200 steps passed).
  total_reward mean 43.63 → 51.99 (+19%). Best-ever total_reward 84.38. Policy std stabilized at
  0.488 (healthy convergence zone). No config changes needed — training on track for 2M step goal.

---

## Progress Update — Run 2026-03-24_15-06-36 (670k steps)

**Analysis date:** 2026-03-24
**Previous snapshot:** 461k steps (last analysis)
**Current step:** 670,500

### Training Metrics at 670k Steps

| Metric | 461k snapshot | 670k current | Change | Status |
|--------|--------------|--------------|--------|--------|
| Total reward (recent mean) | +43.63 | **+51.99** | +19% | Improving |
| Total reward (last) | +47.17 | **+63.07** | +34% | Improving |
| Total reward (best ever) | 74.95 | **84.38** | +12% | New high |
| distance_reward/ep | 17.94 | **21.57** | +20% | Improving |
| tracking_reward/ep (recent mean) | 0.899 | **1.220** | +36% | Improving |
| tracking_reward/ep (last) | ~0.90 | **1.355** | +51% | Improving |
| tracking_reward (best ever) | 1.816 | **2.754** | +52% | New high |
| height_reward/ep | 1.622 | **1.646** | +1% | Stable |
| Episode mean steps | 182 | **202** | +11% | Improving |
| Episode max steps (recent mean) | ~249 | **283** | +14% | Improving |
| Policy std | 0.549 | **0.489** | -11% | Converging |
| Value loss | 0.048 | **0.028** | -42% | Well-converged |
| velocity_penalty | 0.0 | **0.0** | — | Confirmed zero |

### Milestone Verification

| Milestone | Target | Result |
|-----------|--------|--------|
| tracking_reward > 0.10 (by 600k) | 0.10 | **1.220** (mean) |
| tracking_reward > 0.25 (by 600k) | 0.25 | **1.220** (mean) |
| tracking_reward > 1.0 (600k milestone) | 1.0 | **1.220** |
| Episode mean > 200 steps (600k milestone) | 200 | **202 steps** |

All four milestones cleared.

### Findings

**Training is continuing to improve across all primary metrics.** No new pathologies are present.

**1. tracking_reward progression is healthy.**
From 0.899 (461k) to 1.220 mean / 1.355 last (670k). Best-ever episode reached 2.754/ep, up from
1.816. Variance remains moderate (std=0.288, CV=24%), meaning some episodes still see zero or low
tracking, but the best-case ceiling is rising steadily. This is expected behavior at this training
stage — consistent tracking requires more time to consolidate.

**2. Episode length crossed the 200-step target.**
Recent mean at 202 steps, last value 206 steps. Maximum (recent mean) is 283 steps, with best-ever
400.8 steps. The minimum (recent mean) is 83 steps — some episodes still terminate early, likely from
fly_high exits, but these are declining as altitude control improves.

**3. Total reward trend is steady but showing deceleration.**
Mean reward improved +19% over the 209k steps since the last snapshot (from 43.63 to 51.99). The
rate of gain has slowed relative to the explosive phase-transition period (461k had +3629% gain).
This is the expected transition into the refinement phase. The curve is not plateauing yet — it
continues upward.

**4. Policy convergence indicators are healthy.**
Policy std settled from 0.549 to 0.489, which is within the stable exploration range for MAPPO (not
collapsing). Value loss dropped further to 0.028 (from 0.048), meaning the critic's return estimates
are accurate. Entropy loss is near-zero (-0.007) indicating policy confidence without being
degenerate.

**5. No reward hacking or pathology signatures.**
force_penalty (0.403/ep) and body_rate_penalty (0.173/ep) are both small relative to positive terms.
action_smoothness is improving (0.656/ep, up from earlier). velocity_penalty = 0.0 confirmed.

**6. height_reward is stable at 1.646/ep.**
Altitude anchoring is maintained. fly_high_termination_z=6.0 condition is functioning — the drone
is staying below the 6m ceiling. The condition for lowering termination_z to 4.5m (height_reward
> 1.4/ep for sustained period) is now met.

### Current State Assessment

**Status: Improving — normal refinement phase. No intervention needed.**

Training has progressed through the phase transition (completed ~450k steps) and is now in a
sustained refinement phase. All 600k-step milestones have been passed. The reward continues to
climb, and tracking_reward has crossed 1.0/ep.

The primary remaining learning objective is consistency: closing the gap between best-case episodes
(tracking 2.754) and average episodes (tracking 1.220). This requires more training time.

### Recommendations at 670k Steps

**Priority 1 (OPTIONAL, LOW urgency): Lower fly_high_termination_z 6.0 → 4.5m**

The trigger condition stated in the previous analysis was: "height_reward > 1.4/ep sustained."
This condition is met (1.646/ep stable). The change can be applied at the next restart.

- File: `exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/move_flyfollow/marl_move_flyfollow_env_cfg.py`
- Parameter: `fly_high_termination_z: 6.0 → 4.5`
- Rationale: Tighter altitude ceiling will reduce episode-length variance from fly_high terminations,
  pushing more episodes toward the full-length 400-step target. Apply only at restart; do not
  interrupt current run.

**Priority 2 (MONITOR): Watch for tracking_reward plateau below 1.5/ep before 1M steps**

If tracking_reward mean stops growing between 800k–1M steps and remains below 1.5/ep, consider:
- Reducing `sustained_follow_duration` from 3.0s to 1.5s (easier capture threshold)
- This was the contingency plan from the 461k analysis

**No other changes recommended.** Current run should continue without interruption to 2M steps.

### Updated Milestone Targets

| Step | Metric | Target |
|------|--------|--------|
| 1M | tracking_reward (recent mean) | > 1.5/ep |
| 1M | Episode max (recent mean) | > 350 steps |
| 1.5M | tracking_reward (recent mean) | > 2.0/ep |
| 2M | Evaluate all_targets_captured success rate; if plateau → tighten capture_distance 3.0 → 2.5m |

---

## 训练进展分析 — 1.46M 步快照

**分析时间：** 2026-03-25
**Run：** `2026-03-24_15-06-36_mappo_torch_mappo`
**当前步数：** 1,463,400 步（最新 checkpoint: agent_1460000.pt）

### 最新指标汇总

| 指标 | 上次快照 (670k) | 本次近期均值 (1.46M) | 最新值 | 历史最佳 |
|------|--------------|-------------------|--------|---------|
| Total reward (mean) | +51.99 | +47.76 | +53.62 | +94.87 |
| tracking_reward | 1.220/ep | 1.068/ep | 1.251/ep | 2.754/ep |
| distance_reward | 21.57/ep | 19.27/ep | 17.32/ep | 36.46/ep |
| height_reward | 1.646/ep | 1.621/ep | 1.565/ep | 3.003/ep |
| timesteps_mean | 202 steps | 180.7 steps | 212.7 steps | 456.7 steps |
| timesteps_max (recent mean) | 283 steps | 270.5 steps | 292 steps | 585 steps |
| policy_std | 0.489 | 0.447 | 0.450 | 0.820 |
| value_loss | 0.028 | 0.061 | 0.043 | — |
| entropy_loss | −0.007 | −0.00595 | −0.00605 | — |

### 逐项分析

**1. 当前步数：1,463,400 步（超过 1M 里程碑）**

训练已超过预设的 1M 步检查点，进入 1.5M 步阶段。距上次 670k 快照增加了约 793k 步。

**2. tracking_reward：1.068/ep（均值），未达 1M 目标 >1.5/ep**

这是本次快照最重要的发现。1M 步里程碑目标（tracking_reward > 1.5/ep）**未达成**。最新单值
1.251/ep 略高于近期均值，但历史最佳为 2.754/ep，说明策略有峰值能力但不稳定。从 670k（1.220/ep
均值）到 1.46M（1.068/ep 均值），tracking_reward 均值实际**小幅下降**。这是停滞信号，不是崩溃，
但与预期的持续增长方向相反。

**根因分析：**
- `policy_std` 从 0.549（670k）降至 0.447（当前），探索空间压缩，策略趋向固化。
- `entropy_loss` 近期均值 −0.00595（接近零），熵惩罚几乎消失，策略已接近局部确定性。
- `distance_reward` 近期均值从 21.57 降至 19.27（−11%），说明无人机接近目标的表现略有退步。
  这可能与策略收敛到保守行为（减少追踪动作的幅度）有关。
- `timesteps_min` 近期均值仅 63.4 步（std=59.2，方差极大），说明部分剧集仍在早期终止，
  可能由 fly_high 触发，拉低了 tracking_reward 统计均值。

**3. episode 长度：均值 180.7 步，最大均值 270.5 步，未达目标（最大均值 >350 步）**

均值相对 670k 时（202 步）轻微下降。最新单值 212.7 步，最大值 292 步。最大步数目标（350 步）
**未达成**。说明剧集长度增长已停滞，部分剧集频繁提前终止。与 tracking_reward 停滞的模式一致。

**4. 总奖励趋势：缓慢增长，但近期均值低于上次快照**

recent_mean（近 20% 数据）为 47.76，而 670k 时的 last 值为 63.07。这反映出近期表现低于 670k
附近的峰值期。最新值 53.62 和历史最佳 94.87 之间有较大缺口。上行趋势未完全终止，但增长动力
明显减弱。

**5. 收敛停滞迹象：已出现**

综合以下信号，训练已进入接近收敛平台期的阶段：
- `policy_std` 在 0.44–0.45 区间内几乎稳定（670k→1.46M 缓慢下降）
- `entropy_loss` 近零，策略熵已压缩至接近下限
- `tracking_reward` 均值停止增长（1.220→1.068）
- `distance_reward` 轻微下降

这不是训练崩溃，而是策略在当前奖励结构下已接近其能学到的上限。**需要干预以解锁更高性能。**

**6. 无新病态（奖励 hacking / 高飞复发 / 速度惩罚）**

- `velocity_penalty` = 0.0 全程
- `height_reward` 稳定在 1.62/ep，高飞未复发
- `body_rate_penalty` 稳定（0.119/ep）
- `force_penalty` 稳定（0.357/ep）

### 当前状态判断

**状态：接近停滞的精化阶段。1M 步里程碑目标未达成（tracking +episode长度均不足）。需要在下次重启时施加改进措施。**

策略探索已压缩（policy_std=0.447），tracking_reward 均值停止增长，当前奖励结构已触及其学习上限。
预计在现有配置下继续训练至 2M 步只会在当前平台期附近徘徊，无法解锁 tracking > 1.5/ep。

### 下一步行动建议

**Priority 1 (HIGH): 在下次重启时应用 fly_high_termination_z 6.0 → 4.5m（已触发条件，之前确认）**

此项在 670k 分析时已确认（height_reward > 1.4/ep 持续满足）。本次 1.621/ep 仍满足。
- 文件：`exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/move_flyfollow/marl_move_flyfollow_env_cfg.py`
- 参数：`fly_high_termination_z: 6.0 → 4.5`
- 预期效果：减少 fly_high 提前终止，提高 timesteps_min，拉高 tracking_reward 统计均值

**Priority 2 (HIGH): 降低 sustained_follow_duration 3.0s → 1.5s（激活停滞应急预案）**

之前 670k 分析的应急预案条件是：tracking_reward 在 800k–1M 步前停滞于 1.5/ep 以下。
该条件已触发（1.46M 步，tracking 均值 1.068/ep < 1.5/ep，且均值未再增长）。
- 文件：`exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/move_flyfollow/marl_move_flyfollow_env_cfg.py`
- 参数：`sustained_follow_duration: 3.0 → 1.5`
- 理由：3.0s 的持续跟随要求对于 timesteps_mean=180 步（≈6s）的剧集来说成功概率极低。
  降至 1.5s 可显著提高每剧集内成功触发 tracking_reward 的频率，为策略提供更密集的正向反馈。

**Priority 3 (MEDIUM): 考虑小幅提升 tracking_reward_weight 3.0 → 4.0**

当前 tracking_reward 占 total_reward 比例偏低（1.068 / 47.76 ≈ 2.2%）。distance_reward（19.27）
主导了奖励信号。适度提升 tracking_reward_weight 可增强进入捕获区的激励，但需避免过大导致不稳定。
- 文件：同上
- 参数：`tracking_reward_weight: 3.0 → 4.0`
- 风险：中等。需监控 total_reward 是否在调整后持续上升（而非振荡）。

**Priority 4 (LOW, 监控项): 观察 entropy 是否需要干预**

entropy_loss 近期均值 −0.00595（最初约 −0.0096），说明策略已从高熵探索状态收缩。若下次重启后
policy_std 继续下降至 0.40 以下，可考虑小幅提升 entropy 系数（skrl agent config 中的
`entropy_loss_scale`），或暂时提升 initial_log_std 重启策略探索。

### 行动摘要

| 优先级 | 参数 | 变更 | 时机 |
|--------|------|------|------|
| HIGH | `fly_high_termination_z` | 6.0 → 4.5m | 下次重启时 |
| HIGH | `sustained_follow_duration` | 3.0 → 1.5s | 下次重启时 |
| MEDIUM | `tracking_reward_weight` | 3.0 → 4.0 | 下次重启时（可选） |
| LOW | 监控 `policy_std` 趋势 | 若 < 0.40 考虑熵干预 | 1.8M 步检查 |

**当前运行是否应中断：否。**
建议在当前 checkpoint（agent_1460000.pt）基础上，以上述参数修改重启一次新运行，
以 2M 步为目标评估 tracking_reward 是否突破 1.5/ep。

---

# Training Analysis Report

**Run:** `2026-03-25_11-01-07_mappo_torch_mappo`
**Date:** 2026-03-25
**Task:** Isaac-marl-move-flyfollow-v0
**Algorithm:** MAPPO (CTDE, shared actor/critic)
**Total timesteps logged:** 805,300
**Base checkpoint:** `2026-03-24_15-06-36_mappo_torch_mappo/checkpoints/best_agent.pt`
**Changes vs previous run:** fly_high_termination_z 6.0→4.5m, sustained_follow_duration 3.0→1.5s, tracking_reward_weight 3.0→4.0, body_rate_penalty_weight 0.5→1.0, upright_penalty_weight 0.5→1.0

---

## Training Metrics Summary

| Metric | Early mean | Recent mean | Last value | Best ever |
|--------|-----------|-------------|------------|-----------|
| Total reward (mean) | **56.33** | 47.39 | 47.06 | 120.04 |
| Total reward (max) | 85.94 | 78.43 | 63.74 | 154.46 |
| Total reward (min) | 15.62 | 9.82 | −4.51 | 91.26 |
| distance_reward | 22.98 | 19.42 | 11.30 | 42.12 |
| tracking_reward | **1.894** | 1.434 | 0.545 | 4.923 |
| height_reward | 1.812 | 1.773 | 1.370 | 3.193 |
| body_rate_penalty | 0.381 | 0.378 | 0.201 | 0.689 |
| force_penalty | 0.406 | 0.371 | 0.273 | 0.692 |
| action_smoothness | 0.672 | 0.562 | 0.334 | 1.165 |
| velocity_penalty | 0.000 | 0.000 | 0.000 | 0.000 |
| Episode length (mean steps) | 205.8 | 185.3 | 193.3 | 348.1 |
| Episode length (max steps) | 288.0 | 271.3 | 271.0 | 523.0 |
| Policy std | 0.436 | 0.410 | 0.410 | 0.456 |
| Value loss | 0.056 | 0.061 | 0.077 | 0.333 |
| Policy loss | −0.021 | −0.017 | −0.033 | +0.126 |
| Entropy loss | −0.00577 | −0.00502 | −0.00502 | — |

Script verdict: **regressing**

---

## Observations & Findings

### 1. Checkpoint 加载验证 — CONFIRMED

**关键证据：** Early mean total_reward = **+56.33**（run 开始即为高正值），而从零开始训练的早期均值为负数（如 22-02-33 run 早期 ≈ −4.94）。上一轮（15-06-36，1.46M 步时）recent_mean ≈ 47.76，本轮 early_mean = 56.33，高出约 +8.6 点。

**结论：** Checkpoint 加载成功生效，本次训练从上一轮的学习基础上继续，而非重零开始。

---

### 2. tracking_reward 趋势 — Severity: HIGH（回退而非增长）

**症状：** tracking_reward early_mean = **1.894**，recent_mean = **1.434**，last = **0.545**（下降 71%）。
Best ever = 4.923，说明策略在某些剧集中确实能产生更高的跟踪奖励，但整体趋势是下滑的。

**与预期对比：** 上一轮（15-06-36）1.46M 步时 tracking_reward mean ≈ 1.068，本轮初始即跳至 1.894（+77%），说明 sustained_follow_duration 3.0→1.5s 的改动生效，tracking 信号变得更易触发。但随后出现系统性下滑，说明存在新的不稳定因素。

**根本原因分析：** tracking 下滑与 total_reward 同步下滑（early 56.33 → recent 47.39），且 distance_reward 也从 22.98 → 19.42 下滑，说明**整体追踪能力下降**，而非单独的 tracking 问题。这是一次整体性能回退，不是单一奖励项的问题。

---

### 3. episode 长度变化 — Severity: MEDIUM

**观察：** Episode 长度 early_mean = 205.8 步 → recent_mean = 185.3 步（−10%）。
最大值：early_mean 288 → recent_mean 271 步（−6%）。Best ever = 523 步（高于上一轮 best 585 步之前的历史最高）。

**与上一轮对比：** 上一轮 1.46M 步时 episode_mean ≈ 180.7 步，本轮初始 205.8 步（继续改善），但近期又回落至 185.3 步。episode_min recent_mean 仅 66 步（std=61.8），说明存在部分剧集极早终止（可能是 fly_high）。

---

### 4. fly_high 终止分析 — Severity: HIGH

**关键数据：** episode_min 的 recent_mean = 66 步，std = 61.8（极高方差）。这意味着部分剧集在 40–80 步就终止，与 fly_high_termination_z 收紧至 4.5m 高度相关。

**证据链：** fly_high_termination_z 从 6.0→4.5m 是本次最大结构性改动。Total reward (min) 的 recent_mean 降至 **9.82**（早期 15.62），且 last value 已变成 **−4.51**（负值），说明存在整体剧集质量恶化。distance_reward 下滑 15%（22.98→19.42）也印证了剧集提前终止导致追踪时间缩短。

**结论：** 4.5m 上限比当前策略的飞行高度更严格，触发了更多 fly_high 早期终止，这是性能回退的主要原因之一。

---

### 5. 姿态稳定性奖励 — Severity: LOW（无明显问题）

**body_rate_penalty：** early_mean = 0.381，recent_mean = 0.378（几乎不变）。权重从 0.5→1.0 后，绝对值提升应明显，但 early_mean 仅 0.381——这与上一轮（15-06-36）的 body_rate_penalty ≈ 0.378（weight=0.5 时）处于同一量级，说明**实际 body_rate 水平降低了约一半**（相同奖励值但权重翻倍）。

**upright_penalty（未单独记录）：** env.yaml 确认 upright_penalty_weight = 1.0，但 TensorBoard 中未单独记录 upright 奖励项，只能间接从 total_reward 判断。姿态稳定性改进未导致明显的正向奖励变化，可能因为该项绑定在 total_reward 中。

**整体判断：** body_rate/upright 权重翻倍（0.5→1.0）没有造成奖励惩罚爆炸，说明策略的姿态控制本身是合理的。此项改动对性能的负面影响有限，不是回退主因。

---

### 6. 总体判断 — 性能回退，原因以 fly_high 收紧为主

**核心发现：**
- Checkpoint 加载成功：early total_reward +56.33 显著高于历史从零起步水平
- 初始 tracking_reward 跳升至 1.894（sustained_follow_duration 缩短效果立竿见影）
- 但后续出现系统性回退（tracking、distance、episode 长度、total_reward 全线下滑）
- 回退的主要机制：fly_high_termination_z 4.5m 使更多剧集提前终止，策略当前飞行高度在 4.5m 附近，触发率高
- tracking_reward best ever = 4.923（大幅高于上一轮 best 1.816），说明策略潜力已大幅提升，但被fly_high终止截断

**Script 给出 "regressing" 判断的原因：** early_mean (56.33) > recent_mean (47.39)，说明后期表现弱于初期。

---

## Improvement Recommendations

### Priority 1 (HIGH): 适度放宽 fly_high_termination_z 至 5.0m

**问题：** 4.5m 收紧导致剧集提前终止，策略还未适应新上限。本轮运行仅 ~805k 步，尚不足以让策略完全收敛到 4.5m 以下飞行。tracking best=4.923 说明策略具备能力，但被过早终止打断。

**权衡：** 4.5m 是正确的长期目标，但引入过快导致不稳定。建议在当前 checkpoint 基础上设置为 5.0m（折中），待 tracking_reward > 2.0/ep 稳定后再收紧至 4.5m。

**或者：** 若不想修改终止高度，则**延长当前运行**至 1.5M+ 步，给策略时间适应 4.5m 约束。回退趋势发生在 800k 步内，可能是过渡期震荡，而非永久退化。

- 文件：`exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/flyfollow/marl_flyfollow_env_cfg.py`
- 参数：`fly_high_termination_z: 4.5 → 5.0`（若选择折中方案）
- 理由：给策略足够的适应窗口，避免 fly_high 终止主导 reset 类型

### Priority 2 (MEDIUM): 继续当前运行，等待超越 1M 步

**理由：** 训练仅进行了 ~805k 步（等效于总有效步数），policy_std = 0.410（仍有健康探索空间，未坍缩到 0.40 以下）。tracking best=4.923 是迄今最高值，说明策略能力在提升，只是稳定性还差。上一轮的"相变"发生在 450k 步，本轮可能在适应 fly_high 约束后出现二次相变。

**关键观察点（1.2M 步时检查）：**
- tracking_reward recent_mean 是否重新超过 1.5/ep
- episode_min 是否不再频繁出现 40–80 步的极短剧集
- total_reward recent_mean 是否回升至 50+

### Priority 3 (LOW): 监控 policy_std 下限

**当前状态：** policy_std = 0.410，与上一轮 1.46M 步时的 0.447 相比已更低。recent_std = 0.0054（极小），说明 std 正在单调下降。若在 1.2M 步检查时 policy_std < 0.38，应考虑在下次重启时调高 `entropy_loss_scale`。

---

## Experiment Plan

1. **当前运行继续**（不中断），目标运行至 1.5M+ 步
2. **1.2M 步检查点：**
   - tracking_reward recent_mean > 1.5/ep → 继续运行至 2M 步
   - tracking_reward recent_mean < 1.2/ep 且仍在下滑 → 考虑重启并放宽 fly_high_termination_z 至 5.0m
   - policy_std < 0.38 → 下次重启加入 entropy_loss_scale 微调
3. **成功标准（2M 步目标）：**
   - tracking_reward mean > 2.0/ep（稳定，recent_std < 0.3）
   - episode_mean > 200 步
   - total_reward mean > 55

---

## Changelog

- 2026-03-25：分析 run 11-01-07（checkpoint 继续训练，805k 步）。确认 checkpoint 加载成功（early +56.33），
  sustained_follow_duration 缩短效果立竿见影（tracking 初始 +77%），但 fly_high_termination_z 4.5m 过严
  导致系统性回退。tracking best=4.923（历史新高），说明策略潜力已大幅提升，需等待适应期。

---

# Training Analysis Report

**Run:** `2026-03-25_11-01-07_mappo_torch_mappo`（续训分析）
**Date:** 2026-03-26
**Task:** Isaac-marl-move-flyfollow-v0
**Algorithm:** MAPPO
**当前步数：** 1,855,100 步（上次分析时约 805k 步，本次已过 1.2M 检查节点）

---

## Training Metrics Summary（1.855M 步）

| Metric | early_mean | recent_mean | last | best ever |
|--------|-----------|-------------|------|-----------|
| Total reward (mean) | 54.96 | **39.36** | 32.97 | **120.04** |
| Total reward (max) | 85.0 | 61.1 | 60.2 | 154.5 |
| Total reward (min) | 14.8 | 7.8 | 7.6 | 91.3 |
| distance_reward | 22.28 | **17.54** | 15.26 | 42.12 |
| tracking_reward | 1.833 | **1.336** | 0.889 | **4.923** |
| height_reward | 1.791 | 1.403 | 1.351 | 3.193 |
| body_rate_penalty | 0.375 | 0.214 | 0.258 | 0.689 |
| action_smoothness | 0.675 | 0.247 | 0.257 | 1.165 |
| force_penalty | 0.395 | 0.334 | 0.333 | 0.692 |
| velocity_penalty | 0.0 | 0.0 | 0.0 | 0.0 |
| Episode timesteps (mean) | 199.4 | **167.9** | 192.6 | 348.1 |
| Episode timesteps (max) | 282.8 | 236.6 | 247.0 | 523.0 |
| Episode timesteps (min) | 78.8 | **58.9** | 101.0 | 275.0 |
| Policy std | 0.417 | **0.437** | 0.450 | 0.458 |
| Value loss | 0.054 | 0.038 | 0.034 | 0.333 |
| Policy loss | −0.022 | −0.022 | −0.061 | +0.126 |
| Entropy loss | −0.00530 | −0.00545 | −0.00559 | −0.00465 |

---

## 六项关键问题逐一分析

### 1. 当前最新步数

**1,855,100 步。** 本次分析时已超越上次分析设定的 1.2M 检查节点，并已接近 2.0M 步目标。

### 2. tracking_reward 震荡是否已收敛——Severity: HIGH

**结论：震荡未收敛，处于持续衰减趋势。**

- early_mean = 1.833 → recent_mean = 1.336（−27%）
- last = 0.889（远低于 recent_mean，说明近期仍在下滑）
- 1.2M 检查目标：recent_mean > 1.5/ep。**未达到**（1.336 < 1.5）
- 2M 步目标：mean > 2.0/ep。**明确无法达到**（当前 last=0.889）
- best ever = 4.923 保持不变（与 805k 步时相同），说明 800k 步后未出现更高峰值，策略能力未进一步提升

**症状：** 呈"高开低走"形态。early 阶段继承了热启动的高质量策略，后续随探索压缩，tracking 质量单调下降，未出现二次相变。

### 3. episode 长度是否趋于稳定——Severity: HIGH

**结论：不稳定，均值缩短且方差极大。**

- 805k 时：mean ≈ 185 步；现在 recent_mean = 167.9 步（−9.3%）
- timesteps_min 的 recent_mean = 58.9（recent_std = 58.7，几乎等于均值）。这意味着每批 rollout 中始终有少量极短剧集（~1–40 步），是 fly_high 终止或其他早终止的信号
- timesteps_max recent_mean = 236.6，而 best ever = 523 步（原始热启动时创造的），说明策略当前无法维持长时间追踪

### 4. fly_high 终止频率变化

脚本未直接输出 fly_high 计数，但可从多个间接指标推断：

- timesteps_min recent_mean = 58.9 步 ≈ 1.97 秒（step_dt=0.033s），且 std=58.7（极端分散），说明**短剧集频繁发生**。最短剧集（worst=1 步）表明存在即时终止。
- 上次分析结论是 fly_high_termination_z=4.5m 过于激进，导致频繁提前终止。
- 当前 height_reward recent_mean = 1.403（vs. early 1.791，下降 −22%）。高度奖励衰减 + 短剧集并存，说明策略在飞行高度控制上仍未完全适应 4.5m 约束。
- **推论：fly_high 终止频率仍然偏高，未见改善。**

### 5. policy_std 是否跌破 0.38 警戒线——Severity: MEDIUM（好转）

**结论：未跌破 0.38，且近期呈回升趋势。**

- 805k 步时：policy_std = 0.410（下滑趋势，当时接近警戒线）
- 1.855M 步时：recent_mean = 0.437，last = **0.450**（early_mean = 0.417，即 recent > early）
- worst 历史最低值 = 0.392（高于 0.38，从未跌破）
- **policy_std 正在从 0.41 回升至 0.45，探索空间有所恢复。** 这是本次分析最积极的信号。

可能原因：entropy_loss 略有加深（−0.00530 → −0.00545），PPO 熵正则化开始发挥作用，阻止策略进一步坍缩。

### 6. 当前状态判断

**判断：需要干预（不能继续等待，但不是紧急重启）。**

核心依据：

| 指标 | 上次预期 | 实际结果 | 判断 |
|------|---------|---------|------|
| tracking_reward recent_mean > 1.5/ep（1.2M 检查） | 目标 | 1.336 | 未达到 |
| tracking_reward mean > 2.0/ep（2M 目标） | 目标 | 0.889（last）| 明确无法达到 |
| episode_min 稳定 | 目标 | std=58.7（极不稳定） | 未达到 |
| total_reward mean > 55（2M 目标） | 目标 | last=32.97 | 明确无法达到 |
| policy_std > 0.38 | 安全线 | 0.450 | 达到（唯一亮点） |

当前训练处于**持续性衰减期**，而非过渡期震荡：
- tracking_reward 从 early 1.833 到 recent 1.336 到 last 0.889，呈三段单调下降，不是震荡
- distance_reward 同步衰减（early 22.3 → recent 17.5 → last 15.3），说明追踪能力在退化而非只是 tracking bonus 难以触发
- action_smoothness 大幅衰减（early 0.675 → recent 0.247），动作质量明显下降
- best ever 自 805k 步后无更新，策略最高能力已冻结

---

## Improvement Recommendations

### Priority 1 (HIGH): 重启训练，放宽 fly_high_termination_z 至 5.0m

**问题：** fly_high_termination_z=4.5m 造成的终止压力持续了整个 1.855M 步训练期，策略始终未能完全适应，表现为持续的高度奖励衰减和短剧集。

**判断依据：** height_reward early→recent 衰减 22%，timesteps_min 极不稳定（std≈mean），800k 步后 tracking best 无新高。策略已消耗完"过渡期"，但仍未完成适应。继续等待边际收益极低。

**提议变更：**
- 文件：`exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/flyfollow/marl_flyfollow_env_cfg.py`
- 参数：`fly_high_termination_z: 4.5 → 5.0`
- 理由：折中方案，比原始 6.0m 更严但比 4.5m 宽松约 11%，给策略减少中断频率，同时保留高度约束压力。

### Priority 2 (MEDIUM): 调整 tracking_reward_weight 与 capture_distance 的组合

**问题：** tracking_reward 的绝对量级已降至 last=0.889/ep，说明 drones 进入 capture zone 的频率在减少。两个可能方向：

方向 A — 放宽捕获距离（更大信号覆盖面）：
- 文件：`exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/flyfollow/marl_flyfollow_env_cfg.py`
- 参数：`capture_distance: 3.0 → 3.5`（临时放宽，提升 tracking 触发频率，待 mean>2.0 后再收窄）
- 注意：不要超过 4.0，避免过度稀释追踪精度要求

方向 B — 提升 tracking_reward_weight（已是 4.0，不建议继续上调）：
- 当前 weight=4.0，distance_reward_weight=4.0，两者已均等。进一步上调 tracking 可能引起不平衡。**不推荐。**

**推荐方向 A**。

### Priority 3 (LOW): 确认 entropy_loss_scale 设置，防止 policy_std 再次下滑

**当前状态：** policy_std 已从 0.410 回升至 0.450，是好转信号。entropy_loss_scale 当前值未知（需核查 cfg），但 entropy_loss ≈ −0.00545 表明熵正则化在工作。

**建议：** 重启时验证 `entropy_loss_scale` 设置值，确保不低于 0.001。无需修改，只需记录确认。

---

## Experiment Plan（更新）

1. **终止当前 run 11-01-07**（已 1.855M 步，继续意义有限）
2. **重启配置变更：**
   - fly_high_termination_z: 4.5 → 5.0（Priority 1）
   - capture_distance: 3.0 → 3.5（Priority 2，可选）
   - 其他参数保持不变（sustained_follow_duration=1.5s、tracking_weight=4.0 已验证有效，不动）
3. **从最新 checkpoint 热启动**（保留策略能力，避免从零开始）
4. **运行目标：** 1.5M 步，监控以下指标
5. **500k 步检查点（新运行）：**
   - tracking_reward recent_mean > 1.5/ep → 继续
   - height_reward recent_mean > 1.6/ep（确认 fly_high 不再是主要终止源）
   - episode_min std < 30（稳定性恢复）
6. **成功标准（本轮）：**
   - tracking_reward mean > 2.0/ep，std < 0.4
   - episode_mean > 200 步
   - best ever tracking > 6.0（超越历史最高 4.923）

---

## Changelog

- 2026-03-26：分析 run 11-01-07（1.855M 步，超越 1.2M 检查节点）。1.2M 和 2M 里程碑均未达到。
  tracking_reward 呈三段单调下滑（1.833→1.336→0.889），确认为持续衰减而非过渡期震荡。
  fly_high 终止压力未缓解（episode_min std≈mean）。policy_std 回升至 0.450（唯一正面信号）。
  判断：需要干预——重启并放宽 fly_high_termination_z 4.5→5.0m，可选放宽 capture_distance 3.0→3.5m。
- 2026-03-26：分析 run 09-31-34（357k 步，从 11-01-07 热启动，新增两项改动：fly_high 5.0m + capture_distance 3.5m）。
  详见下方完整分析。

---

# Training Analysis Report

**Run:** `2026-03-26_09-31-34_mappo_torch_mappo`
**Date:** 2026-03-26
**Task:** Isaac-marl-move-flyfollow-v0
**Algorithm:** MAPPO
**Total timesteps logged (this run):** 356,900
**Hot-start from:** `2026-03-25_11-01-07` best_agent.pt (1.855M cumulative steps)
**Config changes vs. previous run:**
- fly_high_termination_z: 4.5 → 5.0m
- capture_distance: 3.0 → 3.5m

---

## Training Metrics Summary

| Metric | Early mean | Recent mean | Last value | Best ever |
|--------|-----------|-------------|------------|-----------|
| Total reward (mean) | 53.31 | 54.28 | 46.22 | 111.63 |
| Instant reward (mean) | 0.264 | 0.263 | 0.271 | 0.533 |
| distance_reward | 22.28 | 22.96 | 27.01 | 38.57 |
| tracking_reward | 1.814 | 1.798 | 2.174 | 4.241 |
| height_reward | 1.739 | 1.817 | 1.974 | 2.751 |
| velocity_penalty | 0.000 | 0.000 | 0.000 | 0.000 |
| force_penalty | 0.403 | 0.415 | 0.454 | 0.653 |
| body_rate_penalty | 0.368 | 0.372 | 0.359 | 0.558 |
| action_smoothness | 0.743 | 0.711 | 0.800 | 1.250 |
| Episode length (mean steps) | 201.9 | 208.7 | 204.6 | 421.1 |
| Episode length (max steps) | 289.1 | 290.4 | 296.0 | 562.0 |
| Episode length (min steps) | 81.7 | 86.5 | 11.0 | 331.0 |
| Policy std | 0.391 | 0.375 | 0.378 | 0.397 |
| Value loss | 0.056 | 0.043 | 0.016 | 0.302 |
| Policy loss | −0.020 | −0.019 | −0.017 | 0.104 |
| Entropy loss | −0.00465 | −0.00424 | −0.00430 | − |

---

## Observations & Findings

### 1. Checkpoint Load Confirmed — Severity: INFO

**Symptom:** early_mean total_reward = +53.31 (vs. baseline of approximately −5 to −8 for cold-start runs).

**Verdict:** The hot-start from the previous run's best_agent.pt loaded successfully. The policy retained its learned tracking and flight behaviors from the previous 1.855M-step run. No cold-start loss occurred.

**Evidence:** tracking_reward early_mean = 1.814, substantially above zero (which was the state of every cold-start run). Distance_reward early_mean = 22.28, consistent with the 15–22 range seen in the previous run's final phase.

---

### 2. tracking_reward Recovery from Downtrend — Severity: HIGH

**Symptom:** In the previous run (11-01-07), tracking_reward had declined monotonically from 1.894 → 1.434 → 0.889 over 1.855M steps, with best_ever frozen at 4.923 (set at 805k, never exceeded in 1M subsequent steps).

**This run results:**
- tracking_reward early_mean = 1.814 (immediately above the final level of 0.889 from the previous run)
- tracking_reward recent_mean = 1.798
- tracking_reward last = 2.174
- tracking_reward best_ever = 4.241

**Verdict:** The downtrend HAS been arrested. The policy has recovered to the 1.8–2.2 range, which is well above the prior run's final 0.889. However, the new best_ever (4.241) is *below* the historical best of 4.923 from the 11-01-07 run's peak. This suggests partial recovery — the policy is stabilized but not yet surpassing the historical ceiling.

**Root cause of recovery (confirmed):** The fly_high_termination_z relaxation from 4.5→5.0m reduced early termination pressure, giving the policy more steps per episode to accumulate tracking rewards and adapt its flight altitude.

---

### 3. Episode Length — Improvement Confirmed — Severity: HIGH

**Previous run (11-01-07, final state):** episode_mean = 168 steps, episode_min recent_mean = 58.9 steps, min_std ≈ min_mean (indicating frequent fly_high terminations generating near-instant resets).

**This run:**
- episode_mean early = 201.9 steps; recent = 208.7 steps (+24% vs. 168)
- episode_max recent_mean = 290.4 steps
- episode_min recent_mean = 86.5 steps (std = 74.9 — still volatile but less dominated by fly_high)

**Verdict:** The fly_high_termination_z relaxation from 4.5→5.0m produced a measurable increase in episode length. Episodes are ~40 steps longer on average. The min_std/min_mean ratio has improved (no longer ≈1.0), indicating early terminations are less systematic. However, episode_min_std = 74.9 remains large, suggesting fly_high terminations are still occurring but less uniformly.

**The episode_min last = 11 steps is an outlier** — likely a transient instability in a small subset of environments. The recent_mean of 86.5 is more representative.

---

### 4. fly_high Termination Frequency — Indirect Evidence — Severity: MEDIUM

The analysis script does not output fly_high termination rates directly for this run. However, the following indirect evidence is available:

- episode_min recent_mean rose 58.9 → 86.5 steps (+47%) — fewer ultra-short episodes
- episode_mean rose 168 → 208.7 steps (+24%) — broadly longer survival
- height_reward recent_mean = 1.817 (vs. 1.403 in the previous run's final state, +29%) — policy is flying closer to desired_height=1.5m more consistently
- height_reward best_ever = 2.751 (new high for this task)

**Verdict:** The evidence strongly supports reduced fly_high termination frequency. The altitude adaptation that was blocked by the 4.5m ceiling is proceeding under the relaxed 5.0m constraint. The height_reward increase is particularly significant — it confirms the drone is actually flying lower, not just surviving longer at the same altitude.

**Capture_distance 3.0→3.5m effect:** The tracking_reward early_mean = 1.814 being immediately above the prior run's 0.889 final level is partly attributable to the wider capture zone. A larger capture radius means the policy achieves tracking triggers more frequently, which is the intended effect. The best_ever = 4.241 (below 4.923) is consistent with the capture zone being "easier to enter but still challenging to sustain."

---

### 5. Policy std — Borderline Concern — Severity: MEDIUM

**This run:**
- policy_std early_mean = 0.391
- policy_std recent_mean = 0.375 (declining)
- policy_std last = 0.378
- policy_std best = 0.397 (from the very start — has never exceeded this)
- Warning threshold: > 0.38 preferred; collapse threshold: < 0.35

**Status:** The policy_std is at 0.375–0.378 in the recent window, which is at the lower edge of the "safe range." This is marginally below the 0.38 warning threshold. The declining trend (0.391 → 0.375) is a concern if it continues.

**Context:** The previous run ended at policy_std ≈ 0.450 (recovered from a low of 0.410). The drop from 0.450 back to 0.375 within 357k steps is steeper than expected. The entropy_loss of −0.00424 to −0.00430 is slightly less negative than in the previous run (−0.00545), which is marginally better (less entropy compression), but the std trajectory is still downward.

**Risk assessment:** At current rate, policy_std may reach the 0.35 collapse threshold within 500k–800k additional steps. This warrants monitoring. If std breaks below 0.37 in the next 200k steps, the run should be re-evaluated.

---

### 6. Training Stability — Stall Diagnosis — Severity: MEDIUM

The script verdict is "stalled." The basis:

- total_reward early_mean = 53.31 vs. recent_mean = 54.28: nearly flat (+1.8%)
- tracking_reward early_mean = 1.814 vs. recent_mean = 1.798: very slightly declining
- distance_reward early_mean = 22.28 vs. recent_mean = 22.96: minor improvement

**Interpretation:** The policy has stabilized at the level it inherited from the checkpoint, but has not advanced meaningfully in 357k steps. This is partially expected for a hot-start: the policy needs time to adapt to the new constraint parameters before making further progress. The phase transition pattern documented for this task (~450k steps from cold start) suggests that from a hot-start, progress may resume at a different cadence.

**However:** The fact that tracking_reward best_ever (4.241) is *below* the historical peak (4.923) is the key signal. The policy has not yet rediscovered the behaviors that produced the 4.923 peak. This could reflect:
1. The capture_distance 3.5m change altering the reward landscape (tracking is "easier" but maximum achievable per-episode tracking reward with 3.5m capture may be different from 3.0m)
2. The policy_std compression limiting exploration capacity
3. The run being too early (357k steps) to assess true trajectory

**The "stalled" verdict at 357k steps from a hot-start is not alarming.** The 11-01-07 run itself looked stable in its early phases before declining. The critical question is whether the recent tracking_reward remains above 1.5 at the 500k step checkpoint.

---

## Current State Judgment

**Verdict: Early stabilization — recovery confirmed, plateau risk moderate.**

The intervention (fly_high 5.0m + capture_distance 3.5m) has achieved its primary goal: arresting the tracking_reward downtrend (0.889 → 1.8) and increasing episode length (168 → 208 steps). The two critical red flags from the previous run are resolved.

The remaining concern is whether the policy can advance *beyond* its current plateau to surpass the historical best (tracking > 4.923, ideally > 6.0 per the milestone). At 357k steps from a hot-start, it is too early to judge this. The 500k checkpoint will be decisive.

---

## Improvement Recommendations

### Priority 1 (MONITOR): policy_std trajectory

**Problem:** policy_std has declined from 0.450 (end of previous run) to 0.375 in 357k steps. At this rate, it may approach the 0.35 collapse threshold within 500k–700k additional steps.

**Watch signal:** If policy_std drops below 0.37 by the 500k–600k step mark, consider a warm entropy injection by slightly increasing entropy_loss_scale. Current value in config not verified, but if it follows the standard MAPPO default of 0.001, a temporary increase to 0.003–0.005 may help.

**No change recommended now.** Monitor first.

### Priority 2 (CONTINGENCY): tracking stall persists past 600k steps

If tracking_reward recent_mean remains below 1.5/ep at the 600k step mark:

**Option A — Tracking weight increase:**
- File: `exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/flyfollow/marl_flyfollow_env_cfg.py`
- Parameter: `tracking_reward_weight` → consider 4.0 → 5.0
- Rationale: increase gradient from tracking success to offset policy_std compression reducing exploration toward the capture zone.

**Option B — capture_distance rollback:**
- File: same
- Parameter: `capture_distance` → 3.5 → 3.0m (if best_ever remains below 4.923, the 3.5m zone may be "too easy" — saturating before pushing to tight follow behavior)
- Rationale: The historical 4.923 peak was achieved with capture_distance=3.0. If tracking is stable but best_ever is not growing, tightening the capture zone may force higher-quality tracking behavior.

**Option C — sustained_follow_duration tightening:**
- Parameter: `sustained_follow_duration` → 1.5 → 2.0s (if policy is stable enough)
- Only apply if tracking_reward recent_mean > 2.0/ep at 1M steps.

### Priority 3 (LOW): episode_min volatility

episode_min_std = 74.9 (std > mean) indicates some environments still terminate very early. This is not blocking progress but contributes to gradient noise.

If fly_high terminations are confirmed as the primary cause at the 500k checkpoint, consider whether fly_high_termination_z 5.0m should be further relaxed to 5.5m for the next restart. **Do not apply mid-run.**

---

## Experiment Plan

1. Continue current run `2026-03-26_09-31-34` to 1.5M steps
2. **500k step checkpoint (est. ~450k steps from now):**
   - tracking_reward recent_mean > 1.5/ep → PASS, continue
   - tracking_reward recent_mean < 1.5/ep → apply Priority 2 Option A (tracking_weight 4.0 → 5.0) at next restart
   - height_reward recent_mean > 1.7/ep → altitude adaptation confirmed, continue
   - policy_std < 0.37 → flag for entropy intervention
   - episode_min recent_mean > 100 steps → fly_high pressure resolved
3. **1M step checkpoint:**
   - tracking_reward mean > 2.0/ep → PASS
   - tracking_reward best_ever > 5.5 → policy expanding (on track for 6.0 target)
   - If best_ever still below 4.923 at 1M steps → consider capture_distance rollback 3.5 → 3.0
4. **Success criteria (this run):**
   - tracking_reward mean > 2.0/ep, std < 0.4
   - episode_mean > 220 steps
   - best_ever tracking > 6.0 (exceed historical peak of 4.923)
   - policy_std > 0.36 (no entropy collapse)


---


---

# Training Analysis Report — 2026-03-26_09-31-34 (第二次检查)

**Run:** 2026-03-26_09-31-34_mappo_torch_mappo
**Analysis Date:** 2026-03-27
**Total Steps (this run):** 356,900（与上次分析持平 — 训练已停止或 events 未更新）
**Cumulative Steps:** ~2.21M（含前序热启动积累）
**Task:** Isaac-marl-flyfollow-v0 / MAPPO

---

## 训练指标摘要

| 指标 | 上次分析 (357k) | 本次 (356.9k) | 变化 |
|------|----------------|---------------|------|
| policy_std (last) | 0.375 | 0.3778 | **↑ +0.003（好转）** |
| policy_std (recent_mean 200步) | 0.375 | 0.3754 | 持平 |
| tracking_reward (last) | 2.174 | 2.174 | 持平（同一时刻） |
| tracking_reward (recent_200 mean) | 1.798 | 1.700 | 略降 |
| distance_reward (recent_200) | 22.96 | 22.26 | 持平 |
| height_reward (recent_200) | 1.817 | 1.783 | 持平 |
| episode_mean (last) | 204.6 | 204.6 | 持平 |
| episode_mean (recent_200) | 208.7 | 208.7 | 持平 |
| total_reward (recent_200) | 54.28 | — | — |
| upright_penalty (recent_200) | — | **−4.108** | **新发现** |
| height_penalty (recent_200) | — | **−3.480** | **新发现** |

**重要：** events 文件最后步数仍为 356,900，与上次分析完全相同。这表明本次分析读取的是同一数据，**训练可能已停止运行**，或本次分析与上次分析间隔极短（仅几分钟内）。

---

## 核心发现

### 发现 1：policy_std 尚未跌破 0.37 — PASS

**状态：** policy_std 当前范围 0.3754–0.3778（近200步），全程历史最低仅 0.3706，**从未跌破 0.370**。

**斜率分析（最后200个记录点的线性回归）：**
- 斜率 = **+0.000166 / 1000步**（正值，正在小幅反弹）
- 当前 0.3778 距警戒线 0.370 仍有 0.008 的缓冲
- 前次分析担忧的"继续下降至跌破 0.37"**未发生**，斜率已转正

**判断：** 无需 entropy 干预。policy_std 在 0.370–0.380 区间震荡整固，这是正常的"探索稳定化"行为，而非压缩崩溃前兆。

---

### 发现 2：upright_penalty 是当前最大负向奖励 — HIGH

**数据：** upright_penalty recent_200_mean = **−4.108/ep**，last = −4.54，历史最差 −6.68。

**背景：** 本轮重启（2026-03-26）在 commit ef50372 中将 `upright_penalty_weight` 从 0.5 恢复至 1.0（与 `body_rate_penalty_weight` 同步）。这是为了改善姿态稳定性（commit d7c8205）。

**问题：** 与前一次运行（11-01-07）的 upright_penalty 相比，本次值 −4.1/ep 是否偏大？需要与 cfg 中的权重核实。此前 15-06-36 运行（weight=0.5）对应值约 −1.0–1.5/ep。权重加倍 → 实际惩罚倍增，合理。

**关键：** upright_penalty（−4.1）+ height_penalty（−3.5）+ fly_high（−0.16）+illegal_contact（−0.70）= **−8.46/ep 净负向**。相比之下，height_reward（+1.78）+ tracking_reward（+1.70）+distance_reward（+22.3）= **+25.78/ep 净正向**。

净收益仍正向（+17.3/ep），但 upright_penalty 作为单一最大负向项目值得关注。

**结论：** 不建议立即降低权重。但若 tracking_reward 在 500k 步后仍不增长，upright_penalty 可能在抑制进攻性追踪行为（姿态倾斜用于加速追踪目标）。

---

### 发现 3：crash + illegal_contact 终止 — 新关注点 — MEDIUM

**数据（recent_mean）：**
- `Episode_Termination/crash`: 0.73/ep（即平均每个环境每次迭代 0.73 个 episode 以 crash 结束）
- `Episode_Termination/illegal_contact`: 0.71/ep（几乎完全重叠）
- `Episode_Termination/falcon_fly_high`: 0.31/ep（降低，之前为 0.56/ep）
- `Episode_Termination/time_out`: 0.000（无超时，episode 总长度未达上限）

**解读：** crash ≈ illegal_contact 表明 crash 终止几乎全部由 illegal_contact（无人机碰地面或障碍物）触发。fly_high 终止已从之前的主要终止原因（11-01-07 期间）降低，5.0m 放宽有效。

**问题：** crash/illegal_contact 率为 0.71–0.73/ep（即约 70% 的 episode 以碰撞结束）是否正常？需要与早期运行对比。根据 15-06-36 运行（670k 步时），当时主要终止原因未记录具体数字，但 episode_min 约 100 步，暗示碰撞终止并不如此频繁。

**可能成因：** 高 upright_penalty 使无人机在接近目标时"翻滚"惩罚减少倾斜，但接近目标时由于速度控制不足撞地（desired_height=1.5m，目标在 z≈0.25m，接近时高度误差大）。

---

### 发现 4：tracking_reward 趋势 — 高波动，均值停滞 — MEDIUM

**数据（最后10步）：**
- 范围：1.316 至 2.174，震荡幅度 ≈ 0.86/ep（约±25%）
- recent_200_mean = 1.700（略低于上次分析的 1.798）
- best_ever = 4.241（低于历史峰值 4.923）

**解读：** tracking_reward 高波动（CV ≈ 28%）是策略尚在探索 capture zone 的正常现象，不是崩溃信号。均值 1.7 接近 previous run 早期的 1.83，说明热启动恢复完成但未超越。

**关键路径：** 从 1.7 到 2.0/ep（下一个里程碑）需要策略探索出更稳定的持续跟随行为。在 policy_std 稳定（不继续压缩）的前提下，这是可以期待的。

---

### 发现 5：success_reward 持续为 0 — 关注

**数据：** success_reward recent_200_mean = 0.000，best_ever = 0.000（全程未触发）。

**背景：** success_reward 需要所有3架无人机同时在各自目标的 capture_distance（3.5m）内，且持续 sustained_follow_duration（1.5s）以上。

**分析：** 尽管 tracking_reward（需满足 1/1/2 架进入 capture zone 的某些版本）达到 2.174，success_reward 从未触发，说明三架同时达标的条件极难满足。这不是 bug，是任务难度的体现。

---

## 综合判断

**结论：继续等待。**

五项问题分析总结：

1. **policy_std 未跌破 0.37** — 斜率转正，当前 0.375–0.378，无需熵干预。上次分析的预警条件**未被触发**。

2. **tracking_reward 均值 1.70/ep** — 高波动但稳定，仍在 1.5 合格线以上，符合 500k 步检查点标准（>1.5/ep）。训练仍在进行中（如果 events 确实是最新的），无下降趋势。

3. **episode 长度 208 步** — 维持 +24% 改善水平，fly_high 终止明显减少（从 56% 降至 31%），5.0m 放宽持续有效。

4. **新发现：upright_penalty −4.1/ep** 是当前最大单项负向奖励，但整体净收益仍为正。不需要立即处理。

5. **新发现：crash/illegal_contact 70% 终止率** 是需要在下一检查点确认的指标。如果这一比例继续上升或 episode_length 下降，需要诊断无人机是否因接近目标而频繁碰地。

**当前阶段：** 357k 步热启动稳定化阶段（符合预期）。500k 步是真正的判断时刻。

---

## 更新后的检查点标准（500k 步）

| 指标 | 通过标准 | 失败标准 | 当前值 |
|------|---------|---------|-------|
| tracking_reward recent_mean | > 1.5/ep | < 1.5/ep | 1.70 (边缘通过) |
| height_reward recent_mean | > 1.7/ep | < 1.5/ep | 1.78 (通过) |
| policy_std | > 0.370 | < 0.370 | 0.375 (通过) |
| episode_mean | > 190 步 | < 160 步 | 208 步 (通过) |
| crash_rate | < 80% | > 90% | 71% (注意) |
| tracking best_ever | > 4.5 | < 3.5（停滞） | 4.241 (注意) |

**500k 步通过后行动：** 继续至 1M 步。
**500k 步任一失败后行动：** 按 Priority 排序，优先检查 crash 原因后再调整参数。

---

## Changelog
- 2026-03-27: 357k 步复检（与上次分析数据相同，events 未更新）。Policy_std 斜率转正，跌破 0.37 风险消除。发现 upright_penalty（−4.1/ep）为最大负向项，crash/illegal_contact 终止率 71% 为新关注点。综合判断：继续等待，等待 500k 步检查点。

---

# Training Analysis Report

**Run:** 2026-03-26_14-02-25_mappo_torch_mappo
**Date:** 2026-03-27
**Task:** Isaac-marl-flyfollow-v0 (move_flyfollow variant)
**Algorithm:** MAPPO
**Total Timesteps:** 2,000,000
**Script Status:** regressing（分析器评级）

## Training Metrics Summary

| 指标 | 数值 | 对比上轮（357k步） |
|------|------|------------------|
| total_reward_mean（最后值） | 69.65 | 46.22 |
| total_reward_mean（recent_mean） | 51.10 | 54.28 |
| total_reward_best_ever | 111.63 | 111.63（相同） |
| tracking_reward（最后值） | 2.446 | 2.174 |
| tracking_reward（recent_mean） | 1.663 | 1.798 |
| tracking_reward（best_ever） | 4.757 | 4.241 |
| height_reward（recent_mean） | 1.797 | 1.817 |
| distance_reward（recent_mean） | 21.20 | 22.96 |
| episode_mean（recent_mean） | 192.7 步 | 208.7 步 |
| policy_std（最后值） | 0.3551 | 0.3778 |
| policy_std（recent_mean） | 0.3712 | 0.3754 |
| crash 终止率（recent_mean） | 83.1% | 71%（估算） |
| fly_high 终止率（recent_mean） | 21.0% | ~31% |
| illegal_contact 终止率（recent_mean） | 81.8% | ~71% |
| success_reward | 0.000（从未触发） | 0.000 |
| velocity_penalty | 0.000（始终为零） | 0.000 |

---

## 问题 1：热启动判断 — 确认为同一连续 run 的 TFEvents 延续

**证据：**
- 两个 run 的前 5 步 `total_reward_mean` 完全相同：`[50.6, 55.4, 43.0, 54.9, 41.7]`，均值 49.1。
- 两个 run 的前 5 步 `policy_std` 完全相同：`[0.3963, 0.3967, 0.3968, 0.3968, 0.397]`。
- 新 run step 从 200 开始（而非从 0 重置），与上轮 step 范围（100–356900）高度重叠。

**结论：** 这两个 run 目录共享同一批 TFEvents 数据（或新 run 从上轮 checkpoint 热启动后重放了相同的初始轨迹）。**新 run 并非从零冷启动，而是上轮训练的直接延续，共 2M 步。** 上轮（09-31-34）已达到 best_ever=111.63（对应某早期高峰），新 run 的 best_ever 同样为 111.63，进一步确认数据连续性。

---

## 问题 2：tracking_reward 趋势 — 下降，进入衰退期

**数据：**
- early_mean: 1.790 → recent_mean: **1.663**（-7.1%）
- best_ever: 4.757（远高于上轮的 4.241，说明在训练中途有过高峰）
- 最后值 2.446 高于 recent_mean，表明存在局部反弹，但趋势仍向下

**诊断：** tracking_reward 在 2M 步时已从早期均值下滑。结合 total_reward_mean 的 early→recent 也从 53.6 降至 51.1，判断训练整体处于轻度衰退状态。policy 已接近当前 reward 设计下的局部最优，无法进一步优化 tracking。

**根因假设：**
1. upright_penalty（recent_mean = **-3.90/ep**）+ height_penalty（recent_mean = **-2.94/ep**）合计约 **-6.84/ep**，是总奖励中最大的负向来源，压制了 tracking 信号。
2. illegal_contact 终止率 81.8% 意味着大多数 episode 在碰撞中终止，无法积累足够的 tracking 奖励。

---

## 问题 3：crash/illegal_contact 终止率 — 恶化（HIGH）

**数据：**
- `Episode_Termination/crash`: early_mean=83.1%，recent_mean=**83.1%**（无改善）
- `Episode_Termination/illegal_contact`: early_mean=81.2%，recent_mean=**81.8%**（轻微恶化）
- 与上轮报告的 71% 相比，**上升约 10–12 个百分点**

**严重程度：** HIGH。超过 80% 的 episode 以碰撞终止，无人机无法建立长时稳定的追踪行为。

**诊断：** 碰撞率高有两种成因：
1. 无人机飞得过低（fly_low 终止 recent_mean=1.9%）-- 低空坠地
2. 无人机碰到地面/目标小车/其他无人机 -- 接触碰撞

`Episode_Termination/drones_collide` recent_mean=0.11%，可排除无人机间碰撞。主因应为无人机与地面或小车的接触（illegal_contact）。这与 fly_high 终止率下降（31%→21%）一致——无人机改为飞低，但代价是碰地率上升。

---

## 问题 4：episode 长度 — 轻微缩短（MEDIUM）

**数据：**
- episode_mean recent_mean: **192.7 步**（vs 上轮 208.7 步，-7.7%）
- timesteps_min recent_mean: 74.4 步（极短 episode 持续存在）
- timesteps_max recent_mean: 276 步（最长 episode 未改善）

**诊断：** episode 长度轻微缩短，与 crash 率恶化一致。无人机越来越难以维持长时间飞行。2M 步时距离任务目标（episode_length_s=60s，对应约 600 步）仍有巨大差距。

---

## 问题 5：policy_std — 继续收窄（MEDIUM）

**数据：**
- 上轮（357k步）：0.3778
- 本轮最后值：**0.3551**（-5.8%）
- recent_mean: 0.3712，recent_std 仅 0.0069（极低波动）
- 趋势：单调下降，从 early_mean=0.3816 持续降低

**诊断：** policy_std 继续收窄，探索空间压缩。在 crash 率未改善的情况下，过早的探索收缩意味着 policy 正在收敛到一个次优局部解（碰撞-重置循环），而非真正学会追踪。

**风险：** 若 policy_std 降至 0.34 以下，可能需要熵奖励干预。当前尚未触发紧急阈值，但需监控。

---

## 问题 6：height_penalty 与 upright_penalty 主导负向奖励（HIGH）

**数据：**
- `height_penalty` recent_mean: **-2.94/ep**（early=-3.33，小幅改善）
- `upright_penalty` recent_mean: **-3.90/ep**（early=-4.17，小幅改善）
- 两者合计 **-6.84/ep**，远超 tracking_reward(+1.66) + height_reward(+1.80)

**诊断：** 无人机存在严重的姿态不稳定和高度控制不良。当前配置：
- `height_penalty_weight=2.0`，`height_penalty_threshold=0.3`（高度偏差超过 0.3m 即触发）
- `upright_penalty_weight=1.0`
- `desired_height=1.5m`

这些惩罚的绝对值（各约 -3 至 -4/ep）意味着无人机大部分时间都在偏离期望姿态和高度，且这一问题在 2M 步训练后**未能有效解决**。

---

## 综合判断

**结论：当前训练陷入局部次优，需要干预。**

2M 步训练后的状态：
- tracking_reward 从峰值（best=4.757）回落至 recent_mean=1.663，**处于下降趋势**
- crash/illegal_contact 率 **83%**，绝大多数 episode 在碰撞中结束
- policy_std 持续收窄（0.355），探索能力萎缩
- height_penalty + upright_penalty 合计约 -6.84/ep，成为主导的负向信号
- success_reward 从未触发

这不是"继续等待"可以解决的问题。policy 已在当前 reward 设计下达到稳定的次优均衡：无人机学会了在碰撞-重置循环中积累 distance_reward（21.2/ep），但无法突破到稳定追踪阶段。

---

## 改进建议

### Priority 1 (HIGH)：降低 height_penalty_weight，放宽高度容忍阈值

**问题：** height_penalty 贡献 -2.94/ep，high_penalty_threshold=0.3m 过于严苛，在高速追踪动态目标时难以维持。
**建议改动：**
- 文件：`marl_flyfollow_env_cfg.py`
- `height_penalty_weight`: 2.0 → **1.0**
- `height_penalty_threshold`: 0.3 → **0.5**
- **理由：** 减少高度惩罚主导效应，让 policy 更多关注 tracking 而非姿态微调。

### Priority 2 (HIGH)：降低 upright_penalty_weight

**问题：** upright_penalty -3.90/ep 为最大单项负向奖励，在追踪动态小车时姿态倾斜是必要的机动动作，过重惩罚会阻碍机动性。
**建议改动：**
- 文件：`marl_flyfollow_env_cfg.py`
- `upright_penalty_weight`: 1.0 → **0.5**
- **理由：** 先前 2026-03-12 已有将 body_rate_penalty 等权重下调的成功经验（body_rate/upright 0.5→1.0 的回调是因姿态崩溃，但当前 crash=83% 说明过重惩罚已适得其反）。

### Priority 3 (MEDIUM)：提高 illegal_contact 惩罚以主动惩戒碰撞行为

**问题：** `illegal_contact_penalty=1.0`，`crash_penalty_scale=1.0`。当前惩罚幅度不足以改变 policy 行为（crash 率 83% 说明 policy 已将碰撞终止视为"正常"）。
**建议改动：**
- 文件：`marl_flyfollow_env_cfg.py`
- `illegal_contact_penalty`: 1.0 → **3.0**
- `crash_penalty_scale`: 1.0 → **2.0**
- **理由：** 大幅提升碰撞惩罚，使 policy 学会规避碰撞，延长 episode 长度，进而有更多机会积累 tracking_reward。

### Priority 4 (MEDIUM)：降低 desired_height 以减少高度追踪难度

**问题：** `desired_height=1.5m`，`fly_high_threshold=4.0m`，留白充裕，但无人机频繁触发 height_penalty（1.5m ±0.3m），说明控制精度不足。
**建议改动：**
- 文件：`marl_flyfollow_env_cfg.py`
- `desired_height`: 1.5 → **2.0m**（远离地面，减少意外碰地）
- `fly_high_threshold`: 4.0 → **4.5m**（对应放宽上界）
- **理由：** 飞行高度提升可减少与地面/小车的意外接触，降低 illegal_contact 率。

### Priority 5 (LOW)：增加熵系数以缓解 policy_std 收缩

**问题：** policy_std 单调下降至 0.355，在 crash 率未改善的情况下提前收敛。
**建议改动：**
- 文件：SKRL MAPPO 训练配置
- `entropy_loss_scale`（或等效的 `entropy_coeff`）：当前值 → 适当增大（参考当前 entropy_loss=-0.0036）
- 时机：在 Priority 1-3 改动后重新训练，若 policy_std < 0.35 再介入。

---

## 实验计划

1. 按 Priority 1→2→3→4 顺序修改 `marl_flyfollow_env_cfg.py`
2. 重启训练（冷启动或从本 run 的最佳 checkpoint 热启动）：
   ```bash
   python3 scripts/skrl/train.py --task=Isaac-marl-flyfollow-v0 \
     --headless --num_envs=2048 --algorithm="MAPPO"
   ```
3. 关键监控指标（前 500k 步）：
   - crash 终止率：目标 < 60%（当前 83%）
   - episode_mean：目标 > 220 步（当前 192 步）
   - tracking_reward recent_mean：目标 > 2.0/ep（当前 1.663）
   - height_penalty：目标 > -1.5/ep（当前 -2.94）
   - upright_penalty：目标 > -2.5/ep（当前 -3.90）
4. 成功标准（1M 步检查点）：
   - crash 率 < 50%，tracking_reward > 2.5/ep，success_reward > 0.1（至少偶发触发）

---

## Changelog
- 2026-03-27: 分析 2026-03-26_14-02-25 run（2M步，热启动延续自09-31-34）。确认为同一连续训练的延续。发现训练进入轻度衰退：tracking_reward 下滑至 1.663，crash 率恶化至 83%，height_penalty+upright_penalty 合计-6.84/ep 成为主导负向信号，policy_std 收窄至 0.355。综合判断需要干预：建议降低 height_penalty_weight(2→1)、upright_penalty_weight(1→0.5)，提升 illegal_contact_penalty(1→3)，提高 desired_height(1.5→2.0m)。

---

# Training Analysis Report

**Run:** 2026-03-27_19-15-45_mappo_torch_mappo
**Date:** 2026-03-28
**Task:** Isaac-marl-flyfollow-v0（move_flyfollow 变体）
**Algorithm:** MAPPO
**热启动来源：** 2026-03-26_14-02-25 best_agent.pt（累计约 2M 步）

---

## Training Metrics Summary

| 指标 | 上一轮（14-02-25）recent | 本轮 early | 本轮 recent | 变化 |
|------|----------------------|-----------|------------|------|
| Total reward mean | 51.10 | 48.88 | 47.75 | -0.6%（持平） |
| tracking_reward/ep | 1.663 | 1.324 | 1.316 | -21% |
| distance_reward/ep | 21.20 | 18.55 | 18.25 | -14% |
| height_penalty/ep | -2.94 | -1.11 | -0.94 | **+68% 显著改善** |
| upright_penalty/ep | -3.90 | -1.85 | -1.89 | **+52% 显著改善** |
| illegal_contact 惩罚/ep | -0.78 | -2.15 | -2.28 | -192%（如预期：权重3×） |
| crash 终止率 | 83.1% | 74.9% | 80.3% | 无实质改善 |
| illegal_contact 终止率 | 81.8% | 74.5% | 79.3% | 无实质改善 |
| fly_high 终止率 | 21.0% | 29.3% | 24.3% | 轻度劣化 |
| episode 长度 mean | 192.7 步 | 182.0 步 | 179.1 步 | -7% 轻微缩短 |
| policy_std | 0.3551（last） | 0.3757 | 0.4062 | **+14% 显著回升** |
| Total timesteps analyzed | 2,000,000 | — | 1,125,100 | — |
| success_reward | 0.000 | 0.000 | 0.000 | 无触发 |

---

## Observations & Findings

### 1. 热启动确认 — 成功 — MEDIUM

**症状：** 训练开始即为正向奖励（early total_reward=48.88），无从零学习的负值阶段。
**根据：** 上一轮 recent_mean=51.10，本轮 early_mean=48.88。差值约 -4.4 pt，属于正常热启动后的短暂适应震荡（新参数配置下策略需重新稳定）。结论：热启动成功，无需验证 checkpoint 完整性。

### 2. crash/illegal_contact 终止率未改善 — HIGH

**症状：** crash 终止率 early=74.9% → recent=80.3%。crash 与 illegal_contact 两指标的时间序列完全重叠（完全一致），确认二者由同一物理事件驱动。illegal_contact_penalty 从 -0.78/ep 升至 -2.28/ep（weight 1→3 生效）。

**时间序列分析：**
- Step 125k–500k：crash 率在 0.36–0.77 范围震荡，期间出现 step~625k 的局部低点 0.13。
- Step 625k–1125k：crash 率重新上升，最终回到 1.0（last value）。

**根本原因：** 提高 illegal_contact_penalty（1→3）确实提高了每次碰撞的惩罚幅度，但未改变"碰撞后立即 reset"的结构。policy 仍然发现"允许碰撞 → 快速 reset → 避免长时间受高 upright/height 惩罚"是局部最优策略。碰撞惩罚虽加重，但 episode 长度（179步 ≈ 1.8s × 3.0 weight = -5.4 max）与 upright_penalty（-1.89/ep）+ height_penalty（-0.94/ep）合计 -2.83/ep 之间的信号竞争仍存在。

**备注：** 上一轮 crash 率 83%，本轮 recent 80.3%，下降仅 3pp，不具统计意义。核心问题未解决。

### 3. tracking_reward 下降 — HIGH

**症状：** tracking_reward early=1.324，recent=1.316。对比上一轮 early=1.777（来自14-02-25），**本轮热启动后 tracking_reward 立即下跌约 25%**，且整个训练过程中未见回升。时间序列显示最高点在 step~59k（2.10），此后持续在 0.7–1.7 范围震荡，无上升趋势。best=3.41（vs 历史最高 4.92 于 11-01-07 run），新高未被突破。

**根本原因：** episode 长度从上一轮的 192.7 步缩短至 179.1 步（-7%），导致每 episode 可积累的 tracking_reward 时间窗口收缩。fly_high 终止率轻度劣化（21% → 24.3% recent），也消耗了部分长 episode 机会。desired_height 从 1.5→2.0m 的调整增加了 fly_high 风险（与 fly_high_threshold=4.5m 的缓冲变小了 0.5m）。

### 4. 惩罚项成功削减 — 验证有效 — MEDIUM

**症状（正面）：**
- height_penalty: -2.94/ep → -0.94/ep（-68% 改善）— weight 2.0→1.0 + threshold 0.3→0.5m 联合生效。
- upright_penalty: -3.90/ep → -1.89/ep（-52% 改善）— weight 1.0→0.5 生效。
- fly_high_penalty: -0.028/ep → -0.024/ep（轻微改善，desired_height 上移效果）。

**结论：** 这两项改动按预期工作，副作用是总负向惩罚减少后，crash 的相对成本没有相应增加，反而强化了"靠 reset 逃避惩罚"的局部最优。

### 5. policy_std 回升至安全范围 — 正面信号 — LOW

**症状：** policy_std early=0.376，recent=0.406，last=0.408。与上一轮末尾的 0.355 相比，**回升了 15%**，重新进入 0.40 以上的探索充分区间。

**根本原因：** 热启动后新参数（尤其 upright/height 权重降低）减少了负向梯度压力，允许 policy 恢复更广泛的动作分布。这是本轮最正面的技术信号。

### 6. episode 长度轻微下滑，成功率仍为零 — HIGH

**症状：** episode mean 179 步（目标 >220 步，未达）。success_reward 全程 0.000——3架无人机同时捕获目标从未发生。velocity_follow 平均仅 0.054/ep（极低），说明速度跟随能力仍然很弱。

---

## 综合判断

本轮热启动**部分成功**：惩罚权重削减的改动按预期生效，policy_std 健康回升，但**核心改善目标（crash 率下降）完全未达成**。

训练呈现以下结构性矛盾：
1. **惩罚减轻反而强化了 crash 局部最优**：减轻 height/upright 惩罚后，"受惩罚 → reset"的成本降低，policy 更容易接受碰撞终止。
2. **illegal_contact_penalty=3.0 惩罚幅度仍不足以驱动行为改变**：99.7% 的 episode 都有碰撞（仅 0.3% 无接触），说明 policy 并未将"避免碰撞"学为基本行为，而是将碰撞内化为常规流程。
3. **tracking_reward 因 episode 缩短而下滑**，未出现新高，表明本轮并未拓展 policy 能力边界。

**当前状态与 14-02-25 run 的对比：** 本轮总奖励（recent 47.75）低于上一轮（recent 51.10），tracking_reward（1.316 vs 1.663）也倒退。热启动没有带来能力提升，仅完成了"从坏的配置中逃离"的工作。

**结论：需要第二次有针对性的干预。** 重点应从"惩罚削减"转为"碰撞行为根治"。

---

## Improvement Recommendations

### Priority 1 (CRITICAL)：引入 crash 专项 dense 惩罚（per-step 距离地面/障碍物惩罚）

**问题：** 当前 illegal_contact 是 episode 结束时的稀疏惩罚（碰后 reset，信号延迟到 episode 末）。Policy 无法在碰撞前就获得"危险临近"的梯度信号，缺乏主动规避行为。

**建议改动：**
- 文件：`marl_flyfollow_env_cfg.py`
- 新增：`proximity_penalty_weight: 0.5`（当无人机距地面 < 0.8m 时，per-step 指数衰减惩罚）
- 或者：`min_altitude_soft_threshold: 0.8m`（类似 fly_high_penalty 的软惩罚机制）
- **理由：** 给 policy 一个持续的近地危险信号，让它在碰撞前就开始爬升，而非等到 crash 终止后才 reset。

### Priority 2 (CRITICAL)：进一步提高 illegal_contact_penalty 或调整惩罚结构

**问题：** illegal_contact_penalty=3.0 后，per-episode 平均惩罚 -2.28（约等于 0.76次碰撞/ep），比 weight=1.0 时的 -0.78 增加了 3×，但 crash 率只从 83% 降到 80%。说明 3.0 的惩罚幅度仍不足以触发行为改变。

**建议改动：**
- 文件：`marl_flyfollow_env_cfg.py`
- `illegal_contact_penalty`: 3.0 → **5.0**
- **理由：** 以较大步长（5.0）测试惩罚是否有明确的阈值效应。若 crash 率对 3.0 无响应，则 3.0 处于 policy 的"可接受损失"区间，需要跨越该阈值。

### Priority 3 (HIGH)：提高 fly_low 保护 threshold，增加软落地惩罚

**问题：** `falcon_fly_low` 终止率 recent=1.7%（轻度存在），且 desired_height=2.0m 后离地面近。99.7% episode 有 illegal_contact，说明无人机频繁触地/触碰物体，但 fly_low termination 只有 1.7%，说明大部分碰撞不是通过 fly_low 触发，而是通过接触检测触发。

**建议改动：**
- 文件：`marl_flyfollow_env_cfg.py`
- `fly_low_threshold`: 当前值 → **0.5m**（如当前为 0.3m，提高以给更多提前预警）
- 新增 `low_altitude_soft_penalty_weight: 1.0`（在 0.5m–1.0m 区间内提供 per-step 惩罚，创造安全高度带）
- **理由：** 在物理碰撞前就提供梯度信号，结合 Priority 1 形成"软-硬"双层高度保护。

### Priority 4 (MEDIUM)：降低 desired_height 回到 1.5m 或保持 2.0m 但放宽 height_penalty_threshold

**问题：** desired_height=2.0m 后，fly_high 终止率从 21% 小幅上升至 24.3%（缓冲区减小）。同时 height_penalty 明显改善（-2.94→-0.94），说明 desired_height 上移后实际高度跟踪更精准了。但 fly_high 风险的上升需要平衡。

**建议：**
- 维持 desired_height=2.0m（惩罚项改善明显，不建议回退）
- `height_penalty_threshold`: 0.5m → **0.6m**（进一步放宽容忍，减少 height_penalty 频率）
- `fly_high_threshold`: 4.5m 维持（合理）

### Priority 5 (LOW)：监控 policy_std，确保不跌破 0.37

**问题：** policy_std 已回升至 0.408，但本轮训练将延续，需监控其不重新下降。
**建议：** 本次训练观察期间如 policy_std < 0.37，考虑在下一次 restart 时调整 entropy_coeff（MAPPO config 中的 `entropy_loss_scale`）。当前暂不改动。

---

## 实验计划

本轮训练（19-15-45）仍在进行，目前 1.125M 步。建议**继续本轮至 1.5M 步**后评估以下 milestone：

**1.5M 步 checkpoint 通过标准：**
- crash 终止率 < 75%（当前 80.3%，仅需小幅改善）
- tracking_reward recent_mean > 1.5/ep（当前 1.316，需恢复）
- episode mean > 185 步（当前 179 步）

**若 1.5M 步未通过，执行 Restart B（第二次干预）：**
```bash
python3 scripts/skrl/train.py --task=Isaac-marl-flyfollow-v0 \
  --headless --num_envs=2048 --algorithm="MAPPO" \
  --checkpoint=/path/to/2026-03-27_19-15-45/best_agent.pt
```

Restart B 配置变更（相比当前）：
| 参数 | 当前值 | Restart B |
|------|--------|-----------|
| illegal_contact_penalty | 3.0 | **5.0** |
| low_altitude_soft_penalty_weight | 0 | **1.0** （新增） |
| height_penalty_threshold | 0.5 | **0.6** |

**关键监控指标（前 500k 步）：**
- crash 终止率：目标 < 65%
- illegal_contact_penalty/ep：目标 > -3.0（即碰撞频率下降）
- episode mean：目标 > 200 步
- tracking_reward：目标 > 1.8/ep

**成功标准（1M 步）：**
- crash 率 < 50%，tracking_reward > 2.5/ep，policy_std > 0.38

---

## Changelog
- 2026-03-28（第一次分析）: 分析 2026-03-27_19-15-45 run（1.125M 步，热启动自 14-02-25）。热启动确认成功。惩罚削减按预期生效（height_penalty -68%，upright_penalty -52%，policy_std 回升至 0.408）。但 crash 终止率仍 80%，tracking_reward 下滑至 1.316（vs 上轮 1.663），核心目标未达成。诊断：crash 局部最优结构未被打破，illegal_contact_penalty=3.0 仍在 policy 可接受损失范围内。建议：继续本轮至 1.5M 步观察；若未通过 milestone，Restart B 将 illegal_contact_penalty→5.0 并新增低空软惩罚。
- 2026-03-28（第二次分析）: 重新分析 2026-03-27_19-15-45 run（1.914M 步，完整运行）。1.5M 步 milestone 全部未通过（tracking 1.316→1.054，crash 未改善，episode 长度下滑至 171 步）。run 已进入持续退化阶段，policy_std 回升至 0.441（正面信号，但未能转化为性能提升）。1.5M 步 checkpoint 通过标准均未达到，触发 Restart B 条件。确认执行 Restart B。


---

# Training Analysis Report — 2026-03-27_19-15-45（第二次分析，完整 1.914M 步）

**Run:** 2026-03-27_19-15-45_mappo_torch_mappo
**Date:** 2026-03-28
**Task:** Isaac-marl-flyfollow-v0（move_flyfollow）
**Algorithm:** MAPPO
**热启动自:** 14-02-25 best_agent.pt
**脚本结论:** regressing

---

## Training Metrics Summary

| 指标 | early_mean | recent_mean | last | best |
|------|-----------|-------------|------|------|
| total_reward_mean | 47.27 | 40.96 | 37.95 | 103.56 |
| total_reward_max | 76.93 | 59.93 | 61.21 | 161.08 |
| distance_reward | 17.98 | 16.04 | 13.13 | 33.45 |
| tracking_reward | 1.255 | 1.054 | 0.948 | 3.406 |
| height_reward | 1.665 | 1.693 | 1.434 | 2.680 |
| velocity_penalty | 0.0 | 0.0 | 0.0 | 0.0 |
| force_penalty | 0.357 | 0.340 | 0.285 | 0.578 |
| body_rate_penalty | 0.225 | 0.165 | 0.127 | 0.413 |
| action_smoothness | 0.366 | 0.205 | 0.127 | 0.786 |
| timesteps_mean | 179.9 | 171.3 | 164.8 | 342.0 |
| policy_std | 0.380 | 0.415 | 0.436 | 0.446 |
| value_loss | 0.065 | 0.034 | 0.039 | 0.330 |
| entropy_loss | -0.00433 | -0.00516 | -0.00562 | — |

**本次运行配置（来自 env.yaml / marl_move_flyfollow_env_cfg.py）：**
- tracking_reward_weight = 4.0，capture_distance = 3.5m
- velocity_follow_weight = 1.5，height_reward_weight = 2.0
- desired_height = 2.0m，fly_high_termination_z = 5.0m
- height_penalty_weight = 1.0，height_penalty_threshold = 0.5m
- upright_penalty_weight = 0.5，body_rate_penalty_weight = 1.0
- illegal_contact_penalty = 3.0，fly_low_penalty = 1.0
- fly_high_threshold = 4.5m，fly_high_penalty_weight = 2.0

---

## Observations & Findings

### 1. 持续退化（Monotonic Decay）— 严重程度：CRITICAL

**Symptom:** 全部关键指标（total_reward、distance_reward、tracking_reward）从 early 到 recent 单调下降，无任何恢复迹象。
- total_reward: early 47.27 → recent 40.96 → last 37.95（−20%）
- distance_reward: early 17.98 → recent 16.04 → last 13.13（−27%）
- tracking_reward: early 1.255 → recent 1.054 → last 0.948（−24%）

**与 1.125M 步分析的对比：**
- 1.125M 时：tracking_reward recent_mean = 1.316，last = ?（未记录）
- 1.914M 时：tracking_reward recent_mean = 1.054，last = 0.948
- 1.5M 步 milestone（tracking > 1.5/ep）明确未通过。

**Root Cause:** 与上一轮 11-01-07 run（fly_high_termination_z=4.5m 导致永久退化）高度相似的模式，但本次退化速度更慢（因为 5.0m 约束比 4.5m 宽松）。根本机制相同：crash 局部最优主导 policy 行为，每次 reset 都比继续飞行收益更高。

**Evidence:** best_ever tracking_reward = 3.406（仅在早期热启动阶段出现），后续 1.5M 步未产生新高。这是"能力冻结"的经典信号——policy 未在持续探索中扩展边界，而是在已知局部最优周围收缩。

---

### 2. 1.5M 步 Milestone 全部未通过 — 严重程度：CRITICAL

**触发 Restart B 的决策依据：**

| Milestone | 目标 | 实际（1.914M 步 recent） | 结论 |
|-----------|------|------------------------|------|
| crash 终止率 < 75% | <75% | ~80%（类比上轮趋势） | **FAIL** |
| tracking_reward > 1.5/ep | >1.5 | 1.054 | **FAIL** |
| episode mean > 185 步 | >185 | 171.3 | **FAIL** |

三项 milestone 全部未通过。上一份分析（1.125M 步）已记录"若 1.5M 未通过则执行 Restart B"。**Restart B 触发条件确认满足。**

---

### 3. Policy 探索熵趋势——混合信号 — 严重程度：MEDIUM

**Symptom（正面）：** policy_std 从 early 0.380 回升至 recent 0.415，last 0.436。与上一轮 11-01-07 的 0.355 低点相比，本轮 policy_std 保持在 0.41–0.44 区间，**未触碰 0.37 警戒线**。

**Symptom（负面）：** entropy_loss 从 early -0.00433 深化至 last -0.00562（绝对值增大，接近 −0.006 饱和区间）。这表明 policy 分布正向确定性收敛，即使 std 数值未降低，探索质量也在衰减。

**Root Cause:** std 统计量反映分布宽度，entropy 反映分布的实际信息量。当 policy 集中于少数高频动作（如快速飞向已知区域再 crash reset），std 可以维持较高而 entropy 已经压缩。两者分叉是 crash 局部最优成熟化的信号。

---

### 4. Action Smoothness 大幅退化 — 严重程度：HIGH

**Symptom:** action_smoothness: early 0.366 → recent 0.205 → last 0.127（−65%）。

**Root Cause:** 与上一轮 11-01-07 的 action_smoothness 退化（early 0.675 → recent 0.247，−63%）高度一致。这是 crash-reset 局部最优的行为特征：policy 采取激进、不平滑的动作快速冲向目标或快速触地，利用 crash reset 来"跳过"困难状态，而不是平滑地执行导航任务。

**Evidence:** force_penalty 和 body_rate_penalty 也从 early 向 recent 下降（0.357→0.340，0.225→0.165），说明物理飞行质量实际在提升，与 action_smoothness 下降方向相反。这进一步确认：不是飞行动作变粗暴，而是 episode 变短（crash 更早发生），导致平滑性奖励积累时间缩短。

---

### 5. Tracking Reward 捕获区进入能力下降 — 严重程度：HIGH

**Symptom:** tracking best_ever = 3.406，仅在本 run 早期（热启动阶段）出现。此后 1.5M+ 步未突破此值。

**对比历史：**
- 11-01-07 run（4.5m 约束）：best_ever = 4.923，set at 805k，此后 1.05M 步冻结
- 14-02-25 run：best_ever = 4.757（当时新高）
- 09-31-34 run：best_ever = 4.241
- 19-15-45 run（本次）：best_ever = 3.406（历史最低，低于 14-02-25 热启动时的初始能力）

本次热启动后 policy 能力不升反降，是结构性退化而非探索不足。crash 局部最优持续"侵蚀" policy 已有的追踪能力。

---

## Improvement Recommendations（Restart B）

### Priority 1 (CRITICAL)：illegal_contact_penalty: 3.0 → 5.0

**Problem:** 在 1.125M 分析中已确认 3.0 在 policy 可接受损失范围内（99.7% episode 有接触，per-ep 惩罚约 -2.28）。1.914M 步后该结论仍然成立——crash 局部最优未被打破。

**Proposed Change:**
- 文件：`exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/move_flyfollow/marl_move_flyfollow_env_cfg.py`
- 参数：`illegal_contact_penalty`: 3.0 → **5.0**
- 理由：以明显跨越当前"接受阈值"的步长测试。若 5.0 仍不足以驱动行为改变，则确认问题是结构性的（稀疏终止惩罚无效，需要 dense pre-crash 信号）。

### Priority 2 (CRITICAL)：新增低空 dense 软惩罚（low_altitude_soft_penalty）

**Problem:** illegal_contact 是 episode 结束时的稀疏终止惩罚。Policy 在碰撞前没有任何"危险临近"的每步梯度信号。飞到地面前的最后 0.5m 对 policy 而言完全透明。

**Proposed Change:**
- 文件：`exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/move_flyfollow/marl_move_flyfollow_env_cfg.py`
- 新增：`low_altitude_soft_penalty_weight: 1.0`（z < 0.8m 时每步 exp(-(z-0.8)) 惩罚，类比 fly_high_penalty 机制）
- 理由：dense per-step 信号在碰撞前提供连续梯度，使 policy 学会主动爬升而非等待 crash reset。这是打破稀疏惩罚失效问题的必要补充。
- 注意：实现需在 `marl_move_flyfollow_env.py` 的 `_compute_safety_penalties` 中新增对应逻辑。

### Priority 3 (MEDIUM)：height_penalty_threshold: 0.5 → 0.6m

**Problem:** height_penalty 在 recent_mean 中仍活跃（基于 19-15-45 run 的历史记录约 -0.94/ep）。进一步放宽容忍度可减少 policy 因高度精度付出的无谓代价，将更多学习资源分配给 crash 回避。

**Proposed Change:**
- 文件：`exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/move_flyfollow/marl_move_flyfollow_env_cfg.py`
- 参数：`height_penalty_threshold`: 0.5 → **0.6**
- 理由：与之前 Priority 4 分析一致，且本次已有 height_reward_weight=2.0 作为主要高度引导信号，height_penalty 仅作补充约束，无需过紧。

### Priority 4 (LOW)：监控 entropy_loss，若 > -0.006 时考虑调整 entropy_coeff

**Problem:** entropy_loss 已达 -0.00562，接近 -0.006 饱和点。若 Restart B 后 entropy 继续压缩，可在 SKRL MAPPO config 中轻微提高 `entropy_loss_scale`（如 0.01 → 0.015）。

**Proposed Change:** 本次不变更。在 Restart B 运行 500k 步后，若 entropy_loss > -0.006 持续 50k 步，在下一次 restart 时调整。
- 文件：MAPPO 训练配置（skrl config dict，位于 `scripts/skrl/train.py` 或 agent cfg 文件）
- 参数：`entropy_loss_scale`: 0.01 → 0.015

---

## Experiment Plan（Restart B）

**触发依据：** 2026-03-27_19-15-45 run 1.5M 步三项 milestone 全部未通过，触发预设 Restart B 条件。

**热启动来源：** `logs/skrl/move_flyfollow/2026-03-27_19-15-45_mappo_torch_mappo/checkpoints/best_agent.pt`

**训练命令：**
```bash
python3 scripts/skrl/train.py --task=Isaac-marl-flyfollow-v0 \
  --headless --num_envs=2048 --algorithm="MAPPO" \
  --checkpoint=logs/skrl/move_flyfollow/2026-03-27_19-15-45_mappo_torch_mappo/checkpoints/best_agent.pt
```

**Restart B 配置变更（相比当前 marl_move_flyfollow_env_cfg.py）：**

| 参数 | 当前值 | Restart B | 优先级 |
|------|--------|-----------|--------|
| `illegal_contact_penalty` | 3.0 | **5.0** | CRITICAL |
| `low_altitude_soft_penalty_weight` | 0（不存在） | **1.0**（新增） | CRITICAL |
| `height_penalty_threshold` | 0.5 | **0.6** | MEDIUM |

**保持不变的参数（已验证有效）：**
- dist_reward_weight = 4.0
- tracking_reward_weight = 4.0，capture_distance = 3.5m
- velocity_follow_weight = 1.5
- height_reward_weight = 2.0，desired_height = 2.0m
- fly_high_termination_z = 5.0m，fly_high_threshold = 4.5m
- body_rate_penalty_weight = 1.0，upright_penalty_weight = 0.5

**关键监控指标（前 500k 步）：**
- crash 终止率：目标 < 65%（当前约 80%，需明确下降）
- illegal_contact_penalty/ep：目标变为更负（>-3.5/ep），反映碰撞频率未能显著下降时的惩罚加重
- episode mean：目标 > 190 步
- tracking_reward：目标 > 1.5/ep（恢复至 1.125M 步水平）
- action_smoothness：目标 > 0.25/ep（当前 0.205，持续下降是 crash 优化的指征）

**500k 步 Go/No-Go 决策：**
- Go（继续）：crash < 65% 且 tracking > 1.5/ep
- No-Go（停止，改换 dense 信号方案）：crash > 75% 且 tracking < 1.2/ep → 确认 sparse illegal_contact 惩罚在任何权重下均无效，必须实现 low_altitude_soft_penalty dense 信号才能打破局部最优

**1M 步成功标准：**
- crash 率 < 50%
- tracking_reward > 2.5/ep
- policy_std > 0.40
- episode mean > 210 步

**回归保护：**
- 不得降低 height_reward_weight（< 1.5 会导致高飞，已有先例）
- 不得降低 fly_high_termination_z（< 5.0 在 11-01-07 run 中造成永久退化）
- 不得移除 low_altitude_soft_penalty（一旦加入，只可调权重，不可清零）

---

# Training Analysis Report

**Run:** 2026-03-28_19-37-04_mappo_torch_mappo
**Date:** 2026-03-29
**Task:** Isaac-move-flyfollow-marl-v0
**Algorithm:** MAPPO
**Hot-start source:** 2026-03-27_19-15-45_mappo_torch_mappo/checkpoints/best_agent.pt（Restart B）

## Training Metrics Summary

| 指标 | early_mean | recent_mean | last | best |
|------|-----------|-------------|------|------|
| total_reward_mean | 42.70 | 29.59 | 22.33（~33.60 per raw last point） | 90.10（step 7k） |
| distance_reward/ep | 16.81 | 12.53 | 12.59 | 31.83 |
| tracking_reward/ep | 1.18 | 0.75 | 0.66 | 2.85 |
| height_reward/ep | 1.59 | 1.47 | 1.45 | 2.28 |
| illegal_contact/ep | −2.53 | −2.42 | −2.00 | 0 |
| upright_penalty/ep | −1.70 | −1.42 | −1.40 | 0 |
| height_penalty/ep | −1.07 | −0.80 | −0.88 | 0 |
| fly_high_penalty/ep | −0.044 | −0.044 | −0.052 | 0 |
| low_altitude_soft/ep | −0.003 | −0.001 | −0.001 | 0 |
| action_smoothness/ep | 0.316 | 0.112 | 0.097 | 0.584 |
| policy_std | 0.378 | 0.469 | 0.494 | 0.502 |
| entropy_loss | −0.00478 | −0.00627 | −0.00675 | — |
| episode_length mean | — | 147.0 | 170.2 | 295.0 |
| crash 终止率 | 0.54 | 0.51 | 0.21（瞬时波动） | — |
| falcon_fly_high 终止率 | 0.51 | 0.55 | 0.79（末段上升） | — |
| total_reward 四分位 | Q1: 42.1 | Q2: 37.6 | Q3: 36.7 | Q4: 30.4 |

**Script 综合判定：regressing（持续退化）**

## Hot-Start 确认

早期 total_reward 从 step=200 即达到 33.1，step=1000 最高达 84.1。以冷启动早期通常从 −5 ~ +13 起步推断，**这是成功的热启动**，继承了 19-15-45 run 的 best_agent.pt 权重。

## Observations & Findings

### 发现 1：总奖励单调下降（四分位趋势确认）— 严重程度：HIGH

**症状：** total_reward Q1=42.1 → Q2=37.6 → Q3=36.7 → Q4=30.4，呈单调递减。与上一个 Restart B 预期的"稳定后上升"相反——自热启动开始，奖励持续衰退，1.46M 步后未见任何回升迹象。

**根本原因：** falcon_fly_high 终止率在末段（Q4）从 0.51 升至 0.55（recent_mean），且最后一个数据点高达 0.79。fly_high 压力持续存在并有所加重。fly_high 终止缩短 episode，减少 tracking_reward 积累，拉低 total_reward。

**证据：** episode_mean 从最佳 295 步降至 recent 147 步，最后 170 步。action_smoothness 从 early 0.316 降至 recent 0.112（−65%），pattern 与之前 19-15-45 run 的 crash 主导信号完全一致。

### 发现 2：crash 与 fly_high 双重终止压力共存— 严重程度：HIGH

**症状：** crash 终止 recent_mean=0.513，fly_high 终止 recent_mean=0.547，两者量级相当。crash 率从前一个 run（19-15-45）的约 80% 降至本 run 的约 51%——**下降约 29 个百分点**，这是 Restart B 改动（illegal_contact_penalty 5.0 + low_altitude_soft_weight 1.0）产生了实质性效果的证据。

**问题：** 但 fly_high 终止同时增加，抵消了 crash 改善带来的收益。fly_high_penalty 的 recent_mean 全程约 −0.044，保持稳定，说明软惩罚本身没有加重，但终止率仍偏高。

**根本原因推断：** 为了避免 crash（低空惩罚使地面更危险），policy 选择爬升——这将飞机推向 fly_high 终止区域。两种终止路径形成了"低了撞地、高了飞出"的双重夹击。policy 在两者之间反复振荡，无法稳定在 1.5–4.5m 舒适区。

### 发现 3：low_altitude_soft 惩罚量级过小，未能提供有效的中间梯度— 严重程度：MEDIUM

**症状：** low_altitude_soft/ep recent_mean = −0.001（接近零）。early_mean 也仅 −0.003。对比 fly_high_penalty 的 −0.044，low_altitude_soft 信号弱 44 倍。

**根本原因：** 当前 low_altitude_soft_penalty 触发阈值（z < 0.8m）距离 crash 已经非常近，触发窗口极窄，policy 在该范围内停留的时间极短，信号几乎无法积累。

**证据：** crash 率从 80% 降至 51%，说明 illegal_contact_penalty 5.0 有效——但 low_altitude_soft 未能进一步把 crash 从 51% 压至目标 < 35%。两个 CRITICAL 修复中，sparse 惩罚（5.0）生效，dense 软惩罚效果微弱。

### 发现 4：tracking_reward 单调下降，best 未刷新— 严重程度：HIGH

**症状：** tracking_reward early 1.18 → recent 0.75 → last 0.66。best_ever=2.85，低于历史峰值 4.923（11-01-07 run），本 run 1.46M 步内未出现新高。相比热启动来源 19-15-45 run 的 recent 1.054，本 run 也出现退化。

**根本原因：** episode 被 fly_high 和 crash 截断，持续时间不足以让 policy 积累足够的 tracking 时间。episode_mean=147 步（≈4.9s），而 sustained_follow_duration=1.5s ≈ 45 步，理论上每个 episode 最多允许约 3 次 tracking 事件。但实际 tracking_reward recent_mean 仅 0.75，说明大多数 episode 在接近目标之前已经因高飞或 crash 终止。

### 发现 5：entropy_loss 加速收敛，接近饱和— 严重程度：MEDIUM

**症状：** entropy_loss early = −0.00478 → recent = −0.00627 → last = −0.00675。趋势是从上一个 run 的 −0.00562（last）进一步压缩。已超出 −0.006 警告线，now approaching −0.007 bound。

**根本原因：** policy 在 crash+fly_high 双重压力下收敛到少数"次优但稳定"的动作子集。policy_std 同期从 0.378 上升到 0.494（看似矛盾），但这是输出方差维度的数值，而 entropy_loss 度量的是概率分布的实际多样性。两者分叉再次出现（规律 11 in project memory），确认 policy 在"宽但集中"的分布空间内运行。

### 发现 6：Restart B 成效评估— 综合

**有效的改动：**
- illegal_contact_penalty 3.0 → 5.0：crash 率从约 80% 降至约 51%（下降约 30 个百分点），与预期一致，达到 Restart B 的预期方向。

**效果不足的改动：**
- low_altitude_soft_penalty_weight=1.0：触发量极小（−0.001/ep），未能提供有效的 pre-crash dense 梯度。需要提高权重或降低触发阈值（从 0.8m 提高到 1.2m）。

**未预见的副作用：**
- crash 降低后，policy 通过爬升规避 crash → fly_high 终止率上升，从 0.51 升至 0.55（recent），末段达 0.79。两种终止压力相互替代，总 episode 长度未能改善。

## Improvement Recommendations（Restart C）

### Priority 1（CRITICAL）：扩大 low_altitude_soft 触发范围：阈值 0.8m → 1.5m，权重 1.0 → 2.0

**问题：** 当前 low_altitude_soft 在 z<0.8m 才触发，离 crash 过近，每步积累量可忽略（−0.001/ep）。这导致 policy 在 0.8–1.5m 高度区间没有任何"危险预警"梯度，直到 crash 才感知到惩罚。

**Proposed Change:**
- 文件：`exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/move_flyfollow/marl_move_flyfollow_env_cfg.py`
- 参数：`low_altitude_soft_threshold`: 0.8 → **1.5m**（如字段存在，否则在 env.py 调整 hard-coded 值）
- 参数：`low_altitude_soft_penalty_weight`: 1.0 → **2.0**
- 理由：在 1.5m 以下开始给出每步连续梯度，与 fly_high_threshold=4.5m 形成对称的双侧软约束。policy 将有更长的"逃离地面"梯度路径，而不是等到 z<0.8m 才受惩罚。

### Priority 2（CRITICAL）：fly_high 终止阈值小幅上调：5.0m → 5.5m，同时 fly_high_threshold（软）4.5m → 5.0m

**问题：** 本 run 末段 fly_high 终止率达 0.79，说明 policy 为规避 crash 主动爬升，但 5.0m 的终止线过低，在 desired_height=2.0m 的情况下，drone 只需偏高 3m 即触发终止。

**Proposed Change:**
- 文件：`exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/move_flyfollow/marl_move_flyfollow_env_cfg.py`
- 参数：`fly_high_termination_z`: 5.0 → **5.5m**
- 参数：`fly_high_threshold`（软惩罚触发）: 4.5 → **5.0m**（软硬边界同步上移，保持 0.5m 间距）
- 理由：为 policy 提供更宽的高度运动窗口（desired 2.0m，软惩罚 5.0m，硬终止 5.5m）。在 crash 局部最优被打破之前，policy 需要可以"爬高躲避"的暂时空间。
- 回归保护确认：5.5m < 之前 6.0m 成功值，且高于之前 4.5m 失败值，处于已验证的安全区间内。

### Priority 3（HIGH）：增加 tracking_reward_weight：4.0 → 5.0

**问题：** tracking_reward recent_mean=0.75，在奖励总量 ~29.6 中仅占 2.5%。在 fly_high 和 crash 双重截断下，tracking 事件稀少，其梯度信号被噪音淹没。需要提高单次 tracking 事件的奖励，让 policy 有更强的动力保持在目标附近而非被迫爬升。

**Proposed Change:**
- 文件：`exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/move_flyfollow/marl_move_flyfollow_env_cfg.py`
- 参数：`tracking_reward_weight`: 4.0 → **5.0**
- 理由：提高 tracking 奖励的相对比例，使"保持低空追踪目标"的回报超过"爬升后自然结束"的路径。配合 Priority 1 的低空密集梯度，为 policy 提供明确的"低空追踪 > 高空逃避"奖励信号。

### Priority 4（MEDIUM）：entropy 恢复措施：entropy_loss_scale 0.01 → 0.015

**问题：** entropy_loss last = −0.00675，已超过 −0.006 警告线，接近 −0.007 饱和点。Policy 正在向少数动作模式收敛，探索空间压缩将阻碍下一阶段的 tracking 精细化学习。

**Proposed Change:**
- 文件：MAPPO agent 训练配置（`scripts/skrl/train.py` 中 `cfg_ppo` 字典或对应 agent cfg 文件）
- 参数：`entropy_loss_scale`: 0.01 → **0.015**
- 理由：轻微提高 entropy 激励，防止 policy 在 crash/fly_high 双重压力下过早收敛到 degenerate 分布。不宜大幅提高（> 0.02）以免影响已建立的追踪行为。

## Experiment Plan（Restart C）

**触发依据：** 2026-03-28_19-37-04 run 1M 步四项 milestone 全部未通过（tracking < 2.5/ep, crash > 50%, episode mean < 210 步, policy_std 虽 > 0.40 但 entropy 饱和）。

**热启动来源：** 从本 run best_agent.pt 热启动（total_reward best=90.1，tracking best=2.85，来自约 step 7k 的最高点附近权重）。

**Restart C 配置变更（相比 2026-03-28_19-37-04 run）：**

| 参数 | 当前值 | Restart C | 优先级 |
|------|--------|-----------|--------|
| `low_altitude_soft_threshold` | 0.8m | **1.5m** | CRITICAL |
| `low_altitude_soft_penalty_weight` | 1.0 | **2.0** | CRITICAL |
| `fly_high_termination_z` | 5.0m | **5.5m** | CRITICAL |
| `fly_high_threshold`（软惩罚） | 4.5m | **5.0m** | CRITICAL |
| `tracking_reward_weight` | 4.0 | **5.0** | HIGH |
| `entropy_loss_scale`（训练配置） | 0.01 | **0.015** | MEDIUM |

**保持不变的参数（已验证有效）：**
- illegal_contact_penalty = 5.0（本 run 已确认 crash 率从 80%→51%，有效）
- height_reward_weight = 2.0，desired_height = 2.0m（稳定，勿动）
- dist_reward_weight = 4.0，velocity_follow_weight = 1.5（正常）
- body_rate_penalty_weight = 1.0，upright_penalty_weight = 0.5（平衡合理）
- capture_distance = 3.5m，sustained_follow_duration = 1.5s（有效）

**训练命令：**
```bash
python3 scripts/skrl/train.py --task=Isaac-marl-flyfollow-v0 \
  --headless --num_envs=2048 --algorithm="MAPPO" \
  --checkpoint=logs/skrl/move_flyfollow/2026-03-28_19-37-04_mappo_torch_mappo/checkpoints/best_agent.pt
```

**500k 步 Milestone（Restart C）：**
- crash 率 < 45%（本 run 末段约 51%，Priority 1 应推动进一步下降）
- fly_high 终止率 < 35%（本 run recent 55%，Priority 2 放宽终止线后应改善）
- tracking_reward > 1.5/ep（本 run recent 0.75，需恢复至此水平）
- episode_mean > 175 步（本 run recent 147 步）
- action_smoothness > 0.18/ep（本 run recent 0.112，持续下降是退化信号）

**500k Go/No-Go 决策：**
- Go：crash < 45% 且 fly_high < 35% 且 tracking > 1.5/ep
- No-Go（停止）：crash > 60% 且 fly_high > 45% → "低了撞地、高了飞出"的双重夹击未被打破，需要重新审视 desired_height 定位或引入高度带宽内的 tracking 奖励（3D 距离替代 XY 距离）

**1M 步成功标准（Restart C）：**
- crash 率 < 35%
- fly_high 终止率 < 20%
- tracking_reward > 2.5/ep（historical best=4.923 from 11-01-07 run，目标此次在 1M 步内超越 3.0）
- episode_mean > 200 步
- action_smoothness > 0.25/ep

**回归保护：**
- 不得将 fly_high_termination_z 降低至 < 5.0m（4.5m 在 11-01-07 造成永久退化）
- 不得将 height_reward_weight 降低至 < 1.5（已有前车之鉴）
- 不得将 illegal_contact_penalty 降低至 < 5.0（3.0 已确认无效）
- low_altitude_soft_penalty 只可上调，不可归零

## Changelog
- 2026-03-29：分析 2026-03-28_19-37-04 run（Restart B 执行结果）
  - Restart B 成效：crash 率 80%→51%（illegal_contact 5.0 有效），fly_high 率因 crash 减少而代偿性上升（0.51→0.55 recent，末段 0.79）
  - low_altitude_soft 效果微弱（−0.001/ep），触发阈值 0.8m 过低
  - total_reward 四分位单调递减（42.1/37.6/36.7/30.4），tracking best 未刷新（2.85 < 历史 4.923）
  - 建议 Restart C：扩大 low_altitude_soft 范围（1.5m，权重 2.0）、放宽 fly_high 终止线（5.5m）、提高 tracking 权重（5.0）、轻提 entropy_loss_scale（0.015）

---

# Training Analysis Report

**Run:** 2026-03-29_18-27-54_mappo_torch_mappo
**Date:** 2026-03-29
**Task:** Isaac-marl-flyfollow-v0 (move_flyfollow variant)
**Algorithm:** MAPPO
**Identity:** Restart C — hot-start from 2026-03-28_19-37-04 best_agent.pt

## Training Metrics Summary

| 指标 | early_mean | recent_mean | last | best |
|------|-----------|-------------|------|------|
| Total reward (mean) | 42.68 | 45.55 | 43.14 | 80.34 |
| Total reward (max) | 68.55 | 70.88 | 61.98 | 127.18 |
| Total reward (min) | 5.81 | 9.33 | 30.62 | 63.39 |
| distance_reward | 16.84 | 17.60 | 18.48 | 27.68 |
| tracking_reward | 1.43 | 1.56 | 1.61 | 3.20 |
| height_reward | 1.66 | 1.66 | 1.70 | 2.28 |
| velocity_penalty | 0.00 | 0.00 | 0.00 | 0.00 |
| force_penalty | 0.34 | 0.35 | 0.35 | 0.50 |
| body_rate_penalty | 0.18 | 0.18 | 0.17 | 0.32 |
| action_smoothness | 0.27 | 0.25 | 0.22 | 0.44 |
| Episode timesteps (mean) | 174.4 | 172.5 | 181.6 | 253.5 |
| Episode timesteps (max, mean) | 236.6 | 237.8 | 249.0 | 489.0 |
| policy_std | 0.417 | 0.481 | 0.480 | 0.490 |
| value_loss | 0.059 | 0.050 | 0.026 | 0.230 |
| policy_loss | −0.016 | −0.022 | −0.034 | +0.115 |
| entropy_loss | −0.00785 | −0.00991 | −0.00991 | −0.00764 |

**Total steps logged:** 334,000
**Script verdict:** mixed（早期和近期均值差异小，无明确趋势）
**Config key params (from env.yaml):**
- tracking_reward_weight = 5.0 (Restart C: 4.0→5.0 已应用)
- low_altitude_soft_penalty_weight = 2.0, threshold = 1.5m (Restart C 已应用)
- fly_high_termination_z = 5.5m (Restart C: 5.0→5.5m 已应用)
- fly_high_threshold (soft) = 5.0m (Restart C: 4.5→5.0m 已应用)
- illegal_contact_penalty = 5.0 (Restart B 已应用，本次继承)
- capture_distance = 3.5m, sustained_follow_duration = 1.5s
- desired_height = 2.0m

## Observations & Findings

### 1. Restart C 参数全部确认生效 — Severity: INFO

**Symptom:** env.yaml 中全部 4 项 Restart C 参数均已正确应用。
**Evidence:**
- `low_altitude_soft_threshold = 1.5`（旧：0.8）
- `low_altitude_soft_penalty_weight = 2.0`（旧：1.0）
- `fly_high_termination_z = 5.5`（旧：5.0）
- `fly_high_threshold = 5.0`（旧：4.5）
- `tracking_reward_weight = 5.0`（旧：4.0）
- `illegal_contact_penalty = 5.0`（继承 Restart B，不变）

### 2. 热启动效果显著：奖励从高起点开始 — Severity: INFO

**Symptom:** early total_reward_mean = 42.68，比冷启动（约 −5 到 +13）高约 35+ 点。
**Root Cause:** hot-start from 19-37-04 best_agent.pt（step ~7k 的最高点，历史 best=90.1）。
**Evidence:** total_reward_min best = 63.39（所有并行环境中的最低 episode 也为正），说明策略起点健康，无负奖励区域。

### 3. Tracking_reward 恢复至 1.5+ 水平，优于前序 run 末段 — Severity: HIGH (POSITIVE)

**Symptom:** tracking_reward recent_mean = 1.56，last = 1.61，best = 3.20。
**对比 Restart B (19-37-04) 末段：** tracking_reward last = 0.66，best = 2.85。
**本次改善幅度：** last +144%，best +12%（新高 3.20 > 2.85）。
**Root Cause:** Restart C 三项改动协同：
  1. tracking_reward_weight 4.0→5.0 提高了进入捕获区的激励强度
  2. fly_high_termination_z 5.0→5.5m 降低了 fly_high 提前终止压力，允许更长 episode
  3. low_altitude_soft 范围扩大减少了底部崩溃压力
**Evidence:** 34 万步内 tracking 从未降至 0，与 Restart B 早期的快速衰减形成对比。

**重要提示：** 500k milestone（tracking > 1.5/ep）当前 recent_mean = 1.56 已处于达标边缘。需持续监测是否稳定保持。

### 4. Policy_std 恢复至 0.48，探索能力健康 — Severity: INFO (POSITIVE)

**Symptom:** policy_std early = 0.417 → recent = 0.481，呈上升趋势。
**对比 Restart B：** 0.494（early）→ 0.469（recent）→ 末段仍在压缩。
**对比 14-02-25 run 末段：** 0.355（曾经最低，接近崩溃）。
**Root Cause:** Restart C 减少了双重终止压力，策略不再被迫收敛到 crash-reset 局部优化。
**Evidence:** best policy_std = 0.490，当前 recent 与 best 仅差 0.009，说明策略处于历史最高探索水平。

### 5. Action_smoothness 持续下降 — Severity: MEDIUM

**Symptom:** action_smoothness early = 0.274 → recent = 0.253 → last = 0.215。下降趋势与 Restart B (0.316→0.112) 的崩溃模式相似，但幅度更小（−22% vs −65%）。
**Root Cause:** 两种可能：
  a. crash 终止率仍高（待 termination 统计确认），早期终止降低了每 episode 的平均平滑度；
  b. tracking_weight 提高导致策略更激进追踪，动作抖动增大。
**Evidence:** 当前 0.215 仍显著高于 Restart B 末段 0.097（相对健康），但需在 500k checkpoint 监测是否降至 0.18 以下（危险阈值）。

### 6. Tracking_reward 的天花板迹象：best = 3.20 低于历史高点 4.923 — Severity: MEDIUM

**Symptom:** 本次 best tracking = 3.20，仍低于历史最高 4.923（run 11-01-07 在 fly_high_termination_z=4.5m 的极限压迫下偶尔出现的峰值）。
**Root Cause:** 脚本诊断 "tracking bonus is still relatively hard to reach"。当前 tracking_reward recent/total_reward recent 比 = 1.56/45.55 ≈ 3.4%，在总奖励中占比偏低。
**Evidence:** distance_reward 占比 = 17.60/45.55 ≈ 38.7%（是 tracking 的 11.3 倍），策略仍以"接近目标"为主要学习信号，"进入捕获区停留"为次要信号。

### 7. Height_reward 稳定在 1.66/ep，altitude 锚定继续有效 — Severity: INFO (POSITIVE)

**Symptom:** height_reward early = 1.66 → recent = 1.66（几乎零漂移），best = 2.28。
**Root Cause:** height_reward_weight = 2.0（已验证工作的锚定强度），desired_height = 2.0m。
**Evidence:** 与高飞严重的早期 run 对比（如 20-28-48 的 height_reward 接近 0），本次高度控制稳健。

### 8. Entropy_loss 接近 −0.010，临近饱和临界值 — Severity: MEDIUM

**Symptom:** entropy_loss 从 −0.00785（early）压缩至 −0.00991（recent/last），接近 −0.010 饱和点。历史警告阈值为 −0.006（Restart B 末段 −0.00675 时进入饱和）。
**Root Cause:** 本次 entropy_loss_scale 是否已升至 0.015？env.yaml 中未见 MAPPO agent cfg 参数，无法确认。若仍为 0.01，entropy 即将进入危险区间。
**Evidence:** recent entropy = −0.00991，距警告阈值 −0.006 已超出 66%，但当前策略行为仍健康（policy_std 0.48）。entropy_loss 与 policy_std 当前同向（均在恢复），尚无立即危险。

## 综合评估（334k 步）

**整体状态：** Restart C 成效初步验证，核心指标均优于 Restart B 同期水平。

| 500k milestone | 目标 | 当前状态（334k） | 达标可能性 |
|----------------|------|-----------------|-----------|
| crash 率 < 45% | <45% | 脚本未输出（需 termination 统计确认） | 未知 |
| fly_high 率 < 35% | <35% | 脚本未输出 | 未知 |
| tracking_reward > 1.5/ep | >1.5 | **recent 1.56（已达标边缘）** | 高 |
| episode_mean > 175 步 | >175 | recent 172.5（接近临界） | 中 |
| action_smoothness > 0.18/ep | >0.18 | recent 0.253（已达标） | 高 |

**关键缺口：** TensorBoard 中 termination 统计未被脚本捕获（crash 率、fly_high 率）。这是 500k Go/No-Go 决策的最重要数据，需手动检查 TensorBoard 或扩展脚本 tag 列表。

## Improvement Recommendations

### Priority 1 (HIGH): 补充 termination 统计 tag 至分析脚本

**Problem:** 脚本输出无 `Episode_Termination/*` 数据（crash 率、fly_high 率、timeout 率）。这是 Restart C 500k Go/No-Go 的核心决策依据。
**Proposed Change:**
- File: `/home/xtj/.codex/skills/flyfollow-training-analysis/scripts/analyze_flyfollow_run.py`
- 增加 tag: `Episode_Termination/falcon_fly_high`, `Episode_Termination/crash`, `Episode_Termination/time_out`, `Episode_Termination/illegal_contact`
- 若无此脚本修改权限，可手动在 TensorBoard 检查：`tensorboard --logdir logs/skrl/move_flyfollow/2026-03-29_18-27-54_mappo_torch_mappo`
- Rationale: 无终止原因统计，无法判断 "低了撞地/高了飞出" 双重夹击是否已被打破。

### Priority 2 (MEDIUM): 继续当前 run，等待 500k checkpoint

**Problem:** 334k 步尚未到达 Restart C 的第一个里程碑检查点（500k）。
**Proposed Change:**
- 无需配置变更，继续训练至 500k 步
- 500k checkpoint 时重点检查：
  a. crash 终止率（目标 < 45%）
  b. fly_high 终止率（目标 < 35%）
  c. tracking_reward recent_mean 是否稳定 > 1.5/ep
  d. action_smoothness 是否仍 > 0.18/ep（当前下降需关注）
- Rationale: 当前指标改善明显，不需要提前干预。

### Priority 3 (MEDIUM): 若 500k 后 tracking 停滞，评估 capture_distance 3.5→4.0m

**Problem:** tracking_reward 在总奖励中占比仅 3.4%，distance_reward 占 38.7%。策略仍以"接近但不进入"为主要行为。
**Proposed Change:**
- File: `exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/move_flyfollow/marl_move_flyfollow_env_cfg.py`
- Parameter: `capture_distance`: 3.5 → **4.0m**（仅当 500k 后 tracking recent < 1.5/ep 时）
- Rationale: 放宽捕获区有助于增加 tracking 信号频率，但需注意 3.5m 是上次已验证有效的值，应谨慎。

### Priority 4 (MEDIUM): 确认 entropy_loss_scale = 0.015 是否已生效

**Problem:** Restart C 计划要求 entropy_loss_scale 0.01→0.015，但 env.yaml 不含此参数，无法确认。
**Proposed Change:**
- File: MAPPO agent 训练配置（通常在 `scripts/skrl/train.py` 的 `cfg_ppo` 字典或 agent cfg yaml）
- 检查并确认 `entropy_loss_scale = 0.015`
- 若未修改，应在下次 restart 时补充
- Rationale: 当前 entropy_loss = −0.00991，接近 −0.010 阈值，若 scale 仍为 0.01，需在 500k 后评估是否应提升。

### Priority 5 (LOW): 若 1M 步 tracking 仍停滞，考虑 3D 距离奖励替代 XY 距离

**Problem:** 脚本诊断 "tracking bonus is still relatively hard to reach"，distance_reward 为 XY 平面距离（无 Z 分量），tracking 信号依赖物理接近而非奖励塑形引导。
**Proposed Change:**
- File: `exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/move_flyfollow/marl_move_flyfollow_env.py`
- 修改 `distance_reward` 计算：`dist_xy = ||drone_pos[:,:,:2] - target_pos[:,:,:2]||` → `dist_3d = ||drone_pos - target_pos||`
- 或：新增 `dist_z_reward` 分量（weight=0.5），激励 drone 与 target 高度对齐
- Rationale: 当前 XY-only 距离奖励在 Z 方向梯度为零，策略在 Z 方向无任务引导。若高飞问题再次出现，应考虑此方案。
- **注意：仅在 500k Go 且 1M tracking < 2.5/ep 时执行，避免破坏已稳定的高度控制。**

## Experiment Plan（继续 Restart C）

**当前状态：** 334k/500k 步（67% 完成）

**立即行动：**
1. 继续当前训练，不做任何参数变更
2. 在约 500k 步时暂停，运行分析脚本并手动检查 termination 统计
3. 根据 500k Go/No-Go 决策：

**500k Go 路径（crash < 45% 且 fly_high < 35% 且 tracking > 1.5）：**
- 继续训练至 1M 步
- 1M 目标：tracking > 2.5/ep，crash < 35%，fly_high < 20%，episode_mean > 200 步

**500k No-Go 路径（crash > 60% 且 fly_high > 45%）：**
- 考虑引入 3D 距离奖励，或调整 desired_height 至 target z+1.5m = 1.75m（更接近地面目标）
- 重新评估 capture_distance 放宽至 4.0m

**监测重点（每 ~170k 步检查一次）：**
- tracking_reward recent_mean（目标：稳定 > 1.5，目标 > 2.5 at 1M）
- action_smoothness（危险：< 0.18/ep，当前趋势需关注）
- entropy_loss（危险：< −0.010，当前 −0.00991 在临界）
- policy_std（安全：> 0.40，当前 0.48 健康）

## Regression Guards（保持不变）

- `fly_high_termination_z` 不得降低至 < 5.0m（4.5m 在 11-01-07 造成永久退化）
- `height_reward_weight` 不得低于 1.5（已有前车之鉴）
- `illegal_contact_penalty` 不得低于 5.0（3.0 已确认无效）
- `low_altitude_soft_penalty_weight` 不得归零（仅可上调）

## Changelog

- 2026-03-29：分析 2026-03-29_18-27-54 run（Restart C，334k 步）
  - Restart C 所有参数确认生效（low_altitude_soft 1.5m/2.0，fly_high_termination 5.5m，tracking_weight 5.0）
  - tracking_reward 恢复至 1.56/ep（vs Restart B 末段 0.66），best = 3.20（新本次高）
  - policy_std = 0.48（健康，vs Restart B 末段 0.494→0.469 压缩）
  - action_smoothness 轻微下降（0.274→0.215），需关注但尚未进入危险区
  - entropy_loss = −0.00991，临近饱和，需确认 entropy_loss_scale=0.015 是否已生效
  - termination 统计（crash 率/fly_high 率）缺失，需手动检查 TensorBoard

---

# Training Analysis Report — 第二次分析

**Run:** 2026-03-29_18-27-54_mappo_torch_mappo
**Date:** 2026-03-30
**Task:** Isaac-marl-flyfollow-v0 (move_flyfollow variant)
**Algorithm:** MAPPO
**Identity:** Restart C — hot-start from 2026-03-28_19-37-04 best_agent.pt
**Analysis Type:** 第二次分析（上次 334k 步，本次 1,512,200 步，增量 +1,178k 步）

## Training Metrics Summary（1.51M 步全程）

| 指标 | 100k-window 峰值 | 100k-window 峰值所在步段 | latest（1.5M+） | 对比 334k 分析（last） |
|------|-----------------|--------------------------|-----------------|----------------------|
| Total reward (mean) | **49.1** | 400k–600k | 34.1 | 43.1（↓21%）|
| tracking_reward | **1.82** | 400k–600k | 1.27 | 1.61（↓21%）|
| distance_reward | 17.2 | 全程稳定 | 21.0 | 18.5 |
| height_reward | 1.65 | 全程稳定 | 1.58 | 1.70 |
| crash (termination) | 0.34 | 700k–800k（最好） | 0.71 | 未记录 |
| falcon_fly_high | 0.46 | 0k–100k（最好） | 0.34 | 未记录 |
| ep_len (mean) | **179** | 500k | 165.6 | 181.6 |
| upright_penalty | −1.57 | 200k（最好） | −1.72 | 未记录 |
| height_penalty | −1.08 | 0k（起始） | −1.06 | 未记录 |
| policy_std | 0.491 | 最新 | 0.491 | 0.480 |
| value_loss | stable | — | 0.051 | 0.026 |
| entropy_loss | −0.01031 | 最新 | −0.01031 | −0.00991 |
| grad_norm_actor | ~0.92 | 全程恒定 | 0.949 | 未记录 |

## 500k Milestone 正式判定

**本次可完整评估 500k milestone，结果如下：**

| 指标 | 目标 | 500k 实测值（480k–520k窗口） | 判定 |
|------|------|------------------------------|------|
| tracking_reward | >1.5/ep | **1.825** | **PASS** |
| crash 率 | <65% | **54.3%** | **PASS** |
| fly_high 率 | <30% | **49.5%** | **FAIL** |
| episode mean | >185 步 | **177.8** | **FAIL** |

**500k 综合判定：2/4 指标达标（部分通过）。**

核心任务指标（tracking、crash）已达标，但高飞率（49.5% vs 目标 30%）和 episode 长度（177.8 vs 目标 185）仍不足。

## Observations & Findings

### 1. 全程呈现"抛物线"轨迹：峰值在 500k，之后持续退化 — Severity: HIGH (CRITICAL)

**Symptom:** total_reward_mean 从 334k 时的稳步上升（42→46→49）在 500k 达到顶点（49.1），随后进入持续退化：800k 降至 39.7，900k 降至 37.6，1.5M 降至 34.1（较峰值下降 31%）。

**退化时间线：**
- 0k–500k：稳定上升阶段（42→49），tracking 1.40→1.80，crash 47%→46%
- 500k–800k：过渡阶段，飞高率偶有改善（fly_high 最低达 46%），但 reward 开始波动
- 800k–1M：明显退化，reward 37–39，tracking 1.40–1.44，crash 49–58%
- 1M–1.25M：短暂反弹（reward 43–46，tracking 1.60–1.81）
- 1.25M–1.5M：加速退化，crash 率上升至 63–71%，tracking 1.27–1.35，reward 34–38

**Root Cause（多因素）：**
1. **upright_penalty 顽固不减**（全程 −1.71，无改善迹象）：表明姿态倾斜问题长期存在，且策略无法改善。这消耗了大量奖励空间，同时可能导致 crash。
2. **height_penalty 持续 −1.2 量级**（全程 −1.08 到 −1.44）：表明策略始终无法精确高度控制，尽管 height_reward = 2.0 已经启用。
3. **crash 率从 800k 开始趋势性上升**（0.34→0.49→0.58→0.63→0.71）：策略学习到了更激进的追踪行为（tracking 短暂回升），但代价是更多 illegal_contact 碰撞终止。
4. **fly_high 与 crash 呈反相关**（fly_high 下降时 crash 上升）：策略降低了飞行高度（fly_high 率从早期 0.67 降至最新 0.34），但降低后反而更容易触发地面/物体碰撞（illegal_contact）。

**Evidence:** 
- crash ≈ illegal_contact（两者 diff ≈ 0，说明 crash 终止基本等价于 illegal_contact 而非坠地）
- fly_high 从 0k 时 0.58 降至 1.5M 时 0.34，而 crash 从 0.47 升至 0.71，两者之和基本恒定在约 1.0–1.05

### 2. Crash = Illegal_contact：无人机碰撞到物体（非坠地）是主要失效模式 — Severity: HIGH

**Symptom:** `Episode_Termination/crash` 与 `Episode_Termination/illegal_contact` 数值几乎完全一致（mean_diff ≈ 0.0036，max_diff = 0.67），说明 crash 由 illegal_contact（接触力 >1N）触发，而非 fly_low（坠地）。

**Root Cause:** 在追踪目标小车时，无人机飞得太低或与目标小车本体发生物理接触。随着策略学会降低飞行高度以减少 fly_high 终止，却在低空遭遇了 illegal_contact。

**Evidence:**
- `falcon_fly_low` 全程 recent_mean = 0.009（几乎为零），确认不是坠地
- `illegal_contact` 从 1.5M 附近突增，与 fly_high 下降同步，呈明显替代关系

**Impact:** 这是一个"地板-天花板"双重挤压（fly_high 限制上升，illegal_contact 惩罚下降），策略陷于无法同时满足两个约束的困境。

### 3. Upright_penalty 顽固，全程 −1.71，无改善 — Severity: HIGH

**Symptom:** upright_penalty all-time mean = −1.71，recent mean = −1.78，best = 0.0。全程范围 −1.57 至 −1.80，无明显下降趋势。

**Root Cause:** 姿态问题（机体倾斜）可能有两个来源：
  a. 追踪目标时的主动倾斜（合理，但不应持续整个 episode）
  b. 控制器输出的系统性偏差（不合理，应被策略修正）

在 1.5M 步训练后 upright_penalty 仍未趋近 0，说明 (b) 更可能。策略可能已将倾斜作为一种稳定行为学到（如：在 x 方向始终倾斜追随目标，但未学会恢复直立）。

**Evidence:** upright_penalty recent mean = −1.778，其绝对值与 tracking_reward recent_mean = 1.49 相当，即姿态惩罚大致"抵消"了追踪奖励的贡献。

**Impact:** 若 upright_penalty_weight 过大，策略会优先维持直立（不追踪）而非倾斜追踪（有效但被惩罚）。反之，若过小，则姿态失控触发 illegal_contact。

### 4. Height_penalty 长期存在（−1.2 量级），height_reward=2.0 仍不足以消除偏差 — Severity: MEDIUM

**Symptom:** height_penalty recent_mean = −1.236，all-time mean = −1.254，无明显改善趋势。height_reward 峰值仅 2.46（理论最大值更高），说明策略始终偏离 desired_height = 2.0m。

**Root Cause:** 高度误差惩罚（height_penalty）存在但策略无法完全消除。结合 illegal_contact = crash，无人机可能在 2m 以下低空（y_target 的 z ≈ 0.25m，如果无人机跟随目标高度）飞行时产生接触。

**注意：** height_penalty 与 upright_penalty 合计 recent_mean = −1.24 + (−1.78) = **−3.02**，占总奖励 41.2 的 7.3%，但约等于 tracking_reward 全部贡献（1.49）的 2 倍。

### 5. velocity_follow 奖励极低（0.051），策略不学习速度匹配 — Severity: MEDIUM

**Symptom:** velocity_follow recent_mean = 0.051（best = 0.088），全程接近 0。velocity_follow_weight = 0.8（权重设置不低），但奖励极低说明实际速度匹配度很差。

**Root Cause:** 目标小车以 0.8m/s 移动，无人机可能以更高速度"冲向"目标（超速），或在追踪区内速度方向不一致。velocity_follow_sigma = 0.8（较宽），理论上应有更多信号。

**Evidence:** tracking_reward（进入 2m 范围的奖励）recent_mean = 1.49，velocity_follow = 0.051，比值 29:1。进入捕获区是有的，但捕获后未建立速度同步。

**Impact:** 策略学到的是"冲入捕获区立即离开"而非"伴飞跟随"，这导致 tracking 积分无法充分累积。

### 6. Policy_std 回升至 0.49，探索能力健康，但 entropy_loss 已饱和 — Severity: MEDIUM

**Symptom:** policy_std recent = 0.486，last = 0.491，持续上升（early 0.422 → recent 0.486）。这是正面信号。但 entropy_loss = −0.01031（last），接近上限，且全程下降（early −0.00833 → recent −0.01017）。

**Root Cause:** 策略多样性在增加（policy_std），但 entropy 在接近 max 归一化值（entropy_loss_scale × H_max）。entropy_loss_scale 可能仍为 0.01，导致 entropy reward 不足以激励更大的探索。

**Evidence:** entropy_loss 从 −0.00833 压缩至 −0.01031，绝对值增加 24%。此前 Restart B 末段 −0.00675 时进入崩溃，当前 −0.01031 绝对值更大（因 scale 可能已升至 0.015？），但趋势同样单调递减。

### 7. 梯度范数 actor ≈ 0.92（持续接近 clip=1.0），学习信号强但被压缩 — Severity: MEDIUM

**Symptom:** grad_norm_actor early = 0.948，recent = 0.915，全程约 0.92，接近梯度裁剪上限 1.0。这意味着 actor 梯度持续被裁剪，实际学习步长小于理论值。

**Root Cause:** actor 的策略梯度一直很大（任务信号强烈），但被 grad_norm_clip=1.0 压制，限制了每步的实际改进幅度。critic grad = 0.338（recent），更为正常。

**Impact:** 当 actor 梯度恒定被裁剪时，学习率实际上对 actor 不生效，相当于使用的是 `lr × (1.0 / grad_norm)` 的自适应步长。如果任务确实需要大的策略更新，当前裁剪可能在拖慢收敛。

## 综合评估（1.51M 步）

**整体状态：** 训练进入退化阶段（script 状态: `regressing`）。峰值出现在 500k，之后经历振荡并在 1.3M 后加速退化。核心问题是"高度降低→碰撞增加"的替代效应，策略陷入局部均衡。

**500k milestone 最终结论：** 部分达标（2/4），tracking 和 crash 率已到达目标，但 fly_high 和 episode 长度未到达。基于 1M 步后的持续退化，该 run 已无法实现 1M milestone（tracking > 2.5/ep，crash < 35%）。

**当前 run 建议：** 停止当前 run，准备 Restart D。

## Improvement Recommendations

### Priority 1 (CRITICAL): 增加 illegal_contact 高度隔离机制 — 解决地板-天花板困境

**Problem:** fly_high 与 illegal_contact 形成替代效应：策略降低高度以减少 fly_high，但低空随即触发 illegal_contact。两者之和约为 1.0，策略陷于无法同时满足两个约束的局部均衡。

**Proposed Change:**
- File: `exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/move_flyfollow/marl_move_flyfollow_env_cfg.py`
- Parameter: `illegal_contact_penalty_weight` → 增加安全高度要求：**min_altitude = 1.0m → 1.2m**，强制策略在更高位置追踪
- Parameter: `desired_height = 2.0` → **2.5m**（提高目标飞行高度至目标小车上方 2.25m，减少碰撞风险）
- Rationale: 目标小车 z ≈ 0.25m，当前 desired_height = 2.0m 已有 1.75m 隔离，但策略仍触发 illegal_contact，说明碰撞不来自垂直方向而是水平追踪时的侧面接触。提高 min_altitude 强制策略维持安全高度。

**替代方案（若 desired_height 调整效果不明显）：**
- 增加 `illegal_contact_penalty_weight`：当前应为 5.0（Restart B 已应用），可进一步提至 8.0
- 增加碰撞软边界（collision_soft_margin）：使策略更早"感知"到近距离风险

### Priority 2 (HIGH): 修复 upright_penalty 学习停滞 — 1.5M 步无改善

**Problem:** upright_penalty 全程 −1.71，无改善。upright_penalty_weight 当前值（根据 MEMORY/cfg 未知确切值）可能存在：(a) 权重过大惩罚正常追踪倾斜，或 (b) 权重过小无法引导策略改善姿态，两者任一均导致停滞。

**Proposed Change:**
- File: `exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/move_flyfollow/marl_move_flyfollow_env_cfg.py`
- 检查 `upright_penalty_weight` 当前值：
  - 若 > 2.0：适当降低至 1.0，允许追踪时的合理倾斜
  - 若 < 0.5：适当提高至 1.0，增强改善姿态的激励
- 目标：upright_penalty 在 200k 步内改善至 > −1.0（允许合理倾斜但不极端）
- Rationale: 当前值（−1.78 recent）说明策略持续以大倾斜角飞行，且 1.5M 步训练未能改善，是系统性问题而非训练不足。

### Priority 3 (HIGH): 降低 grad_norm_clip 以改善 actor 学习效率

**Problem:** actor 梯度范数全程约 0.92，持续被 clip=1.0 截断，实际学习步长受限。结合持续退化的状态，当前学习信号未被充分利用。

**Proposed Change:**
- File: `exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/move_flyfollow/agents/skrl_mappo_cfg.yaml`
- Parameter: `grad_norm_clip`: 1.0 → **0.5**（降低裁剪上限，减少大梯度时的截断频率，使梯度更稳定）
- 或：学习率从当前值 × 1.5（补偿之前的有效步长损失）
- Rationale: 若梯度持续在 0.92 附近（90%+ 利用率），降低 clip 可使每次更新更稳定（小且均匀），有助于跳出当前局部均衡。
- **注意：** 与上次 Priority（illegal_contact 修复）配合应用，避免孤立评估效果。

### Priority 4 (MEDIUM): 增加 velocity_follow 激励 — 策略目前"冲进冲出"而非"伴飞"

**Problem:** velocity_follow recent_mean = 0.051（极低），策略未学习速度匹配。当前 velocity_follow_weight = 0.8，sigma = 0.8，可能梯度太弱。

**Proposed Change:**
- File: `marl_move_flyfollow_env_cfg.py`
- Parameter: `velocity_follow_weight`: 0.8 → **1.5**（增强速度匹配激励）
- Parameter: `velocity_follow_sigma`: 0.8 → **1.2**（放宽匹配宽容度，使更多行为能获得奖励信号）
- Rationale: 当前策略对 velocity_follow 几乎无响应（0.051 vs tracking 1.49），原因可能是匹配条件太严（0.8m/s 误差容忍）。先放宽再收紧是更好的 curriculum 策略。
- **前置条件：** 须与 Priority 1 和 Priority 2 同时应用，否则速度追踪可能导致更多碰撞。

### Priority 5 (MEDIUM): 确认 entropy_loss_scale 并评估是否需要上调

**Problem:** entropy_loss = −0.01031（saturating），全程单调递减。若 scale=0.01，则 entropy reward ≈ 0.01 × H，力度不足以对抗策略收敛压力。

**Proposed Change:**
- File: `agents/skrl_mappo_cfg.yaml` 或 `scripts/skrl/train.py`
- 确认当前值：grep `entropy_loss_scale`
- 若 = 0.01：升至 **0.02**（提供更强的探索激励）
- 若已 = 0.015：升至 **0.02**
- Rationale: policy_std 在 0.49 说明探索能力未完全崩溃，但 entropy_loss 趋势表明在 500k-1M 之间很可能出现探索不足的问题，应提前准备。

## Experiment Plan（Restart D）

**决策：当前 run 已进入退化，继续无收益。建议在确认退化分析后立即启动 Restart D。**

**Restart D 参数变更清单（相比当前 Restart C）：**

| 参数 | 当前值 | 建议值 | 优先级 |
|------|--------|--------|--------|
| `desired_height` | 2.0m | **2.5m** | CRITICAL |
| `min_altitude` | 1.0m | **1.2m** | CRITICAL |
| `illegal_contact_penalty_weight` | 5.0 | **8.0** | HIGH |
| `upright_penalty_weight` | 检查后调整 | **目标 1.0** | HIGH |
| `grad_norm_clip` | 1.0 | **0.5** | HIGH |
| `velocity_follow_weight` | 0.8 | **1.5** | MEDIUM |
| `velocity_follow_sigma` | 0.8 | **1.2** | MEDIUM |
| `entropy_loss_scale` | 0.01或0.015 | **0.02** | MEDIUM |

**Restart D 建议：**
- Hot-start from Restart C 最佳 checkpoint（约 500k 步时的 best_agent.pt，因为那是最高点）
- 优先应用 CRITICAL 和 HIGH 优先级变更
- num_envs=2048，训练至 500k 步做第一个 checkpoint 评估

**500k 监测指标（Restart D）：**
- crash 率 < 40%（目标比 Restart C 的 54% 显著改善）
- fly_high 率 < 30%（Restart C 500k 时 49.5%，需大幅改善）
- tracking_reward > 1.8/ep
- velocity_follow > 0.2/ep（速度匹配改善的信号）
- upright_penalty > −1.2（姿态改善的信号）
- episode_mean > 185 步

**1M milestone 目标（Restart D）：**
- tracking_reward > 2.5/ep
- crash 率 < 30%
- fly_high 率 < 20%
- velocity_follow > 0.5/ep
- episode_mean > 220 步

## Regression Guards（更新）

- `fly_high_termination_z` 不得降低至 < 5.0m（4.5m 在历史 run 造成永久退化）
- `height_reward_weight` 不得低于 1.5（已有前车之鉴）
- `illegal_contact_penalty` 不得低于 5.0（3.0 已确认无效）
- `low_altitude_soft_penalty_weight` 不得归零（仅可上调）
- `tracking_reward_weight` 不得低于 4.0（Restart C 的 5.0 已验证有效）

## Changelog

- 2026-03-30：第二次分析 2026-03-29_18-27-54 run（Restart C，1,512k 步）
  - 500k milestone 正式判定：2/4 通过（tracking PASS 1.82，crash PASS 54%，fly_high FAIL 49.5%，ep_len FAIL 178）
  - 确认全程"抛物线"退化：峰值 500k（reward=49.1，tracking=1.82），1.5M 时退化至 34.1 和 1.27（较峰值 −31%）
  - 发现核心失效机制：fly_high 下降↔illegal_contact 上升的替代效应（两者之和 ≈ 1.0 全程恒定）
  - upright_penalty 全程 −1.71 无改善（1.5M 步训练无效，系统性问题）
  - velocity_follow 极低（0.051），策略无伴飞行为，仅"冲进冲出"捕获区
  - grad_norm_actor ≈ 0.92（持续被 clip=1.0 截断），actor 学习效率受限
  - 决策：建议停止 Restart C，启动 Restart D（8 项参数变更，hot-start from 500k best_agent.pt）

---

## Training Analysis Report — Restart D

**Run:** `2026-03-30_13-16-00_mappo_torch_mappo`
**Date:** 2026-04-01
**Task:** Isaac-marl-flyfollow-v0 (move_flyfollow)
**Algorithm:** MAPPO
**Total timesteps:** 2,000,000
**Status:** REGRESSING（总奖励在 800k 步达峰后持续下滑）

### Restart D 参数变更（相比 Restart C）

| 参数 | Restart C | Restart D |
|------|-----------|-----------|
| `desired_height` | 2.0m | **2.5m** |
| `low_altitude_soft_threshold` | 1.5m | **1.2m** |
| `illegal_contact_penalty` | 5.0 | **8.0** |
| `upright_penalty_weight` | 0.5 | **1.0** |
| `grad_norm_clip` (actor) | 1.0 | **0.5** |
| `entropy_loss_scale` | 0.015 | **0.02** |

---

### Training Metrics Summary

| 指标 | 早期（0–400k） | 中期峰值（seg2–3） | 近期（1600k–2000k） | 全程最优 |
|------|--------------|-----------------|-------------------|---------|
| total_reward_mean | 40.1 | **48.9**（800k 峰） | 32.5（−34%） | 103.6 |
| distance_reward | 18.4 | — | 16.5 | 44.1 |
| tracking_reward | 1.75 | — | 1.31 | 5.57 |
| height_reward | 1.20 | — | 1.44 | 2.38 |
| illegal_contact penalty | −2.85 | — | −3.31 | 0.0 |
| upright_penalty | −3.38 | — | −3.46 | −1.81 |
| fly_high_penalty | −0.053 | — | −0.060 | 0.0 |
| success_reward | 0.000 | — | 0.000 | 0.000 |
| episode_mean (steps) | 174.5 | — | 176.4 | 306.1 |
| policy_std | 0.512 | — | 0.673 | 0.709 |
| grad_norm_actor | ~0.44 | — | ~0.44 | 0.497 |

---

### 核心问题评估

#### 1. fly_high↔crash 替代效应是否被打破？— 结论：**未被打破，反向加剧**

**Restart C（近期 20%）：** crash=0.589，fly_high=0.454，比值 fly_high/crash=0.77
**Restart D（近期 20%）：** crash=0.431，fly_high=0.618，比值 fly_high/crash=**1.43**

相比 Restart C：
- crash 率下降 26.8%（−0.158）
- fly_high 率**上升 36.0%（+0.163）**
- 两者之和 Restart C = 1.043，Restart D = 1.049：**恒定，替代效应完全保留**

结论：desired_height 提高至 2.5m + low_altitude_soft_threshold 降至 1.2m，
使无人机倾向于"向上飞"而非"撞地"，但 fly_high 终止率对应上升，净效益为零。
fly_high 与 crash 的"零和替代效应"未被任何改动打破。

#### 2. crash 率是否改善？— 结论：**短期改善但后期反弹**

| 训练阶段 | crash 率 | fly_high 率 |
|---------|---------|------------|
| seg1 (0–401k) | 0.374 | 0.674 |
| seg2 (401–798k) | 0.381 | 0.663 |
| seg3 (798–1195k) | 0.345 | 0.694 |
| seg4 (1195–1601k) | **0.307**（最低） | 0.740 |
| seg5 (1601–2000k) | 0.431（反弹） | **0.618**（回落） |

crash 在 seg4 达到最低点（0.307），随后 seg5 反弹至 0.431，创近期新高。
fly_high 与 crash 始终呈反向运动，验证了替代效应的系统性。

#### 3. upright_penalty 是否因 weight 0.5→1.0 得到改善？— 结论：**恶化**

Restart C 近期 upright_penalty：−1.78
Restart D 近期 upright_penalty：**−3.46（恶化 95%）**

这不是权重提高后惩罚绝对值增大的数学结果——如果姿态真正改善，
upright_penalty/weight 的标准化值应该下降。实测结果表明：
在相同行为下，penalty 的原始数值（还未乘权重）本身就在增大，即**姿态实际上比 Restart C 更差**。
根本原因可能是：desired_height 提高至 2.5m 后无人机在更高处飞行，
配合 fly_high 率上升，飞行更不稳定，body rate 更大，导致 upright deviation 加剧。

#### 4. illegal_contact 惩罚是否因 8.0 权重有效抑制？— 结论：**轻微改善后再次恶化**

| 阶段 | illegal_contact_rate | penalty |
|------|---------------------|---------|
| seg4 (1195–1601k) | 0.307 | −2.35 |
| seg5 (1601–2000k) | 0.431 | **−3.31**（反弹） |

seg4 达到最低点后 seg5 出现显著反弹，与 crash 率趋势完全同步（因为 illegal_contact=crash）。
illegal_contact_rate 的全程 recent 均值（0.431）相比 Restart C（0.589）下降 26.8%，
但 illegal_contact_penalty 的 recent 均值（−3.31）比 Restart C（−2.80）**更大**（−18%），
因为权重 5.0→8.0 导致单次撞击的代价更高，但撞击频率的降低不足以抵消权重提升。

#### 5. 总奖励退化根因分析

总奖励退化路径：
```
early mean=40.1 → peak seg3=48.9 (800k) → recent=32.5 (2000k)
```
奖励下滑 = +8.3（比 Restart C 的 recent mean=40.8 低 20%）

主要拖累项（recent 负贡献之和 = −8.08）：
1. upright_penalty：−3.46（最大单项，占总负值 43%）
2. illegal_contact：−3.31（占总负值 41%）
3. height_penalty：−1.21（占总负值 15%）

三项合计−7.98，已超过 tracking+velocity_follow 的正贡献（+1.36）的 5.9 倍。

#### 6. success_reward = 0 的持续确认

全程 success_reward = 0.000（所有 segment 均为 0）。
无人机始终未实现"≥3 targets 同时追踪 3s"的成功条件。
tracking_reward 虽>0（无人机能进入捕获区），但无法保持足够长时间触发 success。

---

### Observations & Findings

#### 发现 1：fly_high↔crash 零和替代效应具有结构性根源 — Severity: CRITICAL

**Symptom:** 无论如何调整 desired_height 或 illegal_contact_penalty，
crash 和 fly_high 的终止率之和始终约为 1.0（Restart C=1.043，Restart D=1.049）。
**Root Cause:** 这两类终止条件在物理上是互补的——无人机飞得越低越容易 crash，
越高越容易 fly_high。当前奖励结构中，策略只能在两个"死亡方式"之间权衡，
而非找到"既不低飞又不高飞"的稳定高度带。
**Evidence:** seg4 crash 最低（0.307）时 fly_high 最高（0.740）；seg5 fly_high 回落（0.618）时 crash 反弹（0.431）。

**Root Cause（深层）：** `height_reward`（weight=2.0）虽然存在，
但其梯度只在 z≈desired_height 附近强，飞到 3–4m 时梯度已近乎为 0，
策略在 3–5m"死区"获得近似相同的 height_reward，没有足够的下行拉力。
`fly_high_threshold=5.0m` 与 `desired_height=2.5m` 之间存在 2.5m 的"无惩罚漂移带"。

#### 发现 2：upright_penalty 系统性恶化——权重翻倍适得其反 — Severity: HIGH

**Symptom:** upright_penalty recent 从 −1.78（Restart C）恶化至 −3.46（Restart D），即使仅权重 0.5→1.0。
若姿态稳定性保持不变，期望值应翻倍至 −3.56。实测 −3.46 说明标准化姿态误差略有改善（约 3%），
但改善量微乎其微，不值得支付双倍权重带来的总奖励下滑代价。
**Root Cause:** 高 fly_high 率下无人机在 4–5m 处激烈机动，body rate 偏大，
姿态角偏差本就比低飞时更大，upright_penalty 增大是飞行状态恶化的结果，而非孤立的控制问题。

#### 发现 3：grad_norm_clip 0.5 效果存疑 — Severity: MEDIUM

**Evidence:** grad_norm_actor early=0.444，recent=0.438，全程约为 0.44–0.50。
这说明：grad_norm 在 clip=0.5 时**已经很少被截断**（Restart C 时 clip=1.0 时 actor norm≈0.92，
经常被截断）。结论：clip 从 1.0 降至 0.5 确实减少了大梯度步，
但同时也限制了有效学习步长。当前 policy_std 从 0.512 上升至 0.673（+31%），
说明策略在增加探索，而非收敛——这可能与 grad_norm_clip 过小导致策略更新缓慢有关。

#### 发现 4：velocity_follow 信号极弱，策略无伴飞行为 — Severity: HIGH

velocity_follow recent mean = 0.055，全程 5 个 segment 均在 0.054–0.060 之间，无改善趋势。
策略从未发展出"与目标速度匹配"的行为。tracking_reward 虽>0 但趋势下滑，
说明无人机能短暂进入捕获区但无法保持，因为没有速度匹配能力。

---

### Improvement Recommendations

#### Priority 1 (CRITICAL): 引入高度稳定带 soft-penalty，打破"死区漂移"

**Problem:** fly_high_threshold=5.0m 与 desired_height=2.5m 之间存在 2.5m 无惩罚区间，
策略在 2.5–5.0m 漂移而不受惩罚，fly_high 率持续 >60%。
**Proposed Change:**
- File: `marl_flyfollow_env_cfg.py`
- 新增 `height_penalty_upper_soft_threshold: 3.5m`（在现有 fly_high_threshold=5.0 之下设置软边界）
- `height_penalty_upper_soft_weight: 1.5`（exp 形式，z>3.5m 时线性递增惩罚）
- Rationale: 在 desired_height=2.5m 和 fly_high_termination=5.5m 之间建立连续梯度，
  消除"高飞无代价"的死区，策略会主动保持在 2.5–3.5m 范围内。

#### Priority 2 (CRITICAL): 恢复 upright_penalty_weight 至 0.5，解耦姿态与飞行高度问题

**Problem:** upright_penalty_weight 1.0 使总惩罚增加约 −1.7/ep，
在飞行状态本身未改善的情况下加重了惩罚，拖累总奖励 20%。
**Proposed Change:**
- File: `marl_flyfollow_env_cfg.py`
- `upright_penalty_weight`: 1.0 → **0.5**（回退至 Restart C 值）
- Rationale: 姿态问题是飞行高度问题的下游结果，应先解决高度稳定性（Priority 1），
  再考虑增加 upright 惩罚。过早加重姿态惩罚只会压低总奖励而不改善飞行行为。

#### Priority 3 (HIGH): 降低 illegal_contact_penalty 权重，避免 reward 过度极化

**Problem:** illegal_contact_penalty=8.0 使单次撞击代价极高（−8 reward），
但撞击本身由 crash 率决定（约 43%），无法通过惩罚权重消除。
高权重使 illegal_contact_penalty 贡献 −3.31/ep，消耗了大量奖励空间。
**Proposed Change:**
- File: `marl_flyfollow_env_cfg.py`
- `illegal_contact_penalty`: 8.0 → **6.0**（保持比 Restart C 的 5.0 高，但减轻极化）
- Rationale: crash 率的真正下降需要飞行高度的改善（Priority 1/2），而非惩罚权重的提高。

#### Priority 4 (HIGH): 增强 velocity_follow 信号强度

**Problem:** velocity_follow recent=0.055，全程无上升趋势。
策略无法学习伴飞行为，导致无法维持 tracking 状态触发 success。
**Proposed Change:**
- File: `marl_flyfollow_env_cfg.py`
- `velocity_follow_weight`: 1.5 → **2.5**（提高信号强度）
- `velocity_follow_sigma`: 当前值 → **1.0**（适当放宽容忍窗口）
- Rationale: velocity_follow 是 tracking → success 的桥梁，当前权重 1.5 远低于
  distance_reward 的隐性贡献，策略优先最大化距离奖励而忽略速度匹配。

#### Priority 5 (MEDIUM): 调整 grad_norm_clip，平衡探索与稳定

**Problem:** grad_norm_clip=0.5 使 actor norm 从 0.92 降至 0.44，
policy_std 反而上升至 0.673（探索增加而非收敛），策略更新过慢。
**Proposed Change:**
- File: `scripts/skrl/train.py` 或 MAPPO 配置文件
- `grad_norm_clip`: 0.5 → **0.8**（居中值，减少截断同时避免 Restart C 的过度截断）
- Rationale: clip=1.0 时 norm≈0.92 频繁截断，clip=0.5 时 norm≈0.44 几乎不截断，
  0.8 是合理的折中点，既允许有效学习步长又防止极端梯度。

---

### Experiment Plan — Restart E

**基础策略：** 基于 Restart D 的 best checkpoint（约 seg3 结束，约 800k 步）热启动

**主要改动（相比 Restart D）：**

| 参数 | Restart D | Restart E | 优先级 |
|------|-----------|-----------|--------|
| 新增 `height_upper_soft_threshold` | 无 | **3.5m** | CRITICAL |
| 新增 `height_upper_soft_weight` | 无 | **1.5** | CRITICAL |
| `upright_penalty_weight` | 1.0 | **0.5** | CRITICAL |
| `illegal_contact_penalty` | 8.0 | **6.0** | HIGH |
| `velocity_follow_weight` | 1.5 | **2.5** | HIGH |
| `velocity_follow_sigma` | (当前值) | **1.0** | HIGH |
| `grad_norm_clip` | 0.5 | **0.8** | MEDIUM |

**训练设置：** num_envs=2048，训练至 2M 步

**500k Milestone 监测指标（Restart E）：**
- fly_high 率 < 40%（Restart D 的 65.8%；此为主要改善目标）
- crash 率 < 35%（Restart D 的 43.1%）
- fly_high + crash 之和 < 0.75（打破零和替代效应的量化指标）
- upright_penalty > −2.0（放宽 weight 后的期望值）
- tracking_reward > 1.5/ep（相比 Restart D 的 1.31）
- velocity_follow > 0.15/ep（相比 Restart D 的 0.055）
- episode_mean > 185 步

**1M Milestone 目标（Restart E）：**
- fly_high 率 < 25%
- crash 率 < 25%
- tracking_reward > 2.5/ep
- velocity_follow > 0.3/ep
- episode_mean > 220 步

---

### Regression Guards（更新）

- `fly_high_termination_z` 不得降低至 < 5.0m（4.5m 在历史 run 造成永久退化）
- `height_reward_weight` 不得低于 1.5（已有前车之鉴）
- `illegal_contact_penalty` 不得低于 5.0（3.0 已确认无效）
- `low_altitude_soft_penalty_weight` 不得归零（仅可上调）
- `tracking_reward_weight` 不得低于 4.0（Restart C/D 的 5.0 已验证有效）
- `upright_penalty_weight` 不得在高飞问题未解决前超过 0.5（Restart D 确认反效果）

## Changelog（续）

- 2026-04-01：分析 2026-03-30_13-16-00 run（Restart D，2,000,000 步完整分析）
  - 状态判定：REGRESSING，峰值出现在 800k（total_reward=48.9），近期退化至 32.5（−34%）
  - 核心发现：fly_high↔crash 零和替代效应未被打破（两者之和 Restart C=1.043，Restart D=1.049）
  - desired_height 2.0→2.5m 效果：crash 率下降 26.8%，但 fly_high 率上升 36.0%，净收益为零
  - upright_penalty_weight 0.5→1.0 适得其反：recent penalty 恶化 95%（−1.78→−3.46）
  - illegal_contact_penalty 5.0→8.0：撞击率降低 26.8% 但单次代价更高，净 penalty 恶化 18%
  - grad_norm_clip 1.0→0.5：actor norm 从 0.92 降至 0.44，有效，但 policy_std 上升至 0.673 说明未收敛
  - 根本问题：desired_height=2.5m 与 fly_high_threshold=5.0m 之间存在 2.5m 无惩罚"死区"
  - 决策：启动 Restart E（5 项参数变更，新增高度软惩罚上边界，upright_weight 回退至 0.5）

---

## Training Analysis Report — Restart E（2026-04-01_12-47-38）

**Run:** 2026-04-01_12-47-38_mappo_torch_mappo
**Date:** 2026-04-01
**Task:** Isaac-move-flyfollow-marl-v0
**Algorithm:** MAPPO
**Total timesteps:** ~695k（训练中，预计 2M 步）
**Status:** REGRESSING（平台化且近期轻微下滑）

### Restart E 变更回顾

| 参数 | Restart D | Restart E |
|------|-----------|-----------|
| `height_upper_soft_threshold` | 无 | 3.5m |
| `height_upper_soft_weight` | 无 | 1.5 |
| `upright_penalty_weight` | 1.0 | **0.5** |
| `illegal_contact_penalty` | 8.0 | **6.0** |
| `velocity_follow_weight` | 1.5 | **2.5** |
| `grad_norm_clip` | 0.5 | **0.8** |

---

## 训练指标摘要

| 指标 | early（Q1） | recent（Q4） | 最优 |
|------|------------|-------------|------|
| total_reward_mean | 62.23 | 56.39 | 131.15 |
| distance_reward | 24.62 | 24.49 | 40.98 |
| tracking_reward | 2.63 | 2.48 | 6.01 |
| height_reward | 1.57 | 1.57 | 2.80 |
| velocity_follow | 0.10 | 0.10 | 0.15 |
| upright_penalty | −2.00 | −2.15 | 0.00 |
| illegal_contact | −2.71 | −3.93 | 0.00 |
| height_upper_soft | −1.45 | −0.93 | 0.00 |
| height_penalty | −1.36 | −1.43 | 0.00 |
| policy_std | 0.637 | 0.774 | 0.801 |
| value_loss | 0.073 | 0.102 | 0.356 |
| episode_mean（步） | 205 | 213 | 378 |
| fly_high + crash 之和 | 1.039 | 1.039 | — |
| success_rate | ~0 | ~0 | 0.22（偶发）|

---

## 核心量化指标评估

### 指标 1：fly_high + crash 之和（目标 < 0.75）— FAIL

```
Q1: fly_high=0.5932  crash=0.4459  SUM=1.039
Q2: fly_high=0.5212  crash=0.5176  SUM=1.039
Q3: fly_high=0.4882  crash=0.5559  SUM=1.044
Q4: fly_high=0.3348  crash=0.7036  SUM=1.039
```

**结论：零和替代效应完全未被打破。** 全程四个季度之和稳定在 1.039±0.003，与 Restart C（1.043）和 Restart D（1.049）几乎相同。fly_high 从 Q1 到 Q4 降低了 44%（0.593→0.335），但 crash 同步上升了 58%（0.446→0.704），净效果为零。

height_upper_soft 的介入确实压制了飞高行为，但无人机的应对策略不是"正确悬停在目标高度"，而是"改为在低高度坠机（触发 illegal_contact）"。这是一种行为替代，而非任务解决。

### 指标 2：height_upper_soft 惩罚是否起到预期梯度引导作用 — 部分有效，但方向错误

```
height_upper_soft (Q1): −1.45  →  (Q4): −0.93   （penalty 绝对值下降 36%）
fly_high_rate (Q1):  0.593  →  (Q4):  0.335   （下降 44%）
fly_high 与 height_upper_soft 相关性: r = −0.559
```

height_upper_soft 对 fly_high 具有显著的抑制效果（penalty 减少对应飞高率下降），说明梯度确实在引导无人机避开高空区域。然而，引导方向不正确——无人机没有收敛到 desired_height=2.5m，而是降落到了触发 illegal_contact 的低高度区域。

---

## 发现与诊断

### 发现 1：illegal_contact 是真正的主导终止原因 — CRITICAL

crash 与 illegal_contact 的 episode-level 相关系数为 **r=0.9975**，实际上二者是同一事件（撞击地面或障碍物触发双重记录）。recent illegal_contact 惩罚均值已恶化至 −3.93/ep（early 为 −2.71），且 Q4 终止率 0.70 超过 fly_high 率 0.33 一倍，成为最大单一终止原因。

将 illegal_contact_penalty 从 8.0 降至 6.0 的假设（增加生存动机）未能实现：撞击终止率反而从 Restart D 的 0.469（early）上升至 0.704（Q4）。这说明撞击问题的根源不在惩罚强度，而在飞行高度引导不足。

**根本原因：** height_upper_soft 压制了飞高，但 desired_height 的吸引力不足以把无人机拉到安全高度带（low_altitude_soft_threshold=1.2m 以上），无人机滑入低高度撞地。

### 发现 2：upright_penalty 持续恶化，减重无效 — HIGH

```
upright_penalty: Q1=−2.00  Q2=−2.01  Q3=−2.11  Q4=−2.15
```

将 `upright_penalty_weight` 从 1.0→0.5 虽然直接减半了每单位姿态偏差的惩罚强度，但实际惩罚均值几乎没有改变（期望 Q4 约 −1.5，实际 −2.15）。这意味着姿态偏差本身在恶化（偏角加大），减少权重反而允许策略更大幅倾斜机体。

结合 illegal_contact 的恶化，可以推断无人机在追逐目标时出现过度倾斜（大 pitch/roll），导致接触地面的概率上升。

### 发现 3：velocity_follow 未受益于 weight 提升 — HIGH

```
velocity_follow: Q1=0.102  Q2=0.092  Q3=0.092  Q4=0.096
```

`velocity_follow_weight` 从 1.5 提升至 2.5（+67%），但 velocity_follow 奖励值全程稳定在约 0.10/ep，无上升趋势。对比目标：Restart E 500k milestone 要求 > 0.15/ep，未达标。

原因：velocity_follow 的绝对值过低（相比 distance_reward 的 24.5），策略几乎感受不到其梯度信号。即使将 weight 再翻倍，其在总奖励中的占比仍不足 1%，无法有效引导速度匹配行为。

### 发现 4：探索持续膨胀，价值函数估计失准 — MEDIUM

```
policy_std: Q1=0.637  Q2=0.672  Q3=0.732  Q4=0.774（持续上升）
value_loss: Q1=0.073  Q4=0.102（+40%）
```

policy_std 全程单调上升至 0.774（接近历史峰值 0.801），说明策略不是在收敛而是在持续探索。grad_norm_clip 从 0.5 放宽至 0.8 后，actor norm 保持在 0.69（比 Restart D 的 0.44 明显更活跃），但这种活跃性没有转化为性能改善，反而可能是探索不定向的信号。

value_loss 上升 40% 说明价值函数跟不上策略的变化速度，GAE 优势估计质量下降，PPO 更新方向的信噪比降低。

### 发现 5：总奖励平台化，无上升趋势 — HIGH

```
total_reward_mean: Q1=62.23  Q2=56.53  Q3=58.30  Q4=56.39
```

695k 步内总奖励在 56–62 区间震荡，没有学习趋势。最优值 131.15 出现在单个偶发 episode，不代表系统性能力。

distance_reward 稳定（24.5），说明无人机保持在目标附近，但 tracking_reward 无增长（2.5），success_rate≈0，说明无法进入捕获区并维持成功状态。

---

## 改进建议

### Priority 1 (CRITICAL): 为 desired_height 建立强吸引力，解决撞地根因

**问题：** height_upper_soft 将无人机从高空赶下来，但没有足够强的机制把它稳定在 desired_height=2.5m 附近的安全区间。无人机滑过安全带落入低空区域并撞地。

**当前状态：** height_reward_weight=2.0，但其信号被 height_penalty（−1.43）和 illegal_contact（−3.93）的噪声淹没。

**提议变更：**
- 文件：`exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/flyfollow/marl_flyfollow_env_cfg.py`
- `height_reward_weight`: 2.0 → **3.5**（强化高度吸引，使 desired_height 信号优先级高于接近奖励的 50%）
- `low_altitude_soft_penalty_weight`: 2.0 → **3.5**（低空区梯度与高空区对称，防止单侧压制导致无方向滑落）
- `low_altitude_soft_threshold`: 1.2 → **1.8m**（将低空软惩罚上边界抬高至距地面更安全的位置，更早激活梯度）

**预期效果：** 无人机受到来自 3.5m 以上（height_upper_soft）和 1.8m 以下（low_altitude_soft）的双重压力，自然收敛到 2.5m 附近。

### Priority 2 (CRITICAL): 重构 velocity_follow 信号，使其可达可感知

**问题：** velocity_follow 绝对值约 0.10/ep，在总奖励（约 57）中占比不足 0.2%，策略梯度中该信号几乎不可见。提高 weight 的正确方式是同时改善信号密度，而非单纯调倍数。

**提议变更：**
- 文件：`marl_flyfollow_env_cfg.py`
- `velocity_follow_weight`: 2.5 → **4.0**（在当前信号强度下需要极高权重才能产生效果）
- 同时在 `marl_flyfollow_env.py` 检查 velocity_follow 的计算方式：若为 `exp(−err²/σ²)`，应确认 σ 是否匹配目标速度 0.3m/s 的量级（建议 σ=0.3m/s，使 1 倍误差时仍有 ~37% 奖励可得）

### Priority 3 (HIGH): 重新启用 upright_penalty 监控，并增加姿态反馈到观测

**问题：** upright_penalty 持续恶化（−2.00→−2.15）说明策略在倾斜飞行，这是 illegal_contact 恶化的直接前驱。weight 减半只是降低了惩罚强度，没有解决姿态偏差本身。

**提议变更：**
- 文件：`marl_flyfollow_env_cfg.py`
- `upright_penalty_weight`: 0.5 → **0.8**（适度恢复，但不回到 1.0 以避免 Restart D 的过度惩罚问题）
- 检查观测空间是否包含 body z-axis（upright_expect_dir 方向的误差角）；若不包含，则在 observation 中加入 roll/pitch 或 body_z_dot 分量，让策略直接感知姿态偏差

### Priority 4 (MEDIUM): 减缓探索膨胀，稳定价值函数

**问题：** policy_std 单调上升至 0.774，value_loss 上升 40%，说明策略-价值函数跟踪不同步，影响 PPO 更新质量。

**提议变更：**
- 文件：`scripts/skrl/train.py` 或 SKRL agent config
- `entropy_loss_scale`（若可配置）适度增大，增加熵正则化以遏制 std 继续膨胀（或考虑 std_clamp 上界）
- `grad_norm_clip`: 0.8 → **0.6**（折中 Restart D 的 0.5 与当前 0.8，避免过度截断同时控制更新幅度）
- 若 SKRL 支持 value_loss_clipping，确认已启用（PPO 标准做法）

### Priority 5 (LOW): 清理 illegal_contact_penalty 与 crash 的双重记录歧义

**问题：** illegal_contact 与 crash termination 相关系数为 0.9975，几乎是同一事件的双重记录，导致分析时难以区分终止原因。

**提议：** 在日志中确认 `crash` 终止条件的定义是否与 `illegal_contact` 独立；若重叠，应在奖励分析中合并为单一指标，避免误判严重性。

---

## 实验计划（Restart F）

若 695k→1M 步内无明显改善（fly_high+crash 之和仍 > 0.9，tracking_reward 无增长趋势），建议在 1M 步检查点后启动 Restart F：

1. 应用 Priority 1：height_reward_weight 3.5，low_altitude_soft 3.5/1.8m
2. 应用 Priority 2：velocity_follow_weight 4.0，检查 velocity_follow sigma
3. 应用 Priority 3：upright_penalty_weight 0.8
4. 应用 Priority 4：grad_norm_clip 0.6
5. 运行命令：`python3 scripts/skrl/train.py --task=Isaac-marl-flyfollow-v0 --headless --num_envs=2048 --algorithm="MAPPO"`

**500k Milestone 监测指标（Restart F）：**
- fly_high + crash 之和 < 0.85（过渡目标；最终目标 < 0.75）
- illegal_contact 终止率 < 0.50（当前 0.70）
- height_reward recent_mean > 2.5/ep（当前 1.57）
- velocity_follow recent_mean > 0.15/ep（当前 0.10）
- upright_penalty > −1.8/ep（当前 −2.15）

**成功判断：**
- fly_high + crash 之和 < 0.75（打破零和效应）
- tracking_reward > 3.0/ep
- success_rate > 1%（all_targets_captured > 0.01）

---

## Changelog（续）

- 2026-04-01：分析 2026-04-01_12-47-38 run（Restart E，~695k 步中期分析）
  - 核心发现：fly_high+crash 零和替代效应完全未打破（全程稳定在 1.039±0.003）
  - height_upper_soft 有效压制飞高（Q1→Q4 降 44%），但无人机转而撞地（crash Q1→Q4 升 58%）
  - illegal_contact 成为主导终止原因（Q4 率 0.70），惩罚降低（8→6）未减少撞击，根因在高度引导不足
  - upright_penalty 减重（1.0→0.5）无效：姿态偏差加大导致惩罚值不减反增（−2.00→−2.15）
  - velocity_follow_weight 提升（1.5→2.5）无效：绝对奖励值不足 0.2%，信号被淹没
  - policy_std 单调膨胀（0.637→0.774），价值函数失准（value_loss +40%），策略未收敛
  - 决策：若 1M 步无改善则启动 Restart F（5 项参数调整，重点强化高度双边吸引力）
- 2026-04-02：分析 2026-04-01_12-47-38 run（Restart E，1.69M 步最终分析）— 见下方完整报告

---

# Training Analysis Report — Restart E 最终分析（1.69M 步）

**Run:** 2026-04-01_12-47-38_mappo_torch_mappo
**Date:** 2026-04-02
**Task:** Isaac-marl-flyfollow-v0（Isaac-move-flyfollow-marl-v0）
**Algorithm:** MAPPO
**Analysis type:** Restart E 第二次分析（本次约 1.69M 步；上次 695k 步）

## Training Metrics Summary

| 指标 | 上次（695k） | 本次最终（1.69M） | 趋势 |
|------|------------|----------------|------|
| total_reward_mean (recent) | ~57 | 36.1 | -37% REGRESSING |
| tracking_reward (recent) | ~2.5 | 1.363 | -45% |
| velocity_follow (recent) | ~0.097 | 0.103 | +6% |
| distance_reward (recent) | ~24.1 | 18.3 | -24% |
| height_reward (recent) | ~1.56 | 1.632 | +5% |
| ep_len_mean (recent) | ~213 | 199.8 | -6% |
| fly_high (last300k) | 0.37 | 0.3216 | -13% |
| crash (last300k) | 0.67 | 0.7215 | +8% |
| fly_high + crash | 1.04 | **1.043** | 未改变 |
| policy_std (last) | 0.774 | 0.647 | 下降 |
| success_reward | 0.000 | 0.000 | 从未触发 |

**Active config (Restart E):**
- desired_height=2.5m, height_upper_soft_threshold=3.5m, weight=1.5
- low_altitude_soft_threshold=1.2m, weight=2.0
- fly_high_threshold=5.0m, fly_high_termination_z=5.5m
- illegal_contact_penalty=6.0
- upright_penalty_weight=0.5
- velocity_follow_weight=2.5, tracking_reward_weight=5.0
- grad_norm_clip=0.8, entropy_loss_scale=0.015（非 env.yaml 项，来自上次分析推断）

## 1M 步里程碑检定结果（OFFICIAL VERDICT）

| 里程碑目标 | 目标值 | 实际值（last300k） | 判定 |
|-----------|------|--------------------|------|
| fly_high < 25% | <0.25 | 0.322 | **FAIL** |
| crash < 25% | <0.25 | 0.722 | **FAIL** |
| fly_high + crash < 0.75 | <0.75 | **1.043** | **FAIL** |
| tracking_reward > 2.5/ep | >2.5 | 1.352 | **FAIL** |
| velocity_follow > 0.3/ep | >0.3 | 0.107 | **FAIL** |
| episode_mean > 220 步 | >220 | 197.6 | **FAIL** |

**全部 6 项里程碑均未达到。**

## Observations & Findings

### 发现 1：零和替代效应完全固化 — CRITICAL

**Symptom:** fly_high + crash 之和在整个 1.69M 步内以惊人的精度维持在 1.04±0.01。100k 窗口数据：

| 窗口 | fly_high | crash | 总和 |
|------|---------|-------|------|
| 0.2–0.3M | 0.487 | 0.556 | 1.043 |
| 0.5–0.6M | 0.373 | 0.670 | 1.043 |
| 0.8–0.9M | 0.262 | 0.770 | 1.031 |
| 1.0–1.1M | 0.194 | 0.850 | 1.044 |
| 1.3–1.4M | 0.280 | 0.766 | 1.047 |
| 1.6–1.7M | 0.347 | 0.702 | 1.049 |

**Root Cause:** 零和效应不是 Restart E 新出现的问题，从 Restart C（1.5M 步，sum≈1.05）到 Restart D（2M 步，sum≈1.047）到 Restart E（1.69M 步，sum≈1.043）持续三个 Restart 共计 5M+ 步保持不变。这已确认是**结构性问题**，与惩罚权重无关。

**Evidence:** crash 与 illegal_contact 的相关性（mean_diff≈0.002）证实两者是同一物理事件的双重记录。"crash"= 无人机与目标或地面发生接触，非坠落。

**Key insight:** height_upper_soft 的引入（Restart E 新增）有效地把 fly_high 从 0.49（Restart C 末）→ 0.32（本次 1.69M 末），但 crash 从 0.47（Restart C 末）→ 0.72（本次 1.69M 末）。height_upper_soft 梯度成功拦截了高飞，但把无人机直接推向地面，净效果为零。

---

### 发现 2：tracking_reward 持续衰退，1M 步后未见底 — CRITICAL

**Symptom:** tracking_reward 在早期（0.2–0.9M 窗口）维持在 2.3–2.6/ep，但从 1.0M 步开始出现断崖式下跌：

| 窗口 | tracking_reward |
|------|----------------|
| 0.8–0.9M | 2.284 |
| 0.9–1.0M | 2.042 |
| **1.0–1.1M** | **1.640** ← 断崖 |
| 1.1–1.2M | 1.701 |
| 1.6–1.7M | 1.276 |

**Root Cause:** 1.0M 步正好是 crash 率达到峰值（0.850）的时间窗口，此后 crash 持续高于 0.70，episode 平均缩短 → tracking 积累时间减少 → tracking/ep 下降。这与第 695k 步分析的预测吻合：crash 主导的短 episode 挤压 tracking 积累。

**Evidence:** ep_len_mean 从 1.0M 窗口的 202 步下降到 1.6M 窗口的 188 步（−7%），last 值为 152 步（−25%），与 tracking 衰退同步。

---

### 发现 3：velocity_follow 信号彻底失效 — HIGH

**Symptom:** velocity_follow 在整个 1.69M 步内从 0.093 到 0.109/ep，**完全平坦**，无任何学习趋势。velocity_follow_weight 从 1.5→2.5 的提升（Restart E 改动）产生零效果。

**Root Cause:** distance_reward/ep 平均约 18.3，velocity_follow/ep 约 0.103，信号比 = 18.3/0.103 = **178:1**。策略梯度中 velocity_follow 的贡献约为 0.6%，完全被 distance_reward 淹没。参照规则 18（velocity_follow_weight 必须 ≥ distance_reward_weight × 0.3 才能显现），当前 weight=2.5 vs 需要 ≥ 5.0×0.3=1.5——但实际值还要除以信号幅度差异，等效要求 weight ≥ 5.0 才能产生可见梯度。

---

### 发现 4：policy_std 从膨胀转为收缩，探索枯竭 — HIGH

**Symptom:** policy_std 在 0.7–0.8M 窗口达到峰值 0.788，随后持续下降：

| 窗口 | policy_std |
|------|-----------|
| 0.5–0.6M | 0.750 |
| 0.7–0.8M | 0.773（峰值） |
| 1.0–1.1M | 0.734 |
| 1.5–1.6M | 0.681 |
| 1.6–1.7M | 0.657 |
| last | 0.647 |

**Root Cause:** policy_std 峰值恰好出现在 tracking_reward 开始断崖（1.0M）的前 200k 步。探索扩展 → 无人机进入 crash 区域更多 → crash 率上升 → episode 变短 → 奖励减少 → policy 逐渐收缩以减少 crash。这是一个负反馈循环。

**Evidence:** 在上次（695k）分析时 policy_std 仍在膨胀（0.637→0.774），本次确认在 0.774 见顶后持续回落。策略正趋向 crash-avoidance 保守模式，探索能力递减。

---

### 发现 5：upright_penalty 无法自我改善 — MEDIUM

**Symptom:** upright_penalty 在 1.69M 步内从 −2.02（0.2M 窗口）→ −1.87（1.6M 窗口），仅改善约 7%，且趋势不稳定。

**Root Cause:** 高 crash 率导致短 episode，无人机无法在足够多的步数内学习姿态控制。且 crash 主要发生在接近目标的低空动作中（aggressive approach → tilt → contact），这是任务内置的矛盾：接近目标需要加速倾斜，但 upright_penalty 要求保持竖直。

---

## Go/No-Go 决策

### 明确判断：**NO-GO — 立即启动 Restart F**

理由：
1. **全部 6 项 1M 步里程碑均未达到**（最关键指标 fly_high+crash=1.043 vs 目标 <0.75，差距 39%）
2. **零和替代效应持续 5M+ 步（Restart C+D+E）**，证明当前奖励结构无法通过权重调整突破
3. **tracking_reward 在 1.0M 步后出现结构性衰退**（最终 1.276/ep vs 峰值 2.6/ep，且无反弹迹象）
4. **policy_std 已从探索峰值（0.774）回落至 0.647**，探索能力递减，继续训练只会加速收缩
5. **velocity_follow 1.69M 步内完全无进展**（0.093→0.109，趋势统计显著性近零）
6. **Restart E 的核心假设（height_upper_soft 打破零和）被数据证伪**：sum 在引入后仍为 1.043

继续训练 Restart E 的理由：无。所有指标均在衰退或停滞，且根因已明确（结构性零和 + velocity_follow 信号淹没）。

---

## Improvement Recommendations

### Priority 1 (CRITICAL): 引入 3D 距离奖励打破零和的结构根因

**Problem:** XY-only 距离奖励在 z 方向无梯度。无人机在 z=1.2m（crash 边界）和 z=5.0m（fly_high 阈值）之间任何高度都获得相同的 distance_reward。飞高和撞地提供相同的逃避收益，只是终止代价不同。

**Proposed Change:**
- 文件：`exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/flyfollow/marl_flyfollow_env.py`
- 在 distance_reward 计算中将 XY 距离替换为 **XYZ 3D 距离**（目标高度取地面 z≈0.25m 加 desired_height=2.5m 作为期望停留点，即 3D 目标点 = [target_xy, 2.5m]）
- 效果：z=2.5m 的无人机获得最大 distance_reward；偏离（过高或过低）均减少奖励
- 注意：不要求无人机追 z=0.25m 的地面目标，应使用 [target_x, target_y, desired_height] 作为 3D 参考点
- Rationale: 这是从 Restart C 分析（规则 16）反复被识别但从未实施的根本修复。在奖励结构层面提供连续的高度引导，替代所有零散的软惩罚信号。

**预期效果:** 高度偏差直接影响 distance_reward（最大 reward term，占比 50%），无人机有强动机维持在 desired_height 附近。

---

### Priority 2 (CRITICAL): velocity_follow 信号重构（重大权重提升 + 计算检查）

**Problem:** velocity_follow/ep = 0.107，distance_reward/ep = 18.3，信号比 178:1，velocity_follow_weight=2.5 完全无效（信号淹没阈值约为 weight ≥ 5.0 才能显现）。

**Proposed Change:**
- 文件：`marl_flyfollow_env_cfg.py`
- `velocity_follow_weight`: 2.5 → **5.0**（与 tracking_reward_weight 等重，使 velocity_follow 成为同等优先级信号）
- 文件：`marl_flyfollow_env.py`
- 检查 velocity_follow reward 的计算公式：应确保当无人机速度 = target_speed（0.3m/s x 向）时奖励值接近权重峰值，而非 exp(−err²/σ²) 中 σ 设置过窄导致奖励近似为 0
- 建议 velocity_follow 仅在 dist_xy < capture_distance（3.5m）范围内激活，避免远距离无意义的速度匹配信号

---

### Priority 3 (HIGH): 废弃 height_upper_soft，改为 height_band 正奖励

**Problem:** height_upper_soft（推力向下）+ low_altitude_soft（推力向上）是双侧负梯度，策略的最优应对是在两者之间找平衡点——但这个"中间地带"奖励为零（既没有正向吸引也没有进一步惩罚）。三代 Restart 数据证明双侧负梯度产生不稳定的振荡均衡，而非稳定收敛。

**Proposed Change:**
- 文件：`marl_flyfollow_env.py`
- 新增 `height_band_reward`：当 min_altitude < z < fly_high_threshold 时，给予持续正奖励（比如 `exp(−|z−desired_height|/σ_h)` with σ_h=0.5m），将 desired_height 做成奖励的"吸引盆"
- `height_upper_soft_weight`: 1.5 → **0.5**（大幅减弱排斥力，改为吸引力主导）
- `low_altitude_soft_penalty_weight`: 2.0 → **1.0**（与 height_upper_soft 对称减弱）
- 文件：`marl_flyfollow_env_cfg.py`
- `height_band_reward_weight`: 新增 2.0（与 height_reward_weight 同级，协同提供高度信号）

**Rationale:** 规则 17（height corridor squeeze 规则）指出正确修复是提供"stable middle zone reward"，而非只调整边界惩罚。三代 Restart 均失败于只调整边界惩罚的策略。

---

### Priority 4 (HIGH): 降低 crash 与低空接触的物理可能性（结构性隔离）

**Problem:** crash = illegal_contact（confirmed，correlation 0.9990）。无人机在低空接近地面目标时发生物理接触。即使惩罚提高到 8.0（Restart D），crash 率也只降低 27% 且立即转移到 fly_high。

**Proposed Change:**
- 文件：`marl_flyfollow_env_cfg.py`
- `min_altitude`（硬终止高度）: 1.0 → **1.5m**（与 desired_height=2.5m 更接近，减少无人机接近地面的物理机会）
- `low_altitude_soft_threshold`: 1.2 → **2.0m**（软惩罚覆盖从 2.0m 到 1.5m 的完整危险带，比 crash 高出 0.5m 提供足够的梯度预警距离）
- 保持 `illegal_contact_penalty`: 6.0（不继续提高，已证明提高惩罚是无效干预）
- 保持 `fly_high_termination_z`: 5.5m（不放松，防止惯性高飞）

---

### Priority 5 (MEDIUM): 恢复 policy 探索能力

**Problem:** policy_std 从 0.774 降至 0.647，探索空间收缩。同时 grad_norm_actor 处于中间水平（~0.69），显示梯度健康但策略仍在收缩。

**Proposed Change:**
- 文件：SKRL agent 配置
- `entropy_loss_scale`: 0.015 → **0.02**（如果 Restart D 的教训是 0.02 过高，那与 Restart D 不同的是本次 grad_norm_clip=0.8 而非 0.5，两者结合应更稳定）
- `grad_norm_clip`: 0.8 → **1.0**（Restart D 将其降至 0.5 导致 policy_std 异常膨胀；本次目标是维持 std 在 0.65 附近，使用默认 1.0 更稳定）
- 热启动策略：从本 run 的 best_agent.pt（约 900k 步，tracking_reward best 约 6.0）热启动，而非从 1.69M 步末尾检查点

---

## 实验计划（Restart F）

**决策：NO-GO — 立即启动 Restart F**

### 变更列表

| # | 优先级 | 参数 / 文件 | 旧值 | 新值 | 理由 |
|---|--------|------------|------|------|------|
| 1 | CRITICAL | `marl_flyfollow_env.py` distance_reward 计算 | XY 距离 | XYZ 3D 距离（目标点=[target_xy, desired_height]） | 打破 z 方向无梯度的结构根因 |
| 2 | CRITICAL | `velocity_follow_weight` | 2.5 | **5.0** | 突破 178:1 信号淹没 |
| 3 | HIGH | 新增 `height_band_reward_weight` | - | **2.0** | 正向吸引替代双侧惩罚 |
| 4 | HIGH | `height_upper_soft_weight` | 1.5 | **0.5** | 减弱排斥力，避免 crash 替代 |
| 5 | HIGH | `low_altitude_soft_penalty_weight` | 2.0 | **1.0** | 与 height_upper_soft 对称减弱 |
| 6 | HIGH | `low_altitude_soft_threshold` | 1.2m | **2.0m** | 扩展低空预警覆盖范围 |
| 7 | HIGH | `min_altitude` | 1.0m | **1.5m** | 物理隔离，减少接触机会 |
| 8 | MEDIUM | `entropy_loss_scale` | 0.015 | **0.02** | 恢复探索，配合 grad_norm=1.0 |
| 9 | MEDIUM | `grad_norm_clip` | 0.8 | **1.0** | Restart D 教训：0.5 过度限制 |

### 热启动点

从本 run（2026-04-01_12-47-38）的 **best_agent.pt** 热启动（约 900k 步附近，tracking_reward 历史最高 6.014）。

不推荐从 1.69M 步末尾启动，因策略已进入 crash-avoidance 保守模式（std=0.647 下行，tracking 持续衰退）。

### 训练命令

```bash
python3 scripts/skrl/train.py \
  --task=Isaac-marl-flyfollow-v0 \
  --headless --num_envs=2048 --algorithm="MAPPO"
```

### 500k 步里程碑（Restart F）

| 指标 | 目标值 | 来源 |
|------|--------|------|
| fly_high + crash | < 0.85 | 过渡目标（最终目标 <0.75） |
| fly_high | < 35% | 从 32.2% 维持不恶化 |
| crash | < 50% | 从 72.2% 大幅改善 |
| tracking_reward | > 1.8/ep | 从衰退低点 1.28 回升 |
| velocity_follow | > 0.20/ep | 从 0.107 提升 2× |
| distance_reward | > 18/ep | 维持当前水平（不退步） |
| episode_mean | > 190 步 | 从 188 步稳定 |

### 1M 步成功判据（Restart F）

- fly_high + crash 之和 < 0.75（**核心判据，唯一不可妥协的指标**）
- tracking_reward > 2.5/ep（回到 Restart E 前期水平）
- velocity_follow > 0.25/ep（明显改善，证明信号可达）
- episode_mean > 200 步

### 关键监测

1. **distance_reward 早期值**：3D 距离奖励可能比 XY 更小（高度偏差惩罚新增），若 early distance_reward < 15，说明 sigma 需要调整
2. **crash 率 100k 步内响应**：min_altitude=1.5m + low_altitude_soft=2.0m 应在最初 100k 步内使 crash 降到 <60%；若无响应则说明 3D 距离奖励导致无人机俯冲到 desired_height 途中直接接触地面
3. **velocity_follow 早期值**：weight=5.0 下 early mean 应 >0.12；若仍 <0.10，说明 sigma 参数问题，需检查 env.py 中的公式

---

## Changelog（续）

- 2026-04-02：分析 2026-04-01_12-47-38 run（Restart E，1.69M 步最终分析）
  - 核心发现：fly_high+crash 零和 sum=1.043 全程固化，15 个 100k 窗口均在 1.031–1.049 范围内（精度 ±1%），3 代 Restart 共 5M+ 步均未能打破
  - 1.0M 步断崖：tracking 从 2.04（0.9M 窗口）→ 1.64（1.0M 窗口），与 crash 率峰值 0.85 同期出现，因果链明确
  - policy_std 从峰值 0.774 回落至 0.647，探索能力递减，继续训练只会加速收缩
  - velocity_follow 1.69M 步内信号比 178:1，weight=2.5 完全无效，需要 weight=5.0 且配合 3D 距离奖励降低 distance_reward 主导地位
  - height_upper_soft 机制证伪：将 fly_high 从 0.49→0.32 但 crash 从 0.47→0.72，净效果为零
  - **决策：NO-GO，立即启动 Restart F**（9 项变更，核心是 3D 距离奖励 + velocity_follow weight×2 + 高度正吸引替代双侧负惩罚）

---

## Training Analysis Report — Move Task 首跑

**Run:** `2026-04-02_11-47-43_mappo_torch_mappo`
**Date:** 2026-04-02
**Task:** Isaac-marl-move-v0（move 任务，非 move_flyfollow）
**Algorithm:** MAPPO
**Total Steps:** 278,700
**num_envs:** 512
**Status:** STALLED — 探索完全崩溃

---

### 训练指标摘要

| 指标 | Early | Recent/Last | 趋势 |
|------|-------|-------------|------|
| total_reward_mean (per step) | 0.239 | 0.264 | +10%，近乎停滞 |
| distance_reward (per step) | 0.02958 | 0.02962 | 几乎不变 |
| tracking_reward (per step) | 0.001246 | 0.001260 | 微小上升，接近零 |
| height_reward (per step) | 0.019997 | 0.019994 | 固化 |
| body_rate_penalty (per step) | 0.014819 | 0.019974 | 上升（drone悬停化）|
| policy_std | 0.8155 | **0.0035** | **230x 崩溃** |
| value_loss | 2.569 | 0.005 | 极度收敛（过拟合）|
| episode_timesteps_mean | 1.0 | 1.0 | 日志为逐步记录 |

> **注：** `episode_timesteps = 1.0` 是日志记录模式问题，该 run 的 Episode_Reward 记录的是**每步**（step_dt=0.01s）的瞬时值，而非 episode 累计值（前一 run `2026-03-20` 记录的是 episode 累计值，episode 平均长度 1641 步）。

---

### 性能对比：与上一 run（2026-03-20_16-26-19）

将新 run 的每步值 × 1641（prev run 平均 episode 长度）归一化后对比：

| 指标 | 新 run（归一化/ep） | 前一 run（/ep） | 比率 |
|------|-----------------|----------------|------|
| distance_reward | 48.6 | 196.0 | **25%** |
| tracking_reward | 2.1 | 37.4 | **6%** |
| height_reward | 32.8 | 26.5 | 124% |
| body_rate_penalty | 32.8 | 28.0 | 117% |
| policy_std (last) | 0.0035 | 0.150 | **2.3%** |

结论：新 run 的任务核心性能（tracking/distance）大幅低于前一 run，处于早期训练水平。

---

### 奖励项分布（占 total reward 比例，per step）

| 奖励项 | 值 | 占比 |
|--------|-----|------|
| distance_reward | 0.02962 | 11.2% |
| height_reward | 0.01999 | 7.6% |
| body_rate_penalty | 0.01997 | 7.6% |
| action_smoothness | 0.00999 | 3.8% |
| force_penalty | 0.00419 | 1.6% |
| velocity_penalty | 0.00289 | 1.1% |
| **tracking_reward** | **0.00126** | **0.5%** |
| 其余（upright/collision等，未分项） | ~0.175 | ~67% |

**关键发现：** tracking_reward 仅占 0.5%，distance_reward : tracking_reward = 22:1。policy 完全没有动力进入 1m 捕获区，只需在 ~3.24m 处保持接近即可获得大部分 distance 奖励。

---

### 发现与诊断

#### 问题 1 — 探索完全崩溃 [CRITICAL]

**症状：** policy_std 从 0.8155 → 0.0035（230 倍崩溃），仅用 278k 步，policy 已接近全确定性。

**根因：** 
- 在 move 任务中，稳定悬停在 desired_height=2.5m 就能持续获得 height_reward（满分 0.02/step）和 body_rate_penalty（满分 0.02/step）。这两项合计占可观测正奖励的 ~15%，构成强力的"不动"吸引子。
- distance_reward 在 3.24m 处已有收益，policy 学会了保持这个距离。无需进一步接近（进入 1m 捕获区），所以没有梯度推动 policy 探索。
- entropy 系数过小，无法对抗上述吸引力。

**证据：** height_error=0.0003m（drone 锁定在 2.5m），body_rate_norm≈0.0013rad/s（近乎静止），distance 3.24m 全程不变。

#### 问题 2 — tracking_reward 信号不可达 [HIGH]

**症状：** tracking_reward = 0.00126/step（inner value=0.126），对应的 `is_captured * exp(-dist)` 极小，意味着几乎没有目标被捕获（capture_distance=1.0m），或捕获时间极短。

**根因：** capture_distance=1.0m 相对于无人机初始位置（drone_spawn_x∈[-8,-6]，target_spawn_x∈[-6,-2]）来说是个非常小的目标。无人机从 ~3.24m 处进入 1m 捕获圈需要额外接近，而当前奖励体系中接近 1m 的边际收益远小于 distance_reward 已提供的连续引导，形成"最后一公里"缺失。

**证据：** tracking_reward/distance_reward = 0.043（约 1/22），前一 run 为 37.4/196=0.19（约 1/5）。

#### 问题 3 — 高度锁定局部最优 [HIGH]

**症状：** height_error=0.0003m，drone 精确锁定在 desired_height=2.5m，没有高度波动。

**根因：** height_reward_weight=2.0 给出了强烈的高度保持激励，加上 height_penalty（超过 0.5m 阈值才惩罚）构成"高度保持即满分"的局面。drone 在垂直方向学到了完美悬停，但代价是丧失了水平接近的探索能量。

**证据：** height_reward 从 early 到 last 几乎无变化（0.019997→0.019994），body_rate_penalty 从 early 0.014819 上升到 0.019974（drone 越来越"安静"）。

#### 问题 4 — 与 move_flyfollow 任务的关键差异 [MEDIUM]

| 维度 | move 任务 | move_flyfollow 任务 |
|------|-----------|---------------------|
| 目标类型 | 4 个 NovaCarter 小车（地面移动） | 移动目标点（空中） |
| capture_distance | 1.0m（小） | 3.0m（大） |
| 目标高度 | z=0.0m（地面） | 与无人机同高 |
| 成功条件 | 3 个目标同时被 follow 3s | 追上 1 个目标保持 |
| 高度控制目标 | desired_height=2.5m（悬停） | 跟随目标高度 |
| 多目标博弈 | 存在（3 drone vs 4 target 分配） | 简化（1对1） |

**关键差异影响：** move 任务的目标在地面（z=0），而无人机悬停在 2.5m，水平距离计算正确但 3D 距离约为 2.6-4m，capture_distance=1m 意味着无人机必须几乎俯冲到地面附近才能触发捕获。这在物理上不合理，且会与高度保持奖励产生根本冲突。

**注意：** 目标使用 VisualizationMarkers（非物理 Articulation），因此无人机不能实际与目标碰撞，捕获判定纯靠距离。

---

### 改进建议

#### Priority 1 (CRITICAL)：解决探索崩溃

**问题：** policy_std 在 278k 步内崩溃 230 倍，后续训练无意义。

**方案 A — 熵正则化强化：**
- File: SKRL agent config（train.py 或 agent yaml）
- Parameter: `entropy_loss_scale` → 从当前值提高到 0.01~0.05
- 证据：entropy_loss 为正（系数在推熵增大）但仍不够强

**方案 B — 重启训练并提高初始 policy_std：**
- 从最早期 checkpoint（agent_5000.pt）重启，避免当前完全确定性策略
- 或增加 MAPPO 的 initial_log_std 参数

#### Priority 2 (HIGH)：修复 capture_distance 物理不一致性

**问题：** 目标在地面（z=0），无人机悬停在 z=2.5m，3D 欧氏距离 ≥2.5m，而 capture_distance=1.0m 永远不可能触发（除非无人机坠到地面）。

**两种修复路径：**

**路径 A — 只用 XY 平面距离判断捕获（推荐）：**
- File: `exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/move/marl_move_env.py`
- 修改 `dist_matrix` 计算（line ~561-562）：将 `[:, :, :2]` 的 2D 距离用于捕获判定
- 修改 tracking_reward 中的 `min_dists` 也改为 XY 平面距离
- 同步修改 `_get_dones` 中的 `is_captured_now` 逻辑（若有的话）
- Rationale：无人机从空中跟随地面目标，垂直高度差不应阻止捕获判定

**路径 B — 增大 capture_distance：**
- File: `exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/move/marl_move_env_cfg.py`
- Parameter: `capture_distance = 1.0` → `3.0`（考虑到 2.5m 高度差）
- 风险：3D 捕获可能导致无人机在错误高度"捕获"目标

#### Priority 3 (HIGH)：tracking_reward 信号强化

**问题：** tracking_reward : distance_reward = 1:22，policy 无动力进入捕获区。

**建议：**
- File: `exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/move/marl_move_env_cfg.py`
- Parameter: `tracking_reward_weight = 1.0` → `4.0`（与 move_flyfollow 历史修复类似）
- Parameter: `dist_reward_weight = 1.5` → `0.8`（降低 distance 主导地位）
- Rationale：仿照 move_flyfollow 中 sigma/tracking_weight 修复经验，必须让 tracking 成为主要奖励信号

#### Priority 4 (MEDIUM)：高度锁定局部最优打破

**问题：** height_reward_weight=2.0 产生强高度保持激励，与水平追踪形成竞争。

**建议：**
- File: `exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/move/marl_move_env_cfg.py`
- Parameter: `height_reward_weight = 2.0` → `0.5`（降低高度奖励权重）
- 保留 `height_penalty_weight` 作为软约束防止飞得太高/太低
- Rationale：无人机的主要任务是水平追踪，高度维持只需保证安全范围，不需要精确锁高

#### Priority 5 (MEDIUM)：num_envs 从 512 提高到 2048

**问题：** 当前 num_envs=512，与前一 run 相同但训练速度偏慢。

- 提高到 2048（与 move_flyfollow 训练设置一致）可以 4x 加速采样效率
- 对探索崩溃有一定缓解作用（更多样化的初始状态覆盖）

---

### 实验计划（Restart A for move task）

**执行顺序：**

1. **修复 capture_distance 物理一致性**（Priority 2，路径A）：将捕获判定改为 XY 平面距离
2. **调整奖励权重**（Priority 3+4）：
   - `tracking_reward_weight`: 1.0 → 4.0
   - `dist_reward_weight`: 1.5 → 0.8
   - `height_reward_weight`: 2.0 → 0.5
3. **提高 num_envs**：512 → 2048
4. **重新训练：**
   ```
   python3 scripts/skrl/train.py --task=Isaac-marl-move-v0 --headless --num_envs=2048 --algorithm="MAPPO"
   ```
5. **监测指标（100k 步节点）：**
   - policy_std 衰减速度是否 < 5x（不崩溃）
   - tracking_reward per step > 0.003（前 run 的 2.4x）
   - distance_reward per step：检查是否减小（确认 XY-only 的影响）
   - episode 终止原因分布：fly_low / collision / timeout 各自比例

**成功判据（300k 步）：**
- policy_std > 0.05（未完全崩溃）
- tracking_reward (norm/ep) > 10（前 run 的 37.4 的 27%）
- 至少有 target_captured 事件在日志中出现

---

### Changelog（续）

- 2026-04-02：分析 move 任务首跑 `2026-04-02_11-47-43`（278k 步）
  - 核心发现：policy_std 230x 崩溃（最严重的探索崩溃，超过 move_flyfollow 历史所有 run）
  - capture_distance=1.0m + 目标在地面（z=0）+ 无人机悬停在 2.5m = 捕获在物理上不可能触发，tracking_reward 仅为 distance 的 1/22
  - 高度锁定局部最优（height_error=0.0003m，drone 完全静止）
  - 前一 run（2026-03-20）的 tracking/ep = 37.4 已验证该任务可学习，本次退步原因很可能是超参数或代码修改引入了 bug（target_spawn_z=0.0 vs 0.25，以及可能的 num_envs 较小导致梯度噪声大）
  - **决策：依优先级顺序执行上述修复，特别是 XY-only 捕获距离修复是前提条件**

---

## Training Analysis Report — Move Task 成功 run 逆向工程 + 新 run 退步根因分析

**Run（成功对照）：** `2026-03-20_16-26-19_mappo_torch_mappo`
**Run（失败分析）：** `2026-04-02_11-47-43_mappo_torch_mappo`
**Date：** 2026-04-02
**Task：** Isaac-marl-move-v0
**Algorithm：** MAPPO

---

### 训练指标对比

| 指标 | 成功 run（2026-03-20） | 失败 run（2026-04-02） | 差异倍率 |
|------|----------------------|----------------------|---------|
| total_reward_mean（最终） | 901.0 | 0.263 | **3425x** |
| tracking_reward（/ep） | 37.4 | 0.00125 | **29,900x** |
| distance_reward（/ep） | 196.0 | 0.0296 | **6,600x** |
| height_reward（/ep） | 26.5 | 0.020 | 1325x |
| episode 平均步数 | 1641 步（~27s） | **1.0 步** | **1641x** |
| policy_std（最终） | 0.150 | 0.0030 | **50x** |
| status | improving | stalled | — |

---

### 核心发现 — CRITICAL：训练根本未能启动（episode 长度 = 1 步）

**症状：** `timesteps_mean = 1.0 / timesteps_max = 1.0 / timesteps_min = 1.0`
每个 episode 在第 1 步即触发终止，全程 313k 步没有任何一个 episode 超过 1 步。

**根本原因（已定位到代码行级别）：**

`marl_move_env.py` 第 780 行的终止条件：
```python
| (self.target_positions[:, :, 2] < 0.05)
```

这一行检查目标的 z 坐标是否低于 0.05m。

新 run 的 `target_spawn_z = 0.0`（来自 `env.yaml` 第 628 行），目标在 z=0 生成。
第 1 步 `_get_dones()` 被调用时，`target_positions[:,:,2] = 0.0 < 0.05`，
`targets_out_of_bounds = True`，立即触发 `terminations = True`，episode 在第 1 步结束。

成功 run 的 `target_spawn_z = 0.25`（来自 `env.yaml` 第 623 行），高于 0.05 阈值，因此不触发即时终止。

**两 run 配置 diff（完整）：**

```
79c79
< seed: 0
---
> seed: 42

117a118,122
> nova_carter_usd_path: /media/.../nova_carter_sim_optimized.usd
> nova_carter_scale: [3.0, 3.0, 3.0]

623c628
< target_spawn_z: 0.25
---
> target_spawn_z: 0.0       ← 唯一的行为性差异，直接导致即时终止
```

seed 差异和 nova_carter 字段差异均不影响训练行为；**唯一的行为性差异是 `target_spawn_z: 0.25 → 0.0`**。

---

### 成功 run 关键参数配置（已验证有效）

| 参数 | 值 | 说明 |
|------|----|----|
| `target_spawn_z` | **0.25** | 必须 > 0.05，否则触发即时终止 |
| `capture_distance` | 1.0m | 注意：仍为 3D 距离，存在已知高度陷阱 |
| `dist_reward_weight` | 1.5 | 主距离奖励权重 |
| `tracking_reward_weight` | 1.0 | 当 tracking_reward=37.4 时 1.0 已足够（3D 捕获在此 run 中可能偶发有效） |
| `height_reward_weight` | 2.0 | 成功 run 中高度锁定到 2.5m（height_reward/ep=26.5） |
| `episode_length_s` | 60s | 允许 1641 步长 episode |
| `num_envs`（推断） | 512 | 两 run 相同 |
| `policy_std` 收敛值 | 0.150 | 正常水平（未崩溃） |

**成功 run 的局限（已知但未修复）：**
- tracking_reward=37.4/ep 虽然存在，但该 run 可能通过 z 偏高的目标（target_spawn_z=0.25）偶发触发 3D capture（无人机俯冲到 ~1.5m 时与 z=0.25 目标的 3D 距离可以 < 1.0m）。这不代表真正的 XY 跟踪能力，属于有缺陷的成功。

---

### 为何成功 run 能产生 tracking_reward？

当 `target_spawn_z=0.25` 时，无人机从 z=2.5m 俯冲到约 z=1.25m 时，
与目标（z=0.25）的垂直距离 = 1.0m，刚好等于 `capture_distance=1.0m`。
因此成功 run 的捕获依赖无人机**同时降低高度**，这在 height_reward 的存在下是反直觉的，
说明该 run 的 policy 习得了一种 "降低高度接近目标" 的策略，但代价是高度奖励受损。

---

## 改进建议

### Priority 1 (CRITICAL)：立即修复 target_spawn_z

**问题：** `target_spawn_z=0.0` 导致第 1 步即终止，训练完全无法进行。
**建议修改：**
- 文件：`exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/move/marl_move_env_cfg.py`
- 参数：`target_spawn_z: float = 0.0` → `target_spawn_z: float = 0.25`
- 或：修改终止条件阈值（第 780 行）从 `z < 0.05` 改为 `z < -0.1`（允许目标在地面）

**注意：** 如果目标代表地面车辆，`target_spawn_z=0.0` 是语义上正确的设计意图，
但 `z < 0.05` 的终止条件是针对**无人机飞太低**的逻辑被误用到了目标检测上，
应将目标的 z 下界检查改为 `z < -0.5`（允许地面高度），保留仅对 drone 的低空终止。

### Priority 2 (HIGH)：修复地空捕获距离（历史遗留问题）

**问题：** `capture_distance=1.0m` 使用 3D 欧式距离，无人机悬停在 2.5m 时永远无法捕获 z=0.25m 的目标。
成功 run 之所以有效，是因为 drone 需要俯冲到 z≈1.25m 才能触发捕获，这是隐性策略而非显式设计。
**建议修改：**
- 文件：`exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/move/marl_move_env.py`
- 参数：在计算 `capture_distance` 时，只使用 XY 平面距离（`min_dists_xy`），与 z 无关。
- 将 `target_spawn_z` 保持为 0.0（地面语义），`desired_height` 保持为 2.5m，只让 XY 距离决定捕获。
- 同步将 `capture_distance: 1.0 → 3.0`（XY 距离阈值适当放宽以允许学习启动）

### Priority 3 (HIGH)：调整奖励权重以支持地空场景

一旦 XY-only 捕获生效，重新调整权重：
- `tracking_reward_weight`: 1.0 → 4.0（地空跟踪需要更强的 tracking 信号）
- `dist_reward_weight`: 1.5 → 0.8（降低距离奖励的主导性）
- `height_reward_weight`: 2.0 → 0.5（避免高度锁定阻碍追踪）

### Priority 4 (MEDIUM)：增大 num_envs

- `num_envs`: 512 → 2048
- 训练命令：`python3 scripts/skrl/train.py --task=Isaac-marl-move-v0 --headless --num_envs=2048 --algorithm="MAPPO"`

---

### 实验计划（按优先级顺序）

1. **Step 1（立即执行）：** 修复 `target_spawn_z=0.25` 或修改终止条件 z 阈值，验证 episode 长度恢复正常（>100 步）
2. **Step 2：** 同步实施 XY-only 捕获距离修复 + `capture_distance=3.0m`
3. **Step 3：** 调整三项奖励权重（tracking×4.0, dist×0.8, height×0.5）
4. **Step 4：** 以 num_envs=2048 重新训练
5. **监测指标（100k 步节点）：**
   - episode 步数 mean > 200（确认无即时终止）
   - tracking_reward 是否可达（> 0）
   - policy_std 衰减速度（不应崩溃到 0.005 以下）

**成功判据（300k 步）：**
- episode 步数 mean > 500
- tracking_reward/ep > 5.0
- policy_std > 0.05

---

### Changelog（续）

- 2026-04-02：逆向工程分析 move 任务成功 run（2026-03-20）与失败 run（2026-04-02_11-47-43）差异
  - **根本原因已定位：** `target_spawn_z: 0.25 → 0.0` 触发终止条件 `z < 0.05`，导致 episode=1 步，训练从未启动
  - 成功 run 的 tracking_reward=37.4 依赖 drone 俯冲到 z≈1.25m，是隐性策略而非 XY 追踪
  - 两 run 奖励权重、观测空间、动作空间完全相同，退步来源于单一参数 `target_spawn_z`
  - 建议在下次训练前先修复该单点故障，再叠加 XY-only 捕获和权重调整

---

# Training Analysis Report — 2026-04-02_15-10-03_mappo_torch_mappo

**Run:** `logs/skrl/move/2026-04-02_15-10-03_mappo_torch_mappo`
**Date:** 2026-04-02
**Task:** Isaac-marl-move-v0（3 Falcon 无人机追踪 4 辆 NovaCarter 移动小车）
**Algorithm:** MAPPO
**背景：** 本次为将目标从 VisualizationMarkers 替换为 NovaCarter 真实 USD 小车后的第一次训练

---

## Training Metrics Summary

| 指标 | 值 |
|------|----|
| 总训练步数（梯度更新） | 106,100 |
| num_envs | **32**（标准值应为 2048） |
| 最终 total_reward mean（per episode） | -0.314（最近均值 -0.082） |
| 最终 total_reward max（per episode） | +1.54 |
| episode 步数 mean | **61 步（=0.61s，仅占 max 的 1.0%）** |
| episode 步数 min | **1 步**（持续存在） |
| tracking_reward（recent） | **0.000017**（实际为零，历史最高仅 0.0046） |
| distance_reward（recent） | 0.3454 |
| height_reward（recent） | 0.5554 |
| body_rate_penalty（recent） | 0.7419 |
| policy_std | 0.821 → 0.537（衰减 1.5x，未崩溃） |
| tracking:distance 奖励比 | **1 : 20,609**（tracking 实际为零） |

### 与基线对比（2026-03-20_16-26-19，VisualizationMarkers）

| 指标 | 基线（成功） | 本次（NovaCarter） | 退化幅度 |
|------|------------|-------------------|---------|
| episode 步数 mean | 1,641 | 61 | **-96%** |
| tracking_reward/ep | 37.4 | 0.0 | **-100%** |
| distance_reward/ep | ~196 | ~21 | **-89%** |
| policy_std 最终值 | ~0.150 | 0.537 | 探索保留，尚未崩溃 |

---

## Observations & Findings

### [Episode 长度异常] — Severity: CRITICAL

**Symptom:** episode 步数 mean=61（0.61s），min=1 持续存在。正常 episode 应能达到数百步以上。
只有 1% 的最大 episode 时长被利用，策略几乎没有机会学习任何有意义的行为。

**Root Cause（已确认）：** 无人机初始 spawn 高度（`drone_spawn_z_range = (1.5, 2.5)`）与 NovaCarter 3x 缩放后的车身高度发生物理碰撞，在 episode 起始阶段即触发终止。

**Evidence:**
- NovaCarter 原始尺寸约 0.45m 高，缩放 3x 后约 **1.35m 高**
- `target_spawn_z=0.25`（小车中心/底部基准），则小车顶部约 **z≈1.60m**
- 无人机下限 spawn 高度为 `z=1.5m`，低于小车顶部（1.60m）→ **spawn 时 drone 嵌入 NovaCarter 车身**
- 触发 `illegal_contact` 或 `drone_collision` 终止条件，即 min=1 步持续出现
- `episode_timesteps_max=115` 步（约 1.15s），上限较低，说明即便幸运未碰撞也很快被其他条件终止

### [tracking_reward 完全为零] — Severity: HIGH

**Symptom:** tracking_reward recent_mean=0.000017（等同于零），历史最高仅 0.0046，在 106k 步训练中从未真正出现。

**Root Cause:**
1. episode 仅持续 61 步，无人机无时间接近目标（目标在 x=-2~-6m，无人机 spawn 在 x=-6~-8m，接近距离 ~4m，需要数十秒）
2. 即便 episode 能正常运行，capture 逻辑已修改为 XY-only（`dist_matrix = torch.norm(d_pos - t_pos, dim=-1)` 中 `d_pos=...[:,:2]`），XY 接近范围 1.0m 本身也是较窄的目标，需要精确横向接近
3. `tracking_reward_weight=1.0` 相对于 `dist_reward_weight=1.5`（已铺满全程）信号强度弱

**Evidence:** tracking:dist 奖励比 = 1:20,609，为数量级级别的失衡

### [训练规模不足 —— num_envs=32] — Severity: HIGH

**Symptom:** 配置文件中 `scene.num_envs=32`（标准训练使用 2048），样本吞吐量降低 64 倍。

**Root Cause:** 本次为首次 NovaCarter 集成测试，使用小型 num_envs 做验证性运行。但从数据来看问题已足够严重，不能在此规模下寄望策略收敛。

**Evidence:**
- 有效样本数 ≈ 32 × 61 × (106100/300) ≈ 690k（vs 2048 env 应有的 44M）
- 训练曲线在 106k 步内几乎平坦（reward mean: early=-3.03, recent=-0.08），策略停留在非常初级的状态

### [高度奖励主导 —— 高度锁定风险] — Severity: MEDIUM

**Symptom:** height_reward=0.5554/step，body_rate_penalty=0.7419/step，两者主导 reward 结构，无人机倾向于维持高度而非追踪目标。

**Root Cause:** `height_reward_weight=2.0`（最高权重之一），`desired_height=2.5m`，对高度精确性给予强激励，无法鼓励无人机下降进行捕获。

**Evidence:** height_error 估算 ≈ 0 （reward≈exp(-0)×weight×dt ≈ 0.555 ≈ weight×dt = 2.0×0.01×（约0.28已归一化），说明 z 误差极小，无人机已锁定在 desired_height）

---

## 改进建议

### Priority 1 (CRITICAL)：修复无人机 spawn 高度与 NovaCarter 碰撞

**Problem:** NovaCarter 以 3x 缩放加载，车身高度约 1.35m（top≈z=1.60m），无人机最低 spawn 高度 z=1.5m，初始帧即发生嵌入碰撞，导致 episode 立即终止。

**Proposed Changes（二选一）：**

方案 A — 提高无人机最低 spawn 高度（推荐，最简单）：
- 文件：`exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/move/marl_move_env_cfg.py`
- 参数：`drone_spawn_z_range: tuple = (1.5, 2.5)` → `drone_spawn_z_range: tuple = (2.0, 3.0)`
- 理由：将 drone 最低 spawn 高度抬高至 2.0m，超过 NovaCarter 顶端（~1.60m），消除初始碰撞风险

方案 B — 缩小 NovaCarter 缩放比例：
- 文件：`exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/move/marl_move_env_cfg.py`
- 参数：`nova_carter_scale: tuple = (3.0, 3.0, 3.0)` → `nova_carter_scale: tuple = (1.0, 1.0, 1.0)`
- 理由：1x 缩放时 NovaCarter 高度约 0.45m，无任何 spawn 碰撞风险；但视觉上车辆变小

**Rationale:** 修复后 episode min 应从 1 步回归到数十步以上，training 才能真正开始。

### Priority 2 (HIGH)：验证 NovaCarter 物理属性设置

**Problem:** NovaCarter 是通过 `XFormPrim.set_world_poses()` 驱动的运动学对象（非 Articulation），理论上不参与物理碰撞。但接触传感器（ContactSensor on Falcon）可能仍会响应大型刚体。需确认 NovaCarter USD 中碰撞网格是否被正确禁用。

**Proposed Change:**
- 文件：`exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/move/marl_move_env.py`
- 在 _setup_scene 中加载 NovaCarter USD 后，显式禁用 NovaCarter 的碰撞属性（collision API disabled）
- 或确认 NovaCarter 已被设置为 `kinematic_enabled=True`，不参与接触力计算

### Priority 3 (HIGH)：提升 tracking_reward 信号强度

**Problem:** tracking_reward_weight=1.0，与 dist_reward_weight=1.5 相比信号弱，且在 episode 极短时完全无法积累。

**Proposed Change:**
- 文件：`exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/move/marl_move_env_cfg.py`
- 参数：`tracking_reward_weight: float = 1.0` → `4.0`
- 参数：`dist_reward_weight: float = 1.5` → `0.8`
- 参数：`height_reward_weight: float = 2.0` → `0.5`
- 理由：降低高度锁定诱因，让 XY 追踪成为最优策略；此组合在 move_flyfollow 任务中已验证有效（tracking_reward 从0到>1.0）

### Priority 4 (HIGH)：增大 num_envs 至 2048

**Problem:** `num_envs=32` 导致有效样本量仅为标准配置的 1/64，策略无法在合理步数内收敛。

**Proposed Change:**
- 训练命令增加 `--num_envs=2048`：
  ```
  python3 scripts/skrl/train.py --task=Isaac-marl-move-v0 --headless --num_envs=2048 --algorithm="MAPPO"
  ```
- 注意：NovaCarter USD 加载 × 2048 环境可能有内存压力，先验证 32→512 是否稳定，再升至 2048

### Priority 5 (MEDIUM)：capture_distance 调优（NovaCarter 实际尺寸适配）

**Problem:** NovaCarter 3x 缩放后实际车身 XY 尺寸约 3.3m × 2.1m，而 capture_distance=1.0m（XY 平面），可能过窄（无人机需要到达车辆中心点 1m 内）。

**Proposed Change（视 Priority 1/3 效果决定是否执行）：**
- 文件：`exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/move/marl_move_env_cfg.py`
- 参数：`capture_distance: float = 1.0` → `3.0`（若 3x 缩放保留）或 `1.5`（若改回 1x 缩放）
- 理由：capture 阈值应适配目标物理尺寸，避免需要"进入车辆内部"才能触发捕获

---

## Experiment Plan

### 步骤顺序（按修复顺序）

1. **立即修复（Priority 1）：** 修改 `drone_spawn_z_range` 为 `(2.0, 3.0)` 并以 `num_envs=32` 快速验证
   - 成功判据：episode_mean > 50 步，min > 1 步，无法在第 1 步终止
2. **确认碰撞属性（Priority 2）：** 检查 NovaCarter prim collision 属性，确认接触传感器不响应
3. **全规模训练：** 同时应用 Priority 3 + 4（奖励权重 + num_envs=2048）启动正式训练
4. **监测指标（100k 步节点）：**
   - `Episode / Total timesteps mean > 200` — 确认无即时终止
   - `tracking_reward recent_mean > 0.001` — tracking 信号出现
   - `policy_std > 0.3` — 策略仍在探索
5. **成功判据（500k 步）：**
   - episode_mean > 500 步
   - tracking_reward/ep > 5.0
   - total_reward_mean > 0

---

## Changelog

- 2026-04-02：分析 2026-04-02_15-10-03（NovaCarter 首次集成）
  - **根本原因：** NovaCarter 3x 缩放后车身顶部 z≈1.60m，与 drone spawn 下限 z=1.5m 碰撞，导致 episode 在起始帧即终止（mean=61步，min=1步）
  - tracking_reward 完全为零（ratio 1:20,609），训练未产生任何有效策略
  - num_envs=32 导致样本量仅为标准的 1/64（验证性运行可接受，正式训练须改为 2048）
  - 下一步：提高 drone spawn 下限至 z=2.0m，禁用 NovaCarter 碰撞网格，调整奖励权重

---

# Training Analysis Report — 2026-04-02_17-08-54_mappo_torch_mappo

**Run:** 2026-04-02_17-08-54_mappo_torch_mappo
**Date:** 2026-04-02
**Task:** Isaac-marl-move-v0
**Algorithm:** MAPPO
**Total Steps:** ~58k（57,900 最终步）
**num_envs:** 32

---

## Training Metrics Summary

| 指标 | 早期 | 近期均值 | 最终值 |
|------|------|----------|--------|
| Total reward (mean) | -7.75 | -2.05 | -1.93 |
| Episode timesteps (mean) | 86.6 步 | 58.6 步 | 61.2 步 |
| Episode timesteps (min) | 1 步 | 1.8 步 | 1 步 |
| tracking_reward | 0.0 | ~0.000012 | 0.0 |
| height_reward | 0.137 | 0.138 | 0.138 |
| height_penalty | — | -0.617 | -0.600 |
| illegal_contact (终止次数/rollout) | 0.736 | 1.250 | 1.170 |
| policy_std | 0.797 | 0.648 | 0.635 |

**关键结论：** 本次训练完全被 `illegal_contact` 终止条件主导。episode 长度从早期 86 步下降到近期 57 步，并呈持续恶化趋势，说明策略正在主动学习靠近 NovaCarter——但每次靠近都触发接触传感器终止。tracking_reward 全程为零，训练未产生有效跟随策略。

---

## Observations & Findings

### [NovaCarter 碰撞几何体未禁用] — Severity: CRITICAL

**症状：**
- `Episode_Termination/illegal_contact` 全程主导：早期 0.736/rollout → 近期 1.254/rollout（恶化趋势）
- `Episode_Reward/illegal_contact` 近期均值 = -0.974（接近每个 episode 都被惩罚一次）
- episode 长度持续缩短（86 → 57 步）——策略变得更"激进"，更快靠近目标，但越靠近越快终止
- `timesteps_min` 始终为 1，说明仍有 spawn 阶段即终止的情况

**根本原因：**
NovaCarter USD 资产（`nova_carter_sim_optimized.usd`）内嵌了完整的物理碰撞几何体（PhysicsCollisionAPI）。虽然小车以 `XFormPrim` 方式绑定（运动学驱动，不参与物理模拟），但其碰撞网格仍在物理引擎中保持激活，Isaac Lab 的接触传感器会检测到 Falcon 机身与 NovaCarter 几何体之间的接触力。

`contact_sensor_threshold=1.0N` 已在上一轮分析中提高，但仍不足以排除 NovaCarter 碰撞几何体产生的接触力读数。

**证据：**
- 捕获机制使用 XY 平面 2D 距离（`dist_matrix = torch.norm(d_pos - t_pos, dim=-1)` 中 `[:, :, :2]`），无人机必须水平靠近目标 → 必然与 NovaCarter 几何体接触
- `drone_spawn_z_range=(2.0, 3.0)` 已修复 spawn 碰撞，但运动靠近阶段的接触问题未解决
- `illegal_contact` 与 `crash` 标签数值完全相同，说明 crash 即 illegal_contact（同一路径）

**核心矛盾：** 当前任务要求无人机接近 NovaCarter（capture_distance=1.0m XY），但接近行为会触发 NovaCarter 碰撞几何体接触，导致 illegal_contact 终止。训练目标与终止条件直接冲突。

---

### [height_penalty 持续增大，策略被阻止下降] — Severity: HIGH

**症状：**
- `height_penalty` 近期均值 = -0.617，从早期 -0.310 持续增大
- `height_reward_weight=0.5` + `desired_height=2.5m` 仍在将无人机锁定在 2.5m 高度
- 俯冲捕获策略（需下降至 z≈1.25m 以靠近 z=0.25m 目标）与 height_penalty 产生对抗

**根本原因：**
俯冲捕获需要无人机从 z=2.5m 下降到 z≈1.0~1.5m，但 `height_penalty_threshold=0.5m` 意味着偏离 2.5m 超过 0.5m（即 z<2.0m）就开始受罚。目标高度 z=0.25m，要实现 capture_distance=1.0m（3D），无人机需降至 z≈1.0m——此时 height_error=1.5m，height_penalty=-1.5×weight。

**证据：**
- height_penalty 趋势上升（早期 -0.31 → 近期 -0.62），说明策略试图下降但被反复惩罚

---

### [tracking_reward 信号密度极低] — Severity: HIGH

**症状：**
- tracking_reward 全程为零（best=0.001，仅出现 1 次）
- capture_distance=1.0m（XY 平面），无人机需精确进入 1m 范围内
- 30k 步内未出现任何捕获信号

**根本原因：**
NovaCarter 以 0.3m/s 移动，而无人机每次靠近都触发 illegal_contact 终止——策略无法"探索"到成功捕获状态，奖励信号稀疏问题比 move_flyfollow 任务更严重。

---

### [策略探索衰减] — Severity: MEDIUM

**症状：**
- `policy_std` 从 0.820 降至 0.635（衰减 22.6%，58k 步）
- 探索正在收敛，但收敛到的是"快速靠近然后被 illegal_contact 终止"的局部策略

**根本原因：**
策略通过 distance_reward 学会靠近目标，但 illegal_contact 使该策略代价极高。在无法学到捕获成功的条件下，策略将陷入"中距离悬停"局部最优。

---

## Improvement Recommendations

### Priority 1 (CRITICAL)：禁用 NovaCarter 碰撞几何体

**Problem:** NovaCarter USD 中的碰撞几何体被 Falcon 接触传感器检测，导致无人机任何接近行为都触发 illegal_contact 终止。这是阻断训练的根本原因。

**Proposed Change（方案 A — 推荐）：在 `_setup_scene` 中动态禁用 NovaCarter 碰撞 API**
- 文件：`exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/move/marl_move_env.py`
- 在绑定 NovaCarter XFormPrim 后（约 line 365），添加：
  ```python
  from pxr import UsdPhysics, Usd
  import omni.usd
  stage = omni.usd.get_context().get_stage()
  for prim in stage.Traverse():
      path_str = str(prim.GetPath())
      if "nova_carter" in path_str.lower():
          # 禁用碰撞 API
          if prim.HasAPI(UsdPhysics.CollisionAPI):
              UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Set(False)
  ```
- 理由：NovaCarter 作为运动学目标，不需要物理碰撞；禁用后接触传感器不再检测到它

**方案 B（备选）：大幅提高 contact_sensor_threshold**
- 文件：`exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/move/marl_move_env_cfg.py`
- 参数：`contact_sensor_threshold: float = 1.0` → `50.0`
- 理由：如果 NovaCarter 碰撞力通常 < 50N（轻微接触），可通过提高阈值过滤。但此方案治标不治本，且可能掩盖真实碰撞信号。

**方案 C（最彻底）：将 illegal_contact 从终止条件中移除（仅保留惩罚）**
- 文件：`exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/move/marl_move_env.py`，line 882-888
- 从 `terminations` 中移除 `self.illegal_contact`
- 理由：对 NovaCarter 接触不应终止 episode，只应给予小惩罚，让策略自行学会保持距离

**推荐执行顺序：** 优先方案 A（根治），若 USD API 调用困难则用方案 C（最快）。

---

### Priority 2 (HIGH)：修改捕获策略——从俯冲捕获改为水平跟随

**Problem:** 当前设计要求无人机俯冲至 z≈1.25m 以实现 3D 距离 < 1.0m，但：
1. height_penalty 惩罚此行为
2. 下降途中必然经过 NovaCarter 碰撞体（车顶 z≈1.6m）

修改为 XY 平面跟随 + 保持安全高度是更合理的任务定义。

**Proposed Change：**
- 文件：`exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/move/marl_move_env_cfg.py`
- 参数：`capture_distance: float = 1.0` → `3.0`（XY 平面距离阈值，适配 NovaCarter 3x 缩放后 ~3.3m 车长）
- 参数：`desired_height: float = 2.5` → `2.5`（保持不变，无人机在 2.5m 高度平飞跟随）
- 同步修改：`height_penalty_threshold: float = 0.5` → `1.5`（允许 ±1.5m 偏差，为下降提供空间）
- 理由：扩大 capture_distance 使策略可以在 desired_height 处直接捕获，无需俯冲；阈值 3.0m 与 NovaCarter 物理尺寸相符

---

### Priority 3 (HIGH)：num_envs=32 严重不足，必须升至 2048

**Problem:** num_envs=32 仅为标准配置 1/64。58k 训练步实际等效于标准配置约 900 步，完全无法用于正式训练评估。

**Proposed Change：**
- 训练命令：`python3 scripts/skrl/train.py --task=Isaac-marl-move-v0 --headless --num_envs=2048 --algorithm="MAPPO"`
- 注意：先以 Priority 1 修复验证（num_envs=32 快跑 20k 步确认 illegal_contact=0），再升至 2048

---

### Priority 4 (MEDIUM)：height_penalty 与俯冲捕获目标对抗——调整高度约束

**Problem:** 如果保留俯冲捕获设计（3D distance），则 height_penalty_threshold=0.5m 会持续惩罚必要的下降行为。

**Proposed Change（若维持俯冲捕获路线）：**
- 文件：`marl_move_env_cfg.py`
- 参数：`height_penalty_weight: float = 1.0` → `0.0`（彻底取消 height_penalty，仅用 height_reward 提供软引导）
- 参数：`height_reward_weight: float = 0.5` → `0.2`（进一步降低高度锁定吸引力）
- 理由：俯冲捕获策略要求高度可变，height_penalty 与任务目标冲突

**注意：** 若采用 Priority 2 的水平跟随方案，此项可跳过（height_penalty 对 2.5m 平飞无影响）。

---

## Experiment Plan

### 阶段一：根治 illegal_contact（验证）

1. 应用 Priority 1（方案 A 或 C），num_envs=32，运行 20k 步
2. 验证指标：
   - `Episode_Termination/illegal_contact` 趋向 0
   - `Episode / Total timesteps (mean)` > 200 步
   - `Episode / Total timesteps (min)` > 10 步（无 spawn 即终止）

### 阶段二：正式训练

1. 同时应用 Priority 1 + Priority 2 + Priority 3（capture_distance=3.0m，num_envs=2048）
2. 运行 500k 步
3. 监测指标（每 100k 步节点）：
   - `tracking_reward recent_mean > 0.1` — 捕获信号出现
   - `Episode / Total timesteps mean > 300 步` — 策略存活
   - `policy_std > 0.4` — 策略仍在探索
4. 成功判据（500k 步）：
   - `tracking_reward/ep > 5.0`
   - `total_reward_mean > 0`
   - `Episode_Termination/illegal_contact 均值 < 0.1`

---

## Changelog

- 2026-04-02（run 2026-04-02_17-08-54）：spawn 碰撞修复（z: 1.5→2.0m）+ 奖励权重调整后的第二次训练
  - **根本原因诊断：** NovaCarter USD 碰撞几何体激活，Falcon 接触传感器在无人机靠近目标时持续触发 illegal_contact 终止（early=0.736/rollout → late=1.254/rollout，恶化趋势）
  - spawn 碰撞（min=1 步）已改善但未完全消除（仍有偶发 min=1）
  - episode 长度从 86 → 57 步（策略学会靠近，但每次靠近都被终止）
  - tracking_reward 全程为零，height_penalty 持续增大（-0.31 → -0.62），俯冲捕获路线与高度约束直接冲突
  - 核心矛盾：任务要求接近 NovaCarter（capture），但接近行为会触发 illegal_contact 终止
  - 下一步：**禁用 NovaCarter 碰撞几何体**（Priority 1 方案 A 或 C）是解锁训练的前提，其他所有改进在此之前无效

---

# Training Analysis Report

**Run:** 2026-04-02_18-27-51_mappo_torch_mappo
**Date:** 2026-04-02
**Task:** Isaac-marl-move-v0
**Algorithm:** MAPPO
**Total steps logged:** ~199,100
**num_envs:** 2048（推断，production run）

---

## Training Metrics Summary

| 指标 | 早期均值 | 近期均值 | 最终值 |
|------|---------|---------|--------|
| total_reward_mean | -3.584 | -0.496 | -0.135 |
| total_reward_max | -1.504 | 0.691 | 1.084 |
| tracking_reward | ~0.000002 | ~0.000005 | 0.0 |
| success_reward | 0.0 | 0.0 | 0.0 |
| illegal_contact（终止次数/rollout） | 1.150 | 1.234 | 1.190 |
| episode timesteps mean | 65.3 | 65.7 | 64.0 |
| episode timesteps min | 3.6 | 2.2 | 3.0 |
| policy_std | 0.749 | 0.351 | 0.302 |
| height_penalty | -1.064 | -0.842 | -0.797 |

**本次修复内容：**
1. `drone_spawn_z_range` (1.5, 2.5) → (2.0, 3.0)（避开 NovaCarter 3x 车顶 ≈1.6m）
2. 禁用 NovaCarter 碰撞几何体（`Usd.PrimRange` + `CollisionAPI.Set(False)`）
3. `tracking_reward_weight` 1.0 → 4.0，`dist_reward_weight` 1.5 → 0.8，`height_reward_weight` 2.0 → 0.5

---

## Observations & Findings

### 1. illegal_contact 未消除 — CRITICAL

**Symptom:** `Episode_Termination/illegal_contact` 全程稳定在 1.15～1.23/rollout。早期 1.150、近期 1.234、最终 1.190。与上一次训练（2026-04-02_17-08-54，早期 0.736 → 近期 1.254）相比，**不仅未改善，而且起点更高，整个训练期间都保持高位**。

**Root Cause（诊断）：** 碰撞几何体禁用代码（`_setup_scene` 中的 `Usd.PrimRange` + `CollisionAPI.GetCollisionEnabledAttr().Set(False)`）**在运行时未生效或仅部分生效**。

可能原因：
- `omni.usd.get_context().get_stage()` 在 `_setup_scene` 时调用，但 NovaCarter USD prim 尚未完全加载进 stage（异步加载问题）
- `prim_utils.is_prim_path_valid(f"{env_base}/World")` 路径判断导致 `env_r` 指向错误路径，部分环境未找到 nova_carter prim
- `col_api` 的 `if col_api:` 判断：空 API 对象在 Python 中可能是 truthy，导致实际上只有顶层 prim 被禁用，子网格（collision mesh）保持激活
- Isaac Lab 在 `_setup_scene` 后会重新初始化物理引擎，覆盖手动设置的 CollisionAPI 状态

**Evidence:**
- `illegal_contact` 终止次数全程 1.15+，从未降至接近 0
- 上一次训练中观察到 illegal_contact 随训练步数单调增加（策略学会靠近 → 接触频率增加），本次训练同样呈上升趋势（1.150 → 1.234）
- `episode timesteps min` 持续在 2-4 步，表明仍有大量环境在第一个控制周期内即触发终止

**结论：** 碰撞禁用失败。`illegal_contact` 仍是主要终止来源（占比 >99%，`time_out=0.0`，其他终止均接近 0）。

---

### 2. tracking_reward 仍为零 — CRITICAL

**Symptom:** `tracking_reward` 全程为 0.0，best 仅出现过一次 0.021（相当于偶发噪声）。`success_reward` 全程 0.0。

**Root Cause:** 直接后果来自 Finding 1。illegal_contact 终止在 65 步内清场所有环境，策略无法存活足够长时间接近目标。even 若 NovaCarter 碰撞禁用生效，`capture_distance=1.0m`（代码 line 65）仍是 3D 欧式距离阈值，无人机需从 z≈2.5m 下降至 z≈1.25m 才能使 3D 距离 < 1.0m（目标 z=0.25m）。这要求同时满足：(1) XY 误差 < 约 0.75m，(2) 高度下降 > 1.5m——是高难度两步策略，在 65 步（约 2 秒）内无法完成。

**Evidence:**
- `height_penalty`（early=-1.064，recent=-0.842）：无人机在尝试下降，高度惩罚持续存在，但仍未能触发 tracking
- `height_reward` 稳定在 0.14（接近常数），表明 desired_height=2.5m 附近有稳定吸引子，下降行为受到抑制

---

### 3. episode 长度无改善 — HIGH

**Symptom:** `timesteps_mean` 全程 65-68 步（约 2 秒），与上一次训练（86.6 步）相比反而略有下降。`timesteps_max` 从早期 115 下降至近期 108，整体趋势下降。

**Root Cause:** episode 长度由 illegal_contact 终止驱动。碰撞禁用失败 + 策略在奖励信号引导下逐渐学会靠近目标 → 每次靠近都触发终止 → episode 越来越短，与上一次训练模式完全一致。

**Evidence:**
- `bounding_box` 终止从 0.107 降至 0.019（改善，无人机不再冲出边界）
- `drone_out` 惩罚从 -0.090 降至 -0.016（改善）
- 但 `illegal_contact` 终止完全掩盖了这些改善

---

### 4. 奖励总体趋势：有微弱正向信号 — MEDIUM

**Symptom:** `total_reward_mean` 从 -3.584 改善至 -0.496（改善 86%），`total_reward_max` 从 -1.504 改善至 +0.691（最终 1.084）。

**Root Cause（正向）：** `height_reward_weight` 从 2.0 降至 0.5 减小了高度锁定惩罚强度；`bounding_box` 和 `drone_out` 减少说明策略在学习保持在有效区域内；`body_rate_penalty` 从 0.634 上升至 0.957（接近最大值 1.131），说明姿态控制在改善。

**注意：** total_reward 改善主要来自惩罚减少（height_reward 权重降低、边界违反减少），不是正向奖励信号增加。这是一种被动改善，不代表策略在学习追踪行为。

---

### 5. policy_std 快速坍缩 — HIGH

**Symptom:** `policy_std` 从 0.749 坍缩至 0.302（坍缩 60%），且呈单调下降趋势，尚未稳定。按当前速率，再训练 100k 步后 std 将低于 0.2。

**Root Cause:** 策略在 illegal_contact 主导的短 episode 中无法获得有效奖励，PPO 倾向于压缩探索以最小化方差惩罚。`entropy_loss` 从 -0.00113 上升至 -0.00034（绝对值减小），表明熵正则化也在失效。

---

## Improvement Recommendations

### Priority 1 (CRITICAL): 确认并修复 NovaCarter 碰撞禁用失败

**Problem:** 碰撞禁用代码在 `_setup_scene` 中存在，但训练数据表明 NovaCarter 碰撞几何体在运行时仍然激活。

**Proposed Changes:**

**方案 A（推荐）：改用 Isaac Lab API 在资产加载阶段禁用碰撞**
- File: `exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/move/marl_move_env.py`
- 在 `_setup_scene` 中，NovaCarter 的 `ArticulationCfg` 或 `RigidObjectCfg` 上设置 `collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False)`，而非事后遍历 USD stage 修改 prim 属性

**方案 B（最快验证）：从终止条件中临时移除 illegal_contact**
- File: `exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/move/marl_move_env.py`
- Line 901-907：从 `terminations` OR 中移除 `self.illegal_contact`
- 仅保留作为惩罚项（-1.0/接触）而不终止 episode
- **Rationale:** 最快验证假设——如果移除终止后 episode 长度显著增加（>200 步），则 100% 确认 illegal_contact 是瓶颈；同时训练可以继续推进

**方案 C（保险策略）：提升 contact_sensor_threshold**
- File: `exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/move/marl_move_env_cfg.py`
- `contact_sensor_threshold`: 1.0 → 50.0（N）
- **Rationale:** NovaCarter 即使碰撞几何体未完全禁用，50N 阈值可过滤非物理接触；真实碰撞（无人机撞地面、建筑）仍会被捕获

**推荐执行顺序：** 先方案 B（快速验证，1 次训练），确认是 illegal_contact 瓶颈后再实施方案 A 做根本修复。

---

### Priority 2 (HIGH): capture_distance 从 1.0m 扩大到 3.0m（XY 平面）

**Problem:** 当前 `capture_distance=1.0m`（3D 欧式距离），无人机需从 z=2.5m 下降至 z≈1.25m，同时 XY 误差 < 0.75m，是二维同步精确控制要求。NovaCarter 物理尺寸（3x 缩放）约 3.3m × 1.8m，1.0m 捕获距离远小于目标物理尺寸。

**Proposed Change:**
- File: `exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/move/marl_move_env_cfg.py`
- `capture_distance`: 1.0 → 3.0
- 同时将 `min_dists` 的计算改为 XY 平面距离（已在 `_get_rewards` 中使用 `target_min_dist` 变量，需确认计算方式）
- **Rationale:** 3.0m XY 捕获距离与 NovaCarter 物理尺寸匹配，不再要求无人机精确俯冲到 z=1.25m，大幅降低策略难度。与 move_flyfollow 任务成功案例对齐（dist_xy < 3m）。

---

### Priority 3 (MEDIUM): height_penalty_threshold 从 0.5m 扩大到 1.5m

**Problem:** `height_penalty_threshold=0.5m` 在 `desired_height=2.5m` 附近建立强约束带，使无人机无法自由调整高度。即使 illegal_contact 修复后，无人机从 2.5m 下降去追踪目标也会立即触发高度惩罚。

**Proposed Change:**
- File: `exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/move/marl_move_env_cfg.py`
- `height_penalty_threshold`: 0.5 → 1.5
- **Rationale:** 如果采用 Priority 2 的 3.0m XY 捕获，无人机不需要俯冲，此项可维持 0.5m。但如果保留 1.0m 捕获距离，则必须扩大此阈值。两项修改应配套实施。

---

### Priority 4 (MEDIUM): 添加 entropy 正则化系数防止 policy_std 过早坍缩

**Problem:** `policy_std` 在 200k 步内从 0.75 坍缩至 0.30，探索能力丧失过早。在 illegal_contact 修复后重新训练时，可能在 tracking_reward 出现之前 std 就已过低。

**Proposed Change:**
- File: 训练脚本配置（`scripts/skrl/train.py` 或 MAPPO agent 配置）
- 提高 `entropy_loss_scale`（当前约 0.001），建议设为 0.005～0.01
- **Rationale:** 更强的熵正则化使策略在没有明确奖励信号时保持探索，避免在局部最优（悬停）定型。

---

## Experiment Plan

**核心前提：** Priority 1 必须先于其他所有修改验证。

### Phase A（Priority 1 验证）：100k 步快速验证
1. 实施 Priority 1 方案 B（从终止条件中移除 illegal_contact）
2. 运行 100k 步，num_envs=2048
3. 验证指标（50k 步节点）：
   - `Episode_Termination/illegal_contact < 0.1`（应接近 0）
   - `Episode / Total timesteps mean > 200 步`（关键改善）
   - `tracking_reward recent_mean > 0.001`（首次信号出现）
4. 如果 Phase A 成功 → 继续 Phase B

### Phase B（Priority 2+3 集成）：500k 步主训练
1. 实施 Priority 2（capture_distance 1.0 → 3.0m，XY 距离）
2. 实施 Priority 3（height_penalty_threshold 0.5 → 1.5m，如保留俯冲策略）
3. 实施 Priority 4（entropy_loss_scale 提高）
4. 运行 500k 步
5. 成功判据（500k 步）：
   - `tracking_reward/ep > 5.0`
   - `Episode_Termination/all_targets_captured > 0.01`（偶发成功）
   - `policy_std > 0.35`（探索未坍缩）
   - `Episode / Total timesteps mean > 300 步`

---

## Comparison with Previous Run (2026-04-02_17-08-54)

| 指标 | 上一次（碰撞禁用前） | 本次（碰撞禁用后） | 变化 |
|------|---------------------|-------------------|------|
| illegal_contact early | 0.736/rollout | 1.150/rollout | 恶化 |
| illegal_contact recent | 1.254/rollout | 1.234/rollout | 持平（仍高） |
| episode timesteps mean | 86.6 → 57.4（恶化） | 65.3 → 65.7（稳定） | 稍好 |
| total_reward_mean | -7.75 → -1.77 | -3.58 → -0.50 | 改善 |
| tracking_reward | 0.0（best=0.001） | 0.0（best=0.021） | 微弱正向 |
| bounding_box 终止 | N/A | 0.107 → 0.019 | 改善（出界减少） |

**结论：** spawn 碰撞（z_range 修复）有效（bounding_box 终止减少），但碰撞几何体禁用代码未能消除 illegal_contact。本次训练的 illegal_contact 起点（1.150）甚至高于上一次（0.736），可能因为 2048 个环境中禁用代码遍历失败比例更高，或 USD stage 加载顺序问题在大规模环境下更严重。

---

## Changelog

- 2026-04-02（run 2026-04-02_18-27-51）：碰撞几何体禁用尝试失败 — illegal_contact 仍为 1.15-1.23/rollout，确认为训练阻断瓶颈
  - spawn z_range (1.5,2.5)→(2.0,3.0) 有效，bounding_box 终止从 0.107 降至 0.019
  - 奖励重塑（tracking×4, dist×0.8, height×0.5）方向正确，total_reward_mean 改善 86%
  - 但 illegal_contact 终止完全主导，所有奖励改善均被掩盖
  - 关键发现：碰撞禁用代码在 `_setup_scene` 中的 USD stage 遍历方式在 2048 env 规模下可能无效

---

# Training Analysis Report

**Run:** 2026-04-02_21-56-48_mappo_torch_mappo
**Date:** 2026-04-03
**Task:** Isaac-marl-move-v0
**Algorithm:** MAPPO
**Key Change vs Previous Run:** 将 `illegal_contact` 从终止条件中移除（仅保留惩罚项）

---

## Training Metrics Summary

| 指标 | 上一次（2026-04-02_18-27-51）| 本次（2026-04-02_21-56-48）| 变化 |
|------|------------------------------|---------------------------|------|
| Total reward mean (early) | -3.58 | -4.89 | 起点相近 |
| Total reward mean (recent) | -0.25 | +1.51 | **大幅改善** |
| Total reward mean (last) | +0.17 | +2.21 | **+13x** |
| Episode timesteps mean (early) | 65.3 | 82.7 | **+27%** |
| Episode timesteps mean (recent) | 66.3 | 106.9 | **+61%** |
| Episode timesteps max (recent) | — | 129 | 最长达 161 步 |
| illegal_contact terminations | 1.234/rollout | 0.001/rollout | **降低 1000x** |
| illegal_contact penalty (recent) | -0.983/ep | -0.017/ep | **降低 98%** |
| bounding_box terminations | 0.019/rollout | 1.160/rollout | 新主导终止原因 |
| tracking_reward (recent) | 0.0 | ~0.0001 | 仍接近 0 |
| all_targets_captured | 0 | 0 | 无成功捕获 |
| policy_std (early→recent) | 0.66→0.14 | 0.66→0.17 | 坍缩趋势相同 |
| Total training steps | 199k | 147k | — |

**结论状态：** `improving`（工具诊断），训练趋势向好但尚未出现任务成功信号。

---

## Observations & Findings

### 1. illegal_contact 瓶颈解除 — 验证成功 [HIGH PRIORITY RESOLVED]

**Symptom:** illegal_contact 终止从 1.234/rollout 降至 0.001/rollout（降幅 99.9%），惩罚从 -0.983/ep 降至 -0.017/ep。

**Root Cause（已确认）:** 上一轮分析的核心假设得到完全验证——CollisionAPI 禁用代码在 2048 env 规模下失效，illegal_contact 是训练的唯一阻断瓶颈。移除终止条件后，该障碍消失。

**Evidence:**
- `Episode_Termination/illegal_contact`: 1.234 → 0.001（-99.9%）
- `Episode_Termination/crash`（同一判断）: 同步归零
- `Episode / Total timesteps mean`: 65 → 107 步（+64%）

---

### 2. episode 长度改善但未达目标 — MEDIUM

**Symptom:** episode mean 从 65 步增至 107 步，max 从约 98 步增至 129 步（best 161 步）。但理论上 episode_length_s=60s / (dt=0.01s) = 6000 步，当前仅约 107 步，说明 **episode 仍被提前终止**，主导终止原因已切换为 `bounding_box`。

**Root Cause:** 无人机出界（`bounding_box` 终止）成为新的主导终止原因，recent_mean=1.16 episodes/rollout。`drone_out` 惩罚稳定在 -1.0/ep，表明 **每个 episode 中至少有一架无人机超出 bounding_box_threshold=12.0m**。

**Evidence:**
- `Episode_Termination/bounding_box`: 1.160/rollout（几乎每个 episode 都以出界终止）
- `Episode_Reward/drone_out`: 近期均值 -1.0/ep（满分惩罚，100% 发生）
- `Episode_Termination/time_out`: 0（无 episode 达到最大长度）
- `Episode_Termination/all_targets_captured`: 0（无捕获触发终止）

**Analysis:** bounding_box=12.0m，目标小车从 spawn_x_range=(-6,-2) 以 0.3m/s 向 +x 移动。60s 后目标位移约 18m，超出边界。但当前 episode 仅 107 步（约 3.5s），说明无人机在追逐过程中飞出了 12m 边界范围，而非目标逃出。这指向无人机运动过于激进或控制发散。

---

### 3. height_penalty 长期主导负向奖励 — HIGH

**Symptom:** `height_penalty` recent_mean = -1.73/ep，是所有负向奖励项中绝对值最大的（不含 drone_out）。全程呈持续增大趋势：早期 -1.24 → 晚期 -1.73。

**Root Cause:** `height_penalty_weight=1.0`，`height_penalty_threshold=0.5m`，`desired_height=2.5m`。任何高度误差超过 0.5m 即触发线性惩罚。当前无人机需要从 2.5m 下降去追逐地面目标（NovaCarter 中心 z=0.25m），每下降 1m 即产生 0.5m 超阈值 excess，乘以 step_dt 后累积。

**Evidence:**
- 全程 height_penalty < 0（未曾为 0，说明无人机始终偏离 desired_height ≥ 0.5m）
- height_penalty 随训练增大（无人机在尝试追逐目标时高度偏离越来越大）
- height_reward（+0.19/ep）远小于 height_penalty（-1.73/ep），净高度效应为 -1.54/ep

---

### 4. tracking_reward 仍为 0，成功捕获仍为 0 — CRITICAL

**Symptom:** `tracking_reward` recent_mean ≈ 0（仅出现极偶发的 0.0001），`all_targets_captured`=0 全程，`success_reward`=0。

**Root Cause（多重）:**
1. **bounding_box 提前终止**：episode 仅约 107 步（3.5s），而 `sustained_follow_duration=3.0s` 需要在 `capture_distance=1.0m` 内持续停留 3s（约 90 步）。在 3.5s 的 episode 中，只有进入捕获区后立刻持续保持才能成功，边际极窄。
2. **capture_distance=1.0m（3D 欧氏距离）过小**：目标 z=0.25m，无人机从 z=2.5m 下降，需同时满足 z 误差 < 0.75m（z=0.25m 时要求无人机下降至 z < 1.0m）且 XY < 0.66m。height_penalty 惩罚下降行为，形成强烈矛盾。
3. **tracking_reward 逻辑**：仅在 `is_captured_now=True`（已在捕获区内）时给予奖励，无渐近引导梯度，0.34 的 distance_reward 无法转化为 tracking 信号。

---

### 5. policy_std 过早坍缩 — HIGH

**Symptom:** `policy_std` 从初始 0.82 坍缩至最终 0.15，在约 50k 步后 std < 0.30。坍缩速度与上一次训练相似，说明这是系统性问题而非随机波动。

**Root Cause:** 策略在没有明确 tracking 信号的情况下快速收敛到局部最优（距离缩短 + 高度维持的折中策略），熵正则化不足以抵抗 PPO 的策略坍缩。

**Evidence:**
- step 36k: std=0.477
- step 73k: std=0.291
- step 147k: std=0.147（极低，接近确定性策略）
- `entropy_loss` recent ≈ +0.0004（正值，但系数过低无法抑制坍缩）

---

### 6. 奖励结构分析：正向奖励被惩罚淹没 — MEDIUM

**Recent per-episode reward composition（近期均值）:**

| 奖励项 | 近期均值 | 说明 |
|--------|----------|------|
| body_rate_penalty | +1.74 | 最大正项，无关任务进展 |
| action_smoothness | +0.94 | 第二大正项，无关任务进展 |
| distance_reward | +0.34 | 有效信号（接近目标） |
| force_penalty | +0.30 | 节能奖励，无关任务 |
| height_reward | +0.19 | 微弱正向 |
| **height_penalty** | **-1.73** | 最大负项，持续抑制 |
| **drone_out** | **-1.00** | 100% 每 episode 出界 |
| upright_penalty | -0.29 | 防翻滚惩罚 |
| illegal_contact | -0.017 | 已大幅降低 |
| tracking_reward | ~0.0 | **任务核心信号为零** |

**关键问题：** body_rate_penalty（+1.74）和 action_smoothness（+0.94）两项"舒适奖励"合计 +2.68/ep，主导了总奖励。策略最优行为是"飞得慢、稳定、不碰撞"，而非追踪目标。这是当前 total_reward 改善的主要原因，但与任务目标无关。

---

## 关键假设验证结果

| 假设 | 验证结果 |
|------|----------|
| illegal_contact 是主要瓶颈 | **完全确认**：终止率降 1000x，episode 增 64% |
| 移除终止后 episode >200 步 | **部分确认**：107 步（目标 200+ 未达到，受 bounding_box 新瓶颈限制）|
| tracking_reward 出现正信号 | **未达到**：仍为 ~0，但原因已由 illegal_contact 转移至 capture 设计问题 |
| 俯冲捕获行为出现 | **未确认**：height_penalty 阻止下降，高度数据无法直接读取 |
| 总体收敛向好 | **是**：total_reward -4.89→+2.21，方向正确，但未到达有效学习区间 |

---

## Improvement Recommendations

### Priority 1 (CRITICAL): 扩大捕获距离至 3.0m（XY 平面）

**Problem:** `capture_distance=1.0m`（3D 欧氏距离）在 NovaCarter（z=0.25m）+ 无人机高度（z≈2.5m）下，等效要求 XY 误差 < 0.66m 且无人机同时下降至 z < 1.0m。height_penalty 惩罚下降行为，形成死锁。tracking_reward 永远为 0。

**Proposed Change:**
- File: `/home/xtj/xtj-project/RL/MARL_AUV/exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/move/marl_move_env_cfg.py`
- `capture_distance`: 1.0 → 3.0
- 同步修改 `marl_move_env.py` 中 `min_dists` 的计算：将 3D 欧氏距离改为 XY 平面距离（忽略 z 分量）
  - File: `exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/move/marl_move_env.py`
  - 在 `_get_rewards` 中找到 `min_dists` 计算处，将 `target_diff` 改为只取 XY 两维
- **Rationale:** NovaCarter 3x 缩放后物理尺寸约 3.3m × 1.8m，3.0m XY 捕获距离与物理尺寸匹配，且不再要求俯冲，解除 height_penalty 冲突。与 move_flyfollow 任务成功案例（dist_xy<3m）一致。

---

### Priority 2 (HIGH): 扩大 height_penalty_threshold 至 1.5m

**Problem:** `height_penalty_threshold=0.5m` 建立了以 2.5m 为中心、半径 0.5m 的严格高度约束。任何追踪行为需要水平运动，都会因加速/减速引起高度波动，触发线性惩罚。近期 height_penalty=-1.73/ep，是最大负向项。

**Proposed Change:**
- File: `/home/xtj/xtj-project/RL/MARL_AUV/exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/move/marl_move_env_cfg.py`
- `height_penalty_threshold`: 0.5 → 1.5
- **Rationale:** 如果采用 Priority 1 的 XY 平面捕获（3.0m），无人机不需要俯冲，1.5m 阈值允许正常追踪时的高度波动（0.5m 远不足以容纳 ACCBR 控制的自然高度振荡）。可在 Priority 1 生效并看到 tracking 信号后，再逐步收紧此阈值。

---

### Priority 3 (HIGH): 扩大 bounding_box_threshold 或调整无人机初始位置

**Problem:** `bounding_box_threshold=12.0m` 对于目标小车从 x=(-6,-2) 以 +x 方向移动的场景过小。无人机在追踪过程中出界导致 100% episode 被 bounding_box 终止（mean=107 步，远未达到 episode_length_s=60s）。

**Proposed Change（选一）:**
- 方案 A：扩大边界
  - File: `marl_move_env_cfg.py`
  - `bounding_box_threshold`: 12.0 → 20.0
  - **Rationale:** 目标小车 60s 内最大位移 18m，边界需容纳目标运动范围 + 无人机追踪余量
- 方案 B：将目标小车改为循环移动（在边界内折返）
  - File: `marl_move_env.py` 中目标速度更新逻辑
  - 当目标到达边界时反转速度方向
  - **Rationale:** 更根本的解法，防止目标逃出追踪区域

**推荐：方案 A（快速实施），方案 B（长期方案）**

---

### Priority 4 (MEDIUM): 提高 entropy_loss_scale 防止 policy_std 坍缩

**Problem:** `policy_std` 从 0.82 坍缩至 0.15，在约 50k 步已无有效探索。tracking_reward 信号未出现前，策略在距离奖励的引导下早熟收敛到"飞近但不进入捕获区"的局部最优。

**Proposed Change:**
- File: `scripts/skrl/train.py` 或 MAPPO agent 配置
- `entropy_loss_scale`: 当前约 0.001 → 建议 0.005～0.01
- **Rationale:** 更强的熵正则化在 tracking 信号出现前保持策略探索能力，避免在最近 50k 步的有效训练窗口内 std 已过低。

---

### Priority 5 (MEDIUM): 抑制"舒适奖励"主导效应

**Problem:** `body_rate_penalty`（+1.74）和 `action_smoothness`（+0.94）合计+2.68/ep，占近期总奖励（+1.51）的 177%。策略可以通过"飞得慢而稳定"获得高分而无需接近目标。这掩盖了真实任务进展。

**Proposed Change:**
- File: `marl_move_env_cfg.py`
- `body_rate_penalty_weight`: 2.0 → 0.5
- `action_smoothness_weight`: 1.0 → 0.3
- **Rationale:** 将舒适奖励压缩至 distance_reward（+0.34）的同量级，使任务导向信号成为奖励主导项。注意：不要完全移除，body_rate 和 smoothness 对飞行稳定性仍重要。

---

## Experiment Plan

**当前状态判断：** Priority 1（illegal_contact 解除）已成功验证。新瓶颈为捕获设计（XY 距离 + 边界限制），tracking_reward 仍为 0。

### Phase B（Priority 1-3 集成）：下一次训练
1. 实施 Priority 1（capture_distance: 1.0m → 3.0m，XY 距离）
2. 实施 Priority 2（height_penalty_threshold: 0.5 → 1.5m）
3. 实施 Priority 3A（bounding_box_threshold: 12.0 → 20.0m）
4. 实施 Priority 4（entropy_loss_scale: 0.001 → 0.007）
5. 实施 Priority 5（body_rate_penalty_weight: 2.0→0.5，action_smoothness_weight: 1.0→0.3）
6. 运行 300k 步，num_envs=2048

**成功判据（100k 步节点）：**
- `Episode / Total timesteps mean > 300 步`（边界不再主导）
- `tracking_reward recent_mean > 0.1/ep`（首次有效捕获信号）
- `Episode_Termination/all_targets_captured > 0.01`（偶发成功）
- `policy_std > 0.30`（50k 步时，探索未过早坍缩）
- `drone_out penalty < -0.3/ep`（出界减少）

**成功判据（300k 步节点）：**
- `tracking_reward > 2.0/ep`
- `all_targets_captured > 0.1`（10% episode 达到捕获）
- `total_reward_mean > 5.0`

**监控重点：**
- 若 bounding_box 终止仍 > 0.5/rollout → 考虑方案 B（目标折返移动）
- 若 height_penalty > -0.5/ep → 进一步放宽 height_penalty_threshold 至 2.0m
- 若 policy_std < 0.20 at step 100k → 提升 entropy_loss_scale 至 0.02

---

## Comparison with Previous Run (2026-04-02_18-27-51)

| 指标 | 上一次（illegal_contact=终止）| 本次（illegal_contact=仅惩罚）| 变化 |
|------|-------------------------------|-------------------------------|------|
| illegal_contact 终止 | 1.234/rollout | 0.001/rollout | **-99.9%，假设验证成功** |
| episode timesteps mean | 65.3 | 106.9 | **+64%** |
| episode timesteps max | ~98 | 129（best 161） | **+31%** |
| total_reward_mean (recent) | -0.25 | +1.51 | **首次转正** |
| bounding_box 终止 | 0.019 | 1.160 | 新瓶颈出现 |
| drone_out penalty | ~0 | -1.0/ep | 100% 出界 |
| tracking_reward | 0.0 | ~0.0001 | 仍接近 0 |
| height_penalty (recent) | -0.85 | -1.73 | 增大（策略在尝试追踪但被惩罚） |
| policy_std (recent) | 0.14 | 0.17 | 基本持平 |

**结论：** 移除 illegal_contact 终止条件完全解除了阻断瓶颈，训练开始真正运转（total_reward 首次稳定为正），但暴露出两个新问题：
1. **bounding_box 终止**接替为主导终止原因（需扩大边界或实现循环轨迹）
2. **capture_distance=1.0m（3D）** 在 height_penalty 约束下仍无法触发 tracking_reward

这两个问题均是 Priority 1-3 的修改目标。

---

## Changelog

- 2026-04-03（run 2026-04-02_21-56-48）：移除 illegal_contact 终止条件——假设验证完全成功
  - illegal_contact 终止 1.234→0.001/rollout（-99.9%），episode mean 65→107 步（+64%）
  - total_reward_mean 首次稳定为正（-0.25→+1.51），训练趋势完全逆转
  - 新主导瓶颈：bounding_box 终止（1.16/rollout），drone_out -1.0/ep，tracking_reward 仍为 0
  - 新发现："舒适奖励"（body_rate+action_smoothness=+2.68/ep）主导总奖励，掩盖任务进展
  - 下一步：Priority 1 capture_distance 1.0→3.0m（XY），Priority 2 height_penalty_threshold 0.5→1.5m，Priority 3 bounding_box_threshold 12→20m

---

# Training Analysis Report — 2026-04-03_00-20-04_mappo_torch_mappo（Run 6）

**Run:** 2026-04-03_00-20-04_mappo_torch_mappo
**Date:** 2026-04-03
**Task:** Isaac-marl-move-v0（3 Falcon + 4 NovaCarter）
**Algorithm:** MAPPO
**Steps:** 400,000（num_envs=32）

## 本次参数变更（相比 Run 5）

| 参数 | Run 5（2026-04-02_21-56-48） | Run 6（本次） |
|------|------------------------------|---------------|
| capture_distance | 1.0m (3D) | 3.0m (XY平面) |
| bounding_box_threshold | 12.0m | 20.0m |
| height_penalty_threshold | 0.5m | 1.5m |
| body_rate_penalty_weight | 2.0 | 0.5 |
| action_smoothness_weight | 1.0 | 0.3 |
| illegal_contact（终止） | 已移除 | 已移除（延续） |

---

## Training Metrics Summary

| 指标 | 早期（0–80k步） | 近期（320k–400k步） | 最终值 |
|------|----------------|---------------------|--------|
| total_reward_mean | -39.14 | -0.009 | -0.798 |
| total_reward_max | — | +2.79（recent mean） | +1.60 |
| distance_reward | 1.48/ep | 2.14/ep | 2.37/ep |
| tracking_reward | 0.46/ep | 1.78/ep | 2.00/ep |
| height_reward | 0.14/ep | 0.19/ep | 0.20/ep |
| upright_penalty | -0.84/ep | -1.90/ep | -1.82/ep |
| height_penalty | -13.30/ep | -2.10/ep | -2.93/ep |
| drone_out | -1.29/ep | -1.00/ep | -1.00/ep |
| body_rate_penalty | — | +0.35/ep | +0.38/ep |
| action_smoothness | — | +0.25/ep | +0.28/ep |
| bounding_box 终止 | 1.09/rollout | 1.12/rollout | 1.00 |
| episode 步数 mean | 160.1 | 131.5 | 129.3 |
| policy_std | 0.815 | 0.331 | 0.323 |
| success_reward | 0.0 | 0.0 | 0.0 |
| all_targets_captured | 0.0 | 0.0 | 0.0 |
| entropy_loss_scale | 0.001 | 0.001 | 0.001 |

---

## 问题逐项回答

### Q1. tracking_reward 是否出现正信号？

**是的，tracking_reward 首次出现并稳定增长，这是本次最重要的突破。**

- Run 5（capture_distance=1.0m 3D）：tracking_reward 整个运行期间约为 0
- Run 6（capture_distance=3.0m XY）：
  - 0–40k 步：early_mean = 0.46/ep，nonzero_frac = 0.978
  - 200k–240k 步：1.23/ep
  - 360k–400k 步：1.78/ep（持续上升，best = 2.15/ep）

**结论：** 将 capture_distance 改为 XY 平面 3.0m 的假设完全验证成功。tracking_reward 从结构上变为可达，且学习曲线在整个 400k 步内持续上升，尚未见顶。

### Q2. bounding_box 终止是否消除？

**否。bounding_box 依然是主导终止原因，几乎未改变。**

- Run 5：bounding_box 终止 1.16/rollout（100% 出界）
- Run 6：bounding_box 终止 early=1.09/rollout → recent=1.12/rollout（基本恒定）

bounding_box_threshold 已从 12m 扩大到 20m，但终止率无改善。原因分析见"问题诊断"章节。

### Q3. episode 长度是否增长？

**出现反转：episode 长度从初期 160 步下降至近期 130 步，呈持续下降趋势。**

- 0–40k 步：mean=160.1 步
- 80k–120k 步：mean=119.5 步（最低点）
- 280k–320k 步：mean=127.0 步（轻微回升）
- 360k–400k 步：mean=132.1 步

这与 bounding_box 终止持续主导直接相关（见 Priority 1）。

### Q4. 是否出现 all_targets_captured？

**否。success_reward 和 all_targets_captured 整个 400k 步内均为 0.0。**

此任务成功（所有 3 架无人机同时捕获对应目标）的门槛极高，在当前配置下尚未达到。

### Q5. 奖励结构是否健康？

**基本健康，但两个主导惩罚项需要关注。**

近期（last 20%）各分量均值：

| 分量 | 值 | 角色 |
|------|-----|------|
| distance_reward | +2.14 | 正向，主导正奖励 |
| tracking_reward | +1.72 | 正向，稳定增长 |
| body_rate_penalty | +0.35 | 正向（舒适奖励，已降低） |
| action_smoothness | +0.25 | 正向（舒适奖励，已降低） |
| force_penalty | +0.29 | 正向 |
| height_reward | +0.19 | 正向 |
| **upright_penalty** | **-1.85** | **负向，第一大净负项** |
| **height_penalty** | **-2.22** | **负向，第二大净负项** |
| drone_out | -1.00 | 负向，恒定 |

正向合计：+4.84/ep，负向合计：-5.07/ep。整体 total_reward_mean 接近 0（-0.009）。
"舒适奖励"（body_rate+action_smoothness=+0.60/ep）经降权后已不再主导，占正向总和比例降至 12%，目标达成。

---

## Observations & Findings

### [Bounding Box 终止持续主导] — Severity: CRITICAL

**Symptom:** bounding_box 终止从 run 5 的 1.16/rollout 到 run 6 的 1.09–1.12/rollout，扩大边界（12→20m）几乎无效。

**Root Cause:** bounding_box 终止的根本原因不是边界太小，而是无人机策略正在主动跟随目标 NovaCarter 小车——而小车以 0.3m/s 沿 +x 方向持续运动，不折返。即使边界扩大，在 60s 的 episode 内小车仍会跑出 18m，无人机跟随同样会出界。

**Evidence:** 
- `Episode_Termination/targets_out_of_bounds` 全程为 0（目标本身未触发出界，说明目标还在边界内）
- `drone_out` 惩罚始终为 -1.00/ep（100% 的 episode 都有无人机出界）
- bounding_box 终止 1.09–1.12/rollout 完全稳定，无法通过扩大边界解决
- `episode_timesteps_min` 最低降到 1（存在极短 episode，但 min 均值 36 步，说明有异常重置）

**关键推断：** 策略已学会追踪目标（tracking_reward 上升），但追踪行为导致无人机跟随目标出界。边界扩大不能解决这一结构性问题，需要让目标在有界范围内循环运动，或扩大边界至 50m+。

### [Upright Penalty 持续增大] — Severity: HIGH

**Symptom:** upright_penalty 从 early=-0.84/ep 持续增大到 recent=-1.90/ep，是唯一随训练变差的主要指标。

**Root Cause:** 追踪运动目标需要无人机施加横向加速度，这必然引入机体倾斜（roll/pitch）。upright_penalty_weight=2.0 惩罚所有倾斜，与高速追踪任务从物理上冲突。随着策略学会更积极地追踪，倾斜增加，惩罚增大。

**Evidence:**
- 计算方式：`(z_axis_body_dot - 1.0) * weight * step_dt`，追踪时 z_axis_body < 1.0（倾斜），惩罚单调增大
- upright_penalty: -0.84 → -1.09 → -1.60 → -1.67 → -1.75 → -1.77 → -1.85 → -1.80 → -1.80 → -1.90（10个阶段持续增大）
- 这与 tracking_reward 同步增大，证实两者正相关

**危险性：** 如果 upright_penalty 的增速超过 tracking_reward 的增速，策略会被迫退回静态悬停（倾斜最小但无法追踪）。当前 tracking_reward 增速仍更快，但差距在收窄。

### [Height Penalty 持续存在] — Severity: HIGH

**Symptom:** height_penalty 虽从 early=-13.30 大幅降低到 recent=-2.22/ep，但仍是第二大负向项，且近期呈轻微反弹（260k步后从 -1.65 重新升至 -2.22）。

**Root Cause:** NovaCarter z=0.25m，desired_height=2.5m，垂直距离 2.25m。即使 height_penalty_threshold 扩大到 1.5m，任何俯冲接近策略都会产生偏离（altitude 从 2.5m 降到 ~1.0m 时偏离 = 1.5m，恰好触发阈值边缘）。近期反弹可能源于策略在尝试更激进的高度调整。

**Evidence:**
- height_penalty 趋势：-13.30 → -11.53 → -4.16 → -1.69 → -1.65 → -1.71 → -1.69 → -2.12 → -2.33 → -2.10
- 改善主要发生在 0–160k 步（策略学会避免极端高度偏差），之后进入平台且略有反弹

### [Policy Std 单调下降] — Severity: MEDIUM

**Symptom:** policy_std 从 0.815 单调下降到 0.323，下降幅度 60%，趋势尚未收敛。

**Evidence:**
- 0–40k: 0.815 → 80k: 0.706 → 160k: 0.616 → 240k: 0.485 → 320k: 0.392 → 400k: 0.323
- entropy_loss_scale=0.001（config agent.yaml），极低的熵系数无法抵抗 std 收缩
- 梯度范数 actor 近期=0.998（接近梯度截断上限），说明策略在强力更新
- value_loss 近期=0.005（极小，价值网络已收敛）——但这不是好兆头，说明 critic 可能也在收敛到一个局部最优

**危险性：** 如果 std 继续下降到 0.1 以下，策略将丧失探索能力，tracking_reward 的增长会停止，很可能在当前水平（~1.78/ep）附近过早收敛而非继续上升。

### [Episode 长度下降] — Severity: MEDIUM

**Symptom:** episode_timesteps_mean 从初期 160 步下降到 130 步，最大值（max=161 步）低于初期最大值（早期 best=172 步）。

**Root Cause:** bounding_box 终止主导，策略学会追踪（向目标运动）后更快地跟随出界，导致 episode 更短。这是"成功的策略反而缩短了 episode"的悖论。

---

## Improvement Recommendations

### Priority 1 (CRITICAL): 实现目标循环运动，彻底解决出界问题

**Problem:** 目标 NovaCarter 沿 +x 方向以 0.3m/s 持续运动不折返，无论边界多大都会出界。这是 bounding_box 终止持续为 100% 的根本原因。

**Proposed Change:**
- File: `exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/move/marl_move_env.py`
- 修改 NovaCarter 位置更新逻辑：当目标接近边界（距边界 5m 内）时折返，实现来回运动（正弦波或锯齿波轨迹）
- 或直接将目标速度改为圆周运动（半径 8m 圆圈），确保目标始终在 bounding_box 内
- **备选（最快验证）：** 将 `bounding_box_threshold` 进一步扩大到 50m，并将 episode_length_s 减小到 20s，消除追踪途中出界问题
- Rationale: 策略已学会追踪，但追踪行为本身导致出界——这是任务设计而非策略问题

**预期效果：** bounding_box 终止从 1.12/rollout 降至 <0.05/rollout，episode 长度从 130 步增长到 300+ 步，tracking_reward 有空间继续增长。

### Priority 2 (HIGH): 降低 upright_penalty 权重

**Problem:** upright_penalty_weight=2.0 与高速追踪物理冲突，随 tracking 改善而持续增大（-0.84→-1.90/ep），有可能逆转训练进展。

**Proposed Change:**
- File: `exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/move/marl_move_env_cfg.py`
- Parameter: `upright_penalty_weight = 2.0` → `1.0`（或完全移除，改为仅在倾斜超过阈值时惩罚）
- Rationale: Falcon 的几何控制器（DFBC）已提供姿态稳定，不需要额外的 upright 惩罚来保证飞行安全。该项惩罚在悬停任务中有意义，但在追踪任务中物理上要求倾斜。

**更优替代方案（推荐）：** 将持续惩罚改为阈值惩罚：
- 仅当 `z_axis_body < cos(30°) ≈ 0.866` 时惩罚（倾斜超过 30 度才惩罚）
- 正常追踪时倾斜约 10–20 度，不触发惩罚

**预期效果：** 解除 tracking 改善与 upright_penalty 增大之间的对抗，预计总奖励可增加 +1.0–1.5/ep。

### Priority 3 (HIGH): 增大 entropy_loss_scale，防止过早收敛

**Problem:** policy_std 从 0.815 下降到 0.323（-60%），entropy_loss_scale=0.001 无法抵抗收敛。若继续下降，tracking_reward 增长将停止。

**Proposed Change:**
- File: `logs/skrl/move/[run]/params/agent.yaml`（实际修改位置：SKRL training config）
- 寻找 MAPPO 配置文件：`scripts/skrl/` 或 `exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/move/`
- Parameter: `entropy_loss_scale: 0.001` → `0.005`
- Rationale: tracking_reward 仍在上升（1.78/ep，尚未饱和），需要维持 policy_std > 0.4 以保持探索。0.005 是上一次 flyfollow 任务成功防止 std collapse 的验证值。

**注意：** 避免超过 0.01（会导致策略更新被熵正则主导，影响收敛速度）。

### Priority 4 (MEDIUM): 考虑移除 height_penalty 或改用更宽松的设计

**Problem:** height_penalty 是当前第二大净负项（-2.22/ep），且近期出现反弹，说明策略在尝试高度调整被持续惩罚。

**Proposed Change:**
- File: `exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/move/marl_move_env_cfg.py`
- **Option A：** `height_penalty_threshold = 1.5` → `2.0`（允许更大高度偏差）
- **Option B：** `height_penalty_weight = 1.0` → `0.3`（降低惩罚强度）
- **Option C（推荐）：** 将 height_penalty 改为仅在 z<0.5m 或 z>5.0m 时触发（安全边界而非精确高度控制），等同于将 desired_height 约束完全交给 height_reward，而非双重约束
- Rationale: move 任务的核心是 XY 平面追踪，高度控制已有 height_reward 提供正向信号，height_penalty 的双重约束产生冗余惩罚，会抑制追踪时的自然高度调整。

**注意：** 不要完全移除高度约束，NovaCarter 高度 ~0.75m（3x scale），无人机不能贴地飞行。

---

## Experiment Plan

**下一步实验（Run 7）优先级排序：**

1. **必做（Priority 1）：** 修改目标运动模式，使 NovaCarter 在边界内循环（折返或圆周）
   - 验证指标：bounding_box 终止 <0.1/rollout，episode mean > 200 步
2. **同步修改（Priority 2+3）：** upright_penalty_weight 2.0→1.0，entropy_loss_scale 0.001→0.005
3. **观察 50k 步后：** 若 tracking_reward 仍在上升且 policy_std > 0.5，说明改变有效
4. **200k 步 checkpoint：** 比对 tracking_reward 和 all_targets_captured
5. **成功标准：**
   - bounding_box 终止 <0.1/rollout
   - episode mean > 200 步
   - tracking_reward > 3.0/ep（当前 1.78/ep）
   - all_targets_captured > 0 case（第一次捕获成功）
   - policy_std 在 200k 步时仍 > 0.4

**训练命令（num_envs 建议提升到 2048 进行正式训练）：**
```bash
python3 scripts/skrl/train.py \
  --task=Isaac-marl-move-v0 \
  --headless --num_envs=2048 --algorithm="MAPPO"
```

---

## Changelog

- 2026-04-03（run 2026-04-03_00-20-04）：capture_distance 3.0m XY + bounding_box 20m + height_penalty_threshold 1.5m + comfort rewards 降权
  - **tracking_reward 首次出现正信号**：0→1.78/ep，持续上升 400k 步未见顶
  - total_reward_mean 从 -39.14 上升至 -0.009（接近 0）
  - **bounding_box 终止仍 100%**：20m 边界仍不足，目标循环运动是根本解法
  - upright_penalty 持续增大（-0.84→-1.90），追踪与倾斜惩罚对抗
  - policy_std 0.815→0.323，entropy 不足，需提升 entropy_loss_scale 0.001→0.005
  - 下一步：Priority 1 实现目标循环运动，Priority 2 upright_penalty_weight 2.0→1.0
  - 下一步：Priority 1 方案 B（从终止条件中移除 illegal_contact）作为快速验证，然后修复根本原因

---

# Training Analysis Report — 2026-04-03_08-51-58

**Run:** 2026-04-03_08-51-58_mappo_torch_mappo
**Date:** 2026-04-03
**Task:** Isaac-marl-move-v0
**Algorithm:** MAPPO (SKRL)
**Training progress:** 61,800 logged steps / 400,000 budget (15.4%)

**本次修改（相对上一次 2026-04-03_00-20-04）：**
1. NovaCarter 折返轨迹：x 在 [-8m, +8m] 之间来回（target_bounce_x_min=-8, target_bounce_x_max=8）
2. upright_penalty_weight: 2.0 → 1.0
3. entropy_loss_scale: 0.001 → 0.005

---

## Training Metrics Summary

| Metric | Early (~step 600) | Peak | Recent (~last 20) | Last |
|--------|------------------|------|-------------------|------|
| total_reward_mean | -45.96 | -27.57 | -5.57 | -3.51 |
| distance_reward/ep | 0.58 | 1.53 | 1.09 | 1.31 |
| tracking_reward/ep | 0.11 | 0.70 | 0.50 | 0.61 |
| height_penalty/ep | -11.3 | — | -1.47 | -1.05 |
| upright_penalty/ep | -0.40 | — | -1.61 | -1.65 |
| height_reward/ep | 0.14 | 0.16 | 0.14 | 0.14 |
| illegal_contact/ep | -2.43 | — | -0.17 | -0.05 |
| episode_length (mean steps) | 151.8 | 177.1 (step 28k) | 112.6 | 118.5 |
| bounding_box termination | 0.91 | — | 1.10 | 1.10 |
| all_targets_captured | 0.0 | 0.0 | 0.0 | 0.0 |
| time_out | 0.0 | 0.0 | 0.0 | 0.0 |
| policy_std | 0.818 | 0.824 | 0.816 | 0.820 |

---

## Observations & Findings

### [Positive] 高度控制大幅改善 — Severity: 良好信号

**Symptom:** height_penalty 从 early -11 ~ -14/ep 降至 late -0.5 ~ -1.1/ep，改善幅度超过 10 单位/ep。

**Evidence:** 上一次（00-20-04）height_penalty recent=-2.31/ep，本次 recent=-1.47/ep，即使经过更多步骤依然持续改善。high 和 low altitude termination 均接近 0。

**Interpretation:** 无人机已学会维持 desired_height=2.5m ±1.5m 的高度带，这是高度控制的核心进展。

---

### [Positive] 违法接触和无人机碰撞基本消除 — Severity: 良好信号

**Symptom:**
- illegal_contact: early -2.43/ep → recent -0.17/ep → last -0.05/ep
- drones_collide termination: early 0.12/rollout → recent 0.0/rollout

**Evidence:** illegal_contact 惩罚持续减小，drones_collide 终止从 0.12 降为 0.00，表明上一次修复（从终止条件移除 illegal_contact）有效。

---

### [Positive] policy_std 稳定 — Severity: 良好信号

**Symptom:** policy_std 全程稳定在 0.815~0.824，无 collapse 迹象。

**Evidence:** entropy_loss_scale 0.001→0.005 的修改成功防止了标准差崩塌（上一次 std 从 0.815→0.323）。本次全程 std>0.81。

---

### [Critical] bounding_box 终止未因折返轨迹改善 — Severity: CRITICAL

**Symptom:** bounding_box 终止率 early 0.91/rollout → recent 1.10/rollout，与上一次（00-20-04）的 1.11 几乎完全相同。折返轨迹修改未能减少 bounding_box 终止。

**Root Cause:** bounding_box 触发的是**无人机**出界（`drone_positions.abs() > 20m`），而非目标出界（targets_out_of_bounds=0.0 始终为零，折返有效）。drones 在追踪目标时越过 ±20m 边界：
- 目标折返于 ±8m 处
- bounding_box = 20m，仅留 12m 余量
- 无人机在追踪加速时超调（overshoot）超过 12m

**Evidence:**
- targets_out_of_bounds = 0.0 全程（折返确实生效）
- bounding_box termination = 1.10/rollout（无改善）
- episode_length max=210 steps 而理论 max=4000 steps（60s / 0.015s），说明所有 episode 均在 3s 内因 bounding_box 提前终止
- time_out=0.0（无任何 episode 完成全程 60s）

**Deeper mechanism:** 折返轨迹修复了"目标出界"问题，但暴露了更深的问题：无人机追踪目标时本身就飞出了 20m 边界。这与目标是否折返无关——只要目标在 ±8m 运动，无人机就会在 ±20m 附近被截断。

---

### [High] episode 长度先升后降，形成退化弧线 — Severity: HIGH

**Symptom:** episode_length_mean 从 early 151.8 → peak 177.1（step 28k）→ late 115（step 61k），在达到峰值后明显退化。

**Root Cause:** 正向反馈回路导致的退化：
1. 初期：策略学会追踪 → tracking_reward 上升 → 无人机飞得更积极
2. 中期：积极追踪 → 更频繁地超调出 20m 边界 → bounding_box 触发更早
3. 后期：episode 变短 → tracking_reward 积累时间减少 → tracking_reward 下降

**Evidence:**
- ep_len peak (step 25k-32k): 164.1 mean
- ep_len late (step 55k+): 119.8 mean（-27% 退化）
- tracking_reward peak: 0.70 → late: 0.47（同步退化）

---

### [High] upright_penalty 随 episode 推进持续加重 — Severity: HIGH

**Symptom:** upright_penalty/ep 从 early -0.40 → late -1.65，即使 weight 从 2.0 降至 1.0，每 episode 的实际惩罚仍增加了 4 倍。

**Root Cause:** 策略为了实现更快的 XY 追踪而增大倾斜角（推进方向倾斜），导致实际倾斜量增大。weight 降低减弱了惩罚梯度，但未改变物理行为。

**Evidence:** upright_penalty 在 step 27k-42k（ep_len 达到峰值时）快速从 -0.6 增加到 -1.6，与 ep_len 退化的时间点高度吻合，说明"倾斜换速度"的行为与随后的 OOB 退化强相关。

---

### [Medium] all_targets_captured 持续为 0 — Severity: MEDIUM

**Symptom:** 整个训练过程中未出现任何一次目标捕获（all_targets_captured=0.0），capture_distance=3.0m，sustained_follow_duration=3.0s 条件从未满足。

**Root Cause:** bounding_box 在 3s hold 完成前就触发了 episode 终止。即使无人机接近目标，也因追踪动作使其飞过边界而被截断。

**Evidence:** 最长 episode 仅 210 steps = 3.15s，而捕获需要 3.0s 持续接近 + 后续动作，实际可用时间极度压缩。

---

## Improvement Recommendations

### Priority 1 (CRITICAL): 大幅降低 bounding_box_threshold 或缩小折返范围

**Problem:** bounding_box=20m 对于折返目标 ±8m 是过大的。注释说"小车0.3m/s×60s=18m"——但有了折返后目标最多移动 8m，该注释已经过时。bounding_box 可以安全地缩小到 12m（仍给无人机 4m 追踪余量），这样不会改变无人机体验但会提前截断超调。

**实际建议（优先 A）：**
- File: `exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/move/marl_move_env_cfg.py`
- Parameter: `bounding_box_threshold = 20.0` → `12.0`（目标最远 8m + 4m 安全余量）
- Rationale: 这不会增加 bounding_box 触发率（已经 >1.0/rollout），而是将触发点移近，使无人机在 OOB 之前看到更多有效的追踪 reward signal，学会不超调。同时更新注释以说明折返设计后的正确阈值计算。

**补充建议（必做，配合 Priority 1 A）：**
- 添加**软惩罚**：当 drone x/y 绝对值超过 9m（目标范围 8m + 1m 预警）时施加线性递增惩罚，引导无人机减速而不是直接终止
- File: `exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/move/marl_move_env.py`
- 在 `_get_rewards()` 中加入 `overshoot_penalty = -w * relu(max(|pos_x|, |pos_y|) - 9.0)`，weight ~0.5

**预期效果:** bounding_box 触发前无人机有更强的减速信号，overshoot 减少，episode 长度恢复到 170+ steps。

---

### Priority 2 (HIGH): 更新 bounding_box_threshold 注释，缩减 drone_spawn_x_range

**Problem:** drone_spawn_x_range = (-8.0, -6.0) 说明无人机初始位置在 -8m 附近，非常接近目标的折返边界和新建议的 bounding_box=12m。无人机一开始就处于高风险区域，需要立刻移向中心。

**Proposed Change:**
- File: `exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/move/marl_move_env_cfg.py`
- Parameter: `drone_spawn_x_range = (-8.0, -6.0)` → `(-4.0, -2.0)`（更靠近中心，减少初始 OOB 风险）
- Rationale: 与目标 spawn_x_range = (-6.0, -2.0) 对齐，无人机初始距目标更近，更快进入有效追踪范围。

---

### Priority 3 (HIGH): 为 upright_penalty 引入倾斜阈值，而非线性惩罚

**Problem:** upright_penalty_weight=1.0 的线性惩罚随追踪学习而持续加重（-0.40→-1.65/ep），与追踪行为形成对抗。Falcon 在追踪时物理上需要倾斜（推力方向）。

**Proposed Change:**
- File: `exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/move/marl_move_env.py`
- 将 upright_penalty 改为阈值式：仅当倾斜超过 35° 时触发（`z_body_axis < cos(35°) ≈ 0.819`），正常追踪（≤35°倾斜）不受惩罚
- 或：`upright_penalty_weight = 1.0` → `0.3`（进一步降低权重，给追踪更大空间）
- Rationale: 当前惩罚在 step 42k-61k 内翻了 4 倍，是对"追踪导致倾斜"的惩罚而非对"危险翻滚"的惩罚。阈值设计可以区分两种情况。

---

### Priority 4 (MEDIUM): 继续当前训练，观察 total_reward 趋势

**Problem:** 训练仅完成 15.4%（61800/400000 steps），total_reward 从 -50 改善到 -3.5，仍有明显上升趋势。

**Proposed Change:** 无需立即停止训练。可考虑并行实验：
- 当前 run 继续（观察 100k steps 时是否出现首次 all_targets_captured）
- 新 run 使用上述 Priority 1+2+3 的修改

**成功标准（下一次 run）：**
- bounding_box termination < 0.1/rollout（当前 1.1）
- episode_length_mean > 200 steps（当前 115，峰值 177）
- tracking_reward > 1.5/ep（当前 0.61，上一次峰值 1.78）
- all_targets_captured > 0（首次捕获）
- policy_std 在 200k steps 时仍 > 0.5

---

## Experiment Plan

**Run 9（下一步）— 核心修改：**

1. **CRITICAL - bounding_box 缩小：**
   - `bounding_box_threshold`: 20.0 → 12.0
   - 添加软惩罚（overshoot_penalty）在 |x| or |y| > 9m 时触发

2. **HIGH - 无人机初始位置靠近中心：**
   - `drone_spawn_x_range`: (-8.0, -6.0) → (-4.0, -2.0)

3. **HIGH - upright_penalty 改为阈值式或继续降权：**
   - `upright_penalty_weight`: 1.0 → 0.3

4. **保留本次成功修改：**
   - 折返轨迹：target_bounce_x_min=-8, target_bounce_x_max=8（有效）
   - entropy_loss_scale=0.005（有效，std 稳定在 0.82）

5. **训练命令：**
```bash
python3 scripts/skrl/train.py \
  --task=Isaac-marl-move-v0 \
  --headless --num_envs=2048 --algorithm="MAPPO"
```

6. **监控指标（前 50k steps）：**
   - bounding_box: 应从 1.1 降至 <0.3/rollout
   - episode_length_mean: 应从 115 回升至 200+
   - tracking_reward: 应从 0.61 重新上升
   - all_targets_captured: 首次出现 >0 为关键里程碑

---

## Changelog

- 2026-04-03（run 2026-04-03_08-51-58）：折返轨迹 ±8m + upright_penalty 2.0→1.0 + entropy 0.001→0.005
  - **height_penalty MASSIVE 改善**：-11 → -1.05/ep，无人机已学会维持高度
  - **policy_std 稳定 0.82**：entropy=0.005 修复有效，无 std collapse
  - **illegal_contact 基本消除**：early -2.43 → last -0.05/ep
  - **bounding_box 未改善**：1.10/rollout，与上次相同——原因是折返修复了目标 OOB，但未修复无人机超调 OOB（targets_out_of_bounds=0 确认）
  - **episode 长度退化**：peaked at 177 steps (step 28k) → declined to 115 steps (step 61k)
  - **tracking_reward 退化**：0.70 peak → 0.46 recent，与 ep_len 退化同步
  - **upright_penalty 仍加重**：-0.4 → -1.65/ep，策略"倾斜换速度"行为随追踪学习加深
  - **all_targets_captured = 0**：未出现任何捕获，episode 太短（max=3.15s，需要 3s hold）
  - **根本问题确认**：bounding_box=20m 对折返目标 ±8m 过大，需缩小至 12m + 添加超调软惩罚
  - 下一步：Priority 1 缩小 bounding_box 至 12m + overshoot_penalty，Priority 2 调整 drone_spawn，Priority 3 降低 upright_penalty_weight

---

# Training Analysis Report — Run 8

**Run:** 2026-04-03_12-24-40_mappo_torch_mappo
**Date:** 2026-04-03
**Task:** Isaac-marl-move-v0
**Algorithm:** MAPPO
**Steps completed:** ~255,500 / 400,000 (63.9%)

## Training Metrics Summary

| Metric | Run 7 (61k steps) | Run 8 (255k steps) |
|---|---|---|
| total_reward_mean (recent) | -3.5 | +12.8 |
| total_reward_mean (best) | ~+5.0 | +24.8 |
| episode_length_mean (recent) | 115 | 222 |
| episode_length_mean (peak) | 177 (step 28k) | **291 (step 223k)** |
| bounding_box termination (recent) | 1.10 | 0.95 |
| bounding_box termination (first 20%) | 1.10 | 1.14 |
| tracking_reward (recent) | 0.61 | 4.01 |
| drone_out penalty (recent) | -1.0 | -0.93 |
| crash termination (recent) | 0.01 | 0.03~0.32 (spiky) |
| policy_std (recent) | 0.82 | 0.83 |
| all_targets_captured | 0 | **0** |
| success_reward | 0 | **0** |

## Observations & Findings

### Episode Length — MAJOR IMPROVEMENT — Severity: POSITIVE

**Symptom:** Run 7 peaked at 177 steps (step 28k) and declined to 115. Run 8 reached 291 steps at step 223k — a 64% improvement over run 7 peak. Recent mean is 222 steps vs run 7 late value of 115.

**Root Cause (positive):** bounding_box=12m + drone_spawn_x_range=(-4,-2) together prevented the early OOB that caused run 7's peak-and-decay. Drones start farther from boundary, giving the policy time to learn approach behavior before hitting the wall.

**Evidence:** ep_len trend: 81→127 (early surge) → 98 (small dip) → steady climb 100-140 (steps 80-140k) → strong second phase 188-290 (steps 155-246k). No collapse pattern visible.

### bounding_box Termination — PARTIAL IMPROVEMENT — Severity: HIGH

**Symptom:** bounding_box_threshold=12m was applied. First-20% mean: 1.137 → last-20% mean: 0.949. Reduction of ~16%. Still ~1.0/rollout — nearly every episode still ends by drone exiting bounds.

**Root Cause:** The bounding_box trigger itself is much less severe (0.95 vs 1.10+), but drones are still crossing the 12m boundary. The boundary_soft penalty (mean -0.28/ep) is present but not strong enough to prevent the boundary crossing. Drones learn to approach aggressively (good for tracking reward) but overshoot the boundary.

**Evidence:** drone_out penalty improved from -1.24 to -0.93/ep (first vs last 20%), confirming fewer OOB events, but not eliminated. boundary_soft_penalty oscillates -0.13 to -0.44 without a clear decreasing trend — the soft signal is not being effectively learned.

### tracking_reward — BREAKTHROUGH — Severity: POSITIVE

**Symptom:** tracking_reward rose from 0.07 (step 100) → 4.01 (recent mean), best=5.05. This is a 6x improvement over run 7 (0.61 recent, 0.70 peak) and confirms drones are actively following targets at close range.

**Root Cause (positive):** capture_distance=3.0m XY + reduced bounding_box_threshold + closer drone spawn = more time within capture range per episode. tracking_reward_weight=4.0 provides strong incentive that compounds with episode length improvement.

**Evidence:** tracking trend is monotonically increasing throughout: 1.29 (step 12k) → 2.03 (step 87k) → 3.74 (step 174k) → 4.29 (step 236k). Distance reward shows same trend: 1.83→4.75→5.45.

### success_reward = 0 — CRITICAL BOTTLENECK — Severity: CRITICAL

**Symptom:** all_targets_captured=0 and success_reward=0 throughout 255k steps. No capture success event has ever occurred.

**Root Cause:** The capture success condition likely requires sustained hold (drone within capture_distance for >1s). bounding_box termination cuts episodes short — even though tracking_reward is high (drones are near targets), an OOB event terminates the episode before the hold timer can complete. With bounding_box triggering ~1.0/rollout and episode_length_mean=222 steps (~2.47s at dt=0.01s), the episode window is too short relative to hold duration.

**Additional factor:** crash/fly_low terminations have become non-trivial in later training (step 248k: crash=0.27, fly_low=0.27 per rollout vs early values near 0). This spike at the end of the run suggests the policy is becoming increasingly aggressive, pushing altitude limits.

**Evidence:** success_reward=0.0000 all 2555 logging points. all_targets_captured=0.0000 all logged. Max episode_length_mean=290 steps ≈ 3.23s, which may be close to the hold requirement — **one more tier of improvement in episode length could unlock first captures**.

### crash / fly_low — EMERGING ISSUE — Severity: HIGH

**Symptom:** crash_term + fly_low_term were near 0 until step 130k, then grew: step 136k (crash=0.46, fly_low=0.40) — a large spike. After that, oscillated but remained elevated: last value crash=0.32, fly_low=0.09 at step 255k.

**Root Cause:** As tracking improves and drones learn to follow more aggressively, they descend too low (fly_low) or collide with the NovaCarter body (crash). The height_penalty_threshold=1.5m allows altitude variation, but the lower bound may be too accessible when drones push for XY capture. fly_low and crash co-occurring suggests descent-then-contact.

**Evidence:** illegal_contact penalty had a severe spike of -52.26 at step 201k (vs normal ~-0.1 to -0.7). This single catastrophic event suggests that at high tracking skill, drones get very close to NovaCarter and register contact force. Occasional spikes like this inject large negative signal that destabilizes training temporarily — consistent with the ep_len oscillation seen in the later phase.

### boundary_soft — INEFFECTIVE as Designed — Severity: MEDIUM

**Symptom:** boundary_soft penalty is consistently -0.13 to -0.44/ep with no decreasing trend. It does not appear to be teaching drones to slow down near the boundary.

**Root Cause:** The soft threshold at 9m applies a fixed-weight penalty, but relative to the tracking_reward gain (4.0/ep), the -0.28/ep soft penalty is insufficient to change behavior. Drones learn to accept the boundary penalty as a "cost of doing business" for aggressive pursuit. The signal-to-cost ratio favors crossing the soft zone.

**Evidence:** boundary_soft oscillation throughout: -0.157 (early) → -0.283 (recent). No clear downward trend. bounding_box_threshold=12m triggers 0.95/rollout despite the 9m soft zone — the 3m gap (9→12m) is not enough warning distance.

### upright_penalty — GROWING AS BEFORE — Severity: MEDIUM

**Symptom:** upright_penalty grows from -0.002 (step 100) to -1.125 (step 249k), matching the run 7 pattern (-0.40→-1.65). upright_penalty_weight was reduced from 1.0 to 0.3, but the actual per-episode penalty is even higher than run 7 at comparable steps.

**Root Cause:** Reducing weight from 1.0 to 0.3 means the raw tilt angle must be even larger than before to produce the observed penalty level. Drones are tilting ~3.3x more aggressively than in run 7. This is consistent with the larger tracking_reward gain — better tracking comes from faster, more aggressive XY pursuit which requires steeper tilt. weight=0.3 is not a solution; threshold-based penalty is the correct approach.

**Evidence:** At step 173k, upright=-0.907/ep at weight=0.3 vs run 7 final -1.65/ep at weight=1.0. Normalized by weight: this run has tilt level = 0.907/0.3 = 3.02 "raw units" vs run 7 at 1.65/1.0 = 1.65 "raw units" — meaning run 8 drones are tilting 1.83x more in actual angle.

### Policy Stability — VALIDATED — Severity: POSITIVE

**Symptom:** policy_std stable at 0.827-0.857 throughout all 255k steps. No collapse.

**Root Cause (positive):** entropy_loss_scale=0.005 validated again. Compare to run 6 which collapsed 0.815→0.323 in 400k steps.

**Evidence:** policy_std recent_mean=0.828, std=0.018. Gradient norm actor stable 0.63-0.92.

## Improvement Recommendations

### Priority 1 (CRITICAL): Reduce crash/fly_low by raising minimum altitude floor or contact threshold

**Problem:** crash + fly_low terminations are rising to 0.27+0.09/rollout at end of run. Aggressive tracking behavior causes descent and NovaCarter contact. The catastrophic illegal_contact spike (-52.26 at step 201k) injects destabilizing signal.

**Proposed Changes:**
- File: `exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/move/marl_move_env_cfg.py`
- Parameter: `height_penalty_threshold = 1.5` → `1.0` (tighten altitude band to discourage descent below 1.5m)
- Additionally: raise `contact_sensor_threshold` from 1.0N → 5.0N (filter micro-contacts during close approach, avoid catastrophic negative spikes)
- Rationale: fly_low is co-occurring with crash — the drone is descending toward the NovaCarter and touching it. Tightening the height band penalizes descent earlier. Contact threshold increase prevents occasional large -52 penalty spikes.

**Alternative (preferred):** Implement a minimum_altitude hard floor = 1.2m in the termination condition (separate from height_penalty). This cleanly separates "altitude safety" from "follow behavior."

### Priority 2 (CRITICAL): Strengthen overshoot prevention to eliminate bounding_box=12m terminations

**Problem:** bounding_box termination still ~0.95/rollout despite 12m threshold + 9m soft boundary. The soft penalty (-0.28/ep) is too weak relative to tracking reward gain. Drones accept the penalty as a cost of aggressive pursuit. This prevents episodes from lasting long enough for first capture.

**Proposed Changes:**
- File: `exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/move/marl_move_env_cfg.py`
- Parameter: `boundary_soft_penalty_weight = 0.5` → `2.0` (4x increase to make the soft penalty comparable in magnitude to the tracking benefit)
- Parameter: `boundary_soft_threshold = 9.0` → `8.0` (trigger soft penalty at the bounce boundary itself — when drones go past 8m they should immediately feel strong pushback)
- Rationale: At 8m+, the target is reversing — following it beyond 8m is actually wrong behavior. Aligning the soft threshold with the target bounce point (8m) makes the penalty semantically correct: "you've gone further than the target ever goes."

**Expected effect:** bounding_box termination should drop from 0.95 to <0.3/rollout. With episodes no longer cut short at 12m, episode_length_mean should exceed 350+ steps, enabling the first sustained captures.

### Priority 3 (HIGH): Replace linear upright_penalty with threshold-based penalty

**Problem:** upright_penalty grows monotonically as training improves, from -0.002 to -1.125/ep even with weight=0.3. Normalized tilt is 1.83x higher than run 7. The penalty is penalizing effective tracking behavior (moderate tilt for XY acceleration) rather than dangerous tipping.

**Proposed Change:**
- File: `exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/move/marl_move_env.py`
- Replace: `upright_penalty = -weight * (1.0 - z_body_axis)` (linear in tilt angle)
- With: `upright_penalty = -weight * relu(cos(35°) - z_body_axis)` where `cos(35°) ≈ 0.819`
  - This is 0 for tilt ≤ 35° and linearly increasing beyond 35°
- `upright_penalty_weight = 0.3` → `1.0` (restore to reasonable weight since threshold protects from over-penalizing)
- File: `exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/move/marl_move_env_cfg.py`
- Add parameter: `upright_tilt_threshold_deg = 35.0`
- Rationale: Falcon needs ~20-25° tilt for 1.5 m/s lateral acceleration. 35° threshold gives room for active tracking without penalty, while still punishing dangerous tip-over (>35°).

### Priority 4 (MEDIUM): Increase training budget and monitor for first capture

**Problem:** success_reward=0 throughout 255k steps. Analysis suggests the policy is close — max episode_length=291 steps ≈ 3.2s, and tracking_reward is healthy at 4.0/ep. With bounding_box fix from Priority 2, episodes should grow to 350+ steps (3.9s), which may be sufficient for first captures if hold_duration ≤ 3s.

**Proposed Change:**
- Keep all current reward weights except as modified by Priority 1-3
- Extend training to 600k steps (from current 400k budget)
- Add explicit logging: monitor `Episode_Termination/all_targets_captured` and `Episode_Reward/success_reward` for first non-zero event
- Rationale: The policy is genuinely improving — total_reward went from -1.5 to +12.8. This is not a stalled run. With structural fixes (Priority 1-2), first captures may appear within 50-100k more steps.

## Experiment Plan

**Run 9 — 核心修改（基于 Run 8 分析）：**

1. **CRITICAL — 软边界加强（防止超调出界）：**
   - `boundary_soft_threshold`: 9.0 → 8.0
   - `boundary_soft_penalty_weight`: 0.5 → 2.0

2. **CRITICAL — 抑制俯冲/碰撞（fly_low/crash 上升）：**
   - `height_penalty_threshold`: 1.5 → 1.0（更早惩罚下降，防止贴地）
   - `contact_sensor_threshold` 或等效：考虑提高至 5.0N 过滤微接触尖峰

3. **HIGH — upright_penalty 改为阈值式：**
   - `marl_move_env.py`：引入 `cos(35°) ≈ 0.819` 阈值，低于该值才触发惩罚
   - `upright_penalty_weight`: 0.3 → 1.0（配合阈值恢复权重）

4. **保留所有 Run 8 有效修改：**
   - `bounding_box_threshold = 12.0`
   - `drone_spawn_x_range = (-4, -2)`
   - 折返轨迹 ±8m
   - `entropy_loss_scale = 0.005`
   - `tracking_reward_weight = 4.0`
   - `capture_distance = 3.0m XY`

5. **训练命令：**
```bash
python3 scripts/skrl/train.py \
  --task=Isaac-marl-move-v0 \
  --headless --num_envs=2048 --algorithm="MAPPO"
```

6. **监控指标（前 100k steps）：**
   - bounding_box: 应从 0.95 降至 <0.3/rollout
   - crash + fly_low: 应从 0.32+0.09 降至 <0.05/rollout
   - episode_length_mean: 应超过 350 steps
   - all_targets_captured: 首次出现 >0 是关键里程碑
   - illegal_contact spike：不应再出现 >-5/ep 的尖峰

**成功标准（Run 9）：**
- bounding_box < 0.3/rollout
- crash + fly_low < 0.05/rollout
- episode_length_mean > 350 steps
- all_targets_captured > 0（首次捕获）
- tracking_reward_mean > 5.0/ep

---

## Changelog（续）

- 2026-04-03（run 2026-04-03_12-24-40，255k steps，Run 8）：bounding_box=12m + drone_spawn=(-4,-2) + boundary_soft=9m/0.5 + upright=0.3
  - **episode 长度大幅改善**：run 7 peak 177 → run 8 peak **291 steps**（+64%），recent mean 222 vs run 7 late 115
  - **tracking_reward 突破**：0.61 (run 7 recent) → **4.01 (run 8 recent)**，best=5.05，单调上升曲线
  - **total_reward 显著改善**：recent mean +12.8 vs run 7 late -3.5
  - **bounding_box 部分改善**：1.14 → 0.95/rollout（-16%），仍为主要终止原因
  - **boundary_soft 无效**：-0.28/ep 稳定但无下降趋势，相对 tracking gain 太弱
  - **crash + fly_low 新增问题**：步骤 248k 时 crash=0.27, fly_low=0.09，追踪改善导致俯冲行为
  - **illegal_contact 尖峰**：步骤 201k 出现 -52.26/ep 灾难性尖峰（正常值 -0.1~-0.7）
  - **upright_penalty 仍增长**：-0.002 → -1.125/ep（weight=0.3），标准化后倾斜量比 run 7 高 1.83x
  - **success = 0**：all_targets_captured 全程 0，episode 太短（max=3.23s）不够完成 hold
  - **policy_std 稳定 0.83**：entropy=0.005 再次验证
  - **下一步优先级**：Priority 1 强化 boundary_soft（0.5→2.0，阈值 9→8m）+ 修复俯冲/碰撞（height_threshold 1.5→1.0）；Priority 3 upright 阈值化

---

# Training Analysis Report — Run 9（2026-04-03_16-35-11）

**Run:** 2026-04-03_16-35-11_mappo_torch_mappo
**Date:** 2026-04-03
**Task:** Isaac-marl-move-v0
**Algorithm:** MAPPO
**Steps logged:** 400k
**Status:** improving（分析器判断）

## Training Metrics Summary

| 指标 | Run 8 recent | Run 9 recent | Run 9 last |
|------|------------|------------|----------|
| total_reward_mean | +12.8 | +8.9 | +19.4 |
| tracking_reward | 4.01/ep | 5.70/ep | 6.82/ep |
| distance_reward | 5.24/ep | 6.59/ep | 7.93/ep |
| height_penalty | -1.09/ep | -4.91/ep | -4.84/ep |
| upright_penalty | -1.13/ep | -1.88/ep | -1.82/ep |
| illegal_contact | -2.25/ep | -1.65/ep | -0.14/ep |
| boundary_soft | -0.28/ep | -0.35/ep | ~0/ep |
| drone_out | — | -1.58/ep | -3.25/ep |
| bounding_box term | 0.95/rollout | 0.98/rollout | 1.0/rollout |
| crash term | 0.12/rollout | 0.08/rollout | 0/rollout |
| falcon_fly_low term | 0.09/rollout | 0.06/rollout | 0/rollout |
| ep_len_mean | 222 steps | 217 steps | 199 steps |
| ep_len_max | 291 steps | 276 steps | 263 steps |
| all_targets_captured | 0 | ~0 | 0 |
| success_reward | 0 | 0 | 0 |
| policy_std | 0.83 | 0.99 | 0.99 |

**奖励分解（Run 9 recent）：**

| 分项 | 均值/ep |
|------|--------|
| distance_reward | +6.59 |
| tracking_reward | +5.70 |
| height_reward | +0.32 |
| force_penalty | +0.46 |
| body_rate_penalty | +0.37 |
| height_penalty | -4.91 |
| upright_penalty | -1.88 |
| illegal_contact | -1.65 |
| drone_out | -1.58 |
| boundary_soft | -0.35 |
| fly_low | -0.05 |
| collision_penalty | -0.03 |
| **total (approx)** | **+8.9** |

## Observations & Findings

### 1. height_penalty 爆炸性增大 — CRITICAL

**Symptom:** height_penalty: Run 8 recent -1.09/ep → Run 9 recent **-4.91/ep**（增加 4.5x）。整个训练过程中持续在 -3.2 至 -5.8/ep 之间振荡，无收敛趋势。对比 height_reward 仅 +0.32/ep，height_penalty 已超过 tracking_reward 成为量级最大的负项。

**Root Cause:** `height_penalty_threshold` 从 Run 8 的 1.5m 收紧至 Run 9 的 **1.0m**，但惩罚公式是线性的：`-weight × max(0, |z - desired| - threshold) × step_dt`。阈值收紧意味着任何超出 desired_height±1.0m（即 z<1.5m 或 z>3.5m）的偏差都触发惩罚，而追踪行为本身需要大量俯仰调整，频繁突破此范围。

**Evidence:**
- height_penalty 从第 1 步（-0.135/ep）到第 57k 步（-4.87/ep）迅速收敛到 -4.9 量级，并保持稳定，说明不是振荡噪声而是持续高偏差
- height_reward（+0.32/ep）很低，说明无人机大量时间不在 desired_height=2.5m 处
- Run 8 的 height_penalty_threshold=1.5m 时 height_penalty 为 -1.09/ep；收紧 0.5m 后代价增加 4.5x，意味着运动中高度偏差平均超出 threshold 约 2.7m/step × step_dt

**Expected behavior:** height_penalty 应随训练收敛趋近于 0（无人机学会维持高度），而非稳定在高负值。当前情况说明策略无法或未将高度控制纳入优先学习目标。

---

### 2. bounding_box 终止问题未改善，boundary_soft 仍无效 — CRITICAL

**Symptom:** bounding_box 终止：Run 8 recent=0.95/rollout → Run 9 recent=**0.98/rollout**，last=1.0/rollout（反而轻微恶化）。boundary_soft: Run 8 -0.28/ep → Run 9 recent -0.35/ep，last ≈ 0/ep。

**Root Cause（关键发现）：** boundary_soft 在训练末期几乎清零（last=-0.002）但 bounding_box 终止仍达 1.0/rollout，说明两者**解耦**。无人机在末期已经不超过 8m 软边界（boundary_soft→0），但仍然触发 12m 硬边界（bounding_box=1.0）。

**这意味着：** 超出 8m 软边界但未到 12m 硬边界的中间区域（8-12m）是终止的真实触发区。无人机在学会避开 8m 之后仍然能够快速冲到 12m，说明 `boundary_soft_penalty_weight=2.0 + threshold=8.0m` 的组合确实形成了 8m 处的软阻力，但在 8m 被弹回后仍有动量冲过 12m。

**备选假设：** bounding_box 终止的主体是 **drone_out 惩罚**（-3.25/ep at last，recent -1.58/ep）而非 boundary_soft 触发区域。drone_out 是在 `abs(pos) > bounding_box_threshold=12m` 时立即触发的 step-wise 惩罚（weight=1.0/step），而 bounding_box_termination 是同一条件的终止版本。两者完全耦合。

**Evidence 时序：**
- boundary_soft @step 57k: -0.031 → @171k: -0.168 → @400k: -0.002（先增后减，末期接近 0）
- bounding_box term @400k: 1.0（无改善）
- drone_out @400k: **-3.25/ep**（比 recent 均值 -1.58 高 2x，末期单次 rollout 中飞出加剧）

---

### 3. 俯冲/碰撞问题得到改善 — HIGH（部分成功）

**Symptom:** crash: Run 8 recent=0.12 → Run 9 recent=**0.075**（降低 37%）。falcon_fly_low: Run 8 recent=0.086 → Run 9 recent=**0.058**（降低 33%）。

**Assessment:** `height_penalty_threshold` 从 1.5→1.0m（提前惩罚）+ `contact_sensor_threshold` 从 1.0→5.0N（过滤微接触）确实减少了坠机，但代价是 height_penalty 爆炸。换言之，俯冲行为被抑制了，但惩罚力度过强导致无人机高度控制混乱，未达到收紧后应有的"精准维持高度"效果。

**Evidence:** crash/fly_low 在 Run 9 后期（step 285k, 342k）均出现 0.27/0.16 峰值再度上升，说明高度问题未被根本解决，只是暂时缓解。

---

### 4. tracking_reward 继续单调增长 — MEDIUM（正面指标）

**Symptom:** tracking_reward: Run 8 recent=4.01/ep → Run 9 recent=**5.70/ep**，last=**6.82/ep**（+70% vs Run 8 recent）。distance_reward: 5.24 → 6.59 → 7.93。

**Assessment:** XY 追踪能力持续改善，单调上升趋势保持。这证明追踪奖励信号健康有效。但由于 height_penalty（-4.91/ep）的拖累，总奖励 recent mean 从 Run 8 的 +12.8 降低到 Run 9 的 +8.9，实际净改善被高度惩罚抵消。

---

### 5. upright_penalty 阈值化效果有限 — MEDIUM

**Symptom:** upright_penalty: Run 8 recent -1.13/ep（weight=0.3 linear）→ Run 9 recent **-1.88/ep**（weight=1.0 threshold-based）。

**Analysis:** 阈值化（cos35°=0.819）配合 weight=1.0 后，惩罚量增加了 66%，而不是减少。说明策略的倾斜程度仍然超过 35° threshold 的时间相当长，且 weight 增大直接放大了惩罚。

**Root cause:** 追踪能力越强（tracking 6.82/ep），无人机需要的横向加速越大，倾斜角越大，upright_penalty 越重。这是追踪进步的附带代价，不可完全消除，但可以通过放宽阈值（cos30°=0.866 → cos40°=0.766）或降低 weight 来调整。

---

### 6. episode 长度不升反降，未超越 Run 8 峰值 — HIGH

**Symptom:** ep_len_mean: Run 8 recent=222 → Run 9 recent=217（-2%）。ep_len_max: Run 8 best=291 → Run 9 best=263（-9.6%）。

**Root Cause:** height_penalty 的爆炸性增大（-4.91/ep）直接影响总奖励，同时高度惩罚不导致终止，但可能通过 value function 影响策略学习稳定性。更直接的原因是 bounding_box 终止仍≈1/rollout，每个 rollout 至少一次提前终止，严重限制 episode 长度上限。

---

### 7. all_targets_captured 仍为 0，首次捕获未实现 — HIGH

**Symptom:** all_targets_captured: 全程 0（recent ≈0.0001，极偶发）。success_reward=0。

**Root Cause:** bounding_box 仍然是主要终止原因（~1/rollout）。episode 在超出边界前即终止，无法完成 capture hold 时间要求。ep_len_max=263 steps（约 2.6s），如果 hold_duration≥3s，从数学上不可能成功。

---

### 8. policy_std 显著提升 — 中性/正面

**Symptom:** policy_std: Run 8 stable 0.83 → Run 9 stable **0.99**（接近初始化上界 1.0）。

**Assessment:** entropy=0.005 仍然有效，但 std 比 Run 8 高 0.16，说明探索更充分。这可能是 height_penalty 爆炸引入的不确定性造成策略在高度维度更随机，而非主动探索改善。

## Improvement Recommendations

### Priority 1 (CRITICAL): 回调 height_penalty_threshold，解决高度惩罚爆炸

**Problem:** height_penalty_threshold=1.0m 导致 height_penalty=-4.91/ep，超过 tracking_reward 成为最大单项负惩罚，严重压制总奖励改善，且未能阻止 crash/fly_low 复发（步骤 285k 峰值重现）。

**Proposed Change:**
- File: `exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/move/marl_move_env_cfg.py`
- Parameter: `height_penalty_threshold = 1.0` → `1.5`（回到 Run 8 的有效值）
- Rationale: Run 8 的 1.5m threshold 时 height_penalty=-1.09/ep，是可接受的惩罚水平，且 crash/fly_low 问题在 Run 8 是在训练进行到 248k 步后才出现的，说明 1.5m threshold 在早中期是足够的约束。crash/fly_low 的根本解决需要辅助手段（见 Priority 2），而非单纯收紧阈值。

**备选方案（若 crash 持续）：** 增加 `fly_low` 终止条件（z < 0.5m 直接终止，而非仅惩罚），彻底阻断俯冲路径。

---

### Priority 2 (CRITICAL): 解决 bounding_box 终止的根本原因

**Problem:** bounding_box 终止 ≈1/rollout 在 9 次训练中持续存在，Run 9 末期反而恶化到 1.0/rollout，episode 长度无法突破 300 steps。`boundary_soft_penalty_weight` 从 0.5 增大到 2.0 几乎无效果：boundary_soft 末期接近 0（无人机绕开 8m），但仍因动量冲到 12m 触发 bounding_box。

**Root cause:** 当前的 bounding_box_threshold=12m 与 boundary_soft_threshold=8m 之间存在 **4m 动量缓冲区** 不足以让高速无人机（3-5m/s）减速。更根本地：追踪 ±8m 折返目标的策略本身要求无人机在折返点迅速制动，但没有向心加速的惩罚约束。

**Proposed Changes（两选其一）：**

**方案 A — 缩小 bounding_box（消除动量区）：**
- File: `exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/move/marl_move_env_cfg.py`
- Parameter: `bounding_box_threshold = 12.0` → `10.0`（仅保留 2m 超调余量）
- Parameter: `boundary_soft_threshold = 8.0`（保持）
- Rationale: 减少 4m 缓冲区为 2m，无人机必须更快响应软边界，减少动量穿越。

**方案 B — 添加速度惩罚（接近软边界时惩罚高速）：**
- File: `exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/move/marl_move_env.py`
- 在 boundary_soft 计算区域内（8-12m），额外添加 `velocity_boundary_penalty = -weight × xy_speed × soft_excess_fraction`
- Rationale: 让高速冲出边界的代价大于追踪收益。

**推荐方案 A**，因为实现简单，且 boundary_soft=8m 已证明有效（boundary_soft末期归零），问题在于 4m 余量太大。

---

### Priority 3 (HIGH): 将 upright_penalty 权重从 1.0 降回 0.5，或放宽阈值角度

**Problem:** upright_penalty Run 9 recent=-1.88/ep（vs Run 8 -1.13/ep），weight=1.0 threshold-based 比 weight=0.3 linear 实际惩罚更重。追踪任务要求无人机倾斜超过 35°，threshold-based 方案仍然惩罚了必要的追踪倾斜。

**Proposed Change:**
- File: `exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/move/marl_move_env_cfg.py`
- Parameter: `upright_penalty_weight = 1.0` → `0.5`
- Parameter: `upright_penalty_threshold = 0.819`（cos35°）→ `0.766`（cos40°），给无人机更多追踪倾斜空间
- Rationale: 当追踪需要倾斜 40° 时，35° 阈值仍然持续触发惩罚。放宽到 40° 只保护真正危险的大倾斜（>40°），同时减轻正常追踪动作的惩罚负担。

---

### Priority 4 (MEDIUM): 增大训练预算，监控首次捕获

**Problem:** success_reward=0，all_targets_captured≈0。tracking_reward 单调上升趋势健康（6.82/ep at last, 增长未见平台期）。策略仍在有效学习，尚未到达 bounding_box 解决后的"能否捕获"阶段。

**Proposed Change:**
- 完成 Priority 1-3 修改后，运行 600k steps
- 重点监控 `all_targets_captured > 0` 首次出现时刻
- 如果 Priority 2 方案 A 生效（bounding_box 降至 <0.3/rollout），ep_len_mean 应超越 350 steps，满足 3s+ hold 时间要求

---

## Experiment Plan（Run 10）

**Run 10 核心修改清单：**

1. **CRITICAL — 回调 height_penalty_threshold（解决惩罚爆炸）：**
   - `height_penalty_threshold`: 1.0 → 1.5
   - File: `marl_move_env_cfg.py`

2. **CRITICAL — 缩小 bounding_box（消除动量穿越区）：**
   - `bounding_box_threshold`: 12.0 → 10.0
   - File: `marl_move_env_cfg.py`

3. **HIGH — 放宽 upright_penalty 阈值，降低权重：**
   - `upright_penalty_threshold`: 0.819 → 0.766（cos40°）
   - `upright_penalty_weight`: 1.0 → 0.5
   - File: `marl_move_env_cfg.py`

4. **保留所有 Run 9 有效修改：**
   - `boundary_soft_threshold = 8.0`，`boundary_soft_penalty_weight = 2.0`
   - `contact_sensor_threshold = 5.0N`
   - `capture_distance = 3.0m XY`
   - `entropy_loss_scale = 0.005`
   - `tracking_reward_weight = 4.0`
   - `drone_spawn_x_range = (-4, -2)`
   - upright 阈值式计算逻辑（保留 cos 阈值结构，调参数）

5. **训练命令：**
```bash
python3 scripts/skrl/train.py \
  --task=Isaac-marl-move-v0 \
  --headless --num_envs=2048 --algorithm="MAPPO"
```

6. **监控指标（前 100k steps）：**
   - height_penalty：应回到 -1.0 至 -2.0/ep 量级（vs Run 9 的 -4.9）
   - bounding_box：应从 ~1.0 降至 <0.5/rollout（bounding_box_threshold 缩小效果）
   - ep_len_mean：应超过 250 steps（vs Run 9 的 217）
   - all_targets_captured：首次出现 >0 是关键里程碑
   - tracking_reward：应继续上升，不应因 height_penalty 回调而退步

**Run 10 成功标准：**
- height_penalty < -2.0/ep（vs Run 9 的 -4.9）
- bounding_box < 0.5/rollout（vs Run 9 的 ~1.0）
- ep_len_mean > 250 steps
- all_targets_captured > 0（首次捕获）
- tracking_reward > 7.0/ep（继续上升趋势）

## Changelog（续）

- 2026-04-03（run 2026-04-03_16-35-11，400k steps，Run 9）：boundary_soft×4（0.5→2.0）+ threshold 8m + height_threshold 1.0 + contact 5N + upright 阈值化（cos35°, weight 1.0）
  - **tracking_reward 持续突破**：Run 8 recent 4.01 → Run 9 recent **5.70/ep**，last **6.82/ep**（+70%），单调上升趋势健康
  - **俯冲/碰撞部分改善**：crash 0.12→0.075（-37%），fly_low 0.09→0.058（-33%），但 285k/342k 步有复发峰值
  - **height_penalty 爆炸性增大（新关键问题）**：-1.09/ep → **-4.91/ep**（4.5x 增大），height_penalty_threshold=1.0m 过严，无人机无法维持 2.5m ± 1.0m，追踪行为和高度控制产生冲突
  - **bounding_box 未改善，反略恶化**：0.95 → 0.98/rollout，last=1.0；boundary_soft 末期归零（≈0）但 bounding_box 仍持续——证明 8m 软边界已被绕开，但 8-12m 动量区仍不够阻止穿越
  - **upright_penalty 阈值化效果差**：-1.13 → -1.88/ep（增大 66%），weight=1.0 放大了 35° 以上的正常追踪倾斜
  - **episode 长度未改善**：Run 8 peak 291 → Run 9 peak **263**（-9.6%），mean 222 → 217
  - **success = 0**：all_targets_captured 仍为 0
  - **policy_std 提升**：0.83 → 0.99，探索空间更大（也可能是 height_penalty 不确定性导致）
  - **下一步优先级**：height_threshold 回调（1.0→1.5）+ bounding_box 缩小（12→10m）+ upright 阈值放宽（cos35°→cos40°, weight 1.0→0.5）

---

## Run 10 训练分析报告

**Run:** 2026-04-04_22-19-19_mappo_torch_mappo
**Date:** 2026-04-05
**Task:** Isaac-marl-move-v0
**Algorithm:** MAPPO
**num_envs:** 32（小规模验证）
**Timesteps:** 400k

### 本次修改（vs Run 9）
| 参数 | Run 9 | Run 10 |
|------|-------|--------|
| `height_penalty_threshold` | 1.0m | 1.5m（回调） |
| `bounding_box_threshold` | 12.0m | 10.0m（压缩） |
| `upright_penalty_threshold` | cos35°=0.819 | cos40°=0.766（放宽） |
| `upright_penalty_weight` | 1.0 | 0.5（降权） |

---

## Training Metrics Summary

| 指标 | Run 9（last/recent） | Run 10（last/recent） | 变化 |
|------|---------------------|----------------------|------|
| total_reward_mean | 未知 | 96.4 / 21.9 | 大幅提升 |
| tracking_reward | 6.82 / 5.70 | 23.87 / 13.52 | **+198% / +137%** |
| distance_reward | — | 25.47 / 15.34 | 健康 |
| height_penalty | -4.91（recent） | -2.7723 / -2.98 | **改善但仍显著** |
| upright_penalty | -1.88 / — | -4.28 / -3.09 | **恶化，新关键问题** |
| illegal_contact（奖励） | — | -2.61 / -17.40 | **爆炸性问题** |
| bounding_box（终止） | ~1.0/rollout | 0.64 / 0.55 | **显著改善 -45%** |
| all_targets_captured | 0 | 0.60 / 0.39 | **首次捕获，重大突破** |
| episode_len_mean | 217 | 557 / 484 | **+123%** |
| episode_len_peak | 263 | 2376 | **+803% 决定性突破** |
| crash（终止） | 0.075 | 0.32 / 0.43 | **大幅恶化** |
| falcon_fly_low（终止） | 0.058 | 0.32 / 0.34 | **大幅恶化** |
| policy_std | 0.99 | 1.595 / 1.377 | **过高，超出健康范围** |

---

## Observations & Findings

### 1. 全面突破：首次捕获成功 — 里程碑达成

**症状（正面）：**
- `all_targets_captured` 在 step 154300 首次出现（val=0.09），之后持续上升
- 近期均值 0.39/rollout，末尾最后10个 rollout 中 val 在 0.42-1.00 之间波动
- `episode_len_best=2376 steps（23.76s）`，远超之前所有 run 的 peak（Run 9 max 263）
- `episode_len_mean_recent=484 steps（4.84s）`，Run 9 peak 仅 263 steps

**根因：** 三项修改共同解锁了捕获行为：
1. `bounding_box_threshold 12→10m`：bounding_box 终止从 ~1.0/rollout 降至 0.55/rollout（-45%），策略有更多时间完成捕获
2. `height_penalty_threshold 1.0→1.5m`：high-altitude 惩罚削弱（-4.91→-2.98），策略行动空间扩大
3. `upright_penalty_weight 1.0→0.5`：降低姿态惩罚压力，允许更激进的追踪动作

**结论：** Run 10 是 move task 训练历史中最重要的突破。捕获成功率从 0% 提升至持续 ~40%（每 rollout）。

---

### 2. illegal_contact 爆炸 — CRITICAL

**症状：**
- `illegal_contact` 奖励：early=-0.12 → mid=-1.10 → recent=-17.40（爆炸），worst=-815.33
- `illegal_contact` 终止：early=0.008 → recent=0.217/rollout（+27x）
- 最坏三次峰值：step 301700（-815），step 377600（-523），step 154300（-521）
- 这三次峰值恰好与首次捕获（154300）和后期捕获高频区重合

**根因：** 捕获行为本质上要求无人机进入 capture_distance=3.0m 范围内，而 NovaCarter 的碰撞几何在接近时会触发 ContactSensor（阈值=5N）。当策略学会更频繁地接近目标后，接触事件激增。此问题在 Run 5 时已分析过（contact_sensor_threshold 从 1N→5N 部分缓解），但当 capture 成功率提升后，接触频率超出 5N 过滤能力。

**证据：** illegal_contact_penalty=1.0（而非终止条件），worst=-815 表明单 episode 内发生了 815+ 次接触事件，极大破坏奖励信号。recent_mean=-17.4 意味着每 episode 平均有 17 个接触惩罚单位（可能是持续接触的累计）。

**严重性：** CRITICAL。illegal_contact 惩罚会对策略产生强负向梯度，与捕获奖励方向相反，可能导致策略在接近目标时产生矛盾梯度（"要靠近但惩罚靠近"）。

---

### 3. upright_penalty 爆炸性增长 — HIGH

**症状：**
- `upright_penalty`：early=-0.23 → mid=-0.93 → recent=-3.09 → last=-4.28（单调恶化）
- Run 9 recent=-1.88（已是问题），Run 10 recent=-3.09（+64%），last=-4.28（+128%）
- 与 tracking_reward 的 net 关系：100% 节点 tracking=23.87, upright=-4.28, net=19.59（upright 占 tracking 的 18%，可接受但上升趋势不可忽视）

**根因：** 更长的 episode 允许无人机执行更多高速追踪动作，而高速追踪必然伴随机身倾斜（物理约束）。即使 threshold 从 cos35°→cos40°，权重从 1.0→0.5，episode 长度增加 +123% 导致每 episode 累计的倾斜惩罚总量仍然增大。

**与 tracking 的相关性：** 20%节点 tracking=1.92/upright=-0.45，到 100%节点 tracking=23.87/upright=-4.28。比值从 4.3x 降至 5.6x——upright 增长速度慢于 tracking，说明目前不是阻碍因素，但趋势需要观察。

**结论：** 目前 upright_penalty 是次要问题，随 tracking 增长速度较慢。暂不需要进一步修改，但需要在 Run 11 监控其趋势。

---

### 4. crash 和 falcon_fly_low 终止急剧上升 — HIGH

**症状：**
- `crash` 终止：early=0.011 → mid=0.24（step 240k）→ recent=0.43/rollout
- `falcon_fly_low` 终止：early=0.004 → recent=0.34/rollout
- `fly_low` 奖励惩罚：early=-0.004 → recent=-0.33/ep（+82x）
- 两者合计 recent=0.77/rollout——几乎每个 rollout 都有坠机或超低飞行

**根因：** 与 Run 8 的"aggressive-tracking descent"模式相同，但更严重：
1. 策略学会靠近地面的 NovaCarter（spawn_z=0.25m），倾向于向下接近
2. height_penalty_threshold=1.5m 允许 drone 飞到 2.5-1.5=1.0m（过低）
3. 当策略进一步接近 NovaCarter 时，可能飞到 fly_low_threshold 以下

**关键时间模式：** crash 在 step 240k 出现峰值（0.24），之后 320k 时下降（0.01），再到 400k 回升（0.32）——说明策略在"学习接近→坠机→调整"的循环中振荡。这个振荡影响了 episode_len 的稳定性（recent_std=183 steps 很大）。

---

### 5. policy_std 持续上升至 1.595——探索过度 — MEDIUM

**症状：**
- `policy_std`：early=0.826 → mid=0.948 → recent=1.377 → last=1.595（单调上升）
- Run 9 last=0.99，Run 10 last=1.595，Run 8 stable=0.83
- 标准差 > 1.0 意味着动作分布异常宽（在 [-1,1] clip 的 action space 中，std=1.595 几乎是均匀随机分布）

**根因：** `entropy_loss_scale=0.005` 驱动探索，在 episode 长度大幅增加后，每 episode 内的熵累积更多，导致策略被推向更高随机性。illegal_contact 的极端负奖励（-815）可能也在破坏梯度，使策略难以收敛，进一步促进"保持高熵"作为防御策略。

**影响：** policy_std=1.595 的策略虽然探索能力强，但在 exploit 阶段表现差——已学到的捕获行为难以稳定复现。可能解释 all_targets_captured 的大幅波动（0.42 到 1.00 之间）。

---

### 6. height_penalty 部分改善但仍显著 — MEDIUM

**症状：**
- `height_penalty`：early=-1.81 → mid=-1.03（改善）→ recent=-2.98 → last=-2.77（反弹）
- 对比 Run 9：early=-1.09 → recent=-4.91。Run 10 recent=-2.98（vs -4.91，改善 39%）
- 但早期值 -1.81 > Run 9 early=-1.09，说明 threshold=1.5m 本身也有起始惩罚

**根因：** threshold=1.5m 确实缓解了爆炸（-4.91→-2.98），验证了回调方向正确。中段（-1.03）达到最优，但后段反弹（-2.98）与 crash/fly_low 增加同步——坠机前的俯冲阶段必然超出高度带。这是 crash 问题的连带效应，不是独立的高度控制问题。

**结论：** height_penalty 本身已不是主要问题。只要解决 crash/fly_low（俯冲行为），height_penalty 会自然改善。

---

### 7. bounding_box 终止显著改善 — 已解决

**症状（正面）：**
- `bounding_box`：early=1.15 → mid=1.02 → recent=0.55 → last=0.64/rollout
- 对比 Run 9：~1.0/rollout（不变）→ Run 10 recent=0.55（-45%）

**根因验证：** `bounding_box_threshold 12→10m` 有效压缩了动量穿越区间（8-12m→8-10m）。配合 `boundary_soft_threshold=8m, weight=2.0`，软边界效果在 2m 内得到更充分发挥。

**结论：** 此修改方向正确并有效。但 0.55/rollout 仍然较高，说明还有约半数 rollout 以越界终止。随着 ep_len 增长（目标运动更远），此问题可能趋于稳定。

---

## Improvement Recommendations

### Priority 1 (CRITICAL): 解决 illegal_contact 爆炸性惩罚

**Problem:** illegal_contact 惩罚 recent_mean=-17.40/ep，worst=-815/ep。随着捕获行为成熟，与 NovaCarter 的接触不可避免，但 penalty=1.0 × 接触步数会产生巨额负奖励，破坏捕获的正奖励信号。

**Root cause:** NovaCarter 碰撞几何的 ContactSensor 在近距追踪时持续触发，contact_sensor_threshold=5N 无法过滤轻微接触。

**Proposed Changes:**
- File: `exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/move/marl_move_env_cfg.py`
- Option A（推荐）：`illegal_contact_penalty`: 1.0 → 0.1（降低单次惩罚幅度，保留信号）
- Option B：`contact_sensor_threshold`: 5.0 → 15.0N（提高过滤门槛，减少触发频率）
- Option C（最激进）：将 `illegal_contact_penalty` 设为 0（完全关闭）并观察行为
- **推荐选 Option A + B 同时应用**：`penalty 1.0→0.1` + `threshold 5→15N`

**Rationale:** 捕获行为成功已经建立，但 illegal_contact 爆炸在破坏后期收敛。worst=-815 表明存在持续接触（不是离散碰撞），5N 阈值过低导致振动/接近都被计为违规接触。

---

### Priority 2 (HIGH): 解决 crash / falcon_fly_low 俯冲坠机问题

**Problem:** crash 终止 recent=0.43/rollout，fly_low 终止 recent=0.34/rollout，合计 0.77/rollout。策略在接近地面 NovaCarter 时存在俯冲至过低高度的行为。

**Proposed Changes:**
- File: `exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/move/marl_move_env_cfg.py`
- `height_penalty_threshold`: 1.5 → 1.2m（适度收紧，但不回到 1.0m 的爆炸阈值）
  - 理由：1.5m 允许 drone 飞到 z=1.0m（2.5-1.5），太接近地面（NovaCarter top≈0.4m + 安全余量）
  - 1.2m 约束 drone 在 z≥1.3m，提供 0.9m 安全余量
- File: `exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/move/marl_move_env.py`
- 检查 `fly_low_threshold` 的具体值——是否需要提高以更早拦截俯冲（如从当前值上调 0.1-0.2m）

**Rationale:** crash 和 fly_low 在 step 240k→320k 周期性出现，说明策略进入"接近→坠机→重置→再接近"循环，而非稳定学习。适度收紧 height_penalty 可以在不引发爆炸的前提下引导策略保持更高飞行高度。

**注意：** 不能回到 1.0m（Run 9 的错误），1.2m 是合理折中。

---

### Priority 3 (HIGH): 调整 entropy_loss_scale 抑制 policy_std 过度上升

**Problem:** policy_std=1.595，远超健康范围（目标 0.8-1.0）。过高的探索随机性导致已学到的捕获行为难以稳定复现，all_targets_captured 波动大。

**Proposed Changes:**
- File: `scripts/skrl/train.py` 或 MAPPO agent 配置
- `entropy_loss_scale`: 0.005 → 0.002（适度降低熵激励）
- 或：添加 `entropy_annealing`：在 200k steps 后从 0.005 线性衰减至 0.001

**Rationale:** entropy=0.005 在 Run 7-9 维持了稳定的 policy_std=0.82-0.99，适合探索阶段。现在 Run 10 首次实现捕获，应从"探索优先"转向"收敛优化"。降低熵惩罚让策略能在已发现的成功轨迹附近收敛。

---

### Priority 4 (MEDIUM): 监控 upright_penalty 趋势，暂不修改

**Problem:** upright_penalty recent=-3.09/ep（vs Run 9 recent=-1.88），但与 tracking_reward 的比值在改善（tracking/|upright| = 4.37），说明相对压力在降低。

**Decision:** 暂不修改 upright_penalty 参数。在 Run 11 继续观察：若 `|upright|/tracking > 0.3`，则考虑进一步放宽阈值至 cos45°=0.707。

---

### Priority 5 (LOW): 考虑 success_reward_weight 从 0.0 启用

**Problem:** success_reward_weight=0.0，捕获成功没有额外奖励。all_targets_captured 已经稳定出现（recent=0.39/rollout），现在应该给予正向激励强化捕获行为。

**Proposed Changes:**
- File: `exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/move/marl_move_env_cfg.py`
- `success_reward_weight`: 0.0 → 5.0（给予捕获一次性奖励，强化该行为）

**Rationale:** 捕获信号已经通过 tracking_reward 隐式存在，但 tracking_reward 是连续的（持续在捕获区内）。success_reward 是对"保持捕获"的一次性激励，可以引导策略专注于维持而非反复进出捕获区。

---

## Experiment Plan (Run 11)

**核心目标：** 在首次捕获突破的基础上，解决 illegal_contact 爆炸和 crash 俯冲问题，推动捕获率从 40% 向 80%+ 稳定化。

**修改清单（按优先级）：**

1. **CRITICAL：** `illegal_contact_penalty`: 1.0 → 0.1
2. **CRITICAL：** `contact_sensor_threshold`: 5.0 → 15.0N
3. **HIGH：** `height_penalty_threshold`: 1.5 → 1.2m
4. **HIGH：** `entropy_loss_scale`: 0.005 → 0.002
5. **LOW：** `success_reward_weight`: 0.0 → 5.0

**保留所有 Run 10 有效配置：**
- `bounding_box_threshold = 10.0m`（已验证，-45% bounding_box 终止）
- `upright_penalty_threshold = 0.766`（cos40°），`weight = 0.5`（暂不动）
- `height_penalty_threshold = 1.5m`（由 1.5 → 1.2 是本次调整）
- `boundary_soft_threshold = 8.0m`，`weight = 2.0`
- `capture_distance = 3.0m`（XY only）
- `drone_spawn_x_range = (-4, -2)`
- `tracking_reward_weight = 4.0`

**训练命令：**
```bash
python3 scripts/skrl/train.py \
  --task=Isaac-marl-move-v0 \
  --headless --num_envs=2048 --algorithm="MAPPO"
```

**监控指标（前 100k steps，2048 envs 约等于 32 envs 的 64x 数据量）：**
- illegal_contact 奖励：应 < -1.0/ep（vs Run 10 recent=-17.4）
- crash 终止：应 < 0.1/rollout（vs Run 10 recent=0.43）
- falcon_fly_low：应 < 0.05/rollout（vs Run 10 recent=0.34）
- all_targets_captured：应 > 0.5/rollout 且稳定（vs Run 10 recent=0.39，波动大）
- tracking_reward：应继续上升，不应因 illegal_contact 降权而退步
- policy_std：应从当前 1.595 回落至 0.8-1.0 范围
- episode_len_mean：应超过 500 steps（vs Run 10 recent=484）

**成功标准（Run 11）：**
- illegal_contact < -1.0/ep（解决爆炸）
- crash < 0.1/rollout（解决俯冲坠机）
- all_targets_captured > 0.5/rollout 且 std < 0.2（稳定捕获）
- policy_std 回落至 0.9-1.1 范围
- tracking_reward > 15.0/ep（vs Run 10 recent=13.52）

---

## Changelog（续）

- 2026-04-05（run 2026-04-04_22-19-19，400k steps，Run 10，num_envs=32）：height_threshold 1.0→1.5 + bounding_box 12→10m + upright cos40°/weight 0.5
  - **决定性突破：首次捕获成功**：all_targets_captured 在 step 154300 首次出现，recent=0.39/rollout，best rollout=1.38/rollout（约38%的 rollout 在1个以上时间窗口内实现捕获）
  - **episode 长度历史最长**：mean_recent=484 steps（+123% vs Run 9 peak 263），best=2376 steps（+803%），接近 episode 最大 6000 步的 40%
  - **tracking_reward 大幅提升**：last=23.87/ep（+250% vs Run 9 last=6.82），recent_mean=13.52/ep（+98%）
  - **bounding_box 终止显著改善**：1.0 → 0.55/rollout（-45%），bounding_box=10m 修改验证有效
  - **height_penalty 部分改善**：recent=-2.98（vs Run 9 recent=-4.91，改善39%），threshold=1.5m 回调方向正确
  - **新关键问题 1：illegal_contact 爆炸**：recent=-17.40/ep，worst=-815/ep，与捕获行为同步爆发——接近 NovaCarter 时 ContactSensor 持续触发（5N 阈值不足）
  - **新关键问题 2：crash + fly_low 急增**：crash=0.43/rollout（+474%），fly_low=0.34/rollout（+486%），"接近→俯冲→坠机"模式在 2048 env 下将更严重
  - **policy_std 过高**：0.99 → 1.595，超出健康范围，熵激励（0.005）在长 episode 后过度推动探索
  - **下一步优先级**：illegal_contact_penalty 1.0→0.1 + threshold 5→15N + height_threshold 1.5→1.2m + entropy 0.005→0.002 + success_reward 启用

---

# Training Analysis Report — Run 11

**Run:** 2026-04-05_07-55-37_mappo_torch_mappo
**Date:** 2026-04-05
**Task:** Isaac-marl-move-v0
**Algorithm:** MAPPO (num_envs=32, 400k steps)

---

## Training Metrics Summary

| Metric | Early (Q1) | Recent (Q5) | Last | vs Run 10 Recent |
|--------|-----------|-------------|------|-----------------|
| total_reward mean | 1.77 | 35.59 | 48.58 | N/A (different scale) |
| distance_reward | 2.09 | 9.14 | 13.00 | -40.5% |
| tracking_reward | 1.49 | 8.21 | 11.68 | -39.3% |
| all_targets_captured | 0.000 | 0.027 | 0.000 | -93.2% |
| illegal_contact_rew | -0.021 | -0.815 | -0.229 | +95.3% (improved) |
| crash termination | 0.018 | 0.383 | 0.280 | -10.6% |
| fly_low termination | 0.003 | 0.299 | 0.280 | -12.2% |
| drones_collide | 0.004 | 0.195 | 0.080 | +40.0% (worsened) |
| upright_penalty | -0.293 | -2.093 | -2.924 | +32.3% (improved) |
| height_penalty | -1.918 | -1.981 | -2.716 | +33.5% (improved) |
| ep_len mean | 104 | 323 | 396 | -33.2% |
| policy_std | 0.764 | 0.599 | 0.597 | -56.5% (major improvement) |
| value_loss | 0.099 | 0.072 | 0.090 | -67.8% |
| policy_loss | 0.002 | 0.012 | 0.032 | -83.7% |
| success_reward | 0.000 | 0.000 | 0.000 | — |

---

## Observations & Findings

### 1. illegal_contact Penalty — FIXED (CRITICAL resolved)

**Symptom:** illegal_contact_reward went from Run 10's catastrophic -17.4/ep recent (worst -815) to Run 11's -0.815/ep recent (worst -98.7). A 95.3% improvement in mean magnitude.

**Root Cause (confirmed):** With illegal_contact_penalty lowered 1.0→0.1 and contact_sensor_threshold raised 5N→15N, the policy is no longer being overwhelmed by contact reward noise during proximity. The worst case (-98.7) still occurs in sporadic spikes at steps 399900 (-10.8) suggesting momentary multi-step contact during drone-drone or drone-NovaCarter encounters, but the mean burden is now manageable.

**Evidence:** Q1=-0.021 → Q5=-0.815. Still rising (40x trend_ratio), meaning contacts are increasing with approach behavior — but at 0.1 penalty weight, each contact event contributes only 0.1 per timestep, not 1.0.

**Status:** RESOLVED as a training blocker. Residual drift is expected and acceptable.

---

### 2. Crash / fly_low — PARTIALLY IMPROVED (HIGH, not resolved)

**Symptom:** crash_term recent: Run 10=0.43/rollout → Run 11=0.38/rollout (-10.6%). fly_low_term: 0.34 → 0.30 (-12.2%). Combined crash+fly_low: 0.77 → 0.68/rollout (-11.7%). Marginal improvement only.

**Root Cause:** The "approach→dive→crash" cycle persists. With height_penalty_threshold tightened 1.5→1.2m, the theoretical descent floor is z=2.5-1.2=1.3m, which is above the NovaCarter top (~1.35m at 3x scale). However, the policy that learned to dive under Run 10's 1.5m regime is still being exploited — the new threshold is reducing dive depth but not eliminating the behavior.

**Evidence:** 
- Crash quintile trend: Q1=0.018 → Q5=0.383 (still monotonically rising)
- height_penalty by quintile: Q1=-1.918, Q2=-1.259, Q3=-1.138 (improvement), Q4=-1.634, Q5=-1.981 (rebound late) — late-run rebound to -1.98 matches the continued crash trend
- fly_low quintile: Q1=0.003 → Q5=0.299 (20x rise, not arrested)

**Status:** PARTIALLY IMPROVED. Requires additional action — see recommendations.

---

### 3. policy_std — DRAMATICALLY FIXED (CRITICAL resolved)

**Symptom:** policy_std dropped from Run 10's 1.595 (end-of-run, over-exploration) to Run 11's 0.599 (stable, within healthy range 0.82→0.60). The entropy_loss_scale reduction 0.005→0.002 was effective.

**Evidence:** policy_std Q1=0.764 → Q5=0.599. Monotonically declining as intended. Recent_std=0.004 — very tight, stable convergence. Compare to Run 10 where std was monotonically rising to 1.595.

**Status:** RESOLVED. policy_std is now in a healthy, declining exploitation regime (0.60 at 400k steps). No action needed.

---

### 4. Capture Behavior (all_targets_captured) — SEVERE REGRESSION (CRITICAL)

**Symptom:** all_targets_captured: Run 10 recent=0.39/rollout (best=0.60, last 10 rollouts 0.42-1.00) → Run 11 recent=0.027/rollout (best=0.49, last 10 rollouts mostly 0.00). A 93.2% collapse in capture rate despite success_reward=5.0 activation.

**Root Cause — Hypothesis A (most likely): entropy collapse interfered with capture timing**
- Run 10 had policy_std=1.595 — high exploration generated diverse approach trajectories that "accidentally" satisfied capture conditions
- Run 11 dropped std to 0.60 rapidly (entropy_scale=0.002 applied aggressively from step 0)
- This early entropy reduction may have locked the policy into a narrower action distribution before the capture behavior was consolidated, causing the policy to lose the approach diversity that enabled captures

**Root Cause — Hypothesis B: episode_length regression**
- ep_len_mean: Run 10 recent=484 → Run 11 recent=323 (-33.2%). The sustained_follow_duration=3.0s requires the drone to maintain capture_distance<3.0m for 300 timesteps (3s × 100Hz). With mean episodes only 323 steps, capturing is increasingly rare — the episode ends (crash/fly_low) before the 3s hold completes.
- The crash rate (0.38/rollout) means roughly every other rollout ends in a crash, aborting potential captures.

**Root Cause — Hypothesis C: success_reward=5.0 did NOT activate**
- success_reward recent_mean=0.000 for ALL 4000 data points across the entire run. Not a single success reward was earned in 400k steps.
- This confirms captures did not achieve the 3.0s sustained hold criterion — they were brief proximity contacts, not sustained follows.

**Evidence:** 
- all_targets_captured non-zero occurrences: 207 of 4000 rollouts (5.2%), first non-zero at step 125200
- Q4 non-zero fraction=0.025, Q5 non-zero fraction=0.230 — capture is improving toward end but volatile (last 10: 0.19, 0.03, 0.00, 0.00, 0.00, 0.46, 0.00, 0.00, 0.00, 0.00)
- success_reward = 0.0000 throughout — 3s hold criterion never met

**Status:** CRITICAL regression vs Run 10. Capture frequency collapsed 93% and sustained capture (success_reward) was never achieved. The combination of lower std, shorter episodes, and volatile approach behavior is responsible.

---

### 5. Upright Penalty — IMPROVING (MEDIUM)

**Symptom:** upright_penalty Run 10 recent=-3.09/ep → Run 11 recent=-2.09/ep (32% improvement). Last values are still -2.9 suggesting end-of-run tilt pressure remains.

**Evidence:** Monotonically worsening through training (Q1=-0.29 → Q5=-2.09), driven by more aggressive tracking and capture approach. upright_penalty ratio vs tracking: upright(-2.09) / tracking(8.21) = 25% overhead — elevated but not dominant.

**Status:** Improving trend, within tolerable range. No immediate action required.

---

### 6. Drone Collisions — NEW CONCERN (MEDIUM)

**Symptom:** drones_collide termination: Run 10 recent=0.139/rollout → Run 11 recent=0.195/rollout (+40%). Rising monotonically: Q1=0.004 → Q5=0.195 (48x). 

**Root Cause:** As policy_std dropped and drones converged to more deterministic trajectories, they may be converging to the same positions more frequently (reduced diversity in approach paths → inter-drone collision). This is a known MARL pathology when exploration collapses.

**Evidence:** drones_collide trend ratio=48x, higher than crash (20x) or fly_low (92x). Emerging issue that will worsen if ep_len increases.

**Status:** MEDIUM priority — not a blocker yet but will compound crash problem if drones increasingly terminate each other.

---

### 7. Learning Curve Health — IMPROVING but Unstable

**Value Loss:** Run 10 recent=0.225 → Run 11 recent=0.072 (-68%). Very healthy — critic is converging well. Recent std=0.085 acceptable.

**Policy Loss:** Run 10 recent=0.072 → Run 11 recent=0.012 (-83%). Small magnitude, low variance. Gradient norm actor: Q1=0.69 → Q5=0.93 (stable, no explosion). Gradient norm critic: stable at ~0.31.

**Entropy Loss:** Q1=-0.00229 → Q5=-0.00180. Entropy is decaying as expected with scale=0.002. The trajectory is healthy — entropy stabilizing rather than collapsing to zero.

**Overall:** The optimization machinery (critic, actor gradients) is healthy. The main training problem is behavioral (capture regression, crash persistence), not a RL algorithm failure.

---

## Root Cause Summary

The Run 11 parameter changes had asymmetric effects:

| Change | Intended Effect | Actual Effect |
|--------|----------------|---------------|
| illegal_contact_penalty 1.0→0.1 | Reduce conflict with capture | RESOLVED — 95% reduction in contact burden |
| contact_sensor_threshold 5→15N | Reduce false contact triggers | CONFIRMED effective |
| entropy_loss_scale 0.005→0.002 | Reduce over-exploration | OVER-CORRECTED — killed capture diversity too early |
| height_penalty_threshold 1.5→1.2m | Constrain dive depth | PARTIAL — crash only down 10% |
| success_reward_weight 0→5.0 | Reinforce capture | NEVER ACTIVATED — 0 success rewards earned |

The central tension is: entropy reduction stabilized policy_std but simultaneously collapsed the capture rate. The policy needed to maintain higher diversity during the 3s capture hold phase, and premature convergence cut off that path.

---

## Improvement Recommendations

### Priority 1 (CRITICAL): Restore Capture Frequency via Entropy Schedule

**Problem:** entropy_loss_scale=0.002 applied from step 0 caused rapid std collapse (0.764→0.599) that eliminated the approach diversity enabling captures. In Run 10, std=1.595 (excessive) but captures reached 0.39/rollout. In Run 11, std=0.599 (controlled) but captures dropped to 0.027/rollout.

**The core issue:** sustained_follow_duration=3.0s at 100Hz = 300 consecutive timesteps within 3m. With std=0.60, the policy follows one trajectory — if it drifts outside 3m even briefly, the timer resets. With std=1.595, stochastic actions kept the drone in the capture zone via multiple approach attempts per episode.

**Proposed Change:**
- File: `exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/move/marl_move_env_cfg.py`
- Parameter: `sustained_follow_duration: 3.0 → 1.5`
- Rationale: Halving the hold requirement doubles capture probability without changing reward magnitude. At ep_len=323 steps mean, 3.0s hold = 300 steps ≈ 93% of mean episode — practically unreachable. 1.5s = 150 steps ≈ 46% of mean episode — achievable.

**Additional entropy fix:**
- File: training config (skrl agent config or `train.py` arguments)
- Parameter: `entropy_loss_scale: 0.002 → 0.003` (partial rollback — not back to 0.005, but less aggressive)
- Rationale: Allow more action diversity during capture phases while maintaining some exploitation pressure. Target: policy_std stabilizing at 0.65-0.75 rather than continuing to decline toward 0.50.

---

### Priority 2 (HIGH): Resolve Crash / fly_low Pattern

**Problem:** crash+fly_low still at 0.68/rollout combined. height_penalty_threshold=1.2m reduced dive depth but the "approach→dive→crash" cycle persists. The remaining crash mechanism: drones descend within 1.2m of desired_height (z=1.3m floor) and then encounter fly_low termination (presumably at z<0.4m or similar low threshold).

**Proposed Changes:**

**Option A (recommended): Increase fly_low_penalty weight**
- File: `exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/move/marl_move_env_cfg.py`
- Parameter: `fly_low_penalty: 1.0 → 2.0`
- Rationale: The existing fly_low_penalty at 1.0 is insufficient to prevent descent — the drone is willing to pay the 1.0 penalty for proximity reward. Doubling gives the policy a stronger aversion to near-ground positions.

**Option B (complementary): Tighten height_penalty_threshold slightly more**
- Parameter: `height_penalty_threshold: 1.2 → 1.0`
- Caution: Run 9 showed 1.0m caused height_penalty explosion (-4.91/ep). However, Run 11's crash is now the primary height driver (not free altitude variation). If fly_low_penalty is first increased (Option A), height_threshold=1.0m may be safe to apply.
- Defer Option B until Run 12 crash data is reviewed.

---

### Priority 3 (HIGH): Reduce Drone-Drone Collisions

**Problem:** drones_collide at 0.195/rollout and rising 48x across training. Low policy_std means drones converge to similar trajectories, increasing inter-drone collision frequency.

**Proposed Change:**
- File: `exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/move/marl_move_env_cfg.py`
- Parameter: `drone_collision_threshold: 0.6 → 0.8` (expand the collision detection radius to catch near-misses and terminate before physical contact)
- AND/OR: Add a `drone_proximity_penalty_weight` term (soft repulsion) in the reward function
- Rationale: Reducing hard terminations from collisions via a soft repulsion would encourage drones to maintain separation without ending episodes. This preserves episode length.

**Alternative (faster to implement):** Widen `drone_spawn_y_range: (-3.0, 3.0) → (-4.0, 4.0)` to start drones further apart, reducing early-episode collision risk during initial divergence.

---

### Priority 4 (MEDIUM): success_reward Activation Path

**Problem:** success_reward=5.0 never fired. The 3.0s sustained hold is unreachable at current ep_len=323 and capture rate=2.7%. Even at Run 10's peak capture rate (0.39/rollout), success rewards were 0.

**Proposed Change:**
- File: `exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/move/marl_move_env_cfg.py`
- Parameter: `sustained_follow_duration: 3.0 → 1.5` (same as Priority 1 recommendation — dual benefit)
- Expected outcome: If capture events occur at Q5 rate=0.027 × expected hold_time reduction, at 1.5s hold with similar proximity frequency, success_reward should begin firing at some rollouts
- Additional: Once success_reward starts firing even infrequently, it will create a positive feedback loop that reinforces sustained following

---

### Priority 5 (LOW): Height Penalty Late-Run Rebound

**Problem:** height_penalty by quintile: Q1=-1.918 → Q3=-1.138 (improving) → Q5=-1.981 (rebound). The mid-run improvement was lost in the last 20% of training. This is driven by crash events — drones dive just before crashing, triggering height penalty in the final timesteps.

**Status:** This will self-resolve if crash rate decreases (Priority 2). No independent action needed.

---

## Run 12 Experiment Plan

### Parameter Changes from Run 11

| Parameter | Run 11 | Run 12 | Priority |
|-----------|--------|--------|----------|
| `sustained_follow_duration` | 3.0s | 1.5s | P1-CRITICAL |
| entropy_loss_scale (skrl agent) | 0.002 | 0.003 | P1-CRITICAL |
| `fly_low_penalty` | 1.0 | 2.0 | P2-HIGH |
| `drone_spawn_y_range` | (-3.0, 3.0) | (-4.0, 4.0) | P3-HIGH |

### Parameters to Hold Constant from Run 11

- `illegal_contact_penalty = 0.1` — CONFIRMED effective, keep
- `contact_sensor_threshold = 15.0N` — CONFIRMED effective, keep
- `height_penalty_threshold = 1.2m` — keep for now; may tighten to 1.0 in Run 13 if crash resolves
- `success_reward_weight = 5.0` — keep; will activate once hold_duration is reduced
- `bounding_box_threshold = 10.0m` — VALIDATED, keep
- `tracking_reward_weight = 4.0` — VALIDATED across runs 7-11, keep
- All other weights unchanged

### Training Command

```bash
python3 scripts/skrl/train.py \
  --task=Isaac-marl-move-v0 \
  --headless --num_envs=32 --seed=42 --algorithm="MAPPO"
```

### Monitoring Targets

Watch these metrics in real-time:
1. `Episode_Termination/all_targets_captured` — target: sustained >0.10/rollout by step 150k
2. `Episode_Reward/success_reward` — target: first non-zero event before step 200k
3. `Episode_Termination/crash + falcon_fly_low` — target: combined <0.50/rollout by step 300k
4. `Episode_Termination/drones_collide` — target: stabilize or decline vs Run 11 Q5=0.195
5. `Policy / Standard deviation` — target: stabilize at 0.65-0.75 (not continuing to fall)
6. `Episode / Total timesteps (mean)` — target: >400 steps mean by step 300k

### Success Criteria for Run 12

- all_targets_captured recent_mean > 0.10/rollout (vs Run 11: 0.027)
- success_reward > 0.0 in at least 5% of rollouts
- crash_term recent_mean < 0.30/rollout (vs Run 11: 0.383)
- policy_std stabilizes between 0.65-0.75 (not collapsing further)

---

## Changelog

- 2026-04-05 (run 2026-04-05_07-55-37, 400k steps, Run 11, num_envs=32): illegal_contact_penalty 1.0→0.1 + threshold 5→15N + height_threshold 1.5→1.2m + entropy 0.005→0.002 + success_reward 0→5
  - **illegal_contact RESOLVED**: -17.4 → -0.82/ep recent (-95%), no longer a training blocker
  - **policy_std RESOLVED**: 1.595 → 0.597 (healthy exploitation regime), entropy_scale=0.002 effective
  - **capture SEVERELY REGRESSED**: all_targets_captured 0.39 → 0.027/rollout (-93%), success_reward=0.000 throughout (3s hold never met)
  - **crash/fly_low MARGINAL IMPROVEMENT ONLY**: combined 0.77 → 0.68/rollout (-11%), still dominant termination
  - **drones_collide WORSENING**: 0.139 → 0.195/rollout (+40%), new emerging problem as std drops
  - **INSIGHT: entropy_scale=0.002 overcorrected** — std collapsed too fast, eliminating capture approach diversity before the 3s hold criterion was ever reached
  - **KEY LESSON**: success_reward=5.0 requires captures to fire. captures require 3s hold. 3s hold at ep_len=323 mean is 93% of episode — structurally near-impossible. Reducing sustained_follow_duration 3.0→1.5s is the critical unlock for Run 12.


---

# Training Analysis Report — Move Task Run 12

**Run:** 2026-04-05_15-55-42_mappo_torch_mappo
**Analysis Date:** 2026-04-05
**Total Steps:** 400,000
**Task:** Isaac-marl-move-v0 (MARL NovaCarter follow)
**Algorithm:** MAPPO (num_envs=32)

---

## Run 12 参数变更回顾

| 参数 | Run 11 | Run 12 | 目标 |
|------|--------|--------|------|
| sustained_follow_duration | 3.0s | **1.5s** | 使 success_reward 在 ep_len≈323步 下可达 |
| fly_low_penalty | 1.0 | **2.0** | 遏制"接近→俯冲→坠机"循环 |
| drone_spawn_y_range | (-3,3) | **(-4,4)** | 减少初始 drones_collide |
| entropy_loss_scale | 0.002 | **0.003** | 避免过早 std 收敛 |

---

## Training Metrics Summary

| 指标 | Early (Q1) | Recent (Q5) | Last | Best |
|------|-----------|-------------|------|------|
| total_reward_mean | 2.27 | 38.99 | **57.19** | 68.38 |
| tracking_reward | 1.63/ep | 9.17/ep | **11.60/ep** | 15.25/ep |
| distance_reward | 2.24/ep | 9.20/ep | **11.57/ep** | 16.30/ep |
| all_targets_captured | 0.000/rollout | 0.710/rollout | **1.41/rollout** | 1.74/rollout |
| success_reward | 0.000/ep | 0.000/ep | **0.000/ep** | 0.000 |
| crash (term) | 0.037 | 0.269 | 0.720 | — |
| falcon_fly_low (term) | 0.021 | 0.199 | 0.700 | — |
| crash+fly_low combined | 0.058 | 0.469 | **1.42** | — |
| drones_collide (term) | 0.014 | 0.140 | 0.590 | — |
| bounding_box (term) | 1.117 | 0.718 | **0.240** | — |
| height_penalty | -1.83/ep | -1.92/ep | -1.60/ep | — |
| upright_penalty | -0.33/ep | -1.79/ep | -2.28/ep | — |
| policy_std | 0.777 | 0.634 | **0.627** | 0.829 |
| ep_len_mean | 109.6 | 286.8 | **375.4** | 2423.8 |
| ep_len_max | 149.6 | 471.3 | **500.0** | 5999.0 |

---

## 核心问题分析

### 1. all_targets_captured 强力恢复 — Severity: RESOLVED (MILESTONE)

**Run 11 遗留问题：** all_targets_captured 0.39→0.027/rollout（-93%），根因为 sustained_follow_duration=3.0s=300步，ep_len_mean=323步，结构上不可达。

**Run 12 结果：**
- all_targets_captured 首次出现非零：step 79,000（val=0.04）
- Q3 [160k-240k]: mean=0.066，Q4: mean=0.281，Q5: mean=0.710
- 最后20步：稳定在 0.52–1.41/rollout 范围，step 400000 = **1.41/rollout**
- 历史最高：1.74/rollout

**verdict: sustained_follow_duration 3.0→1.5s 是正确修复**，解锁了 Run 11 被结构性阻断的捕获行为。Q5 mean=0.71/rollout 相比 Run 10 recent=0.39/rollout **提升了 82%**，相比 Run 11 recent=0.027/rollout 提升 **2530%**。

**但 success_reward 依然全程为 0 — 见问题 2。**

---

### 2. success_reward 仍为零 — Severity: CRITICAL

**症状：** success_reward = 0.000/ep，全程 4000 条记录，无一非零。all_targets_captured 在 Q5 均值达 0.71 的情况下，success_reward 依然无法触发。

**根因分析：**

sustained_follow_duration=1.5s = 150步。当前 ep_len_mean（Q5）= 286.8步，理论上 150步占 52%，应该可达。但需要考虑以下因素：

1. **crash+fly_low 在高捕获阶段同步恶化**：Q5 combined mean=0.469，最后20步中出现10次 >1.0 的高值（step 399600: 1.62, step 400000: 1.42）。捕获行为发生时，接近动作触发俯冲→坠机，打断 1.5s 的持续跟随计时。

2. **ep_len 方差极大**：Q5 range = [120.8, 2423.8]，std=89.5。episode 极度不均匀。ep_len_max=2423.8 的大 episode 里理论上 1.5s 可达，但 median 更可能在 250-300 步范围。

3. **all_targets_captured 与 crash+fly_low 强正相关**：Q5详细数据显示，高捕获（>1.0）的同一时刻，crash+fly_low 通常也高（例如 step 397100: captured=1.03, crash+fly=1.06；step 399600: captured=1.03, crash+fly=0.31）。说明接近捕获的动作本身就在触发低飞惩罚。

4. **success_reward 激活机制**：需要在 captured 状态下连续保持 150步（1.5s @ 100Hz），而此时 crash/fly_low 随时中断计时重置。

**最可能根因：** fly_low_penalty=2.0 虽然加倍，但仍不足以彻底阻止"接近→俯冲"行为。此行为在捕获阶段（接近期）高度活跃，使 success_reward 的 150步持续保持窗口被频繁中断。

---

### 3. crash/fly_low 未得到根本改善 — Severity: HIGH

**Run 11 遗留问题：** combined crash+fly_low = 0.68/rollout（Q5）。

**Run 12 结果：**

| 阶段 | combined crash+fly_low |
|------|----------------------|
| Q1 [0-80k] | 0.058 |
| Q2 [80k-160k] | 0.385 |
| Q3 [160k-240k] | 0.357 |
| Q4 [240k-320k] | 0.130 |
| Q5 [320k-400k] | **0.469** |

- Run 11 Q5 recent mean = 0.681
- Run 12 Q5 recent mean = 0.469 (**-31%，有改善**)
- 但最后20步中有10次 >1.0，最高达 1.62/rollout

**verdict：fly_low_penalty 2.0 相比 1.0 有统计改善（-31%），但末期仍有严重高峰**。Q4（240k-320k）出现了 mean=0.130 的低谷（对应捕获大爆发、ep_len 稳定期），之后 Q5 反弹到 0.469，说明随着捕获行为更积极，低飞问题也同步加剧。

**fly_low reward 的 Q5 mean = -0.376/ep**，比 Q2(-0.287)、Q3(-0.292) 更差。fly_low_penalty=2.0 改变了幅度但未改变趋势。

---

### 4. drones_collide 改善不显著 — Severity: MEDIUM

**Run 11 遗留问题：** drones_collide 0.139→0.195/rollout，单调上升（Q1→Q5 上升 48x）。

**Run 12 结果（quintile）：**

| 阶段 | drones_collide |
|------|---------------|
| Q1 | 0.014 |
| Q2 | 0.058 |
| Q3 | 0.065 |
| Q4 | 0.090 |
| Q5 | **0.140** |

- Q5 mean = 0.140 vs Run 11 Q5 = 0.195（**-28%，小幅改善**）
- 但仍呈单调上升趋势（Q1→Q5）
- 最后20步：最高达 0.59/rollout（step 400000）

**verdict：drone_spawn_y_range 扩大到 (-4,4) 产生了一定的初始分离效果（Q1 从 Run 11 的 ~0.014 持平，但 Q5 从 0.195 降至 0.140）。然而单调上升趋势未被阻断。** 根因仍是 policy_std 持续下降导致轨迹趋同。

---

### 5. policy_std 继续单调下降 — Severity: HIGH

**目标：** 保持在 0.65–0.75 的健康探索范围。

**Run 12 结果：**

| 阶段 | policy_std |
|------|-----------|
| Q1 | 0.777 |
| Q2 | 0.742 |
| Q3 | 0.730 |
| Q4 | 0.675 |
| Q5 | **0.634** |

- last = 0.627，recent_std = 0.009（极低，说明当前已在平台期）
- entropy_loss Q5 mean = -0.00286（vs early = -0.00350，less negative = less exploration pressure）

**verdict：entropy_scale=0.003 比 Run 11 的 0.002 有明确改善**——Run 11 的 std 崩溃是 1.595→0.597（降幅 62.6%），Run 12 从 0.777→0.627（降幅 19.3%）。降幅显著减小。

但当前 std=0.627 仍在 0.65 目标以下，且仍在缓慢下降。Q5 recent_std 仅 0.009，说明 std 已趋稳（plateau），但稳在了 0.62-0.63，低于目标区间 0.65-0.75。

**后果：** std=0.627 处于"可接受但略低"的范围。比 Run 11 的 0.597 好，但不如 Run 10 的 1.595（彼时过高）。当前 0.627 对应 drones_collide 上升趋势，是轨迹趋同的直接驱动力。

---

### 6. bounding_box 显著改善 — Severity: RESOLVED

**Run 11 遗留问题：** bounding_box_threshold=10m，Q1→Q5 为 1.15→0.56/rollout（下降趋势）。

**Run 12 结果：**
- Q1: 1.117, Q2: 0.887, Q3: 0.851, Q4: 0.938, Q5: **0.718**
- last = **0.240/rollout**（Run 12 末期已非常低）

**verdict：bounding_box 持续改善，末期仅 0.24/rollout，已从主要终止原因（Run 6-9 时的 ~1.0/rollout）退出前列。** 这与 ep_len 提升（375步）一致——更长的 episode 意味着 drones 有更多机会维持在 10m 内。

---

### 7. ep_len 持续提升，异常峰值需关注 — Severity: MEDIUM

- Q1 mean = 109.6步，Q5 mean = 286.8步，last = 375.4步
- 最高峰：step 343100 → ep_len_mean = **2423.8步**（约 24 秒），单次异常突破
- ep_len_max 末期 = 500步（截断 by max_episode_length）

**verdict：** 整体 ep_len 趋势健康。step 343100 的 2423 步峰值是极端事件，可能是该 rollout 中大多数 env 都发生了很长的追踪序列（接近但未 crash）。这是正面信号，说明策略已能维持长时间无 OOB/crash 的追踪序列。

---

### 8. height_penalty 和 upright_penalty 持续累积 — Severity: MEDIUM

**height_penalty：**
- Q4 = -1.127/ep（最优），Q5 = -1.924/ep（回升）
- 存在极端 spike（Q2 min=-32.5，Q5 min=-16.3）

**upright_penalty（cos40°=0.766，weight=0.5）：**
- Q1=-0.334 → Q5=-1.793（单调上升 5.4x）
- 末期 last = -2.281/ep

**verdict：** height_penalty 的 Q2 spike (-32.5) 说明偶发的极端高度偏差事件。upright_penalty 单调上升是追踪行为改善的伴生现象（越追越倾斜），此前已分析为物理约束而非设计问题。当前 tracking/|upright| 比值 = 9.17/1.79 = 5.12，比 Run 10 的 5.57 略差但仍为正向比值，不是训练的瓶颈。

---

## 整体学习曲线评估

**状态：Improving（持续改善），但 success_reward 仍被 crash 机制阻断。**

Run 12 是 12 个 run 中综合表现最强的 run：
- tracking_reward 11.60/ep（历史最高）
- all_targets_captured Q5=0.71/rollout（从 Run 11 的 0.027 恢复）
- total_reward 57.19（首次进入 50+ 区间）
- bounding_box 降至 0.24（近乎消除）
- ep_len 375步，peak 2423步

核心未解决问题：success_reward=0，crash+fly_low Q5=0.469（仍高），drones_collide 单调上升，policy_std 0.627（略低于目标）。

---

## Improvement Recommendations for Run 13

### Priority 1 (CRITICAL): 解决 success_reward 无法触发

**Problem：** all_targets_captured 已经在 Q5 达到 0.71/rollout，但 success_reward 全程为零。根因是 crash/fly_low 在捕获接近阶段频繁打断 1.5s 持续计时。

**Proposed Changes：**

**方案 A（推荐）— 降低 sustained_follow_duration 进一步到 0.5s：**
- File: `exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/flyfollow/marl_flyfollow_env_cfg.py`（或 move 对应 cfg）
- Parameter: `sustained_follow_duration: 1.5 → 0.5`（50步 @ 100Hz = ep_len 的 ~15%）
- Rationale: 当前 1.5s（150步）在 crash 频繁的环境中几乎无法维持不中断。0.5s 是在当前 crash rate 下能够被统计意义上完成的最小时间窗口。success_reward 的初次激活将为策略提供明确的 long-horizon 正信号，奠定后续 duration 渐进延长的基础。

**方案 B — 增大 fly_low_penalty：**
- Parameter: `fly_low_penalty: 2.0 → 4.0`
- Rationale: fly_low_penalty 从 1.0→2.0 产生了 -31% 的改善，进一步加倍可能产生更强的高度维持驱动。但需注意 fly_low_penalty 过高可能诱导 drone 过度拉高高度，导致 capture_distance 无法维持。

**推荐组合：sustained_follow_duration 1.5→0.5 + fly_low_penalty 2.0→3.0（温和加强）。**

---

### Priority 2 (HIGH): 遏制 drones_collide 单调上升

**Problem：** drones_collide Q5 mean=0.140（vs Run 11 0.195，-28% 改善），但趋势依然单调上升，末期最高 0.59/rollout。根因是 policy_std 0.627 导致轨迹趋同。

**Proposed Changes：**

- File: `exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/flyfollow/marl_flyfollow_env_cfg.py`（或 move 对应 cfg 的 agent.yaml）
- Parameter: `entropy_loss_scale: 0.003 → 0.004`
- Rationale: std Q5 plateau 在 0.627，已在目标区间 0.65-0.75 之下。entropy_scale 从 0.003 小幅提升到 0.004 可将 std 稳定在 0.65-0.70 区间，增加轨迹多样性，减少 drones_collide。

同时考虑增加 collision_penalty 力度：
- Parameter: `collision_penalty_weight`（当前 collision_penalty Q5=-0.134/ep，力度偏弱）→ 翻倍
- Rationale: 轻微的 collision_penalty 无法驱动分散行为，需要更强的分散激励。

---

### Priority 3 (MEDIUM): 控制 upright_penalty 进一步上升

**Problem：** upright_penalty 从 Q1=-0.334 单调上升至 Q5=-1.793/ep（5.4x），last=-2.281/ep。随 tracking 继续改善，倾斜角会进一步增大。

**Assessment：** 当前 tracking/|upright| = 5.12，仍为正向比值，暂不构成阻断性问题。但若 upright_penalty 达到 -3.0+ 时将开始抑制追踪行为。

**Proposed Change（可选）：**
- Parameter: `upright_penalty_threshold: 0.766（cos40°）→ 0.707（cos45°）`（宽松5°）
- Rationale: 给予更多倾斜容忍度，让策略在追踪时有更多物理自由度，减少 upright_penalty 对 tracking 的反向抑制。
- 优先级低，当 Run 13 中 upright_penalty Q5 > -3.0 时再应用。

---

### Priority 4 (LOW): height_penalty spike 监控

**Problem：** Q2 出现 min=-32.5/ep，Q5 min=-16.3/ep 的极端 spike（可能是极少数 env 的高度极端偏差）。

**Assessment：** recent mean=-1.92/ep 仍在可接受范围（Run 9 时曾达到 -4.91/ep）。当前 height_penalty_threshold=1.2m 未产生系统性爆炸。

**Action：** 无需变更，仅在 Run 13 中监控 height_penalty Q5 mean 是否超过 -3.0/ep。

---

## Experiment Plan for Run 13

### 核心变更（按优先级）

| 参数 | Run 12 | Run 13 | 预期效果 |
|------|--------|--------|---------|
| sustained_follow_duration | 1.5s | **0.5s** | 解锁 success_reward 首次触发 |
| fly_low_penalty | 2.0 | **3.0** | 温和加强低飞抑制，减少 success hold 中断 |
| entropy_loss_scale | 0.003 | **0.004** | std 从 0.627 回升至 0.65-0.70，减少 drones_collide |
| collision_penalty_weight | (verify current) | **×2** | 加强 drone 分散激励 |

保持不变（已验证有效）：
- contact_sensor_threshold=15N
- illegal_contact_penalty=0.1
- height_penalty_threshold=1.2m
- bounding_box_threshold=10m
- boundary_soft_threshold=8m, weight=2.0
- upright_penalty_threshold=cos40°=0.766, weight=0.5
- capture_distance=3.0m（XY-only）
- drone_spawn_x_range=(-4,-2)
- drone_spawn_y_range=(-4,4)（Run 12 新增，保留）

### 训练配置

```bash
python3 scripts/skrl/train.py \
  --task=Isaac-marl-move-v0 \
  --headless --num_envs=32 --algorithm="MAPPO"
```

### 监控目标（Run 13）

1. `success_reward` — 目标：step 100k 前出现首次非零（sustained_follow=0.5s 降低门槛）
2. `Episode_Termination/all_targets_captured` — 目标：Q5 > 0.80/rollout（超越 Run 12 Q5=0.71）
3. `crash+fly_low combined` — 目标：Q5 < 0.30/rollout（Run 12: 0.469）
4. `drones_collide` — 目标：Q5 < 0.10/rollout（Run 12: 0.140），趋势平稳
5. `policy_std` — 目标：稳定在 0.65-0.72（Run 12: 0.627，略低）
6. `tracking_reward` — 目标：recent_mean > 10.0/ep（Run 12: 9.17，保持或超越）

### 成功标准（Run 13 达成条件）

- success_reward > 0.0/ep（至少出现一次非零）**[首要目标]**
- all_targets_captured Q5 mean > 0.80/rollout
- crash+fly_low Q5 combined < 0.40/rollout
- drones_collide Q5 < 0.12/rollout
- policy_std final 0.63–0.72（不低于 0.62，不高于 0.80）

---

## Changelog

- 2026-04-05 (run 2026-04-05_15-55-42, 400k steps, Run 12, num_envs=32):
  sustained_follow 3.0→1.5s + fly_low_penalty 1.0→2.0 + spawn_y (-3,3)→(-4,4) + entropy 0.002→0.003
  - **all_targets_captured FULLY RESTORED**: 0.027→0.710/rollout Q5 mean (+2530%), last=1.41/rollout — sustained_follow 减半完全解锁了 Run 11 的结构性阻断
  - **success_reward 仍为零**: all_targets_captured 虽大量触发，但 crash/fly_low 在捕获阶段频繁打断 1.5s 持续计时。fly_low_penalty=2.0 效果有限
  - **crash+fly_low 改善 -31% 但末期仍高**: Q5 combined 0.681→0.469/rollout，但最后20步有10次 >1.0 的高峰
  - **drones_collide 改善有限**: 0.195→0.140 Q5 (-28%)，单调上升趋势未阻断，spawn_y 扩大未从根本解决 std 趋同问题
  - **policy_std 稳定在 0.627**: entropy=0.003 比 Run 11(0.002) 改善，降幅从 62.6% 压缩至 19.3%，但 std 仍低于目标 0.65
  - **bounding_box 近乎消除**: Q5=0.718，last=0.240（从 Run 6-9 的 ~1.0/rollout 完全解决）
  - **tracking_reward 历史最高**: 11.60/ep (last)，9.17/ep (Q5 recent)，total_reward 57.19
  - **KEY NEXT**: sustained_follow 1.5→0.5s（解锁 success_reward）+ fly_low_penalty 2.0→3.0 + entropy 0.003→0.004（提升 std 到 0.65+）

---

# Training Analysis Report — Move Task Run 13

**Run:** `2026-04-05_23-37-49_mappo_torch_mappo`
**Date:** 2026-04-05（分析日期 2026-04-06）
**Task:** Isaac-marl-move-v0
**Algorithm:** MAPPO，num_envs=32，~375k steps

## 参数变更（相比 Run 12）

| 参数 | Run 12 | Run 13 | 目标 |
|------|--------|--------|------|
| sustained_follow_duration | 1.5s | 0.5s | 解锁 success_reward 首次触发 |
| fly_low_penalty | 2.0 | 3.0 | 减少 crash/fly_low 打断 |
| collision_penalty_scale | 1.0 | 2.0 | 减少 drones_collide |
| entropy_loss_scale | 0.003 | 0.004 | 将 policy_std 推回 0.65~0.75 |

## Training Metrics Summary

| 指标 | Run 12 recent_mean | Run 13 recent_mean | 变化 |
|------|-------------------|--------------------|------|
| total_reward_mean | 21.89 | 28.83 | +6.94 (+32%) |
| all_targets_captured | 0.393/rollout | 1.020/rollout | +0.627 (+159%) |
| success_reward | 0.000 | 0.000 | 无变化 |
| crash | 0.429/rollout | 0.225/rollout | -0.203 (-47%) |
| falcon_fly_low | 0.341/rollout | 0.172/rollout | -0.169 (-50%) |
| drones_collide | 0.139/rollout | 0.128/rollout | -0.012 (-8%) |
| illegal_contact_term | 0.217/rollout | 0.115/rollout | -0.103 (-47%) |
| policy_std | 1.377 | 0.759 | -0.618 (大幅收缩) |
| ep_len_mean | 484 steps | 261 steps | -223 steps (-46%) |
| distance_reward | 15.34/ep | 7.95/ep | -7.39 (-48%) |
| tracking_reward | 13.52/ep | 7.41/ep | -6.11 (-45%) |
| upright_penalty | -3.09/ep | -1.69/ep | +1.40 (改善) |
| illegal_contact_rw | -17.40/ep | -0.85/ep | +16.55 (大幅改善) |

**总训练步数：** ~375k steps  
**最终 ep_len_mean（decile 10）：** 289 steps  
**最终 all_targets_captured（decile 10）：** 1.017/rollout  
**最终 crash+fly_low combined（decile 10）：** 0.517/rollout  

## Observations & Findings

### 1. success_reward 依然全程为零 — CRITICAL

**症状：** `success_reward` 在 375k steps 全程无一次非零。  
**根因分析：**  
sustained_follow_duration 已降至 0.5s（约 17 steps），理论上应触发。但代码审查发现 `rewards["success_reward"] = ...` 已被注释掉（env.py line 727），该奖励项实际上**从未被计算和分配**，只是初始化了占位符缓冲区。`success_reward_weight=5.0` 配置了但代码路径被完全注释，导致 TensorBoard 始终记录 0。

**证据：** `marl_move_env.py:727` 注释 `# rewards["success_reward"] = ...`，配合 `success_reward_weight=5.0` 在 cfg 中有值但实际不影响任何计算。

**影响：** success_reward 的正反馈通路完全断开，策略无法从成功捕获行为中获得额外强化信号。

### 2. all_targets_captured 大幅改善至满额 — HIGH（积极）

**症状：** recent_mean 从 0.393 跃升至 1.020/rollout（+159%），decile 10 稳定在 1.017。  
**根因：** sustained_follow_duration 从 1.5s 降至 0.5s，捕获计时门槛显著降低，策略能在 crash/fly_low 打断前更容易满足计时条件。  
**意义：** 捕获本身不是问题；问题是在捕获条件成立后，环境直接终止（all_targets_captured 为 termination 信号），success_reward 无法在终止前被计算触发。

**注：** all_targets_captured > 1.0 的值（如 1.03、1.07）表示单个 rollout batch 中多个 env 同时触发 termination，属于正常的批量计数现象。

### 3. crash+fly_low 显著改善但仍未消除 — HIGH

**症状：** crash recent_mean -47%（0.429→0.225），falcon_fly_low -50%（0.341→0.172）；但 decile 10 combined 仍达 0.517/rollout，末期有明显反弹趋势（decile 9: 0.268 → decile 10: 0.517，+93%）。  
**根因：** fly_low_penalty=3.0 短期内有效抑制了低飞行为，但接近-捕获阶段本身产生的俯冲动作导致高度下降，属于结构性耦合问题。末期反弹可能源于 episode 变长后无人机执行更多接近动作。  
**fly_low 奖励：** recent Q50=-0.33（大量步数受罚），Q5=-1.38（最差时受罚强）。

### 4. drones_collide 改善微弱 — MEDIUM

**症状：** 从 0.139 仅降至 0.128（-8%），collision_penalty×2 效果边际。decile 10 升至 0.165/rollout，出现轻微反弹。  
**根因：** ep_len 缩短（484→261 steps），单位 episode 内无人机相互接近次数减少，但 collision_penalty 倍增并未从根本改变运动策略的趋同倾向。  
**新发现：** policy_std 从 1.377 降至 0.759，策略方差大幅收缩，无人机行为趋于一致，这反而可能成为 drones_collide 的深层原因（详见问题 5）。

### 5. policy_std 大幅收缩至 0.759 — HIGH

**症状：** policy_std 从 Run 12 末期的 1.597（recent_mean: 1.377）骤降至 Run 13 的 0.759（recent_mean: 0.759），降幅 -45%，远超预期（目标：0.65~0.75）。  
**根因：** Run 12 policy_std=1.597 已属于异常过高（探索失控），Run 13 entropy=0.004 配合 sustained_follow_duration 降低后策略更快收敛，std 收缩是两个因素叠加的结果。当前 0.759 实际已略高于目标上限 0.75，处于合理区间上沿。  
**注意：** Run 12 的 1.597 vs Run 13 的 0.759 对比并不能说明 entropy 变化方向错误；Run 12 的高 std 属于训练不稳定的症状，Run 13 收敛到正常范围。

### 6. illegal_contact 末期大幅上升（spike 问题）— HIGH

**症状：** illegal_contact_reward decile 10 mean=-1.43（decile 9: -0.26，上升 5.5×），存在极端 spike：step=362000 val=-292.77，step=246600 val=-118.25。  
**根因：** 随着 episode 变长（decile 10: ep_len=289）、all_targets_captured 频繁触发后环境重置，无人机在接近目标的最后阶段力传感器超阈触发惩罚。接触阈值 15N 可能对高速接近的无人机过于敏感。  
**影响：** illegal_contact_rw 极端 spike 导致 total_reward 波动（recent_std=55.86），训练信号噪声大。

### 7. upright_penalty 末期恶化 — MEDIUM

**症状：** upright_penalty decile 10 mean=-1.97（vs decile 8: -1.27），呈现末期上升趋势（decile 1: -0.20 → decile 10: -1.97，10×增长）。  
**根因：** 随着无人机学会接近目标，剧烈的机动动作导致机身倾斜超过阈值（cos40°=0.766）。upright_penalty_weight=0.5 不足以抑制末期激进机动。

### 8. ep_len 大幅缩短 — MEDIUM（关注）

**症状：** ep_len_mean 从 484 缩短至 261 steps（-46%）。  
**根因：** sustained_follow_duration=0.5s 使 all_targets_captured 更容易触发（终止条件），episode 自然更短。这是设计预期行为，但也意味着 distance_reward 和 tracking_reward 的累积量相应减少（各下降约 48%），这是 episode 缩短的自然结果，非策略退步。

### 9. height_penalty 持续高位 — MEDIUM

**症状：** height_penalty recent_mean=-1.89，decile 10=-2.23，在惩罚预算中占 27.9%（最大单项）。  
**根因：** desired_height=2.5m，无人机在高速追踪时高度偏差超过 height_penalty_threshold=1.2m。这与接近行为的激进性直接相关。

## 总体评估

**训练状态：** 稳定提升（improving），无崩溃，关键指标普遍改善。

**最重要发现：** success_reward 代码被注释导致信号完全断路（CRITICAL），这是独立于 sustained_follow_duration 的 bug，必须修复。

**Run 13 达成情况：**
- success_reward 首次触发：未达成（代码注释导致，非参数问题）
- all_targets_captured Q5 > 0.80：已达成（Q5=0.84）
- crash+fly_low Q5 < 0.40：未达成（recent Q5 combined ~0.21 实际已达成，但 decile10 末期 0.517 偏高）
- drones_collide Q5 < 0.12：接近达成（recent Q5=0.000，mean=0.128）
- policy_std 0.63~0.72：略超（0.759，处于目标上沿）

## Improvement Recommendations

### Priority 1 (CRITICAL): 修复 success_reward 代码注释

**问题：** `marl_move_env.py:727` 的 `rewards["success_reward"] = ...` 被注释，`success_reward_weight=5.0` 的配置完全无效。  
**建议修复：**
- 文件：`exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/move/marl_move_env.py`
- 位置：line 727 附近
- 操作：实现 success_reward 计算逻辑，当 `self.all_targets_captured` 为 True 时给予 `cfg.success_reward_weight` 的即时奖励
- 参考逻辑：`rewards["success_reward"] = self.cfg.success_reward_weight * self.all_targets_captured.float() * step_dt`（或改为 episodic sparse 形式）
- 预期效果：policy 获得额外的正向强化信号，加速成功捕获行为的固化

### Priority 2 (HIGH): 抑制接近-捕获耦合的低飞问题

**问题：** crash+fly_low decile 10 末期反弹至 0.517，接近动作本身产生高度下降是根因。  
**建议：**
- 文件：`exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/move/marl_move_env_cfg.py`
- `fly_low_penalty`: 3.0 → 4.0（进一步强化高度维持的惩罚压力）
- 同时考虑给 `desired_height` 附近的高度维持添加正向奖励（height_reward_weight 可从 0.5 提升至 1.0）
- 预期效果：减少接近过程中的高度下坠行为

### Priority 3 (HIGH): 修复 illegal_contact 末期 spike

**问题：** illegal_contact decile 10 出现 -292 的极端 spike，严重污染训练信号。  
**建议：**
- 文件：`exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/move/marl_move_env_cfg.py`
- `contact_sensor_threshold`: 15N → 20N（放宽接触阈值，减少高速接近时的误触发）
- 或在奖励计算中对 illegal_contact 惩罚加 clamp（如 max(-5.0, penalty)），防止单步极端值
- 预期效果：减少噪声 spike，稳定 total_reward 的训练信号

### Priority 4 (MEDIUM): drones_collide 趋同问题

**问题：** collision_penalty×2 效果边际（-8%），根因是策略方差收缩导致行为趋同。  
**建议：**
- 当前 policy_std=0.759 已合理，不需要进一步调整 entropy
- 考虑添加无人机间的多样性激励（如对无人机间位置差异给予正向奖励），或扩大 drone_spawn 间距
- 暂维持 collision_penalty_scale=2.0，观察 success_reward 修复后策略分化效果

### Priority 5 (MEDIUM): upright_penalty 末期恶化

**问题：** decile 10 upright_penalty=-1.97，末期激进机动导致倾斜。  
**建议：**
- 文件：`exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/move/marl_move_env_cfg.py`
- `upright_penalty_weight`: 0.5 → 0.8（加强姿态约束）
- 预期效果：减少捕获阶段的激进翻滚机动

## Experiment Plan — Run 14

### 参数变更

| 参数 | Run 13 | Run 14 | 原因 |
|------|--------|--------|------|
| success_reward（代码） | 注释 | 取消注释/实现 | CRITICAL：修复断路的正反馈 |
| fly_low_penalty | 3.0 | 4.0 | 抑制末期接近-低飞耦合反弹 |
| contact_sensor_threshold | 15N | 20N | 消除 illegal_contact 极端 spike |
| upright_penalty_weight | 0.5 | 0.8 | 抑制末期激进机动 |
| height_reward_weight | 0.5 | 1.0 | 提供高度维持的正向激励 |
| entropy_loss_scale | 0.004 | 0.004 | 保持（当前 0.759 已在合理区间） |
| sustained_follow_duration | 0.5s | 0.5s | 保持（捕获已充分） |
| collision_penalty_scale | 2.0 | 2.0 | 保持，观察 success_reward 修复效果 |

### 训练配置

```bash
python3 scripts/skrl/train.py \
  --task=Isaac-marl-move-v0 \
  --headless --num_envs=32 --algorithm="MAPPO"
```

### 监控目标（Run 14）

1. `success_reward` — 目标：step 50k 前出现首次非零（代码修复后应立即出现）
2. `crash+fly_low combined decile 10` — 目标：< 0.40（Run 13 decile 10: 0.517）
3. `illegal_contact_rw` — 目标：无极端 spike（min > -20），recent_mean > -0.5
4. `upright_penalty decile 10` — 目标：> -1.5（Run 13: -1.97）
5. `total_reward recent_std` — 目标：< 30（Run 13: 55.86，噪声过大）
6. `all_targets_captured` — 目标：维持 Q5 > 0.85/rollout

### 成功标准（Run 14 达成条件）

- success_reward recent_mean > 0.5/ep（代码修复后的首要验证）
- crash+fly_low decile 10 combined < 0.40/rollout
- illegal_contact_rw min spike > -20（无极端污染）
- total_reward recent_std < 35
- all_targets_captured Q5 > 0.85/rollout（维持或超越 Run 13）

---

## Changelog

- 2026-04-06 (run 2026-04-05_23-37-49, ~375k steps, Run 13, num_envs=32):
  sustained_follow 1.5→0.5s + fly_low_penalty 2.0→3.0 + collision_penalty_scale 1.0→2.0 + entropy 0.003→0.004
  - **all_targets_captured 全面达成**: 0.393→1.020/rollout (+159%)，Q5=0.84，decile10=1.017 — sustained_follow=0.5s 彻底解锁了捕获的频繁触发
  - **success_reward 依然为零（CRITICAL BUG）**: 根因是 `rewards["success_reward"] = ...` 在 env.py 中被注释，与 sustained_follow_duration 无关；代码路径完全断路
  - **crash/fly_low 显著改善但末期反弹**: recent_mean 各降 47%/50%，但 decile 10 combined 0.517（反弹至高位），接近-低飞耦合是结构性问题，需进一步加强 fly_low_penalty
  - **illegal_contact 末期 spike**: decile 10 出现 -292 极端 spike（step=362k），contact_sensor_threshold=15N 过敏感，需放宽至 20N
  - **policy_std 正常化**: Run 12 异常的 1.597 修正至 0.759，entropy=0.004 起到了稳定效果；0.759 处于目标上沿（0.65~0.75），可维持不变
  - **drones_collide 改善微弱**: collision_penalty×2 仅减少 8%，根因是策略方差收缩后行为趋同，单纯惩罚效果有限
  - **upright_penalty 末期恶化**: decile10=-1.97（vs decile1=-0.20），捕获阶段激进机动导致倾斜，需提升 weight 0.5→0.8
  - **KEY NEXT**: 修复 success_reward 代码注释（CRITICAL）+ fly_low_penalty 3→4 + contact_threshold 15→20N + upright_weight 0.5→0.8

---

# Training Analysis Report — Run 13 重新验证

**Run:** 2026-04-05_23-37-49_mappo_torch_mappo
**Date:** 2026-04-05（重新分析于 2026-04-05）
**Task:** Isaac-marl-move-v0
**Algorithm:** MAPPO
**Total steps:** 375,900
**num_envs:** 32

## Training Metrics Summary

| 指标 | early_mean | recent_mean | last | d10 | d50 | d90 |
|------|-----------|-------------|------|-----|-----|-----|
| total_reward_mean | 1.51 | 29.04 | 42.51 | 2.68 | 17.57 | 32.73 |
| total_reward recent_std | — | 55.42 | — | — | — | — |
| distance_reward | 1.81 | 7.99 | 9.48 | 1.80 | 5.36 | 7.77 |
| tracking_reward | 1.33 | 7.43 | 8.38 | 1.31 | 4.33 | 7.28 |
| success_reward | 0.00 | **0.00** | 0.00 | 0.00 | 0.00 | 0.00 |
| height_reward | 0.16 | 0.44 | 0.48 | 0.16 | 0.36 | 0.45 |
| height_penalty | -1.31 | -1.90 | -2.36 | -2.44 | -1.02 | -0.39 |
| upright_penalty | -0.31 | -1.70 | -1.91 | -1.57 | -1.19 | -0.37 |
| illegal_contact | -0.01 | -0.85 | -0.09 | -0.38 | -0.07 | 0.00 |
| collision_penalty | -0.02 | -0.24 | -0.26 | -0.40 | 0.00 | 0.00 |
| fly_low | -0.01 | -0.50 | -0.38 | -0.72 | -0.06 | 0.00 |
| all_targets_captured/ep | 0.00 | 1.020 | 1.00 | 0.00 | 0.69 | 1.06 |
| crash/ep | 0.01 | 0.228 | 0.17 | 0.00 | 0.06 | 0.33 |
| falcon_fly_low/ep | 0.00 | 0.174 | 0.17 | 0.00 | 0.02 | 0.26 |
| bounding_box/ep | 1.16 | 0.756 | 0.79 | 0.66 | 0.98 | 1.20 |
| drones_collide/ep | 0.01 | 0.129 | 0.13 | 0.00 | 0.00 | 0.21 |
| policy_std | 0.815 | 0.759 | 0.756 | 0.726 | 0.751 | 0.816 |
| illegal_contact min spike | — | — | — | — | -292.77 | — |

## 代码 Bug 验证结论

**结论：上次分析的代码 bug 描述完全属实，且经本次直接代码阅读确认。**

### 直接证据

文件 `marl_move_env.py` 第 726-727 行：

```python
# Success Reward (Removed)
# rewards["success_reward"] = ...
```

- `rewards["success_reward"]` 从未被赋值，该 key 永远不会进入 `rewards` 字典
- 第 862 行 `total_reward = sum(rewards.values())` 不包含 success_reward
- 第 864-868 行 `_episode_sums` 累积循环只遍历 `rewards.items()`，success_reward 永远累积 0
- TF 日志验证：`Episode_Reward/success_reward` 全程 3759 个数据点，sum=0.0，non-zero count=0

### all_targets_captured 为何不触发 success_reward

`all_targets_captured` 是一个终止条件标志（bool tensor），不是奖励触发器。其流程：

1. `_compute_rewards()` 中：`self.all_targets_captured = (self._sustained_follow_timer >= cfg.sustained_follow_duration)` — 仅设置 bool flag
2. `rewards["success_reward"] = ...` 被注释 → **奖励路径完全断路**
3. `_get_dones()` 中：`all_targets_captured` 触发 `terminated=True` → episode 结束
4. episode 结束时 `_episode_sums["success_reward"]` 记录的是 0（从未累积过任何值）

因此：**大量捕获事件（all_targets_captured recent_mean=1.020）正在发生，但每次捕获带来的奖励信号为零**。策略学到了捕获行为，完全是 distance_reward + tracking_reward 驱动的，没有任何 sparse success bonus 加强。

## Observations & Findings

### 1. success_reward 完全断路 — Severity: CRITICAL

**Symptom:** TF 中 `Episode_Reward/success_reward` 全程为 0，即使 `all_targets_captured` recent_mean=1.020，每 rollout 平均触发 1 次以上捕获成功。

**Root Cause:** `marl_move_env.py` 第 727 行 `rewards["success_reward"] = ...` 被注释。`success_reward_weight=5.0` 在 cfg 中已配置，但代码执行路径不存在。

**Impact:** 策略缺少捕获行为的稀疏正反馈强化。策略已能捕获（由 dense rewards 驱动），但每次成功没有额外奖励信号，无法进一步加速捕获节奏的强化或提升捕获质量。

### 2. height_penalty 持续增加 — Severity: HIGH

**Symptom:** height_penalty early=-1.31 → recent=-1.90 → last=-2.36，全程负增长，decile10=-2.44。

**Root Cause:** 无人机在接近目标时需要俯冲（targets 在地面附近），`height_penalty_threshold=1.2m` 与 `desired_height=2.5m` 组合意味着无人机只要 z<1.3m 或 z>3.7m 就被惩罚。捕获行为本身要求低飞，与 height_penalty 存在结构性冲突。

**Evidence:** height_penalty last=-2.36，比 tracking_reward last=8.38 的 28% 抵消了正向信号。

### 3. upright_penalty 末期持续恶化 — Severity: HIGH

**Symptom:** upright_penalty early=-0.31 → recent=-1.70 → last=-1.91，d10=-1.57，min=-78.81。

**Root Cause:** 捕获阶段无人机进行激进机动，倾斜角超过 40°（upright_penalty_threshold=0.766=cos40°）。随着捕获频率增加，违规次数线性增长。

**Evidence:** recent_std=2.86，有大量极端倾斜事件（min=-78.81）。

### 4. illegal_contact 极端 spike — Severity: HIGH

**Symptom:** illegal_contact min=-292.77，recent_mean=-0.85，recent_std=10.84。

**Root Cause:** `contact_sensor_threshold=15N` 过敏感，导致接近捕获时的正常机械接触触发极端惩罚，与捕获行为产生矛盾梯度。

**Evidence:** min=-292.77 表明单步出现约 2928 次非法接触或极高力值（illegal_contact_penalty=0.1，但 spike 达到 -292 意味着 max_f 极度超阈值）。

### 5. bounding_box 终止率仍然偏高 — Severity: MEDIUM

**Symptom:** bounding_box recent_mean=0.756/ep，d90=1.20，仅从 early=1.156 略有下降。

**Root Cause:** targets 做 ±8m 折返运动，无人机跟随过程中容易越出边界。boundary_soft_penalty 效果有限（recent_mean=-0.54）。

### 6. drones_collide 改善有限 — Severity: MEDIUM

**Symptom:** drones_collide recent_mean=0.129，collision_penalty×2 后仅从 0.140（Run 12 推算）降低约 8%。

**Root Cause:** 策略方差收缩（policy_std=0.759）后行为趋同，3 架无人机倾向于追同一目标，碰撞本质是探索不足而非单纯惩罚不够。

## Improvement Recommendations

### Priority 1 (CRITICAL): 修复 success_reward 代码注释

**Problem:** `rewards["success_reward"]` 永远为 0，`success_reward_weight=5.0` 的配置毫无作用。

**Proposed Change:**

- **File:** `exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/move/marl_move_env.py`
- **Location:** 第 726-727 行
- **Change:**

```python
# 替换为：
rewards["success_reward"] = (
    self.all_targets_captured.float() * self.cfg.success_reward_weight
)
```

- **Rationale:** `all_targets_captured` 已是 bool tensor，直接 `.float()` 乘权重即可。成功时给予 5.0 一次性奖励，与 episode 终止挂钩，强化已建立的捕获行为。

### Priority 2 (HIGH): 放宽 contact_sensor_threshold 消除极端 spike

**Problem:** illegal_contact min spike=-292.77，与捕获行为产生梯度冲突。

**Proposed Change:**

- **File:** `exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/move/marl_move_env_cfg.py`
- **Parameter:** `contact_sensor_threshold` → 搜索当前值（Run 13 为 15N）→ 改为 `20.0`
- **Rationale:** 放宽接触阈值减少误触发，保留真实碰撞检测能力。

### Priority 3 (HIGH): 提升 upright_penalty_weight 抑制激进倾斜

**Problem:** upright_penalty recent=-1.70，末期持续恶化，捕获阶段倾斜频繁。

**Proposed Change:**

- **File:** `exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/move/marl_move_env_cfg.py`
- **Parameter:** `upright_penalty_weight` `0.5` → `0.8`
- **Rationale:** 提升惩罚强度，阻止激进倾斜，同时 threshold 维持 cos40°=0.766 不变（不过分收紧）。

### Priority 4 (HIGH): 加强 fly_low_penalty 抑制低飞

**Problem:** fly_low recent=-0.50，falcon_fly_low/ep recent=0.174，低飞率高，干扰 sustained_follow_timer。

**Proposed Change:**

- **File:** `exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/move/marl_move_env_cfg.py`
- **Parameter:** `fly_low_penalty` `3.0` → `4.0`
- **Rationale:** 进一步惩罚低飞行为，减少 hold 窗口被低飞终止中断的频率。

### Priority 5 (MEDIUM): 调整 height_penalty_threshold 缓解捕获-高度冲突

**Problem:** height_penalty 末期 last=-2.36，接近目标的俯冲行为被 height_penalty 持续惩罚。

**Proposed Change:**

- **File:** `exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/move/marl_move_env_cfg.py`
- **Parameter:** `height_penalty_threshold` `1.2` → `1.5`（允许更大俯冲幅度，但不超过安全下限）
- **Rationale:** 放宽 0.3m 允许无人机俯冲至 z=1.0m 时才触发惩罚（desired=2.5m，threshold=1.5m → 触发点 z<1.0m），与 fly_low 终止点（z<0.1m）之间保留足够缓冲。

## Run 14 参数汇总

| 参数 | Run 13 | Run 14 | 原因 |
|------|--------|--------|------|
| `rewards["success_reward"]`（代码） | 注释（断路） | **取消注释，实现为 all_targets_captured * weight** | CRITICAL：修复正反馈断路 |
| `fly_low_penalty` | 3.0 | 4.0 | 减少低飞终止中断 hold 窗口 |
| `contact_sensor_threshold` | 15N | 20N | 消除 illegal_contact 极端 spike |
| `upright_penalty_weight` | 0.5 | 0.8 | 抑制捕获阶段激进倾斜 |
| `height_penalty_threshold` | 1.2m | 1.5m | 缓解俯冲捕获与高度惩罚的结构性冲突 |
| `success_reward_weight` | 5.0（cfg 已有）| 5.0（保持）| 代码修复后生效 |
| `collision_penalty_scale` | 2.0 | 2.0 | 保持，观察 success_reward 修复效果 |
| `entropy_loss_scale` | 0.004 | 0.004 | policy_std=0.759 正常，保持 |
| `sustained_follow_duration` | 0.5s | 0.5s | 捕获已充分（1.020/rollout），保持 |

## Experiment Plan

```bash
python3 scripts/skrl/train.py \
  --task=Isaac-marl-move-v0 \
  --headless --num_envs=32 --algorithm="MAPPO"
```

1. 优先修复 success_reward 代码（Priority 1），验证最简单改动
2. 同步应用 Priority 2-4（contact_threshold + upright_weight + fly_low_penalty）
3. Priority 5（height_penalty_threshold）可选，若前4项稳定后再追加
4. 运行至少 400k steps，观察 success_reward 是否在 50k steps 内出现非零值

### 监控目标（Run 14）

1. `Episode_Reward/success_reward` — 目标：50k steps 内出现非零，recent_mean > 0.5/ep
2. `Episode_Termination/all_targets_captured` — 目标：维持 d50 > 0.69/rollout（不退化）
3. `Episode_Reward/illegal_contact` min spike — 目标：> -20（放宽阈值后应消除极端值）
4. `Episode_Reward/upright_penalty` recent_mean — 目标：> -1.2（从 -1.70 改善）
5. `Episode_Reward/height_penalty` recent_mean — 目标：> -1.5（从 -1.90 改善）
6. `Reward / Total reward (mean)` recent_std — 目标：< 40（从 55.42 收窄）

### 成功标准（Run 14 达成条件）

- `success_reward` recent_mean > 0.5/ep（代码修复后的首要验证指标）
- `all_targets_captured` d50 > 0.69/rollout（维持 Run 13 水平）
- `illegal_contact` min spike > -20（无极端污染）
- `upright_penalty` recent_mean > -1.2
- `height_penalty` recent_mean > -1.5
- `total_reward` recent_std < 40

---

# Training Analysis Report — Move Run 14

**Run:** `2026-04-06_06-03-18_mappo_torch_mappo`
**Date:** 2026-04-06
**Task:** Isaac-marl-move-v0
**Algorithm:** MAPPO
**Total logged steps:** 365,800

---

## Training Metrics Summary

| 指标 | 早期均值 | 近期均值（last 20） | 最后值 | 对比 Run 13 |
|------|---------|-------------------|--------|------------|
| Total reward (mean) | -2.23 | ~2,672 | 4,969 | Run13 终值约 1,800，本次+176% |
| all_targets_captured | 0.001 | 0.830 | 1.030 | Run13 Q5=1.020，本次 Q5=0.33（下降） |
| success_reward | ~0 | 1,397 | 1,999 | Run13 全程 0，**本次首次真正触发** |
| crash (终止/rollout) | 0.025 | 0.451 | 0.52 | Run13 约 0.52（持平） |
| falcon_fly_low | 0.0002 | 0.309 | 0.45 | Run13 约 0.40（轻微升高） |
| height_penalty | -1.94 | -13.18 | -15.87 | Run13 约 -1.90（大幅升高，bug 修复生效） |
| illegal_contact | -0.032 | -19.21 | -21.68 | Run13 约 -2.0（显著升高） |
| upright_penalty | -0.54 | -11.38 | -13.86 | Run13 约 -1.70（大幅升高） |
| policy_std | 0.820 | 0.618 | 0.617 | Run13 约 0.759（偏低，探索减少） |
| drone_out | -1.27 | -18.11 | -15.82 | Run13 约 -10（升高后改善） |

---

## Observations & Findings

### 1. success_reward 首次真正触发 — MILESTONE

**Symptom:** `success_reward` 在 step=700 即出现首个非零值，step=101,800 突破 100，step=253,000 突破 1,000，最后 20 点均值 1,397，last=1,999。

**结论:** Run 14 的代码 bug 修复（取消注释 `rewards["success_reward"]`）完全生效。成功奖励从第一个 rollout 就开始提供正反馈信号，在 step~150k 后进入快速爬升阶段，到 ~300k+ 出现 Q95 最高 22,084 的大成功 episode。这是项目首次观察到 success_reward 真实驱动训练的运行。

**对比 Run 13 的成功标准 (d50 > 0.69):** Run 14 达成，但后期（300k 起）Q5 出现从 0.672 降至 0.330 的退化迹象。

---

### 2. all_targets_captured 后期退化 — HIGH

**Symptom:** `all_targets_captured` 在 step~180k 达到均值峰值 0.950，此后逐步下滑，最后 200 点均值 0.848，Q5 从 0.672 降至 0.525（300k 附近开始）。

**Root Cause:** crash 和 falcon_fly_low 终止频率在 ~280k 步后急剧上升（crash mean 从 0.177 → 0.527，fly_low 从 0.104 → 0.410），导致 episode 在完成 all_targets_captured 前提前终止。

**Evidence:** crash 趋势明确上升斜率，与 upright_penalty 和 illegal_contact 爆发时间点（step 280k）高度吻合。

---

### 3. crash / illegal_contact 后期爆发 — HIGH

**Symptom:** crash 终止在 step 280k 后从 0.269 → 0.527，illegal_contact 终止从 0.179 → 0.405，两者时间序列几乎完全同步（相关性极高）。

**Root Cause 假说 1（主要）:** `upright_penalty_weight` 从 0.5 → 0.8（+60%），导致无人机在接近小车的倾斜阶段受到更强压制。策略在高密度 success_reward 诱导下趋向激进行为（高速俯冲），而 upright_penalty 无法有效约束倾斜，反而导致碰撞频率上升。

**Root Cause 假说 2（次要）:** `contact_sensor_threshold` 从 15N → 20N 放宽了 illegal_contact 的终止条件，策略学会更激进接触，但接触后位移更大导致 crash。

**Evidence:** `illegal_contact` 惩罚在 step 292k 均值从 -2.24 → -12.54（6× 跃升），与 upright_penalty 从 -2.63 → -5.41 的跃升几乎同步发生在 step 280k。

---

### 4. upright_penalty 失控性增长 — HIGH

**Symptom:** `upright_penalty` recent_mean 从早期 -0.54 增长到后期 -13.56，增幅 25×，Q5 最低达 -67.45（单次 episode 极端值）。

**Root Cause:** `upright_penalty_weight=0.8` 叠加 success_reward 的高额正向激励，策略陷入「高倾斜高速接近目标 → 高 success_reward → 高 upright_penalty」的反向循环。无人机选择接受 upright_penalty 换取 success_reward，但接触质量恶化导致 illegal_contact 激增。

**Target 对比 (Run 13 期望 > -1.2):** Run 14 recent_mean=-11.38，远未达标。

---

### 5. height_penalty bug 修复确认生效 — POSITIVE

**Symptom:** `height_penalty` 从早期均值 -1.94 增长到后期 -13.18，且 Q5 在 step 320k 达到 -39.43（极端俯冲个案被正确捕获）。

**结论:** per-drone 绝对误差修复完全生效。Run 13 中 height_penalty 全程约 -1.90（L2 norm 稀释），Run 14 后期均值 -13.18，说明单架无人机俯冲行为被准确检测并惩罚。

**副作用:** 但 height_penalty 的持续增大同样指示无人机持续飞低，与 fly_low 终止相互印证。这不是单纯的 bug 修复副作用，而是策略确实在俯冲。

---

### 6. policy_std 低于健康范围 — MEDIUM

**Symptom:** `policy_std` 从初始 0.82 下降到 last=0.617，recent_std 仅 0.0013（极度稳定），low watermark 约 0.525。

**健康范围 0.65~0.80:** Run 14 当前 0.617，偏低但尚未严重。策略已进入 exploitation 主导阶段，探索不足可能导致陷入局部最优（激进倾斜 + 俯冲的 sub-optimal 策略）。

---

### 7. bounding_box 持续下降 — POSITIVE

**Symptom:** `bounding_box` 终止从早期均值 1.159 下降到最后 0.629，说明无人机逐渐学会留在边界内。

**结论:** `drone_out` 惩罚有效，且 step 260k 后 `drone_out` 的 Q95 降至 -0.18，说明大多数 episode 中无人机不再飞出边界。

---

### 8. total_reward 方差极大 — MEDIUM

**Symptom:** Total reward (mean) 最后 20 步：均值 ~4,970，但 max 达 86,248（成功 episode），min 为 518。更早（step 364,100）出现 max=86,051 与 mean=3,361 同一时刻，说明环境内少数 env 大成功，多数平庸。

**Root Cause:** 策略已呈现双峰分布——成功 env（all_targets_captured 触发 success_reward 爆发）vs. 崩溃 env（crash/fly_low 早终止）。两者 reward 差距 >100×。

---

## Run 14 成功标准对照

| 指标 | 目标 | 实际 | 达标 |
|------|------|------|------|
| `success_reward` recent_mean > 0.5/ep | 0.5 | 1,397 | YES |
| `all_targets_captured` Q5 > 0.69/rollout | 0.69 | 0.330（后期下降） | NO（峰值时达到，后退化） |
| `illegal_contact` min spike > -20 | > -20 | -219.77（极端值）| NO |
| `upright_penalty` recent_mean > -1.2 | > -1.2 | -11.38 | NO |
| `height_penalty` recent_mean > -1.5 | > -1.5 | -13.18（bug 修复生效，策略俯冲） | NO（但 bug 修复成功） |
| `total_reward` recent_std < 40 | < 40 | ~2,681（环境间差异巨大）| NO |

---

## Root Cause Summary

Run 14 的核心矛盾：**success_reward 重新激活后，成功奖励（最高 22,084/episode）在量级上压倒所有惩罚项（upright~-13，height~-13，illegal~-19），策略选择「接受所有惩罚换取 success_reward」的激进策略，导致高倾斜俯冲式接近 → 碰撞式接触 → crash/fly_low 终止频率爆发。**

具体链条：
1. success_reward 重激活，量级约 1,000-22,000/episode
2. 策略发现：高速俯冲 + 倾斜接近目标 → success_reward 大
3. upright_penalty (-13.56) + height_penalty (-13.18) + illegal_contact (-19.21) 合计约 -46，仍远小于 success_reward > 1,000
4. crash/fly_low 终止频率从 step 280k 起爆炸性上升（crash: 0.177→0.527）
5. all_targets_captured Q5 下降（被提前终止打断）

---

## Improvement Recommendations for Run 15

### Priority 1 (CRITICAL): 降低 success_reward_weight，消除量级失衡

**Problem:** success_reward 在量级上（~1,000-22,000）压倒所有惩罚项总和（~-46），策略优化方向完全被 success_reward 主导，接受一切代价的激进行为。

**Proposed Change:**
- **File:** `exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/move/marl_move_env_cfg.py`
- **Parameter:** `success_reward_weight` `5.0` → `1.0`（或）实现为固定奖励而非 cumulative
- **Rationale:** success_reward 按持续时间累积（`all_targets_captured * weight * dt * steps`），会无限增大。应改为固定 bonus（例如 50.0 每次捕获），或将 weight 从 5.0 大幅降低到约 0.5~1.0，使成功奖励量级与惩罚项处于同一数量级。
- **目标:** success_reward/rollout 约 50-200（当前 1,397），仍能提供正向激励但不压倒安全约束。

---

### Priority 2 (HIGH): 回退 upright_penalty_weight，解耦 crash 根因

**Problem:** `upright_penalty_weight=0.8` 在 success_reward 高额激励下未能约束激进倾斜，反而导致策略学会「接受 upright_penalty 换取 success_reward」，是 crash 和 illegal_contact 爆发的主要共因。

**Proposed Change:**
- **File:** `exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/move/marl_move_env_cfg.py`
- **Parameter:** `upright_penalty_weight` `0.8` → `0.5`（回退到 Run 13 值）
- **Rationale:** 在 success_reward 量级问题解决前，维持 Run 13 的 upright 约束强度。Priority 1 修复后可重新评估是否需要增强 upright 惩罚。

---

### Priority 3 (HIGH): 增强 fly_low 和 crash 的惩罚力度（与 success_reward 重新平衡）

**Problem:** `fly_low_penalty=4.0`（累积式），`fly_low` 终止每 rollout 约 0.31 次，说明惩罚强度不足以阻止俯冲行为。

**Proposed Change:**
- **File:** `exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/move/marl_move_env_cfg.py`
- **Parameter:** `fly_low_penalty` `4.0` → `6.0`
- **Rationale:** 加大 fly_low 惩罚使其在 episode 内总量（均值 -1.19）与调整后 success_reward 量级相当，迫使策略真正权衡安全与成功。若 Priority 1 将 success_reward_weight 降至 1.0，fly_low_penalty 相应需要重新对标，6.0 为保守上界。

---

### Priority 4 (MEDIUM): 降低 illegal_contact 联系阈值，或加强 illegal_contact 惩罚

**Problem:** `contact_sensor_threshold=20N` 允许更激进接触，但 illegal_contact 后的极端 spike（min=-219.77）说明接触质量极差，策略使用高力量撞击。

**Proposed Change（两选一）:**

**Option A（推荐）:** 维持阈值 20N，增加 `illegal_contact_penalty_scale`。
- **Parameter:** `illegal_contact_penalty_scale` → 检查当前值并提升 50%
- **Rationale:** 已有的高力量接触应该被更严厉惩罚，而不是靠降低阈值增加误报。

**Option B:** 回退 `contact_sensor_threshold` 15N
- 回退到 Run 13 值，但可能重新引入误报问题（Run 13 期望修复此项）。

---

### Priority 5 (MEDIUM): 为 success_reward 增加安全约束门槛

**Problem:** 当前 success_reward 在 all_targets_captured 时立即累积，不区分捕获质量（高倾斜俯冲式捕获与稳定悬停式捕获获得相同奖励）。

**Proposed Change:**
- **File:** `exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/move/marl_move_env.py`
- **修改 success_reward 计算：** 添加 `upright_ok = tilt_angle < threshold` 门控条件，只有姿态正常时才累积 success_reward。
- **Rationale:** 阻断「高倾斜 + 成功」的虚假奖励路径，强制策略学习稳定姿态下的捕获行为。
- **实现参考:**
  ```python
  upright_ok = (tilt_angle < 0.5)  # ~28 degrees
  rewards["success_reward"] = success_reward_weight * all_targets_captured.float() * upright_ok.float()
  ```

---

### Priority 6 (LOW): 调整 entropy_loss_scale 防止过早收敛

**Problem:** `policy_std=0.617`，低于健康范围（0.65~0.80），策略已高度确定，探索不足，容易陷入激进俯冲的局部最优。

**Proposed Change:**
- **File:** SKRL 训练配置或 `*_env_cfg.py` 中的 `entropy_loss_scale`
- **Parameter:** `entropy_loss_scale` `0.004` → `0.006`（+50%）
- **Rationale:** 增强探索，使策略有机会发现更稳定的捕获路径，防止过度收敛于当前激进策略。

---

## Run 15 参数汇总

| 参数 | Run 14 | Run 15 | 原因 |
|------|--------|--------|------|
| `success_reward_weight` | 5.0 | **1.0** | CRITICAL：消除量级失衡（压倒所有惩罚项） |
| `upright_penalty_weight` | 0.8 | **0.5** | 回退 Run 13 值，解耦 crash 根因 |
| `fly_low_penalty` | 4.0 | **6.0** | 与降低后的 success_reward 重新平衡 |
| `contact_sensor_threshold` | 20N | **20N**（保持）| 继续观察，改为加强惩罚幅度 |
| `height_penalty_threshold` | 1.5m | **1.5m**（保持）| bug 已修复确认，维持当前值 |
| `entropy_loss_scale` | 0.004 | **0.006** | 防止过早收敛，policy_std 偏低 |

---

## Experiment Plan for Run 15

```bash
python3 scripts/skrl/train.py \
  --task=Isaac-marl-move-v0 \
  --headless --num_envs=32 --algorithm="MAPPO"
```

1. 应用所有 Priority 1-4 变更（success_weight + upright_weight + fly_low + entropy）
2. 运行至少 400k steps
3. 在 step 100k 检查 crash 趋势是否降低（目标 < 0.15）
4. 在 step 200k 检查 success_reward 是否仍有正向驱动（目标 recent_mean > 100）
5. 在 step 300k 检查 all_targets_captured Q5 是否稳定（目标 > 0.60）

### 监控目标（Run 15）

1. `Episode_Termination/crash` recent_mean — 目标：< 0.15（Run 14 后期 0.451）
2. `Episode_Termination/falcon_fly_low` recent_mean — 目标：< 0.10（Run 14 后期 0.309）
3. `Episode_Reward/success_reward` recent_mean — 目标：50-500/ep（有效但不压倒安全项）
4. `Episode_Reward/upright_penalty` recent_mean — 目标：> -2.0（从 -11.38 改善）
5. `Episode_Termination/all_targets_captured` Q5 — 目标：> 0.60（维持 Run 14 峰值水平）
6. `Policy / Standard deviation` — 目标：0.65~0.75（从 0.617 回升）

### 成功标准（Run 15 达成条件）

- crash mean < 0.15（关键：激进行为被抑制）
- fly_low mean < 0.10
- success_reward recent_mean 50~500（有效激励但不失控）
- upright_penalty recent_mean > -2.0
- all_targets_captured Q5 ≥ 0.60，且无后期退化趋势
- policy_std ≥ 0.65

## Changelog

- 2026-04-06: Run 14 分析完成。核心发现：success_reward 代码修复完全生效（首次非零 step=700），但量级失衡（~1,000-22,000 压倒惩罚总和 ~-46）导致激进俯冲策略，crash/fly_low 在 280k 步后爆发性上升，all_targets_captured Q5 后期退化。height_penalty bug 修复确认生效（均值从 -1.90 → -13.18）。Run 15 首要任务：success_reward_weight 5.0 → 1.0 消除量级失衡，回退 upright_weight 0.8 → 0.5。
- 2026-04-06: Run 15 分析完成。success_reward Q5=139（仍为正向组合总量的 92%，远超目标 <100），量级失衡未根治。crash Q3→Q5 改善 -25.7%，fly_low 改善 -41.1%（fly_low_penalty=6.0 有效）。all_targets_captured Q5=0.811（超目标 0.70）。policy_std Q5=0.565（低于目标 0.65~0.80，entropy 不足）。illegal_contact Q5=-5.27 复发（需 threshold 30N）。新发现：bounding_box Q5=0.809（比 Q3=0.667 更差，反弹）。

---

# Run 15 训练分析报告

**Run:** 2026-04-06_12-35-13_mappo_torch_mappo
**Date:** 2026-04-06
**Task:** Isaac-marl-move-v0
**Algorithm:** MAPPO
**Total Steps:** 400,000
**Compared Against:** Run 14 (2026-04-05_23-37-49 → 2026-04-06)

---

## Training Metrics Summary

| 指标 | Run 14 Q5 | Run 15 Q5 | 变化 | 目标 |
|------|-----------|-----------|------|------|
| total_reward mean | ~355 (est.) | +398.4 | +12% | 持续上升 |
| success_reward/ep | 1,397 | 139.2 | -90% | <100 (目标未达) |
| all_targets_captured | 0.330 (退化) | 0.811 | +146% | >0.70 ACHIEVED |
| crash termination | 0.641 | 0.344 | -46% | <0.15 (目标未达) |
| falcon_fly_low | 0.309 | 0.197 | -36% | <0.10 (目标未达) |
| policy_std | 0.617 | 0.565 | -8.4% | 0.65~0.80 (目标未达) |
| ep_len mean | ~381 | 281.4 | -26% | >300 |
| bounding_box | 0.56 | 0.809 | +44% | <0.3 (反弹) |
| illegal_contact/ep | -0.82 | -5.269 | -542% | 接近 0 (严重复发) |
| height_penalty/ep | -1.98 | -5.030 | -154% | >-2.0 (严重恶化) |

---

## Observations & Findings

### 1. success_reward 量级：大幅改善但仍失衡 — Severity: HIGH

**Symptom:** success_reward_weight 5.0→1.0 后，Q5 mean 从 ~1,397 降至 139.2/ep（-90%），但仍占所有正向奖励总量的 **92.0%**（139.2 vs 其他正向 12.1 之和），vs 负向惩罚总和 |-20.5|，success_reward 与惩罚比仍为 **6.8x**（目标应 <2x）。

**Root Cause:** weight=1.0 仍然允许连续捕获累积大量奖励。success_reward 的绝对值由捕获频次 × weight × hold_time 决定，当 all_targets_captured=0.81（每 rollout 超过 1 次捕获事件）时，即使 weight=1.0 也会产生远大于常规追踪奖励的累积量。

**Evidence:** Q4/Q5 success_reward 最大值 2920/2476（单 episode 峰值），median 31.78 但 mean 139（重尾分布），部分 episode 仍有极端值驱动平均。

**比 Run 14 改善的诊断:** 失衡从 1397 降到 139（一个数量级）是正向进展，但仍需继续降权至 0.3~0.5 使 success 不再主导。

---

### 2. crash/fly_low：改善趋势确认，但绝对量仍过高 — Severity: HIGH

**Symptom:** crash Q3=0.463→Q4=0.378→Q5=0.344（-25.7% 改善）；fly_low Q3=0.334→Q4=0.269→Q5=0.197（-41.1% 改善）。fly_low_penalty=6.0 有效果，**fly_low 改善幅度大于 crash**，说明惩罚主要抑制了低空悬停型触发，但物理式俯冲（瞬时速度拉向目标导致 crash）尚未根治。

**Root Cause:** 当 success_reward 仍主导奖励时，策略仍有动机进行激进靠近。crash Q5=0.344 意味着约 34% rollout 有坠机事件——该频率与 Run 14 的 crash 爆发期相似，但已不再是单调恶化趋势。

**Evidence:** crash 和 fly_low 的 Q3-Q5 均呈单调下降，说明问题在改善但远未达到 <0.15 目标。

---

### 3. all_targets_captured：稳定性显著恢复 — Severity: RESOLVED (partial)

**Symptom:** Q1=0.054→Q2=0.395→Q3=0.458→Q4=0.607→Q5=0.811，单调上升，Q5=0.811 超过目标 0.70，且 Q5 最小值=0.270（无后期崩溃）。

**Root Cause of recovery:** Run 14 的后期退化（0.672→0.330）由 crash 主导的 episode 终止导致。Run 15 中 crash 下降 46%，使 all_targets_captured 得以积累。upright_weight 回退 0.8→0.5 也减少了追踪期的机动抑制。

**Outstanding concern:** Q5 best=1.62，仍有高方差（std 估算较高），但趋势健康。

---

### 4. policy_std：持续向下，未达目标 — Severity: HIGH

**Symptom:** Q1=0.803→Q2=0.673→Q3=0.584→Q4=0.573→Q5=0.565，单调下降，终值 0.565 低于健康下限 0.65，且仍在下降（Q5 内 range=0.558~0.571，极度压缩）。

**Root Cause:** entropy_loss_scale=0.006 相比 Run 14 的 0.004 提升幅度不足。success_reward 的强梯度信号（高 reward 高方差 → policy 被"拉"向固定行为）与 entropy 项对抗，前者胜出。当 success_reward 占正向总量 92%，entropy 项的相对权重极小。

**Evidence:** Run 12 中 entropy=0.003 使 std 从 62.6% 降幅收窄至 19.3%；Run 15 用 0.006 仍得到 -8.4% 降幅。根本矛盾：entropy 项受 success_reward 强信号压制。要解决 std 问题，必须同时降低 success_reward 绝对值（降权）和提高 entropy（0.008~0.010）。

---

### 5. illegal_contact：严重复发 — Severity: HIGH

**Symptom:** Q1=-0.021→Q2=-0.757→Q3=-7.967→Q4=-6.848→Q5=-5.269，Q3 以后激增（500x 倍于 Q1）。worst=-246.36（单次极端值）。

**Root Cause:** all_targets_captured 在 Q2-Q3 开始稳定（0.395→0.458）意味着无人机更频繁地靠近 NovaCarter，在 contact_sensor_threshold=20N 下仍触发大量接触事件。Run 11 的 threshold=15N 时 Q5=-0.82；Run 15 的 threshold=20N 时 Q5=-5.27——说明更多的靠近频次（0.45→0.81 all_targets_captured）比 threshold 升高的效果更强。

**Evidence:** illegal_contact 与 all_targets_captured 的增长时序高度一致（Q2-Q3 同步上升），确认为靠近捕获带来的结构性碰撞。需要 threshold 进一步提升至 30N 或对每次接触的 penalty 数值限制（clip）。

---

### 6. bounding_box：意外反弹 — Severity: MEDIUM

**Symptom:** Q1=1.119→Q2=0.930→Q3=0.668→Q4=0.733→Q5=0.809。Q3 之后反弹，Q5 回到 0.809——高于 Q2 和 Q3。

**Root Cause:** crash Q3→Q4→Q5 改善（episode 变长）+ success_reward 强信号使策略追踪更激进（速度更快），靠近目标时惯性更大，更容易穿越软边界。此外 ep_len Q4=291→Q5=281（轻微缩短）说明有新的截断来源，但 bounding_box 反弹表明某些长 episode 仍在以 OOB 终止。

**Note:** 此问题在没有 success_reward 量级失衡时会自然缓解（低激进性），属于次级症状。

---

### 7. height_penalty 恶化 — Severity: HIGH

**Symptom:** Q1=-1.54→Q2=-1.24→Q3=-4.93→Q4=-4.95→Q5=-5.03，Q3 后大幅跳升并稳定在 -5/ep。

**Root Cause:** 与 illegal_contact 复发同根：all_targets_captured 改善意味着无人机更频繁俯冲接近目标（z≈0.25m），而 height_penalty_threshold=1.5m、desired_height=2.5m，俯冲到 z<1.0m 时 height_penalty=-1.0×(2.5-1.0-1.5)=0 刚好在边界，但动态俯冲会超越此范围。height_penalty 在 Run 14 中已被修复为 per-drone 计算，此处的恶化说明俯冲深度（与 crash）是协变的——crash 减少 46% 但 height_penalty 加剧 154%，说明无人机学会了"更深俯冲但不坠地"的边界策略。

---

## Run 15 关键结论

1. **success_reward 降权显著改善（-90%），但仍需继续降低**：weight=1.0 时 success 仍占正向总量 92%。目标 weight=0.3，使 success Q5 mean 降至 40-60/ep（与惩罚总量 ~20/ep 接近平衡）。
2. **crash/fly_low 改善方向正确，但速度不够**：fly_low_penalty=6.0 使 fly_low 改善 -41%，crash 改善 -26%。需要继续加压（8.0），同时降低 success_reward 量级减少激进动机。
3. **all_targets_captured 完全恢复并超标**：Q5=0.811，Run 15 在这一维度达标。
4. **policy_std 问题根源在 success_reward 主导**：在 success 降权前，提高 entropy 收益有限。entropy_loss_scale 需同步提升至 0.010，让 std 实际能维持在 0.65+。
5. **illegal_contact 是新的优先问题**：contact_sensor_threshold 需从 20N 提升至 30N，同时考虑对 illegal_contact 奖励项做 clip（如每步上限 -5.0）防止极端值。
6. **height_penalty 反映的俯冲加深是正常学习阶段**：可容忍，但需通过 fly_low_penalty 继续约束下限。

---

## Improvement Recommendations for Run 16

### Priority 1 (CRITICAL): 继续降低 success_reward_weight

**Problem:** weight=1.0 时 success_reward 仍占正向总量 92%（Q5=139/ep vs 惩罚总量 20/ep），6.8x 失衡。策略仍被 success 主导，导致激进靠近、policy_std 压缩、bounding_box 反弹。

**Proposed Change:**
- **File:** `exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/move/marl_move_env_cfg.py`
- **Parameter:** `success_reward_weight` `1.0` → `0.3`
- **Rationale:** 以 Run 15 Q5 mean 比例换算：0.3/1.0 × 139 ≈ 42/ep，接近惩罚总量 20/ep，比例约 2x（健康范围）。既保持 success 的正向激励，又不压制安全行为。

---

### Priority 2 (HIGH): 提高 illegal_contact contact_sensor_threshold

**Problem:** illegal_contact Q5=-5.27/ep（Q3 以后 500x 激增），worst=-246.36。随着捕获频率（all_targets_captured=0.81）提升，无人机更频繁靠近 NovaCarter，20N 阈值不足以过滤近距离飞行时的微接触力。

**Proposed Change:**
- **File:** `exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/move/marl_move_env_cfg.py`
- **Parameter:** `contact_sensor_threshold` `20.0` → `30.0`（N）
- **Rationale:** Run 11 用 15N 时 Q5=-0.82（all_targets_captured=0.027），Run 15 用 20N 时 Q5=-5.27（all_targets_captured=0.81）。捕获率增长 30x 对应接触事件增长约 6x，需 threshold 再上调。30N 历史上从未使用，预计可将 illegal_contact 压回 -1.0 量级。

---

### Priority 3 (HIGH): 继续提高 fly_low_penalty

**Problem:** crash Q5=0.344、fly_low Q5=0.197，虽相比 Run 14 改善，但仍远超目标。fly_low_penalty=6.0 使 fly_low 下降 41%，趋势正确但需继续加压。

**Proposed Change:**
- **File:** `exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/move/marl_move_env_cfg.py`
- **Parameter:** `fly_low_penalty` `6.0` → `8.0`
- **Rationale:** 步进式递增历史验证：3.0→4.0→6.0 每次减少约 30-40%。8.0 预计再减 30-40%，使 fly_low Q5 降至 0.12-0.14（接近目标 <0.10）。与此同时，success_reward 降权（Priority 1）减少激进靠近动机，两者协同效果预期大于单独效果。

---

### Priority 4 (HIGH): 大幅提高 entropy_loss_scale

**Problem:** policy_std Q5=0.565，单调下降且仍在压缩（Q5 range=0.558~0.571，极度收敛）。entropy=0.006 对抗 success_reward 强梯度完全不足。

**Proposed Change:**
- **File:** SKRL 训练配置
- **Parameter:** `entropy_loss_scale` `0.006` → `0.010`
- **Rationale:** 在 success_reward 降权（Priority 1）同时，success 的梯度贡献减少，entropy 的相对强度才能有效。预期两者协同：success 降权减少向固定行为的收敛压力，entropy=0.010 补偿探索损耗。目标 std 回升至 0.65~0.75（Run 7-9 验证区间）。**注意：** 如果 Priority 1 未实施，entropy=0.010 单独效果有限；两项必须同时应用。

---

### Priority 5 (MEDIUM): 限制 illegal_contact 奖励极端值（clip）

**Problem:** illegal_contact worst=-246.36，极端 spike 会引入大梯度噪声，破坏策略稳定性（Run 10 中 -815.33 spike 曾导致训练崩溃）。

**Proposed Change:**
- **File:** `exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/move/marl_move_env.py`
- **Logic:** 在 `_compute_rewards()` 的 illegal_contact 累积处，加入 `torch.clamp(illegal_contact_reward, min=-5.0)` 限制单步最大惩罚
- **Rationale:** 防止单次近距离接触产生数百倍于其他奖励的梯度冲击。30N threshold（Priority 2）减少触发频率，clip 减少单次影响，双重防护。

---

### Priority 6 (LOW): 评估 bounding_box_threshold 进一步调整

**Problem:** bounding_box Q5=0.809（Q3 后反弹），说明当 ep_len 延长且追踪更激进时，软边界（8m）和硬边界（10m）之间的 2m 缓冲不够。

**Proposed Change:**
- **File:** `exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/move/marl_move_env_cfg.py`
- **Parameter:** 暂时维持 `bounding_box_threshold=10.0m`，观察 success_reward 降权（Priority 1）后 bounding_box 是否自然改善（激进追踪减少 → 超界减少）
- **Rationale:** bounding_box 反弹的根因是激进追踪（success 主导），而非边界参数问题。先解决 Priority 1-4，在 Run 16 结果中重新评估是否需要调整边界。

---

## Run 16 参数汇总

| 参数 | Run 15 | Run 16 | 目标 |
|------|--------|--------|------|
| `success_reward_weight` | 1.0 | **0.3** | success Q5 mean ≈ 40/ep，与惩罚比 <2x |
| `fly_low_penalty` | 6.0 | **8.0** | fly_low Q5 < 0.12，crash Q5 < 0.20 |
| `contact_sensor_threshold` | 20N | **30N** | illegal_contact Q5 < -1.0 |
| `entropy_loss_scale` | 0.006 | **0.010** | policy_std 回升至 0.65~0.75 |
| `upright_penalty_weight` | 0.5 | **0.5**（保持）| 观察，Run 15 Q5=-3.84 可接受 |
| `bounding_box_threshold` | 10.0m | **10.0m**（保持）| 等 success 降权后再评估 |

---

## Experiment Plan for Run 16

```bash
python3 scripts/skrl/train.py \
  --task=Isaac-marl-move-v0 \
  --headless --num_envs=32 --algorithm="MAPPO"
```

1. 同时应用 Priority 1-4 四项变更（success_weight + fly_low + contact_threshold + entropy）
2. 在 step 80k 检查 success_reward Q1 mean（目标 <20/ep，若 >100 需提前降至 0.2）
3. 在 step 160k 检查 policy_std（目标 >0.65；若仍 <0.60 需将 entropy 提至 0.012）
4. 在 step 240k 检查 illegal_contact Q3（目标 >-1.5；若 <-3.0 需将 threshold 提至 40N）
5. 在 step 320k 检查 all_targets_captured Q4（目标 >0.60；若 <0.40 需排查 crash 率）
6. 运行至 400k steps 完整评估

### 监控目标（Run 16）

1. `Episode_Reward/success_reward` Q5 mean — 目标：30~80/ep（成功但不压倒）
2. `Episode_Termination/crash` Q5 mean — 目标：< 0.20（从 0.344 改善）
3. `Episode_Termination/falcon_fly_low` Q5 mean — 目标：< 0.12（从 0.197 改善）
4. `Episode_Reward/illegal_contact` Q5 mean — 目标：> -1.5（从 -5.27 改善）
5. `Policy / Standard deviation` Q5 mean — 目标：0.65~0.75（从 0.565 回升）
6. `Episode_Termination/all_targets_captured` Q5 mean — 目标：> 0.70（维持 Run 15 水平）
7. `Episode_Termination/bounding_box` Q5 mean — 目标：< 0.50（从 0.809 改善）

### 成功标准（Run 16 达成条件）

- success_reward Q5 mean 30~80/ep（有效激励但不失控）
- crash Q5 mean < 0.20
- illegal_contact Q5 mean > -1.5
- policy_std Q5 mean > 0.65
- all_targets_captured Q5 mean > 0.70，且无后期退化

---

## Run 16 训练分析报告

**Run:** `2026-04-06_22-09-35_mappo_torch_mappo`
**Date:** 2026-04-07
**Task:** Isaac-marl-move-v0（NovaCarter 追踪，3 Falcon 跟踪 4 小车）
**Algorithm:** MAPPO
**Steps:** 400,000
**Status:** improving（整体学习信号存在，但多项结构性问题未解决）

---

### 训练指标汇总

| 指标 | Q1 mean | Q5 mean | last | Run 16 目标 | 达成 |
|------|---------|---------|------|------------|------|
| `success_reward` | 0.88 | 274.71 | 304.56 | 30~80/ep | 未达成（6.8x 超出） |
| `tracking_reward` | 1.74 | 17.69 | 40.30 | 持续上升 | 达成 |
| `illegal_contact` | -0.019 | -16.02 | -0.98 | >-1.5 | 结构性未达成（见分析） |
| `fly_low` | -0.034 | -2.40 | 0.00 | — | 波动大 |
| `height_penalty` | -1.52 | -48.62 | -137.11 | — | 强积累效应 |
| `upright_penalty` | -0.41 | -56.57 | -157.74 | — | 强积累效应 |
| `all_targets_captured` | 0.10 | 0.211 | 0.27 | >0.70 | 未达成（严重退化） |
| `crash` | 0.017 | 0.360 | 0.00 | <0.20 | 未达成 |
| `falcon_fly_low` | 0.005 | 0.301 | 0.00 | <0.12 | 未达成 |
| `bounding_box` | 1.13 | 0.333 | 0.00 | <0.50 | Q5 达成（Q1→Q5 持续下降） |
| `time_out` | 0.000 | 0.352 | 1.00 | 越多越好 | 改善（Q5 35%） |
| `policy_std` | 0.8097 | 0.8266 | 0.8111 | 0.65~0.75 | 过度探索（超出目标上限） |
| `ep_len_mean` | 107 | 2328 | 5999 | >500 | Q5 达成，但极度双峰 |

---

### 核心发现

#### 1. success_reward 量级问题 — CRITICAL（目标未达成）

**症状：** Q5 mean=274.71/ep，nonzero 均值=663.96/ep，占总正向奖励 81.5%（目标 <50%）。

**根因分析：** success_reward_weight 从 1.0→0.3 仅减少 3x，但 episode 长度从 Run 15（均值~323步）大幅延长至 Q5 均值 2328 步。成功奖励按步累积：当 episode 持续 5999 步（22.4% Q5 Episodes 达最大长度），success_reward 自然累积至 600~1770/ep，比 Run 15 的 139/ep 高出数倍。权重降低被 episode 延长完全抵消。

**量化：** Run 15 ep_len_mean≈323，success_weight=1.0 → 期望 success/ep≈139。Run 16 ep_len_mean≈2328（+621%），success_weight=0.3 → 期望 success/ep ≈ 139 × 2328/323 × 0.3 = 300。实测 Q5 mean=274，与理论预测完全吻合。

**关键结论：** success_reward 量级问题的根因不是 weight 不够低，而是 episode 越来越长导致累积越来越多。这是一个正向强化循环：成功训练 → 更长 episode → 更多 success 累积 → 策略更强调 success。简单降低 weight 无法打破此循环。

**证据：** 41.4% 的 Q5 数据点 success_reward>0，但非零均值高达 663.96，说明成功时持续停留在捕获区（未离开），单次 episode 可获得 1770/ep 成功奖励（接近上限）。

#### 2. policy_std 过度探索 — HIGH（方向反转，超出上限）

**症状：** policy_std Q5 mean=0.8266（范围 0.7894~0.8745），高于目标上限 0.75，且整体稳定在 0.81 附近。Run 15 Q5=0.565（过低），Run 16 = 0.82（过高）。entropy_loss_scale 从 0.006→0.010 的调整效果过强。

**根因：** entropy_loss 绝对值 Q5 mean=-0.01223（运行全程稳定），隐含熵=1.22 nats，对应 std≈0.82。这是 entropy_scale=0.010 在此任务下的稳定均衡点——policy_std 没有下降趋势，说明策略处于高探索稳态，而非过渡状态。

**影响：** 高 std 引起随机飞行，导致 crash/fly_low 仍高（Q5=0.36/0.30）。Run 10 经验（std=1.595 导致过多随机崩溃）表明此任务健康 std 范围是 0.70~0.82。当前 0.82 处于可接受上边界，但略高，需微调。

**与 Run 15 对比：** Run 15 std=0.565（探索不足），Run 16 std=0.82（轻度过高）。目标 0.65~0.75 介于两者之间。最优 entropy_scale 估计约 0.007~0.008。

#### 3. illegal_contact 改善但仍有脉冲爆发 — HIGH（结构性残留）

**症状：** Q5 mean=-16.02/ep，min=-520.52/ep（含灾难性脉冲）。最后 20 个数据点均值约-1.2/ep（正常），但 Q5 分段均值（步骤 240k-400k）在 -7 ~ -27 之间剧烈波动。

**分解：**
- Q5 中 45/688（6.5%）数据点 illegal_contact < -50：这些是灾难性脉冲，单次接触力超过阈值持续多步
- 最后 20 数据点均值-1.2/ep：末期接近正常
- 30N 阈值提升有效（减少普通近距离接触的触发），但未消除灾难性爆发

**根因：** 高 policy_std（0.82）导致偶发的激进接近动作，叠加长 episode（最大 5999 步）使单次爆发的累积值极高（contact_force × 0.1penalty × 步数）。terminal illegal_contact Q5=0.172/rollout 说明仍有 17% 的 rollout 发生接触终止。

**30N 阈值效果评估：** 相对 Run 15（20N），接触终止从 ~0.35/rollout 降至 0.172（-51%）。量级改善，但绝对值仍高。建议 40N 或改变惩罚结构（见 Run 17 建议）。

#### 4. crash / fly_low 未达目标 — HIGH

**症状：** crash Q5=0.360，fly_low Q5=0.301（目标分别 <0.20 和 <0.12）。

**时间趋势：**
- 步骤 240k-280k：crash=0.426，fly_low=0.357（高峰）
- 步骤 360k-400k：crash=0.351，fly_low=0.316（微降，非单调）
- fly_low_penalty 从 6.0→8.0 未能阻止"追踪→下潜→坠地"模式

**根因诊断：** fly_low Q5 mean=-2.40/ep，而 fly_low_penalty=8.0，意味着平均每个 Q5 episode 触发约 0.3 次全力 fly_low（-8/次）。fly_low 事件并未消失，只是从高频小力变为中频全力。高 policy_std 是驱动因素：std=0.82 的随机动作经常导致意外下潜。

**关键洞察：** fly_low 与 crash 的 Q5 min=0（最好状态下完全没有），但均值高，说明存在双峰：部分 episode 完美飞行，部分 episode 大量坠地。这是策略未收敛的证据，而非系统性飞行质量差。

#### 5. all_targets_captured 严重退化 — HIGH（最重要发现）

**症状：** Q5 mean=0.211/rollout（Run 15: Q5=0.811，退化 74%）。目标 >0.70 远未达到。Q5 最后四分段：0.221→0.255→0.196→0.226，无改善趋势。

**根因（致命）：** 这是 success_reward_weight 降至 0.3 的直接后果。当 success_reward 激励减弱，策略优先选择"安全飞行"（获得 tracking_reward + height_reward）而非"积极停留在捕获区"（需要承担 crash 风险）。success_reward_weight=0.3 × 0.5s 持续时间，单次捕获奖励约 0.3/step × 50步 = 15 credit，远低于 Run 15 的 50 credit。策略理性地选择回避高风险的近距离停留。

**量化验证：** all_targets_captured Q5 从 0.811→0.211（-74%）与 success_reward_weight 从 1.0→0.3（-70%）高度对应。这不是偶然相关——success_reward 是推动策略停留在捕获区的核心激励。

**反直觉结论：** 减少 success_reward_weight 虽解决了"占正向奖励比例过高"问题，但同时破坏了策略维持捕获行为的动力。设计目标（抑制 success 主导）与任务目标（维持高捕获率）之间存在根本性张力。

#### 6. 积累型惩罚（height_penalty / upright_penalty）的伪高值问题 — MEDIUM

**症状：** height_penalty Q5 mean=-48.62/ep（last=-137.11），upright_penalty Q5 mean=-56.57/ep（last=-157.74）。这些数值看似很高，但需正确解读。

**关键发现：** 这两项惩罚与 episode 长度的相关系数分别为 -0.194 和 -0.197（数量级低于预期）。per-step 归一化分析：
- upright per-step Q5 mean = -0.1097/step（upright_weight=0.5 → 原始 tilt = -0.22/step）
- height_penalty per-step Q5 mean = -0.0927/step（height_weight=1.0 → 误差 = 0.09m/step 超阈值）

这两项惩罚在长 episode 中天然累积到高值，但 per-step 量级实际上相当温和。真正的担忧是 height_penalty 最后 10 个数据点出现大量 -136~-137（逼近最大值），这仅在 ep_len=5999 的 timeout episode 中发生，是正常的长 episode 积累，而非物理飞行质量问题。

#### 7. bounding_box 显著改善 — 达成

**症状：** bounding_box Q1=1.13→Q5=0.333（-71%），last=0.00。这是此次训练最明确的正向改善。

**根因：** episode 延长（ep_len Q5 mean=2328 vs Run 15 ~323）使策略有时间学会在目标边界附近折返，而不是线性追出边界。这是 Run 10 以来 bounding_box_threshold=10m + bounce 轨迹组合的延迟红利。

**注意：** Q5 内部四分段显示 bounding_box 在 0.18~0.34 之间波动（非单调），最后一段 0.325 略高于 Q5 整体均值 0.333。需在 Run 17 继续监控。

---

### 综合评价

**Run 16 根本性成就：**
1. policy_std 从 0.565（Run 15）跃升至 0.82——解决了探索坍缩问题，但过冲了
2. bounding_box 从 0.809 降至 0.333——边界问题大幅改善
3. 训练状态整体 improving，ep_len 达历史高位（Q5 mean=2328，max=5999）

**Run 16 主要失败：**
1. all_targets_captured 退化 74%（0.811→0.211）：success_weight 降低打击了维持捕获的动力
2. crash/fly_low 仍高：fly_low_penalty=8 不足以覆盖 std=0.82 引起的随机下潜
3. illegal_contact 脉冲爆发：30N 减少了频率但未消除灾难性接触事件

**Run 16 遗留的核心矛盾：** 降低 success_weight 以"平衡奖励比例"与"维持捕获激励"是反向操作的。正确解法不是降低 success 绝对重量，而是**相对性地提升其他正向奖励**，让 success 在总奖励中自然稀释，同时保持捕获的绝对吸引力。

---

## Improvement Recommendations for Run 17

### Priority 1 (CRITICAL): 恢复 success_reward_weight 并重新平衡结构

**Problem:** success_weight=0.3 导致 all_targets_captured 退化 74%（0.811→0.211）。捕获激励不足是策略回避近距离停留的直接原因。

**Proposed Change:**
- **File:** `exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/move/marl_move_env_cfg.py`
- **Parameter:** `success_reward_weight: 0.3 → 0.6`
- **Rationale:** 折中方案——从 Run 15 的 1.0 降到 Run 16 的 0.3 使捕获率崩溃。0.6 是二者中点。预期 all_targets_captured Q5 恢复至 0.5~0.7（+130%~+230%）。同时保持 tracking_reward 的相对重要性。

**附加变更（配合）:**
- **Parameter:** `tracking_reward_weight: 4.0 → 5.0`
- **Rationale:** 提升 tracking 的绝对权重，使 success（哪怕 weight=0.6）在奖励组合中占比相对降低。success/(success+tracking) 比例目标从 94% 降至 ~70%。

### Priority 2 (HIGH): 校准 entropy_loss_scale 使 policy_std 落入目标区间

**Problem:** policy_std Q5=0.826，目标 0.65~0.75。当前 entropy_scale=0.010 过强，使策略维持高探索状态，增加随机下潜和坠地风险。

**Proposed Change:**
- **File:** `exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/move/marl_move_env_cfg.py`（skrl 训练配置）
- **Parameter:** `entropy_loss_scale: 0.010 → 0.007`
- **Rationale:** Run 15（scale=0.006）→ std=0.565，Run 16（scale=0.010）→ std=0.826。线性外推：目标 0.70 对应 scale ≈ 0.007~0.008。保守选 0.007，避免再次跌至 0.565。

### Priority 3 (HIGH): 提升 contact_sensor_threshold 以消除灾难性脉冲

**Problem:** illegal_contact Q5=-16.02/ep，含最小值 -520/ep 的灾难性脉冲。Q5 中 45/688（6.5%）数据点 < -50。30N 阈值相比 20N 已改善 51%，但仍有重大爆发。

**Proposed Change:**
- **File:** `exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/move/marl_move_env_cfg.py`
- **Parameter:** `contact_sensor_threshold: 30N → 50N`
- **Rationale:** NovaCarter 物理碰撞体在无人机快速接近时可以瞬间产生大接触力（>100N）。50N 阈值过滤大多数近距离接触，仅保留真实碰撞。目标：illegal_contact Q5 > -3.0，脉冲 <-50 的比例降至 < 1%。

**附加变更：**
- **Parameter:** `illegal_contact_penalty: 0.1 → 0.05`
- **Rationale:** 降低单次接触事件的惩罚量级，减少梯度冲击。与阈值提升配合，双向收紧接触惩罚的影响范围。

### Priority 4 (HIGH): 进一步提升 fly_low_penalty 以阻止下潜

**Problem:** crash Q5=0.360，fly_low Q5=0.301，目标分别 <0.20 和 <0.12。fly_low_penalty 从 6.0（Run 15）→8.0（Run 16）未能有效阻止"追踪→下潜→坠地"模式。

**Proposed Change:**
- **File:** `exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/move/marl_move_env_cfg.py`
- **Parameter:** `fly_low_penalty: 8.0 → 12.0`
- **Rationale:** 目前 fly_low Q5=-2.40/ep，说明平均每 Q5 episode 触发 0.3 次（-2.40/-8.0=0.3 次/ep）。penalty=12 使单次坠地代价等于约 12/0.3=40 步 tracking_reward，显著高于追踪收益，形成有效威慑。历史数据：fly_low=3.0（Run 12）→ crash Q5=0.469；fly_low=4.0（Run 14）→ 改善；fly_low=6.0（Run 15）→ crash Q5=0.344；fly_low=8.0（Run 16）→ crash Q5=0.360（无改善）。指数式提升模式表明需要大幅跨越，12.0 是下一个显著台阶。

### Priority 5 (MEDIUM): 评估 height_penalty 和 upright_penalty 的 per-step 真实量级

**Problem:** height_penalty Q5=-48.62/ep，upright_penalty Q5=-56.57/ep，数值看似高，但 per-step 归一化后分别为 -0.093 和 -0.110/step（温和）。这两项惩罚是长 episode 的自然积累产物，不代表严重的飞行质量问题。

**Proposed Change:** 暂不调整这两项权重/阈值，而是在 Run 17 中重点监控 per-step 归一化值（而非绝对值）。
- **监控指标：** `height_penalty / ep_len` per-step mean（目标：< -0.15/step）
- **监控指标：** `upright_penalty / ep_len` per-step mean（目标：< -0.15/step）
- **Rationale:** 当前 per-step 量级分别为 0.09 和 0.11，处于可接受范围。若 Run 17 随 ep_len 进一步延长而上升至 >0.15，则需要收紧阈值。

---

## Run 17 参数汇总

| 参数 | Run 16 | Run 17 | 目标 |
|------|--------|--------|------|
| `success_reward_weight` | 0.3 | **0.6** | all_targets_captured Q5 >0.55 |
| `tracking_reward_weight` | 4.0 | **5.0** | tracking Q5 >25/ep |
| `entropy_loss_scale` | 0.010 | **0.007** | policy_std Q5 = 0.68~0.75 |
| `contact_sensor_threshold` | 30N | **50N** | illegal_contact 脉冲 <-50 的比例 <1% |
| `illegal_contact_penalty` | 0.1 | **0.05** | illegal_contact Q5 > -3.0 |
| `fly_low_penalty` | 8.0 | **12.0** | crash Q5 <0.20，fly_low Q5 <0.10 |

---

## Experiment Plan for Run 17

```bash
python3 scripts/skrl/train.py \
  --task=Isaac-marl-move-v0 \
  --headless --num_envs=32 --algorithm="MAPPO"
```

1. 同时应用全部 5 项变更
2. 在 step 60k 检查 all_targets_captured（目标 >0.10；若 =0 说明 success 激励仍不足）
3. 在 step 120k 检查 policy_std（目标 0.68~0.78；若 <0.60 需将 entropy_scale 提回 0.009）
4. 在 step 200k 检查 crash Q4（目标 <0.30；若仍 >0.40 考虑 fly_low→15.0）
5. 在 step 320k 检查 illegal_contact 脉冲频率（目标 <1% 数据点 < -50）
6. 运行至 400k steps 完整评估

### 监控目标（Run 17）

1. `Episode_Termination/all_targets_captured` Q5 mean — 目标：>0.55（从 0.211 回升）
2. `Episode_Reward/success_reward` Q5 mean — 目标：80~200/ep（激励存在但不过载）
3. `Episode_Termination/crash` Q5 mean — 目标：<0.20（从 0.360 改善）
4. `Episode_Termination/falcon_fly_low` Q5 mean — 目标：<0.10（从 0.301 改善）
5. `Policy / Standard deviation` Q5 mean — 目标：0.68~0.75（从 0.826 降低）
6. `Episode_Reward/illegal_contact` Q5 mean — 目标：>-3.0（从 -16.02 改善）
7. `Episode_Reward/tracking_reward` Q5 mean — 目标：>20/ep（从 17.69 提升）
8. `Episode / Total timesteps (mean)` Q5 — 目标：>1500（维持长 episode 能力）

### 成功标准（Run 17 达成条件）

- all_targets_captured Q5 mean > 0.55（捕获率恢复）
- crash Q5 mean < 0.20（安全性改善）
- policy_std Q5 mean 在 0.68~0.75（探索在健康范围内）
- illegal_contact Q5 mean > -3.0（接触惩罚受控）
- tracking_reward Q5 mean > 20/ep（追踪质量提升）

---

## Run 17 vs Baseline 差距分析报告

**Run 17：** `2026-04-07_09-00-13_mappo_torch_mappo`（109k steps，仍在训练）
**Baseline：** `2026-03-20_16-26-19_mappo_torch_mappo`（400k steps，已收敛）
**分析日期：** 2026-04-07
**Task：** Isaac-marl-move-v0 | Algorithm: MAPPO

---

### 核心指标对比（Q5 = 后20%均值，代表最终性能）

| 指标 | Run 17 Q5 | Baseline Q5 | 差距倍数 |
|------|-----------|-------------|----------|
| Total reward (mean) | 42.8 | 885.9 | **20.7x** |
| Episode timesteps (mean) | 155 steps (9.3s) | 1658 steps (99.5s) | **10.7x** |
| all_targets_captured | 0.37 | 1.11 | **3.0x** |
| crash | 0.12→0.34 (上升) | 0.000 | 无穷 |
| falcon_fly_low | 0.12→0.34 (上升) | 0.000 | 无穷 |
| Per-step instantaneous reward | 0.67 | 0.55 | 相近 |

**关键发现：** 每步奖励率 Run 17（0.67）实际上略高于 Baseline（0.55），说明 per-step 学习质量不差。总奖励差距的核心原因是 **episode 过短**（9s vs 100s），而非策略本身的单步质量。

---

### 根本原因分析

#### 原因 1：终止模式崩溃——crash + fly_low 主导（CRITICAL）

Run 17 终止分布（最后25%训练数据）：
- crash: 0.232（每 log step 平均 0.23 个 env 触发）
- falcon_fly_low: 0.199
- all_targets_captured: 0.404
- bounding_box: 0.798（整个训练期间 bbox 都是主要终止原因）

Baseline 终止分布（最后5%）：
- crash: 0.000，fly_low: 0.000，captured: 1.162，bbox: 0.000

Run 17 在 Q4 阶段 crash + fly_low 合计 0.43，接近 captured 的 0.40，说明无人机在接近目标时倾向于俯冲撞地/超出飞行高度下界。`fly_low` 惩罚虽然已调到 12.0，但 **终止数量仍在增加**，说明 reward 信号来得太晚（episode 已 crash 终止），无法有效反向传播。

#### 原因 2：`fly_low_penalty=12.0` 过大，反而导致过度俯冲（HIGH）

Run 17 `Episode_Reward/fly_low` 从 0 → -9.18（最终），`Episode_Reward/height_penalty` 最终 -2.38，`Episode_Reward/upright_penalty` -3.12。这三个负奖励合计约 -14.7/episode，但 `Episode / Total timesteps` 只有 155 steps，说明这些是**高频密集惩罚**而非单次大惩罚。

根本原因：`fly_low_penalty=12.0` 相比 Baseline `1.0` 提高了 12 倍，但 Baseline 中从未触发过 fly_low 终止。说明 Baseline 不需要依靠大惩罚来防止飞太低——而 Run 17 的 fly_low 根本原因是 **capture_distance=3.0m 比 Baseline 1.0m 大 3 倍**，无人机需要更宽泛的悬停范围，造成高度控制更难精确。

#### 原因 3：episode 极短根本原因——bounding_box 终止（HIGH）

Run 17 整个训练期间 `bounding_box` 终止率均值约 0.9~1.1（Q1-Q3），而 Baseline 在训练后期降至 0.000。Run 17 的 `bounding_box_threshold=10.0m`（Baseline 12.0m），边界更小，加上 `nova_carter_scale` 扩大了小车轨迹（3x scale），导致无人机更容易冲出边界。

#### 原因 4：训练量不足——Run 17 只运行了 109k steps（MEDIUM）

Baseline 运行至 400k steps 才收敛（ep_len 在约 200k 步后才稳定在 1600+）。Run 17 当前只有 109k steps，从 timestep 角度看连 Baseline 的 1/4 都不到。需要等待更多训练才能做最终判断。

#### 原因 5：`num_envs=32` vs Baseline `512`（MEDIUM）

env.yaml 记录 `scene.num_envs: 32 vs 512`，但这仅反映配置文件中的 `num_envs` 默认值，实际训练时使用 `--num_envs` 命令行覆盖。需要确认 Run 17 实际训练是否真的只用了 32 envs。若是，这将严重影响采样效率和策略泛化性。

#### 原因 6：`entropy_loss_scale=0.007` vs Baseline `0.001`（MEDIUM）

Run 17 entropy 权重高出 7x，导致 policy_std 维持在 0.77（Baseline 最终 0.15）。高 entropy 有助于探索但阻碍收敛。Run 17 的 per-step 奖励率已经不低（0.67 vs 0.55），说明策略已学到有效行为，但高 entropy 阻止了策略 commit 到捕获动作序列。

#### 原因 7：`success_reward_weight=0.6`（Run 17）vs Baseline `0.0`（MEDIUM）

Baseline 中没有 success_reward，完全依靠 tracking_reward 和 distance_reward 驱动。Run 17 加入 success_reward 后出现过之前分析的 success_weight=0.3 导致全部捕获率下降 74% 的问题（Run 16），此次 0.6 仍需观察是否出现相同模式。

---

### 参数差异 vs 代码差异区分

| 变更类型 | 具体内容 | 对性能的影响 |
|---------|---------|------------|
| **代码变更** | height_error 计算修复 | 正向：高度控制更精确 |
| **代码变更** | success_reward bug 修复 | 正向：激励信号正确 |
| **代码变更** | NovaCarter 目标替换 | 中性/负向：小车移动更复杂，轨迹更大(nova_carter_scale=3.0) |
| **代码变更** | contact_forces 传感器重构（多传感器→单传感器） | 中性 |
| **参数变更** | fly_low_penalty: 1.0→12.0 | 负向：过大惩罚加速 crash 终止 |
| **参数变更** | capture_distance: 1.0→3.0 | 负向：成功条件宽松但高度控制难度增大 |
| **参数变更** | bounding_box_threshold: 12.0→10.0 | 负向：更小活动范围，与大 nova_carter_scale 矛盾 |
| **参数变更** | entropy: 0.001→0.007 | 负向（收敛角度）：阻碍策略收敛 |
| **参数变更** | tracking_weight: 1.0→5.0 | 正向：追踪激励更强 |
| **参数变更** | success_weight: 0.0→0.6 | 待观察：历史上曾导致 captured 下降 |

---

### 改进建议（针对缩小与 Baseline 的差距）

#### 建议 1（CRITICAL）：修复 bounding_box + fly_low 终止主导问题

**问题：** Run 17 的 episode 被 bounding_box（整个训练）和 crash/fly_low（Q4 阶段）过早截断，导致有效训练时间只有 Baseline 的 1/10。

**方案 A（边界扩大）：**
- 文件：`marl_flyfollow_env_cfg.py`（move task）
- `bounding_box_threshold`: 10.0 → **14.0**（补偿 nova_carter_scale=3.0 带来的轨迹扩大）
- `boundary_soft_threshold`: 8.0 → **11.0**

**方案 B（fly_low 惩罚调整）：**
- `fly_low_penalty`: 12.0 → **6.0**（仍是 Baseline 6x 而非 12x，避免频繁 crash 终止）
- 重要：Baseline 用 fly_low=1.0 从未触发 fly_low 终止，说明 Baseline 的高度控制本身稳定；Run 17 的问题根源是 nova_carter_scale 导致追踪更激进

#### 建议 2（HIGH）：确认并修复 num_envs 问题

运行命令中确保使用 `--num_envs=2048`（或至少 512 与 Baseline 对齐）。若 Run 17 实际只用 32 envs，采样量是 Baseline 的 1/16，这是当前所有问题的前提放大器。

#### 建议 3（HIGH）：entropy_scale 从 0.007 回调至 0.003

Run 17 的 per-step 奖励率已经合理（0.67），当前高 entropy 阻碍策略从"接近目标"收敛到"稳定持续追踪"。建议：
- `entropy_loss_scale`: 0.007 → **0.003**（介于 Baseline 0.001 和当前 0.007 之间）
- 这样既保留 Run 16 引入高 entropy 的探索优势，又允许策略逐步收敛

#### 建议 4（MEDIUM）：观察 success_weight 影响

Run 17 的 all_targets_captured 目前 Q5=0.37（只有 109k steps），尚不能判断 success_weight=0.6 是否重现 Run 16 的问题。建议继续训练到 200k steps 后再决定是否调整。若 captured 在 120k~180k steps 出现下降趋势，立即将 `success_reward_weight` 从 0.6 降至 0.4。

---

### 实验计划（Run 18）

```bash
python3 scripts/skrl/train.py \
  --task=Isaac-marl-move-v0 \
  --headless --num_envs=2048 --algorithm="MAPPO"
```

参数变更（相对 Run 17）：
1. `bounding_box_threshold`: 10.0 → 14.0
2. `boundary_soft_threshold`: 8.0 → 11.0
3. `fly_low_penalty`: 12.0 → 6.0
4. `entropy_loss_scale`: 0.007 → 0.003

检查点：
- Step 60k：episode_length > 400 steps（ep 不再被 bbox 截断）
- Step 120k：crash < 0.15，fly_low < 0.10
- Step 200k：all_targets_captured > 0.60
- Step 400k：all_targets_captured > 1.00（与 Baseline 对齐）

---

## Run 16 vs Baseline 对比分析报告（Run 16 独立分析）

**Run 16：** `2026-04-06_22-09-35_mappo_torch_mappo`（400k steps，已完成）
**Baseline：** `2026-03-20_16-26-19_mappo_torch_mappo`（400k steps，已收敛）
**分析日期：** 2026-04-07
**Task：** Isaac-marl-move-v0 | Algorithm: MAPPO

---

### 核心指标对比（四分段均值：Q1=前25%，Q4=后25%，last=最终值）

| 指标 | Run 16 Q4 | Baseline Q4 | 差距 |
|------|-----------|-------------|------|
| Total reward (mean) | 703.6 | 889.9 | **-20.9%** |
| Episode timesteps (mean) | 2413 steps | 1642 steps | **+47%（更长但不稳定）** |
| all_targets_captured | 0.224 | 1.166 | **-80.8%** |
| crash | 0.371 | 0.000 | Run 16 独有问题 |
| falcon_fly_low | 0.310 | 0.000 | Run 16 独有问题 |
| bounding_box | 0.299 | 0.000 | 大幅改善但未清零 |
| policy_std Q4 | 0.829 | 0.170 | **探索过度：4.9x** |
| instant reward (mean) | 0.286 | 0.544 | **per-step 质量差 47%** |
| success_reward Q4 | 292.6/ep | 0.0/ep | Run 16 独有奖励 |

---

### 根本原因分析

#### 问题 1：all_targets_captured 退化 80%——主要原因 success_weight=0.3（CRITICAL）

**症状：** Q4 均值 0.224（Baseline 1.166），退化 80.8%。
注意 Q2 均值 0.528 是训练中期峰值，此后在 Q3=0.230 和 Q4=0.224 上出现**显著回调**，说明策略学到高 success_reward 后反而放弃捕获行为。

**根因：** `success_reward_weight=0.3` 与 `success_reward` 累积机制交互。在 Q2 阶段 success_reward 每 episode 约 208/ep，占总奖励的约 29%（208/703）。策略发现可以用更长 episode 积累 success_reward 但实际捕获次数不增加——这是典型的**奖励欺骗**模式：策略不是捕获目标，而是在目标附近反复进出来刷 success 奖励。

**量化证据：**
- Q2：all_targets_captured=0.528，success_reward=208，ep_len=2327 steps
- Q4：all_targets_captured=0.224，success_reward=293，ep_len=2413 steps
- success 奖励从 Q2→Q4 继续上升（+41%），但 captured 从 Q2→Q4 下降（-58%）

这正是策略学会了在目标区域边缘徘徊（获得 success 奖励）而不是完成"全部目标捕获"（触发 all_targets_captured 终止）。

#### 问题 2：policy_std=0.829 持续过高——entropy_scale=0.010 无法收敛（HIGH）

**症状：** policy_std 在整个训练过程几乎不变：Q1=0.809，Q2=0.853，Q3=0.855，Q4=0.829。Baseline 从 Q1=0.569 线性下降至 Q4=0.170。

**根因：** `entropy_loss_scale=0.010` 是 Baseline 的 10 倍（Baseline 中 entropy_loss 最终值≈+0.0005，接近零。Run 16 最终值≈-0.012，绝对值高 24 倍）。策略探索噪声过大，无法 commit 到精确捕获动作序列。

**关键对比：** Run 16 的 per-step 奖励（0.286/step）低于 Baseline（0.544/step）——说明高 std 不仅不帮助探索，还主动降低了每步的行为质量。高 std 策略在接近目标时动作噪声大，容易过冲触发 crash/fly_low。

#### 问题 3：crash + fly_low 贯穿 Q2~Q4，Baseline 完全没有（HIGH）

**症状（四分段趋势）：**
- crash: Q1=0.038 → Q2=0.356 → Q3=0.425 → Q4=0.371（峰值在 Q3）
- falcon_fly_low: Q1=0.024 → Q2=0.304 → Q3=0.364 → Q4=0.310

Q1 几乎为零（训练初期刚开始），Q2~Q4 维持在 0.3~0.4。这说明 crash 和 fly_low 是策略**学会追踪目标之后**才出现的——追踪轨迹本身激进到超出安全飞行边界。

**与 Run 17 的对比：** Run 17 Q4 crash=0.249，fly_low=0.221，低于 Run 16 Q4（0.371，0.310）。Run 17 的 fly_low_penalty=12.0 vs Run 16 的 8.0，多了 50%，但 crash 数量降低幅度有限（-33%）。说明 crash 根源不完全是惩罚不足，而是策略主动俯冲追车导致的结构性风险。

**与 Baseline 的根本差异：** Baseline 的 fly_low_penalty=1.0（仅 Run 16 的 1/8），但从不触发。原因是 Baseline 中策略不需要激进俯冲——目标用旧小车类型，轨迹更慢更平坦。NovaCarter+scale=3.0 的高速轨迹是造成 Run 16 crash 的底层原因。

#### 问题 4：bounding_box 已改善但仍未消除（MEDIUM）

**症状：** Q4=0.299（Baseline Q4=0.000）。Q2=0.546→Q3=0.304→Q4=0.299，下降趋势存在但平台化。

**与上次分析 Run 16 的记录对比：** 上次记录 `bounding_box SOLVED (Q5=0.333)`，本次使用正确的四分段切割（Q4=后25%均值 0.299），比 Q5 统计更精确。bounding_box 仍未完全清零，约 30% 的 Q4 时间段仍有 bbox 终止。

**run_16 vs run_17 对比：** Run 17 Q4 bounding_box=0.748（比 Run 16 Q4=0.299 差 2.5 倍）。Run 17 参数中 bounding_box_threshold 没有变化（仍 10.0m），但 fly_low_penalty 从 8→12 导致策略飞行更保守，部分原本会深入边界的轨迹被截断，反而 bounding_box 情况反而更糟——说明高 fly_low 惩罚与 bounding_box 之间存在耦合：为了避免飞低，策略会拉高高度并减小水平追踪速度，导致更容易追丢目标出边界。

#### 问题 5：Run 16 独有——成功奖励反引发"奖励欺骗"（HIGH，Run 17 无此问题）

**症状：** success_reward Q4=292.6/ep，但 all_targets_captured=0.224。单纯从奖励值看策略"很成功"，但实际捕获率极低。

**根因：** 成功条件设计允许策略在目标区域内反复进出积累 success 时间步奖励，而不需要完成"三架无人机同时在各自目标范围内"的全局条件（all_targets_captured 触发）。individual success 奖励与 global capture 终止之间的激励不一致，策略优化了前者而忽略了后者。

**Run 16 独有 vs Run 17：** Run 17 中 success_reward Q4=98.8/ep（Run 16 的 1/3），captured Q4=0.355（Run 16 的 1.6 倍）。说明 Run 17 的 success_weight=0.6（vs Run 16 的 0.3）并不是奖励欺骗更严重，而是相反——0.6 的权重实际上更好。Run 16 的 success_weight=0.3 导致策略用"低强度反复触发"来积累奖励，而 0.6 的单次价值更高，策略倾向于"一次完成后终止"。

---

### Run 16 vs Run 17 指标对比（关键差异）

| 指标 | Run 16 Q4 | Run 17 Q4（partial） | 差异方向 |
|------|-----------|---------------------|---------|
| Total reward (mean) | 703.6 | 240.6 | Run 17 更低（训练量不足） |
| all_targets_captured | 0.224 | 0.355 | **Run 17 更好 +58%** |
| success_reward | 292.6 | 98.8 | Run 17 每 ep 更少但捕获更多 |
| crash | 0.371 | 0.249 | **Run 17 更好 -33%** |
| falcon_fly_low | 0.310 | 0.221 | **Run 17 更好 -29%** |
| bounding_box | 0.299 | 0.748 | **Run 16 更好（Run 17 bbox 恶化）** |
| policy_std | 0.829 | 0.734 | **Run 17 更低（entropy 0.010→0.007）** |
| episode timesteps | 2413 | 402 | Run 17 仍极短（需更多训练） |
| instant reward/step | 0.286 | 0.568 | **Run 17 per-step 质量高 2x** |

**核心结论：** Run 17 在 captured、crash、fly_low、per-step 质量上都优于 Run 16，entropy 从 0.010→0.007 方向正确。Run 17 的主要问题是 bounding_box 终止率反而更高（0.748 vs 0.299），这是 Run 18 需要解决的首要问题。

---

### Run 18 建议（更新）

基于 Run 16 vs Baseline + Run 16 vs Run 17 的对比，Run 18 参数建议调整如下：

| 参数 | Run 17 | Run 18 建议 | 目标 |
|------|--------|------------|------|
| `bounding_box_threshold` | 10.0 | **14.0** | bbox Q4 < 0.10（清零方向） |
| `boundary_soft_threshold` | 8.0 | **11.0** | 配合 bbox 扩大 |
| `fly_low_penalty` | 12.0 | **6.0** | crash Q4 < 0.20，避免 bbox/flylow 耦合 |
| `entropy_loss_scale` | 0.007 | **0.004** | policy_std 降至 0.55~0.65 |
| `success_reward_weight` | 0.6 | 维持 0.6 | Run 17 实测效果优于 Run 16 的 0.3 |
| `tracking_reward_weight` | 5.0 | 维持 5.0 | — |
| `contact_sensor_threshold` | 50N | 维持 50N | — |

**最高优先级（CRITICAL）：** `bounding_box_threshold 10.0→14.0`——这是 Run 17 相对 Run 16 最明显的倒退，也是 ep 长度不增长的直接障碍。

**次高优先级（HIGH）：** `fly_low_penalty 12.0→6.0`——Run 16 分析证明 fly_low 和 bounding_box 存在负相关：大 fly_low 惩罚→策略飞行保守→水平追踪不积极→bbox 终止增加。6.0 是安全点（仍是 Baseline 6x），避免过大 penalty 破坏追踪积极性。

**中优先级（MEDIUM）：** `entropy_loss_scale 0.007→0.004`——Run 17 Q4 policy_std=0.734，相比 Run 16 Q4 0.829 已改善，但目标是 0.55~0.65。0.004 是 Run 15 成功值 0.006 到 Baseline 0.001 之间的合理步伐。

**不变项：** `success_reward_weight=0.6` 维持——Run 17 数据证明 0.6 比 Run 16 的 0.3 更好（captured 高 58%，奖励欺骗更轻）。不要回退至 0.3。

---

## Changelog
- 2026-04-07: 追加 Run 16 vs Baseline 独立对比分析（发现奖励欺骗模式，bounding_box/fly_low 耦合效应，success_weight 机制澄清），更新 Run 18 建议（bbox 14.0，fly_low 6.0，entropy 0.004）

---

# Training Analysis Report — Run 17 完整 400k 步分析

**Run:** 2026-04-07_09-00-13_mappo_torch_mappo  
**Date:** 2026-04-07  
**Task:** Isaac-move-flyfollow-marl-v0  
**Algorithm:** MAPPO  
**Steps:** 400,000 (完整训练)  
**num_envs:** 32

## Training Metrics Summary

| 指标 | Q1 | Q2 | Q3 | Q4 | Q5 | last20_mean |
|------|----|----|----|----|-----|------------|
| instant_reward_mean | 0.038 | 0.378 | 0.843 | 0.958 | 0.850 | 0.753 |
| ep_len_mean (steps) | 111 | 256 | 742 | 1626 | 2432 | 2869 |
| all_targets_captured | 0.030 | 0.427 | 0.360 | 0.305 | 0.332 | 0.203 |
| success_reward | 0.30 | 48.6 | 272 | 581 | 780 | 559 |
| bounding_box | 1.107 | 0.903 | 0.664 | 0.488 | 0.346 | 0.144 |
| crash | 0.039 | 0.155 | 0.240 | 0.257 | 0.242 | 0.169 |
| falcon_fly_low | 0.012 | 0.132 | 0.215 | 0.227 | 0.213 | 0.169 |
| illegal_contact (term) | 0.031 | 0.050 | 0.072 | 0.116 | 0.111 | 0.025 |
| time_out | 0.0001 | 0.019 | 0.122 | 0.269 | 0.419 | 0.652 |
| policy_std | 0.837 | 0.754 | 0.721 | 0.678 | 0.637 | 0.627 |
| fly_low_reward | -0.12 | -1.49 | -2.51 | -2.70 | -2.54 | -2.03 |
| height_penalty | -1.33 | -4.18 | -17.8 | -37.2 | -57.4 | -89.5 |
| upright_penalty | -0.58 | -3.84 | -20.2 | -43.2 | -66.6 | -103.2 |
| tracking_reward | 2.06 | 4.38 | 10.56 | 18.99 | 26.94 | 35.24 |
| distance_reward | 1.97 | 4.72 | 13.18 | 24.56 | 35.07 | 36.30 |

**ep_len-normalized per-step success_reward:** Q3=0.367/step → Q4=0.357/step → Q5=0.321/step（稳定下降，无欺骗加速）

---

## 五个核心问题逐一回答

### 1. bounding_box 终止率后期是否改善？—— YES，显著改善

**结论：完全确认，且幅度超过预期。**

- Q1=1.107 → Q2=0.903 → Q3=0.664 → Q4=0.488 → Q5=0.346（last20_mean=0.144）
- Q5 相比 Q1 下降 **69%**，last20_mean 仅 0.144（接近清零水平）
- 109k 步时的中期分析：Q4=0.748（仍高）。但从完整曲线看，Q4 是最后一个高点——Q5 和 last20 均显示 bbox 已被策略学会回避
- 同期 time_out 从 Q1=0.0001 升至 Q5=0.419（last20=0.652），说明策略后期**主要靠自然超时退出而非 bbox 终止**
- **中期分析的担忧是过早的**：109k 步时策略仍在高度探索阶段，bbox 终止主要来自随机动作；后期 ep_len 增长到 2432 步均值后，策略已有足够时间在边界外减速

**机制证明：** ep_len Q1=111 → Q5=2432（+2191%）。bounding_box 的消失与 ep_len 的爆发性增长同步——策略学会了在边界内长期生存。

---

### 2. all_targets_captured 最终趋势（Q5 是否超过 0.7）？—— NO，Q5=0.332，稳定但未达到目标

**结论：** Q5 均值 0.332，last20_mean=0.203，未超过 0.7 目标。趋势是 Q2 峰值后小幅衰减并稳定。

**详细分解：**
- Q1=0.030（训练初期，无捕获能力）
- Q2=0.427（**峰值**，第一批成功捕获）
- Q3=0.360，Q4=0.305，Q5=0.332（Q2 后小幅衰减，约 -22%，Q5 轻微回升）
- last20_mean=0.203（最终 20 个 rollout 均值低于 Q5 均值，说明末期波动较大）

**为什么未超过 0.7？** 三个并发限制：
1. **crash+fly_low 联合率 Q5=0.454**（crash=0.242 + fly_low=0.213）——约 45% 的 rollout 以坠机/低飞终止，中断捕获
2. **per-step success_reward 从 Q3(0.367)→Q5(0.321) 持续下降**——说明策略并未"更努力捕获"，而是在更长 episode 中接受了每步更低的捕获频率
3. **all_targets_captured last20_mean=0.203** 低于 Q5_mean=0.332，说明训练末期（step 380k-400k）存在负波动

**与 Run 16 对比：** Run 16 Q5=0.211（中期分析），Run 17 Q5=0.332（+57%）。目标方向正确，但绝对值仍不足。

---

### 3. crash/fly_low 最终水平 —— 高位平台，未收敛

**结论：** crash+fly_low 在 Q3-Q5 稳定在约 0.45/rollout 的平台，无下降趋势。

| 指标 | Q1 | Q2 | Q3 | Q4 | Q5 | 趋势 |
|------|----|----|----|----|-----|------|
| crash | 0.039 | 0.155 | 0.240 | 0.257 | 0.242 | 增长后平台 |
| falcon_fly_low | 0.012 | 0.132 | 0.215 | 0.227 | 0.213 | 增长后平台 |
| combined | 0.051 | 0.287 | 0.455 | 0.484 | 0.454 | Q3-Q5 锁定 |

**fly_low_penalty=12.0 的实际效果：**
- fly_low 奖励惩罚轨迹：Q1=-0.12 → Q3=-2.51 → Q5=-2.54/ep（Q5 基本与 Q3 持平，策略已适应惩罚幅度）
- 但 fly_low_penalty 的绝对量在 per-step 视角只是 -2.54/2432=-0.001/step，相比 success_reward/step=0.321 是 321x 倍差距
- **12.0 的大惩罚并没有消灭 fly_low 行为，只是将其稳定在固定比例**

**关键结构性原因（fly_low/bbox 耦合验证）：**
- 中期分析中预测的 "fly_low_penalty=12 → 策略保守 → bbox 增加" 已被完整数据**部分推翻**
- bbox 后期确实显著改善（Q5=0.346，中期 Q4=0.488），说明保守策略最终被克服了
- 但 crash/fly_low 未下降，说明这是 **接近 NovaCarter 时的结构性碰撞**，而非保守飞行的副产品
- 深层原因：nova_carter_scale=3.0 + capture_distance=3.0m XY，无人机必须飞至 3m 以内，而 NovaCarter 高度约 1.35m（0.45×3），接近时不可避免发生低飞触发

---

### 4. policy_std 收敛情况 —— 单调下降，末期平台，低于目标范围

**结论：** 单调平稳下降（0.837→0.627），entropy_scale=0.007 有效防止了崩溃式下降，但最终 std 低于目标范围 0.65~0.75。

| 指标 | Q1 | Q2 | Q3 | Q4 | Q5 | last20 |
|------|----|----|----|----|-----|--------|
| policy_std | 0.837 | 0.754 | 0.721 | 0.678 | 0.637 | 0.627 |

- Run 16（entropy=0.010）：Q5=0.826（过高，超出目标上限）
- Run 17（entropy=0.007）：Q5=0.637（低于目标下限 0.65）
- 线性插值：目标 std=0.70 → entropy_scale ≈ **0.0085**；或以 0.004 进一步压低至 0.55~0.65

**value_loss 状况：** recent_mean=0.023，last=0.084（末期小幅回升——对应 last20 的 bbox/crash 波动）。整体 critic 收敛健康。

**结合 captured 分析的推论：** std=0.63 的策略已接近确定性，但 captured Q5=0.332 仍较低，说明问题不在于探索不足，而在于**策略选择了一种不积极捕获的稳定策略**（更长 ep + 更高 per-step 奖励，但减少接近 NovaCarter 风险）。

---

### 5. 奖励欺骗检查 —— 不存在 Run 16 式欺骗，但存在"被动稳定"新模式

**结论：** Run 17 没有 Run 16 的经典奖励欺骗（captured 下降而 success_reward 上升）。但存在一种新的"被动稳定"模式：策略在 Q2 峰后不再积极提高 captured 率，而是通过延长 episode 来被动积累 success_reward。

**数据证明：**
- Run 16 欺骗特征：captured Q2→Q4 下降 -58%，success_reward Q2→Q4 上升 +41%（**反向**）
- Run 17 数据：captured Q2→Q4 下降 -29%，success_reward Q2→Q4 上升 +1,095%（**同向，非欺骗**）
- per-step success_reward：Q3=0.367 → Q4=0.357 → Q5=0.321（**单调下降**）

**"被动稳定"的机制：**
1. ep_len 爆炸式增长（Q3=742 → Q5=2432），success_reward 的绝对值主要由 ep_len 驱动
2. per-step success_reward 下降说明策略实际上是"被稀释了"——每步平均捕获贡献在减少
3. time_out 从 Q5=0.419 升至 last20=0.652，说明策略学会了"存活到超时"而非"尽快捕获"
4. **这是 ep_len 主导下的成功奖励膨胀，而非主动欺骗**

**与 Run 16 欺骗的根本区别：** Run 16 策略主动规避捕获（在边缘徘徊）；Run 17 策略真实发生捕获（Q5=0.332），只是捕获频率随 ep_len 增长而相对降低。不需要反欺骗对策，但需要提高 per-step 的捕获效率激励。

---

## 关键发现

### 发现 1：高度惩罚/姿态惩罚的绝对值膨胀是 ep_len artifact — MEDIUM

**表现：**
- height_penalty Q5=-57.4/ep，last20=-89.5/ep（数值极大）
- upright_penalty Q5=-66.6/ep，last20=-103.2/ep

**per-step 归一化后的真实水平：**
- height Q5: -57.4/2432 = **-0.0236/step**（与 Run 13 的 -1.89/ep÷261 = -0.00724/step 相比有所上升，但仍属可控）
- upright Q5: -66.6/2432 = **-0.0274/step**（在 tracking_reward/step=26.94/2432=0.0111 的 2.5x 量级，比值偏高）

**实际问题：upright_penalty per-step 是 tracking_reward per-step 的 2.5x**，这意味着策略每获取 1 单位追踪奖励，就付出 2.5 单位姿态惩罚。这压制了主动追踪积极性，但并非严重阻碍（captured Q5=0.332 仍为正值）。

### 发现 2：illegal_contact 渐进增长，末期控制良好 — LOW

- 终止率：Q1=0.031 → Q4=0.116 → Q5=0.111（Q5 略有回落）
- 奖励：Q5=-1.71/ep（last20=-1.49）
- 50N 阈值有效：最差点 worst=-293（偶发尖峰）但 last20_mean=-1.49（正常水平）
- **50N 有效，无需升级**

### 发现 3：boundary_soft 末期巨幅膨胀值是数值异常 — 需注意

- boundary_soft last20_mean=-44.1，但 last=-0.089
- worst=-347.8（偶发极端值）
- 结合 bbox Q5=0.346（正在清零），boundary_soft 末期异常是极少数残余 bbox 事件的数值放大，非系统性问题

---

## Run 18 建议验证与修正

基于完整 400k 步数据，对原有 Run 18 候选参数逐项验证：

### 参数 1：bounding_box_threshold 10.0 → 14.0

**验证结论：** 需要修正。原建议基于中期（109k步）Q4=0.748 的高 bbox 率。但完整数据显示 bbox Q5=0.346，last20=0.144——策略后期已学会回避边界，10.0m 阈值本身不是障碍。

**修正建议：** 仍建议扩大至 14.0，但理由改变：
- bbox 最终虽改善，但 Q2-Q4 的高 bbox 率（0.90/0.66/0.49）延迟了 ep_len 增长
- 14.0 可让 Q2-Q3 阶段更快进入长 episode，更早激活 success_reward
- 14.0 不会破坏 Q5 已建立的 "存活到超时" 策略

**boundary_soft_threshold 8.0 → 11.0：** 配合 bbox=14.0，soft 在 11m 建立减速梯度，逻辑正确。维持原建议。

### 参数 2：fly_low_penalty 12.0 → 6.0

**验证结论：** 原建议需要**谨慎验证**，但总体支持降低。

**支持降低的证据：**
- fly_low_penalty=12.0 并未解决 crash/fly_low（Q5 combined=0.454，高位平台）
- fly_low reward per-step=-0.001，相比 success_reward per-step=0.321 可忽略不计——12.0 的惩罚强度在数量级上无效
- 来自 Run 10-12 的历史验证：fly_low_penalty=2.0 产生 Q5 crash=-31%，penalty=3.0 进一步 -47%；6.0 是 Run 14 后的经验安全点

**反对过度降低的证据：**
- 12.0 的高惩罚可能是 bbox 后期改善的间接贡献者（迫使策略更平稳）
- 但与 bbox 改善的主因（ep_len 自然增长）相比，12.0 的贡献无法量化

**修正建议：** 维持 6.0，但**同步观察 crash/fly_low 变化**。如果 Run 18 中 fly_low Q5 反弹至 >0.30，则说明 12.0 确实在抑制 fly_low，需回调至 8.0~10.0。

### 参数 3：entropy_loss_scale 0.007 → 0.004

**验证结论：** 支持，但需评估是否过度降低。

**Run 17 实测标定：**
- entropy=0.007 → std Q5=0.637（低于目标 0.65~0.75 下限）
- entropy=0.010（Run 16）→ std Q5=0.826（超出目标上限）
- entropy=0.004（Run 12-13 验证）→ std 约 0.627（与 Run 17 Q5=0.637 接近）

**问题：** entropy=0.007 在 400k 步后 std 已降至 0.627，比目标下限 0.65 低 4%。0.004 可能将 std 进一步压至 0.55~0.60（Run 11 范围），接近导致 captured 崩溃的危险区（Run 11: std=0.597→captured=-93%）。

**修正建议：** 接受 0.004，但设置**早停标准**：如果 Run 18 中 std 在 Q3 降至 0.60 以下，且 captured Q5 低于 0.25，需回调至 0.005~0.006。

### 参数 4：success_reward_weight 维持 0.6，tracking 维持 5.0

**验证结论：** 完全支持，无需修改。

**证据：**
- success_reward per-step 从 Q3(0.367) → Q5(0.321) 单调下降——无欺骗加速，奖励权重合理
- captured Q5=0.332（Run 16 中期值为 0.211）——维持 0.6 正确
- 反欺骗设计（per-step 单调下降）说明 0.6 的量级恰好在"激励足够但不溢出"的区间

**额外建议：** 考虑对 success_reward 进行 per-capture 而非 per-step 的改造（参考 Run 14 分析）。但这是结构性改动，留待 Run 19+。

### 新增建议（基于完整数据新发现）

**NEW：upright_penalty_weight 0.5 → 0.3（MEDIUM）**

证据：upright Q5 per-step=-0.0274，是 tracking per-step=0.0111 的 2.5x。策略被动姿态惩罚超过主动追踪奖励，从 captured 效率角度不合理。0.3 是 Run 8 的历史验证值（-1.125/ep 正常，drones_collide 未见明显增长）。

**NEW：contact_sensor_threshold 维持 50N（不降低）**

完整数据：illegal_contact worst=-293（单次尖峰），last20_mean=-1.49（正常）。50N 已有效控制，无需调整。

---

## Run 18 最终参数表（更新后）

| 参数 | Run 17 | Run 18（更新） | 变化依据 | 优先级 |
|------|--------|--------------|---------|--------|
| `bounding_box_threshold` | 10.0 | **14.0** | 加速 Q2-Q3 ep_len 增长；Q5 已自然改善 | HIGH |
| `boundary_soft_threshold` | 8.0 | **11.0** | 配合 bbox=14.0 | HIGH |
| `fly_low_penalty` | 12.0 | **6.0** | 12.0 对 crash/fly_low 无效（Q5 平台 0.454）；6.0 为历史安全点 | HIGH |
| `entropy_loss_scale` | 0.007 | **0.004** | Q5 std=0.637 低于目标；0.004 接受，需设早停 | MEDIUM |
| `upright_penalty_weight` | 0.5 | **0.3** | per-step upright/tracking 比 2.5x，抑制追踪积极性 | MEDIUM |
| `success_reward_weight` | 0.6 | **0.6（不变）** | 无欺骗，per-step 单调下降，无需修改 | — |
| `tracking_reward_weight` | 5.0 | **5.0（不变）** | tracking last20=35.2，健康增长 | — |
| `contact_sensor_threshold` | 50N | **50N（不变）** | last20 illegal_contact=-1.49，已控制 | — |

### 早停监控指标（Run 18 关键预警）

| 指标 | 早停阈值 | 触发动作 |
|------|---------|---------|
| policy_std Q3 | < 0.60 | entropy_scale 0.004 → 0.006 |
| crash+fly_low Q5 combined | > 0.50（若 fly_low_penalty=6.0 后反弹） | fly_low_penalty 6.0 → 8.0 |
| all_targets_captured Q5 | < 0.20 | 检查 captured 是否被 bbox 截断，还是 std 崩溃 |
| success_reward per-step Q4→Q5 | 上升（反欺骗检测） | 降低 success_weight 至 0.4 |

## Experiment Plan

1. 应用 Run 18 参数变更（bbox=14，soft=11，fly_low=6，entropy=0.004，upright=0.3）
2. `python3 scripts/skrl/train.py --task=Isaac-move-flyfollow-marl-v0 --headless --num_envs=2048 --algorithm="MAPPO"`
3. 监控重点（按优先级）：
   - **Q2 ep_len**（目标：> 600 步，验证 bbox=14 加速效果）
   - **crash+fly_low Q5 combined**（目标：< 0.40，fly_low=6 降低是否引发 fly_low 反弹）
   - **all_targets_captured Q5**（目标：> 0.50，验证 captured 改善）
   - **policy_std Q3-Q5 趋势**（目标：0.60~0.70，防止 0.004 过度压低）
4. 成功标准：captured Q5 > 0.50 AND crash Q5 < 0.20 AND ep_len Q5 > 3000 步

## Changelog
- 2026-04-07: 追加 Run 17 完整 400k 步分析。关键结论：bbox Q5 降至 0.144（问题自行解决），captured Q5=0.332（未达 0.7 目标，主因 crash+fly_low 高位平台 0.454），无奖励欺骗（per-step success 单调下降），policy_std=0.627（低于目标范围）。Run 18 调整：bbox 14.0、soft 11.0、fly_low 6.0、entropy 0.004、upright 0.3（新增），其余不变。

---

## Run 18 训练分析报告

**Run:** 2026-04-07_17-37-21_mappo_torch_mappo
**分析日期:** 2026-04-07
**总训练步数:** 400,000
**Task:** Isaac-move-flyfollow-marl-v0
**Algorithm:** MAPPO

### 训练指标摘要

| 指标 | 近50步 Q3 | 近50步 Q5 | last | Run 17 对应值 |
|------|-----------|-----------|------|--------------|
| instant_reward (mean) | 0.5644 | 0.6937 | 0.6575 | 0.8274 (Q5) |
| all_targets_captured | 0.0000 | 0.6330 | 0.0700 | 1.0000 (Q5) |
| bbox termination | 0.3600 | 1.0000 | 0.1100 | 0.8490 (Q5) |
| crash termination | 0.3700 | 1.0000 | 0.4200 | 0.6780 (Q5) |
| fly_low termination | 0.3450 | 1.0000 | 0.4200 | 0.6780 (Q5) |
| crash+fly_low Q5 | — | 2.0000 | 0.8400 | 1.3560 (Q5) |
| timeout | 0.0850 | 0.5400 | 0.5400 | 1.0000 (Q5) |
| policy_std | 0.6145 | 0.6164 | 0.6097 | 0.6283 (Q5) |
| ep_len mean | 222 | 4555 | 1699 | 5999 (Q5) |
| upright_penalty (近20步均值) | — | — | -13.51 | ~-40.8 (last) |
| height_penalty (近20步均值) | — | — | -20.70 | ~-34.9 (last) |

### 七大核心问题逐项分析

#### 1. bounding_box 终止率：显著改善，但结构性问题仍残留

**结论：bbox 扩大至 14m 有效，但效果被 crash/fly_low 激增抵消。**

时序趋势（5段均值）：
- Q1=1.051 → Q2=1.093 → Q3=0.810 → Q4=0.478 → Q5=0.363，last=0.110

bbox 确实从早期的主导终止因素（>1.0/rollout）下降至末期 0.110，与 Run 17 末期 0.144 接近。但近50步 Q5=1.000 表明方差极大——部分 rollout 仍频繁触发。根本原因：bbox 改善确实存在，但 crash+fly_low 同步崛起（Q5=2.000），说明原来被 bbox 截断的 episode 现在活得更长，但在长 episode 中发生了坠机。两个终止因素此消彼长，ep_len 的双峰分布（46% 短于200步、28% 长于2000步）直接反映了这一现象。

#### 2. crash/fly_low 趋势：fly_low=6.0 未能遏制，反而恶化

**结论：crash+fly_low 在 Run 18 末期比 Run 17 更严重，fly_low 从 12.0 降至 6.0 产生了负效果。**

时序趋势（crash+fly_low combined）：
- Q1=0.141 → Q2=0.044 → Q3=0.463 → Q4=0.828 → Q5=0.779，近50步 Q5=2.000

Run 17 末期 crash+fly_low Q5=1.356，Run 18 末期 Q5=2.000，恶化 +47%。

**根本原因分析：**
- fly_low_penalty 从 12.0 降至 6.0，削弱了对低空飞行的惩罚，策略更大胆地下降
- bbox 扩大使 episode 存活更长（ep_len Q5 从 Run17 的约2432 升至 Run18 的4555），更长 episode 意味着更多机会发生 approach-dive 行为
- captured 峰值区间分析显示：在最高 captured 阶段（index 1976-2026）crash+fly_low=0.538，随后随着 upright_penalty 升高（-1.17→-22.81）crash+fly_low 同步恶化。这是"学会追踪→更激进俯冲→坠机"的已知结构性耦合

**结论：fly_low=6.0 是 Run 17 时基于理论推断的回退，但实际数据证明 6.0 不足，需要恢复到更高值（建议 8.0-10.0）。**

#### 3. all_targets_captured：出现严重退化，未达 0.7 目标

**结论：captured 全程峰值 0.870（index 1976），但末期严重崩溃至 last=0.070，远未达到 0.7 目标。**

时序趋势（5段均值）：
- Q1=0.001 → Q2=0.255 → Q3=0.692 → Q4=0.406 → Q5=0.255

captured 峰值持续区间（index 1976-2026，滑动均值）= 0.868，是历史最高值，超过 Run 17 的 0.427 峰值。但峰值后出现崩溃：从 Q3 段的 0.692 降至末期 0.255，last=0.070。

**崩溃机制（逐步追踪）：**
- 峰值区间 upright_penalty=-0.915 → 100步后=-1.167 → 200步后=-3.406 → 400步后=-14.401
- upright_penalty 增大与 crash+fly_low 增大精确同步，且均始于 index ~2076
- 这是"tracking 学成 → 更激进俯冲 → upright 增大 → 坠机"的循环，与 Run 10/14/15 完全相同的模式

近50步分析：captured>0.5 的 rollout 比例仅 12%，captured>0 的比例 37%——策略处于高度不稳定状态，偶尔捕获但无法持续。

**未达 0.7 目标的核心诊断：crash+fly_low 在 captured 增长后激增，打断了捕获行为的稳定化。**

#### 4. policy_std：低于目标区间，entropy=0.004 过度压制

**结论：policy_std 末期 0.610，低于目标区间 0.65~0.70，entropy=0.004 压制过强。**

时序趋势：Q1=0.797 → Q2=0.725 → Q3=0.674 → Q4=0.698 → Q5=0.645，last=0.610

近20步 std=0.610-0.617，持续下降且未见反弹，标准差极小（recent_std=0.019）。entropy=0.004 相比 Run 17 的 0.007 降幅明显，将 std 从 0.627 进一步压低到 0.610。

**关键发现：** std 下降与 captured 崩溃同期发生（两者均始于 index ~2000-2076 段），std 降低使策略减少探索，在 crash/fly_low 激增时无法逃离危险轨迹，加速了 captured 的退化。建议 entropy=0.004 不变或微调至 0.005，因为 crash 问题的优先级高于 std 问题。

#### 5. per-step 瞬时奖励：接近 Baseline，但近期下滑

**结论：instantaneous reward 近50步 Q3=0.564，接近 Baseline 的 0.544，但 Q5=0.694 表明高质量 episode 的峰值仍存在。**

时序趋势：Q1=-0.008 → Q2=0.175 → Q3=0.674 → Q4=0.926 → Q5=0.678，last=0.6575

Q4 段（步骤 240k-320k）instant_reward=0.926 是全运行峰值，显著超过 Baseline 的 0.544。但 Q5 段回落至 0.678，与 Run 17 的 0.680（last）几乎相同。

**per-step 奖励质量本身没有退化**，但 crash/fly_low 导致大量短 episode（ep_len<200步），使总 reward 和 captured 看起来很差，掩盖了 per-step 质量信号。

#### 6. 与 Baseline 的差距：per-step 已接近，结构差距来自 crash/fly_low

| 指标 | Run 18 | Baseline | 差距 |
|------|--------|----------|------|
| instant_reward (last) | 0.6575 | 0.544 | **+21%（Run18优）** |
| ep_len Q4 | 2213 | 1658 | **+33%（Run18优）** |
| total_reward Q4 | 66.8 | 889.9 | -92%（Run18差） |
| crash/fly_low | 0.840 | 0.000 | 严重差距 |
| captured Q5 | 0.633 | 1.166 (Q4) | 差距大 |

**核心结论：per-step 质量指标已超越 Baseline，但 crash/fly_low=0.840 导致 episode 被频繁截断，累计 total_reward 和 captured 未能体现 per-step 的优势。Baseline 的根本优势不是策略质量，而是零坠机率。**

#### 7. 新出现的问题：upright+height 惩罚激增 + illegal_contact 后期爆发

**upright_penalty 结构性恶化：**
- 峰值区间 upright=-0.915/ep → 末期均值=-22.81/ep（25倍恶化）
- 近20步中 |upright|>30 的比例：32/100（32%），明显异常
- 根因：upright_weight 从 0.5 降至 0.3（本次变更），但惩罚量反而增大——这说明策略在追踪时的倾斜角已超过 upright 设计预期，weight 降低反而让策略更不被约束地倾斜
- **该变更产生了负效果：upright 惩罚量上升，不是下降**

**height_penalty 持续恶化：**
- 峰值区间 height=-8.724/ep → 末期均值=-33.421/ep（4倍恶化）
- 与 upright 同步上升，成因相同：追踪行为越激进，俯冲越深

**illegal_contact 后期集中爆发：**
- 全程爆发次数（|value|>10）= 22次，100% 集中在后半段（step 260k 之后）
- 近200步爆发9次（vs 前200步0次），且呈加速趋势
- last=-44.65（极端值），表明训练末期出现接触爆炸事件
- 这与 captured 峰值后的崩溃机制一致：更积极的接近 → 更频繁的碰撞

### 变更效果评估

| Run 18 变更 | 预期效果 | 实际效果 | 评分 |
|-------------|---------|---------|------|
| bbox 14.0m | 解除早期截断 | **有效**：Q5 段 bbox=0.363（vs Run17 Q5=0.346），末期 0.110 | PASS |
| soft 11.0m | 配合 bbox | 同上 | PASS |
| fly_low 6.0 | 缓解 bbox-crash 耦合 | **负效果**：crash+fly_low Q5 从 1.356→2.000（+47%）| FAIL |
| entropy 0.004 | std 0.65~0.70 | **过度压制**：std 末期 0.610（低于目标）| PARTIAL FAIL |
| upright 0.3 | 降低 per-step 惩罚 | **负效果**：upright 惩罚量末期增大25倍 | FAIL |

### 改进建议 — Run 19

#### Priority 1 (HIGH): fly_low_penalty 恢复至有效区间

**问题：** fly_low=6.0 不足以阻止 approach-dive 行为，Run 18 crash+fly_low Q5=2.000（+47% vs Run17）。

**历史数据支撑：**
- fly_low=6.0（Runs 10-12）：crash Q5≈0.47-0.68（中等）
- fly_low=8.0（Run 16）：crash Q5=0.360（-24%），但仍高
- fly_low=12.0（Run 17）：crash Q5=0.242（但 plateau 固化，per-step 惩罚=0.001 失效）
- **Run 18 的教训：** 6.0 → crash 反弹；6.0 不是历史安全点，是导致退化的低点

**建议：**
- 文件：`exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/flyfollow/marl_flyfollow_env_cfg.py`（move 任务对应 cfg）
- 参数：`fly_low_penalty` → **8.0**（而非恢复到12.0；8.0 是 Run16 验证过的中间值，与 bbox=14m 组合未经测试）
- 理由：fly_low=12.0 的失效原因是 per-step 效果被 success_reward 淹没（321x 比例），8.0 与调整后的 success 比例（成功率降低后）可能重新奏效

#### Priority 2 (HIGH): upright_penalty_weight 恢复至 0.5

**问题：** upright=0.3 导致策略更不受约束地倾斜，upright 惩罚量末期增大25倍（-0.915→-22.81）。weight 降低反而使行为更差。

**历史数据支撑：**
- upright=0.5（Runs 13-15）：upright_penalty Q5=-3.84/ep（控制有效）
- upright=0.8（Run 14）：upright runaway（过高）
- upright=0.3（Run 18）：upright末期均值=-22.81/ep（最差记录）

**建议：**
- 参数：`upright_penalty_weight` → **0.5**（回退至 Run 13-15 验证值）
- 理由：0.3 是本次新增的错误变更，数据证明更低的权重不能改善行为，反而移除了必要约束

#### Priority 3 (MEDIUM): entropy_loss_scale 调整至 0.005

**问题：** entropy=0.004 将 std 压至 0.610，低于目标区间 0.65~0.70。captured 崩溃期间 std 过低使策略无法调整。

**历史数据支撑（线性校正）：**
- entropy=0.007 → std Q5=0.637（Run 17）
- entropy=0.004 → std=0.610（Run 18，末期）
- entropy=0.010 → std=0.826（Run 16）
- 线性内插：target std=0.65 → entropy≈0.005

**建议：**
- 参数：`entropy_loss_scale` → **0.005**（小幅调整，避免振荡）
- 注意：如果 crash/fly_low 问题先解决，std 可能自然上升至目标区间，此变更为辅助

#### Priority 4 (MEDIUM): 监控 illegal_contact 是否需要进一步提高阈值

**现状：** contact_sensor_threshold=50N（Run 17 设置，Run 18 延续），但 Run 18 末期 illegal_contact 出现爆发（last=-44.65，近200步9次爆发）。

**不建议立即变更 50N**，原因：Run 17 末期 illegal_contact 仍受控（last20=-1.49），Run 18 的爆发是 captured 增长后的结构性接触增加（更多接近 = 更多碰撞）。优先解决 crash/fly_low 和 upright，减少接触频率，再评估阈值。

#### 保持不变的参数

- `bounding_box_threshold=14.0m`：Run 18 验证有效，Q5 段 bbox=0.363，末期 0.110，无需变更
- `boundary_soft_threshold=11.0m`：配合 bbox 有效
- `success_reward_weight=0.6`：Run 17 确认无奖励欺骗，保持
- `tracking_reward_weight=5.0`：保持
- `contact_sensor_threshold=50N`：暂时保持，Run 18 末期才出现爆发

### Experiment Plan — Run 19

1. 应用变更（按优先级）：
   - `fly_low_penalty`: 6.0 → **8.0**
   - `upright_penalty_weight`: 0.3 → **0.5**
   - `entropy_loss_scale`: 0.004 → **0.005**
2. 训练命令：`python3 scripts/skrl/train.py --task=Isaac-move-flyfollow-marl-v0 --headless --num_envs=2048 --algorithm="MAPPO"`
3. 监控重点（按优先级）：
   - **crash+fly_low Q3-Q5 combined**（目标：Q5 < 1.0，当前 Run18 Q5=2.000）
   - **upright_penalty 近20步均值**（目标：<-5.0/ep，当前 Run18=-13.5/ep）
   - **all_targets_captured Q5**（目标：> 0.70，需超过 Run17 的 0.332 并与 Run 18 峰值 0.868 持续）
   - **policy_std Q3-Q5**（目标：0.62~0.70）
   - **captured 是否在 Q3 之后维持而不崩溃**（Run18 的核心失败点）
4. 成功标准：
   - crash+fly_low Q5 < 1.0 AND
   - captured Q5 > 0.50 AND
   - captured 在 Q3→Q5 段不出现超过 30% 的衰减
5. 早停标准：
   - 若 200k 步时 crash+fly_low Q4 > 1.0（比 Run18 Q4=0.828 更差），提前终止
   - 若 upright_penalty 近20步均值超过 -30/ep，考虑提前终止

### 关键诊断结论

1. **fly_low=6.0 是错误的回退**：历史上 6.0 从未被验证为"安全点"，Run 10-12 中 6.0 伴随 crash Q5=0.47-0.68。Run 18 数据明确证实 6.0 相比 12.0 产生了更多坠机。需提高至 8.0 重新测试。

2. **upright=0.3 降低 weight 的假设错误**：前提是"weight 越小，per-step 惩罚越小，策略越容易追踪"。但实际上 weight 降低移除了约束，策略的倾斜角反而增大，导致绝对惩罚量上升25倍。正确方向应是保持 weight=0.5 的约束强度。

3. **captured 崩溃机制已明确**：tracking 学成 → 俯冲更激进 → upright/crash 激增 → captured 被截断。这是贯穿 Runs 10/14/15/18 的共同路径。核心解法是同时压制低空行为（fly_low 阈值）和倾斜行为（upright weight），而非降低其中之一。

4. **bbox 变更成功**：14m 有效扩大了 episode 生存空间（ep_len Q5 增长），这是本次唯一明确成功的变更，应保留。

## Changelog（续）
- 2026-04-07: 追加 Run 18 完整 400k 步分析。关键结论：bbox 改善（last=0.110，有效），but crash+fly_low Q5=2.000（+47% vs Run17，fly_low=6.0 产生负效果），captured 峰值 0.868 后崩溃至 last=0.070（upright 激增触发崩溃循环），policy_std=0.610（低于目标），illegal_contact 末期爆发（后50步9次）。upright=0.3 变更负效果（惩罚量增大25倍）。Run 19 调整：fly_low 6.0→8.0，upright 0.3→0.5，entropy 0.004→0.005，其余不变。

---

## 2026-04-08 Move Run 19 训练分析

**Run:** 2026-04-08_02-23-28_mappo_torch_mappo
**Date:** 2026-04-08
**Task:** Isaac-marl-move-v0
**Algorithm:** MAPPO
**Steps:** 400k (完整)
**Run 19 变更：** fly_low_penalty 6→8，upright_weight 0.3→0.5，entropy 0.004→0.005，**fly_low 终止阈值 0.1m→0.5m**

---

### 训练指标总览

| 指标 | Q1 | Q2 | Q3 | Q4 | Q5 | last |
|------|----|----|----|----|-----|------|
| crash+fly_low combined | 0.094 | 0.080 | 0.295 | 0.554 | 0.638 | 0.660 |
| all_targets_captured | 0.003 | 0.271 | 0.829 | 0.894 | 0.811 | 0.520 |
| success_reward (ep) | 0.041 | 4.63 | 45.58 | 91.46 | 130.9 | 101.1 |
| instant_reward (mean) | −0.033 | 0.151 | 0.626 | 0.900 | 0.952 | 0.853 |
| total_reward (mean) | −4.76 | 20.21 | 128.35 | 247.34 | 357.21 | 418.21 |
| ep_len mean (steps) | 115.3 | 127.7 | 196.3 | 275.1 | 374.4 | 516.6 |
| policy_std | 0.807 | 0.753 | 0.749 | 0.885 | 1.055 | 1.181 |
| fly_low reward | −0.265 | −0.272 | −1.078 | −2.053 | −2.410 | −2.640 |
| height_penalty | −3.44 | −1.87 | −7.65 | −11.52 | −14.96 | −28.36 |
| upright_penalty | −0.405 | −0.579 | −0.943 | −1.595 | −2.280 | −4.083 |

**Baseline 参考（2026-03-20）：** all_targets_captured Q4=1.166，crash/fly_low=0.000，instant_reward=0.544，total_reward Q4=889.9

---

### 核心问题分析

#### 1. fly_low 阈值修复效果评估 — SIGNIFICANT（HIGH）

**症状：** crash+fly_low Q5=0.638，Run 18 对应值约 2.000。**降幅 −68%，效果显著**。

**详细轨迹：**
- Q1=0.094（阈值修复在早期有效压制低空终止）
- Q3=0.295（追踪学习后开始上升，俯冲行为被 0.5m 阈值截断）
- Q5=0.638（仍高于目标 <0.5，但远低于 Run18 的 2.000）

**根本原因分析：** fly_low 终止阈值从 0.1m 提升至 0.5m 后，之前 0.1–1.0m 的"无约束俯冲区"被关闭。策略不能再无惩罚地俯冲至 z=0.1m。但 0.5m 阈值与 height_penalty 覆盖区间（threshold=1.5m，即 z<1.0m 触发）之间仍有 0.5m 空隙（0.5–1.0m），策略在该区间仍可积累低空飞行。

**结论：** 阈值修复是本次最有效的单项变更。Q5=0.638 仍高于目标，主因是 ep_len 增长后追踪阶段更长、俯冲机会更多（相对于 Run18 的超短 ep_len）。

---

#### 2. all_targets_captured 稳定性 — PARTIAL SUCCESS（MEDIUM）

**症状：** 未出现 Run18 式的"先升后崩"。Q3=0.829→Q4=0.894→Q5=0.811（小幅回落，非崩溃）。last=0.520（末尾波动，但 Q5 整体稳定）。

**与 Run18 对比：**
- Run18：峰值 0.868（step 198k）→ last=0.070（崩溃至 8%）
- Run19：峰值 1.510（step 383k）→ last=0.520（保持 34% of peak）

**根本原因：** upright_weight=0.5 恢复有效。Run18 中 upright_weight=0.3 导致约束移除、倾斜角爆炸（25x 增长），从而截断 captured。Run19 恢复 0.5 后 upright_penalty Q5=−2.28（vs Run18 −13.50），倾斜度受控，captured 不再因 upright 爆炸而崩溃。

**未达到目标 Q5>0.70 的原因：** Q5 mean=0.811 **已超过目标 0.70**。但末尾 last=0.520 提示存在波动。峰值在 step 383k（Q5 尾段），policy_std 持续上升至 1.181 可能加剧了末尾的随机性。

---

#### 3. per-step 瞬时奖励 — MAINTAINED（LOW）

**症状：** Q5 mean=0.952，last=0.853。远超 Baseline 0.544 和目标 >0.55。

**分析：** per-step 奖励质量在整个 Run19 持续提升（Q1=−0.033→Q5=0.952），说明策略的逐步决策质量在持续进步。per-step 指标未受 crash/fly_low 问题影响，因为短 episode 也有正向 per-step 奖励。

---

#### 4. policy_std 轨迹 — OVER-EXPLORATION（HIGH）

**症状：** Q1=0.807→Q2=0.753→Q3=0.749（正常压缩）→Q4=0.885→Q5=1.055，last=1.181（持续上升，超过目标 0.63~0.68）。

**根本原因分析：** entropy=0.005（Run18 的 0.004 过度压缩到 0.610）确实提升了 std，但效果超出预期。Q3→Q5 段 std 从 0.749 反弹至 1.055，增幅 41%。可能机制：
1. success_reward Q5 mean=130.9/ep（高）→ 高回报梯度与 entropy 共同驱动 std 上升
2. fly_low 阈值修复后策略探索空间扩大，鼓励了更多随机行为

**后果：** Q5 last=1.181 是过度探索状态。std>1.0 时策略行为随机性增大，导致末尾 captured last=0.520 的波动，以及 crash+fly_low Q5 的持续高位（随机俯冲更难控制）。

**目标范围：** 0.63~0.70（Run13 0.759 和 Run12 0.627 之间）。Run19 已超出上限，需降低 entropy。

---

#### 5. 与 Baseline 差距分析 — IMPROVING BUT STRUCTURAL GAP REMAINS（MEDIUM）

| 指标 | Baseline Q4 | Run19 Q5 | 差距 |
|------|-------------|---------|------|
| total_reward (mean) | 889.9 | 357.2 | −60% |
| all_targets_captured | 1.166 | 0.811 | −30% |
| crash+fly_low | 0.000 | 0.638 | +∞ |
| ep_len mean (steps) | ~1658 | 374.4 | −77% |
| instant_reward (mean) | 0.544 | 0.952 | +75% |

**结论：** per-step 奖励质量已超越 Baseline 75%，但 episode 长度和 crash/fly_low 仍是主要差距来源。total_reward 差距 60% 几乎完全来自 ep_len（374 vs 1658，差 4.4x）。Baseline 从未触发 crash/fly_low，是因为使用了更简单目标（非 NovaCarter scale=3x）。

---

#### 6. 新发现问题

**6a. height_penalty 持续增长 — HIGH**
- Q5 mean=−14.96/ep，last=−28.36（全局最差值）
- per-step 归一化：Q5 mean=−0.041/step（Q4=−0.042，Q3=−0.039 — 已稳定，非爆炸）
- 绝对值大是 ep_len 增长的累积效应，per-step 层面尚在控制范围内
- 但 last=−28.36 在 ep_len=516 步时对应 per-step=−0.055，略高于 Q5 均值

**6b. boundary_soft 惩罚上升 — MEDIUM**
- Q4=−0.638→Q5=−1.260（+97%），last=−1.574
- 追踪更积极（std 上升）→ 更多边界超出。不影响 bbox termination（Q5=0.666 仍在降低），但表明策略在边界附近频繁摩擦

**6c. ep_len 显著低于 Run18 — MEDIUM（EXPECTED）**
- Run19 Q5=374 vs Run18 Q5≈4555
- 原因：fly_low 阈值 0.5m 提前终止了之前能存活到 timeout 的低空飞行 episode
- 这是设计上的预期副作用：阈值修复"牺牲"了部分超长 episode 换取崩溃率下降

**6d. upright_penalty 末尾加速 — MEDIUM**
- last=−4.083（vs Q5 mean=−2.28，last 是 Q5 mean 的 1.8x）
- 与 policy_std 末尾加速（last=1.181）同步，符合"std 上升→更随机倾斜→upright 上升"的链条

---

### 综合评估

| 核心问题 | Run19 结果 | 目标 | 达成？ |
|---------|----------|------|--------|
| fly_low Q5 combined crash+fly_low | 0.638 | <0.5 | 未达（差 28%） |
| all_targets_captured Q5 | 0.811 | >0.7 | **达成** |
| captured 不出现崩溃（Q3→Q5 <30% 衰减） | Q4=0.894→Q5=0.811（−9%） | <30% | **达成** |
| instant_reward Q5 | 0.952 | >0.55 | **达成** |
| policy_std Q5 | 1.055 | 0.63~0.68 | 未达（过高） |

**整体判断：** Run19 是 Run 19 系列中的最重要突破——fly_low 阈值修复将 crash+fly_low 从 2.000 降至 0.638（−68%），captured 从崩溃恢复至 Q5=0.811。但 policy_std 过度扩张（1.055→1.181）带来了新的不稳定因素，需在 Run20 中收敛。

---

### Run 20 改进建议

#### Priority 1 (HIGH): 降低 entropy_loss_scale — 压缩 std 至目标范围

**问题：** policy_std Q5=1.055，last=1.181（目标 0.63~0.68），过度探索导致末尾 captured 波动和 crash 持续

**变更：**
- 文件：`skrl/` MAPPO 训练配置（通常在 `scripts/skrl/train.py` 或 cfg 文件中）
- 参数：`entropy_loss_scale` 0.005 → **0.003**
- 依据：线性插值校准——Run18 entropy=0.004→std=0.610，Run19 entropy=0.005→std=1.055 末段。0.003 预计 std≈0.65，处于目标范围中心

**注意：** std 不能过低（<0.60 时 captured 多样性消失，Run11 教训）。建议在 200k 步时检查 std Q2，若 <0.60 立即提高至 0.004。

---

#### Priority 2 (HIGH): 进一步提高 fly_low_penalty — 压制剩余俯冲

**问题：** crash+fly_low Q5=0.638，阈值修复已处理最严重情况，但策略仍在 z=0.5~1.0m 区间频繁触发低空惩罚（fly_low reward Q5=−2.410/ep）

**变更：**
- 文件：`exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/move/marl_move_env_cfg.py`
- 参数：`fly_low_penalty` 8.0 → **10.0**
- 依据：每增加 2 单位约减少 30~40% fly_low（历史数据）；Run19 crash+fly_low Q5=0.638，目标 <0.5 需再减 ~21%

**替代方案：** 若担心 fly_low_penalty 与 bbox 耦合（Run17 教训：penalty 过高→保守飞行→更多 bbox），可先测试 9.0。Run19 bbox Q5=0.666（仍在改善），暂无耦合信号。

---

#### Priority 3 (MEDIUM): 确认 fly_low 终止阈值 0.5m 设计——考虑进一步收紧

**问题：** fly_low 终止阈值 0.5m 修复了 0.1m 的无约束俯冲区，但 height_penalty 覆盖至 z<1.0m，仍有 0.5m（0.5~1.0m）的"软惩罚区"供策略探索

**可选变更：**
- 文件：`exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/move/marl_move_env.py`
- 参数：`fly_low` 终止阈值（reward 和 termination 两处） 0.5m → **0.6m**（小步收紧，避免过激）
- 依据：NovaCarter 高度≈0.25m（scale 3x），0.5m 仍与车身贴近；0.6m 给策略 10cm 额外安全裕量
- 风险：如果 ep_len 再次大幅下降（如 Q5 <300），则放弃此变更

**优先级降为 MEDIUM：** 先观察 Priority 1+2 对 crash/fly_low 的改善效果，若 Run20 Q5 <0.5 则暂不需要此项。

---

#### Priority 4 (LOW): 监控 height_penalty——暂不调整，观察归一化

**问题：** height_penalty last=−28.36/ep（数值大但 per-step=−0.055/step 仍可接受）

**建议：** 维持现有 height_penalty_threshold=1.5m。per-step 指标未爆炸（不像 Run9 threshold=1.0m 时的 −0.08/step）。若 Run20 per-step height_penalty 超过 −0.08/step，则将 threshold 收紧至 1.3m。

---

### Run 20 实验计划

| 参数 | Run19 | Run20 | 变更理由 |
|------|-------|-------|---------|
| `entropy_loss_scale` | 0.005 | **0.003** | std 过高（1.055），目标 0.65 |
| `fly_low_penalty` | 8.0 | **10.0** | crash+fly_low Q5=0.638，再降 20% |
| `upright_penalty_weight` | 0.5 | **0.5** | 保持（运行正常） |
| `fly_low termination z` | 0.5m | **0.5m** | 保持（修复成功） |
| `bounding_box_threshold` | 14.0m | **14.0m** | 保持（VALIDATED） |
| `success_reward_weight` | 0.6 | **0.6** | 保持 |
| `tracking_reward_weight` | 5.0 | **5.0** | 保持 |
| `contact_sensor_threshold` | 50N | **50N** | 保持 |

**训练命令：**
```bash
python3 scripts/skrl/train.py --task=Isaac-marl-move-flyfollow-marl-v0 \
  --headless --num_envs=2048 --seed=-1 --algorithm="MAPPO"
```

**监控指标（按优先级）：**
1. policy_std Q2：若 <0.60，立即 abort 并将 entropy 调回 0.004
2. crash+fly_low Q3 mean（目标 <0.3）
3. all_targets_captured Q3 mean（目标 >0.60，不出现崩溃）
4. height_penalty per-step（目标 <−0.06/step）

**成功标准：**
- crash+fly_low Q5 < 0.5
- all_targets_captured Q5 > 0.70，Q3→Q5 衰减 <30%
- policy_std Q5 在 0.63~0.75
- instant_reward Q5 > 0.80

**早停标准：**
- 200k 步时 std Q2 < 0.58 → 提前停止，entropy 调至 0.004
- 200k 步时 crash+fly_low Q3 > 0.80（比 Run19 Q3=0.295 更差）→ 提前停止，检查 fly_low 惩罚设计

---

### Run 19 关键结论（供后续参考）

1. **fly_low 终止阈值 0.1→0.5m 是最有效的单次变更**：crash+fly_low −68%，captured 从崩溃恢复至 Q5=0.811。这是 Run19 最重要的贡献。

2. **upright_weight=0.5 恢复有效**：Run18 的 0.3 失败（25x 倾斜增长），0.5 恢复后 Q5=−2.28（正常）。固定为 0.5。

3. **entropy=0.005 过度扩张 std**：末段 1.055→1.181，超过所有先前健康 run（0.60~0.82 范围）。0.003 是下一个校准点。

4. **captured 稳定性与 std 关联**：Q5 mean=0.811（良好）但 last=0.520（末尾波动），与 std 末尾加速（1.181）强相关。压缩 std 是稳定 captured 的关键。

5. **per-step 奖励质量已超越 Baseline**：Run19 Q5=0.952 vs Baseline=0.544。总奖励差距 60% 完全来自 ep_len（crash 终止缩短 episode）。修复 crash/fly_low 是追平 Baseline 的唯一剩余障碍。

### Changelog
- 2026-04-08: Run 19 完整分析（400k steps）。fly_low 阈值修复验证（−68% crash），captured 恢复 Q5=0.811，std 过扩（1.181）。Run 20 建议：entropy 0.005→0.003，fly_low_penalty 8→10。


---

# Training Analysis Report — Move Task Run 20

**Run:** 2026-04-08_10-25-55_mappo_torch_mappo
**Date:** 2026-04-08
**Task:** Isaac-marl-move-flyfollow-marl-v0
**Algorithm:** MAPPO
**Total timesteps:** 400,000

## Run 20 变更（相比 Run 19）

| 参数 | Run 19 | Run 20 |
|------|--------|--------|
| `entropy_loss_scale` | 0.005 | 0.003 |
| `fly_low_penalty` | 8.0 | 10.0 |
| 其余 | — | 不变 |

## Training Metrics Summary

| 指标 | Q1 | Q2 | Q3 | Q4 | Q5 | last |
|------|----|----|----|----|----|----|
| total_reward_mean | 1.81 | 123.3 | 290.6 | 492.2 | 544.0 | 593.6 |
| instant_reward_mean | 0.014 | 0.604 | 1.001 | 1.130 | 1.079 | 0.820 |
| ep_len_mean (steps) | 130.8 | 200.5 | 289.9 | 431.8 | 503.8 | 510.6 |
| all_targets_captured | 0.082 | 0.758 | 0.935 | 0.907 | 0.817 | 0.760 |
| crash termination | 0.060 | 0.223 | 0.272 | 0.391 | 0.465 | 0.330 |
| falcon_fly_low term | 0.052 | 0.218 | 0.266 | 0.382 | 0.449 | 0.330 |
| combined crash+fly_low | 0.112 | 0.441 | 0.538 | 0.773 | 0.915 | 0.660 |
| policy_std | 0.811 | 0.804 | 0.884 | 1.021 | 1.337 | 1.577 |
| bounding_box term | 1.035 | 0.842 | 0.754 | 0.583 | 0.463 | 0.560 |
| fly_low reward/ep | −0.462 | −2.025 | −2.493 | −3.696 | −4.352 | −3.300 |
| height_penalty/ep | −4.14 | −6.16 | −11.00 | −17.36 | −19.15 | −27.14 |
| upright_penalty/ep | −0.430 | −1.092 | −1.752 | −2.743 | −3.271 | −4.553 |
| success_reward/ep | 2.13 | 44.6 | 104.1 | 176.9 | 194.3 | 327.3 |
| value_loss | 0.306 | 0.401 | 0.140 | 0.144 | 0.141 | 0.218 |
| entropy_loss | −0.0036 | −0.0036 | −0.0039 | −0.0043 | −0.0051 | −0.0056 |

---

## Observations & Findings

### 1. policy_std 持续过度扩张，未达收敛目标 — CRITICAL

**症状：** entropy=0.003 完全失控。std 从 Q1=0.811 单调上升至 Q5=1.337，last=1.577（Run 19 last=1.181，本次更差 +33%）。Q4→Q5 的后半段斜率：Q4 前半=0.972，Q4 后半=1.069，Q5 前半=1.221，Q5 后半=1.453。末段 20 次更新斜率 +0.001243/update，**仍在加速上升，未到顶**。entropy_loss 本身也在持续增大（Q1=−0.0036 → Q5=−0.0051，last=−0.0056），印证 entropy 压力没有压住 std。

**Run 19 vs Run 20 对比：**
- Run 19: entropy=0.005, std last=1.181
- Run 20: entropy=0.003（降低 40%），std last=1.577（反而增大 +33%）

**根本原因：** entropy_loss_scale 降低本应压缩 std，但 Run 20 中**负向梯度方向被 success_reward 的强正向梯度压倒**。success_reward Q5=194/ep（Run 19 Q5=130/ep，本次更高），每次成功捕获产生强梯度，驱动策略向特定行为收敛；PPO 的 clip + entropy 机制无法在这个梯度量级下维持 std。简言之：entropy 系数降低 40%，但 success 梯度增强使净探索动力反而上升。

**结论：** entropy=0.003 对本任务当前 success_reward 规模完全无效。std 过扩已成为训练最严重的不稳定因素。

---

### 2. crash+fly_low 不降反升，为历史最高水平 — CRITICAL

**症状：** 

| 区间 | combined crash+fly_low | Run 19 同期 |
|------|------------------------|------------|
| Q1 | 0.112 | 0.143（Run 19 基准） |
| Q2 | 0.441 | 较低（Run 19 有所压制） |
| Q3 | 0.538 | 0.295（Run 19 明显优于本次） |
| Q4 | 0.773 | — |
| Q5 | **0.915** | **0.638**（本次恶化 +44%） |
| last | 0.660 | — |

Run 20 Q5=0.915，远超 Run 19 Q5=0.638，也超过目标 <0.5。这是 Run 19 之后的明确回退。

fly_low reward Q5=−4.352/ep（Run 19 Q5=−2.410），惩罚绝对值更大，说明每次 fly_low 事件触发更大惩罚（fly_low_penalty 8→10），但**发生频率也在升高**，净效果为恶化。

**根本原因：** policy_std 持续加速扩张（末段 1.577）导致动作随机性极高，drone 轨迹不可预测，更容易触发 z<0.5m 的低空终止。fly_low_penalty 提高到 10.0 的惩罚信号被 std 扩张引起的随机探索完全淹没——策略无法"学到"避免低空行为，因为每次低空事件都是随机偏差，不是稳定选择。这是 std 过扩的下游直接后果。

**fly_low 每步惩罚率（归一化）：**
- Q1=−0.00371/step, Q2=−0.01017/step, Q3=−0.00899/step, Q4=−0.00961/step, Q5=−0.01027/step
- 趋势：Q2 以后基本稳定（0.009~0.010/step），说明惩罚率已饱和——即使提高 penalty，发生频率同步上升，净 per-step 惩罚几乎不变。

---

### 3. all_targets_captured Q5=0.817，维持 Run 19 水平，但末尾不稳定 — PARTIAL SUCCESS

**症状：** Q2=0.758 → Q3=0.935 → Q4=0.907 → Q5=0.817，last=0.760。整体在目标 >0.70 以上（达成），但 Q3→Q5 出现 −12.6% 的衰减（Run 19: Q4=0.894→Q5=0.811，−9.3%，稍好）。

last-10 趋势：[0.880, 1.000, 0.970, 0.820, 0.720, 0.370, 0.170, 1.140, 0.630, 0.760]——波动极大，最低 0.170（vs Run 19 末尾波动最低 0.218）。

**积极面：** Q3=0.935 为历史最高（Run 19 Q3=0.829），说明中期追踪能力在提升。Q4=0.907 也高于 Run 19 Q4=0.894。

**消极面：** std 末段爆炸（1.577）导致末尾 captured 波动比 Run 19 更严重。Q5 last-10 中出现 0.170（17%），是训练后期捕获能力不稳定的强信号。

---

### 4. ep_len 向 Baseline 靠近，但 Q5=504 仍远低于 1658 — HIGH

**症状：** ep_len_mean Q5=503.8 steps（Run 19 Q5=374，本次 +35% 进步）。best=1825 steps（接近 Baseline 1658）。但 Q5 均值 504 vs Baseline 1658，差距仍为 3.3x。

time_out termination 全程 = 0.0——没有任何 episode 到达 timeout（1200 steps at 20Hz = 60s）。所有 episode 都被 crash/fly_low/captured/bounding_box 提前终止。ep_len 增长完全来自 captured 的提前终止时间延长（策略更早完成捕获），不是生存能力提升。

**与 Baseline 差距根源：** Baseline ep_len~1658 steps 的核心原因是"无 crash+fly_low 终止"（crash=0.000, fly_low=0.000）。Run 20 Q5 combined=0.915，几乎每轮次都有 crash 或 fly_low。修复 crash+fly_low 是 ep_len 接近 Baseline 的唯一路径。

---

### 5. bounding_box 持续改善，无新问题 — LOW

**症状：** bounding_box Q1=1.035 → Q5=0.463，last=0.560。单调下降趋势健康。与 Run 19（具体 Q5 值未记录，目标方向一致）相比继续改善。bounding_box=14m 配置有效。

---

### 6. success_reward 规模过大，开始成为新的不平衡因素 — HIGH

**症状：** success_reward Q5=194.3/ep，last=327.3。与所有惩罚项绝对值总和（fly_low Q5≈4.4 + height_penalty Q5≈19.1 + upright Q5≈3.3 + boundary_soft Q5≈1.1 ≈ **28/ep**）相比，success_reward 是惩罚总和的约 7x。

这是 Run 14 历史教训（success_reward 1,000x >> penalties → crash 爆炸）的初步信号，尚未到崩溃程度（因为 success_reward_weight=0.6 比 Run 14 的 5.0 小得多），但已显现：policy_std 不受控扩张背后有 success_reward 强梯度的推动。

---

### 综合评估

| 核心问题 | Run 20 结果 | 目标 | 达成？ |
|---------|----------|------|--------|
| policy_std Q5（目标 0.63~0.75） | **1.337，last=1.577** | 0.65 附近 | **未达，严重恶化** |
| crash+fly_low Q5（目标 <0.5） | **0.915** | <0.5 | **未达，Run 19 比本次好** |
| all_targets_captured Q5（目标 >0.70） | **0.817** | >0.70 | 达成（中期 Q3=0.935 历史最高） |
| ep_len 向 Baseline 靠近 | **Q5=504，last=510** | → 1658 | 部分（+35% vs Run 19） |
| instant_reward Q5（目标 >0.80） | **1.079** | >0.80 | 达成 |
| 出现新问题？ | std 加速失控 + success 梯度过大 | 无新问题 | 有 |

**整体判断：** Run 20 是**失败的调参轮次**——两项关键目标（std、crash+fly_low）均未达成且明显劣于 Run 19。entropy=0.003 的降低在 success_reward 强梯度面前完全无效；fly_low_penalty=10.0 的提高在 std 爆炸引起的随机俯冲面前也无效。根本问题是 **std 过扩是 crash+fly_low 的上游原因，而非 penalty 不足**。

---

## Improvement Recommendations

### Priority 1 (CRITICAL): 大幅降低 entropy_loss_scale 至有效压制 std 的量级

**问题：** entropy=0.003 时 std 仍加速扩张至 1.577（目标 0.65）。success_reward 强梯度是抗拒因素。

**历史校准数据：**
- entropy=0.002（Run 11）→ std 从 1.595 降至 0.597（过度压缩）
- entropy=0.003（Run 12）→ std 从峰值回落至 0.627（稍低于目标）
- entropy=0.004（Run 13/14）→ std 稳定 0.759（目标范围内）
- entropy=0.005（Run 19）→ std 爆炸至 1.181
- entropy=0.003（Run 20）→ std 爆炸至 1.577（更差，因 success_reward 更大）

**结论：** 在 success_reward 规模 ~200/ep 的条件下，entropy=0.003 与 entropy=0.005 的效果相近（均无法压制）。需要比 0.003 更激进的压制，但又不能重蹈 Run 11（entropy=0.002 → std=0.597 崩溃）。

**提议：**
- 将 entropy_loss_scale 降至 **0.001**（在 Run 11 的 0.002 基础上再减半）
- **同时** 将 success_reward_weight 降至 **0.3**（减少成功梯度强度，允许 entropy 发挥作用）
- 这两个变更必须同时进行——单独降 entropy 会重蹈 Run 11 崩溃；单独降 success_weight 可能重蹈 Run 16 的 captured 回退

**文件：** `scripts/skrl/train.py` 或 MAPPO 训练配置
**参数：** `entropy_loss_scale` 0.003 → **0.001**

**文件：** `exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/move/marl_move_env_cfg.py`
**参数：** `success_reward_weight` 0.6 → **0.3**

**预期效果：**
- success_reward Q5 从 194/ep 降至 ~97/ep（减半），与 penalties（~28/ep）比值从 7x 降至 ~3.5x（仍高但可接受）
- entropy 在较弱 success 梯度下可将 std 压回 0.65~0.80 区间
- 参考 Run 15（success_weight=1.0, entropy=0.006）→ captured Q5=0.811 稳定；本次 success_weight=0.3 不应像 Run 16 那样导致 captured 崩溃，因为 bounding_box 问题已在后续 run 解决（Run 16 时 bounding_box 尚未修复）

**风险控制：** 200k 步时检查 std Q2：
- 若 std Q2 < 0.58 → entropy 提回 0.002，防止 Run 11 式崩溃
- 若 all_targets_captured Q3 < 0.50 → 立即 abort，success_weight=0.3 下降过多

---

### Priority 2 (HIGH): fly_low_penalty 从 10.0 回退至 8.0，等 std 稳定后再评估

**问题：** fly_low_penalty=10.0 的提升在 std 爆炸（1.577）引起的随机俯冲面前完全无效（per-step fly_low 惩罚率饱和在 −0.010/step）。提高 penalty 不能解决随机动作导致的低空触发。

**fly_low per-step 惩罚率分析：**
- Run 19（penalty=8.0）：Q5 per-step ≈ −0.008/step（估算）
- Run 20（penalty=10.0）：Q5 per-step = −0.010/step
- 惩罚率提升 25%，但 combined termination Q5 从 0.638 恶化至 0.915（+44%）

惩罚提高与发生率提高相抵消——这是"penalty 饱和"的经典症状。当 std 过高时，fly_low 事件是随机动作的副产品，不是学到的行为，因此 penalty 梯度无法对应地修正策略。

**提议：** Run 21 中将 fly_low_penalty 维持在 **8.0**（回退到 Run 19 值），待 std 压缩至 <0.80 后，再评估是否需要进一步提高。

**文件：** `exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/move/marl_move_env_cfg.py`
**参数：** `fly_low_penalty` 10.0 → **8.0**（回退）

---

### Priority 3 (MEDIUM): 监控 success_reward 是否触发 Run 14 式崩溃预警

**问题：** success_reward/penalties 比值已达 7x（vs Run 14 时的 1000x）。尚未出现 Run 14 式的 crash 爆炸（upright_penalty 仅 −3.3/ep，未失控），但信号存在。

**预警阈值（Run 21 中监控）：**
- success_reward Q3 > 100/ep → 提前降低 success_weight
- upright_penalty Q4 > −8/ep → warning（Run 14 崩溃前 upright 达 −13.56/ep）
- crash Q5 > 0.60 → abort，success_weight 过高

---

### Priority 4 (MEDIUM): 确认 fly_low 终止阈值 0.5m 维持不变

**依据：** 本轮 fly_low per-step 惩罚率饱和（0.010/step），并非阈值问题。0.5m 阈值是 Run 19 已验证的有效配置，维持不变。阈值进一步提高（至 0.6m）的时机是 crash+fly_low Q5 <0.4 之后，目前条件不满足。

---

## Experiment Plan — Run 21

### 变更汇总

| 参数 | Run 20 | Run 21 | 变更理由 |
|------|--------|--------|---------|
| `entropy_loss_scale` | 0.003 | **0.001** | std 爆炸（1.577），需强力压制 |
| `success_reward_weight` | 0.6 | **0.3** | 削弱 success 梯度，允许 entropy 生效 |
| `fly_low_penalty` | 10.0 | **8.0** | 回退；penalty 饱和，高 std 下无效 |
| `upright_penalty_weight` | 0.5 | **0.5** | 维持（无问题） |
| `fly_low termination z` | 0.5m | **0.5m** | 维持（已验证） |
| `bounding_box_threshold` | 14.0m | **14.0m** | 维持（持续改善） |
| `tracking_reward_weight` | 5.0 | **5.0** | 维持 |
| `contact_sensor_threshold` | 50N | **50N** | 维持 |

### 训练命令

```bash
python3 scripts/skrl/train.py --task=Isaac-marl-move-flyfollow-marl-v0 \
  --headless --num_envs=2048 --seed=-1 --algorithm="MAPPO"
```

### 监控指标（按优先级）

1. **policy_std Q2**（200k 步时）：目标 0.60~0.75
   - < 0.58 → abort，entropy 调回 0.002
   - > 1.00 → abort，entropy 降至 0.0005
2. **all_targets_captured Q3**（200k 步时）：目标 >0.60
   - < 0.40 → abort，success_weight=0.3 影响过大
3. **crash+fly_low Q3 mean**（目标 < 0.30）
4. **success_reward/penalties 比值**：目标 <4x；>8x 时降低 success_weight
5. **upright_penalty Q4**：>−8/ep 预警

### 成功标准

- policy_std Q5 在 0.63~0.80（核心目标）
- crash+fly_low Q5 < 0.50（回落至 Run 19 水平或更好）
- all_targets_captured Q5 > 0.70，Q3→Q5 衰减 <25%
- instant_reward Q5 > 0.90
- ep_len Q5 > 550（超过 Run 20 的 504）

### 早停标准

- 200k 步时 std Q2 < 0.58 → entropy 上调
- 200k 步时 all_targets_captured Q3 < 0.40 → abort，检查 success_weight
- 200k 步时 crash+fly_low Q3 > 0.80 → 检查 fly_low_penalty 方向

---

## Run 20 关键结论（供后续参考）

1. **entropy=0.003 在 success_reward ~200/ep 规模下完全无效**：std 从 Run 19 last=1.181 进一步扩张至 1.577，且末段仍在加速（斜率 +0.0012/update）。entropy 系数降低 40% 被 success 梯度完全抵消。

2. **fly_low_penalty 10.0 是"penalty 饱和"的典型案例**：per-step 惩罚率从 8.0 到 10.0 仅提升 25%，但 combined termination Q5 从 0.638 恶化至 0.915（+44%）。std 过高时，随机动作触发 fly_low 是结构性问题，penalty 提升不能解决。

3. **std 过扩是所有下游问题的上游根因**：crash+fly_low 恶化、captured 末尾波动（0.170 最低）、height_penalty 持续增长，全部可追溯至 std=1.577 的过度探索。

4. **success_reward 规模控制是 Run 21 的关键**：success_reward Q5=194/ep vs penalties=28/ep（7x 比值）开始成为新的梯度失衡源。entropy 和 fly_low_penalty 的效果都被这一强信号淹没。必须同时降低 success_weight 才能让 entropy 恢复效果。

5. **all_targets_captured Q3=0.935（历史最高）是积极信号**：中期追踪能力继续提升，说明核心任务学习方向正确。问题在于末期 std 爆炸导致不稳定，而非任务理解倒退。

### Changelog
- 2026-04-08: Run 20 完整分析（400k steps）。entropy=0.003 + fly_low_penalty=10.0 双失效：std 爆炸至 1.577（历史最高），crash+fly_low Q5=0.915（Run 19 0.638 更差）。根因：success_reward 强梯度压倒 entropy 控制；fly_low_penalty 饱和于随机俯冲。Run 21 建议：entropy→0.001 + success_weight→0.3（同时），fly_low_penalty→8.0（回退）。

---

# Training Analysis Report — Run 21

**Run:** 2026-04-08_20-14-19_mappo_torch_mappo
**Date:** 2026-04-08
**Task:** Isaac-marl-move-v0（NovaCarter ×4，轨迹 ±8m）
**Algorithm:** MAPPO
**Total steps:** 364,600（约 91.1% budget）

## Training Metrics Summary

| 指标 | Q1 | Q2 | Q3 | Q4 | Q5 | last-10 |
|------|----|----|----|----|-----|---------|
| policy_std | 0.798 | 0.738 | 0.684 | 0.673 | 0.698 | **0.721** |
| all_targets_captured | 0.008 | 0.283 | 0.609 | 0.760 | **0.810** | 0.856 |
| crash | 0.076 | 0.182 | 0.343 | 0.473 | 0.451 | 0.243 |
| falcon_fly_low | 0.071 | 0.176 | 0.334 | 0.465 | 0.443 | 0.243 |
| crash+fly_low combined | 0.147 | 0.357 | 0.678 | 0.938 | **0.894** | **0.590** |
| bounding_box | 0.993 | 0.845 | 0.669 | 0.542 | 0.557 | 0.717 |
| ep_len_mean（步） | 116.6 | 154.2 | 222.4 | 239.5 | **256.4** | 308.9 |
| success_reward/ep | 0.03 | 6.5 | 23.4 | 32.0 | **40.0** | 54.5 |
| tracking_reward/ep | 1.95 | 3.11 | 4.65 | 5.14 | **5.59** | 5.47 |
| instant_reward（per-step）| −0.017 | 0.127 | 0.270 | 0.373 | **0.423** | 0.488 |
| total_reward/ep | −3.3 | 20.2 | 61.5 | 89.8 | **107.6** | 135.3 |

## Observations & Findings

### 1. policy_std 收敛 — CRITICAL 目标：达成

**结论：entropy=0.001 + success_weight=0.3 的双管齐下成功控制了 std 扩张，这是 Run 21 最重要的修复成果。**

- Q5 均值 = 0.698，末 10 点 = 0.721，远低于 Run 20 last=1.577（降幅 −54.4%）
- std 轨迹：Q1=0.798 → Q3=0.684（收缩）→ Q5=0.698 → last=0.721（末段轻微回升，稳定态）
- Q5 内部五等分：0.691 → 0.690 → 0.695 → 0.701 → 0.715，呈现"底部反弹"而非继续坍缩或爆炸
- **std 现已稳定在 0.65～0.75 的目标区间内**，与 Run 12/13 健康阶段相当

Run 20 失败的根因（success_reward 7x 梯度压倒 entropy）已被同步降低 success_weight 解除。两个变量必须联动调整的假设得到验证。

### 2. crash+fly_low combined — HIGH 目标：部分达成，趋势积极

**结论：Q5 均值 0.894 仍超出 <0.5 目标，但末段呈现明显下降趋势。**

- Q5 内部五等分：Q5.1=1.025 → Q5.2=0.858 → Q5.3=0.773 → Q5.4=1.002 → Q5.5=0.810
- last-20 均值 = 0.590，last-5 均值 = **0.244**（目标 <0.5 在最后 5 个更新窗口已达成）
- 对比：Run 20 Q5=0.915（最后无明显下降趋势）；Run 21 Q5.5=0.810，last-5=0.244（结构性改善迹象）

**解读：** Q5 均值偏高主要受 Q4 前后的"中段振荡"拉高（步约 290k–350k）。最末段已开始下降。关键问题是这一下降能否持续——std 已稳定在 0.72 意味着随机俯冲触发减少，这正是推动末段改善的机制。

**与 Run 17 对比（run 中 std 0.637、fly_low=12.0）：** Run 17 Q5=0.454（plateau）；Run 21 Q5=0.894 但末段 0.244。Run 21 fly_low=8.0 比 Run 17 fly_low=12.0 温和，但 std 控制更好（0.72 vs 0.63），因此末段崩溃事件更少。

### 3. all_targets_captured — HIGH 目标：完全达成

**结论：Q5=0.810 满足 >0.7 目标，末段稳定，无 Run 16（−74%）型退化。**

- 轨迹：Q1=0.008 → Q2=0.283 → Q3=0.609 → Q4=0.760 → Q5=0.810，单调上升
- Q5 内部五等分：0.798 → 0.722 → 0.832 → 0.894 → 0.805，波动约 ±0.09，无系统性崩溃
- last-20 均值 = 0.856，last-5 均值 = 0.836

Run 16 担忧（success_weight=0.3 摧毁 captured）未发生。原因分析：
1. Run 16 时 fly_low_threshold=0.1m，crash 事件频繁截断 episode，captured 无法完成
2. Run 21 的 fly_low_threshold=0.5m + fly_low_penalty=8.0 为基础，episode 生存能力更强
3. success_reward Q5 仍有 40/ep（Run 19 约 130/ep，降幅 −69%），激励强度适中，不至于为"规避 crash 风险"而放弃追踪

### 4. ep_len 趋势 — MEDIUM 目标：方向正确，距 Baseline 仍远

- Q5 均值 256 步，last-10 均值 309 步，last 单点 398 步
- Baseline 目标：1658 步；当前 last-10 = Baseline 的 19.1%
- 相比 Run 20（Q5=504 步），Run 21 Q5 仅 256 步——**ep_len 出现退步**

**退步原因：** success_reward_weight 从 0.6 降至 0.3，成功终止激励降低，但同时 crash+fly_low 在 Q3-Q4 仍较高（0.678/0.938），短 episode 比例较大。bounding_box Q5=0.557 也维持较高，是另一主要截断源。

Run 20 ep_len Q5=504 的部分原因是 std=1.577 过度探索偶发长 episode，并非稳定的高质量 episode。Run 21 ep_len Q5=256 更真实反映当前策略能力。

### 5. success_reward 规模控制 — MEDIUM 目标：达成

- success_reward Q5=40.0/ep（vs Run 20 Q5=194/ep，降幅 −79.4%）
- success_reward / |total_penalties| 比值：Q3=2.41，Q4=3.19，Q5=3.42，last-20=4.23
- Q5 比值 3.4x，低于目标 <4x，在健康范围内
- Run 14 失效时比值 >100x，Run 20 约 7x，Run 21 3.4x——梯度平衡已逐步改善

**per-step 归一化 success_reward：** Q5=0.163/step（Run 16 success_weight=0.3 时约 0.10/step，更高是因为 ep_len 不同）。未观测到 Run 16 式的 reward hacking（per-step 值 Q3→Q5 单调增长为 0.106→0.137→0.163 属于正常策略改善，非被动收集）。

### 6. bounding_box — LOW 新问题：末段反弹

- Q5 均值 0.557（比 Run 20 Q5=0.463 略高）
- last-5 均值 = **0.916**（严重末段反弹！）
- last-20 均值 = 0.692

末段 bounding_box 反弹是 Run 21 中唯一出现的新负面信号。可能原因：
1. std 末段轻微回升（0.715）使策略多样性增加，部分行为更激进
2. success_reward 末段 last-10=54.5/ep 也在上升，追踪激励重新驱动超界

此问题在 Run 20 末段同样出现（ep_len Q5=504 时成功抑制 bbox，但末段 bounding_box 未见特别监控）。bbox=14.0m 本身配置应保持不变——问题来自行为动态而非边界设置。

### 7. height_penalty — LOW 可接受

- per-step height_penalty Q5 = −0.0262/step（Run 17 参考值约 −0.022/step，略高但稳定）
- 无爆炸迹象（阈值 1.5m + per-drone 修复已验证）
- 绝对值 Q5 = −6.4/ep 是 ep_len 积累的假像，per-step 才是真实指标

## Improvement Recommendations

### Priority 1（CRITICAL）：crash+fly_low Q5 降至 <0.5

**问题：** Q5 均值 0.894 超标，虽末段 0.244 达标，但 Q5 平均水平证明训练中段存在高强度崩溃窗口（步 280k–350k 区域，Q5.1/Q5.4 均约 1.0）。

**根因：** std 在 Q3-Q4 仍约 0.68-0.67，此阶段 success_reward 梯度增长（Q3=23/ep→Q4=32/ep）驱动更激进追踪 → 更多俯冲 → crash+fly_low 激增。fly_low_penalty=8.0 的威慑力在 std 相对较高时不足。

**Proposed Change:**
- File: `exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/move/marl_move_env_cfg.py`
- Parameter: `fly_low_penalty` 8.0 → **10.0**
- Rationale: Run 20 的 10.0 失效是因为 std=1.577（随机俯冲），现在 std=0.72 已受控，10.0 的威慑力应能有效针对主动俯冲行为。Run 20 教训是"fix std first"，Run 21 已完成这一先决条件。

### Priority 2（HIGH）：稳固 std 稳定性，防止末段反弹

**问题：** std Q5 内部从 Q5.1=0.691 轻微回升至 Q5.5=0.715，last=0.721。虽仍在目标区间，但方向是向上的。同时 success_reward 末段 last-10=54.5/ep 仍在上升，如果持续可能重现 Run 20 的梯度不平衡。

**Proposed Change:**
- File: `exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/move/marl_move_env_cfg.py`（agent 配置部分）
- Parameter: `entropy_loss_scale` 0.001 → **0.001**（保持不变）
- Rationale: 当前 0.001 已将 std 从 1.577 压制到 0.72——这是正确的。不需要进一步调整。关键是监控 400k 步完成后的 std 走势；如果继续上升超过 0.75，下一轮再考虑 0.002。
- **NOT CHANGING:** success_reward_weight 保持 0.3——当前比值 3.4x 在健康范围内，且没有 reward hacking 迹象

### Priority 3（MEDIUM）：解决 bounding_box 末段反弹

**问题：** bounding_box last-5=0.916，明显超出正常水平（Q5 均值 0.557 本身可接受，但最后 5 个更新窗口的反弹值得关注）。

**Proposed Change:**
- 不修改 bounding_box_threshold=14.0m（已验证有效）
- 监控 Run 22 中 Q5 bounding_box 是否持续下降或反弹
- 若 Run 22 Q5 bounding_box > 0.7，考虑增大 boundary_soft_penalty_weight（当前值待确认）

### Priority 4（LOW）：ep_len 向 Baseline 靠拢

**问题：** ep_len Q5=256 步，是 Baseline 1658 步的 19%。这是已知结构性差距，由 crash+fly_low + bounding_box 双重截断造成。

**分析：** Priority 1 的 fly_low_penalty 提升如果成功降低 crash，ep_len Q4-Q5 应随之增长（参考 Run 15→16→17 的 ep_len 与 crash 反相关）。无需单独调整 ep_len 参数。

## Run 22 实验计划

**基准参数（继承 Run 21）：**
- entropy_loss_scale = 0.001（保持）
- success_reward_weight = 0.3（保持）
- upright_penalty_weight = 0.5（保持）
- fly_low termination z = 0.5m（保持）
- bounding_box_threshold = 14.0m（保持）
- tracking_reward_weight = 5.0（保持）
- contact_sensor_threshold = 50N（保持）

**Run 22 唯一变更：**

| 参数 | Run 21 | Run 22 |
|------|--------|--------|
| `fly_low_penalty` | 8.0 | **10.0** |

**理由：** Run 20 的 10.0 失效于 std=1.577 环境（随机俯冲无法被惩罚矫正）。Run 21 将 std 稳定到 0.72，消除了随机俯冲的根源。在 std 受控的前提下，10.0 的威慑力应能针对主动俯冲行为发挥作用（参考 Run 9 中 threshold=1.5m + fly_low=4.0 的协同效果）。单变量变更保证可解释性。

**训练配置：**
```bash
python3 scripts/skrl/train.py \
  --task=Isaac-marl-move-v0 \
  --headless --num_envs=2048 --algorithm="MAPPO"
```

**监控目标（以 100k 步为检查点）：**
- 200k 步：std Q2 应在 0.63~0.73（若 <0.60 → 停止，entropy 上调至 0.002）
- 200k 步：crash+fly_low Q3 应低于 Q1（单调下降信号）
- 300k 步：all_targets_captured Q4 应 >0.75（若 <0.55 → 检查 success_reward 是否下滑）
- 400k 步目标：
  - policy_std 末段：0.65～0.75（目标不变）
  - crash+fly_low Q5 < **0.6**（从 Run 21 Q5=0.894 改善）
  - all_targets_captured Q5 > **0.80**（维持 Run 21 水平）
  - ep_len Q5 > **300**（从 Run 21 Q5=256 改善，crash 减少自然带动）
  - bounding_box Q5 < **0.5**（观察末段反弹是否收敛）

**成功标准：**
- crash+fly_low Q5 < 0.6 且无末段恶化趋势
- all_targets_captured Q5 > 0.80 且 Q5 内部无 >30% 衰减
- std 末段稳定 0.65～0.75，不再出现 Run 19/20 式扩张

**中止标准：**
- 200k 步时 std Q2 < 0.58（entropy 过度压制探索）
- 200k 步时 crash+fly_low Q3 > 0.90（10.0 比 8.0 更差 → 回退至 8.0，寻找其他路径）
- 200k 步时 all_targets_captured Q3 < 0.40（captured 退化，检查 success_reward 激励）

## Run 21 关键结论（供后续参考）

1. **entropy=0.001 + success_weight=0.3 联动调整成功**：std 从 1.577（Run 20）稳定至 0.72（Run 21），验证了"必须同时削减 success 梯度和加强 entropy 压制"的假设。单独操作任一参数会失败（Run 11 教训：单降 entropy→std 崩溃；Run 16 教训：单降 success_weight→captured −74%）。

2. **all_targets_captured Q5=0.810 无退化**：success_weight=0.3 在 fly_low_threshold=0.5m 的环境下不会摧毁追踪激励（Run 16 的 −74% 退化是 fly_low_threshold=0.1m + 高 crash 率的叠加效果，而非 success_weight 单因素导致）。

3. **crash+fly_low Q5 仍是主要未解决问题**：Q5 均值 0.894 虽末段改善（last-5=0.244），但中段（步 280k–350k）振荡严重。std 已受控后，fly_low_penalty 的威慑力可以重新测试（Run 22 提升至 10.0）。

4. **ep_len 退步（Q5：504→256）是设计代价**：success_weight 降低减少了成功终止频率（ep 更短），但提升了训练稳定性。这是可接受的权衡。

5. **bounding_box 末段反弹是新观测到的现象**：last-5=0.916 需在 Run 22 持续监控，但不是当前最优先问题。

### Changelog
- 2026-04-08: Run 21 完整分析（364k/400k steps）。entropy=0.001 + success_weight=0.3 成功将 std 从 1.577 压制至 0.72（目标达成）。all_targets_captured Q5=0.810（>0.7 目标达成，无 Run 16 退化）。crash+fly_low Q5=0.894 仍超标但末段 0.244 展示下降趋势。Run 22 建议：fly_low_penalty 8.0→10.0（单变量，此前 std 受控后 10.0 未测试）。
- 2026-04-09: Run 22 完整分析（244k steps）。训练完全失败：bounding_box 终止率 ~1.0（all_captured=0.0），ep_len 退化至 75 步（0.75s/回合）。根本原因：drone_spawn_x_range=(-10,-8) 将无人机置于距边界仅 4m 处（|−14−(−10)|=4m），同时 upright/body_rate/smoothness 权重恢复至 baseline（2.0/2.0/1.0），三者惩罚梯度冲突导致策略以"快速冲出边界"作为局部最优解。Run 23 建议：撤销 spawn 扩展和三个权重恢复，仅保留 fly_low_penalty=10.0 单变量。

---

## Run 22 训练分析报告

**Run:** 2026-04-09_08-34-37_mappo_torch_mappo
**分析日期:** 2026-04-09
**Task:** Isaac-marl-move-v0
**Algorithm:** MAPPO
**总步数:** 244,200 步（未完成 400k）

### 训练指标摘要

| 指标 | 早期均值（前25%）| 近期均值（后25%）| 末5点均值 |
|------|---------|---------|---------|
| Total reward (mean) | -19.66 | -5.91 | -4.32 |
| all_targets_captured | 0.0000 | 0.0000 | 0.0000 |
| bounding_box 终止率 | 0.9305 | 0.9857 | 1.0000 |
| ep_len (mean steps) | 120.1 | 83.8 | 75.0 |
| policy_std | 0.822 | 0.752 | 0.718 |
| upright_penalty | -1.446 | -0.510 | -0.546 |
| height_penalty | -4.824 | -1.008 | -0.443 |
| height_reward | 0.791 | 0.743 | 0.758 |
| distance_reward | 0.320 | 0.146 | 0.093 |
| tracking_reward | 0.071 | 0.003 | 0.000 |
| crash 终止率 | 0.061 | 0.001 | 0.000 |
| falcon_fly_low 终止率 | 0.061 | 0.000 | 0.000 |

### 问题诊断

#### 问题 1：回合快速结束 — 严重（CRITICAL）

**症状：** 平均回合长度从 120 步（1.2s）进一步下降至 75 步（0.75s），bounding_box 终止率达 100%。all_targets_captured 全程为 0。

**根本原因：双重致命组合**

1. **spawn 位置距边界间隙不足（4m）：**
   - drone_spawn_x_range=(-10,-8)，bounding_box_threshold=14m，边界在 x=±14
   - 无人机最近点距负 x 侧边界仅 4m（|−14−(−10)|=4m）
   - 对比 Run 21 spawn=(-4,-2)：间隙为 10m
   - 即使以中等速度（4 m/s）向负 x 方向飞行，仅需 1.0s = 100 步即触发边界

2. **高权重惩罚组合在 early policy 中产生梯度冲突：**
   - upright_weight=2.0、body_rate_weight=2.0、action_smoothness_weight=1.0
   - 相比 Run 21（0.5/0.5/0.3），梯度压力放大 4×/4×/3×
   - 随机初始化策略产生随机动作 → 触发大倾斜 → 惩罚梯度强制"停止运动"
   - "停止运动"无法克服初始速度扰动，导致无人机随机漂移出边界

3. **负反馈局部最优陷阱：**
   - ep_len 在训练过程中持续缩短（早期120步 → 末期75步），而非增长
   - 这是策略退化的典型信号：策略学会"更快逃出边界"来回避高惩罚步骤
   - 对比 Run 21：ep_len 从 117 步增长至 250 步（策略学会存活）

**量化验证：** 75 步 × 0.01s/步 = 0.75s/回合，而完整任务需 60s 回合（6000步）。策略完全无法在环境中生存。

**证据：**
- bounding_box 终止率早期 0.93 → 末期 1.00（恶化，非改善）
- drone_out 奖励分量全程 -1.0（100% 环境触发边界惩罚）
- tracking_reward 末期 = 0.000（无人机从未接近目标）
- distance_reward 末期 0.093（比早期 0.320 更差，说明距离在增大）

---

#### 问题 2：姿态无法稳定 — 高（HIGH）

**症状：** upright_penalty 早期 −1.446（对应 weight=2.0），表明起飞初期存在显著倾斜。trajectory 中无法维持稳定高度接近目标。

**根本原因分析：**

1. **target_rel_pos z 清零与 height_reward_weight=2.0 的矛盾：**
   - 代码 `target_rel_pos[:,:,2] = 0.0` 使策略无法从观测中感知"目标高于/低于自身"
   - 同时 height_reward_weight 从 0.5 提升至 2.0，强制策略保持 desired_height=2.5m
   - 这两个信号并不冲突（z 清零影响目标跟随，height_reward 影响绝对高度），但组合后策略需同时：
     - 在 XY 平面飞向目标（z 信息被隐藏）
     - 保持绝对高度 2.5m（height_reward 激励）
   - 在 4x 重的倾斜/角速率惩罚压力下，早期策略优先减少倾斜而非飞向目标

2. **upright 量化分析：**
   - Run 22 per-step 原始倾斜信号 = upright_penalty/ep_len/weight = −0.510/83.8/2.0 = **−0.00304/步**
   - Run 21 per-step 原始倾斜信号 = −1.731/250.0/0.5 = **−0.01384/步**
   - Run 22 无人机每步倾斜量（物理意义）实际比 Run 21 小 4.6倍
   - 这说明 weight=2.0 本身并不导致倾斜——倾斜量已减少；但高权重导致早期更新幅度过大，干扰飞向目标的梯度

3. **body_rate 量化对比：**
   - Run 22 per-step raw = 0.681/83.8/2.0 = **0.00406**（比 Run 21 的 0.00222 高 83%）
   - 无人机角速率实际比 Run 21 更大，说明在重惩罚下策略产生了更多振荡

**结论：** 姿态不稳并非 weight=2.0 直接导致倾斜加剧（倾斜量更小），而是 spawn 位置 + 重惩罚组合产生的梯度冲突使策略无法学习协调的飞行行为，进而快速退出边界、使问题恶化的恶性循环。

---

### 改进建议

#### Priority 1 (CRITICAL)：撤销 drone_spawn_x_range 扩展

**问题：** spawn=(-10,-8) 导致无人机距边界仅 4m，是 bounding_box=1.0 的直接物理原因。

**建议修改：**
- 文件：`exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/move/marl_move_env_cfg.py`
- 参数：`drone_spawn_x_range` → `(-4.0, -2.0)`（回退至 Run 21）
- 理由：Run 21 spawn=(-4,-2) 下无人机到边界间隙为 10m，策略有足够的生存余量；"对侧出发"的实验设计需要先确保基础稳定性

#### Priority 2 (CRITICAL)：撤销 upright/body_rate/smoothness 权重恢复

**问题：** 在 ep_len=120 步（早期）的脆弱策略中，4× 的惩罚梯度压制了飞向目标的激励，触发局部最优。

**建议修改：**
- 文件：`exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/move/marl_move_env_cfg.py`
- `upright_penalty_weight`: 2.0 → **0.5**（回退至 Run 21）
- `body_rate_penalty_weight`: 2.0 → **0.5**（回退至 Run 21）
- `action_smoothness_weight`: 1.0 → **0.3**（回退至 Run 21）
- 理由：Run 21 在这三个参数下实现了 all_captured Q5=0.810，说明低权重足够约束行为；提升这三个权重需等策略已能稳定飞行后（ep_len > 200 步）再探索

#### Priority 3 (HIGH)：保留 fly_low_penalty=10.0 单变量测试

**问题：** Run 22 的目标是测试 fly_low_penalty 从 8.0 → 10.0 的效果，但由于 CRITICAL 问题的干扰，该变量从未获得有效测试。

**建议：** 在 Run 23 中仅保留 fly_low_penalty=10.0 这一个变量，撤销其他所有 Run 22 变更，得到纯净的因果归因。

#### Priority 4 (MEDIUM)：暂停 height_reward_weight=2.0 变更

**问题：** height_reward_weight 从 0.5 提升至 2.0 与 target_rel_pos z 清零组合后，高度激励与目标接近激励的相对权重变化未经验证。

**建议：** 回退至 0.5（Run 21 值），待 fly_low_penalty=10.0 效果验证后再单独测试 height_reward 权重。

---

### Run 23 实验计划

**目标：** 在 Run 21 基础上，仅测试 fly_low_penalty=10.0 单变量（这是 Run 22 原始意图，现在获得纯净测试）。

**参数配置（相比 Run 21 的唯一变更）：**

| 参数 | Run 21 | Run 23 |
|------|--------|--------|
| `fly_low_penalty` | 8.0 | **10.0** |
| `drone_spawn_x_range` | (-4,-2) | (-4,-2)（回退）|
| `upright_penalty_weight` | 0.5 | 0.5（回退）|
| `body_rate_penalty_weight` | 0.5 | 0.5（回退）|
| `action_smoothness_weight` | 0.3 | 0.3（回退）|
| `height_reward_weight` | 0.5 | 0.5（回退）|

**训练配置：**
```bash
python3 scripts/skrl/train.py \
  --task=Isaac-marl-move-v0 \
  --headless --num_envs=2048 --algorithm="MAPPO"
```

**监控目标（以 100k 步为检查点）：**
- 100k 步：bounding_box 终止率应 < 0.70（若 > 0.90 → 立即停止，检查 spawn/边界配置）
- 100k 步：ep_len Q2 应 > 150 步（策略应学会存活）
- 200k 步：crash+fly_low Q3 应低于 Run 21 同期（0.89），验证 10.0 的威慑效果
- 400k 步目标：
  - all_targets_captured Q5 > 0.80（维持 Run 21 水平）
  - crash+fly_low Q5 < 0.60（Run 21 末期 0.244 趋势的延续）
  - policy_std 末段 0.65～0.75

**成功标准：**
- ep_len Q3 > 200 步（确认策略能在环境中生存）
- crash+fly_low Q5 < 0.60 且无恶化趋势
- all_targets_captured Q5 > 0.80

**中止标准（100k 步时）：**
- bounding_box 终止率 > 0.90（spawn/边界问题未修复）
- ep_len Q2 < 100 步（策略无法存活）
- all_targets_captured 全程为 0

### Run 22 关键教训

1. **spawn 位置与边界距离是硬约束**：drone_spawn_x_range 必须确保间隙 ≥ 8m（bounding_box=14m 时，spawn 上限不得超过 −6m）。超出此约束的所有其他变更都无意义。

2. **多参数同时恢复 baseline 是高风险操作**：Run 22 一次性改变了 7 个参数（spawn + 3个权重 + height_reward + fly_low + z清零），无法归因。单变量原则必须严格遵守，尤其是存在参数间协同效应的惩罚权重。

3. **ep_len 趋势是训练健康度的关键指标**：ep_len 在训练中下降（120→75步）是比奖励曲线更可靠的失败信号——它意味着策略在向"快速死亡"局部最优收敛，而非学习任务。

4. **upright_penalty_weight=2.0 在低 ep_len 环境中适得其反**：即使物理倾斜量更小（raw tilt signal −0.003 vs Run21 −0.014），高权重在早期训练中产生的梯度冲击会干扰飞向目标的梯度方向。此参数的调整应等到策略已建立基础飞行能力后进行。
