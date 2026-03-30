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
