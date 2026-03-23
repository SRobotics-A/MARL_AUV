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
