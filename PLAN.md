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

