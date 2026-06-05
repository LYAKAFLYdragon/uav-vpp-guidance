# Stage 6G.5A Geometry Smoke Contract

> **Version**: 6G.5A-R  
> **Status**: Active  
> **Scope**: Baseline geometric feasibility only (`no_prediction` method).

---

## 1. Aspect-Angle Semantic Contract

In Stage 6G.5A, ``aspect_angle_deg`` serves **dual roles**:

1. **LOS bearing**: The angle from ownship to target, measured clockwise from ownship heading.
2. **Target heading**: The target aircraft's flight-path heading equals this same angle.

### Example mappings

| aspect_angle_deg | Geometry name | Target position (relative to own at origin) | Target heading |
|---|---|---|---|
| 0° | Tail-chase | Along +x axis at `initial_range_m` | 0° (same as own) |
| 30° | Oblique pursuit | 30° off +x axis | 30° |
| 60° | Lead pursuit | 60° off +x axis | 60° |
| 90° | Broadside / Crossing | Along +y axis at `initial_range_m` | 90° |

### Why this simplification is acceptable for smoke

- Stage 6G.5A only needs to answer: **does any geometry produce >20% success?**
- Decoupling ``los_bearing_deg`` from ``target_heading_deg`` adds two extra dimensions and is unnecessary until Stage 6G.5B (direct-track probe) or Stage 6G.5C (predictor-policy sweep).

### Known limitation

If the target is at 90° but heading 90°, the target is flying **perpendicular** to the LOS. The closure rate is then purely from ownship velocity projected onto the LOS, which is zero. This means:
- `aspect=90°` will have `closure_rate = 0` under the current simplified model.
- A more realistic crossing scenario would set target heading ≈ 270° (flying toward ownship), but that changes the semantics. Stage 6G.5B should introduce `los_bearing_deg` and `target_heading_deg` as separate parameters if crossing geometries are deemed important.

---

## 2. Closure-Rate Calculation Contract

`compute_geometry_metadata()` computes closure rate via **vector projection**:

```text
LOS_unit = [cos(aspect), sin(aspect)]
own_vel  = [ego_speed, 0]
tgt_vel  = [target_speed * cos(aspect), target_speed * sin(aspect)]
range_rate = dot(tgt_vel - own_vel, LOS_unit)
closure_rate = -range_rate
```

This is exact for the simplified model where target heading = aspect angle. It is **not** a general 3-D engagement closure rate.

---

## 3. Episode Execution Contract

- `episodes_per_point` is the **inner loop** inside each evaluation seed.
- Each episode receives a unique deterministic seed:
  ```text
  episode_seed = eval_seed * 100000 + point_index * 1000 + episode_index
  ```
- `evaluated_count` in the summary must equal the actual length of `all_episodes`.

---

## 4. Policy Loading Contract

| Mode | Checkpoint present | `--allow-random-policy` | Behavior |
|---|---|---|---|
| Dry-run | — | — | No policy loaded |
| Real run | Yes | — | Load checkpoint |
| Real run | No | False | **Raise `FileNotFoundError`** |
| Real run | No | True | Warn and use random policy |

Random policy must never be used silently in a real smoke run.

---

## 5. Output File Contract

Regardless of dry-run or real execution, the runner must produce:

- `geometry_smoke_plan.json`
- `geometry_smoke_points.csv`
- `resolved_config.yaml`
- `geometry_smoke_summary.md`
- `geometry_smoke_summary.csv`
- `feasible_candidates.csv`
- `failed_points.csv`

Dry-run CSV files contain headers only.

---

## 6. Bilevel Gate Contract

`bilevel_unblocked_candidate` is `true` **only if**:

1. At least one geometry combo shows `success_rate > 0.20`.
2. The best combo has `closure_rate` in the range `(20, 250) m/s`.

Rationale: if closure is too low, the geometry is nearly neutral and gains have no leverage. If closure is too high, success is trivial and gains are irrelevant.
