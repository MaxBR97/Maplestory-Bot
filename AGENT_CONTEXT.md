# Maplestory AI Project Context (Agent Reference)

## What this project is
This repository automates MapleStory gameplay from screen parsing and keyboard actions.

- `main.py` is an attempt to make a **non-learning bot** (rule-based / heuristic control).
- The newer direction is to build a **learning-based RL agent** on top of reusable parsing/state interfaces.

## Current repository structure
- `main.py`:
  - Legacy runtime loop (screen capture, heuristic decisions, key pressing).
  - Useful as behavior baseline.
- `maplestory_parse.py`:
  - Shared parser/state module intended for RL and future bots.
  - Defines keys/actions, thresholds, template loading, profile selection, and `detect_all` state refresh.
  - Loads per-profile controls + parser thresholds from JSON config files under `<P>/RL/config`.
  
- `maplestory_rl.py`:
  - RL launcher/runtime (inference, online training, imitation collection).
  - Uses profile-local persistence (config, datasets, checkpoints, logs).
- `Objects/`, `Objects_magician_teddy_bears/`, `Objects-shadower-tot/`:
  - Template profile folders.
- `temporary-ignore/`:
  - Auxiliary assets not used by the main parser flow.

## Agreed architecture
1. Keep parsing + state representation in `maplestory_parse.py`.
2. Keep RL training/inference logic in `maplestory_rl.py` (or future RL modules).
3. Treat `main.py` as non-learning reference implementation.
4. Use **Option C** training strategy: behavior cloning first (human imitation), then online RL fine-tuning.

## Profile mechanism (important)
The parser uses a single profile name variable and assumes consistent internal structure.

- Profile selector variable: `PROFILE_NAME`
- Runtime switch function: `select_profile(profile_name)`
- Expected folder schema for any profile `<P>`:
  - `<P>/My_Character`
  - `<P>/My_Character/Climbing`
  - `<P>/Monsters`
  - `<P>/Climbing`
  - `<P>/Damage`
  - `<P>/RL/config`
  - `<P>/RL/datasets`
  - `<P>/RL/checkpoints`
  - `<P>/RL/logs`

RL persistence is **per-profile only**. Models, datasets, logs, and hyperparameters for one profile must stay in that profile's `RL/` subtree and are not shared globally.

Only subfolder/file contents vary (e.g., monster classes under `Monsters/`).

## Per-profile config files (source of truth)
Each profile keeps two key config files in `<P>/RL/config`:

- `controls.json` (consumed by `maplestory_parse.py` on `select_profile(...)`):
  - `movement_keys`
  - `attack_keys`
  - `attack_metadata` (per-attack range and targeting metadata)
  - `hp_potion_key`
  - `mp_potion_key`
  - `hp_consume_threshold_percent`
  - `mp_consume_threshold_percent`
  - `buffs` (`key`, `interval_seconds`)

- `hyperparameters.json`:
  - Parser detection thresholds:
    - `MONSTER_MATCH_THRESHOLD`
    - `CHARACTER_MATCH_THRESHOLD`
    - `CLIMBING_MATCH_THRESHOLD`
    - `DAMAGE_THRESHOLD`
  - RL learning/reward parameters used by `maplestory_rl.py`.

If either file is missing or invalid, parser/runtime recreates it with defaults.

## Parsing and memory model
`maplestory_parse.py` provides:

- `DetectionMemory`:
  - Snapshot state used by future RL loops.
  - Includes player/climbing status, monsters, climb objects, damage, HP/MP info, timestamp, profile.
- `DetectedObject`:
  - Generic detected item with pixel and normalized coordinates.
- `detect_all(sct, memory=None, include_damage=True)`:
  - Refreshes full parsed memory for one loop tick.

Current scope intentionally excludes notices/bars OCR in this RL-oriented parsing flow.

## Action model and control rules (important)
The project distinguishes between **actions** and **buff timers**.

### Atomic action sets
- `MOVEMENT_KEYS`
- `ATTACK_KEYS`
- `HP_POTION_KEY`
- `MP_POTION_KEY`
- `IDLE_ACTION` (`"idle"`)

### Attack metadata model
Attack keys are represented separately from attack effectiveness metadata.

- `ATTACK_KEYS`: key names used as atomic actions in the control loop.
- `ATTACK_METADATA`: per-attack definitions with these fields:
  - `key`
  - `x_attack_range`
  - `y_attack_range`
  - `minimum_x_attack_range`
  - `is_at_one_direction`
  - `number_of_monsters`
  - optional `name` (defaults to key when not set)
- `ATTACK_METADATA_BY_KEY`: lookup map for algorithm logic.

Directionality rule:
- If `is_at_one_direction` is `true`, attack effectiveness is symmetric around player X (`-x_attack_range` to `+x_attack_range`).
- If `false`, effectiveness is only in the positive X direction (`0` to `+x_attack_range`).

Helpers exposed by parser:
- `get_attack_metadata(attack_key)`
- `is_target_in_attack_range(player_position, target_position, attack_key)`

Backward-compatibility note:
- `ATTACK_RANGE_X` and `ATTACK_RANGE_Y` remain exported aggregate values for legacy callers.
- They are recomputed from current `ATTACK_METADATA` (max per-axis range) and should be treated as compatibility-only, not as the primary attack model.

### `ACTIONS`
`ACTIONS` is composed of movement + attack + potion keys + `idle`.

- `BUFF_KEYS` must **not** be included in `ACTIONS`.
- `ACTIONS` items are atomic choices that can be stacked per loop.
- Attack metadata attributes are **not** embedded into `ACTIONS`; they are used by higher-level decision logic.

### Buff timers
- `BUFF_KEYS` is a sequence of `(key, interval_seconds)` pairs.
- Buff mechanism behavior: press each buff key every configured interval.
- Helper for scheduling: `init_buff_schedule(...)`.

### Stacking constraints
These constraints are required for future RL policy outputs:

1. Multiple actions may be stacked in one loop tick.
2. **Order matters** for stacked actions.
3. `idle` is unique: if chosen, no other action may be in the same stack.
4. Use:
   - `validate_action_stack(actions)`
   - `normalize_action_stack(actions)`

## HP/MP mechanism retained
The parser keeps HP/MP bar-based detection as part of state:

- `_bar_fill_percent(...)`
- `check_hp_status(...)`
- `check_mp_status(...)`

These feed `DetectionMemory.need_hp`, `DetectionMemory.need_mp`, `hp_percent`, and `mp_percent`.

HP/MP consume thresholds are now profile-configurable via `controls.json` (`hp_consume_threshold_percent`, `mp_consume_threshold_percent`).

## Future-conversation guidance for coding agents
When extending this project:

1. Do not reintroduce heavy logic duplication from `main.py` into RL modules.
2. Reuse `maplestory_parse.py` exports as the single source of state and control constants.
3. Preserve action-stack semantics (`idle` exclusivity, ordered stacks).
4. Keep profile handling generic via profile folder names + consistent structure.
5. Prefer minimal, surgical changes; avoid unrelated refactors.

## Suggested next RL steps
- Define RL observation vector encoder from `DetectionMemory`.
- Define action-head mapping from model outputs to ordered action stacks.
- Add reward shaping and episode reset criteria.
- Add a training loop module that depends on `maplestory_parse.py` only for perception/state.

## RL mode contract (implemented direction)
- `inference`: load profile checkpoint and play without policy updates.
- `online_train`: run inference and update policy online from rewards using the loaded checkpoint as initialization.
- `imitation_collect`: do not send game actions; parse state + observe human keys and learn supervised behavior.

All three modes use the same profile-local model lifecycle.

## Reward policy priorities
- Survival has highest priority and outweighs all other objectives.
- Killing speed and damage throughput are optimized after survival.
- Buff lateness penalty must be linear over time:

$$
penalty_{buff} = \lambda \cdot \max(0, \Delta t)
$$

where $\Delta t$ is delay past the scheduled buff time.

## Hyperparameter policy
- Keep hyperparameters easy to tune in profile-local config files.
- Prefer one explicit JSON config per profile under `<P>/RL/config`.
- Allow runtime overrides, but persist merged values so future runs are consistent.
- Preserve parser threshold keys in `hyperparameters.json` when RL runtime writes updated hyperparameters.
