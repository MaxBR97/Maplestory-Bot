from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import time

import cv2
import numpy as np


@dataclass(frozen=True)
class AttackMetadata:
    key: str
    x_attack_range: int
    y_attack_range: int
    minimum_x_attack_range: int = 0
    is_at_one_direction: bool = True
    number_of_monsters: int = 1
    name: Optional[str] = None

    @property
    def resolved_name(self) -> str:
        return self.name or self.key


# --- BASE CONFIGURATION ---
SCREEN = {"top": 0, "left": 0, "width": 800, "height": 625}
BAR_REGION = {"top": 590, "left": 0, "width": 560, "height": 35}
NOTICE_REGION = {"top": 370, "left": 500, "width": 300, "height": 110}

DEFAULT_MOVEMENT_KEYS = ("left", "right", "up", "down", "alt")
DEFAULT_ATTACK_KEYS = ("ctrl", "space")
DEFAULT_ATTACK_METADATA: Tuple[AttackMetadata, ...] = (
    AttackMetadata(
        key="ctrl",
        x_attack_range=250,
        y_attack_range=80,
        minimum_x_attack_range=0,
        is_at_one_direction=True,
        number_of_monsters=1,
        name="ctrl",
    ),
    AttackMetadata(
        key="space",
        x_attack_range=250,
        y_attack_range=80,
        minimum_x_attack_range=0,
        is_at_one_direction=True,
        number_of_monsters=1,
        name="space",
    ),
)
DEFAULT_HP_POTION_KEY = "Page Up"
DEFAULT_MP_POTION_KEY = "Insert"
IDLE_ACTION = "idle"
DEFAULT_HP_CONSUME_THRESHOLD_PERCENT = 50.0
DEFAULT_MP_CONSUME_THRESHOLD_PERCENT = 50.0
HP_CONSUME_THRESHOLD_PERCENT = DEFAULT_HP_CONSUME_THRESHOLD_PERCENT
MP_CONSUME_THRESHOLD_PERCENT = DEFAULT_MP_CONSUME_THRESHOLD_PERCENT
HP_BAR_REGION = {"top": 615, "left": 220, "width": 100, "height": 4}
MP_BAR_REGION = {"top": 615, "left": 327, "width": 100, "height": 4}
DEFAULT_MONSTER_MATCH_THRESHOLD = 0.65
DEFAULT_CHARACTER_MATCH_THRESHOLD = 0.65
DEFAULT_CLIMBING_MATCH_THRESHOLD = 0.7
DEFAULT_DAMAGE_THRESHOLD = 0.5
MONSTER_MATCH_THRESHOLD = DEFAULT_MONSTER_MATCH_THRESHOLD
CHARACTER_MATCH_THRESHOLD = DEFAULT_CHARACTER_MATCH_THRESHOLD
CLIMBING_MATCH_THRESHOLD = DEFAULT_CLIMBING_MATCH_THRESHOLD
DAMAGE_THRESHOLD = DEFAULT_DAMAGE_THRESHOLD
DEDUPE_DISTANCE_PX = 20

ATTACK_RANGE_X = 250
ATTACK_RANGE_Y = 80
WINDOW_NAME = "MapleStory Detections"
DEFAULT_BUFF_KEYS = (("8", 299), ("c", 101), ("v", 91), ("b", 103))

MOVEMENT_KEYS = DEFAULT_MOVEMENT_KEYS
ATTACK_KEYS = DEFAULT_ATTACK_KEYS
ATTACK_METADATA: Tuple[AttackMetadata, ...] = DEFAULT_ATTACK_METADATA
ATTACK_METADATA_BY_KEY: Dict[str, AttackMetadata] = {}
HP_POTION_KEY = DEFAULT_HP_POTION_KEY
MP_POTION_KEY = DEFAULT_MP_POTION_KEY
BUFF_KEYS = DEFAULT_BUFF_KEYS
ACTIONS: List[str] = []


def _refresh_action_bindings() -> None:
    global ATTACK_METADATA_BY_KEY
    global ACTIONS
    global ATTACK_RANGE_X
    global ATTACK_RANGE_Y

    ATTACK_METADATA_BY_KEY = {attack.key: attack for attack in ATTACK_METADATA}
    ACTIONS = list(
        dict.fromkeys(
            list(MOVEMENT_KEYS)
            + list(ATTACK_KEYS)
            + [HP_POTION_KEY, MP_POTION_KEY, IDLE_ACTION]
        )
    )

    ATTACK_RANGE_X = 0
    ATTACK_RANGE_Y = 0
    if ATTACK_METADATA:
        ATTACK_RANGE_X = max(attack.x_attack_range for attack in ATTACK_METADATA)
        ATTACK_RANGE_Y = max(attack.y_attack_range for attack in ATTACK_METADATA)


_refresh_action_bindings()


# --- PROFILE CONFIGURATION ---
PROFILE_NAME = "Objects"


def _profile_paths(profile_name: str) -> Dict[str, Path]:
    base_dir = Path(profile_name)
    rl_dir = base_dir / "RL"
    return {
        "profile_dir": base_dir,
        "character_dir": base_dir / "My_Character",
        "character_climbing_dir": base_dir / "My_Character" / "Climbing",
        "monster_dir": base_dir / "Monsters",
        "climbing_dir": base_dir / "Climbing",
        "damage_dir": base_dir / "Damage",
        "rl_dir": rl_dir,
        "rl_config_dir": rl_dir / "config",
        "rl_datasets_dir": rl_dir / "datasets",
        "rl_checkpoints_dir": rl_dir / "checkpoints",
        "rl_logs_dir": rl_dir / "logs",
    }


def get_profile_paths(profile_name: Optional[str] = None) -> Dict[str, Path]:
    return _profile_paths(profile_name or ACTIVE_PROFILE_NAME)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _default_controls_config() -> Dict[str, Any]:
    return {
        "movement_keys": list(DEFAULT_MOVEMENT_KEYS),
        "attack_keys": list(DEFAULT_ATTACK_KEYS),
        "attack_metadata": [
            {
                "key": attack.key,
                "name": attack.name,
                "x_attack_range": int(attack.x_attack_range),
                "y_attack_range": int(attack.y_attack_range),
                "minimum_x_attack_range": int(attack.minimum_x_attack_range),
                "is_at_one_direction": bool(attack.is_at_one_direction),
                "number_of_monsters": int(attack.number_of_monsters),
            }
            for attack in DEFAULT_ATTACK_METADATA
        ],
        "hp_potion_key": DEFAULT_HP_POTION_KEY,
        "mp_potion_key": DEFAULT_MP_POTION_KEY,
        "hp_consume_threshold_percent": float(DEFAULT_HP_CONSUME_THRESHOLD_PERCENT),
        "mp_consume_threshold_percent": float(DEFAULT_MP_CONSUME_THRESHOLD_PERCENT),
        "buffs": [
            {"key": key, "interval_seconds": int(interval_seconds)}
            for key, interval_seconds in DEFAULT_BUFF_KEYS
        ],
    }


def _default_thresholds_config() -> Dict[str, Any]:
    return {
        "MONSTER_MATCH_THRESHOLD": float(DEFAULT_MONSTER_MATCH_THRESHOLD),
        "CHARACTER_MATCH_THRESHOLD": float(DEFAULT_CHARACTER_MATCH_THRESHOLD),
        "CLIMBING_MATCH_THRESHOLD": float(DEFAULT_CLIMBING_MATCH_THRESHOLD),
        "DAMAGE_THRESHOLD": float(DEFAULT_DAMAGE_THRESHOLD),
    }


def _coerce_attack_metadata(raw_entries: Any, attack_keys: Tuple[str, ...]) -> Tuple[AttackMetadata, ...]:
    metadata_by_key: Dict[str, AttackMetadata] = {
        attack.key: attack for attack in DEFAULT_ATTACK_METADATA
    }
    if isinstance(raw_entries, list):
        for entry in raw_entries:
            if not isinstance(entry, dict):
                continue
            key = str(entry.get("key", "")).strip()
            if not key:
                continue
            metadata_by_key[key] = AttackMetadata(
                key=key,
                x_attack_range=int(entry.get("x_attack_range", 250)),
                y_attack_range=int(entry.get("y_attack_range", 80)),
                minimum_x_attack_range=int(entry.get("minimum_x_attack_range", 0)),
                is_at_one_direction=bool(entry.get("is_at_one_direction", True)),
                number_of_monsters=int(entry.get("number_of_monsters", 1)),
                name=(str(entry.get("name")).strip() or None) if "name" in entry else None,
            )

    resolved: List[AttackMetadata] = []
    for attack_key in attack_keys:
        existing = metadata_by_key.get(attack_key)
        if existing is None:
            resolved.append(
                AttackMetadata(
                    key=attack_key,
                    x_attack_range=250,
                    y_attack_range=80,
                    minimum_x_attack_range=0,
                    is_at_one_direction=True,
                    number_of_monsters=1,
                    name=attack_key,
                )
            )
        else:
            resolved.append(existing)
    return tuple(resolved)


def _apply_controls_config(controls_config: Dict[str, Any]) -> None:
    global MOVEMENT_KEYS
    global ATTACK_KEYS
    global ATTACK_METADATA
    global HP_POTION_KEY
    global MP_POTION_KEY
    global HP_CONSUME_THRESHOLD_PERCENT
    global MP_CONSUME_THRESHOLD_PERCENT
    global BUFF_KEYS

    movement_keys_raw = controls_config.get("movement_keys", list(DEFAULT_MOVEMENT_KEYS))
    if isinstance(movement_keys_raw, list):
        movement_keys = tuple(str(key) for key in movement_keys_raw if str(key).strip())
    else:
        movement_keys = DEFAULT_MOVEMENT_KEYS
    MOVEMENT_KEYS = movement_keys or DEFAULT_MOVEMENT_KEYS

    attack_keys_raw = controls_config.get("attack_keys", list(DEFAULT_ATTACK_KEYS))
    if isinstance(attack_keys_raw, list):
        attack_keys = tuple(dict.fromkeys(str(key) for key in attack_keys_raw if str(key).strip()))
    else:
        attack_keys = DEFAULT_ATTACK_KEYS
    ATTACK_KEYS = attack_keys or DEFAULT_ATTACK_KEYS

    ATTACK_METADATA = _coerce_attack_metadata(
        controls_config.get("attack_metadata", []),
        ATTACK_KEYS,
    )

    hp_potion_key = str(controls_config.get("hp_potion_key", DEFAULT_HP_POTION_KEY)).strip()
    mp_potion_key = str(controls_config.get("mp_potion_key", DEFAULT_MP_POTION_KEY)).strip()
    HP_POTION_KEY = hp_potion_key or DEFAULT_HP_POTION_KEY
    MP_POTION_KEY = mp_potion_key or DEFAULT_MP_POTION_KEY
    HP_CONSUME_THRESHOLD_PERCENT = float(
        controls_config.get("hp_consume_threshold_percent", DEFAULT_HP_CONSUME_THRESHOLD_PERCENT)
    )
    MP_CONSUME_THRESHOLD_PERCENT = float(
        controls_config.get("mp_consume_threshold_percent", DEFAULT_MP_CONSUME_THRESHOLD_PERCENT)
    )

    buffs_raw = controls_config.get("buffs", [])
    parsed_buffs: List[Tuple[str, int]] = []
    if isinstance(buffs_raw, list):
        for buff in buffs_raw:
            if not isinstance(buff, dict):
                continue
            buff_key = str(buff.get("key", "")).strip()
            if not buff_key:
                continue
            interval_seconds = int(buff.get("interval_seconds", 0))
            if interval_seconds <= 0:
                continue
            parsed_buffs.append((buff_key, interval_seconds))
    BUFF_KEYS = tuple(parsed_buffs) if parsed_buffs else DEFAULT_BUFF_KEYS

    _refresh_action_bindings()


def _apply_thresholds_config(hparams_config: Dict[str, Any]) -> None:
    global MONSTER_MATCH_THRESHOLD
    global CHARACTER_MATCH_THRESHOLD
    global CLIMBING_MATCH_THRESHOLD
    global DAMAGE_THRESHOLD

    MONSTER_MATCH_THRESHOLD = float(
        hparams_config.get("MONSTER_MATCH_THRESHOLD", DEFAULT_MONSTER_MATCH_THRESHOLD)
    )
    CHARACTER_MATCH_THRESHOLD = float(
        hparams_config.get("CHARACTER_MATCH_THRESHOLD", DEFAULT_CHARACTER_MATCH_THRESHOLD)
    )
    CLIMBING_MATCH_THRESHOLD = float(
        hparams_config.get("CLIMBING_MATCH_THRESHOLD", DEFAULT_CLIMBING_MATCH_THRESHOLD)
    )
    DAMAGE_THRESHOLD = float(
        hparams_config.get("DAMAGE_THRESHOLD", DEFAULT_DAMAGE_THRESHOLD)
    )


def _load_profile_config_files(profile_name: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    profile_paths = _profile_paths(profile_name)
    profile_paths["rl_config_dir"].mkdir(parents=True, exist_ok=True)
    controls_path = profile_paths["rl_config_dir"] / "controls.json"
    hparams_path = profile_paths["rl_config_dir"] / "hyperparameters.json"

    if not controls_path.exists():
        _write_json(controls_path, _default_controls_config())
    if not hparams_path.exists():
        _write_json(hparams_path, _default_thresholds_config())

    try:
        controls_config = json.loads(controls_path.read_text(encoding="utf-8"))
    except Exception:
        controls_config = _default_controls_config()
        _write_json(controls_path, controls_config)

    try:
        hparams_config = json.loads(hparams_path.read_text(encoding="utf-8"))
    except Exception:
        hparams_config = _default_thresholds_config()
        _write_json(hparams_path, hparams_config)

    return controls_config, hparams_config


@dataclass
class DetectedObject:
    object_type: str
    position_px: Tuple[int, int]
    position_norm: Tuple[float, float]
    bbox: Optional[Tuple[int, int, int, int]] = None
    confidence: Optional[float] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DetectionMemory:
    timestamp: float = 0.0
    frame_size: Tuple[int, int] = (SCREEN["width"], SCREEN["height"])
    profile_name: str = "default"
    player: Optional[DetectedObject] = None
    is_climbing: bool = False
    monsters: List[DetectedObject] = field(default_factory=list)
    climbing_objects: List[DetectedObject] = field(default_factory=list)
    hp_percent: float = 0.0
    mp_percent: float = 0.0
    need_hp: bool = False
    need_mp: bool = False
    damage_count: int = 0


ACTIVE_PROFILE_NAME = PROFILE_NAME
_active_paths = _profile_paths(ACTIVE_PROFILE_NAME)
CHARACTER_TEMPLATES_DIR = _active_paths["character_dir"]
CHARACTER_CLIMBING_TEMPLATES_DIR = _active_paths["character_climbing_dir"]
MONSTER_TEMPLATES_DIR = _active_paths["monster_dir"]
CLIMBING_TEMPLATES_DIR = _active_paths["climbing_dir"]
DAMAGE_TEMPLATES_DIR = _active_paths["damage_dir"]

PLAYER_TEMPLATES: List[Tuple[np.ndarray, Tuple[int, int], bool]] = []
MONSTER_TEMPLATES: List[Tuple[np.ndarray, Tuple[int, int]]] = []
CLIMBING_TEMPLATES: List[Tuple[np.ndarray, Tuple[int, int]]] = []
DAMAGE_TEMPLATES: List[Tuple[np.ndarray, Tuple[int, int]]] = []


def _resolve_path_case(path: Path) -> Path:
    if path.exists():
        return path
    path_str = str(path)
    if not path_str:
        return path
    swapped = path_str[0].swapcase() + path_str[1:]
    swapped_path = Path(swapped)
    if swapped_path.exists():
        return swapped_path
    return path


def validate_action_stack(actions: List[str]) -> List[str]:
    normalized_actions = [action for action in actions if action]
    invalid_actions = [action for action in normalized_actions if action not in ACTIONS]
    if invalid_actions:
        raise ValueError(f"Invalid actions: {invalid_actions}. Valid actions: {ACTIONS}")
    if IDLE_ACTION in normalized_actions and len(normalized_actions) > 1:
        raise ValueError(f"'{IDLE_ACTION}' cannot be stacked with other actions.")
    return normalized_actions


def normalize_action_stack(actions: Optional[List[str]]) -> List[str]:
    if actions is None:
        return [IDLE_ACTION]
    normalized_actions = validate_action_stack(actions)
    return normalized_actions if normalized_actions else [IDLE_ACTION]


def init_buff_schedule(buff_keys: Tuple[Tuple[str, int], ...], now: Optional[float] = None) -> Dict[str, float]:
    timestamp = time.time() if now is None else now
    return {key: timestamp + float(interval_seconds) for key, interval_seconds in buff_keys}


def get_attack_metadata(attack_key: str) -> Optional[AttackMetadata]:
    return ATTACK_METADATA_BY_KEY.get(attack_key)


def is_target_in_attack_range(
    player_position: Tuple[int, int],
    target_position: Tuple[int, int],
    attack_key: str,
) -> bool:
    attack = get_attack_metadata(attack_key)
    if attack is None:
        return False

    delta_x = target_position[0] - player_position[0]
    delta_y = abs(target_position[1] - player_position[1])
    abs_delta_x = abs(delta_x)

    if delta_y > attack.y_attack_range:
        return False

    if attack.is_at_one_direction:
        if abs_delta_x > attack.x_attack_range:
            return False
        if abs_delta_x < attack.minimum_x_attack_range:
            return False
        return True

    if delta_x < 0:
        return False
    if delta_x > attack.x_attack_range:
        return False
    if delta_x < attack.minimum_x_attack_range:
        return False
    return True


def load_templates(directory: Path) -> List[Tuple[np.ndarray, Tuple[int, int]]]:
    templates: List[Tuple[np.ndarray, Tuple[int, int]]] = []
    resolved_dir = _resolve_path_case(directory)
    if not resolved_dir.exists() or not resolved_dir.is_dir():
        return templates

    for entry in sorted(resolved_dir.iterdir()):
        if entry.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp"}:
            continue
        tpl = cv2.imread(str(entry), cv2.IMREAD_GRAYSCALE)
        if tpl is None:
            continue
        templates.append((tpl, tpl.shape[::-1]))
    return templates


def load_templates_recursive(directory: Path) -> List[Tuple[np.ndarray, Tuple[int, int]]]:
    templates: List[Tuple[np.ndarray, Tuple[int, int]]] = []
    resolved_dir = _resolve_path_case(directory)
    if not resolved_dir.exists() or not resolved_dir.is_dir():
        return templates

    for entry in sorted(resolved_dir.rglob("*")):
        if entry.is_file() and entry.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
            tpl = cv2.imread(str(entry), cv2.IMREAD_GRAYSCALE)
            if tpl is None:
                continue
            templates.append((tpl, tpl.shape[::-1]))
    return templates


def load_player_templates(
    base_dir: Path,
    climbing_dir: Path,
) -> List[Tuple[np.ndarray, Tuple[int, int], bool]]:
    templates: List[Tuple[np.ndarray, Tuple[int, int], bool]] = []
    resolved_base = _resolve_path_case(base_dir)
    resolved_climbing = _resolve_path_case(climbing_dir)

    if resolved_base.exists() and resolved_base.is_dir():
        for entry in sorted(resolved_base.iterdir()):
            if entry.is_file() and entry.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
                tpl = cv2.imread(str(entry), cv2.IMREAD_GRAYSCALE)
                if tpl is not None:
                    templates.append((tpl, tpl.shape[::-1], False))

    if resolved_climbing.exists() and resolved_climbing.is_dir():
        for entry in sorted(resolved_climbing.rglob("*")):
            if entry.is_file() and entry.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
                tpl = cv2.imread(str(entry), cv2.IMREAD_GRAYSCALE)
                if tpl is not None:
                    templates.append((tpl, tpl.shape[::-1], True))

    return templates


def available_profiles() -> List[str]:
    profiles = {ACTIVE_PROFILE_NAME}
    for entry in Path(".").iterdir():
        if not entry.is_dir():
            continue
        if (entry / "My_Character").exists():
            profiles.add(entry.name)
    return sorted(profiles)


def select_profile(profile_name: str) -> str:
    global ACTIVE_PROFILE_NAME
    global CHARACTER_TEMPLATES_DIR
    global CHARACTER_CLIMBING_TEMPLATES_DIR
    global MONSTER_TEMPLATES_DIR
    global CLIMBING_TEMPLATES_DIR
    global DAMAGE_TEMPLATES_DIR
    global PLAYER_TEMPLATES
    global MONSTER_TEMPLATES
    global CLIMBING_TEMPLATES
    global DAMAGE_TEMPLATES

    ACTIVE_PROFILE_NAME = profile_name
    profile = _profile_paths(profile_name)

    controls_config, hparams_config = _load_profile_config_files(profile_name)
    _apply_controls_config(controls_config)
    _apply_thresholds_config(hparams_config)

    CHARACTER_TEMPLATES_DIR = profile["character_dir"]
    CHARACTER_CLIMBING_TEMPLATES_DIR = profile["character_climbing_dir"]
    MONSTER_TEMPLATES_DIR = profile["monster_dir"]
    CLIMBING_TEMPLATES_DIR = profile["climbing_dir"]
    DAMAGE_TEMPLATES_DIR = profile["damage_dir"]

    PLAYER_TEMPLATES = load_player_templates(CHARACTER_TEMPLATES_DIR, CHARACTER_CLIMBING_TEMPLATES_DIR)
    MONSTER_TEMPLATES = load_templates_recursive(MONSTER_TEMPLATES_DIR)
    CLIMBING_TEMPLATES = load_templates_recursive(CLIMBING_TEMPLATES_DIR)
    DAMAGE_TEMPLATES = load_templates_recursive(DAMAGE_TEMPLATES_DIR)

    return ACTIVE_PROFILE_NAME


def new_memory(profile_name: Optional[str] = None) -> DetectionMemory:
    return DetectionMemory(profile_name=profile_name or ACTIVE_PROFILE_NAME)


def _grab_screen_bgr_gray(sct: Any) -> Tuple[np.ndarray, np.ndarray]:
    monitor = sct.grab(SCREEN)
    img = np.array(monitor)
    bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return bgr, gray


def _normalize_position(x: int, y: int) -> Tuple[float, float]:
    width = max(1, SCREEN["width"])
    height = max(1, SCREEN["height"])
    return (x / width, y / height)


def _make_detected_object(
    object_type: str,
    x: int,
    y: int,
    bbox: Optional[Tuple[int, int, int, int]] = None,
    confidence: Optional[float] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> DetectedObject:
    return DetectedObject(
        object_type=object_type,
        position_px=(x, y),
        position_norm=_normalize_position(x, y),
        bbox=bbox,
        confidence=confidence,
        meta=meta or {},
    )


def _bar_fill_percent(sct: Any, region: Dict[str, int], bar_type: str) -> float:
    img = np.array(sct.grab(region))
    bgr_img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)

    if bar_type == "hp":
        lower_red_1 = np.array([0, 70, 50], dtype=np.uint8)
        upper_red_1 = np.array([10, 255, 255], dtype=np.uint8)
        lower_red_2 = np.array([170, 70, 50], dtype=np.uint8)
        upper_red_2 = np.array([180, 255, 255], dtype=np.uint8)
        mask = cv2.bitwise_or(
            cv2.inRange(hsv_img, lower_red_1, upper_red_1),
            cv2.inRange(hsv_img, lower_red_2, upper_red_2),
        )
    else:
        lower_blue = np.array([90, 70, 40], dtype=np.uint8)
        upper_blue = np.array([130, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv_img, lower_blue, upper_blue)

    column_color_ratio = (mask > 0).mean(axis=0)
    colored_columns = np.where(column_color_ratio > 0.25)[0]

    if len(colored_columns) == 0:
        return 0.0

    rightmost_filled_idx = int(colored_columns.max())
    fill_percent = ((rightmost_filled_idx + 1) / region["width"]) * 100.0
    return float(np.clip(fill_percent, 0.0, 100.0))


def check_hp_status(sct: Any) -> Tuple[bool, float]:
    hp_percent = _bar_fill_percent(sct, HP_BAR_REGION, "hp")
    return hp_percent < HP_CONSUME_THRESHOLD_PERCENT, hp_percent


def check_mp_status(sct: Any) -> Tuple[bool, float]:
    mp_percent = _bar_fill_percent(sct, MP_BAR_REGION, "mp")
    return mp_percent < MP_CONSUME_THRESHOLD_PERCENT, mp_percent


def _get_player_state_from_gray(gray: np.ndarray) -> Tuple[Optional[Tuple[int, int]], bool, float]:
    best_score = CHARACTER_MATCH_THRESHOLD
    best_coords: Optional[Tuple[int, int]] = None
    best_is_climbing = False

    for template, (w, h), is_climbing_template in PLAYER_TEMPLATES:
        if h > gray.shape[0] or w > gray.shape[1]:
            continue

        res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if max_val > best_score:
            best_score = float(max_val)
            best_coords = (int(max_loc[0] + w / 2), int(max_loc[1] + h / 2))
            best_is_climbing = is_climbing_template

    if best_coords is None:
        return None, False, 0.0
    return best_coords, best_is_climbing, best_score


def get_player_state(sct: Any) -> Tuple[Optional[Tuple[int, int]], bool]:
    _, gray = _grab_screen_bgr_gray(sct)
    coords, is_climbing, _ = _get_player_state_from_gray(gray)
    return coords, is_climbing


def _dedupe_points(points: List[Tuple[int, int]], min_distance: int = DEDUPE_DISTANCE_PX) -> List[Tuple[int, int]]:
    deduped: List[Tuple[int, int]] = []
    for point in points:
        if any(np.hypot(point[0] - existing[0], point[1] - existing[1]) < min_distance for existing in deduped):
            continue
        deduped.append(point)
    return deduped


def _detect_monsters_from_gray(gray: np.ndarray) -> List[Tuple[int, int]]:
    found: List[Tuple[int, int]] = []
    for template, (w, h) in MONSTER_TEMPLATES:
        if h > gray.shape[0] or w > gray.shape[1]:
            continue
        res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
        ys, xs = np.where(res >= MONSTER_MATCH_THRESHOLD)
        for y, x in zip(ys, xs):
            found.append((int(x + w / 2), int(y + h / 2)))
    return _dedupe_points(found)


def detect_monsters(sct: Any) -> List[Tuple[int, int]]:
    _, gray = _grab_screen_bgr_gray(sct)
    return _detect_monsters_from_gray(gray)


def _detect_climbing_objects_from_gray(gray: np.ndarray) -> List[Tuple[int, int]]:
    found: List[Tuple[int, int]] = []
    for template, (w, h) in CLIMBING_TEMPLATES:
        if h > gray.shape[0] or w > gray.shape[1]:
            continue
        res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
        ys, xs = np.where(res >= CLIMBING_MATCH_THRESHOLD)
        for y, x in zip(ys, xs):
            found.append((int(x + w / 2), int(y + h / 2)))
    return _dedupe_points(found)


def detect_climbing_objects(sct: Any) -> List[Tuple[int, int]]:
    _, gray = _grab_screen_bgr_gray(sct)
    return _detect_climbing_objects_from_gray(gray)


def _detect_damage_from_gray(gray: np.ndarray) -> int:
    damage_count = 0
    for template, (w, h) in DAMAGE_TEMPLATES:
        if h > gray.shape[0] or w > gray.shape[1]:
            continue
        res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
        ys, _ = np.where(res >= DAMAGE_THRESHOLD)
        damage_count += int(len(ys))
    return damage_count


def detect_damage(sct: Any) -> int:
    _, gray = _grab_screen_bgr_gray(sct)
    return _detect_damage_from_gray(gray)


def detect_all(
    sct: Any,
    memory: Optional[DetectionMemory] = None,
    include_damage: bool = True,
) -> DetectionMemory:
    global CURRENT_MEMORY

    bgr, gray = _grab_screen_bgr_gray(sct)
    height, width = bgr.shape[:2]

    mem = memory or new_memory(ACTIVE_PROFILE_NAME)
    mem.timestamp = time.time()
    mem.frame_size = (width, height)
    mem.profile_name = ACTIVE_PROFILE_NAME

    player_coords, is_climbing, player_score = _get_player_state_from_gray(gray)
    mem.player = (
        _make_detected_object(
            object_type="player",
            x=player_coords[0],
            y=player_coords[1],
            confidence=player_score,
            meta={"is_climbing": is_climbing},
        )
        if player_coords
        else None
    )
    mem.is_climbing = is_climbing

    monster_points = _detect_monsters_from_gray(gray)
    mem.monsters = [
        _make_detected_object("monster", x=mx, y=my)
        for mx, my in monster_points
    ]

    climbing_points = _detect_climbing_objects_from_gray(gray)
    mem.climbing_objects = [
        _make_detected_object("climbing", x=cx, y=cy)
        for cx, cy in climbing_points
    ]

    mem.damage_count = _detect_damage_from_gray(gray) if include_damage else 0

    need_hp, hp_percent = check_hp_status(sct)
    need_mp, mp_percent = check_mp_status(sct)
    mem.need_hp = need_hp
    mem.hp_percent = hp_percent
    mem.need_mp = need_mp
    mem.mp_percent = mp_percent

    CURRENT_MEMORY = mem
    return mem


# Initialize template caches with the default profile.
select_profile(ACTIVE_PROFILE_NAME)
CURRENT_MEMORY = new_memory(ACTIVE_PROFILE_NAME)


__all__ = [
    "AttackMetadata",
    "DetectionMemory",
    "DetectedObject",
    "SCREEN",
    "BAR_REGION",
    "NOTICE_REGION",
    "MOVEMENT_KEYS",
    "ATTACK_KEYS",
    "ATTACK_METADATA",
    "ATTACK_METADATA_BY_KEY",
    "IDLE_ACTION",
    "HP_POTION_KEY",
    "MP_POTION_KEY",
    "HP_CONSUME_THRESHOLD_PERCENT",
    "MP_CONSUME_THRESHOLD_PERCENT",
    "HP_BAR_REGION",
    "MP_BAR_REGION",
    "ACTIONS",
    "MONSTER_MATCH_THRESHOLD",
    "CHARACTER_MATCH_THRESHOLD",
    "CLIMBING_MATCH_THRESHOLD",
    "DAMAGE_THRESHOLD",
    "ATTACK_RANGE_X",
    "ATTACK_RANGE_Y",
    "WINDOW_NAME",
    "BUFF_KEYS",
    "PROFILE_NAME",
    "ACTIVE_PROFILE_NAME",
    "CHARACTER_TEMPLATES_DIR",
    "CHARACTER_CLIMBING_TEMPLATES_DIR",
    "MONSTER_TEMPLATES_DIR",
    "CLIMBING_TEMPLATES_DIR",
    "DAMAGE_TEMPLATES_DIR",
    "PLAYER_TEMPLATES",
    "MONSTER_TEMPLATES",
    "CLIMBING_TEMPLATES",
    "DAMAGE_TEMPLATES",
    "CURRENT_MEMORY",
    "available_profiles",
    "get_profile_paths",
    "select_profile",
    "new_memory",
    "load_templates",
    "load_templates_recursive",
    "load_player_templates",
    "validate_action_stack",
    "normalize_action_stack",
    "init_buff_schedule",
    "get_attack_metadata",
    "is_target_in_attack_range",
    "check_hp_status",
    "check_mp_status",
    "get_player_state",
    "detect_monsters",
    "detect_climbing_objects",
    "detect_damage",
    "detect_all",
]
