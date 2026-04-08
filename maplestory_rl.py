from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

import maplestory_parse as mparse
from maplestory_parse import DetectionMemory


AGENT_VERSION = "0.1.0"
DEFAULT_CHECKPOINT_NAME = "policy_latest.npz"
DEFAULT_HPARAM_FILENAME = "hyperparameters.json"
DEFAULT_DATASET_FILENAME = "imitation_trajectories.jsonl"
DEFAULT_RUN_LOG_FILENAME = "session_log.jsonl"


@dataclass
class HyperParameters:
	epsilon: float = 0.10
	learning_rate_supervised: float = 0.02
	learning_rate_online: float = 0.005
	sequence_window: int = 1
	reward_damage_to_monsters_weight: float = 1.5
	low_hp_penalty: float = -1.2
	low_mp_penalty: float = -0.2
	reward_idle_penalty: float = 0.03
	reward_buff_lateness_weight: float = 0.04
	reward_no_player_penalty: float = 0.5
	max_monster_norm: float = 40.0
	max_damage_norm: float = 15.0


@dataclass
class RLConfig:
	mode: str = "inference"
	profile_name: str = mparse.ACTIVE_PROFILE_NAME
	tick_seconds: float = 0.08
	max_steps: int = 0
	include_damage: bool = True
	save_every_steps: int = 50
	checkpoint_name: str = DEFAULT_CHECKPOINT_NAME
	debug: bool = False
	debug_print_every_steps: int = 1
	hparams: HyperParameters = field(default_factory=HyperParameters)


@dataclass
class RuntimeStats:
	steps: int = 0
	total_reward: float = 0.0
	supervised_updates: int = 0
	online_updates: int = 0
	last_reward: float = 0.0
	started_at: float = 0.0


class DebugTimer:
	def __init__(self, enabled: bool, report_every_steps: int = 20, history_window: int = 120) -> None:
		self.enabled = bool(enabled)
		self.report_every_steps = max(1, int(report_every_steps))
		self.history_window = max(self.report_every_steps, int(history_window))
		self.current_tick_ms: Dict[str, float] = {}
		self._history_ms: Dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=self.history_window))

	def reset_tick(self) -> None:
		if not self.enabled:
			return
		self.current_tick_ms = {}

	def add_ms(self, label: str, duration_ms: float) -> None:
		if not self.enabled:
			return
		value = max(0.0, float(duration_ms))
		self.current_tick_ms[label] = self.current_tick_ms.get(label, 0.0) + value
		self._history_ms[label].append(value)

	def add_seconds(self, label: str, duration_s: float) -> None:
		self.add_ms(label, float(duration_s) * 1000.0)

	def capture_from_dict_ms(self, metrics: Dict[str, float]) -> None:
		if not self.enabled:
			return
		for key, value in metrics.items():
			self.add_ms(key, value)

	def should_report(self, step: int) -> bool:
		return self.enabled and step > 0 and (step % self.report_every_steps == 0)

	def _avg_ms(self, label: str) -> float:
		samples = list(self._history_ms.get(label, []))
		if not samples:
			return 0.0
		return float(sum(samples) / len(samples))

	def _p95_ms(self, label: str) -> float:
		samples = list(self._history_ms.get(label, []))
		if not samples:
			return 0.0
		return float(np.percentile(np.array(samples, dtype=np.float32), 95))

	def format_report(self, step: int, labels: Sequence[str]) -> str:
		parts: List[str] = [f"[timing] step={step}"]
		for label in labels:
			current = self.current_tick_ms.get(label)
			if current is None:
				continue
			avg = self._avg_ms(label)
			p95 = self._p95_ms(label)
			parts.append(f"{label}={current:.1f}ms(avg={avg:.1f},p95={p95:.1f})")
		return " | ".join(parts)


class LinearPolicyModel:
	def __init__(self, input_dim: int, action_dim: int) -> None:
		self.input_dim = input_dim
		self.action_dim = action_dim
		self.weights = np.zeros((input_dim, action_dim), dtype=np.float32)
		self.bias = np.zeros((action_dim,), dtype=np.float32)

	def logits(self, observation: np.ndarray) -> np.ndarray:
		return observation @ self.weights + self.bias

	def probabilities(self, observation: np.ndarray) -> np.ndarray:
		logits = self.logits(observation)
		logits = logits - np.max(logits)
		exp_scores = np.exp(logits)
		total = np.sum(exp_scores)
		if total <= 0:
			return np.full((self.action_dim,), 1.0 / self.action_dim, dtype=np.float32)
		return exp_scores / total

	def predict_action(self, observation: np.ndarray, epsilon: float = 0.0) -> int:
		if epsilon > 0.0 and np.random.random() < epsilon:
			return int(np.random.randint(0, self.action_dim))
		probabilities = self.probabilities(observation)
		return int(np.argmax(probabilities))

	def update_supervised(self, observation: np.ndarray, target_action: int, lr: float) -> float:
		probabilities = self.probabilities(observation)
		gradient = probabilities
		gradient[target_action] -= 1.0
		self.weights -= lr * np.outer(observation, gradient)
		self.bias -= lr * gradient
		loss = -math.log(max(1e-8, float(probabilities[target_action])))
		return float(loss)

	def update_reinforce(self, observation: np.ndarray, action_index: int, reward: float, lr: float) -> float:
		probabilities = self.probabilities(observation)
		advantage = float(reward)
		gradient = probabilities
		gradient[action_index] -= 1.0
		self.weights -= lr * advantage * np.outer(observation, gradient)
		self.bias -= lr * advantage * gradient
		loss = -advantage * math.log(max(1e-8, float(probabilities[action_index])))
		return float(loss)

	def save(self, checkpoint_path: Path, action_templates: Sequence[List[str]]) -> None:
		checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
		np.savez(
			checkpoint_path,
			weights=self.weights,
			bias=self.bias,
			input_dim=np.array([self.input_dim], dtype=np.int32),
			action_dim=np.array([self.action_dim], dtype=np.int32),
			action_templates=np.array([json.dumps(action_templates)], dtype=object),
			agent_version=np.array([AGENT_VERSION], dtype=object),
		)

	@classmethod
	def load(cls, checkpoint_path: Path) -> Optional[Tuple["LinearPolicyModel", List[List[str]]]]:
		if not checkpoint_path.exists():
			return None
		with np.load(checkpoint_path, allow_pickle=True) as data:
			input_dim = int(data["input_dim"][0])
			action_dim = int(data["action_dim"][0])
			model = cls(input_dim=input_dim, action_dim=action_dim)
			model.weights = data["weights"].astype(np.float32)
			model.bias = data["bias"].astype(np.float32)
			templates_raw = str(data["action_templates"][0])
			templates: List[List[str]] = json.loads(templates_raw)
			return model, templates


class ActionExecutor:
	def __init__(self) -> None:
		self.current_actions: List[str] = [mparse.IDLE_ACTION]
		try:
			import pydirectinput  # type: ignore

			pydirectinput.PAUSE = 0.005
			self._pydirectinput = pydirectinput
			self.enabled = True
		except Exception:
			self._pydirectinput = None
			self.enabled = False

	def _press(self, key: str) -> None:
		if self.enabled and self._pydirectinput is not None:
			self._pydirectinput.press(key)

	def _key_down(self, key: str) -> None:
		if self.enabled and self._pydirectinput is not None:
			self._pydirectinput.keyDown(key)

	def _key_up(self, key: str) -> None:
		if self.enabled and self._pydirectinput is not None:
			self._pydirectinput.keyUp(key)

	def apply_action_stack(self, next_actions: List[str]) -> List[str]:
		next_actions = mparse.normalize_action_stack(next_actions)
		potion_keys = {mparse.HP_POTION_KEY, mparse.MP_POTION_KEY}
		next_set = set(next_actions)
		current_set = set(self.current_actions)

		if next_set.intersection(potion_keys):
			for key in self.current_actions:
				if key != mparse.IDLE_ACTION:
					self._key_up(key)
			for key in next_actions:
				if key in potion_keys:
					self._press(key)
			self.current_actions = [mparse.IDLE_ACTION]
			return self.current_actions

		for key in current_set - next_set:
			if key != mparse.IDLE_ACTION:
				self._key_up(key)

		for key in next_actions:
			if key in mparse.ATTACK_KEYS or key == "alt":
				self._press(key)
			elif key != mparse.IDLE_ACTION and key not in current_set:
				self._key_down(key)

		self.current_actions = next_actions
		return self.current_actions

	def release_all(self) -> None:
		for key in mparse.ACTIONS:
			if key != mparse.IDLE_ACTION:
				self._key_up(key)
		self.current_actions = [mparse.IDLE_ACTION]


class HumanKeyObserver:
	KEY_ALIAS = {
		"left": "left",
		"right": "right",
		"up": "up",
		"down": "down",
		"alt": "alt",
		"ctrl": "ctrl",
		"space": "space",
		"Page Up": "page up",
		"page up": "page up",
		"PgUp": "page up",
		"Insert": "insert",
		"insert": "insert",
		"Home": "home",
		"home": "home",
		"Delete": "delete",
		"delete": "delete",
		"Page Down": "page down",
		"page down": "page down",
		"PgDn": "page down",
	}

	def __init__(self) -> None:
		self.available = False
		self._keyboard = None
		try:
			import keyboard  # type: ignore

			self._keyboard = keyboard
			self.available = True
		except Exception:
			self.available = False

	def _is_pressed(self, key: str) -> bool:
		if not self.available or self._keyboard is None:
			return False
		candidates = [
			key,
			self.KEY_ALIAS.get(key, key),
			key.lower(),
			key.upper(),
			key.replace("_", " "),
		]
		seen: set[str] = set()
		for candidate in candidates:
			if not candidate or candidate in seen:
				continue
			seen.add(candidate)
			try:
				if bool(self._keyboard.is_pressed(candidate)):
					return True
			except Exception:
				continue
		return False

	def get_pressed_action_stack(self) -> List[str]:
		if not self.available:
			return [mparse.IDLE_ACTION]

		pressed: List[str] = []
		for key in mparse.ACTIONS:
			if key == mparse.IDLE_ACTION:
				continue
			if self._is_pressed(key):
				pressed.append(key)
		return mparse.normalize_action_stack(pressed)

	def get_pressed_buff_keys(self) -> List[str]:
		keys: List[str] = []
		for key, _ in mparse.BUFF_KEYS:
			if self._is_pressed(key):
				keys.append(key)
		return keys


def default_action_templates() -> List[List[str]]:
	templates: List[List[str]] = [[mparse.IDLE_ACTION]]

	for movement_key in mparse.MOVEMENT_KEYS:
		templates.append([movement_key])
	for attack_key in mparse.ATTACK_KEYS:
		templates.append([attack_key])

	for movement_key in mparse.MOVEMENT_KEYS:
		for attack_key in mparse.ATTACK_KEYS:
			templates.append([movement_key, attack_key])

	templates.append([mparse.HP_POTION_KEY])
	templates.append([mparse.MP_POTION_KEY])

	valid_templates: List[List[str]] = []
	seen: set[str] = set()
	for actions in templates:
		try:
			normalized = mparse.normalize_action_stack(actions)
		except ValueError:
			continue
		serialized = json.dumps(normalized)
		if serialized in seen:
			continue
		seen.add(serialized)
		valid_templates.append(normalized)
	return valid_templates


def ensure_profile_rl_structure(profile_name: str) -> Dict[str, Path]:
	paths = mparse.get_profile_paths(profile_name)
	for key in ("rl_dir", "rl_config_dir", "rl_datasets_dir", "rl_checkpoints_dir", "rl_logs_dir"):
		paths[key].mkdir(parents=True, exist_ok=True)
	return paths


def load_or_create_hparams(config_path: Path) -> HyperParameters:
	defaults = HyperParameters()
	if config_path.exists():
		try:
			raw = json.loads(config_path.read_text(encoding="utf-8"))
			if not isinstance(raw, dict):
				raw = {}
			merged = asdict(defaults)
			merged.update({k: v for k, v in raw.items() if k in merged})
			return HyperParameters(**merged)
		except Exception:
			pass

	config_path.parent.mkdir(parents=True, exist_ok=True)
	config_path.write_text(json.dumps(asdict(defaults), indent=2), encoding="utf-8")
	return defaults


def encode_observation(
	memory: DetectionMemory,
	previous_memory: Optional[DetectionMemory],
	buff_schedule: Dict[str, float],
	now_ts: float,
	hparams: HyperParameters,
) -> np.ndarray:
	has_player = 1.0 if memory.player else 0.0
	player_x = float(memory.player.position_norm[0]) if memory.player else 0.0
	player_y = float(memory.player.position_norm[1]) if memory.player else 0.0
	is_climbing = 1.0 if memory.is_climbing else 0.0

	monster_count = min(1.0, len(memory.monsters) / max(1.0, hparams.max_monster_norm))
	nearest_dx = 0.0
	nearest_dy = 0.0
	avg_dx = 0.0
	avg_dy = 0.0
	if memory.player and memory.monsters:
		px, py = memory.player.position_norm
		deltas = [(monster.position_norm[0] - px, monster.position_norm[1] - py) for monster in memory.monsters]
		nearest = min(deltas, key=lambda delta: math.hypot(delta[0], delta[1]))
		nearest_dx = float(nearest[0])
		nearest_dy = float(nearest[1])
		avg_dx = float(sum(delta[0] for delta in deltas) / len(deltas))
		avg_dy = float(sum(delta[1] for delta in deltas) / len(deltas))

	climbing_count = min(1.0, len(memory.climbing_objects) / 20.0)
	nearest_climb_dx = 0.0
	nearest_climb_dy = 0.0
	if memory.player and memory.climbing_objects:
		px, py = memory.player.position_norm
		climb_deltas = [(climb.position_norm[0] - px, climb.position_norm[1] - py) for climb in memory.climbing_objects]
		closest = min(climb_deltas, key=lambda delta: math.hypot(delta[0], delta[1]))
		nearest_climb_dx = float(closest[0])
		nearest_climb_dy = float(closest[1])

	hp = float(memory.hp_percent / 100.0)
	mp = float(memory.mp_percent / 100.0)
	need_hp = 1.0 if memory.need_hp else 0.0
	need_mp = 1.0 if memory.need_mp else 0.0

	damage_norm = min(1.0, memory.damage_count / max(1.0, hparams.max_damage_norm))

	lateness_total = 0.0
	for due in buff_schedule.values():
		lateness_total += max(0.0, now_ts - due)
	buff_lateness_norm = min(
		1.0,
		lateness_total / (len(mparse.BUFF_KEYS) * 30.0 if mparse.BUFF_KEYS else 1.0),
	)

	day_phase = now_ts % 60.0
	phase_sin = math.sin((2.0 * math.pi * day_phase) / 60.0)
	phase_cos = math.cos((2.0 * math.pi * day_phase) / 60.0)

	vector = np.array(
		[
			has_player,
			player_x,
			player_y,
			is_climbing,
			monster_count,
			nearest_dx,
			nearest_dy,
			avg_dx,
			avg_dy,
			climbing_count,
			nearest_climb_dx,
			nearest_climb_dy,
			hp,
			mp,
			need_hp,
			need_mp,
			damage_norm,
			buff_lateness_norm,
			phase_sin,
			phase_cos,
		],
		dtype=np.float32,
	)
	return vector


def compute_reward(
	memory: DetectionMemory,
	previous_memory: Optional[DetectionMemory],
	action_stack: List[str],
	buff_schedule: Dict[str, float],
	now_ts: float,
	hparams: HyperParameters,
) -> float:
	_ = previous_memory
	reward = 0.0

	if memory.need_hp:
		reward += hparams.low_hp_penalty
	if memory.need_mp:
		reward += hparams.low_mp_penalty

	if memory.player is None:
		reward += hparams.reward_no_player_penalty

	reward += hparams.reward_damage_to_monsters_weight * float(memory.damage_count)

	if action_stack == [mparse.IDLE_ACTION]:
		reward += hparams.reward_idle_penalty

	lateness_penalty = 0.0
	for due in buff_schedule.values():
		lateness_penalty += max(0.0, now_ts - due) * hparams.reward_buff_lateness_weight
	reward += lateness_penalty

	return float(reward)


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("a", encoding="utf-8") as handle:
		handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def stack_observation_window(observations: Sequence[np.ndarray], window_size: int) -> np.ndarray:
	if not observations:
		raise ValueError("observations must not be empty")
	window_size = max(1, int(window_size))
	latest = np.asarray(observations[-1], dtype=np.float32)
	if window_size == 1:
		return latest

	window_obs: List[np.ndarray] = [np.asarray(obs, dtype=np.float32) for obs in observations[-window_size:]]
	if len(window_obs) < window_size:
		padding = [window_obs[0]] * (window_size - len(window_obs))
		window_obs = padding + window_obs
	return np.concatenate(window_obs, axis=0).astype(np.float32)


def replay_dataset_supervised(
	dataset_path: Path,
	policy: LinearPolicyModel,
	action_templates: Sequence[List[str]],
	hparams: HyperParameters,
	sequence_window: int = 1,
	max_rows: int = 5000,
) -> int:
	if not dataset_path.exists():
		return 0

	sequence_window = max(1, int(sequence_window))
	action_index_by_template = {json.dumps(template): index for index, template in enumerate(action_templates)}
	updates = 0
	observation_history: List[np.ndarray] = []
	with dataset_path.open("r", encoding="utf-8") as handle:
		for index, line in enumerate(handle):
			if index >= max_rows:
				break
			line = line.strip()
			if not line:
				continue
			try:
				row = json.loads(line)
				observation = np.array(row["observation"], dtype=np.float32)
				observation_history.append(observation)
				if len(observation_history) > sequence_window:
					observation_history = observation_history[-sequence_window:]
				stacked_observation = stack_observation_window(observation_history, sequence_window)
				template_key = json.dumps(mparse.normalize_action_stack(row["action"]))
				action_index = action_index_by_template.get(template_key)
				if action_index is None:
					continue
				policy.update_supervised(stacked_observation, action_index, hparams.learning_rate_supervised)
				updates += 1
			except Exception:
				continue
	return updates


def resolve_or_create_policy(
	checkpoint_path: Path,
	input_dim: int,
	action_templates: List[List[str]],
) -> Tuple[LinearPolicyModel, List[List[str]]]:
	loaded = LinearPolicyModel.load(checkpoint_path)
	if loaded is None:
		return LinearPolicyModel(input_dim=input_dim, action_dim=len(action_templates)), action_templates

	model, loaded_templates = loaded
	validated_loaded_templates: List[List[str]] = []
	for template in loaded_templates:
		try:
			validated_loaded_templates.append(mparse.normalize_action_stack(template))
		except ValueError:
			return LinearPolicyModel(input_dim=input_dim, action_dim=len(action_templates)), action_templates

	if model.input_dim != input_dim:
		return LinearPolicyModel(input_dim=input_dim, action_dim=len(action_templates)), action_templates
	if len(validated_loaded_templates) != len(action_templates):
		return LinearPolicyModel(input_dim=input_dim, action_dim=len(action_templates)), action_templates
	if {json.dumps(template) for template in validated_loaded_templates} != {
		json.dumps(template) for template in action_templates
	}:
		return LinearPolicyModel(input_dim=input_dim, action_dim=len(action_templates)), action_templates
	return model, validated_loaded_templates


def mode_requires_keyboard_observer(mode: str) -> bool:
	return mode == "imitation_collect"


def _draw_debug_panel(
	frame: np.ndarray,
	memory: DetectionMemory,
	action_stack: List[str],
	config: RLConfig,
	stats: RuntimeStats,
	reward: float,
) -> np.ndarray:
	annotated = frame.copy()
	if memory.player is not None:
		player_x, player_y = memory.player.position_px
		cv2.circle(annotated, (int(player_x), int(player_y)), 10, (255, 0, 0), 2)

	for monster in memory.monsters:
		mx, my = monster.position_px
		cv2.drawMarker(annotated, (int(mx), int(my)), (0, 255, 0), cv2.MARKER_TILTED_CROSS, 16, 2)

	for climbing in memory.climbing_objects:
		cx, cy = climbing.position_px
		cv2.drawMarker(annotated, (int(cx), int(cy)), (0, 255, 255), cv2.MARKER_SQUARE, 12, 1)

	lines = [
		f"mode={config.mode} step={stats.steps} reward={reward:.3f}",
		f"hp={memory.hp_percent:.1f}% mp={memory.mp_percent:.1f}% dmg_to_monsters={memory.damage_count}",
		f"player={'yes' if memory.player else 'no'} monsters={len(memory.monsters)} climbs={len(memory.climbing_objects)}",
		f"action={action_stack}",
	]

	for index, line in enumerate(lines):
		cv2.putText(
			annotated,
			line,
			(10, 22 + (index * 20)),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.55,
			(0, 0, 0),
			1,
			cv2.LINE_AA,
		)

	return annotated


def run_agent(config: RLConfig) -> RuntimeStats:
	if config.mode not in {"inference", "online_train", "imitation_collect"}:
		raise ValueError("mode must be one of: inference, online_train, imitation_collect")

	mparse.select_profile(config.profile_name)
	profile_paths = ensure_profile_rl_structure(config.profile_name)
	hparams_path = profile_paths["rl_config_dir"] / DEFAULT_HPARAM_FILENAME
	raw_hparams: Dict[str, Any] = {}
	try:
		raw_hparams = json.loads(hparams_path.read_text(encoding="utf-8"))
		if not isinstance(raw_hparams, dict):
			raw_hparams = {}
	except Exception:
		raw_hparams = {}
	file_hparams = load_or_create_hparams(hparams_path)
	hparams = file_hparams
	default_hparams = HyperParameters()
	cli_hparams = config.hparams
	merged_values = asdict(file_hparams)
	for key, cli_value in asdict(cli_hparams).items():
		default_value = getattr(default_hparams, key)
		if cli_value != default_value:
			merged_values[key] = cli_value
	hparams = HyperParameters(**merged_values)
	persisted_hparams: Dict[str, Any] = {}
	for parser_threshold_key in (
		"MONSTER_MATCH_THRESHOLD",
		"CHARACTER_MATCH_THRESHOLD",
		"CLIMBING_MATCH_THRESHOLD",
		"DAMAGE_THRESHOLD",
	):
		if parser_threshold_key in raw_hparams:
			persisted_hparams[parser_threshold_key] = raw_hparams[parser_threshold_key]
	persisted_hparams.update(asdict(hparams))
	persisted_hparams.setdefault("MONSTER_MATCH_THRESHOLD", mparse.MONSTER_MATCH_THRESHOLD)
	persisted_hparams.setdefault("CHARACTER_MATCH_THRESHOLD", mparse.CHARACTER_MATCH_THRESHOLD)
	persisted_hparams.setdefault("CLIMBING_MATCH_THRESHOLD", mparse.CLIMBING_MATCH_THRESHOLD)
	persisted_hparams.setdefault("DAMAGE_THRESHOLD", mparse.DAMAGE_THRESHOLD)
	hparams_path.write_text(json.dumps(persisted_hparams, indent=2), encoding="utf-8")
	config.hparams = hparams
	sequence_window = max(1, int(getattr(hparams, "sequence_window", 1)))

	checkpoint_path = profile_paths["rl_checkpoints_dir"] / config.checkpoint_name
	dataset_path = profile_paths["rl_datasets_dir"] / DEFAULT_DATASET_FILENAME
	run_log_path = profile_paths["rl_logs_dir"] / DEFAULT_RUN_LOG_FILENAME

	action_templates = default_action_templates()
	buff_schedule = mparse.init_buff_schedule(mparse.BUFF_KEYS)

	try:
		from mss import mss  # type: ignore
	except Exception as exc:
		raise RuntimeError("`mss` is required to run the RL agent.") from exc

	human_observer = HumanKeyObserver() if mode_requires_keyboard_observer(config.mode) else None
	action_executor = ActionExecutor()
	stats = RuntimeStats(started_at=time.time())
	debug_timer = DebugTimer(enabled=config.debug, report_every_steps=20)
	timing_report_labels = [
		"parse_capture_ms",
		"parse_player_ms",
		"parse_monsters_ms",
		"parse_climbing_ms",
		"parse_damage_ms",
		"parse_hp_bar_ms",
		"parse_mp_bar_ms",
		"capture_parse_total_ms",
		"state_assembly_ms",
		"human_input_ms",
		"dataset_log_ms",
		"supervised_update_ms",
		"buff_maintenance_ms",
		"policy_inference_ms",
		"action_exec_ms",
		"reward_ms",
		"online_update_ms",
		"run_log_ms",
		"debug_render_ms",
		"checkpoint_save_ms",
		"sleep_ms",
		"tick_total_ms",
	]
	window_name = "MapleStory RL Debug"
	debug_window_initialized = False

	policy: Optional[LinearPolicyModel] = None
	try:
		with mss() as sct:
			warmup_memory = mparse.detect_all(sct, include_damage=config.include_damage)
			warmup_obs = encode_observation(warmup_memory, None, buff_schedule, time.time(), hparams)
			policy_input_dim = len(warmup_obs) * sequence_window
			policy, action_templates = resolve_or_create_policy(checkpoint_path, policy_input_dim, action_templates)

			if config.mode in {"inference", "online_train"} and dataset_path.exists():
				replay_updates = replay_dataset_supervised(
					dataset_path=dataset_path,
					policy=policy,
					action_templates=action_templates,
					hparams=hparams,
					sequence_window=sequence_window,
					max_rows=90000,
				)
				if replay_updates > 0:
					policy.save(checkpoint_path, action_templates)

			previous_memory: Optional[DetectionMemory] = None
			recent_observations: List[np.ndarray] = [warmup_obs]
			while True:
				tick_started_at = time.perf_counter()
				debug_timer.reset_tick()
				now_ts = time.time()
				parse_metrics_ms: Dict[str, float] = {}
				memory = mparse.detect_all(
					sct,
					memory=None,
					include_damage=config.include_damage,
					metrics_ms=parse_metrics_ms,
				)
				debug_timer.capture_from_dict_ms(parse_metrics_ms)
				if parse_metrics_ms:
					debug_timer.add_ms("capture_parse_total_ms", sum(parse_metrics_ms.values()))

				state_started_at = time.perf_counter()
				observation = encode_observation(memory, previous_memory, buff_schedule, now_ts, hparams)
				recent_observations.append(observation)
				if len(recent_observations) > sequence_window:
					recent_observations = recent_observations[-sequence_window:]
				stacked_observation = stack_observation_window(recent_observations, sequence_window)
				debug_timer.add_seconds("state_assembly_ms", time.perf_counter() - state_started_at)
				reward = 0.0

				if config.mode == "imitation_collect":
					human_input_started_at = time.perf_counter()
					action_stack = [mparse.IDLE_ACTION]
					if human_observer is not None:
						action_stack = human_observer.get_pressed_action_stack()
						pressed_buff_keys = human_observer.get_pressed_buff_keys()
						for buff_key in pressed_buff_keys:
							for configured_key, interval in mparse.BUFF_KEYS:
								if configured_key == buff_key:
									buff_schedule[buff_key] = now_ts + float(interval)
									break
					debug_timer.add_seconds("human_input_ms", time.perf_counter() - human_input_started_at)

					dataset_log_started_at = time.perf_counter()
					append_jsonl(
						dataset_path,
						{
							"timestamp": now_ts,
							"profile_name": memory.profile_name,
							"mode": config.mode,
							"observation": observation.tolist(),
							"action": action_stack,
							"hp_percent": memory.hp_percent,
							"mp_percent": memory.mp_percent,
							"monster_count": len(memory.monsters),
							"damage_count": memory.damage_count,
						},
					)
					debug_timer.add_seconds("dataset_log_ms", time.perf_counter() - dataset_log_started_at)

					supervised_started_at = time.perf_counter()
					action_key = json.dumps(mparse.normalize_action_stack(action_stack))
					action_index_lookup = {
						json.dumps(template): index for index, template in enumerate(action_templates)
					}
					action_index = action_index_lookup.get(action_key)
					if action_index is not None:
						policy.update_supervised(stacked_observation, action_index, hparams.learning_rate_supervised)
						stats.supervised_updates += 1
					debug_timer.add_seconds("supervised_update_ms", time.perf_counter() - supervised_started_at)

					if config.debug and (stats.steps % max(1, config.debug_print_every_steps) == 0):
						print(
							" | ".join(
								[
									f"mode={config.mode}",
									f"step={stats.steps}",
									f"captured={action_stack}",
									f"player={'yes' if memory.player else 'no'}",
									f"hp={memory.hp_percent:.1f}",
									f"mp={memory.mp_percent:.1f}",
									f"need_hp={memory.need_hp}",
									f"need_mp={memory.need_mp}",
									f"monsters={len(memory.monsters)}",
									f"climbs={len(memory.climbing_objects)}",
									f"damage_to_monsters={memory.damage_count}",
								]
							)
						)
				else:
					buff_maintenance_started_at = time.perf_counter()
					for buff_key, interval in mparse.BUFF_KEYS:
						due = buff_schedule.get(buff_key, now_ts + float(interval))
						if now_ts >= due:
							action_executor._press(buff_key)
							buff_schedule[buff_key] = now_ts + float(interval)
					debug_timer.add_seconds("buff_maintenance_ms", time.perf_counter() - buff_maintenance_started_at)

					inference_started_at = time.perf_counter()
					epsilon = hparams.epsilon if config.mode == "online_train" else 0.0
					action_index = policy.predict_action(stacked_observation, epsilon=epsilon)
					action_stack = action_templates[action_index]
					debug_timer.add_seconds("policy_inference_ms", time.perf_counter() - inference_started_at)

					action_exec_started_at = time.perf_counter()
					action_executor.apply_action_stack(action_stack)
					debug_timer.add_seconds("action_exec_ms", time.perf_counter() - action_exec_started_at)

					reward_started_at = time.perf_counter()
					reward = compute_reward(
						memory=memory,
						previous_memory=previous_memory,
						action_stack=action_stack,
						buff_schedule=buff_schedule,
						now_ts=now_ts,
						hparams=hparams,
					)
					debug_timer.add_seconds("reward_ms", time.perf_counter() - reward_started_at)
					stats.total_reward += reward
					stats.last_reward = reward

					if config.mode == "online_train":
						online_update_started_at = time.perf_counter()
						policy.update_reinforce(stacked_observation, action_index, reward, hparams.learning_rate_online)
						stats.online_updates += 1
						debug_timer.add_seconds("online_update_ms", time.perf_counter() - online_update_started_at)

					if config.debug and (stats.steps % max(1, config.debug_print_every_steps) == 0):
						print(
							" | ".join(
								[
									f"mode={config.mode}",
									f"step={stats.steps}",
									f"action={action_stack}",
									f"reward={reward:.3f}",
									f"hp={memory.hp_percent:.1f}",
									f"mp={memory.mp_percent:.1f}",
									f"need_hp={memory.need_hp}",
									f"need_mp={memory.need_mp}",
									f"monsters={len(memory.monsters)}",
									f"climbs={len(memory.climbing_objects)}",
									f"damage_to_monsters={memory.damage_count}",
								]
							)
						)

					run_log_started_at = time.perf_counter()
					append_jsonl(
						run_log_path,
						{
							"timestamp": now_ts,
							"profile_name": memory.profile_name,
							"mode": config.mode,
							"reward": reward,
							"action": action_stack,
							"hp_percent": memory.hp_percent,
							"mp_percent": memory.mp_percent,
							"monster_count": len(memory.monsters),
							"damage_count": memory.damage_count,
						},
					)
					debug_timer.add_seconds("run_log_ms", time.perf_counter() - run_log_started_at)

				stats.steps += 1

				if config.debug:
					debug_render_started_at = time.perf_counter()
					if not debug_window_initialized:
						cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
						cv2.resizeWindow(window_name, 840, 640)
						cv2.moveWindow(window_name, mparse.SCREEN["width"], 0)
						debug_window_initialized = True
					captured = np.array(sct.grab(mparse.SCREEN))
					frame = cv2.cvtColor(captured, cv2.COLOR_BGRA2BGR)
					panel = _draw_debug_panel(frame, memory, action_stack, config, stats, reward)
					cv2.imshow(window_name, panel)
					debug_timer.add_seconds("debug_render_ms", time.perf_counter() - debug_render_started_at)
					if cv2.waitKey(1) & 0xFF == 27:
						break

				if stats.steps % max(1, config.save_every_steps) == 0:
					save_started_at = time.perf_counter()
					policy.save(checkpoint_path, action_templates)
					debug_timer.add_seconds("checkpoint_save_ms", time.perf_counter() - save_started_at)

				previous_memory = memory
				if config.max_steps > 0 and stats.steps >= config.max_steps:
					debug_timer.add_seconds("tick_total_ms", time.perf_counter() - tick_started_at)
					if debug_timer.should_report(stats.steps):
						print(debug_timer.format_report(stats.steps, timing_report_labels))
					break

				sleep_seconds = max(0.0, config.tick_seconds)
				sleep_started_at = time.perf_counter()
				time.sleep(sleep_seconds)
				debug_timer.add_seconds("sleep_ms", time.perf_counter() - sleep_started_at)
				debug_timer.add_seconds("tick_total_ms", time.perf_counter() - tick_started_at)
				if debug_timer.should_report(stats.steps):
					print(debug_timer.format_report(stats.steps, timing_report_labels))
	finally:
		if policy is not None:
			policy.save(checkpoint_path, action_templates)
		action_executor.release_all()
		if debug_window_initialized:
			cv2.destroyWindow(window_name)

	return stats


def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="MapleStory RL launcher")
	parser.add_argument("--mode", choices=["inference", "online_train", "imitation_collect"], default="inference")
	parser.add_argument("--profile", default=mparse.ACTIVE_PROFILE_NAME)
	parser.add_argument("--tick-seconds", type=float, default=0.08)
	parser.add_argument("--max-steps", type=int, default=0)
	parser.add_argument("--sequence-window", type=int, default=None)
	parser.add_argument("--save-every-steps", type=int, default=50)
	parser.add_argument("--checkpoint-name", type=str, default=DEFAULT_CHECKPOINT_NAME)
	parser.add_argument("--include-damage", action=argparse.BooleanOptionalAction, default=True)
	parser.add_argument("--epsilon", type=float, default=None)
	parser.add_argument("--learning-rate-supervised", type=float, default=None)
	parser.add_argument("--learning-rate-online", type=float, default=None)
	parser.add_argument("--reward-damage-to-monsters-weight", type=float, default=None)
	parser.add_argument("--low-hp-penalty", type=float, default=None)
	parser.add_argument("--low-mp-penalty", type=float, default=None)
	parser.add_argument("--reward-buff-lateness-weight", type=float, default=None)
	parser.add_argument("--debug", action="store_true")
	parser.add_argument("--debug-print-every-steps", type=int, default=1)
	return parser


def config_from_args(args: argparse.Namespace) -> RLConfig:
	config = RLConfig(
		mode=args.mode,
		profile_name=args.profile,
		tick_seconds=args.tick_seconds,
		max_steps=args.max_steps,
		save_every_steps=args.save_every_steps,
		checkpoint_name=args.checkpoint_name,
		include_damage=bool(args.include_damage),
		debug=bool(args.debug),
		debug_print_every_steps=max(1, int(args.debug_print_every_steps)),
	)
	hparams = HyperParameters()
	if args.sequence_window is not None:
		hparams.sequence_window = max(1, int(args.sequence_window))
	if args.epsilon is not None:
		hparams.epsilon = float(args.epsilon)
	if args.learning_rate_supervised is not None:
		hparams.learning_rate_supervised = float(args.learning_rate_supervised)
	if args.learning_rate_online is not None:
		hparams.learning_rate_online = float(args.learning_rate_online)
	if args.reward_damage_to_monsters_weight is not None:
		hparams.reward_damage_to_monsters_weight = float(args.reward_damage_to_monsters_weight)
	if args.low_hp_penalty is not None:
		hparams.low_hp_penalty = float(args.low_hp_penalty)
	if args.low_mp_penalty is not None:
		hparams.low_mp_penalty = float(args.low_mp_penalty)
	if args.reward_buff_lateness_weight is not None:
		hparams.reward_buff_lateness_weight = float(args.reward_buff_lateness_weight)
	config.hparams = hparams
	return config


def run_cli() -> None:
	parser = build_arg_parser()
	args = parser.parse_args()
	config = config_from_args(args)

	stats = run_agent(config)
	elapsed = max(1e-6, time.time() - stats.started_at)
	print(
		" | ".join(
			[
				f"mode={config.mode}",
				f"profile={config.profile_name}",
				f"steps={stats.steps}",
				f"elapsed_s={elapsed:.2f}",
				f"reward_total={stats.total_reward:.3f}",
				f"reward_last={stats.last_reward:.3f}",
				f"supervised_updates={stats.supervised_updates}",
				f"online_updates={stats.online_updates}",
			]
		)
	)


__all__ = [
	"AGENT_VERSION",
	"DEFAULT_CHECKPOINT_NAME",
	"DEFAULT_DATASET_FILENAME",
	"DEFAULT_HPARAM_FILENAME",
	"DEFAULT_RUN_LOG_FILENAME",
	"HyperParameters",
	"RLConfig",
	"RuntimeStats",
	"LinearPolicyModel",
	"ActionExecutor",
	"HumanKeyObserver",
	"default_action_templates",
	"ensure_profile_rl_structure",
	"load_or_create_hparams",
	"encode_observation",
	"compute_reward",
	"append_jsonl",
	"replay_dataset_supervised",
	"resolve_or_create_policy",
	"run_agent",
	"build_arg_parser",
	"config_from_args",
	"run_cli",
]


if __name__ == "__main__":
	run_cli()
   