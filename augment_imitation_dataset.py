from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

import maplestory_parse as mparse


DEFAULT_DATASET_NAME = "imitation_trajectories.jsonl"


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Augment MapleStory imitation dataset JSONL.")
	parser.add_argument("--input", type=Path, required=True, help="Path to source imitation_trajectories.jsonl")
	parser.add_argument("--output", type=Path, default=None, help="Path to output JSONL (default: *_augmented.jsonl)")
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--idle-to-non-idle-ratio", type=float, default=0.8)
	parser.add_argument("--min-samples-per-action", type=int, default=300)
	parser.add_argument("--hard-monster-count", type=int, default=6)
	parser.add_argument("--hard-hp-percent", type=float, default=40.0)
	parser.add_argument("--hard-mp-percent", type=float, default=25.0)
	parser.add_argument("--hard-state-multiplier", type=float, default=2.0)
	parser.add_argument("--noise-std", type=float, default=0.008)
	parser.add_argument("--noise-probability", type=float, default=0.35)
	parser.add_argument("--preserve-binary-ish", action=argparse.BooleanOptionalAction, default=True)
	return parser.parse_args()


def load_jsonl_rows(path: Path) -> List[Dict[str, Any]]:
	rows: List[Dict[str, Any]] = []
	with path.open("r", encoding="utf-8") as handle:
		for line in handle:
			line = line.strip()
			if not line:
				continue
			try:
				row = json.loads(line)
				if not isinstance(row, dict):
					continue
				if "observation" not in row or "action" not in row:
					continue
				rows.append(row)
			except Exception:
				continue
	return rows


def normalize_action(action: Sequence[str]) -> List[str]:
	try:
		return mparse.normalize_action_stack(list(action))
	except Exception:
		return [mparse.IDLE_ACTION]


def action_key(action: Sequence[str]) -> str:
	return json.dumps(normalize_action(action))


def detect_binary_dims(rows: Sequence[Dict[str, Any]], tolerance_ratio: float = 0.95) -> List[bool]:
	if not rows:
		return []
	obs_dim = len(rows[0].get("observation", []))
	if obs_dim <= 0:
		return []

	counts_binary = np.zeros((obs_dim,), dtype=np.int32)
	counts_total = np.zeros((obs_dim,), dtype=np.int32)
	for row in rows:
		obs = row.get("observation")
		if not isinstance(obs, list) or len(obs) != obs_dim:
			continue
		for i, value in enumerate(obs):
			try:
				f = float(value)
			except Exception:
				continue
			counts_total[i] += 1
			if abs(f - 0.0) < 1e-6 or abs(f - 1.0) < 1e-6:
				counts_binary[i] += 1

	binary_dims: List[bool] = []
	for i in range(obs_dim):
		if counts_total[i] == 0:
			binary_dims.append(False)
			continue
		ratio = float(counts_binary[i]) / float(counts_total[i])
		binary_dims.append(ratio >= tolerance_ratio)
	return binary_dims


def rebalance_actions(
	rows: Sequence[Dict[str, Any]],
	rng: random.Random,
	idle_to_non_idle_ratio: float,
	min_samples_per_action: int,
) -> List[Dict[str, Any]]:
	grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
	for row in rows:
		grouped[action_key(row.get("action", []))].append(row)

	idle_key = json.dumps([mparse.IDLE_ACTION])
	idle_rows = list(grouped.get(idle_key, []))
	non_idle_keys = [k for k in grouped.keys() if k != idle_key]
	non_idle_rows: List[Dict[str, Any]] = []
	for k in non_idle_keys:
		non_idle_rows.extend(grouped[k])

	output: List[Dict[str, Any]] = []
	if non_idle_rows:
		max_idle = int(len(non_idle_rows) * max(0.0, idle_to_non_idle_ratio))
		if len(idle_rows) > max_idle > 0:
			output.extend(rng.sample(idle_rows, k=max_idle))
		elif max_idle == 0:
			pass
		else:
			output.extend(idle_rows)
	else:
		output.extend(idle_rows)

	for key, action_rows in grouped.items():
		if key == idle_key:
			continue
		output.extend(action_rows)
		if len(action_rows) < min_samples_per_action and len(action_rows) > 0:
			needed = min_samples_per_action - len(action_rows)
			output.extend(rng.choices(action_rows, k=needed))

	rng.shuffle(output)
	return output


def is_hard_state(
	row: Dict[str, Any],
	hard_monster_count: int,
	hard_hp_percent: float,
	hard_mp_percent: float,
) -> bool:
	monster_count = int(row.get("monster_count", 0) or 0)
	hp_percent = float(row.get("hp_percent", 100.0) or 100.0)
	mp_percent = float(row.get("mp_percent", 100.0) or 100.0)
	return (
		monster_count >= hard_monster_count
		or hp_percent <= hard_hp_percent
		or mp_percent <= hard_mp_percent
	)


def upsample_hard_states(
	rows: Sequence[Dict[str, Any]],
	rng: random.Random,
	hard_monster_count: int,
	hard_hp_percent: float,
	hard_mp_percent: float,
	hard_state_multiplier: float,
) -> List[Dict[str, Any]]:
	output = list(rows)
	hard_rows = [
		row
		for row in rows
		if is_hard_state(row, hard_monster_count, hard_hp_percent, hard_mp_percent)
	]
	if not hard_rows:
		return output

	multiplier = max(1.0, float(hard_state_multiplier))
	extra_needed = int((multiplier - 1.0) * len(hard_rows))
	if extra_needed > 0:
		output.extend(rng.choices(hard_rows, k=extra_needed))
	rng.shuffle(output)
	return output


def apply_observation_noise(
	rows: Sequence[Dict[str, Any]],
	rng: random.Random,
	noise_std: float,
	noise_probability: float,
	binary_dims: Sequence[bool],
	preserve_binary_ish: bool,
) -> List[Dict[str, Any]]:
	if noise_std <= 0.0 or noise_probability <= 0.0:
		return [dict(row) for row in rows]

	output: List[Dict[str, Any]] = []
	for row in rows:
		new_row = dict(row)
		obs = row.get("observation")
		if not isinstance(obs, list):
			output.append(new_row)
			continue

		obs_array = np.array(obs, dtype=np.float32)
		if rng.random() < noise_probability:
			obs_array = obs_array + np.random.normal(0.0, noise_std, size=obs_array.shape).astype(np.float32)

		for i in range(obs_array.shape[0]):
			if preserve_binary_ish and i < len(binary_dims) and binary_dims[i]:
				obs_array[i] = float(np.clip(obs_array[i], 0.0, 1.0))
			else:
				obs_array[i] = float(np.clip(obs_array[i], -1.5, 1.5))

		new_row["observation"] = [float(x) for x in obs_array.tolist()]
		output.append(new_row)
	return output


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8") as handle:
		for row in rows:
			handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize_actions(rows: Sequence[Dict[str, Any]]) -> Dict[str, int]:
	counts: Dict[str, int] = defaultdict(int)
	for row in rows:
		counts[action_key(row.get("action", []))] += 1
	return dict(sorted(counts.items(), key=lambda item: item[0]))


def derive_default_output_path(input_path: Path) -> Path:
	if input_path.suffix.lower() == ".jsonl":
		return input_path.with_name(f"{input_path.stem}_augmented.jsonl")
	return input_path.with_name(f"{input_path.name}_augmented.jsonl")


def run() -> None:
	args = parse_args()
	if not args.input.exists():
		raise FileNotFoundError(f"Input dataset not found: {args.input}")

	random.seed(args.seed)
	np.random.seed(args.seed)
	rng = random.Random(args.seed)

	rows = load_jsonl_rows(args.input)
	if not rows:
		raise RuntimeError("No valid rows loaded from input dataset.")

	output_path: Path = args.output if args.output is not None else derive_default_output_path(args.input)
	binary_dims = detect_binary_dims(rows)

	rows_rebalanced = rebalance_actions(
		rows,
		rng=rng,
		idle_to_non_idle_ratio=float(args.idle_to_non_idle_ratio),
		min_samples_per_action=max(1, int(args.min_samples_per_action)),
	)
	rows_hard = upsample_hard_states(
		rows_rebalanced,
		rng=rng,
		hard_monster_count=max(1, int(args.hard_monster_count)),
		hard_hp_percent=float(args.hard_hp_percent),
		hard_mp_percent=float(args.hard_mp_percent),
		hard_state_multiplier=float(args.hard_state_multiplier),
	)
	rows_noisy = apply_observation_noise(
		rows_hard,
		rng=rng,
		noise_std=max(0.0, float(args.noise_std)),
		noise_probability=float(np.clip(args.noise_probability, 0.0, 1.0)),
		binary_dims=binary_dims,
		preserve_binary_ish=bool(args.preserve_binary_ish),
	)

	write_jsonl(output_path, rows_noisy)

	src_counts = summarize_actions(rows)
	aug_counts = summarize_actions(rows_noisy)
	print(f"Loaded rows: {len(rows)}")
	print(f"Augmented rows: {len(rows_noisy)}")
	print(f"Input: {args.input}")
	print(f"Output: {output_path}")
	print("Action counts (before -> after):")
	for key in sorted(set(src_counts.keys()) | set(aug_counts.keys())):
		print(f"  {key}: {src_counts.get(key, 0)} -> {aug_counts.get(key, 0)}")


if __name__ == "__main__":
	run()
