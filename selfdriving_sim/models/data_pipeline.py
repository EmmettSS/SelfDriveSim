"""Load camera frames into contiguous, episode-safe control windows."""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

FRAME_WIDTH = 160
FRAME_HEIGHT = 120
SEQUENCE_LENGTH = 3
CAMERA_SCHEMA = "selfdrivesim/camera-frames/v1"
OUTPUT_NAMES = ("steering", "throttle", "brake", "turn_signal")


class TrainingDataError(ValueError):
    """The camera dataset is incomplete or inconsistent with the training contract."""


@dataclass(frozen=True)
class WindowedDataset:
    """Unique frame pixels plus indices and labels for contiguous sequence windows."""

    images: np.ndarray
    sequence_indices: np.ndarray
    labels: dict[str, np.ndarray]
    episodes: np.ndarray
    metadata: dict
    image_paths: tuple[Path, ...]


def _finite_label(record: dict, name: str, low: float, high: float) -> float:
    try:
        value = float(record[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise TrainingDataError(f"Frame label {name!r} is missing or invalid.") from exc
    if not np.isfinite(value) or value < low or value > high:
        raise TrainingDataError(f"Frame label {name!r} must be finite and in [{low}, {high}].")
    return value


def load_camera_dataset(dataset_dir: str | Path, sequence_length: int = SEQUENCE_LENGTH) -> WindowedDataset:
    """Load valid JPEGs and label only windows that stay contiguous within one episode/view."""
    dataset_dir = Path(dataset_dir).resolve()
    metadata_path = dataset_dir / "metadata.json"
    manifest_path = dataset_dir / "manifest.jsonl"
    if not dataset_dir.is_dir() or not metadata_path.is_file() or not manifest_path.is_file():
        raise TrainingDataError(f"{dataset_dir} must contain metadata.json and manifest.jsonl.")
    if sequence_length < 1:
        raise TrainingDataError("sequence_length must be positive.")

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TrainingDataError("metadata.json is not valid JSON.") from exc
    if metadata.get("schema") != CAMERA_SCHEMA:
        raise TrainingDataError(f"Unsupported camera dataset schema {metadata.get('schema')!r}.")

    records = []
    for line_number, line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TrainingDataError(f"Invalid JSON on manifest line {line_number}.") from exc
        if not record.get("valid", True):
            continue

        image_value = record.get("image")
        if not isinstance(image_value, str):
            raise TrainingDataError(f"Manifest line {line_number} has no image path.")
        image_path = (dataset_dir / image_value).resolve()
        try:
            image_path.relative_to(dataset_dir)
        except ValueError as exc:
            raise TrainingDataError(f"Frame path escapes the dataset directory: {image_value!r}.") from exc

        episode = str(record.get("episode", "episode-0000"))
        view_index = int(record.get("view_index", 0))
        frame_index = int(record.get("frame_index", -1))
        if not episode or view_index < 0 or frame_index < 0:
            raise TrainingDataError(f"Invalid episode, view, or frame index on manifest line {line_number}.")
        signal_raw = record.get("turn_signal")
        try:
            signal_float = float(signal_raw)
        except (TypeError, ValueError) as exc:
            raise TrainingDataError(f"Invalid turn_signal on manifest line {line_number}.") from exc
        if not np.isfinite(signal_float) or not signal_float.is_integer() or int(signal_float) not in (0, 1, 2):
            raise TrainingDataError(f"turn_signal must be 0, 1, or 2 on manifest line {line_number}.")

        row = {
            "image_path": image_path,
            "episode": episode,
            "view_index": view_index,
            "frame_index": frame_index,
            "camera_profile_seed": record.get("camera_profile_seed"),
            "horizontal_mirror": bool(record.get("horizontal_mirror", False)),
            "steering": _finite_label(record, "steering", -1.0, 1.0),
            "throttle": _finite_label(record, "throttle", 0.0, 1.0),
            "brake": _finite_label(record, "brake", 0.0, 1.0),
            "turn_signal": int(signal_float),
        }
        records.append(row)

    if not records:
        raise TrainingDataError("No valid camera frames were found.")

    groups: dict[tuple[str, int], list[dict]] = {}
    seen_frames = set()
    for row in records:
        group_key = (row["episode"], row["view_index"])
        frame_key = (*group_key, row["frame_index"])
        if frame_key in seen_frames:
            raise TrainingDataError(f"Duplicate frame index in episode/view {group_key}: {row['frame_index']}.")
        seen_frames.add(frame_key)
        groups.setdefault(group_key, []).append(row)

    windows: list[list[dict]] = []
    for group_key in sorted(groups):
        group_rows = groups[group_key]
        group_rows.sort(key=lambda row: row["frame_index"])
        segment: list[dict] = []
        previous_index = None
        previous_profile = None
        previous_mirror = None
        for row in group_rows:
            contiguous = previous_index is None or row["frame_index"] == previous_index + 1
            same_camera = (
                not segment
                or (
                    row["camera_profile_seed"] == previous_profile
                    and row["horizontal_mirror"] == previous_mirror
                )
            )
            if not contiguous or not same_camera:
                for start in range(max(0, len(segment) - sequence_length + 1)):
                    windows.append(segment[start : start + sequence_length])
                segment = []
            segment.append(row)
            previous_index = row["frame_index"]
            previous_profile = row["camera_profile_seed"]
            previous_mirror = row["horizontal_mirror"]
        for start in range(max(0, len(segment) - sequence_length + 1)):
            windows.append(segment[start : start + sequence_length])

    if not windows:
        raise TrainingDataError(f"No contiguous {sequence_length}-frame windows were found.")

    unique_paths: list[Path] = []
    path_indices: dict[Path, int] = {}
    sequence_indices = []
    labels = {name: [] for name in OUTPUT_NAMES}
    episode_ids = []
    for window in windows:
        indices = []
        for row in window:
            path = row["image_path"]
            if path not in path_indices:
                path_indices[path] = len(unique_paths)
                unique_paths.append(path)
            indices.append(path_indices[path])
        sequence_indices.append(indices)
        target = window[-1]
        labels["steering"].append([target["steering"]])
        labels["throttle"].append([target["throttle"]])
        labels["brake"].append([target["brake"]])
        labels["turn_signal"].append(target["turn_signal"])
        episode_ids.append(target["episode"])

    images = np.empty((len(unique_paths), FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    for index, path in enumerate(unique_paths):
        if not path.is_file():
            raise TrainingDataError(f"Missing camera frame {path}.")
        try:
            with Image.open(path) as opened:
                image = opened.convert("RGB")
        except OSError as exc:
            raise TrainingDataError(f"Could not decode camera frame {path}.") from exc
        if image.size != (FRAME_WIDTH, FRAME_HEIGHT):
            raise TrainingDataError(
                f"Camera frame {path} is {image.size[0]}×{image.size[1]}, expected 160×120."
            )
        images[index] = np.asarray(image, dtype=np.uint8)

    label_arrays = {
        "steering": np.asarray(labels["steering"], dtype=np.float32),
        "throttle": np.asarray(labels["throttle"], dtype=np.float32),
        "brake": np.asarray(labels["brake"], dtype=np.float32),
        "turn_signal": np.asarray(labels["turn_signal"], dtype=np.int32),
    }
    return WindowedDataset(
        images=images,
        sequence_indices=np.asarray(sequence_indices, dtype=np.int32),
        labels=label_arrays,
        episodes=np.asarray(episode_ids),
        metadata=metadata,
        image_paths=tuple(unique_paths),
    )


def split_episodes(
    episodes: np.ndarray,
    labels: dict[str, np.ndarray],
    validation_fraction: float = 0.2,
    seed: int = 20261014,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Choose a deterministic episode-level holdout with representative label coverage."""
    if not 0.05 <= validation_fraction <= 0.4:
        raise ValueError("validation_fraction must be between 0.05 and 0.4.")
    names = tuple(sorted(set(str(name) for name in episodes)))
    if len(names) < 2:
        raise TrainingDataError("At least two episodes are required for a held-out validation set.")
    validation_count = min(len(names) - 1, max(1, round(len(names) * validation_fraction)))
    candidates = itertools.combinations(names, validation_count)
    if len(names) > 18:
        random = np.random.default_rng(seed)
        candidate_list = [tuple(sorted(random.choice(names, validation_count, replace=False))) for _ in range(4096)]
        candidates = iter(dict.fromkeys(candidate_list))

    signals = labels["turn_signal"].reshape(-1)
    brakes = labels["brake"].reshape(-1) > 0.1
    turning = np.abs(labels["steering"].reshape(-1)) > 0.2
    overall = np.asarray([
        np.count_nonzero(signals == 0),
        np.count_nonzero(signals == 1),
        np.count_nonzero(signals == 2),
        np.count_nonzero(brakes),
        np.count_nonzero(turning),
    ], dtype=np.float64)
    random = np.random.default_rng(seed)
    best_score = float("inf")
    best_validation = None
    for candidate in candidates:
        mask = np.isin(episodes, candidate)
        actual = np.asarray([
            np.count_nonzero(signals[mask] == 0),
            np.count_nonzero(signals[mask] == 1),
            np.count_nonzero(signals[mask] == 2),
            np.count_nonzero(brakes[mask]),
            np.count_nonzero(turning[mask]),
        ], dtype=np.float64)
        target = overall * validation_fraction
        score = float(np.sum((actual - target) ** 2 / (target + 1.0)))
        training_counts = overall - actual
        score += 100.0 * np.count_nonzero((overall > 0) & (actual == 0))
        score += 100.0 * np.count_nonzero((overall > 0) & (training_counts == 0))
        score += random.random() * 1e-8
        if score < best_score:
            best_score = score
            best_validation = tuple(candidate)

    if best_validation is None:
        raise TrainingDataError("Could not construct an episode-level validation split.")
    validation = tuple(sorted(best_validation))
    training = tuple(name for name in names if name not in validation)
    return training, validation


def sample_weights(labels: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict]:
    """Balance rare brake events and signal classes without shifting steering targets."""
    signals = labels["turn_signal"].reshape(-1)
    counts = np.bincount(signals, minlength=3).astype(np.float32)
    turn_weights = np.ones(3, dtype=np.float32)
    present = counts > 0
    turn_weights[present] = len(signals) / (np.count_nonzero(present) * counts[present])
    turn_weights = np.minimum(turn_weights, 3.0)
    turn_weights /= np.mean(turn_weights[signals])

    brake_positive = labels["brake"].reshape(-1) > 0.1
    positive_count = int(np.count_nonzero(brake_positive))
    negative_count = len(brake_positive) - positive_count
    positive_weight = min(4.0, float(np.sqrt((negative_count + 1) / (positive_count + 1))))
    brake_weights = np.where(brake_positive, positive_weight, 1.0).astype(np.float32)
    brake_weights /= np.mean(brake_weights)

    steering_weights = np.ones(len(signals), dtype=np.float32)
    throttle_weights = np.ones(len(signals), dtype=np.float32)
    return {
        "steering": steering_weights,
        "throttle": throttle_weights,
        "brake": brake_weights,
        "turn_signal": turn_weights[signals],
    }, {
        "turn_signal_class_counts": counts.astype(int).tolist(),
        "turn_signal_mean_weights": turn_weights.astype(float).tolist(),
        "brake_positive_count": positive_count,
        "brake_positive_weight": positive_weight,
        "steering_maneuver_weight": 1.0,
    }
