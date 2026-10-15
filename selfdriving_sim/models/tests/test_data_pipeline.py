import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from models.data_pipeline import (
    CAMERA_SCHEMA,
    TrainingDataError,
    load_camera_dataset,
    sample_weights,
    split_episodes,
)


class WindowBuilderTests(unittest.TestCase):
    def _write_dataset(self, root: Path):
        dataset = root / "camera"
        episode_dir = dataset / "frames" / "episode-a"
        episode_dir.mkdir(parents=True)
        records = []
        for frame_index in range(7):
            filename = f"frame-{frame_index:05d}-view-00.jpg"
            Image.new("RGB", (160, 120), (frame_index * 20, 40, 80)).save(episode_dir / filename)
            records.append({
                "image": f"frames/episode-a/{filename}",
                "episode": "episode-a",
                "view_index": 0,
                "frame_index": frame_index,
                "valid": frame_index != 3,
                "camera_profile_seed": 111,
                "horizontal_mirror": False,
                "steering": frame_index / 10,
                "throttle": 0.5,
                "brake": 1.0 if frame_index == 3 else 0.0,
                "turn_signal": frame_index % 3,
            })
        episode_dir_b = dataset / "frames" / "episode-b"
        episode_dir_b.mkdir(parents=True)
        for frame_index in range(3):
            filename = f"frame-{frame_index:05d}-view-00.jpg"
            Image.new("RGB", (160, 120), (30, frame_index * 30, 90)).save(episode_dir_b / filename)
            records.append({
                "image": f"frames/episode-b/{filename}",
                "episode": "episode-b",
                "view_index": 0,
                "frame_index": frame_index,
                "valid": True,
                "camera_profile_seed": 222,
                "horizontal_mirror": True,
                "steering": -frame_index / 10,
                "throttle": 0.4,
                "brake": 0.0,
                "turn_signal": 2,
            })
        (dataset / "metadata.json").write_text(json.dumps({
            "schema": CAMERA_SCHEMA,
            "frame_count": len(records) - 1,
            "image_size": [160, 120],
        }), encoding="utf-8")
        (dataset / "manifest.jsonl").write_text(
            "\n".join(json.dumps(row) for row in reversed(records)) + "\n",
            encoding="utf-8",
        )
        return dataset

    def test_windows_are_sorted_and_never_cross_invalid_gaps_or_episodes(self):
        with tempfile.TemporaryDirectory() as temporary:
            dataset_dir = self._write_dataset(Path(temporary))

            dataset = load_camera_dataset(dataset_dir)

            self.assertEqual(dataset.images.shape, (9, 120, 160, 3))
            self.assertEqual(dataset.images.dtype, np.uint8)
            self.assertEqual(dataset.sequence_indices.shape, (3, 3))
            self.assertEqual(dataset.episodes.tolist(), ["episode-a", "episode-a", "episode-b"])
            frame_numbers = [int(path.name.split("-")[1]) for path in dataset.image_paths]
            windows = [[frame_numbers[index] for index in row] for row in dataset.sequence_indices]
            self.assertEqual(windows, [[0, 1, 2], [4, 5, 6], [0, 1, 2]])
            self.assertTrue(np.allclose(dataset.labels["steering"].reshape(-1), [0.2, 0.6, -0.2]))
            self.assertEqual(dataset.labels["turn_signal"].tolist(), [2, 0, 2])

    def test_rejects_wrong_camera_shape(self):
        with tempfile.TemporaryDirectory() as temporary:
            dataset_dir = self._write_dataset(Path(temporary))
            image_path = dataset_dir / "frames/episode-a/frame-00000-view-00.jpg"
            Image.new("RGB", (80, 60)).save(image_path)
            with self.assertRaises(TrainingDataError):
                load_camera_dataset(dataset_dir)


class SplitAndWeightTests(unittest.TestCase):
    def test_episode_split_is_deterministic_disjoint_and_contains_all_labels_when_possible(self):
        episodes = np.repeat(np.array([f"episode-{i}" for i in range(5)]), 6)
        labels = {
            "steering": np.tile(np.array([[-0.4], [0.0], [0.3], [0.0], [0.0], [0.0]], dtype=np.float32), (5, 1)),
            "brake": np.tile(np.array([[0.0], [0.2], [0.0], [0.0], [0.0], [0.0]], dtype=np.float32), (5, 1)),
            "turn_signal": np.tile(np.array([0, 1, 2, 0, 0, 0], dtype=np.int32), 5),
        }

        train_a, validation_a = split_episodes(episodes, labels, 0.2, seed=81)
        train_b, validation_b = split_episodes(episodes, labels, 0.2, seed=81)

        self.assertEqual((train_a, validation_a), (train_b, validation_b))
        self.assertTrue(set(train_a).isdisjoint(validation_a))
        self.assertEqual(len(validation_a), 1)
        validation_rows = np.isin(episodes, validation_a)
        self.assertEqual(set(labels["turn_signal"][validation_rows]), {0, 1, 2})
        self.assertTrue(np.any(labels["brake"][validation_rows] > 0.1))

    def test_sample_weights_are_finite_and_emphasize_rare_positive_events(self):
        labels = {
            "steering": np.asarray([[0.0], [0.4], [-0.4], [0.0], [0.0], [0.0]], dtype=np.float32),
            "throttle": np.ones((6, 1), dtype=np.float32) * 0.5,
            "brake": np.asarray([[0.0], [0.0], [0.2], [0.0], [0.0], [0.0]], dtype=np.float32),
            "turn_signal": np.asarray([0, 0, 1, 0, 0, 2], dtype=np.int32),
        }

        weights, details = sample_weights(labels)

        for values in weights.values():
            self.assertTrue(np.all(np.isfinite(values)))
        self.assertGreater(weights["brake"][2], weights["brake"][0])
        self.assertGreater(weights["turn_signal"][2], weights["turn_signal"][0])
        self.assertEqual(details["brake_positive_count"], 1)


if __name__ == "__main__":
    unittest.main()
