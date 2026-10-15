import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from models.augment_dataset import (
    DatasetError,
    augment_dataset,
    augment_frame,
    build_camera_profile,
    mirror_control_record,
    solve_projective_map,
)


class PerspectiveMapTests(unittest.TestCase):
    def test_identity_map_keeps_all_four_corners(self):
        points = [(0.0, 0.0), (159.0, 0.0), (159.0, 119.0), (0.0, 119.0)]
        coefficients = solve_projective_map(points, points)
        a, b, c, d, e, f, g, h = coefficients

        for x, y in points:
            denominator = g * x + h * y + 1
            self.assertAlmostEqual((a * x + b * y + c) / denominator, x, places=6)
            self.assertAlmostEqual((d * x + e * y + f) / denominator, y, places=6)

    def test_augmentation_is_reproducible_and_matches_the_phone_frame(self):
        source = Image.new("RGB", (640, 360), (83, 112, 139))
        first, first_quality = augment_frame(source, 97)
        second, second_quality = augment_frame(source, 97)

        self.assertEqual(first.size, (160, 120))
        self.assertEqual(first.mode, "RGB")
        self.assertEqual(first.tobytes(), second.tobytes())
        self.assertEqual(first_quality, second_quality)

    def test_camera_profile_is_reproducible(self):
        first = build_camera_profile(314)
        second = build_camera_profile(314)

        self.assertEqual(first["monitor_corners"], second["monitor_corners"])
        self.assertEqual(first["perspective_coefficients"], second["perspective_coefficients"])
        self.assertEqual(first["horizontal_mirror"], second["horizontal_mirror"])
        self.assertEqual(first["room_background"].tobytes(), second["room_background"].tobytes())

    def test_horizontal_reflection_swaps_steering_and_turn_signal(self):
        original = {
            "steering": 0.3,
            "throttle": 0.5,
            "brake": 0.2,
            "turn_signal": 1,
            "state": {"x": 2.0, "heading": 0.1, "z": -4.0},
        }

        mirrored = mirror_control_record(original)

        self.assertEqual(mirrored["steering"], -0.3)
        self.assertEqual(mirrored["turn_signal"], 2)
        self.assertEqual(mirrored["state"], {"x": -2.0, "heading": -0.1, "z": -4.0})
        self.assertEqual(original["turn_signal"], 1)
        self.assertEqual(mirrored["throttle"], original["throttle"])
        self.assertEqual(mirrored["brake"], original["brake"])


class DatasetWriterTests(unittest.TestCase):
    def test_only_valid_rows_are_written_with_the_camera_shape(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_dir = root / "raw"
            episode_dir = input_dir / "frames" / "episode-0000"
            episode_dir.mkdir(parents=True)
            Image.new("RGB", (640, 360), (30, 60, 90)).save(episode_dir / "frame-00000.jpg")
            Image.new("RGB", (640, 360), (90, 60, 30)).save(episode_dir / "frame-00001.jpg")
            Image.new("RGB", (640, 360), (60, 90, 30)).save(episode_dir / "frame-00002.jpg")
            (input_dir / "metadata.json").write_text(json.dumps({
                "schema": "selfdrivesim/frame-labels/v1",
                "frame_rate_hz": 20,
                "labels": {"turn_signal": "integer"},
            }), encoding="utf-8")
            records = [
                {
                    "image": "frames/episode-0000/frame-00000.jpg",
                    "episode": "episode-0000",
                    "frame_index": 0,
                    "valid": True,
                    "steering": 0.1,
                    "throttle": 0.5,
                    "brake": 0.0,
                    "turn_signal": 0,
                },
                {
                    "image": "frames/episode-0000/frame-00001.jpg",
                    "episode": "episode-0000",
                    "frame_index": 1,
                    "valid": False,
                    "steering": 0,
                    "throttle": 0,
                    "brake": 1,
                    "turn_signal": 2,
                },
                {
                    "image": "frames/episode-0000/frame-00002.jpg",
                    "episode": "episode-0000",
                    "frame_index": 2,
                    "valid": True,
                    "steering": -0.1,
                    "throttle": 0.5,
                    "brake": 0.0,
                    "turn_signal": 0,
                },
            ]
            (input_dir / "manifest.jsonl").write_text(
                "\n".join(json.dumps(record) for record in records) + "\n",
                encoding="utf-8",
            )
            output_dir = root / "camera"

            metadata = augment_dataset(input_dir, output_dir, views=1, seed=8)

            self.assertEqual(metadata["frame_count"], 2)
            self.assertEqual(metadata["valid_source_frames"], 2)
            self.assertEqual(metadata["image_size"], [160, 120])
            self.assertEqual(metadata["label_counts"]["off"], 2)
            self.assertEqual(metadata["label_counts"]["brake_positive"], 0)
            rows = (output_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(rows), 2)
            first, second = (json.loads(row) for row in rows)
            self.assertEqual(first["camera_profile_seed"], second["camera_profile_seed"])
            self.assertEqual(first["horizontal_mirror"], second["horizontal_mirror"])
            self.assertNotEqual(first["augmentation_seed"], second["augmentation_seed"])
            for record in (first, second):
                with Image.open(output_dir / record["image"]) as image:
                    self.assertEqual(image.size, (160, 120))
                    self.assertEqual(image.mode, "RGB")

    def test_output_cannot_overwrite_the_input_dataset(self):
        with tempfile.TemporaryDirectory() as temporary:
            input_dir = Path(temporary) / "raw"
            input_dir.mkdir()
            with self.assertRaises(DatasetError):
                augment_dataset(input_dir, input_dir)


if __name__ == "__main__":
    unittest.main()
