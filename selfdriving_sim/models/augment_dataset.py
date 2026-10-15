#!/usr/bin/env python3
"""Convert rendered simulator frames into randomized phone-camera views."""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "models" / "data"
FRAME_SIZE = (160, 120)
INPUT_SCHEMA = "selfdrivesim/frame-labels/v1"
OUTPUT_SCHEMA = "selfdrivesim/camera-frames/v1"


class DatasetError(ValueError):
    """A dataset does not match the collector's frame-and-label format."""


def solve_projective_map(destination: list[tuple[float, float]], source: list[tuple[float, float]]) -> tuple[float, ...]:
    """Return PIL perspective coefficients mapping destination points to source points."""
    if len(destination) != 4 or len(source) != 4:
        raise ValueError("A perspective map requires four source and destination corners.")

    matrix: list[list[float]] = []
    for (x, y), (u, v) in zip(destination, source, strict=True):
        matrix.append([x, y, 1, 0, 0, 0, -u * x, -u * y, u])
        matrix.append([0, 0, 0, x, y, 1, -v * x, -v * y, v])

    for column in range(8):
        pivot = max(range(column, 8), key=lambda row: abs(matrix[row][column]))
        if abs(matrix[pivot][column]) < 1e-10:
            raise ValueError("The monitor corners do not define a stable perspective map.")
        matrix[column], matrix[pivot] = matrix[pivot], matrix[column]
        divisor = matrix[column][column]
        matrix[column] = [value / divisor for value in matrix[column]]
        for row in range(8):
            if row == column:
                continue
            factor = matrix[row][column]
            if factor:
                matrix[row] = [
                    value - factor * pivot_value
                    for value, pivot_value in zip(matrix[row], matrix[column], strict=True)
                ]

    return tuple(matrix[row][8] for row in range(8))


def make_room_background(random: np.random.Generator) -> Image.Image:
    """Make a low-detail wall/background for the screen in the phone's field of view."""
    palettes = (
        ((83, 80, 76), (46, 47, 48)),
        ((95, 91, 82), (54, 52, 48)),
        ((74, 83, 89), (41, 47, 51)),
        ((105, 100, 89), (58, 55, 50)),
        ((64, 69, 72), (35, 38, 40)),
    )
    top, bottom = palettes[int(random.integers(0, len(palettes)))]
    gradient = np.linspace(0, 1, FRAME_SIZE[1], dtype=np.float32)[:, None, None]
    start = np.asarray(top, dtype=np.float32)[None, None, :]
    end = np.asarray(bottom, dtype=np.float32)[None, None, :]
    pixels = start * (1 - gradient) + end * gradient
    pixels = np.broadcast_to(pixels, (FRAME_SIZE[1], FRAME_SIZE[0], 3)).copy()
    pixels += random.normal(0, 1.8, pixels.shape[:2])[:, :, None]
    return Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8), "RGB")


def monitor_corners(random: np.random.Generator, width: int, height: int) -> list[tuple[float, float]]:
    """Select a slightly rotated and perspective-skewed 16:9 screen rectangle."""
    center_x = 80 + random.uniform(-4.5, 4.5)
    center_y = 59 + random.uniform(-5, 5)
    angle = math.radians(random.uniform(-3.5, 3.5))
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    half_w, half_h = width / 2, height / 2
    offsets = [
        (-half_w, -half_h),
        (half_w, -half_h),
        (half_w, half_h),
        (-half_w, half_h),
    ]
    skew = random.uniform(0, 3.0)
    result = []
    for index, (x, y) in enumerate(offsets):
        rotated_x = center_x + x * cos_a - y * sin_a
        rotated_y = center_y + x * sin_a + y * cos_a
        top_bias = -skew if index in (0, 1) else skew
        side_bias = -skew if index in (0, 3) else skew
        result.append((rotated_x + side_bias, rotated_y + top_bias))
    return result


def build_camera_profile(seed: int) -> dict:
    """Create one fixed phone/monitor setup for every frame in an episode view."""
    random = np.random.default_rng(seed)
    room = make_room_background(random)
    width = int(random.integers(136, 158))
    height = round(width * 9 / 16)
    bezel = int(random.integers(2, 5))
    corners = monitor_corners(random, width, height)
    source_corners = [
        (0.0, 0.0),
        (width - 1.0, 0.0),
        (width - 1.0, height - 1.0),
        (0.0, height - 1.0),
    ]
    glare = None
    if random.random() < 0.58:
        glare = {
            "center_x": float(random.uniform(18, 145)),
            "center_y": float(random.uniform(12, 105)),
            "angle": float(random.uniform(-0.8, 0.8)),
            "radius_x": float(random.uniform(16, 55)),
            "radius_y": float(random.uniform(5, 18)),
            "strength": float(random.uniform(0.12, 0.42)),
        }

    return {
        "room_background": room,
        "monitor_width": width,
        "monitor_height": height,
        "bezel": bezel,
        "monitor_corners": corners,
        "perspective_coefficients": solve_projective_map(corners, source_corners),
        "panel_brightness": float(random.uniform(0.72, 1.32)),
        "panel_contrast": float(random.uniform(0.82, 1.22)),
        "exposure": float(random.uniform(0.68, 1.38)),
        "contrast": float(random.uniform(0.72, 1.35)),
        "colour": float(random.uniform(0.75, 1.25)),
        "glare": glare,
        "noise_sigma": float(random.uniform(0.5, 7.0)),
        "blur_radius": float(random.uniform(0.15, 1.1)) if random.random() < 0.72 else 0.0,
        "jpeg_quality": int(random.integers(38, 96)),
        "horizontal_mirror": bool(random.integers(0, 2)),
    }


def put_monitor_on_camera(screen: Image.Image, profile: dict) -> Image.Image:
    """Render a screen into a fixed 4:3 phone-camera view."""
    frame = profile["room_background"].convert("RGBA")
    width = profile["monitor_width"]
    height = profile["monitor_height"]
    bezel = profile["bezel"]
    panel = Image.new("RGB", (width, height), (15, 17, 18))

    displayed = ImageOps.fit(
        screen.convert("RGB"),
        (width - bezel * 2, height - bezel * 2),
        method=Image.Resampling.LANCZOS,
        centering=(0.5, 0.5),
    )
    displayed = ImageEnhance.Brightness(displayed).enhance(profile["panel_brightness"])
    displayed = ImageEnhance.Contrast(displayed).enhance(profile["panel_contrast"])
    panel.paste(displayed, (bezel, bezel))
    warped = panel.convert("RGBA").transform(
        FRAME_SIZE,
        Image.Transform.PERSPECTIVE,
        profile["perspective_coefficients"],
        resample=Image.Resampling.BICUBIC,
        fillcolor=(0, 0, 0, 0),
    )
    return Image.alpha_composite(frame, warped).convert("RGB")


def add_glare(image: Image.Image, glare: dict | None) -> Image.Image:
    """Blend a fixed reflected light patch into a camera view, if configured."""
    if glare is None:
        return image

    pixels = np.asarray(image, dtype=np.float32).copy()
    y, x = np.mgrid[0 : FRAME_SIZE[1], 0 : FRAME_SIZE[0]]
    cos_a, sin_a = math.cos(glare["angle"]), math.sin(glare["angle"])
    dx = (x - glare["center_x"]) * cos_a + (y - glare["center_y"]) * sin_a
    dy = -(x - glare["center_x"]) * sin_a + (y - glare["center_y"]) * cos_a
    mask = np.exp(-0.5 * (
        (dx / glare["radius_x"]) ** 2 + (dy / glare["radius_y"]) ** 2
    )) * glare["strength"]
    pixels = pixels * (1 - mask[:, :, None]) + 255 * mask[:, :, None]
    return Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8), "RGB")


def augment_frame(
    screen: Image.Image,
    seed: int,
    profile: dict | None = None,
) -> tuple[Image.Image, int]:
    """Apply a fixed camera profile plus reproducible frame-level sensor noise."""
    random = np.random.default_rng(seed)
    profile = build_camera_profile(seed) if profile is None else profile
    image = put_monitor_on_camera(screen, profile)
    image = add_glare(image, profile["glare"])
    image = ImageEnhance.Brightness(image).enhance(
        profile["exposure"] * float(random.uniform(0.97, 1.03))
    )
    image = ImageEnhance.Contrast(image).enhance(
        profile["contrast"] * float(random.uniform(0.97, 1.03))
    )
    image = ImageEnhance.Color(image).enhance(
        profile["colour"] * float(random.uniform(0.97, 1.03))
    )

    noise_sigma = profile["noise_sigma"] * float(random.uniform(0.9, 1.1))
    pixels = np.asarray(image, dtype=np.float32)
    pixels += random.normal(0, noise_sigma, pixels.shape)
    image = Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8), "RGB")
    if profile["blur_radius"] > 0:
        image = image.filter(ImageFilter.GaussianBlur(radius=profile["blur_radius"]))

    quality = profile["jpeg_quality"]
    compressed = io.BytesIO()
    image.save(compressed, format="JPEG", quality=quality, optimize=True)
    compressed.seek(0)
    return Image.open(compressed).convert("RGB"), quality


def mirror_control_record(record: dict) -> dict:
    """Reflect left/right commands and lateral state when mirroring an image."""
    mirrored = dict(record)
    if "steering" in mirrored:
        mirrored["steering"] = -float(mirrored["steering"])
    if "turn_signal" in mirrored:
        mirrored["turn_signal"] = {0: 0, 1: 2, 2: 1}.get(int(mirrored["turn_signal"]), 0)
    state = mirrored.get("state")
    if isinstance(state, dict):
        state = dict(state)
        if "x" in state:
            state["x"] = -float(state["x"])
        if "heading" in state:
            state["heading"] = -float(state["heading"])
        mirrored["state"] = state
    return mirrored


def resolve_data_path(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    try:
        path.relative_to(DATA_DIR.resolve())
    except ValueError as exc:
        raise DatasetError(f"{label} must be inside {DATA_DIR} so generated data stays ignored by Git.") from exc
    return path


def augment_dataset(input_dir: Path, output_dir: Path, views: int = 1, seed: int = 20261014) -> dict:
    """Write camera-augmented 160×120 JPEGs and a matching manifest."""
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    if not input_dir.is_dir():
        raise DatasetError(f"Input dataset {input_dir} does not exist.")
    if input_dir == output_dir or output_dir.is_relative_to(input_dir):
        raise DatasetError("Output must not be the input dataset or a directory inside it.")
    if output_dir.exists():
        raise DatasetError(f"Output directory {output_dir} already exists.")
    if views < 1 or views > 16:
        raise DatasetError("views must be between 1 and 16.")

    metadata_path = input_dir / "metadata.json"
    manifest_path = input_dir / "manifest.jsonl"
    if not metadata_path.is_file() or not manifest_path.is_file():
        raise DatasetError("Input must contain metadata.json and manifest.jsonl.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema") != INPUT_SCHEMA:
        raise DatasetError(f"Unsupported input schema {metadata.get('schema')!r}.")

    records = []
    for line_number, line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"Invalid manifest JSON on line {line_number}.") from exc
        if record.get("valid", True):
            records.append(record)
    if not records:
        raise DatasetError("No valid frames were found in the input manifest.")

    output_dir.mkdir(parents=True)
    output_frames = output_dir / "frames"
    output_frames.mkdir()
    output_manifest = []
    output_counts = {"off": 0, "left": 0, "right": 0, "brake_positive": 0}
    random = np.random.default_rng(seed)
    camera_profiles: dict[tuple[str, int], tuple[int, dict]] = {}

    for source_index, record in enumerate(records):
        source_path = (input_dir / record["image"]).resolve()
        try:
            source_path.relative_to(input_dir)
        except ValueError as exc:
            raise DatasetError(f"Frame path escapes the dataset directory: {record['image']!r}.") from exc
        if not source_path.is_file():
            raise DatasetError(f"Missing frame {source_path}.")
        with Image.open(source_path) as opened:
            source = opened.convert("RGB")

        episode = str(record.get("episode", "episode-0000"))
        if not episode or Path(episode).name != episode or episode in {".", ".."}:
            raise DatasetError(f"Invalid episode name {episode!r}.")
        episode_directory = output_frames / episode
        episode_directory.mkdir(exist_ok=True)
        for view_index in range(views):
            profile_key = (episode, view_index)
            if profile_key not in camera_profiles:
                profile_seed = int(random.integers(0, 0x7FFF_FFFF))
                camera_profiles[profile_key] = (profile_seed, build_camera_profile(profile_seed))
            profile_seed, profile = camera_profiles[profile_key]
            augmentation_seed = int(random.integers(0, 0x7FFF_FFFF))
            augmented, quality = augment_frame(source, augmentation_seed, profile)
            mirrored = profile["horizontal_mirror"]
            if mirrored:
                augmented = ImageOps.mirror(augmented)
            filename = f"frame-{int(record.get('frame_index', source_index)):05d}-view-{view_index:02d}.jpg"
            relative_image = Path("frames") / episode / filename
            augmented.save(output_dir / relative_image, format="JPEG", quality=95, optimize=True)
            view_record = {
                **record,
                "image": relative_image.as_posix(),
                "source_image": record["image"],
                "view_index": view_index,
                "augmentation_seed": augmentation_seed,
                "camera_profile_seed": profile_seed,
                "jpeg_input_quality": quality,
                "horizontal_mirror": mirrored,
            }
            if mirrored:
                view_record = mirror_control_record(view_record)
            output_manifest.append(json.dumps(view_record, separators=(",", ":")))
            signal = int(view_record.get("turn_signal", 0))
            output_counts[{0: "off", 1: "left", 2: "right"}.get(signal, "off")] += 1
            output_counts["brake_positive"] += int(float(view_record.get("brake", 0)) > 0.1)

    (output_dir / "manifest.jsonl").write_text("\n".join(output_manifest) + "\n", encoding="utf-8")
    try:
        source_name = str(input_dir.relative_to(PROJECT_ROOT))
    except ValueError:
        source_name = input_dir.name
    output_metadata = {
        "schema": OUTPUT_SCHEMA,
        "source_schema": metadata["schema"],
        "source_dataset": source_name,
        "created_from_seed": seed,
        "valid_source_frames": len(records),
        "views_per_frame": views,
        "frame_count": len(output_manifest),
        "frame_rate_hz": metadata.get("frame_rate_hz", 20),
        "image_size": list(FRAME_SIZE),
        "image_format": "JPEG",
        "labels": metadata.get("labels", {}),
        "camera_model": {
            "monitor_aspect_ratio": "16:9",
            "phone_frame_aspect_ratio": "4:3",
            "screen_occupancy_width": "85–99 percent, perspective-skewed",
            "effects": ["room background", "monitor bezel", "view-angle warp", "glare", "brightness", "colour shift", "sensor noise", "blur", "JPEG compression", "episode-consistent horizontal reflection with mirrored control labels"],
            "profile_scope": "room, monitor geometry, display, glare, focus and reflection stay fixed per episode/view; sensor noise and small exposure shifts vary per frame",
            "assumptions": [
                "The source is a rendered screen, not a physical camera recording.",
                "The monitor fills most of the phone frame; crop and mount pose must be recalibrated for the real setup.",
            ],
        },
        "label_counts": output_counts,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(output_metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output_metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Add randomized phone-camera effects to captured simulator frames.")
    parser.add_argument("--input", type=Path, default=DATA_DIR / "sim-dataset-pilot", help="Raw simulator dataset directory.")
    parser.add_argument("--output", type=Path, default=DATA_DIR / "sim-dataset-camera", help="New augmented dataset directory.")
    parser.add_argument("--views", type=int, default=1, help="Augmented views generated per valid source frame, 1..16.")
    parser.add_argument("--seed", type=int, default=20261014, help="Reproducible augmentation seed.")
    args = parser.parse_args(argv)

    try:
        input_dir = resolve_data_path(args.input, "input")
        output_dir = resolve_data_path(args.output, "output")
        result = augment_dataset(input_dir, output_dir, args.views, args.seed)
    except (DatasetError, OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote {result['frame_count']} camera frames to {output_dir}.")
    print(f"Input valid frames: {result['valid_source_frames']}; image size: {result['image_size'][0]}×{result['image_size'][1]}.")
    print(f"Turn-signal counts: {result['label_counts']['off']}/{result['label_counts']['left']}/{result['label_counts']['right']}; brake-positive {result['label_counts']['brake_positive']}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
