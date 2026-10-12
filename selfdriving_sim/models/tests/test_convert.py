"""Tests for models/convert.py. The converter test runs the real tensorflowjs_converter."""

import json
from pathlib import Path

import pytest

import convert
from models.tests.helpers import TINY_INPUT_SHAPE, build_tiny_checkpoint


def test_build_command_puts_the_quantize_flag_last_without_a_value():
    command = convert.build_command("converter", Path("in.h5"), Path("out"))
    assert command[-1] == "--quantize_float16"
    assert command[-3:-1] == ["in.h5", "out"]
    assert "--input_format=keras" in command
    assert "--output_format=tfjs_layers_model" in command


def test_clear_previous_output_removes_only_model_files(tmp_path):
    for name in ("model.json", "group1-shard1of1.bin", "group1-shard9of9.bin", "keep.txt"):
        (tmp_path / name).write_bytes(b"old")

    convert.clear_previous_output(tmp_path)

    assert sorted(path.name for path in tmp_path.iterdir()) == ["keep.txt"]


def test_check_budget_requires_model_json_and_a_total_below_the_limit():
    assert convert.check_budget({"model.json": 10, "a.bin": 5}, limit=16) == 15
    with pytest.raises(convert.ConversionError, match="limit"):
        convert.check_budget({"model.json": 10, "a.bin": 5}, limit=15)
    with pytest.raises(convert.ConversionError, match="model.json"):
        convert.check_budget({"a.bin": 5}, limit=100)


def test_missing_source_does_not_touch_the_existing_output(tmp_path):
    output = tmp_path / "dave2"
    output.mkdir()
    (output / "model.json").write_text("{}")

    with pytest.raises(convert.ConversionError, match="does not exist"):
        convert.convert(tmp_path / "missing.h5", output)
    assert (output / "model.json").exists(), "a failed precondition must not delete the last good model"


def test_missing_converter_binary_is_reported(monkeypatch, tmp_path):
    monkeypatch.setattr(convert.sys, "executable", str(tmp_path / "python"))
    monkeypatch.setattr(convert.shutil, "which", lambda name: None)
    with pytest.raises(convert.ConversionError, match="was not found"):
        convert.find_converter()


@pytest.fixture(scope="module")
def tiny_extended_h5(tmp_path_factory):
    from extend_model import extend_checkpoint

    path = tmp_path_factory.mktemp("extended") / "extended.h5"
    extended, _ = extend_checkpoint(build_tiny_checkpoint(), TINY_INPUT_SHAPE)
    extended.save(path)
    return path


def test_conversion_writes_float16_weights_and_replaces_stale_shards(tmp_path, tiny_extended_h5):
    output = tmp_path / "dave2"
    output.mkdir()
    (output / "group1-shard9of9.bin").write_bytes(b"stale")

    sizes = convert.convert(tiny_extended_h5, output)

    assert "group1-shard9of9.bin" not in sizes
    assert "model.json" in sizes
    manifest = json.loads((output / "model.json").read_text())
    quantization = [entry.get("quantization", {}).get("dtype") for group in manifest["weightsManifest"] for entry in group["weights"]]
    assert quantization and set(quantization) == {"float16"}
    layer_names = {layer["config"]["name"] for layer in manifest["modelTopology"]["model_config"]["config"]["layers"]}
    assert {"steering", "throttle", "brake", "turn_signal"} <= layer_names


def test_over_budget_conversion_leaves_no_model_behind(tmp_path, tiny_extended_h5, monkeypatch):
    monkeypatch.setattr(convert, "MAX_WEB_MODEL_BYTES", 1_000)
    output = tmp_path / "dave2"

    with pytest.raises(convert.ConversionError, match="limit"):
        convert.convert(tiny_extended_h5, output)
    assert list(output.iterdir()) == []
