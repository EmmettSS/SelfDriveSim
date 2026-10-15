#!/usr/bin/env python3
"""Train and evaluate the lightweight four-control simulator controller."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import tensorflow as tf
import tf_keras as keras

from .controller_model import INPUT_SHAPE, build_controller_model, compile_controller_model
from .data_pipeline import TrainingDataError, load_camera_dataset, sample_weights, split_episodes
from .dave2_config import BRAKE, DATA_DIR, STEERING, THROTTLE, TURN_SIGNAL

REGRESSION_OUTPUTS = (STEERING, THROTTLE, BRAKE)
SIGNAL_NAMES = ("off", "left", "right")


def build_tf_dataset(
    dataset,
    sample_indices: np.ndarray,
    batch_size: int,
    seed: int,
    weights: dict[str, np.ndarray] | None = None,
    shuffle: bool = False,
):
    """Yield sequence tensors in batches without materializing overlapping windows."""
    output_signature = (
        tf.TensorSpec((INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2], INPUT_SHAPE[3]), tf.float32),
        {
            STEERING: tf.TensorSpec((1,), tf.float32),
            THROTTLE: tf.TensorSpec((1,), tf.float32),
            BRAKE: tf.TensorSpec((1,), tf.float32),
            TURN_SIGNAL: tf.TensorSpec((), tf.int32),
        },
    )
    if weights is not None:
        output_signature = (*output_signature, {
            name: tf.TensorSpec((), tf.float32)
            for name in (STEERING, THROTTLE, BRAKE, TURN_SIGNAL)
        })

    epoch = {"index": 0}

    def generate():
        order = np.asarray(sample_indices, dtype=np.int64).copy()
        if shuffle:
            np.random.default_rng(seed + epoch["index"]).shuffle(order)
            epoch["index"] += 1
        for sample_index in order:
            indices = dataset.sequence_indices[sample_index]
            frames = dataset.images[indices].astype(np.float32)
            targets = {
                STEERING: dataset.labels[STEERING][sample_index],
                THROTTLE: dataset.labels[THROTTLE][sample_index],
                BRAKE: dataset.labels[BRAKE][sample_index],
                TURN_SIGNAL: dataset.labels[TURN_SIGNAL][sample_index],
            }
            if weights is None:
                yield frames, targets
            else:
                sample_weight = {name: weights[name][sample_index] for name in weights}
                yield frames, targets, sample_weight

    result = tf.data.Dataset.from_generator(generate, output_signature=output_signature)
    result = result.apply(tf.data.experimental.assert_cardinality(len(sample_indices)))
    result = result.batch(batch_size, drop_remainder=False).prefetch(1)
    options = tf.data.Options()
    options.threading.private_threadpool_size = 1
    options.threading.max_intra_op_parallelism = 1
    return result.with_options(options)


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _confusion_metrics(true: np.ndarray, predicted: np.ndarray) -> dict:
    matrix = np.zeros((3, 3), dtype=np.int64)
    for actual, guess in zip(true, predicted, strict=True):
        matrix[int(actual), int(guess)] += 1
    classes = {}
    f1_scores = []
    recalls = []
    for index, name in enumerate(SIGNAL_NAMES):
        true_positive = int(matrix[index, index])
        false_positive = int(matrix[:, index].sum() - true_positive)
        false_negative = int(matrix[index, :].sum() - true_positive)
        precision = _safe_divide(true_positive, true_positive + false_positive)
        recall = _safe_divide(true_positive, true_positive + false_negative)
        f1 = _safe_divide(2 * precision * recall, precision + recall)
        support = int(matrix[index, :].sum())
        classes[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        f1_scores.append(f1)
        if support:
            recalls.append(recall)
    return {
        "accuracy": _safe_divide(int(np.trace(matrix)), int(matrix.sum())),
        "macro_f1": float(np.mean(f1_scores)),
        "balanced_accuracy": float(np.mean(recalls)) if recalls else 0.0,
        "class_metrics": classes,
        "confusion_matrix_true_rows_predicted_columns": matrix.tolist(),
    }


def evaluate_predictions(labels: dict[str, np.ndarray], predictions: list[np.ndarray]) -> dict:
    """Compute per-head regression metrics and class-aware event metrics."""
    if len(predictions) != 4:
        raise ValueError(f"Expected four output arrays, found {len(predictions)}.")
    predicted = {
        STEERING: np.asarray(predictions[0]).reshape(-1),
        THROTTLE: np.asarray(predictions[1]).reshape(-1),
        BRAKE: np.asarray(predictions[2]).reshape(-1),
    }
    result = {}
    for name in REGRESSION_OUTPUTS:
        truth = labels[name].reshape(-1)
        error = predicted[name] - truth
        metrics = {
            "mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(np.square(error)))),
            "bias": float(np.mean(error)),
        }
        if name == STEERING:
            maneuver = np.abs(truth) > 0.2
            metrics["maneuver_mae"] = (
                float(np.mean(np.abs(error[maneuver]))) if np.any(maneuver) else None
            )
            metrics["maneuver_support"] = int(np.count_nonzero(maneuver))
        if name == BRAKE:
            actual_positive = truth > 0.1
            predicted_positive = predicted[name] > 0.1
            true_positive = int(np.count_nonzero(actual_positive & predicted_positive))
            false_positive = int(np.count_nonzero(~actual_positive & predicted_positive))
            false_negative = int(np.count_nonzero(actual_positive & ~predicted_positive))
            metrics["positive_threshold"] = 0.1
            metrics["positive_support"] = int(np.count_nonzero(actual_positive))
            metrics["positive_rate"] = float(np.mean(actual_positive))
            metrics["predicted_positive_rate"] = float(np.mean(predicted_positive))
            metrics["positive_precision"] = _safe_divide(true_positive, true_positive + false_positive)
            metrics["positive_recall"] = _safe_divide(true_positive, true_positive + false_negative)
            metrics["positive_f1"] = _safe_divide(
                2 * metrics["positive_precision"] * metrics["positive_recall"],
                metrics["positive_precision"] + metrics["positive_recall"],
            )
            metrics["positive_underprediction_mae"] = (
                float(np.mean(np.maximum(truth[actual_positive] - predicted[name][actual_positive], 0)))
                if np.any(actual_positive)
                else None
            )
            threshold_metrics = {}
            for threshold in (0.02, 0.05, 0.1, 0.2, 0.5):
                threshold_positive = predicted[name] > threshold
                tp = int(np.count_nonzero(actual_positive & threshold_positive))
                fp = int(np.count_nonzero(~actual_positive & threshold_positive))
                fn = int(np.count_nonzero(actual_positive & ~threshold_positive))
                precision = _safe_divide(tp, tp + fp)
                recall = _safe_divide(tp, tp + fn)
                threshold_metrics[f"{threshold:.2f}"] = {
                    "precision": precision,
                    "recall": recall,
                    "f1": _safe_divide(2 * precision * recall, precision + recall),
                    "predicted_positive_rate": float(np.mean(threshold_positive)),
                }
            metrics["threshold_sweep"] = threshold_metrics
        result[name] = metrics

    turn_probability = np.asarray(predictions[3])
    turn_predicted = np.argmax(turn_probability, axis=1)
    result[TURN_SIGNAL] = _confusion_metrics(labels[TURN_SIGNAL].reshape(-1), turn_predicted)
    return result


def _label_summary(labels: dict[str, np.ndarray]) -> dict:
    signals = np.bincount(labels[TURN_SIGNAL].reshape(-1), minlength=3)
    brakes = labels[BRAKE].reshape(-1) > 0.1
    return {
        "samples": int(labels[TURN_SIGNAL].size),
        "turn_signal_counts": {name: int(signals[index]) for index, name in enumerate(SIGNAL_NAMES)},
        "brake_positive": int(np.count_nonzero(brakes)),
        "steering_maneuvers": int(np.count_nonzero(np.abs(labels[STEERING].reshape(-1)) > 0.2)),
    }


def constant_baseline_predictions(training_labels: dict[str, np.ndarray], count: int) -> list[np.ndarray]:
    """Construct a label-only baseline from training-set means and majority signal."""
    predictions = [
        np.full((count, 1), float(np.mean(training_labels[name])), dtype=np.float32)
        for name in REGRESSION_OUTPUTS
    ]
    signal_counts = np.bincount(training_labels[TURN_SIGNAL].reshape(-1), minlength=3)
    majority = int(np.argmax(signal_counts))
    probabilities = np.zeros((count, len(SIGNAL_NAMES)), dtype=np.float32)
    probabilities[:, majority] = 1.0
    predictions.append(probabilities)
    return predictions


def _predict(model, dataset, sample_indices: np.ndarray, batch_size: int, seed: int):
    inputs = build_tf_dataset(dataset, sample_indices, batch_size, seed, shuffle=False)
    # Strip the target dictionary because predict() consumes only the first tuple item.
    inputs = inputs.map(lambda frames, targets: frames, num_parallel_calls=1)
    return model.predict(inputs, steps=math.ceil(len(sample_indices) / batch_size), verbose=0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def train(args) -> dict:
    """Train a new candidate in an ignored data directory; never touch the current web model."""
    dataset_dir = Path(args.dataset).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    data_root = DATA_DIR.resolve()
    try:
        output_dir.relative_to(data_root)
    except ValueError as exc:
        raise ValueError(f"Training output must stay inside {data_root}.") from exc
    if output_dir.exists():
        raise ValueError(f"Output directory already exists: {output_dir}")
    if args.epochs < 1 or args.batch_size < 1 or args.threads < 1:
        raise ValueError("epochs, batch_size, and threads must be positive integers.")

    output_dir.mkdir(parents=True)
    dataset = load_camera_dataset(dataset_dir)
    training_episodes, validation_episodes = split_episodes(
        dataset.episodes,
        dataset.labels,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    training_indices = np.flatnonzero(np.isin(dataset.episodes, training_episodes))
    validation_indices = np.flatnonzero(np.isin(dataset.episodes, validation_episodes))
    if not len(training_indices) or not len(validation_indices):
        raise TrainingDataError("The selected episode split has no training or validation windows.")

    train_labels = {name: values[training_indices] for name, values in dataset.labels.items()}
    validation_labels = {name: values[validation_indices] for name, values in dataset.labels.items()}
    weights, weight_details = sample_weights(train_labels)
    training_sample_weights = {
        name: np.ones(len(dataset.sequence_indices), dtype=np.float32)
        for name in weights
    }
    for name, values in weights.items():
        training_sample_weights[name][training_indices] = values

    os.environ["PYTHONHASHSEED"] = str(args.seed)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    np.random.seed(args.seed)
    tf.keras.utils.set_random_seed(args.seed)

    train_dataset = build_tf_dataset(
        dataset,
        training_indices,
        args.batch_size,
        args.seed,
        weights=training_sample_weights,
        shuffle=True,
    )
    validation_dataset = build_tf_dataset(
        dataset,
        validation_indices,
        args.batch_size,
        args.seed,
        shuffle=False,
    )

    model = build_controller_model()
    compile_controller_model(model, learning_rate=args.learning_rate)
    monitor = getattr(args, "monitor", "val_loss")
    best_model_path = output_dir / "controller_best.h5"
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(best_model_path),
            monitor=monitor,
            mode="min",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor=monitor,
            mode="min",
            patience=args.patience,
            restore_best_weights=False,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            mode="min",
            factor=0.5,
            patience=max(2, args.patience // 2),
            min_lr=1e-5,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(str(output_dir / "training_history.csv")),
    ]

    started = time.monotonic()
    history = model.fit(
        train_dataset,
        steps_per_epoch=math.ceil(len(training_indices) / args.batch_size),
        validation_data=validation_dataset,
        validation_steps=math.ceil(len(validation_indices) / args.batch_size),
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=2,
    )
    training_seconds = time.monotonic() - started
    if not best_model_path.is_file():
        raise RuntimeError("Training ended without a saved best validation checkpoint.")

    best_model = keras.models.load_model(best_model_path, compile=False)
    predictions = _predict(best_model, dataset, validation_indices, args.batch_size, args.seed)
    metrics = evaluate_predictions(validation_labels, predictions)
    baseline_metrics = evaluate_predictions(
        validation_labels,
        constant_baseline_predictions(train_labels, len(validation_indices)),
    )
    losses = history.history.get("val_loss", [])
    monitored_values = history.history.get(monitor, [])
    best_epoch = int(np.argmin(monitored_values)) + 1 if monitored_values else None
    best_monitor_value = float(monitored_values[best_epoch - 1]) if best_epoch else None
    best_loss_at_checkpoint = float(losses[best_epoch - 1]) if best_epoch and losses else None
    beats_constant_baseline = {
        "steering_mae": metrics[STEERING]["mae"] < baseline_metrics[STEERING]["mae"],
        "throttle_mae": metrics[THROTTLE]["mae"] < baseline_metrics[THROTTLE]["mae"],
        "brake_mae": metrics[BRAKE]["mae"] < baseline_metrics[BRAKE]["mae"],
        "brake_positive_recall": (
            metrics[BRAKE]["positive_recall"] > baseline_metrics[BRAKE]["positive_recall"]
        ),
        "turn_signal_accuracy": metrics[TURN_SIGNAL]["accuracy"] > baseline_metrics[TURN_SIGNAL]["accuracy"],
    }
    offline_gate = {
        "regression_heads_beat_constant_mae_baseline": all(
            beats_constant_baseline[name] for name in ("steering_mae", "throttle_mae", "brake_mae")
        ),
        "brake_event_recall_beats_baseline": beats_constant_baseline["brake_positive_recall"],
        "turn_signal_beats_majority_baseline": beats_constant_baseline["turn_signal_accuracy"],
        "passes": all(beats_constant_baseline.values()),
        "note": "An offline comparison only; passing this screen would not establish closed-loop or road safety.",
    }

    manifest_path = dataset_dir / "manifest.jsonl"
    report = {
        "schema": "selfdrivesim/controller-stage2/v1",
        "status": "trained_candidate_not_deployed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data": {
            "dataset": str(dataset_dir),
            "schema": dataset.metadata.get("schema"),
            "manifest_sha256": _sha256(manifest_path),
            "unique_frames_loaded": int(len(dataset.images)),
            "sequence_windows": int(len(dataset.sequence_indices)),
            "training_episodes": list(training_episodes),
            "validation_episodes": list(validation_episodes),
            "training_labels": _label_summary(train_labels),
            "validation_labels": _label_summary(validation_labels),
        },
        "model": {
            "name": best_model.name,
            "input_shape": list(best_model.input_shape),
            "output_names": list(best_model.output_names),
            "parameters": int(best_model.count_params()),
            "architecture": "9-channel three-frame MobileNet-style separable CNN; no recurrent/custom ops",
            "weights_file": best_model_path.name,
            "weights_bytes": best_model_path.stat().st_size,
        },
        "training": {
            "framework": f"TensorFlow {tf.__version__} / tf-keras {keras.__version__}",
            "device": "CPU",
            "host_platform": platform.platform(),
            "cpu_threads": args.threads,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "requested_epochs": args.epochs,
            "epochs_completed": len(history.history.get("loss", [])),
            "checkpoint_monitor": monitor,
            "best_epoch": best_epoch,
            "best_monitor_value": best_monitor_value,
            "best_validation_loss_at_checkpoint": best_loss_at_checkpoint,
            "minimum_validation_loss": float(min(losses)) if losses else None,
            "training_seconds": round(training_seconds, 2),
            "learning_rate": args.learning_rate,
            "sample_weighting": weight_details,
        },
        "validation_metrics": metrics,
        "validation_constant_baseline": baseline_metrics,
        "beats_constant_baseline": beats_constant_baseline,
        "offline_evaluation_gate": offline_gate,
        "limitations": [
            "Imitation learning from the synthetic scripted policy; not a road-safety guarantee.",
            (
                "The candidate detected none of the held-out positive-brake windows at threshold 0.1; "
                "it is not suitable for closed-loop control."
                if metrics[BRAKE]["positive_recall"] == 0
                else "Brake-event recall remains an offline metric and is not a safety guarantee."
            ),
            "No TFLite or TensorFlow.js conversion was performed in this stage.",
            "The existing DAVE-2-GRU checkpoint was not modified or replaced.",
        ],
    }
    (output_dir / "evaluation.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    serializable_history = {
        name: [float(value) for value in values]
        for name, values in history.history.items()
    }
    with (output_dir / "training_history.json").open("w", encoding="utf-8") as stream:
        json.dump(serializable_history, stream, indent=2, ensure_ascii=False)
        stream.write("\n")

    print(f"Candidate saved to {best_model_path}")
    print(f"Parameters: {best_model.count_params():,}; size: {best_model_path.stat().st_size:,} bytes")
    print(f"Training windows: {len(training_indices)}; validation windows: {len(validation_indices)}")
    print(f"Best epoch: {best_epoch}; {monitor}: {best_monitor_value:.6f}; time: {training_seconds:.1f} s")
    print("Validation metrics:")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print("Constant baseline comparison:")
    print(json.dumps({
        "baseline": baseline_metrics,
        "beats_constant_baseline": report["beats_constant_baseline"],
        "offline_evaluation_gate": report["offline_evaluation_gate"],
    }, indent=2, ensure_ascii=False))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a compact four-output controller from simulator captures.")
    parser.add_argument("--dataset", type=Path, default=DATA_DIR / "selfdrivesim-stage1-5000-camera")
    parser.add_argument("--output", type=Path, default=DATA_DIR / "controller-stage2-run")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--monitor", choices=("val_loss", "val_steering_mae"), default="val_loss")
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--threads", type=int, default=min(2, os.cpu_count() or 1))
    args = parser.parse_args(argv)
    try:
        train(args)
    except (TrainingDataError, OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
