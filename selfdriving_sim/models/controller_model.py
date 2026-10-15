"""Compact, browser/TFLite-friendly multi-output driving controller."""

from __future__ import annotations

import tf_keras as keras

from .dave2_config import (
    BRAKE,
    FRAME_CHANNELS,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    SEQUENCE_LENGTH,
    STEERING,
    THROTTLE,
    TURN_SIGNAL,
    TURN_SIGNAL_CLASSES,
)

MODEL_NAME = "selfdrivesim_lite_controller"
INPUT_SHAPE = (SEQUENCE_LENGTH, FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS)


def build_controller_model() -> keras.Model:
    """Build a small fully-convolutional temporal stack with four named control heads.

    Three raw RGB frames are rearranged into nine channels at each pixel. This
    keeps the existing 20 Hz phone input contract while avoiding recurrent or
    custom layers, making the graph straightforward to export to TFLite and
    TensorFlow.js. The model performs its own 0..255 to 0..1 normalization.
    """
    frames = keras.layers.Input(shape=INPUT_SHAPE, dtype="float32", name="frames")
    x = keras.layers.Rescaling(1.0 / 255.0, name="normalize_pixels")(frames)
    x = keras.layers.Permute((2, 3, 1, 4), name="reorder_time_to_channels")(x)
    x = keras.layers.Reshape(
        (FRAME_HEIGHT, FRAME_WIDTH, SEQUENCE_LENGTH * FRAME_CHANNELS),
        name="stack_temporal_channels",
    )(x)

    x = keras.layers.Conv2D(
        16,
        kernel_size=5,
        strides=2,
        padding="same",
        use_bias=False,
        name="stem_conv",
    )(x)
    x = keras.layers.BatchNormalization(name="stem_bn")(x)
    x = keras.layers.ReLU(name="stem_relu")(x)

    for index, filters in enumerate((24, 32, 48, 64), start=1):
        x = keras.layers.SeparableConv2D(
            filters,
            kernel_size=3,
            strides=2,
            padding="same",
            use_bias=False,
            name=f"separable_block_{index}",
        )(x)
        x = keras.layers.BatchNormalization(name=f"separable_block_{index}_bn")(x)
        x = keras.layers.ReLU(name=f"separable_block_{index}_relu")(x)

    x = keras.layers.Flatten(name="spatial_features")(x)
    x = keras.layers.Dense(64, activation="relu", name="shared_features")(x)
    x = keras.layers.Dropout(0.15, name="head_dropout")(x)

    steering = keras.layers.Dense(1, activation="tanh", name=STEERING)(x)
    throttle = keras.layers.Dense(1, activation="sigmoid", name=THROTTLE)(x)
    brake = keras.layers.Dense(1, activation="sigmoid", name=BRAKE)(x)
    turn_signal = keras.layers.Dense(
        TURN_SIGNAL_CLASSES,
        activation="softmax",
        name=TURN_SIGNAL,
    )(x)
    model = keras.Model(
        inputs=frames,
        outputs=[steering, throttle, brake, turn_signal],
        name=MODEL_NAME,
    )
    validate_controller_model(model)
    return model


def compile_controller_model(model: keras.Model, learning_rate: float = 3e-4) -> None:
    """Compile with robust regression losses and sparse categorical signal loss."""
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss={
            STEERING: keras.losses.Huber(delta=0.15),
            THROTTLE: keras.losses.Huber(delta=0.10),
            BRAKE: keras.losses.Huber(delta=0.10),
            TURN_SIGNAL: keras.losses.SparseCategoricalCrossentropy(),
        },
        loss_weights={STEERING: 2.0, THROTTLE: 0.7, BRAKE: 1.5, TURN_SIGNAL: 0.35},
        metrics={
            STEERING: [keras.metrics.MeanAbsoluteError(name="mae")],
            THROTTLE: [keras.metrics.MeanAbsoluteError(name="mae")],
            BRAKE: [keras.metrics.MeanAbsoluteError(name="mae")],
            TURN_SIGNAL: [keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
        },
    )


def validate_controller_model(model: keras.Model) -> None:
    """Fail fast if an edit breaks the current phone-side input/output contract."""
    expected_input = (None, *INPUT_SHAPE)
    if tuple(model.input_shape) != expected_input:
        raise ValueError(f"Expected model input {expected_input}, found {model.input_shape}.")
    expected_outputs = (STEERING, THROTTLE, BRAKE, TURN_SIGNAL)
    if tuple(model.output_names) != expected_outputs:
        raise ValueError(f"Expected named outputs {expected_outputs}, found {tuple(model.output_names)}.")
    shapes = [tuple(output.shape) for output in model.outputs]
    expected_shapes = [(None, 1), (None, 1), (None, 1), (None, TURN_SIGNAL_CLASSES)]
    if shapes != expected_shapes:
        raise ValueError(f"Expected output shapes {expected_shapes}, found {shapes}.")
