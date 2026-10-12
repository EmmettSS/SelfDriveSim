"""Constants shared by the DAVE-2-GRU scripts and the phone-side inference.

The checkpoint facts come from the Hugging Face model card, from each
checkpoint's ``meta.json``, and from the training code that produced them. The
SHA-256 digests and sizes come from the Hugging Face LFS metadata for the pinned
revision. Change a value here and the download, extension, conversion and tests
all follow.
"""

from dataclasses import dataclass
from pathlib import Path

HF_REPO_ID = "maxim-igenbergs/dave2-gru"
HF_MODEL_PAGE = f"https://huggingface.co/{HF_REPO_ID}"

# Commit the checkpoints are pinned to. Hugging Face files at a commit do not change.
HF_REVISION = "dddcf9fc24fc258bb809f276c021d4c66aa4ceb5"
CHECKPOINT_FILENAME = "best_model.h5"

# Where the checkpoint's architecture and training pipeline are published. The
# code answers questions the model card leaves open: how frames are windowed and
# what pixel values the network is fed.
TRAINING_SOURCE = "https://github.com/MaximIgenbergs/thesis-robust-e2e-vision-ads"
TRAINING_SOURCE_MODEL = "scripts/udacity/models/dave2_gru/model.py"
TRAINING_SOURCE_DATA = "scripts/udacity/models/dave2_gru/utils/data_stream.py"


@dataclass(frozen=True)
class Checkpoint:
    """One trained checkpoint stored in its own folder of the Hugging Face repository."""

    folder: str
    sha256: str
    size_bytes: int
    train_sequences: int
    val_sequences: int


CHECKPOINTS = {
    "genroads": Checkpoint(
        folder="genroads_20251215-174930",
        sha256="fbee0740ce53b0ad7be35b708c0242fc98143699f8f7024af0d21d28719a32bf",
        size_bytes=33_328_880,
        train_sequences=24_604,
        val_sequences=7_684,
    ),
    "jungle": Checkpoint(
        folder="jungle_20251201-142321",
        sha256="aa3ec2e3e2926828a7e566a6fc2cd49eed5cfecab5c6bd2bc0829ddbe1aef068",
        size_bytes=33_328_880,
        train_sequences=1_170,
        val_sequences=292,
    ),
}
DEFAULT_CHECKPOINT = "genroads"

# Input contract from meta.json: a window of 3 frames, each 120 pixels high, 160
# wide and RGB. The model card's 66x200 diagram does not match the trained weights.
SEQUENCE_LENGTH = 3
FRAME_HEIGHT = 120
FRAME_WIDTH = 160
FRAME_CHANNELS = 3
INPUT_SHAPE = (SEQUENCE_LENGTH, FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS)

# The frames of one window are consecutive. In the training pipeline STRIDE is the
# step between window *starts* (windows overlap), not a gap inside a window:
# iter_windows() takes frames[start:start + seq_len]. The inference adapter feeds
# its frame buffer one frame at a time, the same way.
FRAME_STRIDE = 1

# Pixels are fed to the checkpoint as the raw 0..255 values of the recorded JPEGs,
# and the network divides by 255 itself in a Lambda layer named "norm". That layer
# cannot run in the browser, so extend_model.py folds the division into the first
# convolution and the converted model keeps the same 0..255 input. The phone
# therefore sends canvas pixels without scaling them.
PIXEL_MAX = 255.0
CHECKPOINT_NORMALIZATION = 1.0 / PIXEL_MAX
NORMALIZATION_LAYER_NAME = "norm"

# Output heads in the order the phone receives them.
STEERING = "steering"
THROTTLE = "throttle"
BRAKE = "brake"
TURN_SIGNAL = "turn_signal"
OUTPUT_NAMES = (STEERING, THROTTLE, BRAKE, TURN_SIGNAL)
TURN_SIGNAL_CLASSES = 3  # 0 = off, 1 = left, 2 = right

# The trained head is Dense(2, linear): steering and throttle come out of the
# network as plain regression values, and the phone clamps them to the protocol
# ranges [-1, 1] and [0, 1]. The two heads without training data start from
# constant outputs instead of random weights. sigmoid(-20) is about 2e-9, which
# rounds to 0.000 on the phone, so the brake stays released. Softmax with logit
# 20 on class 0 keeps the turn signal off.
NEUTRAL_BRAKE_LOGIT = -20.0
TURN_SIGNAL_OFF_LOGIT = 20.0

# Optimizer settings from meta.json. They only matter if the model is trained again.
LEARNING_RATE = 1e-4

MODELS_DIR = Path(__file__).resolve().parent
# Large files that Git ignores (see .gitignore and models/README.md).
DATA_DIR = MODELS_DIR / "data"
EXTENDED_MODEL_PATH = DATA_DIR / "dave2_gru_extended.h5"
PARITY_DIR = DATA_DIR / "parity"
# The converted model is committed here and served by Django as a static file.
WEB_MODEL_DIR = MODELS_DIR.parent / "client" / "static" / "client" / "models" / "dave2"
# Budget from the phase-4 brief, in decimal megabytes.
MAX_WEB_MODEL_BYTES = 10 * 1000 * 1000


def local_checkpoint_path(checkpoint: Checkpoint, data_dir: Path = DATA_DIR) -> Path:
    """Return where a checkpoint is stored locally, mirroring its Hugging Face folder."""
    return data_dir / checkpoint.folder / CHECKPOINT_FILENAME
