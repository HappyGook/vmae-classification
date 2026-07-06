
def apply_preset(name: str):
    if name not in PRESETS:
        raise ValueError(f"Unknown preset: {name}")
    p = PRESETS[name]
    globals().update(p)
    print(f"Preset '{name}' applied")

PRESETS = {
    "kinetics": {
        "MODEL_NAME": "MCG-NJU/videomae-base-finetuned-kinetics",
        "NUM_CLASSES": 400,
        "ID2LABEL": None,   # None = trust the checkpoint's own config.json
        "LABEL2ID": None,
        "TOP_K": 5,
    },
    "ucf_crime": {
        "MODEL_NAME": "OPear/videomae-large-finetuned-UCF-Crime",
        "NUM_CLASSES": 14,
        "ID2LABEL": None,   # this checkpoint ships its own 14-class mapping
        "LABEL2ID": None,
        "TOP_K": 3,
    },
    "binary": {
        "MODEL_NAME": "checkpoints/best",  # fine-tuned violence/non-violence ckpt
        "NUM_CLASSES": 2,
        "ID2LABEL": {0: "non-violence", 1: "violence"},
        "LABEL2ID": {"non-violence": 0, "violence": 1},
        "TOP_K": 2,
    },
    "binary-new": {
        "MODEL_NAME": "Nikeytas/videomae-crime-detector-production-v1",
        "NUM_CLASSES": 2,
        "ID2LABEL": {0: "non-violence", 1: "violence"},
        "LABEL2ID": {"non-violence": 0, "violence": 1},
        "TOP_K": 2,
    },
}

DEFAULT_PRESET = "binary"
apply_preset(DEFAULT_PRESET)

# video sampling
N_FRAMES = 16 # must match model's pre-training
STRIDE   = 4  # temporal stride between sampled frames

# hardware
DEVICE = "cuda" if __import__("torch").cuda.is_available() else "mps" if __import__("torch").backends.mps.is_available() else "cpu"

# training
BATCH_SIZE     = 8
NUM_WORKERS = 4
NUM_EPOCHS     = 10
LEARNING_RATE  = 3e-5
WEIGHT_DECAY   = 1e-4
EPOCHS         = 10
SAVE_EVERY     = 3           # save a checkpoint every N epochs
CHECKPOINT_DIR = "checkpoints"
FREEZE_ENCODER = False   # True = only train the head

# paths
DATA_DIR = "bus-violence"
PREDICTIONS_CSV = "predictions.csv"