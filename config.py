

# model
MODEL_NAME = "MCG-NJU/videomae-base"

# models used in experiments
# MODEL_NAME = "MCG-NJU/videomae-base"
# MODEL_NAME = "OPear/videomae-large-finetuned-UCF-Crime"
# MODEL_NAME = "Nikeytas/videomae-crime-detector-production-v1"
# MODEL_NAME = "checkpoints/best"

NUM_CLASSES = 2  # violence=1, non-violence=0
ID2LABEL = {0: "non-violence", 1: "violence"}
LABEL2ID = {"non-violence": 0, "violence": 1}

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

# inference output
TOP_K = 2          # how many predictions print per clip

# paths
DATA_DIR = "bus-violence"
PREDICTIONS_CSV = "predictions.csv"