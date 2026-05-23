

# model
MODEL_NAME = "MCG-NJU/videomae-base"

# drop-in replacements
# MODEL_NAME = "MCG-NJU/videomae-large"
# MODEL_NAME = "MCG-NJU/videomae-base-ssv2"
# MODEL_NAME = "MCG-NJU/videomae-large-finetuned-kinetics"

NUM_CLASSES = 2  # violence=1, non-violence=0
ID2LABEL = {0: "non-violence", 1: "violence"}
LABEL2ID = {"non-violence": 0, "violence": 1}

# video sampling
N_FRAMES = 16 # must match model's  pre-training
STRIDE   = 4  # temporal stride between sampled frames

# hardware
DEVICE      = "cuda"

# training
BATCH_SIZE     = 8
NUM_EPOCHS     = 10
LEARNING_RATE  = 1e-4
WEIGHT_DECAY   = 0.05
FREEZE_ENCODER = False   # True = only train the head

# paths
DATA_DIR = "/bus-violence"
CHECKPOINT_DIR = "./checkpoints"
