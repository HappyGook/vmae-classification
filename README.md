# VideoMAE Violence Detection Experiments

This project contains experiments with several VideoMAE-based video classification models for binary violence / non-violence detection.

The main experimental focus is:

1. Evaluate available VideoMAE checkpoints on the **Bus Violence** dataset.
2. Fine-tune a binary VideoMAE classifier on Bus Violence.
3. Test whether the fine-tuned Bus Violence model transfers to the structurally different **BOSS** dataset.
4. Analyze predictions, error patterns, and window-level performance.

---

## Project Goal

The goal is to determine whether publicly available VideoMAE models are suitable for violence detection in the ISA project context.

The experiments treat violence detection as a **binary classification task**:
- 1 for violence
- 0 for non-violence

The experiments compare:

| Preset       | Model                                                                |
|--------------|----------------------------------------------------------------------|
| `kinetics`   | `MCG-NJU/videomae-base-finetuned-kinetics`                           |
| `ucf_crime`  | `OPear/videomae-large-finetuned-UCF-Crime`                           |
| `binary-new` | `Nikeytas/videomae-crime-detector-production-v1`                     |
| `binary`     | local fine-tuned Bus Violence checkpoint, usually `checkpoints/best` |

> the local checkpoint has been uploaded to HuggingFace (HappyGook/videomae-violence-detector)

The most important model for later experiments is the fine-tuned binary Violence Detector trained on the Bus Violence dataset.

---

## Experiment Summary

### Bus Violence

The Bus Violence dataset is used for both fine-tuning and model comparison.

The expected dataset split files are:
```
text
bus-violence/
├── train.txt
├── test.txt
├── Violence/
│   └── ...
└── NoViolence/
    └── ...
```
Filenames are expected to start with either:
```
text
VIOLENCE_...
NONVIOLENCE_...
```
These prefixes are mapped to binary labels by `BusViolenceDataset`.

### BOSS

The BOSS dataset is used as a transfer/domain-shift evaluation target. The fine-tuned Bus Violence model is evaluated on BOSS videos using the official annotation sheet.

Expected annotation file:
```
text
AnnotationsBOSS_v1.xlsx
```
Default BOSS video directory:
```
text
boss/
```
BOSS evaluation is done at sliding-window level. For each window, the model produces a violence score, which is compared against the ground-truth fight intervals from the annotation sheet.

---

## Installation

Create and activate a virtual environment:
```
bash
python -m venv .venv
source .venv/bin/activate
```
On Windows:
```
bash
python -m venv .venv
.venv\Scripts\activate
```
Install dependencies:
```
bash
pip install -r requirements.txt
```
The project uses PyTorch, HuggingFace Transformers, Decord, TorchCodec, Pandas, NumPy, Matplotlib, and Pillow.

> Depending on your system and GPU setup, you may need to install a PyTorch build matching your CUDA version manually.

---

## Configuration

Most project-wide settings are defined in `config.py`.

Important values include:
```
python
DEFAULT_PRESET = "binary"

N_FRAMES = 16
STRIDE = 3

BATCH_SIZE = 8
NUM_WORKERS = 4
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 1e-4
EPOCHS = 10
SAVE_EVERY = 3
CHECKPOINT_DIR = "checkpoints"
FREEZE_ENCODER = False

DATA_DIR = "bus-violence"
PREDICTIONS_CSV = "predictions.csv"
```
### Presets

`config.py` defines the available model presets:
```
text
kinetics
ucf_crime
binary
binary-new
```
Use a preset with:
```
bash
python main.py --preset binary
```
or together with a mode:
```
bash
python main.py --mode run --preset kinetics
```
---

## Main Entry Point

The main entry point is:
```
bash
python main.py
```
It supports three modes:
```
text
train   fine-tune a binary VideoMAE model on Bus Violence
run     run a selected model preset on the Bus Violence test set
boss    evaluate a binary model on the BOSS dataset
```
General syntax:
```
bash
python main.py --mode <train|run|boss> [options]
```
---

## CLI Parameters

### Common parameters

| Parameter  |  Default | Description                                                                |
|------------|---------:|----------------------------------------------------------------------------|
| `--mode`   |    `run` | Execution mode: `train`, `run`, or `boss`                                  |
| `--preset` | `binary` | Model preset from `config.PRESETS`                                         |
| `--model`  |   `None` | Explicit model/checkpoint path. If omitted, uses the selected preset model |

### BOSS-specific parameters

| Parameter       |                   Default | Description                                                       |
|-----------------|--------------------------:|-------------------------------------------------------------------|
| `--video-dir`   |                    `boss` | Directory containing BOSS videos                                  |
| `--annotations` | `AnnotationsBOSS_v1.xlsx` | BOSS annotation Excel file                                        |
| `--sheet`       |                       `1` | Excel sheet index/name passed to Pandas                           |
| `--clip-len`    |                      `16` | Number of sampled frames fed to VideoMAE                          |
| `--stride`      |           `config.STRIDE` | Frame distance between sampled frames                             |
| `--hop`         |                    `None` | Step between sliding-window starts. If `None`, defaults to stride |
| `--batch-size`  |                       `4` | Batch size for sliding-window inference                           |
| `--threshold`   |                     `0.5` | Score threshold for binary prediction                             |
| `--cameras`     |                    `None` | Optional camera filter, e.g. `--cameras 6 7`                      |
| `--save-dir`    |              `boss-evals` | Output directory for BOSS JSON reports                            |
| `--out`         |         `full_boss-evals` | Dataset-level output JSON filename                                |

---

## Usage

### 1. Fine-tune on Bus Violence

Run:
```
bash
python main.py --mode train --preset binary
```
This will:

1. Load the configured base model.
2. Replace the classification head with a binary head.
3. Train on `bus-violence/train.txt`.
4. Validate on `bus-violence/test.txt`.
5. Save the best checkpoint to:
```
text
checkpoints/best/
```
Periodic checkpoints are saved to:
```
text
checkpoints/epoch_003/
checkpoints/epoch_006/
...
```
depending on `SAVE_EVERY`.

---

### 2. Run a model on the Bus Violence test set

Run the default binary model:
```
bash
python main.py --mode run --preset binary
```
Run the Kinetics model:
```
bash
python main.py --mode run --preset kinetics
```
Run the UCF Crime model:
```
bash
python main.py --mode run --preset ucf_crime
```
Run the external binary crime detector:
```
bash
python main.py --mode run --preset binary-new
```
This mode:

1. Loads the selected model.
2. Runs it on the Bus Violence test set.
3. Prints top-k predictions.
4. Writes a CSV into:
```
text
predictions/
```
The output filename is derived from the model name and `PREDICTIONS_CSV`.

Example:
```
text
predictions/checkpoints-best-predictions.csv
```
---

### 3. Analyze Bus Violence prediction CSVs

After running one or more presets, generate metrics and figures with:
```
bash
python analyze_predictions.py
```
This script reads the configured CSV files from:
```
text
predictions/
```
and writes plots to:
```
text
figures/
```
It computes:

- accuracy
- error rate
- false positive rate
- false negative rate
- precision
- recall
- F1
- most common predicted classes
- false-positive class distributions
- false-negative class distributions

The class-to-violence mapping is defined in `PRESET_RULES` inside `analyze_predictions.py`.

For example:

- Kinetics uses a manually selected set of violence-related class names.
- UCF Crime treats everything except `Normal Video` as violence.
- Binary models treat `violence` / `LABEL_1` as violence.

---

### 4. Evaluate on the full BOSS dataset

Run:
```
bash
python main.py \
  --mode boss \
  --model checkpoints/best \
  --video-dir boss \
  --annotations AnnotationsBOSS_v1.xlsx \
  --out full_boss-evals
```
This will:

1. Load the fine-tuned Bus Violence model.
2. Parse BOSS annotations.
3. Run sliding-window inference over all matching `.avi` videos.
4. Match predictions with fight annotations.
5. Save per-video and dataset-level JSON reports.

Default output directory:
```
text
boss-evals/
```
Dataset-level output:
```
text
boss-evals/full_boss-evals.json
```
---

### 5. Evaluate only BOSS cameras 6 and 7

Run:
```
bash
python main.py \
  --mode boss \
  --model checkpoints/best \
  --video-dir boss \
  --annotations AnnotationsBOSS_v1.xlsx \
  --cameras 6 7 \
  --out boss-cam6-cam7-evals
```
This corresponds to the second BOSS experiment, where only cameras 6 and 7 are evaluated.

Output:
```
text
boss-evals/boss-cam6-cam7-evals.json
```
---

## Sampling Logic

The VideoMAE models receive `N_FRAMES` sampled frames per clip.

The intended experiment setup is:
```
text
N_FRAMES = 16
STRIDE = 3
```
This samples frames approximately like:
```
text
0, 3, 6, 9, ..., 45
```
So each model input covers roughly a 48-frame temporal window.

For BOSS sliding-window evaluation:

- `clip_len` controls the number of sampled frames.
- `stride` controls the spacing between sampled frames.
- `hop` controls how far the next window start moves.
- If `hop` is omitted, it defaults to the same value as `stride`.

Example with more overlap:
```
bash
python main.py \
  --mode boss \
  --model checkpoints/best \
  --stride 3 \
  --hop 1
```
This is slower but can improve temporal localization.

---

## Notes and Caveats

- The non-domain-specific models are not expected to perform well on binary Bus Violence classification without careful label mapping.
- The BOSS evaluation is a domain-shift experiment. Lower performance is expected because camera perspective, scene structure, and interaction types differ from Bus Violence.
- The binary threshold defaults to `0.5`.
- The BOSS evaluation is window-level, not video-level.
- BOSS videos without Fight annotations are skipped.
- The model score is interpreted as the probability of the positive class, usually class index `1`.
- The quality of `analyze_predictions.py` depends on the manually defined mapping from model class names to binary violence/non-violence labels.

---

## References

- VideoMAE Base, HuggingFace: <https://huggingface.co/MCG-NJU/videomae-base>
- VideoMAE UCF Crime, HuggingFace: <https://huggingface.co/OPear/videomae-large-finetuned-UCF-Crime>
- VideoMAE Crime Detector, HuggingFace: <https://huggingface.co/Nikeytas/videomae-crime-detector-production-v1>
- Fine-tuned Violence Detector: <https://huggingface.co/HappyGook/videomae-violence-detector>
- Bus Violence Dataset: <https://zenodo.org/records/7044203>
- Bus Violence Paper: <https://www.mdpi.com/1424-8220/22/21/8345>
- BOSS Dataset: <https://videodatasets.org/BOSSdata/index.html>
```

