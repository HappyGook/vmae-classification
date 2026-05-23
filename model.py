import torch
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
import config

def build_processor() -> VideoMAEImageProcessor:
    """Load the image processor tied to the chosen model."""
    return VideoMAEImageProcessor.from_pretrained(config.MODEL_NAME)

def build_model() -> VideoMAEForVideoClassification:
    """
    Load VideoMAE encoder with a classification head.
    """
    model = VideoMAEForVideoClassification.from_pretrained(
        config.MODEL_NAME,
        num_labels=config.NUM_CLASSES,
        id2label=config.ID2LABEL,
        label2id=config.LABEL2ID,
        ignore_mismatched_sizes=False,
    )

    if config.FREEZE_ENCODER:
        for param in model.videomae.parameters():
            param.requires_grad = False
        print("[model] encoder frozen — only the classification head will be trained")

    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] {config.MODEL_NAME}")
    print(f"[model] parameters: {n_total:,} total  /  {n_trainable:,} trainable")

    return model.to(config.DEVICE)