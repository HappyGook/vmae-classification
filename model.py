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
    model = VideoMAEForVideoClassification.from_pretrained(config.MODEL_NAME)

    n_classes = model.config.num_labels
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] loaded {config.MODEL_NAME}  ({n_params:,} parameters)")
    print(f"[model] output classes: {n_classes}")

    if n_classes < 2:
        raise ValueError(
            f"Model has only {n_classes} output class(es). "
        )

    return model.to(config.DEVICE)