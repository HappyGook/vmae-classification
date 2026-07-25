from torch import nn
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
import config

def build_processor() -> VideoMAEImageProcessor:
    """Load the image processor tied to the chosen model."""
    return VideoMAEImageProcessor.from_pretrained(config.MODEL_NAME)

def build_model(model_name_or_path: str | None = None) -> VideoMAEForVideoClassification:
    """Load a VideoMAE classification checkpoint."""
    source = model_name_or_path or config.MODEL_NAME
    model = VideoMAEForVideoClassification.from_pretrained(source)

    n_classes = model.config.num_labels
    n_params = sum(param.numel() for param in model.parameters())

    print(f"[model] loaded {source} ({n_params:,} parameters)")
    print(f"[model] output classes: {n_classes}")

    if n_classes < 2:
        raise ValueError(
            f"Model has only {n_classes} output class(es). "
        )

    return model.to(config.DEVICE)

def build_model_for_finetuning(num_labels: int = 2):
    """
    Load the pretrained model and REPLACE the classification head
    with a fresh linear layer sized for num_labels.
    """
    model = VideoMAEForVideoClassification.from_pretrained(config.MODEL_NAME)

    # Swap the head — keep all encoder weights intact
    hidden_size = model.config.hidden_size
    model.classifier = nn.Linear(hidden_size, num_labels)
    model.config.num_labels = num_labels
    model.config.id2label = {0: "non-violence", 1: "violence"}
    model.config.label2id = {"non-violence": 0, "violence": 1}

    n_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] {n_params:,} total params, {trainable:,} trainable")
    return model.to(config.DEVICE)


def freeze_encoder(model):
    """Optionally freeze the VideoMAE encoder, train only the head."""
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] encoder frozen — {trainable:,} trainable params (head only)")