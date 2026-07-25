import argparse
import csv
from pathlib import Path
import time
import torch
from torch import nn
from torch.utils.data import DataLoader
import config
from boss.boss_annotations import parse_annotation_sheet
from boss.boss_eval import evaluate_dataset
from bus_dataset import BusViolenceDataset
from model import build_model, build_model_for_finetuning, build_processor, freeze_encoder


def train():
    processor = build_processor()
    model = build_model_for_finetuning(num_labels=2)
    if config.FREEZE_ENCODER:
        freeze_encoder(model)

    print("\n=== MODEL DEBUG INFO ===")
    print(f"Model class: {model.__class__.__name__}")
    print(f"Device target: {config.DEVICE}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Frozen parameters: {sum(p.numel() for p in model.parameters() if not p.requires_grad):,}")
    print(f"Model config: {model.config}")
    print("========================\n")

    train_dataset = BusViolenceDataset(
        root_dir=config.DATA_DIR,
        processor=processor,
        n_frames=config.N_FRAMES,
        stride=config.STRIDE,
        split="train",
    )
    val_dataset = BusViolenceDataset(
        root_dir=config.DATA_DIR,
        processor=processor,
        n_frames=config.N_FRAMES,
        stride=config.STRIDE,
        split="test",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.DEVICE == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.DEVICE == "cuda",
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda param: param.requires_grad, model.parameters()),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.EPOCHS,
    )

    save_dir = Path(config.CHECKPOINT_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0

    for epoch in range(1, config.EPOCHS + 1):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\n--- Starting epoch {epoch}/{config.EPOCHS} | lr={current_lr:.8f} ---")

        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for pixel_values, labels, _ in train_loader:
            pixel_values = pixel_values.to(config.DEVICE)
            labels = labels.to(config.DEVICE)

            optimizer.zero_grad()

            logits = model(pixel_values=pixel_values).logits
            loss = criterion(logits, labels)

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * labels.size(0)
            train_correct += (logits.argmax(1) == labels).sum().item()
            train_total += labels.size(0)

            print(
                f"batch={train_total // config.BATCH_SIZE} "
                f"loss={loss.item():.5f} grad_norm={grad_norm:.5f}"
            )

        scheduler.step()

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for pixel_values, labels, _ in val_loader:
                pixel_values = pixel_values.to(config.DEVICE)
                labels = labels.to(config.DEVICE)

                logits = model(pixel_values=pixel_values).logits
                loss = criterion(logits, labels)

                val_loss += loss.item() * labels.size(0)
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_total += labels.size(0)

        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        train_avg_loss = train_loss / train_total
        val_avg_loss = val_loss / val_total

        epoch_time = time.time() - epoch_start

        print(f"Epoch {epoch}/{config.EPOCHS} finished in {epoch_time:.2f}s")
        print(f"  train loss={train_avg_loss:.4f} acc={train_acc:.3f}")
        print(f"  val   loss={val_avg_loss:.4f} acc={val_acc:.3f}")
        print(f"  lr={optimizer.param_groups[0]['lr']:.8f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = save_dir / "best"

            model.save_pretrained(best_path)
            processor.save_pretrained(best_path)

            print(f"  saved best model val_acc={val_acc:.3f} -> {best_path}")

        if epoch % config.SAVE_EVERY == 0:
            checkpoint_path = save_dir / f"epoch_{epoch:03d}"

            model.save_pretrained(checkpoint_path)
            processor.save_pretrained(checkpoint_path)

            print(f"  saved checkpoint -> {checkpoint_path}")

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.3f}")


def run():
    # setup
    processor = build_processor()
    model = build_model(config.MODEL_NAME)
    model.eval()

    id2label = model.config.id2label  # {int: "class name"} for all classes

    dataset = BusViolenceDataset(
        root_dir=config.DATA_DIR,
        processor=processor,
        n_frames=config.N_FRAMES,
        stride=config.STRIDE,
        split="test"
    )
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory= config.DEVICE == "cuda",
    )

    rows = []  # accumulated for CSV

    print(f"\n{'─' * 64}")
    print(f"  {'filename':<30}  gt label top-1 prediction")
    print(f"{'─' * 64}")

    with torch.no_grad():
        for pixel_values, labels, names in loader:
            pixel_values = pixel_values.to(config.DEVICE)

            logits = model(pixel_values=pixel_values).logits  # (B, 400)
            probs = torch.softmax(logits, dim=-1)
            topk = torch.topk(probs, k=config.TOP_K, dim=-1)

            for i in range(len(names)):
                gt = "violence" if labels[i].item() == 1 else "non-violence"
                top1_name = id2label[topk.indices[i, 0].item()]
                top1_prob = topk.values[i, 0].item()

                print(f"  {names[i]:<30}  {gt:<13}  {top1_name}  ({top1_prob:.2f})")

                # full top-K block for closer inspection
                for rank in range(config.TOP_K):
                    class_id = topk.indices[i, rank].item()
                    class_name = id2label[class_id]
                    prob = topk.values[i, rank].item()
                    if rank > 0:
                        print(f"  {'':30}  {'':13}  {rank + 1}. {class_name:<38} {prob:.3f}")
                    else:
                        print(f"  {'':30}  {'':13}  1. {class_name:<38} {prob:.3f}")

                # row for CSV
                row = {"filename": names[i], "gt_label": gt}
                for rank in range(config.TOP_K):
                    class_id = topk.indices[i, rank].item()
                    row[f"top{rank + 1}_class"] = id2label[class_id]
                    row[f"top{rank + 1}_prob"] = round(topk.values[i, rank].item(), 4)
                rows.append(row)

    print(f"{'─' * 64}\n")

    # save CSV
    out = Path("predictions/"+str.replace((config.MODEL_NAME + '-'+ config.PREDICTIONS_CSV),'/','-'))
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows → {out.resolve()}")

def boss_inference(args):
    processor = build_processor(args.model)
    model = build_model(args.model)
    model.eval()

    annotations_df = parse_annotation_sheet(args.annotations, sheet_name=args.sheet)

    report = evaluate_dataset(
        model,
        args.video_dir,
        processor,
        annotations_df,
        clip_len=args.clip_len,
        stride=args.stride,
        hop=args.hop,
        batch_size=args.batch_size,
        device=config.DEVICE,
        threshold=args.threshold,
        cameras=args.cameras,
        save_dir=args.save_dir,
        dataset_filename=args.out,
    )

    print("Metrics:", report["metrics"])
    print(
        f"{len(report['errors'])} misclassified windows across "
        f"{len(report['per_video'])} evaluated videos "
        f"({len(report['skipped'])} skipped)"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "run", "boss"], default="run")
    parser.add_argument("--preset", choices=list(config.PRESETS), default=config.DEFAULT_PRESET)

    parser.add_argument("--model", default=None)
    parser.add_argument("--video-dir", default="boss")
    parser.add_argument("--annotations", default="AnnotationsBOSS_v1.xlsx")
    parser.add_argument("--sheet", type=int, default=1)

    parser.add_argument("--clip-len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=config.STRIDE)
    parser.add_argument("--hop", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--cameras", type=int, nargs="*", default=None)

    parser.add_argument("--save-dir", default="boss-evals")
    parser.add_argument("--out", default="full_boss-evals")

    args = parser.parse_args()

    config.apply_preset(args.preset)

    if args.model is None:
        args.model = config.MODEL_NAME

    if args.mode == "train":
        train()
    elif args.mode == "run":
        run()
    elif args.mode == "boss":
        boss_inference(args)