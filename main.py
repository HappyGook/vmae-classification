import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import config
from dataset import BusViolenceDataset
from inference import run_inference, aggregate_to_timeline
from model import build_processor, build_model, build_model_for_finetuning


def train():
    processor = build_processor()
    model = build_model()
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


    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE,
                              shuffle=True, num_workers=config.NUM_WORKERS,
                              pin_memory=config.DEVICE == "cuda")
    val_loader   = DataLoader(val_dataset,   batch_size=config.BATCH_SIZE,
                              shuffle=False, num_workers=config.NUM_WORKERS,
                              pin_memory=config.DEVICE == "cuda")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.EPOCHS
    )

    save_dir = Path(config.CHECKPOINT_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0

    for epoch in range(1, config.EPOCHS + 1):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\n--- Starting epoch {epoch}/{config.EPOCHS} | lr={current_lr:.8f} ---")
        # ── Train ──────────────────────────────────────────────
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for pixel_values, labels, _ in train_loader:
            batch_start = time.time()
            print(f"Batch {train_total // config.BATCH_SIZE + 1}: input shape={tuple(pixel_values.shape)}, labels={labels.tolist()}")
            pixel_values = pixel_values.to(config.DEVICE)
            labels       = labels.to(config.DEVICE)

            optimizer.zero_grad()
            logits = model(pixel_values=pixel_values).logits   # (B, 2)
            loss   = criterion(logits, labels)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            batch_time = time.time() - batch_start
            print(f"    loss={loss.item():.5f} | batch_time={batch_time:.3f}s | grad_norm={grad_norm:.5f}")

            train_loss    += loss.item() * labels.size(0)
            train_correct += (logits.argmax(1) == labels).sum().item()
            train_total   += labels.size(0)

        scheduler.step()

        # ── Validate ───────────────────────────────────────────
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for pixel_values, labels, _ in val_loader:
                pixel_values = pixel_values.to(config.DEVICE)
                labels       = labels.to(config.DEVICE)
                logits = model(pixel_values=pixel_values).logits
                loss   = criterion(logits, labels)
                val_loss    += loss.item() * labels.size(0)
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_total   += labels.size(0)

        t_acc = train_correct / train_total
        v_acc = val_correct   / val_total
        t_l   = train_loss    / train_total
        v_l   = val_loss      / val_total

        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch:>3}/{config.EPOCHS} finished in {epoch_time:.2f}s")
        print(f"  train loss={t_l:.4f} acc={t_acc:.3f}")
        print(f"  val   loss={v_l:.4f} acc={v_acc:.3f}")
        print(f"  lr={optimizer.param_groups[0]['lr']:.8f}")
        print(f"  best_val_acc={best_val_acc:.3f}")

        # ── Save best checkpoint ───────────────────────────────
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            best_path = save_dir / "best"
            print("Checkpoint debug:")
            print(f"  Saving epoch: {epoch}")
            print(f"  Validation accuracy: {v_acc:.5f}")
            print(f"  Directory: {best_path.resolve()}")
            print(f"  Files before save: {list(best_path.glob('*')) if best_path.exists() else 'directory does not exist'}")
            model.save_pretrained(best_path)       # saves config.json + model weights
            processor.save_pretrained(best_path)   # saves preprocessor_config.json
            print(f"  Files after save: {[p.name for p in best_path.glob('*')]}")
            print(f"  ✓ saved best model (val_acc={v_acc:.3f}) → {best_path}")

        # ── Save periodic checkpoint every N epochs ────────────
        if epoch % config.SAVE_EVERY == 0:
            ckpt_path = save_dir / f"epoch_{epoch:03d}"
            print(f"Checkpoint debug: periodic save at epoch {epoch}")
            print(f"  Path: {ckpt_path.resolve()}")
            model.save_pretrained(ckpt_path)
            processor.save_pretrained(ckpt_path)
            print(f"  Saved files: {[p.name for p in ckpt_path.glob('*')]}")
            print(f"  ✓ checkpoint saved → {ckpt_path}")

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.3f}")
    print(f"Best model saved to: {(save_dir / 'best').resolve()}")


def run():
    # setup
    processor = build_processor()
    model = build_model("checkpoints/best")
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
    out = Path(config.PREDICTIONS_CSV)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows → {out.resolve()}")

def boss_inference():
    processor = build_processor()
    model = build_model("checkpoints/best")

    results, fps, total_frames = run_inference(
        args.video_path, model, processor,
        clip_len=args.clip_len, stride=args.stride,
        batch_size=args.batch_size, device=config.DEVICE
    )

    timeline = aggregate_to_timeline(results, total_frames, agg="max")

    out = {
        "video": args.video,
        "fps": fps,
        "total_frames": total_frames,
        "num_windows": len(results),
        "windows": results,
        "video_level_score": float(max(r["score"] for r in results)) if results else None,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    np.save(args.out.replace(".json", "_timeline.npy"),timeline)
    print(f"Saved {len(results)} window predictions to {args.out}")
    print(f"Saved per-frame timeline ({total_frames} frames) to "
          f"{args.out.replace('.json', '_timeline.npy')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "run", "boss"], default="run")
    parser.add_argument("--clip_len", type=int, default=48)
    parser.add_argument("--stride", type=int, default=1,
                        help="sliding step between window starts, in sampled-frame units; "
                             "smaller = more overlap = better localization, slower")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--out", default="inference.json")
    args = parser.parse_args()

    if args.mode == "train":
        train()
    elif args.mode == "run":
        run()
    elif args.mode == "boss":
        boss_inference()