import csv
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import config
from dataset import BusViolenceDataset
from model import build_processor, build_model


def run():
    # setup
    processor = build_processor()
    model = build_model()
    model.eval()

    id2label = model.config.id2label  # {int: "class name"} for all classes

    dataset = BusViolenceDataset(
        root_dir=config.DATA_DIR,
        processor=processor,
        n_frames=config.N_FRAMES,
        stride=config.STRIDE,
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


if __name__ == "__main__":
    run()