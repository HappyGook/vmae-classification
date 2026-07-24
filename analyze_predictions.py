
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------------------------
# 1) Per-preset mapping: which top1_class counts as "violence"
# ---------------------------------------------------------------------------
# mode="include" -> classes listed are treated as positive (violence), everything else negative
# mode="exclude" -> classes listed are treated as negative, everything else positive (violence)

PRESET_RULES = {
    "kinetics": {
        "mode": "include",
        "classes": {
            "punching bag", "punching person (boxing)", "slapping", "headbutting",
            "sword fighting", "wrestling", "drop kicking", "high kick",
            "side kick", "capoeira", "tai chi",
        },
    },
    "ucf_crime": {
        "mode": "exclude",
        "classes": {"Normal Video"},  # everything else counts as violence
    },
    "binary": {
        "mode": "include",
        "classes": {"violence", "LABEL_1"},  # handles either label naming
    },
    "binary-new": {
        "mode": "include",
        "classes": {"violence", "LABEL_1"},
    },
}

# assumed file naming: predictions/<preset>.csv -- adjust if yours differ
PREDICTIONS_DIR = Path("predictions")
PRESET_CSV_MAP = {
    "kinetics": "MCG-NJU-videomae-base-finetuned-kineticspredictions.csv",
    "ucf_crime": "OPear-videomae-large-finetuned-UCF-Crime-predictions.csv",
    "binary": "checkpoints-best-predictions.csv",
    "binary-new": "Nikeytas-videomae-crime-detector-production-v1-predictions.csv",
}

OUT_DIR = Path("figures")
OUT_DIR.mkdir(exist_ok=True)

TOP_N_CLASSES = 10  # how many classes to show per bar chart


# ---------------------------------------------------------------------------
# 2) Classification helper
# ---------------------------------------------------------------------------

def classify_as_violent(top1_series: pd.Series, rule: dict) -> pd.Series:
    if rule["mode"] == "include":
        return top1_series.isin(rule["classes"])
    if rule["mode"] == "exclude":
        return ~top1_series.isin(rule["classes"])
    raise ValueError(f"Unknown rule mode: {rule['mode']}")


# ---------------------------------------------------------------------------
# 3) Per-preset analysis
# ---------------------------------------------------------------------------

def analyze_preset(preset_name: str):
    rule = PRESET_RULES[preset_name]
    csv_path = PREDICTIONS_DIR / PRESET_CSV_MAP[preset_name]

    if not csv_path.exists():
        print(f"⚠ {csv_path} not found — skipping '{preset_name}'")
        return None

    df = pd.read_csv(csv_path)
    df["gt_positive"] = df["gt_label"] == "violence"
    df["pred_positive"] = classify_as_violent(df["top1_class"], rule)

    tp = int((df["gt_positive"] & df["pred_positive"]).sum())
    tn = int((~df["gt_positive"] & ~df["pred_positive"]).sum())
    fp = int((~df["gt_positive"] & df["pred_positive"]).sum())
    fn = int((df["gt_positive"] & ~df["pred_positive"]).sum())
    total = len(df)

    accuracy = (tp + tn) / total if total else 0.0
    error_rate = 1 - accuracy
    fpr = fp / (fp + tn) if (fp + tn) else 0.0  # rate among actually non-violent videos
    fnr = fn / (fn + tp) if (fn + tp) else 0.0  # rate among actually violent videos
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    # most common predicted classes, split by ground truth (regardless of correctness)
    violent_classes = df.loc[df["gt_positive"], "top1_class"].value_counts()
    nonviolent_classes = df.loc[~df["gt_positive"], "top1_class"].value_counts()

    # classes responsible for false positives / false negatives
    fp_classes = df.loc[~df["gt_positive"] & df["pred_positive"], "top1_class"].value_counts()
    fn_classes = df.loc[df["gt_positive"] & ~df["pred_positive"], "top1_class"].value_counts()

    print(
        f"{preset_name:12s}  n={total:4d}  acc={accuracy:.2f}  "
        f"error={error_rate:.2f}  FPR={fpr:.2f}  FNR={fnr:.2f}  "
        f"precision={precision:.2f}  recall={recall:.2f} "
        f"(TP={tp} TN={tn} FP={fp} FN={fn})"
    )

    return {
        "preset": preset_name,
        "n": total, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": accuracy, "error_rate": error_rate, "fpr": fpr, "fnr": fnr,
        "precision": precision, "recall": recall,
        "violent_classes": violent_classes,
        "nonviolent_classes": nonviolent_classes,
        "fp_classes": fp_classes,
        "fn_classes": fn_classes,
    }


# ---------------------------------------------------------------------------
# 4) Plots
# ---------------------------------------------------------------------------

def plot_class_distribution(counts: pd.Series, title: str, filename: str, top_n: int = TOP_N_CLASSES):
    counts = counts.head(top_n)
    fig, ax = plt.subplots(figsize=(8, 0.4 * len(counts) + 1.5))
    ax.barh(counts.index[::-1], counts.values[::-1])
    ax.set_xlabel("Count")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=200)
    plt.close(fig)


def plot_metric_comparison(results):
    metrics = ("accuracy", "error_rate", "fpr", "fnr")
    labels = ("Accuracy", "Error rate", "False Positive Rate", "False Negative Rate")

    x = np.arange(len(results))
    width = 0.2

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (metric, label) in enumerate(zip(metrics, labels)):
        values = [r[metric] for r in results]
        ax.bar(x + i * width, values, width, label=label)

    ax.set_xticks(x + width * (len(metrics) - 1) / 2)
    ax.set_xticklabels([r["preset"] for r in results])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Rate")
    ax.set_title("Model comparison on Bus Violence (test set)")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "metric_comparison.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5) Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results = []
    for name in PRESET_RULES:
        r = analyze_preset(name)
        if r is not None:
            results.append(r)

    if not results:
        raise SystemExit("No prediction files found — check PREDICTIONS_DIR / PRESET_CSV_MAP.")

    plot_metric_comparison(results)

    for r in results:
        preset = r["preset"]

        plot_class_distribution(
            r["violent_classes"],
            title=f"{preset}: most common predictions on violent videos",
            filename=f"{preset}_top_classes_violent.png",
        )
        plot_class_distribution(
            r["nonviolent_classes"],
            title=f"{preset}: most common predictions on non-violent videos",
            filename=f"{preset}_top_classes_nonviolent.png",
        )

        # only worth a chart when there's more than one distinct offending class
        # (e.g. binary presets only ever have a single positive label, so a
        # bar chart would just show one bar)
        if r["fp_classes"].nunique() > 1:
            plot_class_distribution(
                r["fp_classes"],
                title=f"{preset}: classes causing false positives",
                filename=f"{preset}_false_positive_classes.png",
            )
        else:
            print(f"  ({preset}) false positives all predicted as a single class — skipping chart")

        if r["fn_classes"].nunique() > 1:
            plot_class_distribution(
                r["fn_classes"],
                title=f"{preset}: classes causing false negatives",
                filename=f"{preset}_false_negative_classes.png",
            )
        else:
            print(f"  ({preset}) false negatives all predicted as a single class — skipping chart")

    print(f"\nFigures saved to {OUT_DIR.resolve()}")