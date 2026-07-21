"""
Evaluation layer: ties parse_annotations.py ground truth into run_inference,
producing per-window predictions with true/predicted labels, isolating
misclassified windows, and aggregating metrics across a whole video set.

Assumes parse_annotations.py (run_inference, get_windows, sample_clip,
preprocess, aggregate_to_timeline, parse_annotation_sheet,
parse_video_filename, get_violence_segments, label_window) is importable.
"""
import json
import os
import numpy as np

from inference import (
    run_inference,
    aggregate_to_timeline,
    parse_video_filename,
    get_violence_segments,
    label_window,
)

DEFAULT_SAVE_DIR = "boss-evals"


def _to_serializable(obj):
    """Recursively convert numpy types / tuples into plain JSON-friendly types."""
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _safe_name(*parts):
    """Build a filesystem-safe filename from arbitrary parts."""
    raw = "_".join(str(p) for p in parts if p is not None)
    return "".join(c if c.isalnum() or c in ("-", "_") else "-" for c in raw)


def save_json(data, save_dir, filename):
    """Serialize `data` to save_dir/filename (adds .json if missing)."""
    os.makedirs(save_dir, exist_ok=True)
    if not filename.endswith(".json"):
        filename += ".json"
    path = os.path.join(save_dir, filename)
    with open(path, "w") as f:
        json.dump(_to_serializable(data), f, indent=2)
    return path


def evaluate_video(model, video_path, processor, df, clip_len=None, stride=None,
                    batch_size=4, device="cuda", threshold=0.5, save_dir=DEFAULT_SAVE_DIR, save=True):
    """
    Runs inference on a single video and attaches ground-truth labels.

    Returns None if the video has no Fight annotation for its
    (situation, camera) -- i.e. it's not violence-relevant and should be
    skipped, matching get_violence_segments' skip behaviour.

    Returns a dict:
      video, camera, fps, total_frames, segments,
      results  -- every window with score/true_label/pred_label/correct
      errors   -- subset of results where pred_label != true_label
    """
    print(f"========Evaluating {video_path}...=======\n")
    situation, camera = parse_video_filename(video_path)

    results, fps, total_frames = run_inference(
        model, video_path, processor, clip_len=clip_len, stride=stride,
        batch_size=batch_size, device=device
    )

    segments = get_violence_segments(df, situation, camera, total_frames=total_frames)
    if segments is None:
        return None  # not violence-relevant, skip

    for r in results:
        r["video"] = situation
        r["camera"] = camera
        r["true_label"] = label_window(r["start_frame"], r["end_frame"], segments)
        r["pred_label"] = int(r["score"] >= threshold)
        r["correct"] = r["pred_label"] == r["true_label"]

    errors = [r for r in results if not r["correct"]]

    out = {
        "video": situation,
        "camera": camera,
        "fps": fps,
        "total_frames": total_frames,
        "segments": segments,
        "results": results,
        "errors": errors,
        "pred_timeline": aggregate_to_timeline(results, total_frames, agg="max"),
        "true_timeline": true_timeline(segments, total_frames),
    }

    if save:
        filename = _safe_name(situation, camera)
        path = save_json(out, save_dir, filename)
        print(f"Saved per-video eval to {path}")

    return out


def true_timeline(segments, total_frames):
    """Per-frame ground-truth curve (1.0 inside fight segments, else 0.0),
    directly comparable to aggregate_to_timeline's output."""
    timeline = np.zeros(total_frames, dtype=np.float32)
    for label, s, e in segments:
        if label == 1:
            timeline[s:e + 1] = 1.0
    return timeline


def compute_metrics(windows):
    """Window-level confusion matrix + precision/recall/F1/accuracy."""
    if not windows:
        return {}
    y_true = np.array([w["true_label"] for w in windows])
    y_pred = np.array([w["pred_label"] for w in windows])

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / len(windows)

    return {
        "n_windows": len(windows),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": accuracy, "precision": precision,
        "recall": recall, "f1": f1,
    }


def evaluate_dataset(model, video_dir, processor, df, clip_len=None, stride=None,
                      batch_size=4, device="cuda", threshold=0.5,
                      extensions=(".avi",), save_dir=DEFAULT_SAVE_DIR, save=True,
                     dataset_filename="full_boss-evals"):
    """
    Walks video_dir, evaluates every matching video, and aggregates
    per-window results + errors + metrics across the whole set.

    Returns:
      per_video -- list of evaluate_video() dicts (one per used video)
      errors    -- all misclassified windows, pooled across videos
      metrics   -- aggregate confusion matrix / precision / recall / F1
      skipped   -- videos skipped (no Fight annotation, or filename didn't
                   match the expected pattern), with the reason why
    """
    video_paths = [
        os.path.join(root, f)
        for root, _, files in os.walk(video_dir)
        for f in files if f.lower().endswith(extensions)
    ]

    per_video, skipped = [], []
    for path in video_paths:
        if ('7' or '6') in path: # <- hardcoded for now, since cameras 6 and 7 show best results
            try:
                out = evaluate_video(model, path, processor, df, clip_len=clip_len,
                                      stride=stride, batch_size=batch_size,
                                      device=device, threshold=threshold)
            except ValueError as e:
                skipped.append({"path": path, "reason": str(e)})
                continue
            if out is None:
                skipped.append({"path": path, "reason": "no Fight annotation"})
                continue
            per_video.append(out)

    all_windows = [r for v in per_video for r in v["results"]]
    all_errors = [r for v in per_video for r in v["errors"]]

    dataset_out = {
        "per_video": per_video,
        "errors": all_errors,
        "metrics": compute_metrics(all_windows),
        "skipped": skipped,
    }

    if save:
        path = save_json(dataset_out, save_dir, dataset_filename)
        print(f"Saved full dataset eval to {path}")

    return dataset_out
