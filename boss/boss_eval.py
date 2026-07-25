import os
from boss_annotations import (
    get_violence_segments,
    label_window,
    parse_video_filename,
    true_timeline,
)
from inference import aggregate_to_timeline, run_inference
from metrics import compute_window_metrics
from serialization import safe_name, save_json

DEFAULT_SAVE_DIR = "boss-evals"


def evaluate_video(
    model,
    video_path,
    processor,
    annotations_df,
    clip_len=None,
    stride=3,
    hop=None,
    batch_size=4,
    device="cuda",
    threshold=0.5,
    save_dir=DEFAULT_SAVE_DIR,
    save=True,
):
    """
    Evaluate one BOSS video.
    Returns None if the video/camera pair has no Fight annotation.
    """
    print(f"======== Evaluating {video_path} ========")

    situation, camera = parse_video_filename(video_path)

    results, fps, total_frames = run_inference(
        model,
        video_path,
        processor,
        clip_len=clip_len,
        stride=stride,
        hop=hop,
        batch_size=batch_size,
        device=device,
    )

    segments = get_violence_segments(
        annotations_df,
        situation,
        camera,
        total_frames=total_frames,
    )

    if segments is None:
        return None

    for result in results:
        result["video"] = situation
        result["camera"] = camera
        result["true_label"] = label_window(
            result["start_frame"],
            result["end_frame"],
            segments,
        )
        result["pred_label"] = int(result["score"] >= threshold)
        result["correct"] = result["pred_label"] == result["true_label"]

    errors = [
        result
        for result in results
        if not result["correct"]
    ]

    output = {
        "video": situation,
        "camera": camera,
        "fps": fps,
        "total_frames": total_frames,
        "segments": segments,
        "results": results,
        "errors": errors,
        "metrics": compute_window_metrics(results),
        "pred_timeline": aggregate_to_timeline(results, total_frames, agg="max"),
        "true_timeline": true_timeline(segments, total_frames),
    }

    if save:
        path = save_json(output, save_dir, safe_name(situation, camera))
        print(f"Saved per-video eval to {path}")

    return output


def iter_video_paths(video_dir, extensions=(".avi",)):
    """Yield video paths below `video_dir` matching the given extensions."""
    for root, _, files in os.walk(video_dir):
        for filename in files:
            if filename.lower().endswith(extensions):
                yield os.path.join(root, filename)


def evaluate_dataset(
    model,
    video_dir,
    processor,
    annotations_df,
    clip_len=None,
    stride=3,
    hop=None,
    batch_size=4,
    device="cuda",
    threshold=0.5,
    extensions=(".avi",),
    cameras=None,
    save_dir=DEFAULT_SAVE_DIR,
    save=True,
    dataset_filename="full_boss-evals",
):
    """
    Evaluate all matching BOSS videos.
    """
    selected_cameras = set(cameras) if cameras is not None else None

    per_video = []
    skipped = []

    for path in iter_video_paths(video_dir, extensions=extensions):
        try:
            _, camera = parse_video_filename(path)
        except ValueError as error:
            skipped.append({"path": path, "reason": str(error)})
            continue

        if selected_cameras is not None and camera not in selected_cameras:
            skipped.append({"path": path, "reason": f"camera {camera} not selected"})
            continue

        try:
            output = evaluate_video(
                model,
                path,
                processor,
                annotations_df,
                clip_len=clip_len,
                stride=stride,
                hop=hop,
                batch_size=batch_size,
                device=device,
                threshold=threshold,
                save_dir=save_dir,
                save=save,
            )
        except ValueError as error:
            skipped.append({"path": path, "reason": str(error)})
            continue

        if output is None:
            skipped.append({"path": path, "reason": "no Fight annotation"})
            continue

        per_video.append(output)

    all_windows = [
        result
        for video_result in per_video
        for result in video_result["results"]
    ]

    all_errors = [
        result
        for video_result in per_video
        for result in video_result["errors"]
    ]

    dataset_output = {
        "per_video": per_video,
        "errors": all_errors,
        "metrics": compute_window_metrics(all_windows),
        "skipped": skipped,
    }

    if save:
        path = save_json(dataset_output, save_dir, dataset_filename)
        print(f"Saved full dataset eval to {path}")

    return dataset_output