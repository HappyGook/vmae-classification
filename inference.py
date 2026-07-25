import argparse
import json
import os
import re
import numpy as np
import pandas as pd
import torch
import decord
from decord import VideoReader, cpu

decord.bridge.set_bridge("torch")

def get_windows(total_frames, clip_len=16, stride=4, hop=None):
    """
    clip_len: number of frames sampled per clip (fed to the model)
    stride:   spacing between sampled frames within a clip
    hop:      how far the window start advances between clips
              (defaults to stride, but is logically independent)
    """
    span = (clip_len - 1) * stride + 1  # raw frames covered by one clip
    if hop is None:
        hop = stride

    if total_frames <= span:
        yield 0
        return


    start = 0
    last_start = 0
    while start + span <= total_frames:
        yield start
        start += hop
        last_start = start

    tail_start = total_frames - span
    if tail_start > last_start:
        yield tail_start


def sample_clip(video_reader: VideoReader, start_index: int, clip_len: int, stride: int):
    """Return sampled frames and their raw frame indices."""
    total_frames = len(video_reader)
    indices = [
        min(start_index + stride * i, total_frames - 1)
        for i in range(clip_len)
    ]
    frames = video_reader.get_batch(indices)
    return frames, indices

def preprocess(frames, processor):
    """Convert Decord/Torch frames to the tensor format expected by VideoMAE."""
    frames_np = [frame.numpy() for frame in frames]
    return processor(images=frames_np, return_tensors="pt").pixel_values

def logits_to_positive_scores(logits: torch.Tensor, positive_class_index: int = 1) -> torch.Tensor:
    """
    Convert model logits to a binary positive-class score.
    """
    return torch.softmax(logits, dim=-1)[:, positive_class_index]

@torch.no_grad()
def run_inference(
    model,
    video_path,
    processor,
    clip_len: int | None = None,
    stride: int = 3,
    hop: int | None = None,
    batch_size: int = 4,
    device: str = "cuda",
    positive_class_index: int = 1,
    progress_seconds: float | None = 10.0,
):
    """
    Run sliding-window inference over one video.

    Returns:
        results:
            List of window dicts containing start/end frame, start/end second,
            and model score.
        fps:
            Average video FPS.
        total_frames:
            Number of frames in the video.
    """
    if clip_len is None:
        clip_len = model.config.num_frames

    video_reader = VideoReader(video_path, ctx=cpu(0))
    fps = video_reader.get_avg_fps()
    total_frames = len(video_reader)

    results = []
    batch_clips = []
    batch_meta = []

    def flush_batch():
        nonlocal batch_clips, batch_meta

        if not batch_clips:
            return

        pixel_values = torch.cat(batch_clips, dim=0).to(device)
        logits = model(pixel_values=pixel_values).logits
        scores = logits_to_positive_scores(logits, positive_class_index)

        for meta, score in zip(batch_meta, scores.cpu().numpy()):
            meta["score"] = float(score)
            results.append(meta)

        batch_clips = []
        batch_meta = []

    previous_progress_sec = 0.0

    for start_index in get_windows(total_frames, clip_len=clip_len, stride=stride, hop=hop):
        frames, indices = sample_clip(video_reader, start_index, clip_len, stride)
        batch_clips.append(preprocess(frames, processor))

        end_sec = round(indices[-1] / fps, 2)
        batch_meta.append(
            {
                "start_frame": indices[0],
                "end_frame": indices[-1],
                "start_sec": round(indices[0] / fps, 2),
                "end_sec": end_sec,
            }
        )

        if len(batch_clips) == batch_size:
            flush_batch()

        if progress_seconds is not None and end_sec - previous_progress_sec >= progress_seconds:
            print(f"Processed {end_sec:.2f} seconds")
            previous_progress_sec = end_sec

    flush_batch()

    return results, fps, total_frames

def aggregate_to_timeline(results, total_frames, agg="max"):
    """window scores -> per-frame violence score curve"""
    timeline = np.zeros(total_frames, dtype=np.float32)
    counts = np.zeros(total_frames, dtype=np.int32)
    for r in results:
        s, e = r["start_frame"], r["end_frame"]
        if agg == "max":
            timeline[s:e + 1] = np.maximum(timeline[s:e + 1], r["score"])
        else:
            timeline[s:e + 1] += r["score"]
            counts[s:e + 1] += 1
    if agg == "mean":
        counts[counts == 0] = 1
        timeline = timeline / counts
    return timeline