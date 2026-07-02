import argparse
import json

import numpy as np
import torch
import decord
from decord import VideoReader, cpu

decord.bridge.set_bridge("torch")

def get_windows(total_frames, clip_len=48, stride=1):
    if total_frames <= clip_len:
        yield 0
        return


    start = 0
    last_start = 0

    while start + clip_len <= total_frames:
        yield start
        start += stride
        last_start = start

    tail_start = total_frames - clip_len
    if tail_start > last_start:
        yield tail_start


def sample_clip(videoReader, start_index, clip_len, stride):
    n = len(videoReader) # what is the length of the video reader ??
    indexes = [min(start_index + stride * i, n - 1) for i in range(clip_len)]
    frames = videoReader.get_batch(indexes)
    return frames, indexes

def preprocess(frames, processor):
    frames_np = [f.numpy() for f in frames]
    inputs = processor(images=frames_np, return_tensors="pt").pixel_values
    return inputs

@torch.no_grad()
def run_inference(model, video_path, processor, clip_len=48, stride=1, batch_size=4, device="cuda"):
    vr = VideoReader(video_path, ctx=cpu(0))
    fps = vr.get_avg_fps()
    total_frames = len(vr)
    windows = list(get_windows(total_frames, clip_len, stride))
    results, batch_clips, batch_meta = [], [], []

    def flush():
        nonlocal batch_clips, batch_meta
        if not batch_clips:
            return
        pixel_values = torch.cat(batch_clips, dim=0).to(device)
        logits = model(pixel_values=pixel_values).logits
        probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
        probs = np.atleast_1d(probs)
        for meta, prob in zip(batch_meta, probs):
            meta["score"] = float(prob)
            results.append(meta)
        batch_clips, batch_meta = [], []

    for start_index in windows:
        frames, indexes = sample_clip(vr, start_index, clip_len, stride)
        inputs = preprocess(frames, processor)
        batch_clips.append(inputs)
        batch_meta.append({"start_frame": indexes[0], "end_frame": indexes[-1],
                           "start_sec": round(indexes[0] / fps, 2),
                           "end_sec": round(indexes[-1] / fps, 2)})
        if len(batch_clips) == batch_size:
            flush()

        flush()

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