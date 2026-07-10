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

CAMERA_HEADER_RE = re.compile(r'^Camera\s*(\d+)$', re.IGNORECASE)
VIDEO_FILENAME_RE = re.compile(r'^(?P<situation>.+)\.Cam(?P<cam>\d+)\.avi$', re.IGNORECASE)


def parse_annotation_sheet(path, sheet_name=1):
    raw = pd.read_excel(path, sheet_name=sheet_name, header=None)
    n_rows = raw.shape[0]

    records = []
    state = "WAITING_FOR_VIDEO"  # WAITING_FOR_VIDEO -> WAITING_FOR_HEADER -> IN_BLOCK
    current_camera = None
    current_video = None
    col_map = None
    event_id = 1

    for i in range(n_rows):
        row = raw.iloc[i]
        non_null = [(c, v) for c, v in row.items()
                    if pd.notna(v) and str(v).strip() != ""]

        if not non_null:
            if state == "IN_BLOCK":
                state = "WAITING_FOR_VIDEO"
                col_map = None
            continue

        values_str = [str(v).strip() for _, v in non_null]

        # "Camera N" marker can appear between video blocks, at any state
        if len(non_null) == 1:
            cam_match = CAMERA_HEADER_RE.match(values_str[0])
            if cam_match:
                current_camera = int(cam_match.group(1))
                state = "WAITING_FOR_VIDEO"
                continue

        if state == "WAITING_FOR_VIDEO":
            current_video = values_str[0]
            state = "WAITING_FOR_HEADER"
            continue

        if state == "WAITING_FOR_HEADER":
            if "Interaction" in values_str:
                col_map = {c: str(v).strip() for c, v in non_null}
                state = "IN_BLOCK"
            # else: stray row before the header, ignore
            continue

        if state == "IN_BLOCK":
            record = {"Camera": current_camera, "Video": current_video}
            has_data = False
            for c, v in non_null:
                field = col_map.get(c)
                if field is None:
                    continue
                record[field] = v
                if field in ("Starting Frame", "Ending Frame"):
                    has_data = True
            if has_data:
                record["id"] = event_id
                event_id += 1
                records.append(record)
            continue

    df = pd.DataFrame(records)
    if df.empty:
        return df

    preferred = ["id", "Camera", "Video", "Interaction", "Number of persons involved",
                 "First person", "Second person", "Third person",
                 "Starting Frame", "Ending Frame"]
    cols = [c for c in preferred if c in df.columns] + \
           [c for c in df.columns if c not in preferred]
    return df[cols]


def parse_video_filename(path):
    """Extract (situation, camera_number) from a filename like
    '.../Cell_phone_Spanish/Cell_phone_Spanish.Cam1.avi'."""
    fname = os.path.basename(path)
    m = VIDEO_FILENAME_RE.match(fname)
    if not m:
        raise ValueError(f"Filename doesn't match expected pattern 'Situation.CamN.avi': {fname}")
    return m.group("situation"), int(m.group("cam"))


def get_violence_segments(df, video, camera, total_frames=None):
    """
    Returns None if this (video, camera) has no Fight-type interaction
    (i.e. it's not "useful" and should be skipped).

    Otherwise returns a sorted list of (label, start_frame, end_frame)
    tuples covering the whole video: label=1 for fight, 0 for everything
    else. If total_frames is given, a trailing (0, last_fight_end,
    total_frames) segment is appended when the video continues past the
    last fight.

    Handles multiple fight intervals per video (e.g. "Fight 1", "Fight 2")
    by sorting and treating each independently -- adjust here if
    overlapping fights should be merged instead.
    """
    subset = df[(df["Video"] == video) & (df["Camera"] == camera)]
    fight_rows = subset[subset["Interaction"].str.contains("Fight", case=False, na=False)]

    if fight_rows.empty:
        return None  # skip: not a violence-relevant video

    intervals = sorted(
        (int(r["Starting Frame"]), int(r["Ending Frame"]))
        for _, r in fight_rows.iterrows()
    )

    segments = []
    cursor = 0
    for start, end in intervals:
        if start > cursor:
            segments.append((0, cursor, start))
        segments.append((1, start, end))
        cursor = end

    if total_frames is not None and total_frames > cursor:
        segments.append((0, cursor, total_frames))

    return segments


def label_window(start_frame, end_frame, segments):
    """Binary label for an inference window: 1 if it overlaps any
    fight segment, else 0."""
    return int(any(
        label == 1 and start_frame <= seg_end and end_frame >= seg_start
        for label, seg_start, seg_end in segments
    ))


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
def run_inference(model, video_path, processor, clip_len=None, stride=None, batch_size=4, device="cuda"):
    if clip_len is None:
        clip_len = model.config.num_frames
    if stride is None:
        stride = 4 # getattr(model.config, "sampling_stride", 4) <- there is no such attribute, therefore hardcoded

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
        probs = torch.softmax(logits, dim=-1)[:, 1]  # P(violence) \in R^1
        probs = probs.cpu().numpy()
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
        print(f"Processed {round(indexes[-1] / fps, 2)} seconds")

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