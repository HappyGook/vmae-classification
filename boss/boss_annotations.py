import os
import re

import numpy as np
import pandas as pd


CAMERA_HEADER_RE = re.compile(r"^Camera\s*(\d+)$", re.IGNORECASE)
VIDEO_FILENAME_RE = re.compile(r"^(?P<situation>.+)\.Cam(?P<cam>\d+)\.avi$", re.IGNORECASE)
FRAME_FIELD_RE = re.compile(r"^(Starting|Ending)\s*Frame\s*(\d*)$", re.IGNORECASE)


def normalize_field(name: str) -> str:
    name = re.sub(r"\s+", " ", str(name).strip())
    match = FRAME_FIELD_RE.match(name)

    if not match:
        return name

    kind, number = match.group(1), match.group(2)
    return f"{kind} Frame {number if number else '1'}"


def parse_annotation_sheet(path, sheet_name=1):
    """
    Parse the BOSS annotation Excel sheet into one row per annotated interaction.
    """
    raw = pd.read_excel(path, sheet_name=sheet_name, header=None)

    records = []
    state = "WAITING_FOR_VIDEO"
    current_camera = None
    current_video = None
    column_map = None
    event_id = 1

    for _, row in raw.iterrows():
        non_null = [
            (column, value)
            for column, value in row.items()
            if pd.notna(value) and str(value).strip()
        ]

        if not non_null:
            if state == "IN_BLOCK":
                state = "WAITING_FOR_VIDEO"
                column_map = None
            continue

        values = [str(value).strip() for _, value in non_null]

        if len(non_null) == 1:
            camera_match = CAMERA_HEADER_RE.match(values[0])
            if camera_match:
                current_camera = int(camera_match.group(1))
                state = "WAITING_FOR_VIDEO"
                continue

        if state == "WAITING_FOR_VIDEO":
            current_video = values[0]
            state = "WAITING_FOR_HEADER"
            continue

        if state == "WAITING_FOR_HEADER":
            if "Interaction" in values:
                column_map = {
                    column: normalize_field(value)
                    for column, value in non_null
                }
                state = "IN_BLOCK"
            continue

        if state == "IN_BLOCK":
            record = {
                "Camera": current_camera,
                "Video": current_video,
            }
            has_frame_data = False

            for column, value in non_null:
                field = column_map.get(column)
                if field is None:
                    continue

                record[field] = value

                if FRAME_FIELD_RE.match(field):
                    has_frame_data = True

            if has_frame_data:
                record["id"] = event_id
                event_id += 1
                records.append(record)

    df = pd.DataFrame(records)

    if df.empty:
        return df

    frame_columns = sorted(
        [column for column in df.columns if FRAME_FIELD_RE.match(column)],
        key=lambda column: (
            column.split()[0],
            int(re.search(r"\d+", column).group()),
        ),
    )

    preferred_columns = [
        "id",
        "Camera",
        "Video",
        "Interaction",
        "Number of persons involved",
        "First person",
        "Second person",
        "Third person",
        *frame_columns,
    ]

    columns = [
        column for column in preferred_columns if column in df.columns
    ] + [
        column for column in df.columns if column not in preferred_columns
    ]

    return df[columns]


def parse_video_filename(path):
    """
    Extract BOSS situation and camera from filenames like:
    `Cell_phone_Spanish.Cam1.avi`.
    """
    filename = os.path.basename(path)
    match = VIDEO_FILENAME_RE.match(filename)

    if not match:
        raise ValueError(
            "Filename does not match expected pattern "
            f"'Situation.CamN.avi': {filename}"
        )

    return match.group("situation"), int(match.group("cam"))


def extract_frame_intervals(row):
    starts = {}
    ends = {}

    for column, value in row.items():
        if pd.isna(value):
            continue

        match = FRAME_FIELD_RE.match(str(column))
        if not match:
            continue

        kind = match.group(1).lower()
        number = match.group(2) or "1"

        if kind == "starting":
            starts[number] = value
        else:
            ends[number] = value

    interval_numbers = sorted(set(starts) & set(ends), key=lambda value: int(value))

    return [
        (int(starts[number]), int(ends[number]))
        for number in interval_numbers
    ]


def get_violence_segments(df, video, camera, total_frames=None):
    """
    Return labeled frame segments for one BOSS video/camera pair.

    Returns:
        None:
            If the video has no Fight interaction.
        list[tuple[int, int, int]]:
            Tuples of `(label, start_frame, end_frame)`.
            Label `1` means fight, label `0` means background.
    """
    subset = df[(df["Video"] == video) & (df["Camera"] == camera)]
    fight_rows = subset[
        subset["Interaction"].str.contains("Fight", case=False, na=False)
    ]

    if fight_rows.empty:
        return None

    intervals = []
    for _, row in fight_rows.iterrows():
        intervals.extend(extract_frame_intervals(row))

    if not intervals:
        return None

    intervals.sort()

    segments = []
    cursor = 0

    for start, end in intervals:
        if start > cursor:
            segments.append((0, cursor, start))

        segments.append((1, start, end))
        cursor = max(cursor, end)

    if total_frames is not None and total_frames > cursor:
        segments.append((0, cursor, total_frames))

    return segments


def label_window(start_frame, end_frame, segments):
    """Return 1 if the window overlaps any fight segment, else 0."""
    return int(
        any(
            label == 1 and start_frame <= segment_end and end_frame >= segment_start
            for label, segment_start, segment_end in segments
        )
    )


def true_timeline(segments, total_frames):
    """Build a per-frame binary ground-truth timeline."""
    timeline = np.zeros(total_frames, dtype=np.float32)

    for label, start, end in segments:
        if label == 1:
            timeline[start:end + 1] = 1.0

    return timeline