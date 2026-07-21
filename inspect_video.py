import json
import os
import matplotlib.pyplot as plt


def _mmss(seconds: float) -> str:
    m, s = divmod(seconds, 60)
    return f"{int(m):02d}:{s:05.2f}"


def load_result(path):
    """Load a result dict previously saved by evaluate_video(..., save=True)."""
    path = os.path.expanduser(path)
    with open(path, "r") as f:
        return json.load(f)



# 1. human-readable listing
def print_windows(result, mode="all"):
    """
    mode: "all" | "correct" | "errors"
    Prints one line per window: [mm:ss - mm:ss] score=.. true=.. pred=.. OK/WRONG
    """
    if mode == "errors":
        rows = result["errors"]
    elif mode == "correct":
        rows = [r for r in result["results"] if r["correct"]]
    else:
        rows = result["results"]

    print(f"\n=== {result['video']} (cam {result['camera']}) — {mode} windows "
          f"({len(rows)}) ===")
    for r in rows:
        flag = "OK   " if r["correct"] else "WRONG"
        print(
            f"[{_mmss(r['start_sec'])} - {_mmss(r['end_sec'])}] "
            f"score={r['score']:.3f}  true={r['true_label']}  "
            f"pred={r['pred_label']}  {flag}"
        )


# 2. plot pred vs. true timeline, seconds on the x-axis
def plot_timelines(result, threshold=0.5, save_path=None):
    fps = result["fps"]
    pred = result["pred_timeline"]
    true = result["true_timeline"]
    n = min(len(pred), len(true))
    t = [i / fps for i in range(n)]

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(t, pred[:n], label="predicted score", color="tab:blue", linewidth=1)
    ax.plot(t, true[:n], label="ground truth", color="tab:orange",
            linewidth=1.5, linestyle="--")
    ax.axhline(threshold, color="gray", linestyle=":", linewidth=1,
               label=f"threshold={threshold}")

    # shade frames where pred != true (using the same binarization as evaluate_video)
    mismatch = [(p >= threshold) != bool(tr) for p, tr in zip(pred[:n], true[:n])]
    in_run = False
    for i, bad in enumerate(mismatch):
        if bad and not in_run:
            start = t[i]
            in_run = True
        if in_run and (not bad or i == n - 1):
            end = t[i]
            ax.axvspan(start, end, color="red", alpha=0.15)
            in_run = False

    ax.set_xlabel("time (s)")
    ax.set_ylabel("score / label")
    ax.set_title(f"{result['video']} — camera {result['camera']}")
    ax.legend(loc="upper right")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved plot to {save_path}")
    return fig



if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("result_json", help="path to a saved evaluate_video() JSON")
    ap.add_argument("--video", help="source video, needed for --clips")
    ap.add_argument("--mode", default="errors", choices=["all", "correct", "errors"])
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--clips", action="store_true")
    ap.add_argument("--out", default="./clip_out")
    args = ap.parse_args()

    result = load_result(args.result_json)
    print_windows(result, mode=args.mode)

    if args.plot:
        fig = plot_timelines(result)
        plt.show()