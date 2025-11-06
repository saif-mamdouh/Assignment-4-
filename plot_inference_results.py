# plot_inference_results.py
"""
Plot inference evaluation results for Assignment 4.

Usage:
    python plot_inference_results.py --phase1 phase1_results.json --phase2 phase2_results.json --out_dir figs

Input format:
    Each JSON file must be a list of dicts with keys including:
      - "phase": "phase1" or "phase2"
      - "epoch": int
      - "class": str
      - "seq": str
      - "frame_count": int
      - "auc": float
      - "mean_iou": float
      - "precision20": float
      - "fps": float
      - "mean_time_ms": float
"""

import json
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def aggregate_by_epoch(records):
    """
    records: list of dicts
    returns dict: epoch -> aggregated metrics (mean over sequences for that epoch)
    """
    by_epoch = {}
    for r in records:
        ep = int(r.get("epoch", 0))
        if ep not in by_epoch:
            by_epoch[ep] = {"auc": [], "mean_iou": [], "precision20": [], "fps": [], "mean_time_ms": [], "frame_count": []}
        # safe get floats
        for k in ("auc","mean_iou","precision20","fps","mean_time_ms","frame_count"):
            v = r.get(k, None)
            if v is None:
                continue
            by_epoch[ep][k].append(float(v))
    # compute mean
    agg = {}
    for ep, d in sorted(by_epoch.items()):
        agg[ep] = {
            "auc": float(np.mean(d["auc"])) if len(d["auc"]) else None,
            "mean_iou": float(np.mean(d["mean_iou"])) if len(d["mean_iou"]) else None,
            "precision20": float(np.mean(d["precision20"])) if len(d["precision20"]) else None,
            "fps": float(np.mean(d["fps"])) if len(d["fps"]) else None,
            "mean_time_ms": float(np.mean(d["mean_time_ms"])) if len(d["mean_time_ms"]) else None,
            "total_frames": int(np.sum(d["frame_count"])) if len(d["frame_count"]) else 0
        }
    return agg

def plot_metric(epoch_list, vals1, vals2, ylabel, outpath, label1="phase1", label2="phase2"):
    plt.figure(figsize=(8,5))
    plt.plot(epoch_list, vals1, marker='o', label=label1)
    plt.plot(epoch_list, vals2, marker='x', label=label2)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def main(args):
    p1 = load_json(args.phase1)
    p2 = load_json(args.phase2)

    agg1 = aggregate_by_epoch(p1)
    agg2 = aggregate_by_epoch(p2)

    # union of epochs
    epochs = sorted(set(list(agg1.keys()) + list(agg2.keys())))
    if not epochs:
        raise RuntimeError("No epochs found in inputs.")

    # build vectors (use np.nan where missing)
    def vec(metric, agg):
        return [agg.get(ep, {}).get(metric, np.nan) for ep in epochs]

    # metrics to plot
    metrics = [
        ("mean_iou", "Mean IoU", "iou_vs_epoch.png"),
        ("precision20", "Precision @20px", "precision20_vs_epoch.png"),
        ("auc", "Success AUC", "auc_vs_epoch.png"),
        ("fps", "FPS (frames/sec)", "fps_vs_epoch.png"),
        ("mean_time_ms", "Mean time per frame (ms)", "mean_time_ms_vs_epoch.png"),
    ]

    os.makedirs(args.out_dir, exist_ok=True)

    for key, ylabel, fname in metrics:
        v1 = vec(key, agg1)
        v2 = vec(key, agg2)
        out = os.path.join(args.out_dir, fname)
        plot_metric(epochs, v1, v2, ylabel, out, label1="phase1", label2="phase2")
        print(f"Saved {out}")

    # also save CSV summary per phase
    def save_csv(agg, outcsv):
        with open(outcsv, "w", encoding="utf-8") as f:
            f.write("epoch,mean_iou,precision20,auc,fps,mean_time_ms,total_frames\n")
            for ep in sorted(agg.keys()):
                a = agg[ep]
                f.write(f"{ep},{a['mean_iou']},{a['precision20']},{a['auc']},{a['fps']},{a['mean_time_ms']},{a['total_frames']}\n")
    save_csv(agg1, os.path.join(args.out_dir, "phase1_summary.csv"))
    save_csv(agg2, os.path.join(args.out_dir, "phase2_summary.csv"))
    print("Saved CSV summaries.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1", required=True, help="JSON file with phase1 results")
    parser.add_argument("--phase2", required=True, help="JSON file with phase2 results")
    parser.add_argument("--out_dir", default="figs", help="output directory for figures/csv")
    args = parser.parse_args()
    main(args)
