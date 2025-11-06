"""
Run SeqTrack inference/evaluation on LaSOT test sequences using checkpoints from Hugging Face.

Example (Windows CMD):
> set HUGGINGFACE_TOKEN=hf_...
> python run_inference_eval.py --phase 2 --hf_repo saifmamdouh11/seqtrack_assignment3 ^
    --local_ckpt "F:/image/assignment_4/checkpoints/epoch_05.pth" ^
    --subset_json "F:/image/assignment_3/lasot_subset/subset_info.json" ^
    --dataset_root "F:/image/assignment_3/dataset" ^
    --out_dir eval_phase2 --device cuda
"""

import os
import time
import json
import importlib
import argparse
import glob
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

try:
    from huggingface_hub import snapshot_download
except Exception:
    snapshot_download = None


# ---------- Utilities ----------
def mkdirp(p):
    os.makedirs(p, exist_ok=True)


def extract_epoch_from_path(ckpt_path: str):
    """Try to extract epoch number from checkpoint filename."""
    import re
    m = re.search(r"epoch[_\-]?(\d+)", ckpt_path)
    if m:
        return int(m.group(1))
    return None


def bbox_iou(boxA, boxB):
    def to_x1y1x2y2(b):
        if len(b) == 4:
            x, y, w, h = b
            return [x, y, x + w, y + h]
        return b

    A = to_x1y1x2y2(boxA)
    B = to_x1y1x2y2(boxB)
    xA, yA, xB, yB = max(A[0], B[0]), max(A[1], B[1]), min(A[2], B[2]), min(A[3], B[3])
    interW, interH = max(0, xB - xA), max(0, yB - yA)
    interArea = interW * interH
    union = (A[2]-A[0])*(A[3]-A[1]) + (B[2]-B[0])*(B[3]-B[1]) - interArea
    return interArea / union if union > 0 else 0.0


def center_error(boxA, boxB):
    ax, ay, aw, ah = boxA
    bx, by, bw, bh = boxB
    return ((ax + aw/2 - (bx + bw/2))**2 + (ay + ah/2 - (bx + bw/2))**2) ** 0.5


# ---------- Dataset ----------
class LaSOTSubsetDataset(Dataset):
    def __init__(self, subset_json, dataset_root):
        with open(subset_json, "r") as f:
            subset = json.load(f)
        self.dataset_root = Path(dataset_root)
        self.samples = []
        for cls, info in subset.items():
            for seq in info.get("test", []):
                img_dir = self.dataset_root / cls / seq / "img"
                gt_file = self.dataset_root / cls / seq / "groundtruth.txt"
                if not img_dir.exists() or not gt_file.exists():
                    continue
                imgs = sorted(glob.glob(str(img_dir / "*.jpg")) + glob.glob(str(img_dir / "*.png")))
                if not imgs:
                    continue
                with open(gt_file) as f:
                    gts = [ln.strip() for ln in f.readlines()]
                self.samples.append((cls, seq, imgs, gts))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


# ---------- Model loader ----------
def build_model_from_repo(device):
    candidates = [
        ("lib.models.seqtrackv2.seqtrackv2", "build_seqtrackv2", "lib.config.seqtrackv2.config"),
        ("lib.models.seqtrack.seqtrack", "build_seqtrack", "lib.config.seqtrack.config"),
    ]
    for mod_name, fn_name, cfg_module in candidates:
        try:
            mod = importlib.import_module(mod_name)
            cfg = importlib.import_module(cfg_module).cfg
            fn = getattr(mod, fn_name)
            model = fn(cfg).to(device)
            print(f"[INFO] Built model using {mod_name}.{fn_name}")
            return model, cfg
        except Exception as e:
            print(f"[WARNING] Failed to load {mod_name}: {e}")
    raise RuntimeError("Failed to build SeqTrack model.")


def load_checkpoint_to_model(ckpt_path, model, device):
    print(f"[INFO] Loading checkpoint: {ckpt_path}")
    import torch.serialization as serialization
    serialization.add_safe_globals([__import__("numpy").core.multiarray._reconstruct])

    try:
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ck = torch.load(ckpt_path, map_location=device)

    if isinstance(ck, dict):
        if "model" in ck:
            state = ck["model"]
        elif "state_dict" in ck:
            state = ck["state_dict"]
        else:
            state = ck
    else:
        state = ck

    new_state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(new_state, strict=False)
    print("[INFO] ✅ Checkpoint loaded successfully.")


# ---------- Inference ----------
def run_single_sequence(model, imgs, gts, device, transform):
    model.eval()
    ious, centers, preds, times = [], [], [], []

    if not imgs or len(imgs) < 2:
        return {"ious": [], "center_errors": [], "preds": [], "times": []}

    with torch.no_grad():
        template_img = Image.open(imgs[0]).convert("RGB")
        template_tensor = transform(template_img).unsqueeze(0).to(device)

        for idx in range(1, len(imgs)):
            search_img = Image.open(imgs[idx]).convert("RGB")
            search_tensor = transform(search_img).unsqueeze(0).to(device)

            images_list = [template_tensor, search_tensor]
            t0 = time.time()
            try:
                out = model(images_list)
            except Exception as e:
                print(f"[WARNING] Forward failed ({e}), retrying...")
                out = model(search_tensor)
            t1 = time.time()
            times.append(t1 - t0)

            pred_box = None
            if isinstance(out, dict):
                for k in ("pred_boxes", "boxes", "output_bbox", "pred_bbox"):
                    if k in out:
                        pb = out[k]
                        if isinstance(pb, torch.Tensor):
                            pred_box = pb.cpu().squeeze().tolist()[:4]
                            break
            elif isinstance(out, torch.Tensor):
                pred_box = out.cpu().view(-1).tolist()[:4]
            if pred_box is None:
                pred_box = [0, 0, 0, 0]

            preds.append(pred_box)

            if idx < len(gts):
                parts = gts[idx].replace(",", " ").split()
                try:
                    gt_box = [float(x) for x in parts[:4]]
                except:
                    gt_box = [0, 0, 0, 0]
                ious.append(bbox_iou(pred_box, gt_box))
                centers.append(center_error(pred_box, gt_box))
            else:
                ious.append(0.0)
                centers.append(float("inf"))

    return {"ious": ious, "center_errors": centers, "preds": preds, "times": times}


# ---------- Metrics ----------
def compute_success_auc(ious):
    t = np.linspace(0, 1, 101)
    rates = [(np.array(ious) >= thr).mean() for thr in t]
    return np.trapz(rates, t)


def compute_precision(center_errors, thr=20):
    ce = np.array(center_errors)
    valid = np.isfinite(ce)
    return float((ce[valid] < thr).mean()) if valid.any() else 0.0


# ---------- Evaluation ----------
def evaluate_checkpoint(hf_repo, hf_path, local_ckpt, phase, subset_json, dataset_root, out_dir, device_str="cpu"):
    device = torch.device(device_str)
    mkdirp(out_dir)

    epoch_num = extract_epoch_from_path(local_ckpt or hf_path or "")
    if epoch_num:
        print(f"[INFO] Detected epoch: {epoch_num}")

    if hf_repo and snapshot_download:
        try:
            print("[INFO] Downloading from HF:", hf_repo)
            repo_local = snapshot_download(hf_repo, local_dir=out_dir)
            ckpt_candidate = Path(repo_local) / hf_path
            if ckpt_candidate.exists():
                local_ckpt = str(ckpt_candidate)
        except Exception as e:
            print("[WARNING] HF download failed:", e)

    model, cfg = build_model_from_repo(device)
    load_checkpoint_to_model(local_ckpt, model, device)

    ds = LaSOTSubsetDataset(subset_json, dataset_root)
    print(f"[INFO] {len(ds)} sequences to evaluate")

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    results = []
    for cls, seq, imgs, gts in ds:
        print(f"[INFO] Running {cls}/{seq} ({len(imgs)} frames)...")
        r = run_single_sequence(model, imgs, gts, device, transform)
        auc = compute_success_auc(r["ious"])
        prec = compute_precision(r["center_errors"])
        fps = len(r["times"]) / (sum(r["times"]) + 1e-8)
        mean_iou = np.mean(r["ious"]) if len(r["ious"]) > 0 else 0.0
        total_time = np.sum(r["times"]) if len(r["times"]) > 0 else 0.0

        results.append({
            "phase": phase,
            "epoch": epoch_num,
            "class": cls,
            "seq": seq,
            "frame_count": len(imgs),
            "auc": auc,
            "mean_iou": mean_iou,
            "precision20": prec,
            "fps": fps,
            "mean_time_ms": np.mean(r["times"]) * 1000 if r["times"] else 0.0,
            "total_time_s": total_time,
            "min_iou": np.min(r["ious"]) if len(r["ious"]) > 0 else 0.0,
            "max_iou": np.max(r["ious"]) if len(r["ious"]) > 0 else 0.0,
        })

    summary_name = f"{phase}_epoch{epoch_num}_summary.json" if epoch_num else f"{phase}_summary.json"
    out_path = Path(out_dir) / summary_name
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print("[INFO] Results saved to", out_path)


# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", type=int, choices=[1, 2], required=True)
    p.add_argument("--hf_repo", type=str)
    p.add_argument("--hf_model_path", type=str)
    p.add_argument("--local_ckpt", type=str)
    p.add_argument("--subset_json", type=str)
    p.add_argument("--dataset_root", type=str)
    p.add_argument("--out_dir", type=str, default="eval_out")
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


def main():
    args = parse_args()
    evaluate_checkpoint(
        args.hf_repo,
        args.hf_model_path,
        args.local_ckpt,
        f"phase{args.phase}",
        args.subset_json,
        args.dataset_root,
        args.out_dir,
        device_str=args.device
    )


if __name__ == "__main__":
    main()
