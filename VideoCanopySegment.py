import os
import sys
import csv
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import argparse
import glob
import numpy as np
import cv2
from PIL import Image

import torch
import time
import matplotlib
matplotlib.use("Agg")  # headless save only
import matplotlib.pyplot as plt

# ---- SAM3 repo path (change if needed) ----
SAM3_REPO_ROOT = Path("/home/kaishen/Documents/sam3")
if str(SAM3_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(SAM3_REPO_ROOT))

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# sklearn/pandas for clustering + export
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture



# ----------------------------
# Config
# ----------------------------
@dataclass
class Cfg:
    # I/O (set your two videos here)
    rgb_video_path: str = "/home/kaishen/Documents/sam3/data/20260120_Plot_Noon_src.mp4"
    depth_video_path: str = "/home/kaishen/Documents/sam3/data/20260120_Plot_Noon_vis.mp4"
    prompt_text: str = "plant"

    # frame sampling (MUST be the same for RGB and depth)
    frames_fps: int = 15

    # ----------------------------
    # HitProcess (KMeans rows + circle hit) (replaces DBSCAN)
    # ----------------------------
    hit_k_rows: int = 2
    hit_circle_r: int = 50
    hit_no_largest_rgb: bool = False

    jpg_quality: int = 2  # ffmpeg -q:v (lower is higher quality)

    # Depth brightness filtering: keep top X% brightest depth pixels
    depth_keep_brightness_frac: float = 0.3
    depth_nonzero_only: bool = True  # ignore zeros when computing percentile

    # Top-K component selection (from your combined pipeline)
    top_k_components: int = 6 #6
    min_component_area_px: float = 200 #20
    rel_area_gate: float = 0.2 # 0

    # Official Y-band (keep center fraction)
    y_keep_center_frac: float = 1 # change to 0.7
	
    # Crop guardrails
    min_crop_radius_px: int = 30
    merge_close_px: int = 1   # if masks/components are within 50 px, merge them
    
    depth_dynamic: bool = True
    depth_gmm_k: int = 3
    depth_p_left: float = 99.0
    depth_min_keep: float = 0.2
    depth_keep_step: float = 0.10
    depth_open_k: int = 50
    depth_close_k: int = 100
    depth_min_area: int = 30
    depth_max_vals_for_gmm: int = 200_000
    depth_downsample_stride: int = 8
    
    # ----------------------------
    # Refinement loop (depth + SAM3 rerun)
    # ----------------------------
    refine_enable: bool = True
    refine_area_mult: float = 1.3        # target: max_area >= 2 * mean_area
    refine_max_iters: int = 8              # keep small; SAM3 is expensive
    refine_thr_step: int = 6               # threshold +5 each iter (uint8 depth)
    refine_min_component_area: int = 5000   # used to count "split" components
    refine_warmup_frames: int = 30         # build mean_area estimate before enforcing

cfg = Cfg()

def resolve_multiple_video_pairs(parent_dir: Path):
    """
    Find paired *_src.mp4 and *_vis.mp4 files and return
    a sorted list of (rgb_path, depth_path).
    """
    parent_dir = Path(parent_dir)
    if not parent_dir.exists():
        raise FileNotFoundError(parent_dir)

    src_videos = sorted(parent_dir.glob("*_src.mp4"))

    pairs = []
    for src in src_videos:
        base = src.name.replace("_src.mp4", "")
        vis = parent_dir / f"{base}_vis.mp4"
        if vis.exists():
            pairs.append((src, vis))
        else:
            print(f"⚠️ Missing depth video for {src.name}")

    if not pairs:
        raise RuntimeError("No valid *_src/_vis video pairs found")

    return pairs
def crop_to_y_band(img: np.ndarray, y_min: int, y_max: int) -> np.ndarray:
    """Crop image to [y_min:y_max, :] (safe clamp)."""
    H = img.shape[0]
    y_min = max(0, min(int(y_min), H))
    y_max = max(y_min + 1, min(int(y_max), H))
    return img[y_min:y_max, :].copy()
    
def resolve_videos_from_seq_dir(seq_dir: Path) -> tuple[str, str]:
    """
    Given a folder like .../20260120_Plot_Noon, find:
      20260120_Plot_Noon_src.mp4  (RGB)
      20260120_Plot_Noon_vis.mp4  (Depth)

    Fallback: if exact names not found, try glob *src*.mp4 / *vis*.mp4.
    """
    seq_dir = Path(seq_dir)
    if not seq_dir.exists() or not seq_dir.is_dir():
        raise FileNotFoundError(f"-s must be an existing folder: {seq_dir}")

    base = seq_dir.name
    rgb = seq_dir / f"{base}_src.mp4"
    dep = seq_dir / f"{base}_vis.mp4"

    if rgb.exists() and dep.exists():
        return str(rgb), str(dep)

    # fallback: glob (handles slightly different naming)
    rgb_cands = sorted(seq_dir.glob("*src*.mp4"))
    dep_cands = sorted(seq_dir.glob("*vis*.mp4"))

    if not rgb.exists() and rgb_cands:
        rgb = rgb_cands[0]
    if not dep.exists() and dep_cands:
        dep = dep_cands[0]

    if not Path(rgb).exists():
        raise FileNotFoundError(f"RGB video not found. Expected {seq_dir}/{base}_src.mp4 (or *src*.mp4)")
    if not Path(dep).exists():
        raise FileNotFoundError(f"Depth video not found. Expected {seq_dir}/{base}_vis.mp4 (or *vis*.mp4)")

    return str(rgb), str(dep)
    
# ----------------------------
# 1) Extract frames at fps
# ----------------------------
def extract_frames_fps(
    video_path: str,
    fps: int = 5,
    jpg_quality: int = 2,
    start_time: str = None,
    duration: str = None
) -> Path:
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    out_dir = video_path.parent / f"{video_path.stem}_{fps}fps"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = ["ffmpeg", "-hide_banner", "-y"]

    # 🔹 FAST seek (before -i)
    if start_time is not None:
        cmd += ["-ss", str(start_time)]

    cmd += ["-i", str(video_path)]

    # 🔹 optional duration
    if duration is not None:
        cmd += ["-t", str(duration)]

    cmd += [
        "-vf", f"fps={fps}",
        "-q:v", str(jpg_quality),
        str(out_dir / "%06d.jpg"),
    ]

    subprocess.run(cmd, check=True)
    print(f"✅ Extracted frames to: {out_dir}")
    return out_dir


# ----------------------------
# 2) SAM3 single-image runner
# ----------------------------
def run_sam3_on_pil(image_pil: Image.Image, processor: Sam3Processor, prompt: str):
    """
    Returns:
      masks_np: (N, H, W) bool
      boxes_np: (N, 4) or None
      scores_np: (N,) or None
    """
    state = processor.set_image(image_pil)
    output = processor.set_text_prompt(state=state, prompt=prompt)

    masks = output["masks"]
    boxes = output.get("boxes", None)
    scores = output.get("scores", None)

    if isinstance(masks, torch.Tensor):
        masks = masks.detach().cpu().numpy()
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.detach().cpu().numpy()
    if isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu().numpy()

    # (N,1,H,W) -> (N,H,W)
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    elif masks.ndim != 3:
        raise ValueError(f"Unexpected mask shape: {masks.shape}")

    if masks.dtype != bool:
        masks = masks > 0.5

    return masks, boxes, scores


def union_mask(masks_bool: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """
    masks_bool: (N,H,W) bool
    return: (H,W) uint8 0/1
    """
    if masks_bool is None or len(masks_bool) == 0:
        return None
    return (masks_bool.sum(axis=0) > 0).astype(np.uint8)


# ----------------------------
# Depth helpers: keep top brightness pixels
# ----------------------------
def denoise_binary_mask(mask01: np.ndarray,
                        open_ksize: int = 3,
                        close_ksize: int = 5,
                        min_area: int = 300) -> np.ndarray:
    if mask01.dtype != np.uint8:
        mask01 = mask01.astype(np.uint8)

    m = (mask01 > 0).astype(np.uint8) * 255

    if open_ksize and open_ksize > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)

    if close_ksize and close_ksize > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=1)

    num, lab, stats, _ = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8), connectivity=8)
    out = np.zeros_like(m)
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area >= int(min_area):
            out[lab == i] = 255

    return (out > 0).astype(np.uint8)  # {0,1}


def threshold_for_keep_fraction(vals: np.ndarray, keep_frac: float) -> int:
    keep_frac = float(np.clip(keep_frac, 0.0, 1.0))
    if keep_frac <= 0:
        return 255
    if keep_frac >= 1:
        return 0
    thr = np.quantile(vals, 1.0 - keep_frac, method="linear")
    return int(np.clip(np.round(thr), 0, 255))


def contours_and_total_area(mask01: np.ndarray):
    mask255 = (mask01 > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total_area = float(sum(cv2.contourArea(c) for c in contours))
    return len(contours), total_area
    
def contour_areas(mask01: np.ndarray) -> list[float]:
    mask255 = (mask01 > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [float(cv2.contourArea(c)) for c in contours]
def refine_depth_keep_with_sam3(
    *,
    frame_bgr: np.ndarray,
    depth_bgr: np.ndarray,
    image_pil: Image.Image,
    processor,
    cfg,
    thr_start: int,
    mean_area_est: float | None,
    use_amp: bool,
    timing_acc: dict | None = None,
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, int, np.ndarray]:
    """
    Refinement loop:
      while (max_area < refine_area_mult * mean_area_est) OR (mask splits to >=2 big components),
      increase depth threshold and rerun SAM3.

    Returns:
      depth_keep01, masks_depth, mask01 (union+depth gate), thr_final
    """
    H, W = frame_bgr.shape[:2]

    # if we don't have a stable mean yet, don't force refinement
    if (mean_area_est is None) or (mean_area_est <= 0) or (mean_area_est != mean_area_est):
        # run once with thr_start
        thr = int(thr_start)
        depth_gray = cv2.cvtColor(depth_bgr, cv2.COLOR_BGR2GRAY) if depth_bgr.ndim == 3 else depth_bgr.copy()
        valid = depth_gray > 0
        depth_keep01 = ((depth_gray >= thr) & valid).astype(np.uint8)

        with torch.inference_mode():
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    t_sam = _now()
                    t_sam = _now()
                    t_sam = _now()
                    t_sam = _now()
                    t_sam = _now()
                    t_sam = _now()
                    t_sam = _now()
                    t_sam = _now()
                    t_sam = _now()
                    t_sam = _now()
                    t_sam = _now()
                    t_sam = _now()
                    t_sam = _now()
                    t_sam = _now()
                    t_sam = _now()
                    t_sam = _now()
                    t_sam = _now()
                    t_sam = _now()
                    t_sam = _now()
                    t_sam = _now()
                    t_sam = _now()
                    t_sam = _now()
                    t_sam = _now()
                    t_sam = _now()
                    t_sam = _now()
                    t_sam = _now()
                    t_sam = _now()
                    t_sam = _now()
                    t_sam = _now()
                    t_sam = _now()
                    t_sam = _now()
                    t_sam = _now()
                    t_sam = _now()
                    t_sam = _now()
                    t_sam = _now()
                    t_sam = _now()
                    t_sam = _now()
                    t_sam = _now()
                    t_sam = _now()
                    t_sam = _now()
                    t_sam = _now()
                    t_sam = _now()
                    t_sam = _now()
                    t_sam = _now()
                    t_sam = _now()
                    t_sam = _now()
                    t_sam = _now()
                    t_sam = _now()
                    t_sam = _now()
                    t_sam = _now()
                    t_sam = _now()
                    t_sam = _now()
                    t_sam = _now()
                    t_sam = _now()
                    t_sam = _now()
                    t_sam = _now()
                    t_sam = _now()
                    t_sam = _now()
                    t_sam = _now()
                    t_sam = _now()
                    t_sam = _now()
                    t_sam = _now()
                    t_sam = _now()
                    t_sam = _now()
                    t_sam = _now()
                    t_sam = _now()
                    t_sam = _now()
                    t_sam = _now()
                    t_sam = _now()
                    t_sam = _now()
                    t_sam = _now()
                    t_sam = _now()
                    t_sam = _now()
                    t_sam = _now()
                    t_sam = _now()
                    t_sam = _now()
                    t_sam = _now()
                    t_sam = _now()
                    t_sam = _now()
                    t_sam = _now()
                    t_sam = _now()
                    t_sam = _now()
                    t_sam = _now()
                    t_sam = _now()
                    t_sam = _now()
                    t_sam = _now()
                    t_sam = _now()
                    t_sam = _now()
                    t_sam = _now()
                    t_sam = _now()
                    masks, _, _ = run_sam3_on_pil(image_pil, processor, prompt=cfg.prompt_text)
                    if timing_acc is not None:
                        timing_acc["stage4_sam3"] = float(timing_acc.get("stage4_sam3", 0.0)) + (_now() - t_sam)
                    if timing_acc is not None:
                        timing_acc["stage4_sam3"] = float(timing_acc.get("stage4_sam3", 0.0)) + (_now() - t_sam)
                    if timing_acc is not None:
                        timing_acc["stage4_sam3"] = float(timing_acc.get("stage4_sam3", 0.0)) + (_now() - t_sam)
                    if timing_acc is not None:
                        timing_acc["stage4_sam3"] = float(timing_acc.get("stage4_sam3", 0.0)) + (_now() - t_sam)
                    if timing_acc is not None:
                        timing_acc["stage4_sam3"] = float(timing_acc.get("stage4_sam3", 0.0)) + (_now() - t_sam)
                    if timing_acc is not None:
                        timing_acc["stage4_sam3"] = float(timing_acc.get("stage4_sam3", 0.0)) + (_now() - t_sam)
                    if timing_acc is not None:
                        timing_acc["stage4_sam3"] = float(timing_acc.get("stage4_sam3", 0.0)) + (_now() - t_sam)
                    if timing_acc is not None:
                        timing_acc["stage4_sam3"] = float(timing_acc.get("stage4_sam3", 0.0)) + (_now() - t_sam)
                    if timing_acc is not None:
                        timing_acc["stage4_sam3"] = float(timing_acc.get("stage4_sam3", 0.0)) + (_now() - t_sam)
                    if timing_acc is not None:
                        timing_acc["stage4_sam3"] = float(timing_acc.get("stage4_sam3", 0.0)) + (_now() - t_sam)
                    if timing_acc is not None:
                        timing_acc["stage4_sam3"] = float(timing_acc.get("stage4_sam3", 0.0)) + (_now() - t_sam)
                    if timing_acc is not None:
                        timing_acc["stage4_sam3"] = float(timing_acc.get("stage4_sam3", 0.0)) + (_now() - t_sam)
                    if timing_acc is not None:
                        timing_acc["stage4_sam3"] = float(timing_acc.get("stage4_sam3", 0.0)) + (_now() - t_sam)
                    if timing_acc is not None:
                        timing_acc["stage4_sam3"] = float(timing_acc.get("stage4_sam3", 0.0)) + (_now() - t_sam)
                    if timing_acc is not None:
                        timing_acc["stage4_sam3"] = float(timing_acc.get("stage4_sam3", 0.0)) + (_now() - t_sam)
                    if timing_acc is not None:
                        timing_acc["stage4_sam3"] = float(timing_acc.get("stage4_sam3", 0.0)) + (_now() - t_sam)
                    if timing_acc is not None:
                        timing_acc["stage4_sam3"] = float(timing_acc.get("stage4_sam3", 0.0)) + (_now() - t_sam)
                    if timing_acc is not None:
                        timing_acc["stage4_sam3"] = float(timing_acc.get("stage4_sam3", 0.0)) + (_now() - t_sam)
                    if timing_acc is not None:
                        timing_acc["stage4_sam3"] = float(timing_acc.get("stage4_sam3", 0.0)) + (_now() - t_sam)
                    if timing_acc is not None:
                        timing_acc["stage4_sam3"] = float(timing_acc.get("stage4_sam3", 0.0)) + (_now() - t_sam)
                    if timing_acc is not None:
                        timing_acc["stage4_sam3"] = float(timing_acc.get("stage4_sam3", 0.0)) + (_now() - t_sam)
                    if timing_acc is not None:
                        timing_acc["stage4_sam3"] = float(timing_acc.get("stage4_sam3", 0.0)) + (_now() - t_sam)
                    if timing_acc is not None:
                        timing_acc["stage4_sam3"] = float(timing_acc.get("stage4_sam3", 0.0)) + (_now() - t_sam)
                    if timing_acc is not None:
                        timing_acc["stage4_sam3"] = float(timing_acc.get("stage4_sam3", 0.0)) + (_now() - t_sam)
                    if timing_acc is not None:
                        timing_acc["stage4_sam3"] = float(timing_acc.get("stage4_sam3", 0.0)) + (_now() - t_sam)
                    if timing_acc is not None:
                        timing_acc["stage4_sam3"] = float(timing_acc.get("stage4_sam3", 0.0)) + (_now() - t_sam)
                    if timing_acc is not None:
                        timing_acc["stage4_sam3"] = float(timing_acc.get("stage4_sam3", 0.0)) + (_now() - t_sam)
                    if timing_acc is not None:
                        timing_acc["stage4_sam3"] = float(timing_acc.get("stage4_sam3", 0.0)) + (_now() - t_sam)
                    if timing_acc is not None:
                        timing_acc["stage4_sam3"] = float(timing_acc.get("stage4_sam3", 0.0)) + (_now() - t_sam)
                    if timing_acc is not None:
                        timing_acc["stage4_sam3"] = float(timing_acc.get("stage4_sam3", 0.0)) + (_now() - t_sam)
                    if timing_acc is not None:
                        timing_acc["stage4_sam3"] = float(timing_acc.get("stage4_sam3", 0.0)) + (_now() - t_sam)
                    if timing_acc is not None:
                        timing_acc["stage4_sam3"] = float(timing_acc.get("stage4_sam3", 0.0)) + (_now() - t_sam)
                    if timing_acc is not None:
                        timing_acc["stage4_sam3"] = float(timing_acc.get("stage4_sam3", 0.0)) + (_now() - t_sam)
                    if timing_acc is not None:
                        timing_acc["stage4_sam3"] = float(timing_acc.get("stage4_sam3", 0.0)) + (_now() - t_sam)
                    if timing_acc is not None:
                        timing_acc["stage4_sam3"] = float(timing_acc.get("stage4_sam3", 0.0)) + (_now() - t_sam)
                    if timing_acc is not None:
                        timing_acc["stage4_sam3"] = float(timing_acc.get("stage4_sam3", 0.0)) + (_now() - t_sam)
                    if timing_acc is not None:
                        timing_acc["stage4_sam3"] = float(timing_acc.get("stage4_sam3", 0.0)) + (_now() - t_sam)
                    if timing_acc is not None:
                        timing_acc["stage4_sam3"] = float(timing_acc.get("stage4_sam3", 0.0)) + (_now() - t_sam)
                    if timing_acc is not None:
                        timing_acc["stage4_sam3"] = float(timing_acc.get("stage4_sam3", 0.0)) + (_now() - t_sam)
                    if timing_acc is not None:
                        timing_acc["stage4_sam3"] = float(timing_acc.get("stage4_sam3", 0.0)) + (_now() - t_sam)
                    if timing_acc is not None:
                        timing_acc["stage4_sam3"] = float(timing_acc.get("stage4_sam3", 0.0)) + (_now() - t_sam)
                    if timing_acc is not None:
                        timing_acc["stage4_sam3"] = float(timing_acc.get("stage4_sam3", 0.0)) + (_now() - t_sam)
                    if timing_acc is not None:
                        timing_acc["stage4_sam3"] = float(timing_acc.get("stage4_sam3", 0.0)) + (_now() - t_sam)
                    if timing_acc is not None:
                        timing_acc["stage4_sam3"] = float(timing_acc.get("stage4_sam3", 0.0)) + (_now() - t_sam)
                    if timing_acc is not None:
                        timing_acc["stage4_sam3"] = float(timing_acc.get("stage4_sam3", 0.0)) + (_now() - t_sam)
                    if timing_acc is not None:
                        timing_acc["stage4_sam3"] = float(timing_acc.get("stage4_sam3", 0.0)) + (_now() - t_sam)
                    if timing_acc is not None:
                        timing_acc["stage4_sam3"] = float(timing_acc.get("stage4_sam3", 0.0)) + (_now() - t_sam)
                    if timing_acc is not None:
                        timing_acc["stage4_sam3"] = float(timing_acc.get("stage4_sam3", 0.0)) + (_now() - t_sam)
                    if timing_acc is not None:
                        timing_acc["stage4_sam3"] = float(timing_acc.get("stage4_sam3", 0.0)) + (_now() - t_sam)
                    if timing_acc is not None:
                        timing_acc["stage4_sam3"] = float(timing_acc.get("stage4_sam3", 0.0)) + (_now() - t_sam)
                    if timing_acc is not None:
                        timing_acc["stage4_sam3"] = float(timing_acc.get("stage4_sam3", 0.0)) + (_now() - t_sam)
                    if timing_acc is not None:
                        timing_acc["stage4_sam3"] = float(timing_acc.get("stage4_sam3", 0.0)) + (_now() - t_sam)
                    if timing_acc is not None:
                        timing_acc["stage4_sam3"] = float(timing_acc.get("stage4_sam3", 0.0)) + (_now() - t_sam)
                    if timing_acc is not None:
                        timing_acc["stage4_sam3"] = float(timing_acc.get("stage4_sam3", 0.0)) + (_now() - t_sam)
                    if timing_acc is not None:
                        timing_acc["stage4_sam3"] = float(timing_acc.get("stage4_sam3", 0.0)) + (_now() - t_sam)
                    if timing_acc is not None:
                        timing_acc["stage4_sam3"] = float(timing_acc.get("stage4_sam3", 0.0)) + (_now() - t_sam)
                    if timing_acc is not None:
                        timing_acc["stage4_sam3"] = float(timing_acc.get("stage4_sam3", 0.0)) + (_now() - t_sam)
                    if timing_acc is not None:
                        timing_acc["stage4_sam3"] = float(timing_acc.get("stage4_sam3", 0.0)) + (_now() - t_sam)
                    if timing_acc is not None:
                        timing_acc["stage4_sam3"] = float(timing_acc.get("stage4_sam3", 0.0)) + (_now() - t_sam)
                    if timing_acc is not None:
                        timing_acc["stage4_sam3"] = float(timing_acc.get("stage4_sam3", 0.0)) + (_now() - t_sam)
                    if timing_acc is not None:
                        timing_acc["stage4_sam3"] = float(timing_acc.get("stage4_sam3", 0.0)) + (_now() - t_sam)
                    if timing_acc is not None:
                        timing_acc["stage4_sam3"] = float(timing_acc.get("stage4_sam3", 0.0)) + (_now() - t_sam)
                    if timing_acc is not None:
                        timing_acc["stage4_sam3"] = float(timing_acc.get("stage4_sam3", 0.0)) + (_now() - t_sam)
                    if timing_acc is not None:
                        timing_acc["stage4_sam3"] = float(timing_acc.get("stage4_sam3", 0.0)) + (_now() - t_sam)
                    if timing_acc is not None:
                        timing_acc["stage4_sam3"] = float(timing_acc.get("stage4_sam3", 0.0)) + (_now() - t_sam)
                    if timing_acc is not None:
                        timing_acc["stage4_sam3"] = float(timing_acc.get("stage4_sam3", 0.0)) + (_now() - t_sam)
                    if timing_acc is not None:
                        timing_acc["stage4_sam3"] = float(timing_acc.get("stage4_sam3", 0.0)) + (_now() - t_sam)
                    if timing_acc is not None:
                        timing_acc["stage4_sam3"] = float(timing_acc.get("stage4_sam3", 0.0)) + (_now() - t_sam)
                    if timing_acc is not None:
                        timing_acc["stage4_sam3"] = float(timing_acc.get("stage4_sam3", 0.0)) + (_now() - t_sam)
                    if timing_acc is not None:
                        timing_acc["stage4_sam3"] = float(timing_acc.get("stage4_sam3", 0.0)) + (_now() - t_sam)
                    if timing_acc is not None:
                        timing_acc["stage4_sam3"] = float(timing_acc.get("stage4_sam3", 0.0)) + (_now() - t_sam)
                    if timing_acc is not None:
                        timing_acc["stage4_sam3"] = float(timing_acc.get("stage4_sam3", 0.0)) + (_now() - t_sam)
                    if timing_acc is not None:
                        timing_acc["stage4_sam3"] = float(timing_acc.get("stage4_sam3", 0.0)) + (_now() - t_sam)
                    if timing_acc is not None:
                        timing_acc["stage4_sam3"] = float(timing_acc.get("stage4_sam3", 0.0)) + (_now() - t_sam)
                    if timing_acc is not None:
                        timing_acc["stage4_sam3"] = float(timing_acc.get("stage4_sam3", 0.0)) + (_now() - t_sam)
                    if timing_acc is not None:
                        timing_acc["stage4_sam3"] = float(timing_acc.get("stage4_sam3", 0.0)) + (_now() - t_sam)
                    if timing_acc is not None:
                        timing_acc["stage4_sam3"] = float(timing_acc.get("stage4_sam3", 0.0)) + (_now() - t_sam)
                    if timing_acc is not None:
                        timing_acc["stage4_sam3"] = float(timing_acc.get("stage4_sam3", 0.0)) + (_now() - t_sam)
                    if timing_acc is not None:
                        timing_acc["stage4_sam3"] = float(timing_acc.get("stage4_sam3", 0.0)) + (_now() - t_sam)
                    if timing_acc is not None:
                        timing_acc["stage4_sam3"] = float(timing_acc.get("stage4_sam3", 0.0)) + (_now() - t_sam)
                    if timing_acc is not None:
                        timing_acc["stage4_sam3"] = float(timing_acc.get("stage4_sam3", 0.0)) + (_now() - t_sam)
                    if timing_acc is not None:
                        timing_acc["stage4_sam3"] = float(timing_acc.get("stage4_sam3", 0.0)) + (_now() - t_sam)
                    if timing_acc is not None:
                        timing_acc["stage4_sam3"] = float(timing_acc.get("stage4_sam3", 0.0)) + (_now() - t_sam)
                    if timing_acc is not None:
                        timing_acc["stage4_sam3"] = float(timing_acc.get("stage4_sam3", 0.0)) + (_now() - t_sam)
                    if timing_acc is not None:
                        timing_acc["stage4_sam3"] = float(timing_acc.get("stage4_sam3", 0.0)) + (_now() - t_sam)
                    if timing_acc is not None:
                        timing_acc["stage4_sam3"] = float(timing_acc.get("stage4_sam3", 0.0)) + (_now() - t_sam)
                    if timing_acc is not None:
                        timing_acc["stage4_sam3"] = float(timing_acc.get("stage4_sam3", 0.0)) + (_now() - t_sam)
                    if timing_acc is not None:
                        timing_acc["stage4_sam3"] = float(timing_acc.get("stage4_sam3", 0.0)) + (_now() - t_sam)
                    if timing_acc is not None:
                        timing_acc["stage4_sam3"] = float(timing_acc.get("stage4_sam3", 0.0)) + (_now() - t_sam)
                    if timing_acc is not None:
                        timing_acc["stage4_sam3"] = float(timing_acc.get("stage4_sam3", 0.0)) + (_now() - t_sam)
            else:
                masks, _, _ = run_sam3_on_pil(image_pil, processor, prompt=cfg.prompt_text)

        masks_depth = []
        for m in masks:
            m01 = m.astype(np.uint8)
            md = (m01 & depth_keep01).astype(np.uint8)
            if int(np.count_nonzero(md)) > 0:
                masks_depth.append(md)

        # merge close masks (same as your existing code)
        if int(cfg.merge_close_px) > 0 and len(masks_depth) > 1:
            masks255 = [(md.astype(np.uint8) * 255) for md in masks_depth]
            merged255 = merge_overlapping_masks(masks255, dilation_px=int(cfg.merge_close_px))
            masks_depth = [(m.astype(np.uint8) > 0).astype(np.uint8) for m in merged255]

        mask01 = union_mask(masks)
        if mask01 is None:
            mask01 = np.zeros((H, W), dtype=np.uint8)
        mask01 = (mask01.astype(np.uint8) & depth_keep01.astype(np.uint8))

        if int(cfg.merge_close_px) > 0:
            mask01 = merge_close_components_on_mask(mask01, dilation_px=int(cfg.merge_close_px))

        return depth_keep01, masks_depth, mask01, thr, masks

    # real refinement
    target_area = float(cfg.refine_area_mult) * float(mean_area_est)
    thr = int(thr_start)

    depth_gray = cv2.cvtColor(depth_bgr, cv2.COLOR_BGR2GRAY) if depth_bgr.ndim == 3 else depth_bgr.copy()
    valid = depth_gray > 0

    best = None  # keep best-so-far if we never satisfy
    for it in range(int(cfg.refine_max_iters)):
        depth_keep01 = ((depth_gray >= thr) & valid).astype(np.uint8)

        with torch.inference_mode():
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    masks, _, _ = run_sam3_on_pil(image_pil, processor, prompt=cfg.prompt_text)
            else:
                masks, _, _ = run_sam3_on_pil(image_pil, processor, prompt=cfg.prompt_text)

        masks_depth = []
        for m in masks:
            m01 = m.astype(np.uint8)
            md = (m01 & depth_keep01).astype(np.uint8)
            if int(np.count_nonzero(md)) > 0:
                masks_depth.append(md)

        if int(cfg.merge_close_px) > 0 and len(masks_depth) > 1:
            masks255 = [(md.astype(np.uint8) * 255) for md in masks_depth]
            merged255 = merge_overlapping_masks(masks255, dilation_px=int(cfg.merge_close_px))
            masks_depth = [(m.astype(np.uint8) > 0).astype(np.uint8) for m in merged255]

        mask01 = union_mask(masks)
        if mask01 is None:
            mask01 = np.zeros((H, W), dtype=np.uint8)
        mask01 = (mask01.astype(np.uint8) & depth_keep01.astype(np.uint8))

        if int(cfg.merge_close_px) > 0:
            mask01 = merge_close_components_on_mask(mask01, dilation_px=int(cfg.merge_close_px))

        max_area, n_large = eval_mask_quality(mask01, min_area=float(cfg.refine_min_component_area))

        # keep best candidate
        if (best is None) or (max_area > best["max_area"]):
            best = {
                "depth_keep01": depth_keep01,
                "masks_depth": masks_depth,
                "mask01": mask01,
                "thr": thr,
                "max_area": max_area,
                "n_large": n_large,
                "masks": masks,
            }

        ok_area = (max_area >= target_area)
        ok_split = (n_large <= 1)  # split => 2+ large components

        if ok_area and ok_split:
            return depth_keep01, masks_depth, mask01, int(thr), masks

        thr += int(cfg.refine_thr_step)

    # fallback
    return best["depth_keep01"], best["masks_depth"], best["mask01"], int(best["thr"]), best["masks"]
    
def eval_mask_quality(mask01: np.ndarray, min_area: float) -> tuple[float, int]:
    """
    Returns:
      max_area: largest component area
      n_large: number of components with area >= min_area
    """
    areas = contour_areas(mask01)
    if not areas:
        return 0.0, 0
    max_area = max(areas)
    n_large = sum(a >= float(min_area) for a in areas)
    return float(max_area), int(n_large)


def choose_best_threshold_gmm_sweep(depth_gray: np.ndarray,
                                   k_gmm: int = 3,
                                   p_left: float = 99.0,
                                   max_vals_for_gmm: int = 200_000,
                                   downsample_stride: int = 2,
                                   min_keep: float = 0.20,
                                   keep_step: float = 0.10,
                                   open_k: int = 50,
                                   close_k: int = 100,
                                   min_area: int = 300) -> int | None:
    """
    depth_gray: uint8 HxW (0..255), where 0 can be invalid background
    Returns best threshold (int) or None if no valid pixels.
    Scoring uses denoised masks ONLY; final keep-mask uses threshold-only.
    """
    if depth_gray is None:
        return None
    if depth_gray.dtype != np.uint8:
        depth_gray = depth_gray.astype(np.uint8)

    valid_mask = depth_gray > 0
    vals_all = depth_gray[valid_mask].astype(np.float32).reshape(-1, 1)
    if vals_all.size == 0:
        return None

    # Downsample for GMM fit if huge
    vals = vals_all
    if vals_all.shape[0] > int(max_vals_for_gmm):
        vals = vals_all[::int(max(1, downsample_stride))]

    # Fit GMM
    gmm = GaussianMixture(n_components=int(k_gmm), random_state=0)
    gmm.fit(vals)

    means = gmm.means_.ravel()
    stds = np.sqrt(gmm.covariances_.ravel())

    # pick "leftmost" component
    order = np.argsort(means)
    left_orig = int(order[0])

    labels = gmm.predict(vals).ravel()
    left_vals = vals[labels == left_orig].ravel()

    if left_vals.size == 0:
        base_thr = int(np.clip(np.round(means[left_orig] + 2.0 * stds[left_orig]), 0, 255))
    else:
        base_thr = int(np.clip(np.round(np.percentile(left_vals, float(p_left))), 0, 255))

    # base keep percent (from base_thr)
    keep_mask0 = (depth_gray >= base_thr) & valid_mask
    keep_pct0 = keep_mask0.sum() / max(1, valid_mask.sum())

    start = float(np.clip(keep_pct0, float(min_keep), 1.0))

    keep_levels = []
    k = start
    while k >= float(min_keep) - 1e-9:
        keep_levels.append(round(k, 6))
        k -= float(keep_step)
    keep_levels = sorted(set(keep_levels), reverse=True)

    valid_vals = depth_gray[valid_mask].astype(np.float32)

    best_thr = None
    best_score = None  # (n_contours, total_area)

    for k in keep_levels:
        thr_k = threshold_for_keep_fraction(valid_vals, k)

        # scoring mask (denoised)
        mask_k = ((depth_gray >= thr_k) & valid_mask).astype(np.uint8)
        mask_dn = denoise_binary_mask(mask_k, open_ksize=int(open_k), close_ksize=int(close_k), min_area=int(min_area))

        n_cont, total_area = contours_and_total_area(mask_dn)
        score = (n_cont, total_area)

        if (best_score is None) or (score > best_score):
            best_score = score
            best_thr = int(thr_k)

    return best_thr


def compute_depth_keep_mask_dynamic(
    depth_bgr: np.ndarray,
    nonzero_only: bool = True,
    # --- dynamic params ---
    k_gmm: int = 3,
    p_left: float = 99.0,
    max_vals_for_gmm: int = 200_000,
    downsample_stride: int = 2,
    min_keep: float = 0.20,
    keep_step: float = 0.10,
    open_k: int = 50,
    close_k: int = 100,
    min_area: int = 300,
) -> tuple[np.ndarray, int | None]:
    """
    Returns uint8 mask01 in {0,1} using dynamic threshold selection.
    """
    if depth_bgr is None:
        raise ValueError("depth_bgr is None")

    if depth_bgr.ndim == 3:
        depth_gray = cv2.cvtColor(depth_bgr, cv2.COLOR_BGR2GRAY)
    else:
        depth_gray = depth_bgr.copy()

    if nonzero_only:
        # choose_best_threshold already treats zeros as invalid
        pass

    thr_best = choose_best_threshold_gmm_sweep(
        depth_gray,
        k_gmm=k_gmm,
        p_left=p_left,
        max_vals_for_gmm=max_vals_for_gmm,
        downsample_stride=downsample_stride,
        min_keep=float(min_keep),

        keep_step=keep_step,
        open_k=open_k,
        close_k=close_k,
        min_area=min_area,
    )

    if thr_best is None:
        return np.zeros(depth_gray.shape[:2], dtype=np.uint8)

    valid_mask = depth_gray > 0
    mask01 = ((depth_gray >= int(thr_best)) & valid_mask).astype(np.uint8)
    return mask01, thr_best

def compute_depth_keep_mask(depth_bgr: np.ndarray,
                            keep_frac: float = 0.70,
                            nonzero_only: bool = True) -> np.ndarray:
    """
    Keep the top keep_frac brightest pixels in the depth visualization frame.
    Returns uint8 mask in {0,1} with same HxW.

    If you later find your foreground is DARKER in the depth vis, flip:
      (depth_gray >= thr)  ->  (depth_gray <= thr)
    """
    if depth_bgr is None:
        raise ValueError("depth_bgr is None")

    if depth_bgr.ndim == 3:
        depth_gray = cv2.cvtColor(depth_bgr, cv2.COLOR_BGR2GRAY)
    else:
        depth_gray = depth_bgr.copy()

    keep_frac = float(np.clip(keep_frac, 0.0, 1.0))
    if keep_frac <= 0.0:
        return np.zeros(depth_gray.shape[:2], dtype=np.uint8)
    if keep_frac >= 1.0:
        return np.ones(depth_gray.shape[:2], dtype=np.uint8)

    vals = depth_gray.reshape(-1)
    if nonzero_only:
        vals = vals[vals > 0]

    # If almost everything is zero/empty, fall back to all pixels
    if vals.size < 64:
        vals = depth_gray.reshape(-1)

    # Keep top keep_frac => threshold at (1-keep_frac) percentile
    p = 100.0 * (1.0 - keep_frac)
    thr = float(np.percentile(vals, p))

    mask01 = (depth_gray >= thr).astype(np.uint8)  # 0/1
    return mask01


def ensure_same_size(mask_or_img: np.ndarray, W: int, H: int, is_mask: bool) -> np.ndarray:
    """
    Resize to (W,H) if needed. Use nearest for masks.
    """
    h, w = mask_or_img.shape[:2]
    if (w, h) == (W, H):
        return mask_or_img
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    return cv2.resize(mask_or_img, (W, H), interpolation=interp)

# ----------------------------
# 3) Helpers: Top-K CC selection + measurements
# ----------------------------
def dynamic_y_bounds_from_height(H: int, keep_center_frac: float) -> Tuple[int, int]:
    keep_center_frac = float(max(0.0, min(1.0, keep_center_frac)))
    if H <= 0:
        raise ValueError("Invalid frame height")
    margin_frac = (1.0 - keep_center_frac) / 2.0
    y_min = int(round(H * margin_frac))
    y_max = int(round(H * (1.0 - margin_frac)))
    y_min = max(0, min(y_min, max(H - 1, 0)))
    y_max = max(y_min + 1, min(y_max, max(H - 1, 1)))
    return y_min, y_max
    
def merge_overlapping_masks(comp_masks: List[np.ndarray],
                            dilation_px: int = 0) -> List[np.ndarray]:
    """
    Merge masks that overlap into a single mask.

    comp_masks: list of uint8 masks (0/255), same HxW
    dilation_px:
        0  -> merge only if masks truly overlap
        >0 -> dilate each mask before testing overlap (merges near-touching masks)

    Returns: list of merged uint8 masks (0/255), contours filled.
    """
    if not comp_masks:
        return []

    # Convert to boolean
    bools = [(m.astype(np.uint8) > 0) for m in comp_masks]

    # Optional dilation for overlap test
    if dilation_px > 0:
        k = 2 * int(dilation_px) + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        test_bools = [cv2.dilate(b.astype(np.uint8) * 255, kernel, iterations=1) > 0 for b in bools]
    else:
        test_bools = bools

    n = len(bools)

    # Union-Find (Disjoint Set) to group overlapping masks
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # Build overlap graph
    for i in range(n):
        for j in range(i + 1, n):
            if np.any(test_bools[i] & test_bools[j]):
                union(i, j)

    # Group indices by root
    groups: Dict[int, List[int]] = {}
    for i in range(n):
        r = find(i)
        groups.setdefault(r, []).append(i)

    merged_masks: List[np.ndarray] = []
    for _, idxs in groups.items():
        merged = np.zeros_like(comp_masks[0], dtype=np.uint8)
        for k in idxs:
            merged = cv2.bitwise_or(merged, comp_masks[k])

        # Fill contours to clean up union result (solid component)
        cnts, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(merged, dtype=np.uint8)
        if cnts:
            cv2.drawContours(filled, cnts, -1, 255, -1)
            merged_masks.append(filled)

    # Optional: sort merged by area desc (so your later steps are stable)
    merged_masks.sort(key=lambda m: cv2.countNonZero(m), reverse=True)
    return merged_masks
    
def merge_close_components_on_mask(mask01: np.ndarray, dilation_px: int) -> np.ndarray:
    """
    Merge components that overlap OR are within `dilation_px` pixels.

    Works by:
      - dilating the binary mask to connect close components
      - finding CCs on the dilated mask (groups)
      - for each dilated CC, pulling back the ORIGINAL pixels inside that group
        so the final mask doesn't get artificially fat

    Returns: merged mask01 (uint8 0/1)
    """
    if mask01 is None:
        return None

    m = (mask01.astype(np.uint8) > 0).astype(np.uint8)  # 0/1
    if dilation_px <= 0:
        return m

    k = 2 * int(dilation_px) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    # Dilate to connect close blobs
    dil = cv2.dilate(m, kernel, iterations=1)

    # CC on the dilated mask defines "groups"
    num, lbl, stats, _ = cv2.connectedComponentsWithStats((dil * 255).astype(np.uint8), connectivity=8)
    if num <= 1:
        return m

    merged = np.zeros_like(m, dtype=np.uint8)

    # For each dilated component, keep ORIGINAL pixels that fall inside that dilated region
    for lab in range(1, num):
        region = (lbl == lab)
        merged_region = (m > 0) & region
        merged[merged_region] = 1

    # Optional clean fill to make each merged region solid (remove holes / cracks)
    merged255 = (merged * 255).astype(np.uint8)
    cnts, _ = cv2.findContours(merged255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(merged255)
    if cnts:
        cv2.drawContours(filled, cnts, -1, 255, -1)

    return (filled > 0).astype(np.uint8)
    

def select_topk_components(mask01: np.ndarray) -> List[np.ndarray]:
    if mask01 is None:
        return []

    # 0/1 -> 0/255
    m = (mask01.astype(np.uint8) * 255)
    m = cv2.threshold(m, 10, 255, cv2.THRESH_BINARY)[1]

    num, lbl, stats, ctrs = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num <= 1:
        return []

    # Build component records with contour-based area
    comps = []  # (area_contour, filled_mask)
    for lab in range(1, num):
        px_area = float(stats[lab, cv2.CC_STAT_AREA])
        if px_area < float(cfg.min_component_area_px):
            continue

        cm = np.zeros_like(m)
        cm[lbl == lab] = 255

        cnts, _ = cv2.findContours(cm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue

        area_c = float(sum(cv2.contourArea(c) for c in cnts))
        if area_c < float(cfg.min_component_area_px):
            continue

        filled = np.zeros_like(cm)
        cv2.drawContours(filled, cnts, -1, 255, -1)

        comps.append((area_c, filled))

    if not comps:
        return []

    # Sort by contour area desc
    comps.sort(key=lambda x: x[0], reverse=True)

    # ---- relative gate vs max area ----
    amax = comps[0][0]
    rel_thr = float(cfg.rel_area_gate) * float(amax)
    comps_rel = [c for c in comps if c[0] >= rel_thr]
    if not comps_rel:
        comps_rel = [comps[0]]

    # ---- keep top-K (NO centroid distance filtering) ----
    k = int(getattr(cfg, "top_k_components", 1))
    k = max(1, k)

    kept = [c[1] for c in comps_rel[:k]]
    return kept
    
def mask_centroid_xy(mask01: np.ndarray):
    """mask01: uint8 0/1. Returns (cx, cy) or (None, None) if empty."""
    m = (mask01 > 0).astype(np.uint8)
    if m.sum() == 0:
        return None, None
    M = cv2.moments((m * 255).astype(np.uint8))
    if M["m00"] == 0:
        return None, None
    cx = float(M["m10"] / M["m00"])
    cy = float(M["m01"] / M["m00"])
    return cx, cy
    
def per_plant_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize each plant (global_id).
    """
    if df is None or df.empty:
        return pd.DataFrame()

    rows = []
    for gid, g in df.groupby("global_id"):
        g = g.sort_values("timestamp_sec")
        rows.append({
            "global_id": int(gid),
            "n_points": int(len(g)),
            "first_ts": float(g["timestamp_sec"].iloc[0]),
            "last_ts":  float(g["timestamp_sec"].iloc[-1]),
            "first_frame_index": int(g["frame_index"].iloc[0]),
            "last_frame_index":  int(g["frame_index"].iloc[-1]),
            "first_x": float(g["x"].iloc[0]),
            "first_y": float(g["y"].iloc[0]),
            "last_x":  float(g["x"].iloc[-1]),
            "last_y":  float(g["y"].iloc[-1]),
            "max_area": float(g["contour_area"].max()),
            "mean_area": float(g["contour_area"].mean()),
        })

    return (pd.DataFrame(rows)
              .sort_values("global_id")
              .reset_index(drop=True))


def write_outputs(clustered: pd.DataFrame, out_dir: Path, prefix: str = "") -> None:
    """
    Writes:
      - clustered_topk_canonical.csv
      - per_plant_summary.csv
      - (optional) xlsx if openpyxl exists
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    clustered_csv = out_dir / f"{prefix}clustered_topk_canonical.csv"
    clustered.to_csv(clustered_csv, index=False)

    summ = per_plant_summary(clustered)
    summ_csv = out_dir / f"{prefix}per_plant_summary.csv"
    summ.to_csv(summ_csv, index=False)

    # optional xlsx
    try:
        import openpyxl  # noqa: F401
        with pd.ExcelWriter(out_dir / f"{prefix}post_dbscan_outputs.xlsx", engine="openpyxl") as xl:
            clustered.to_excel(xl, index=False, sheet_name="clustered_rows")
            summ.to_excel(xl, index=False, sheet_name="per_plant_summary")
    except Exception:
        pass
# ----------------------------
# Geometry helpers
# ----------------------------
def contour_and_centroid_from_mask(mask255: np.ndarray):
    """
    mask255: uint8 mask in {0,255} or {0,1}. Returns:
      contour (Nx1x2 int32) of largest external contour
      area (float) from cv2.contourArea
      center (cx,cy) int tuple from moments (fallback to bbox center)
    """
    if mask255 is None:
        return None, 0.0, None

    m = mask255
    if m.dtype != np.uint8:
        m = m.astype(np.uint8)

    # normalize to 0/255
    if m.max() == 1:
        m = (m * 255).astype(np.uint8)
    else:
        m = (m > 0).astype(np.uint8) * 255

    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, 0.0, None

    # take largest contour by area
    cnt = max(cnts, key=cv2.contourArea)
    area = float(cv2.contourArea(cnt))
    if area <= 0:
        return None, 0.0, None

    M = cv2.moments(cnt)
    if M["m00"] != 0:
        cx = int(round(M["m10"] / M["m00"]))
        cy = int(round(M["m01"] / M["m00"]))
    else:
        x, y, w, h = cv2.boundingRect(cnt)
        cx = int(x + w / 2)
        cy = int(y + h / 2)

    return cnt, area, (cx, cy)


def radius_from_contour(contour, center_xy):
    if contour is None or center_xy is None:
        return None

    cx, cy = center_xy
    pts = contour.reshape(-1, 2).astype(np.float32)
    dx = pts[:, 0] - float(cx)
    dy = pts[:, 1] - float(cy)
    r = float(np.sqrt(dx * dx + dy * dy).max())

    R = int(math.ceil(r))
    if R < int(cfg.min_crop_radius_px):
        return None  # TOO SMALL -> skip
    return R

def crop_square_and_circle(frame_bgr: np.ndarray, center_xy, R: int):
    """
    Returns:
      square crop (BGR) of size (2R x 2R) with padding if needed
      circle crop (BGR) same size but outside-circle pixels set to black
    """
    H, W = frame_bgr.shape[:2]
    cx, cy = center_xy
    R = int(max(int(cfg.min_crop_radius_px), int(R)))

    x0, y0 = cx - R, cy - R
    x1, y1 = cx + R, cy + R

    # pad if out of bounds
    pad_l = max(0, -x0)
    pad_t = max(0, -y0)
    pad_r = max(0, x1 - W)
    pad_b = max(0, y1 - H)

    if any(p > 0 for p in (pad_l, pad_t, pad_r, pad_b)):
        padded = cv2.copyMakeBorder(frame_bgr, pad_t, pad_b, pad_l, pad_r,
                                    borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
        cx_p, cy_p = cx + pad_l, cy + pad_t
        x0, y0 = cx_p - R, cy_p - R
        x1, y1 = cx_p + R, cy_p + R
        square = padded[y0:y1, x0:x1].copy()
    else:
        square = frame_bgr[y0:y1, x0:x1].copy()

    # circle mask in the square crop
    hh, ww = square.shape[:2]
    mask = np.zeros((hh, ww), dtype=np.uint8)
    cv2.circle(mask, (ww // 2, hh // 2), R, 255, -1)

    circle = square.copy()
    circle[mask == 0] = 0

    return square, circle
def _color_from_id(k: int) -> tuple:
    """
    Deterministic bright-ish BGR color for mask index k.
    """
    rng = np.random.RandomState(12345 + int(k) * 97)
    c = rng.randint(60, 256, size=3).tolist()  # avoid too-dark
    return (int(c[0]), int(c[1]), int(c[2]))   # BGR

def colorize_masks_on_frame(frame_bgr: np.ndarray,
                            masks_list,
                            alpha: float = 0.55,
                            draw_border: bool = True,
                            border_thickness: int = 2) -> np.ndarray:
    """
    frame_bgr: HxWx3 uint8
    masks_list: list of HxW masks (bool, 0/1, or 0/255)
    returns: overlayed BGR uint8
    """
    out = frame_bgr.copy()

    H, W = frame_bgr.shape[:2]
    for k, m in enumerate(masks_list):
        if m is None:
            continue

        mm = m
        if mm.dtype != np.uint8:
            mm = mm.astype(np.uint8)

        # normalize to 0/1
        if mm.max() > 1:
            mm01 = (mm > 0).astype(np.uint8)
        else:
            mm01 = (mm > 0).astype(np.uint8)

        if mm01.sum() == 0:
            continue

        color = _color_from_id(k)  # BGR
        color_img = np.zeros((H, W, 3), dtype=np.uint8)
        color_img[mm01 > 0] = color

        out = cv2.addWeighted(out, 1.0, color_img, float(alpha), 0)

        if draw_border:
            cnts, _ = cv2.findContours((mm01 * 255).astype(np.uint8),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                cv2.drawContours(out, cnts, -1, color, int(border_thickness))

        # optional: label
        # cy, cx = np.mean(np.where(mm01 > 0), axis=1)  # simple label position (y,x)

    return out

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p
def _now():
    return time.perf_counter()

def write_timing_csv(rows, out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
# ----------------------------
# Main
# ----------------------------
def main():
    # ----------------------------
    # Args / paths
    # ----------------------------
    t_pipeline_start = _now()

    # ----------------------------
    # Stage timing accumulators (Stages 1–5)
    # ----------------------------
    stage_time = {
        "stage1_extract": 0.0,
        "stage2_depth_model": 0.0,
        "stage3_threshold_search": 0.0,
        "stage4_sam3": 0.0,
        "stage5_row_clustering": 0.0,
    }
    total_frames = 0


    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s", "--seq_dir",
        type=str,
        required=True,
        help="Parent folder containing paired videos: *_src.mp4 and *_vis.mp4 (e.g., xxx_1_src.mp4, xxx_1_vis.mp4, xxx_2_src.mp4, ...)"
    )
    parser.add_argument("--prompt", type=str, default=None, help="SAM3 text prompt (default uses cfg.prompt_text)")
    parser.add_argument("--fps", type=int, default=None, help="Frame extraction fps (default uses cfg.frames_fps)")
    parser.add_argument("--hit_k_rows", type=int, default=None, help="KMeans rows K for HitProcess (default cfg.hit_k_rows)")
    parser.add_argument("--hit_circle_r", type=int, default=None, help="Circle radius (px) for HitProcess (default cfg.hit_circle_r)")
    parser.add_argument("--hit_no_largest_rgb", action="store_true", help="Disable largest_per_plant RGB export in HitProcess")
    parser.add_argument("--start_time", type=str, default="00:00:00", help="ffmpeg start time, e.g. 00:00:23")
    parser.add_argument("--duration", type=str, default=None, help="ffmpeg duration, e.g. 00:01:00 (optional)")
    args = parser.parse_args()

    seq_dir = Path(args.seq_dir)

    # override cfg from CLI
    if args.prompt is not None:
        cfg.prompt_text = args.prompt
    if args.fps is not None:
        cfg.frames_fps = int(args.fps)

    if args.hit_k_rows is not None:
        cfg.hit_k_rows = int(args.hit_k_rows)
    if args.hit_circle_r is not None:
        cfg.hit_circle_r = int(args.hit_circle_r)
    if bool(args.hit_no_largest_rgb):
        cfg.hit_no_largest_rgb = True

    # ----------------------------
    # Resolve multiple segments (fallback to single pair)
    # ----------------------------
    try:
        video_pairs = resolve_multiple_video_pairs(seq_dir)
    except Exception:
        rgb_path, depth_path = resolve_videos_from_seq_dir(seq_dir)
        video_pairs = [(Path(rgb_path), Path(depth_path))]

    print(f"🎬 Found {len(video_pairs)} video segment(s)")

    # ----------------------------
    # Outputs (ONE shared output root for all segments)
    # ----------------------------
    segment_dir = seq_dir / f"{seq_dir.name}_segment"
    segment_dir.mkdir(parents=True, exist_ok=True)

    masks_dir = segment_dir / "masks_union"
    seg_dir = segment_dir / "segs_union"
    comp_dir = segment_dir / "components_topk"
    depthmask_dir = segment_dir / "depth_keep_masks"
    colored_dir = ensure_dir(segment_dir / "masks_colored_depth")

    for d in (masks_dir, seg_dir, comp_dir, depthmask_dir):
        d.mkdir(parents=True, exist_ok=True)

    # timing outputs
    timing_dir = segment_dir / "timing"
    timing_dir.mkdir(parents=True, exist_ok=True)
    per_frame_time_csv = timing_dir / "process_time_per_frame.csv"
    summary_time_csv = timing_dir / "process_time_summary.csv"

    per_frame_rows = []
    summary_rows = []

    # ----------------------------
    # 1) Extract frames (per segment)
    # ----------------------------
    total_extract_sec = 0.0

    # ----------------------------
    # 2) SAM3 init (once)
    # ----------------------------
    t_sam_init_start = _now()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = str(SAM3_REPO_ROOT / "checkpoints" / "sam3.pt")
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(f"SAM3 checkpoint not found: {ckpt_path}")

    model = build_sam3_image_model(checkpoint_path=ckpt_path)
    processor = Sam3Processor(model)
    model.to(device)
    model.eval()
    use_amp = (device == "cuda")
    sam_init_sec = _now() - t_sam_init_start
    print(f"⏱️ SAM3 init time: {sam_init_sec:.2f} sec")

    # ----------------------------
    # 3) Iterate segments + frames continuously
    # ----------------------------
    IMG_EXTS = {".jpg", ".jpeg", ".png"}
    all_rows = []
    frame_h = None

    # global continuity offsets
    global_frame_offset = 0
    global_time_offset = 0.0

    # aggregated step times across ALL frames
    agg_depth_keep = 0.0
    agg_sam3 = 0.0
    agg_merge_masks = 0.0
    agg_union_depth_gate = 0.0
    agg_topk = 0.0
    agg_save_outputs = 0.0
    agg_row_build = 0.0

    t_frame_loop_start = _now()

    for seg_idx, (rgb_video_path, depth_video_path) in enumerate(video_pairs):
        rgb_video_path = Path(rgb_video_path)
        depth_video_path = Path(depth_video_path)

        print(f"\n▶ Segment {seg_idx+1}/{len(video_pairs)}")
        print(f"   RGB:   {rgb_video_path.name}")
        print(f"   Depth: {depth_video_path.name}")

        # ---- Extract frames for this segment ----
        t_extract_start = _now()
        rgb_frames_dir = extract_frames_fps(
            str(rgb_video_path),
            fps=int(cfg.frames_fps),
            jpg_quality=int(cfg.jpg_quality),
            start_time=args.start_time,
            duration=args.duration
        )
        depth_frames_dir = extract_frames_fps(
            str(depth_video_path),
            fps=int(cfg.frames_fps),
            jpg_quality=int(cfg.jpg_quality),
            start_time=args.start_time,
            duration=args.duration
        )
        extract_sec = _now() - t_extract_start
        total_extract_sec += extract_sec
        stage_time["stage1_extract"] += extract_sec
        print(f"⏱️ Segment frame extraction time: {extract_sec:.2f} sec")

        # ---- Align frames ----
        rgb_image_paths = sorted([p for p in Path(rgb_frames_dir).iterdir() if p.suffix.lower() in IMG_EXTS])
        depth_image_paths = sorted([p for p in Path(depth_frames_dir).iterdir() if p.suffix.lower() in IMG_EXTS])

        if not rgb_image_paths:
            raise RuntimeError(f"No RGB frames found in {rgb_frames_dir}")
        if not depth_image_paths:
            raise RuntimeError(f"No depth frames found in {depth_frames_dir}")

        n_local = min(len(rgb_image_paths), len(depth_image_paths))
        rgb_image_paths = rgb_image_paths[:n_local]
        depth_image_paths = depth_image_paths[:n_local]
        print(f"   Aligned frames in segment: RGB={len(rgb_image_paths)} depth={len(depth_image_paths)} (n={n_local})")
        
        running_mean_area = None
        mean_area_count = 0
        # ---- Frame loop ----
        for i_local, (img_path, depth_path) in enumerate(zip(rgb_image_paths, depth_image_paths)):
            total_frames += 1

            # 1️⃣ Compute GLOBAL frame index first
            frame_index = global_frame_offset + i_local
            global_name = f"{int(frame_index):06d}"

            # 2️⃣ Compute timestamp
            timestamp_sec = global_time_offset + float(i_local) / float(cfg.frames_fps)

            # 3️⃣ Compute LOCAL name (optional)
            name = Path(img_path).stem
            local_name = name            
            t_frame_start = _now()

            frame_index = global_frame_offset + i_local
            timestamp_sec = global_time_offset + float(i_local) / float(cfg.frames_fps)

            # ----------------------------
            # Load FULL RGB + FULL depth
            # ----------------------------
            image_pil_full = Image.open(img_path).convert("RGB")
            frame_bgr_full = cv2.cvtColor(np.array(image_pil_full), cv2.COLOR_RGB2BGR)
            H_full, W_full = frame_bgr_full.shape[:2]
            if frame_h is None:
                frame_h = H_full  # keep ORIGINAL height for DBSCAN plots

            depth_bgr_full = cv2.imread(str(depth_path))
            if depth_bgr_full is None:
                raise RuntimeError(f"Failed to read depth frame: {depth_path}")
            depth_bgr_full = ensure_same_size(depth_bgr_full, W=W_full, H=H_full, is_mask=False)

            # ----------------------------
            # Apply y_keep_center_frac DURING LOAD
            # ----------------------------
            y_min, y_max = dynamic_y_bounds_from_height(H_full, cfg.y_keep_center_frac)
            y_offset = int(y_min)

            frame_bgr = crop_to_y_band(frame_bgr_full, y_min, y_max)
            depth_bgr = crop_to_y_band(depth_bgr_full, y_min, y_max)

            H, W = frame_bgr.shape[:2]  # band height
            image_pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))  # SAM3 sees band


            # ----------------------------
            # (1-4) depth keep + SAM3 (+ refinement wrapper)
            # ----------------------------
            t0 = _now()

            # initial threshold from your existing dynamic chooser
            t_stage2 = _now()
            depth_keep01_init, thr_best = compute_depth_keep_mask_dynamic(
                depth_bgr,
                nonzero_only=bool(cfg.depth_nonzero_only),
                k_gmm=int(cfg.depth_gmm_k),
                p_left=float(cfg.depth_p_left),
                max_vals_for_gmm=int(cfg.depth_max_vals_for_gmm),
                downsample_stride=int(cfg.depth_downsample_stride),
                min_keep=float(cfg.depth_min_keep),
                keep_step=float(cfg.depth_keep_step),
                open_k=int(cfg.depth_open_k),
                close_k=int(cfg.depth_close_k),
                min_area=int(cfg.depth_min_area),
            )
            stage_time["stage2_depth_model"] += _now() - t_stage2

            # convert initial mask to an initial "thr_start"
            # (compute_depth_keep_mask_dynamic currently returns mask01 only. It chooses thr internally. :contentReference[oaicite:5]{index=5})
            # So we estimate a starting thr as the minimum depth value that was kept.
            thr = thr_best
            depth_gray = cv2.cvtColor(depth_bgr, cv2.COLOR_BGR2GRAY) if depth_bgr.ndim == 3 else depth_bgr.copy()
            kept_vals = depth_gray[depth_keep01_init > 0]
            thr_start = int(np.min(kept_vals)) if kept_vals.size else 0
            
            mean_area_est = None
            if (running_mean_area is not None) and (mean_area_count >= int(cfg.refine_warmup_frames)):
                mean_area_est = float(running_mean_area)

            if bool(cfg.refine_enable):
                t_stage3 = _now()
                depth_keep01, masks_depth, mask01, thr_final, masks = refine_depth_keep_with_sam3(
                    frame_bgr=frame_bgr,
                    depth_bgr=depth_bgr,
                    image_pil=image_pil,
                    processor=processor,
                    cfg=cfg,
                    thr_start=thr_start,
                    mean_area_est=mean_area_est,
                    use_amp=use_amp,
                    timing_acc=stage_time,
                )
                stage_time["stage3_threshold_search"] += _now() - t_stage3
            else:
                depth_keep01 = depth_keep01_init
                thr_final = thr_start
                # run SAM3 once as before
                t_sam3_once = _now()
                with torch.inference_mode():
                    if use_amp:
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            masks, _, _ = run_sam3_on_pil(image_pil, processor, prompt=cfg.prompt_text)
                    else:
                        masks, _, _ = run_sam3_on_pil(image_pil, processor, prompt=cfg.prompt_text)

                stage_time["stage4_sam3"] += _now() - t_sam3_once

                masks_depth = []
                for m in masks:
                    m01 = m.astype(np.uint8)
                    md = (m01 & depth_keep01).astype(np.uint8)
                    if int(np.count_nonzero(md)) > 0:
                        masks_depth.append(md)

                if int(cfg.merge_close_px) > 0 and len(masks_depth) > 1:
                    masks255 = [(md.astype(np.uint8) * 255) for md in masks_depth]
                    merged255 = merge_overlapping_masks(masks255, dilation_px=int(cfg.merge_close_px))
                    masks_depth = [(m.astype(np.uint8) > 0).astype(np.uint8) for m in merged255]

                mask01 = union_mask(masks)
                if mask01 is None:
                    mask01 = np.zeros((H, W), dtype=np.uint8)
                mask01 = (mask01.astype(np.uint8) & depth_keep01.astype(np.uint8))

                if int(cfg.merge_close_px) > 0:
                    mask01 = merge_close_components_on_mask(mask01, dilation_px=int(cfg.merge_close_px))
             
            depth_keep255 = (depth_keep01 * 255).astype(np.uint8)
            
            areas = contour_areas(mask01)  # uses the helper you added (list of contour areas)
            max_area = float(max(areas)) if len(areas) else 0.0

            if max_area > 0:
                mean_area_count += 1
                if running_mean_area is None:
                    running_mean_area = max_area
                else:
                    alpha = 0.10  # EMA smoothing (0.05-0.2 is typical)
                    running_mean_area = (1.0 - alpha) * running_mean_area + alpha * max_area
            t1 = _now()
            depth_keep_sec = float(t1 - t0)
            agg_depth_keep += depth_keep_sec 

            # ----------------------------
            # (5) topK components
            # ----------------------------
            t0 = _now()
            comp_masks = select_topk_components(mask01)

            # Save FINAL depth_keep mask (after dynamic selection + refinement)
            out_depth_keep = depthmask_dir / f"{global_name}_depth_keep.png"
            cv2.imwrite(str(out_depth_keep), depth_keep255)

            # (Optional) Save each SAM3 mask gated by depth_keep
            for j, md in enumerate(masks_depth):
                out_md = depthmask_dir / f"{global_name}_md{j:02d}.png"

            t1 = _now()
            topk_sec = t1 - t0
            agg_topk += topk_sec

            # ----------------------------
            # (6) save outputs (band-sized)
            # ----------------------------
            t0 = _now()
            overlay = colorize_masks_on_frame(frame_bgr, masks_depth, alpha=0.55, draw_border=True)
            cv2.imwrite(str(colored_dir / f"sam_masks_depth_{global_name}.png"), overlay)

            mask255 = (mask01 * 255).astype(np.uint8)
            out_mask_path = masks_dir / f"mask_{global_name}.png"
            out_seg_path = seg_dir / f"seg_{global_name}.png"
            cv2.imwrite(str(out_mask_path), mask255)

            seg = frame_bgr.copy()
            seg[mask255 == 0] = 0
            cv2.imwrite(str(out_seg_path), seg)
            t1 = _now()
            save_sec = t1 - t0
            agg_save_outputs += save_sec

            # ----------------------------
            # build measurement rows (SAVE full-frame y using y_offset)
            # ----------------------------
            t0 = _now()
            for j, cm in enumerate(comp_masks):
                contour, area, center = contour_and_centroid_from_mask(cm)
                if contour is None or center is None:
                    continue
                cx_band, cy_band = center
                cx_full = int(cx_band)
                cy_full = int(cy_band + y_offset)

                comp_mask_path = comp_dir / f"{global_name}_c{j}.png"
                cv2.imwrite(str(comp_mask_path), cm)

                all_rows.append({
                    "frame_index": int(frame_index),
                    "timestamp_sec": float(timestamp_sec),
                    "x": int(cx_full),
                    "y": int(cy_full),
                    "contour_area": float(area),

                    "frame_path": str(img_path),
                    "union_mask_path": str(out_mask_path),
                    "comp_mask_path": str(comp_mask_path),

                    "segment_index": int(seg_idx),
                    "segment_rgb": str(rgb_video_path),
                    "segment_depth": str(depth_video_path),
                    "local_frame_index": int(i_local),

                    "y_min_band": int(y_min),
                    "y_max_band": int(y_max),
                })
            t1 = _now()
            row_build_sec = t1 - t0
            agg_row_build += row_build_sec

            # per-frame totals
            t_frame_end = _now()
            frame_total_sec = t_frame_end - t_frame_start

            per_frame_rows.append({
                "segment_index": int(seg_idx),
                "local_frame_index": int(i_local),
                "global_frame_index": int(frame_index),
                "frame_name": img_path.name,
                "timestamp_sec": float(timestamp_sec),

                "n_sam_masks": int(len(masks_depth)) if masks_depth is not None else 0,
                "n_masks_after_depth_gate": int(len(masks_depth)),
                "n_topk_components": int(len(comp_masks)),

                "t_depth_keep_sec": float(depth_keep_sec),
                "t_topk_sec": float(topk_sec),
                "t_save_outputs_sec": float(save_sec),
                "t_row_build_sec": float(row_build_sec),
                "t_frame_total_sec": float(frame_total_sec),
            })

            if (frame_index % 50) == 0:
                print(f"   [{frame_index}] ok | topK={len(comp_masks)} | {img_path.name}")

            if device == "cuda":
                torch.cuda.empty_cache()

        # advance offsets AFTER segment completes
        global_frame_offset += n_local
        global_time_offset += float(n_local) / float(cfg.frames_fps)

    frame_loop_sec = _now() - t_frame_loop_start

    # save per-frame timing
    write_timing_csv(per_frame_rows, per_frame_time_csv)
    print(f"🕒 Per-frame process time saved: {per_frame_time_csv}")

    # ----------------------------
    # Measurements CSV
    # ----------------------------
    if not all_rows:
        print("⚠️ No valid components detected. Nothing to cluster/output.")
        total_sec = _now() - t_pipeline_start
        write_timing_csv([{
            "extract_frames_sec": float(total_extract_sec),
            "sam3_init_sec": float(sam_init_sec),
            "frame_loop_sec": float(frame_loop_sec),
            "hitprocess_sec": 0.0,
            "total_pipeline_sec": float(total_sec),
        }], summary_time_csv)
        print(f"🕒 Summary process time saved: {summary_time_csv}")
        print("Done. Output folder:", segment_dir)
        return

    df = pd.DataFrame(all_rows)
    meas_csv = segment_dir / "measurements_topk.csv"
    df.to_csv(meas_csv, index=False)
    print(f"📝 Measurements saved: {meas_csv}")

    # ----------------------------
    # 
    # ----------------------------
    # (7) plant tracking WITHOUT DBSCAN: HitProcess circle-hit + KMeans rows
    # ----------------------------
    from HitProcess_circle import run_hitprocess_circle_global_frameindex
    t_hit_start = _now()
    hit_out_dir = segment_dir / "hitprocess_circle_out"
    run_hitprocess_circle_global_frameindex(
        measurements_csv=str(meas_csv),
        out_dir=str(hit_out_dir),
        k_rows=int(getattr(cfg, "hit_k_rows", 2)),
        circle_r=int(getattr(cfg, "hit_circle_r", 50)),
        export_largest_rgb=(not bool(getattr(cfg, "hit_no_largest_rgb", False))),
    )
    hit_sec = _now() - t_hit_start
    stage_time["stage5_row_clustering"] += hit_sec
    print(f"\n⏱️ HitProcess (KMeans + circle-hit + y-drop) time: {hit_sec:.2f} sec")


    # ----------------------------
    # TOTAL + summary CSV
    # ----------------------------
    total_sec = _now() - t_pipeline_start

    summary_rows.append({
        "extract_frames_sec": float(total_extract_sec),
        "sam3_init_sec": float(sam_init_sec),
        "frame_loop_sec": float(frame_loop_sec),
        "hitprocess_sec": float(hit_sec),
        "total_pipeline_sec": float(total_sec),

        "sum_depth_keep_sec": float(agg_depth_keep),
        "sum_sam3_infer_sec": float(agg_sam3),
        "sum_topk_component_sec": float(agg_topk),
        "sum_save_output_sec": float(agg_save_outputs),
        "sum_row_build_sec": float(agg_row_build),

        "num_segments": int(len(video_pairs)),
        "num_frames_total": int(global_frame_offset),
        "fps_sampled": int(cfg.frames_fps),
        "depth_keep_frac": float(cfg.depth_keep_brightness_frac),
        "top_k_components": int(cfg.top_k_components),
        "avg_frame_sec": float(
            np.mean([r["t_frame_total_sec"] for r in per_frame_rows]) if per_frame_rows else 0.0
        ),
    })

    write_timing_csv(summary_rows, summary_time_csv)
    print(f"\n⏱️ TOTAL PIPELINE TIME: {total_sec:.2f} sec")
    print(f"🕒 Summary process time saved: {summary_time_csv}")
    print("Done. Output folder:", segment_dir)



    # ----------------------------
    # Stage timing summary (Stages 1–5)
    # ----------------------------
    print("\n==============================")
    print("⏱️  PIPELINE STAGE TIMING SUMMARY")
    print("==============================")
    for k, v in stage_time.items():
        per_frame = float(v) / max(1, int(total_frames))
        print(f"{k:24s} total: {float(v):8.2f} sec | per-frame: {per_frame:7.4f} sec")
    total_stage = float(sum(stage_time.values()))
    print("--------------------------------")
    print(f"Total frames processed: {int(total_frames)}")
    print(f"Total stage time (sum): {total_stage:.2f} sec")
    print(f"Avg stage time/frame: {total_stage/max(1,int(total_frames)):.4f} sec")
    print("==============================\n")

if __name__ == "__main__":
    main()

