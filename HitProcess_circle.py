#!/usr/bin/env python3
"""
HitProcess_circle_globalframe.py

UPDATE (requested):
- HitProcess stage uses the TRUE global `frame_index` from measurements_topk.csv
  (supports multiple videos / segments without resetting frame count).
- Does NOT enumerate masks in components_topk/ to define frame_index.
- Uses each `comp_mask_path` row as one candidate mask (one contour per mask).

Core logic:
1) Read measurements_topk.csv (must include: frame_index, frame_path, comp_mask_path)
2) For each comp_mask_path:
   - read mask, get largest contour, compute cx, cy, contour_area
3) Global KMeans on cx -> row_id (fixed centers across whole run)
4) Per-frame row center_x = median cx per (frame_index,row_id) (fallback to global center)
   center_y is fixed = H//2 (from mask height)
5) Circle-hit: contour intersects circle centered at (center_x(frame,row), center_y_fixed), radius = --circle_r
6) Keep best per (frame_index,row_id) (largest area, tie-break closest to center)
7) y-drop split to global_id (per row_id)
8) Outputs:
   - out_dir/hit_center_with_xy_area.csv
   - out_dir/kmeans_time_vs_x_k{K}_circleR{R}.png
   - out_dir/components_topk_with_two_centers/  (circles drawn on every comp mask)
   - out_dir/hit_masks/  (selected masks only)
   - out_dir/largest_per_plant/  (optional RGB export)

Usage:
  python HitProcess_circle_globalframe.py --in_csv /path/to/measurements_topk.csv --out_dir /path/to/out --k_rows 2 --circle_r 30
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# ----------------------------
# Defaults
# ----------------------------
min_area = 30
#abnormal_mult = 2
DIST_STD_MULT = 4 # ground 4 aerial 4 lantana 3

DROP_THRESHOLD = 40
MIN_EVENT_SEP_FRAMES = 2


# ----------------------------
# Geometry helpers
# ----------------------------
def contour_centroid(cnt):
    M = cv2.moments(cnt)
    if M["m00"] <= 1e-9:
        return None
    return float(M["m10"] / M["m00"]), float(M["m01"] / M["m00"])


def contour_hits_circle(cnt, cx0: float, cy0: float, R: float) -> bool:
    if cnt is None or len(cnt) == 0:
        return False
    pts = cnt[:, 0, :].astype(np.float32)
    dx = pts[:, 0] - float(cx0)
    dy = pts[:, 1] - float(cy0)
    return bool(np.any((dx * dx + dy * dy) <= (float(R) * float(R))))


def mask_largest_contour_stats(mask_path: str):
    """
    Read a binary mask (0/255 or 0/1), return (H,W,cnt,cx,cy,area) for largest contour.
    Returns None if mask missing or empty.
    """
    m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    if m.dtype != np.uint8:
        m = m.astype(np.uint8)
    if m.max() == 1:
        m255 = (m * 255).astype(np.uint8)
    else:
        m255 = (m > 0).astype(np.uint8) * 255

    cnts, _ = cv2.findContours(m255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    area = float(cv2.contourArea(cnt))
    cxy = contour_centroid(cnt)
    if cxy is None:
        return None
    cx, cy = cxy
    H, W = m255.shape[:2]
    return H, W, cnt, cx, cy, area


# ----------------------------
# y-drop global_id
# ----------------------------
def assign_global_id_by_y_drop(
    df: pd.DataFrame,
    time_col="frame_index",
    y_col="cy",
    drop_threshold=50,
    min_event_sep_frames=3,
) -> pd.DataFrame:
    df = df.sort_values(["row_id", time_col]).reset_index(drop=True)
    df["global_id"] = -1
    df["y_drop_event"] = False

    for rid in sorted(df["row_id"].unique()):
        d = df[df["row_id"] == rid].copy()
        idxs = d.index.to_numpy()

        t = d[time_col].to_numpy()
        y = d[y_col].to_numpy()

        drop = np.zeros_like(y, dtype=bool)
        for i in range(1, len(y)):
            if (y[i] - y[i - 1]) < -drop_threshold:
                drop[i] = True

        event = np.zeros_like(drop, dtype=bool)
        last_evt = None
        for i in range(len(drop)):
            if not drop[i]:
                continue
            if last_evt is None or (t[i] - last_evt) >= min_event_sep_frames:
                event[i] = True
                last_evt = t[i]

        event_times = t[event]
        if len(event_times) == 0:
            gids = np.zeros_like(t, dtype=int)
        else:
            gids = np.searchsorted(event_times, t, side="right")

        df.loc[idxs, "global_id"] = gids
        df.loc[idxs, "y_drop_event"] = event

    return df


# ----------------------------
# Visualization + largest-per-plant
# ----------------------------
def save_masks_with_center_circles(
    df_all: pd.DataFrame,
    frame_row_center_x: dict[tuple[int, int], float],
    global_centers_x: list[float],
    out_dir: Path,
    *,
    k_rows: int,
    center_y_fixed: int,
    circle_r_px: int,
):
    vis_dir = out_dir / "components_topk_with_two_centers"
    vis_dir.mkdir(parents=True, exist_ok=True)

    for _, r in df_all.iterrows():
        mp = str(r["comp_mask_path"])
        frame_idx = int(r["frame_index"])

        img = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        for rid in range(k_rows):
            cx = frame_row_center_x.get((frame_idx, rid), float(global_centers_x[rid]))
            cy = int(center_y_fixed)
            cx_i = int(round(cx))
            cy_i = int(round(cy))
            cv2.circle(vis, (cx_i, cy_i), int(circle_r_px), (0, 0, 255), 2)
            cv2.circle(vis, (cx_i, cy_i), 3, (0, 0, 255), -1)

        out_path = vis_dir / Path(mp).name
        cv2.imwrite(str(out_path), vis)


def _build_time_order_map_pair(df: pd.DataFrame) -> dict[tuple[int, int], int]:
    d = df.copy()
    d["row_id"] = d["row_id"].astype(int)
    d["global_id"] = d["global_id"].astype(int)
    d["frame_index"] = d["frame_index"].astype(int)
    d["cx"] = d["cx"].astype(float)

    firsts = (
        d.sort_values(["frame_index", "cx"])
         .groupby(["row_id", "global_id"], as_index=False)
         .first()[["row_id", "global_id", "frame_index", "cx"]]
         .sort_values(["frame_index", "cx"])
         .reset_index(drop=True)
    )
    return {(int(r.row_id), int(r.global_id)): int(i + 1) for i, r in firsts.iterrows()}


def _crop_square_and_circle_bgr(frame_bgr: np.ndarray, center_xy: tuple[int, int], R: int):
    H, W = frame_bgr.shape[:2]
    cx, cy = center_xy
    R = max(5, int(R))

    x1 = max(0, cx - R)
    x2 = min(W, cx + R)
    y1 = max(0, cy - R)
    y2 = min(H, cy + R)

    square = frame_bgr[y1:y2, x1:x2].copy()

    circle = np.zeros_like(square)
    hh, ww = circle.shape[:2]
    if hh == 0 or ww == 0:
        return square, circle

    circle_mask = np.zeros((hh, ww), dtype=np.uint8)
    cv2.circle(circle_mask, (ww // 2, hh // 2), min(ww, hh) // 2, 255, -1)
    circle[circle_mask > 0] = square[circle_mask > 0]
    return square, circle


def _output_largest_per_pair_rgb(df: pd.DataFrame, out_dir: Path) -> None:
    largest_dir = out_dir / "largest_per_plant"
    largest_dir.mkdir(parents=True, exist_ok=True)
    all_circle_dir = largest_dir / "all_maxarea_circle"
    all_circle_dir.mkdir(parents=True, exist_ok=True)

    if df.empty:
        print("⚠️ df empty, skip largest-per-plant.")
        return

    pair_to_timeid = _build_time_order_map_pair(df)
    pairs_sorted = sorted(pair_to_timeid.keys(), key=lambda k: pair_to_timeid[k])

    summaries = []

    for (rid, gid) in pairs_sorted:
        sub = df[(df["row_id"] == rid) & (df["global_id"] == gid)].copy()
        if sub.empty:
            continue

        r = sub.sort_values(["contour_area"], ascending=False).iloc[0]

        frame_path = Path(str(r["frame_path"]))
        mask_path = Path(str(r["comp_mask_path"]))

        frame_bgr = cv2.imread(str(frame_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if frame_bgr is None or mask is None:
            continue

        bw = (mask.astype(np.uint8) * 255) if mask.max() == 1 else ((mask > 0).astype(np.uint8) * 255)

        Hf, Wf = frame_bgr.shape[:2]
        Hm, Wm = bw.shape[:2]
        if (Hm != Hf) or (Wm != Wf):
            bw = cv2.resize(bw, (Wf, Hf), interpolation=cv2.INTER_NEAREST)

        cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)

        cxy = contour_centroid(cnt)
        if cxy is None:
            continue
        cx, cy = int(round(cxy[0])), int(round(cxy[1]))

        area = float(cv2.contourArea(cnt))
        (_, _), Rf = cv2.minEnclosingCircle(cnt)
        R = int(round(Rf))

        overlay = frame_bgr.copy()
        cv2.drawContours(overlay, [cnt], -1, (0, 0, 255), 3)
        cv2.circle(overlay, (cx, cy), 8, (0, 0, 255), -1)

        time_id = int(pair_to_timeid[(int(rid), int(gid))])
        cv2.putText(
            overlay,
            f"TimeID {time_id:03d} | row {int(rid)} | gid {int(gid)} | area={area:.0f}",
            (max(5, cx - 260), max(30, cy - 30)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        square, circle = _crop_square_and_circle_bgr(frame_bgr, (cx, cy), R)

        plant_dir = largest_dir / f"plant_{time_id:03d}_row{int(rid):02d}_gid{int(gid):03d}"
        plant_dir.mkdir(parents=True, exist_ok=True)

        out_full = plant_dir / f"plant_{time_id:03d}_row{int(rid):02d}_gid{int(gid):03d}_maxarea_full.png"
        out_sq   = plant_dir / f"plant_{time_id:03d}_row{int(rid):02d}_gid{int(gid):03d}_maxarea_squareR{int(R)}.png"
        out_circ = plant_dir / f"plant_{time_id:03d}_row{int(rid):02d}_gid{int(gid):03d}_maxarea_circleR{int(R)}.png"
        out_circ_all = all_circle_dir / f"plant_{time_id:03d}_row{int(rid):02d}_gid{int(gid):03d}_maxarea_circleR{int(R)}.png"

        cv2.imwrite(str(out_full), overlay)
        cv2.imwrite(str(out_sq), square)
        cv2.imwrite(str(out_circ), circle)
        cv2.imwrite(str(out_circ_all), circle)

        summaries.append({
            "time_order_id": int(time_id),
            "row_id": int(rid),
            "global_id": int(gid),
            "max_area": float(area),
            "frame_index": int(r["frame_index"]),
            "x": int(cx),
            "y": int(cy),
            "R_px": int(R),
            "full_path": str(out_full),
            "square_path": str(out_sq),
            "circle_path": str(out_circ),
            "source_frame_path": str(frame_path),
            "source_mask_path": str(mask_path),
        })

    if summaries:
        pd.DataFrame(summaries).sort_values(["time_order_id"]).to_csv(
            largest_dir / "largest_per_plant_summary.csv", index=False
        )


# ----------------------------
# Main runner
# ----------------------------
def run_hitprocess_circle_global_frameindex(
    measurements_csv: str,
    out_dir: str,
    *,
    k_rows: int = 2,
    circle_r: int = 50,
    export_largest_rgb: bool = True,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meas = pd.read_csv(measurements_csv)
    required = ["frame_index", "frame_path", "comp_mask_path"]
    for c in required:
        if c not in meas.columns:
            raise ValueError(f"measurements CSV missing required column: {c}")

    base = meas[required].copy()
    base["frame_index"] = base["frame_index"].astype(int)
    base["frame_path"] = base["frame_path"].astype(str)
    base["comp_mask_path"] = base["comp_mask_path"].astype(str)

    exists = base["comp_mask_path"].map(lambda p: Path(p).exists())
    base = base[exists].copy().reset_index(drop=True)
    if base.empty:
        raise RuntimeError("No comp_mask_path files exist on disk. Check paths in CSV.")

    rows = []
    H0 = None
    for _, r in base.iterrows():
        st = mask_largest_contour_stats(r["comp_mask_path"])
        if st is None:
            continue
        H, W, cnt, cx, cy, area = st
        if H0 is None:
            H0 = H
        rows.append({
            "frame_index": int(r["frame_index"]),
            "frame_path": r["frame_path"],
            "comp_mask_path": r["comp_mask_path"],
            "cx": float(cx),
            "cy": float(cy),
            "contour_area": float(area),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("All masks were empty (no contours).")

    center_y_fixed = int((H0 if H0 is not None else 0) // 2)

    df = df[df["contour_area"] >= float(min_area)].copy()
    #mean_area = float(df["contour_area"].mean())
    #abnormal_thr = float(abnormal_mult) * mean_area
    #df = df[df["contour_area"] <= abnormal_thr].copy()
    if df.empty:
        raise RuntimeError("No contours left after area filtering.")

    X = df[["cx"]].to_numpy(float)
    km = KMeans(n_clusters=int(k_rows), random_state=0, n_init=20)
    labels = km.fit_predict(X)

    centers = km.cluster_centers_.reshape(-1)
    order = np.argsort(centers)
    centers_sorted = centers[order]
    remap = {int(old): int(new) for new, old in enumerate(order)}

    df["row_id_raw"] = labels
    df["row_id"] = [remap[int(l)] for l in labels]
    df["cluster_center_x"] = df["row_id"].map({i: float(centers_sorted[i]) for i in range(int(k_rows))})
    df["dist_to_center"] = np.abs(df["cx"] - df["cluster_center_x"])

    keep_idx = []
    for rid in range(int(k_rows)):
        d = df[df["row_id"] == rid]
        std = float(d["dist_to_center"].std(ddof=0)) if len(d) > 1 else 0.0
        thr = float(DIST_STD_MULT) * std
        if std == 0.0:
            keep = d.index
        else:
            keep = d[d["dist_to_center"] <= thr].index
        keep_idx.extend(list(keep))
    df = df.loc[keep_idx].copy().reset_index(drop=True)

    frame_row_center_x = df.groupby(["frame_index", "row_id"])["cx"].median().to_dict()

    hit_flags = []
    for _, r in df.iterrows():
        st = mask_largest_contour_stats(r["comp_mask_path"])
        if st is None:
            hit_flags.append(False)
            continue
        H, W, cnt, cx, cy, area = st
        rid = int(r["row_id"])
        fidx = int(r["frame_index"])
        cx0 = float(frame_row_center_x.get((fidx, rid), float(centers_sorted[rid])))
        cy0 = float(center_y_fixed)
        hit_flags.append(contour_hits_circle(cnt, cx0, cy0, float(circle_r)))
    df["hits_circle"] = hit_flags

    before = len(df)
    df = df[df["hits_circle"]].copy()
    print(f"[HIT-CIRCLE] kept {len(df)}/{before}")
    if df.empty:
        raise RuntimeError("No contours hit the circle.")

    df = (
        df.sort_values(["frame_index", "row_id", "contour_area", "dist_to_center"],
                       ascending=[True, True, False, True])
          .drop_duplicates(["frame_index", "row_id"], keep="first")
          .reset_index(drop=True)
    )

    hit_dir = out_dir / "hit_masks"
    hit_dir.mkdir(parents=True, exist_ok=True)
    for mp in df["comp_mask_path"].unique():
        mp = Path(str(mp))
        if mp.exists():
            shutil.copy2(str(mp), str(hit_dir / mp.name))
    df["saved_path"] = df["comp_mask_path"].map(lambda p: str(hit_dir / Path(str(p)).name))

    df = assign_global_id_by_y_drop(df, drop_threshold=DROP_THRESHOLD, min_event_sep_frames=MIN_EVENT_SEP_FRAMES)

    out_csv = out_dir / "hit_center_with_xy_area.csv"
    df.to_csv(out_csv, index=False)
    print(f"[DONE] CSV: {out_csv}")

    plt.figure(figsize=(9, 6))
    for rid in range(int(k_rows)):
        d = df[df["row_id"] == rid]
        plt.scatter(d["cx"], d["frame_index"], s=10, alpha=0.7, label=f"row {rid}")
    for cx0 in centers_sorted:
        plt.axvline(float(cx0), linestyle="--", linewidth=1)
    plt.xlabel("x (cx)")
    plt.ylabel("frame_index (GLOBAL)")
    plt.title(f"KMeans centers + circle-hit R={circle_r}px")
    plt.legend()
    plt.tight_layout()
    png = out_dir / f"kmeans_time_vs_x_k{int(k_rows)}_circleR{int(circle_r)}.png"
    plt.savefig(png, dpi=200)
    plt.close()
    print(f"[PLOT] {png}")

    save_masks_with_center_circles(
        df_all=base,
        frame_row_center_x=frame_row_center_x,
        global_centers_x=list(centers_sorted),
        out_dir=out_dir,
        k_rows=int(k_rows),
        center_y_fixed=center_y_fixed,
        circle_r_px=int(circle_r),
    )

    if export_largest_rgb:
        _output_largest_per_pair_rgb(df, out_dir=out_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True, type=str)
    ap.add_argument("--out_dir", required=True, type=str)
    ap.add_argument("--k_rows", type=int, default=2)
    ap.add_argument("--circle_r", type=int, default=50)
    ap.add_argument("--no_largest_rgb", action="store_true")
    args = ap.parse_args()

    run_hitprocess_circle_global_frameindex(
        measurements_csv=args.in_csv,
        out_dir=args.out_dir,
        k_rows=args.k_rows,
        circle_r=args.circle_r,
        export_largest_rgb=(not args.no_largest_rgb),
    )


if __name__ == "__main__":
    main()
