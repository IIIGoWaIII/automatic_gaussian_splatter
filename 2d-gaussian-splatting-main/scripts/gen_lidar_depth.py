# Generates per-view sparse depth maps from a fused LiDAR point cloud
# (e.g. the device cloud shipped inside a SplatKing capture) by projecting
# the cloud into each training view. Output .npz files are consumed by
# train.py --use_lidar_depth for depth-supervised 2DGS training.

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scene.colmap_loader import (
    read_extrinsics_binary,
    read_intrinsics_binary,
    read_points3D_binary,
    qvec2rotmat,
)


def compute_train_resolution(orig_w, orig_h, r):
    """Mirror utils/camera_utils.loadCam resolution logic exactly."""
    if r in (1, 2, 4, 8):
        return round(orig_w / r), round(orig_h / r)
    if r == -1:
        global_down = orig_w / 1600.0 if orig_w > 1600 else 1.0
    else:
        # any other value means exact target width in loadCam
        global_down = orig_w / float(r)
    return int(orig_w / global_down), int(orig_h / global_down)


_TURBO_STOPS = np.array([
    [48, 18, 59], [70, 107, 227], [40, 187, 235], [32, 229, 172],
    [129, 241, 88], [222, 204, 46], [252, 143, 42], [230, 63, 13], [122, 4, 3],
], dtype=np.float32) / 255.0


def colorize(depth_norm):
    """Map values in [0, 1] to RGB using a turbo-like colormap. Input shape (H, W)."""
    x = np.clip(depth_norm, 0.0, 1.0) * (len(_TURBO_STOPS) - 1)
    lo = np.floor(x).astype(int)
    hi = np.minimum(lo + 1, len(_TURBO_STOPS) - 1)
    frac = (x - lo)[..., None]
    return (_TURBO_STOPS[lo] * (1 - frac) + _TURBO_STOPS[hi] * frac)


def save_overlay(image_path, depth, mask, out_path):
    img = Image.open(image_path).convert("L").resize((depth.shape[1], depth.shape[0]))
    gray = np.asarray(img, dtype=np.float32)[..., None].repeat(3, axis=2) / 255.0

    valid = depth[mask]
    if valid.size == 0:
        return
    dmin, dmax = np.percentile(valid, 1), np.percentile(valid, 99)
    norm = np.clip((depth - dmin) / max(dmax - dmin, 1e-6), 0.0, 1.0)

    overlay = gray.copy()
    overlay[mask] = 0.45 * gray[mask] + 0.55 * colorize(norm[mask])
    Image.fromarray((overlay * 255).astype(np.uint8)).save(out_path, quality=90)


def project_points(xyz, R, T, fx, fy, cx, cy):
    """World -> pixel projection for pinhole cameras. Returns (u, v, z)."""
    cam = xyz @ R.T + T
    z = cam[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = fx * cam[:, 0] / z + cx
        v = fy * cam[:, 1] / z + cy
    return u, v, z


def build_view_depth(u, v, z, tw, th, scale_u, scale_v, rel_thresh, far_pct):
    """Splat projected points into a train-resolution depth grid and reject outliers."""
    px = np.floor(u * scale_u).astype(np.int64)
    py = np.floor(v * scale_v).astype(np.int64)
    ok = np.isfinite(z) & (z > 1e-6)
    ok &= (px >= 0) & (px < tw) & (py >= 0) & (py < th)
    if not ok.any():
        return None, None, 0, 0

    px, py, pz = px[ok], py[ok], z[ok]

    # Robust far-plane cut per view (stray returns pointing past the scene)
    far = np.percentile(pz, far_pct)
    ok_d = pz <= far
    px, py, pz = px[ok_d], py[ok_d], pz[ok_d]
    if pz.size == 0:
        return None, None, 0, 0

    # Closest surface wins when several points fall into one cell
    grid = np.full((th, tw), np.inf, dtype=np.float32)
    np.minimum.at(grid, (py, px), pz.astype(np.float32))
    occupied = np.isfinite(grid)
    total = int(occupied.sum())

    # Local plane consistency: drop cells deviating from their neighborhood
    oi, oj = np.nonzero(occupied)
    d0 = grid[oi, oj]
    samples = []
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            samples.append(grid[np.clip(oi + dy, 0, th - 1), np.clip(oj + dx, 0, tw - 1)])
    stacked = np.where(np.isinf(np.stack(samples)), np.nan, np.stack(samples))
    med = np.nanmedian(stacked, axis=0)
    keep = np.abs(d0 - med) <= rel_thresh * np.maximum(med, 1e-6)

    out = np.zeros((th, tw), dtype=np.float16)
    mask = np.zeros((th, tw), dtype=bool)
    fi, fj = oi[keep], oj[keep]
    out[fi, fj] = grid[fi, fj]
    mask[fi, fj] = True
    return out, mask, int(mask.sum()), total


def main():
    parser = argparse.ArgumentParser(description="Project fused LiDAR cloud into training views.")
    parser.add_argument("--sparse_dir", required=True, help="COLMAP model dir (cameras.bin/images.bin)")
    parser.add_argument("--points_bin", required=True, help="Fused LiDAR cloud (points3D.bin)")
    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--train_res", type=int, default=1,
                        help="The same -r value passed to train.py")
    parser.add_argument("--error_pct", type=float, default=95.0,
                        help="Drop points above this percentile of reprojection error")
    parser.add_argument("--rel_thresh", type=float, default=0.15,
                        help="Relative deviation vs neighborhood median allowed per cell")
    parser.add_argument("--far_pct", type=float, default=99.5,
                        help="Per-view percentile used as robust far-plane cut")
    parser.add_argument("--min_points", type=int, default=200,
                        help="Views with fewer surviving points get no depth map")
    parser.add_argument("--save_overlays", type=int, default=1)
    args = parser.parse_args()

    sparse_dir = Path(args.sparse_dir)
    images_dir = Path(args.images_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    intrinsics = read_intrinsics_binary(sparse_dir / "cameras.bin")
    extrinsics = read_extrinsics_binary(sparse_dir / "images.bin")

    print(f"Reading LiDAR cloud: {args.points_bin}")
    xyz, _, errors = read_points3D_binary(args.points_bin)
    err_cut = np.percentile(errors, args.error_pct)
    xyz = xyz[(errors[:, 0] <= err_cut)]
    print(f"Cloud: {xyz.shape[0]} points (dropped above {args.error_pct:.0f}th pct track error)")

    saved, skipped = 0, 0
    for key in sorted(extrinsics.keys(), key=lambda k: extrinsics[k].name):
        extr = extrinsics[key]
        intr = intrinsics[extr.camera_id]

        if intr.model == "PINHOLE":
            fx, fy, cx, cy = intr.params[:4]
        elif intr.model == "SIMPLE_PINHOLE":
            fx = fy = intr.params[0]
            cx, cy = intr.params[1:3]
        else:
            raise SystemExit(f"Unsupported camera model {intr.model}; only undistorted PINHOLE models are supported.")

        stem = Path(extr.name).stem
        image_path = images_dir / Path(extr.name).name
        if not image_path.exists():
            print(f"[skip] {stem}: source image not found")
            skipped += 1
            continue

        with Image.open(image_path) as im:
            orig_w, orig_h = im.size
        tw, th = compute_train_resolution(orig_w, orig_h, args.train_res)

        R = qvec2rotmat(extr.qvec)
        T = np.asarray(extr.tvec)
        u, v, z = project_points(xyz, R, T, float(fx), float(fy), float(cx), float(cy))
        depth, mask, kept, total = build_view_depth(
            u, v, z, tw, th, tw / orig_w, th / orig_h, args.rel_thresh, args.far_pct
        )

        if depth is None or kept < args.min_points:
            print(f"[skip] {stem}: only {kept}/{total} points survived filtering (< {args.min_points})")
            skipped += 1
            continue

        meta = {
            "orig": [orig_w, orig_h],
            "train": [tw, th],
            "r": args.train_res,
            "points": kept,
        }
        np.savez_compressed(
            output_dir / f"{stem}.npz",
            depth=depth, mask=mask, meta=np.array(json.dumps(meta)),
        )
        if args.save_overlays:
            try:
                save_overlay(image_path, depth.astype(np.float32), mask,
                             output_dir / f"{stem}_overlay.jpg")
            except OSError as e:
                print(f"[warn] overlay failed for {stem}: {e}")

        print(f"[ok] {stem}: {kept}/{total} pts @ {tw}x{th}")
        saved += 1

    print(f"Done. Saved {saved} depth maps to {output_dir} ({skipped} skipped).")
    if saved == 0:
        raise SystemExit("No depth maps were generated.")


if __name__ == "__main__":
    main()
