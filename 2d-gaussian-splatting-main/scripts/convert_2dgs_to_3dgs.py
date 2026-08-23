# Converts a 2D Gaussian Splatting PLY (flat disks: 2 scales) into a standard
# 3DGS-compatible PLY (ellipsoids: 3 scales) so the result can be opened in
# Brush, LichtFeld-Studio, SuperSplat and other 3DGS viewers.
#
# The disk orientation quaternion is kept; the missing axis gets a small
# thickness (1% of the shorter disk axis) so downstream renderers never see a
# degenerate covariance.

import argparse
import os

import numpy as np
from plyfile import PlyData, PlyElement


def convert(src_path, dst_path):
    ply = PlyData.read(src_path)
    vertex = ply["vertex"]
    names = [p.name for p in vertex.properties]

    scale_names = sorted([n for n in names if n.startswith("scale_")], key=lambda s: int(s.split("_")[-1]))
    rot_names = sorted([n for n in names if n.startswith("rot_")], key=lambda s: int(s[-1]))
    if len(scale_names) == 3 and len(rot_names) == 4:
        print(f"{src_path} is already 3DGS-compatible; normalizing quaternions only.")

    arrays = {n: np.asarray(vertex[n]) for n in names}
    n_points = len(arrays["x"])

    # normalize orientation quaternions (raw optimizer state may be unnormalized)
    q = np.stack([arrays[r] for r in rot_names], axis=1)
    q /= np.clip(np.linalg.norm(q, axis=1, keepdims=True), 1e-12, None)
    for i, r in enumerate(rot_names):
        arrays[r] = q[:, i]

    if len(scale_names) == 2:
        # scales are stored in log space; give the disk a z-thickness of 1%
        # of its shorter axis:  log(t) = min(log_s0, log_s1) + log(0.01)
        thick = np.minimum(arrays[scale_names[0]], arrays[scale_names[1]]) + np.log(0.01)
        arrays["scale_2"] = thick.astype(np.float32)
        scale_names.append("scale_2")

    attr_order = ["x", "y", "z", "nx", "ny", "nz"]
    attr_order += [f"f_dc_{i}" for i in range(sum(1 for n in names if n.startswith("f_dc_")))]
    attr_order += sorted([n for n in names if n.startswith("f_rest_")], key=lambda s: int(s.split("_")[-1]))
    attr_order += ["opacity"]
    attr_order += sorted(scale_names, key=lambda s: int(s.split("_")[-1]))
    attr_order += rot_names

    dtype = [(a, "f4") for a in attr_order]
    elements = np.empty(n_points, dtype=dtype)
    for a in attr_order:
        elements[a] = arrays[a]

    el = PlyElement.describe(elements, "vertex")
    os.makedirs(os.path.dirname(dst_path) or ".", exist_ok=True)
    PlyData([el]).write(dst_path)
    print(f"Wrote {dst_path} ({n_points} splats, {len(scale_names)} scales)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert 2DGS PLY to 3DGS-compatible PLY.")
    parser.add_argument("input")
    parser.add_argument("output")
    args = parser.parse_args()
    convert(args.input, args.output)
