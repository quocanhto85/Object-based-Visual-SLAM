#!/usr/bin/env python3
import numpy as np, sys, argparse

def q_to_R(qx,qy,qz,qw):
    # normalize and convert quaternion (x,y,z,w) → R (world↔camera)
    q = np.array([qx,qy,qz,qw], dtype=float)
    q /= np.linalg.norm(q) + 1e-12
    x,y,z,w = q
    R = np.array([
        [1-2*(y*y+z*z), 2*(x*y - z*w),   2*(x*z + y*w)],
        [2*(x*y + z*w),   1-2*(x*x+z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),   2*(y*z + x*w), 1-2*(x*x+y*y)]
    ], dtype=float)
    return R

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="KeyFrameTrajectory.txt (TUM)")
    ap.add_argument("--out_twc", default="kf_traj_Twc_kitti.txt",
                    help="output KITTI [R|t] storing Twc (camera in world)")
    ap.add_argument("--out_tcw", default="kf_traj_Tcw_kitti.txt",
                    help="output KITTI [R|t] storing Tcw (world in camera) — use this for projection")
    args = ap.parse_args()

    twc_lines, tcw_lines = [], []
    with open(args.input, "r") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln[0] == '#':  # skip comments/blank
                continue
            toks = ln.split()
            if len(toks) != 8:
                # ignore non-TUM lines
                continue
            t, tx, ty, tz, qx, qy, qz, qw = map(float, toks)
            Rwc = q_to_R(qx,qy,qz,qw)
            twc = np.array([tx,ty,tz], dtype=float).reshape(3,1)

            # KITTI [R|t] as 3x4
            Twc = np.hstack([Rwc, twc])             # camera pose in world
            Rcw = Rwc.T
            tcw = -Rcw @ twc
            Tcw = np.hstack([Rcw, tcw])             # world->camera (for projection)

            twc_lines.append(" ".join(f"{v:.12f}" for v in Twc.reshape(-1)))
            tcw_lines.append(" ".join(f"{v:.12f}" for v in Tcw.reshape(-1)))

    with open(args.out_twc, "w") as f: f.write("\n".join(twc_lines) + "\n")
    with open(args.out_tcw, "w") as f: f.write("\n".join(tcw_lines) + "\n")

if __name__ == "__main__":
    main()
