#!/usr/bin/env python3
"""
Enhanced Fix for EVO pose alignment mismatch error - Handles ALL cases.

This version handles:
1. GT > KF (more ground truth poses than keyframes)
2. KF > GT (more keyframes than ground truth) 
3. GT ≈ KF (roughly equal)

The issue: ORB-SLAM3 outputs keyframe-only trajectories which may be sparse or dense
compared to ground truth. The Umeyama algorithm requires equal-sized matrices.
"""

import numpy as np
import os
from scipy.spatial.transform import Rotation
from scipy.interpolate import interp1d
import sys


def kitti_to_poses(poses_array):
    """Convert KITTI format (3x4 matrix flattened) to pose objects."""
    num_poses = poses_array.shape[0]
    poses = []
    for i in range(num_poses):
        pose_row = poses_array[i]
        # Reshape 12 values to 3x4 matrix
        pose_matrix = pose_row.reshape(3, 4)
        poses.append(pose_matrix)
    return poses


def extract_translation(pose_matrix):
    """Extract translation vector from 3x4 pose matrix."""
    return pose_matrix[:, 3]


def extract_rotation_matrix(pose_matrix):
    """Extract rotation matrix from 3x4 pose matrix."""
    return pose_matrix[:, :3]


def poses_to_kitti(poses):
    """Convert pose matrices back to KITTI format."""
    kitti_poses = []
    for pose in poses:
        if pose.shape == (4, 4):
            pose = pose[:3, :]
        pose_flat = pose.flatten()
        kitti_poses.append(pose_flat)
    return np.array(kitti_poses)


# ============================================================================
# UTILITY: POSE INTERPOLATION
# ============================================================================

def interpolate_pose_slerp(pose1, pose2, t):
    """
    SLERP interpolation between two poses (3x4 matrices).
    
    Args:
        pose1, pose2: 3x4 pose matrices
        t: interpolation factor (0 to 1)
    """
    # Ensure proper shape
    if pose1.shape == (4, 4):
        pose1 = pose1[:3, :]
    if pose2.shape == (4, 4):
        pose2 = pose2[:3, :]
    
    # Extract components
    R1 = pose1[:, :3]
    t1 = pose1[:, 3]
    R2 = pose2[:, :3]
    t2 = pose2[:, 3]
    
    # Interpolate translation linearly
    t_interp = (1 - t) * t1 + t * t2
    
    # Convert rotation matrices to quaternions and SLERP
    rot1 = Rotation.from_matrix(R1)
    rot2 = Rotation.from_matrix(R2)
    
    # Proper SLERP using scipy
    key_times = [0, 1]
    key_rots = Rotation.from_quat([rot1.as_quat(), rot2.as_quat()])
    slerp = Rotation.from_quat(key_rots.as_quat())
    
    # Interpolate
    quat1 = rot1.as_quat()
    quat2 = rot2.as_quat()
    
    # Ensure shortest path
    if np.dot(quat1, quat2) < 0:
        quat2 = -quat2
    
    quat_interp = (1 - t) * quat1 + t * quat2
    quat_interp = quat_interp / np.linalg.norm(quat_interp)
    
    rot_interp = Rotation.from_quat(quat_interp)
    
    # Reconstruct pose
    pose_interp = np.eye(4)
    pose_interp[:3, :3] = rot_interp.as_matrix()
    pose_interp[:3, 3] = t_interp
    
    return pose_interp[:3, :]


# ============================================================================
# SOLUTION 1A: DOWNSAMPLE DENSER TRAJECTORY (UNIVERSAL)
# ============================================================================

def solution_1a_downsample_denser(gt_path, kf_path, output_dir='./'):
    """
    Downsample whichever trajectory is denser to match the sparser one.
    
    This is the most universal approach - works regardless of which has more poses.
    """
    print("\n" + "="*80)
    print("SOLUTION 1A: DOWNSAMPLE DENSER TRAJECTORY (UNIVERSAL)")
    print("="*80)
    
    gt = np.loadtxt(gt_path)
    kf = np.loadtxt(kf_path)
    
    print(f"Ground truth poses: {len(gt)}")
    print(f"Keyframe poses: {len(kf)}")
    
    # Determine which trajectory to downsample
    if len(gt) > len(kf):
        print(f"\n→ Ground truth is denser. Downsampling GT to match KF.")
        dense_traj = gt
        sparse_traj = kf
        downsample_gt = True
    elif len(kf) > len(gt):
        print(f"\n→ Keyframes are denser. Downsampling KF to match GT.")
        dense_traj = kf
        sparse_traj = gt
        downsample_gt = False
    else:
        print(f"\n→ Trajectories already have equal length!")
        return gt, kf
    
    # Calculate downsample factor
    downsample_factor = max(1, len(dense_traj) // len(sparse_traj))
    print(f"Downsampling by factor: {downsample_factor}")
    print(f"Dense trajectory original: {len(dense_traj)} poses")
    
    # Downsample
    downsampled = dense_traj[::downsample_factor]
    print(f"After downsampling: {len(downsampled)} poses")
    
    # Trim to match exactly
    min_len = min(len(downsampled), len(sparse_traj))
    downsampled = downsampled[:min_len]
    sparse_trimmed = sparse_traj[:min_len]
    
    print(f"After trimming to match: {min_len} poses")
    
    # Save with appropriate names
    if downsample_gt:
        gt_output = os.path.join(output_dir, '08_gt_downsampled.txt')
        kf_output = os.path.join(output_dir, 'KeyFrameTrajectory_trimmed.txt')
        np.savetxt(gt_output, downsampled, fmt='%.12e')
        np.savetxt(kf_output, sparse_trimmed, fmt='%.12e')
        print(f"\n✓ Saved downsampled GT to: {gt_output}")
        print(f"✓ Saved trimmed KF to: {kf_output}")
    else:
        gt_output = os.path.join(output_dir, '08_gt_trimmed.txt')
        kf_output = os.path.join(output_dir, 'KeyFrameTrajectory_downsampled.txt')
        np.savetxt(gt_output, sparse_trimmed, fmt='%.12e')
        np.savetxt(kf_output, downsampled, fmt='%.12e')
        print(f"\n✓ Saved trimmed GT to: {gt_output}")
        print(f"✓ Saved downsampled KF to: {kf_output}")
    
    print("\n📊 Command to run:")
    print(f"  evo_ape kitti {gt_output} {kf_output} --correct_scale -a --pose_relation trans_part -va --plot --plot_mode xz --save_results results/ate_results.zip")
    
    return downsampled, sparse_trimmed


# ============================================================================
# SOLUTION 1B: INTELLIGENT UNIFORM SAMPLING
# ============================================================================

def solution_1b_uniform_sampling(gt_path, kf_path, output_dir='./'):
    """
    Sample both trajectories uniformly to a common length.
    
    This creates a more balanced comparison by sampling both trajectories
    at uniform intervals to a target length (min of both trajectories).
    """
    print("\n" + "="*80)
    print("SOLUTION 1B: INTELLIGENT UNIFORM SAMPLING")
    print("="*80)
    
    gt = np.loadtxt(gt_path)
    kf = np.loadtxt(kf_path)
    
    print(f"Ground truth poses: {len(gt)}")
    print(f"Keyframe poses: {len(kf)}")
    
    # Use the minimum length as target
    target_len = min(len(gt), len(kf))
    print(f"\n→ Sampling both trajectories to length: {target_len}")
    
    # Uniform sampling indices
    gt_indices = np.linspace(0, len(gt) - 1, target_len, dtype=int)
    kf_indices = np.linspace(0, len(kf) - 1, target_len, dtype=int)
    
    gt_sampled = gt[gt_indices]
    kf_sampled = kf[kf_indices]
    
    print(f"GT sampled: {len(gt_sampled)} poses")
    print(f"KF sampled: {len(kf_sampled)} poses")
    
    # Save
    gt_output = os.path.join(output_dir, '08_gt_uniform.txt')
    kf_output = os.path.join(output_dir, 'KeyFrameTrajectory_uniform.txt')
    
    np.savetxt(gt_output, gt_sampled, fmt='%.12e')
    np.savetxt(kf_output, kf_sampled, fmt='%.12e')
    
    print(f"\n✓ Saved uniformly sampled GT to: {gt_output}")
    print(f"✓ Saved uniformly sampled KF to: {kf_output}")
    
    print("\n📊 Command to run:")
    print(f"  evo_ape kitti {gt_output} {kf_output} --correct_scale -a --pose_relation trans_part -va --plot --plot_mode xz --save_results results/ate_results.zip")
    
    return gt_sampled, kf_sampled


# ============================================================================
# SOLUTION 2: INTERPOLATE SPARSER TRAJECTORY (UNIVERSAL)
# ============================================================================

def solution_2_interpolate_sparser(gt_path, kf_path, output_dir='./'):
    """
    Interpolate whichever trajectory is sparser to match the denser one.
    
    This approach preserves all data from the denser trajectory while
    generating intermediate poses for the sparser trajectory.
    """
    print("\n" + "="*80)
    print("SOLUTION 2: INTERPOLATE SPARSER TRAJECTORY (UNIVERSAL)")
    print("="*80)
    
    gt = np.loadtxt(gt_path)
    kf = np.loadtxt(kf_path)
    
    print(f"Ground truth poses: {len(gt)}")
    print(f"Keyframe poses: {len(kf)}")
    
    # Determine which trajectory to interpolate
    if len(gt) > len(kf):
        print(f"\n→ Keyframes are sparser. Interpolating KF to match GT length.")
        sparse_traj = kf
        dense_traj = gt
        target_len = len(gt)
        interpolate_kf = True
    elif len(kf) > len(gt):
        print(f"\n→ Ground truth is sparser. Interpolating GT to match KF length.")
        sparse_traj = gt
        dense_traj = kf
        target_len = len(kf)
        interpolate_kf = False
    else:
        print(f"\n→ Trajectories already have equal length!")
        return gt, kf
    
    print(f"Interpolating from {len(sparse_traj)} to {target_len} poses")
    
    # Convert to pose matrices
    sparse_poses = kitti_to_poses(sparse_traj)
    
    # Create indices for sparse trajectory in the dense trajectory space
    sparse_indices = np.linspace(0, target_len - 1, len(sparse_poses))
    
    # Interpolate
    interpolated_poses = []
    
    for i in range(target_len):
        if i <= sparse_indices[0]:
            # Before first pose
            pose = sparse_poses[0]
        elif i >= sparse_indices[-1]:
            # After last pose
            pose = sparse_poses[-1]
        else:
            # Find surrounding poses
            idx = np.searchsorted(sparse_indices, i)
            
            if idx == 0:
                pose = sparse_poses[0]
            elif idx >= len(sparse_poses):
                pose = sparse_poses[-1]
            else:
                # Interpolate between sparse_poses[idx-1] and sparse_poses[idx]
                idx_before = idx - 1
                idx_after = idx
                
                t_before = sparse_indices[idx_before]
                t_after = sparse_indices[idx_after]
                
                # Interpolation factor
                t = (i - t_before) / (t_after - t_before)
                
                pose = interpolate_pose_slerp(
                    sparse_poses[idx_before],
                    sparse_poses[idx_after],
                    t
                )
        
        interpolated_poses.append(pose)
    
    # Convert back to KITTI format
    interpolated_traj = poses_to_kitti(interpolated_poses)
    
    print(f"Interpolated trajectory: {len(interpolated_traj)} poses")
    print(f"Shape matches target: {interpolated_traj.shape[0] == target_len}")
    
    # Save with appropriate names
    if interpolate_kf:
        gt_output = os.path.join(output_dir, '08_gt_original.txt')
        kf_output = os.path.join(output_dir, 'KeyFrameTrajectory_interpolated.txt')
        np.savetxt(gt_output, dense_traj, fmt='%.12e')
        np.savetxt(kf_output, interpolated_traj, fmt='%.12e')
        print(f"\n✓ Saved original GT to: {gt_output}")
        print(f"✓ Saved interpolated KF to: {kf_output}")
    else:
        gt_output = os.path.join(output_dir, '08_gt_interpolated.txt')
        kf_output = os.path.join(output_dir, 'KeyFrameTrajectory_original.txt')
        np.savetxt(gt_output, interpolated_traj, fmt='%.12e')
        np.savetxt(kf_output, dense_traj, fmt='%.12e')
        print(f"\n✓ Saved interpolated GT to: {gt_output}")
        print(f"✓ Saved original KF to: {kf_output}")
    
    print("\n📊 Command to run:")
    print(f"  evo_ape kitti {gt_output} {kf_output} --correct_scale -a --pose_relation trans_part -va --plot --plot_mode xz --save_results results/ate_results.zip")
    
    return interpolated_traj if not interpolate_kf else dense_traj, interpolated_traj if interpolate_kf else dense_traj


# ============================================================================
# SOLUTION 3: TIMESTAMP-BASED ALIGNMENT (RECOMMENDED)
# ============================================================================

def solution_3_timestamp_alignment():
    """
    Use timestamp-based alignment with EVO (when timestamps available).
    
    This is the gold standard for SLAM evaluation when you have timing info.
    """
    print("\n" + "="*80)
    print("SOLUTION 3: TIMESTAMP-BASED ALIGNMENT (RECOMMENDED)")
    print("="*80)
    print("""
If you have timestamp information (TUM format), this is the best approach:

TUM FORMAT (time x y z qx qy qz qw):
  timestamp tx ty tz qx qy qz qw

CONVERT KITTI TO TUM WITH TIMESTAMPS:
  evo_traj kitti KeyFrameTrajectory.txt --save_as_tum

EVALUATE WITH TIMESTAMP ALIGNMENT:
  evo_ape tum ground_truth.tum slam_trajectory.tum \\
    -va --plot --save_results results/ate_results.zip \\
    --align --correct_scale

This automatically handles pose count mismatches through time-based matching.
""")


# ============================================================================
# SOLUTION 4: EVO WITH PROPER FLAGS
# ============================================================================

def solution_4_evo_advanced():
    """
    Advanced EVO options for handling trajectory comparison.
    """
    print("\n" + "="*80)
    print("SOLUTION 4: ADVANCED EVO OPTIONS")
    print("="*80)
    print("""
EVO has several options to handle pose count mismatches:

1. ALIGN WITH SE(3) Umeyama (handles scale, rotation, translation):
   evo_ape kitti gt.txt slam.txt -va --plot \\
     --align --correct_scale

2. ALIGN ORIGIN ONLY (simpler, may work with different lengths):
   evo_ape kitti gt.txt slam.txt -va --plot \\
     --align_origin --correct_scale

3. ALIGN WITH N POSES (uses first N poses for alignment):
   evo_ape kitti gt.txt slam.txt -va --plot \\
     --align --n_to_align 100 --correct_scale

4. NO ALIGNMENT (compare in same coordinate system):
   evo_ape kitti gt.txt slam.txt -va --plot

5. CHECK YOUR EVO VERSION (newer versions handle mismatches better):
   evo --version
   pip install evo --upgrade

NOTE: Some EVO versions handle trajectory length mismatches automatically.
Try the commands above before resampling trajectories.
""")


# ============================================================================
# AUTOMATIC SOLUTION SELECTOR
# ============================================================================

def auto_select_solution(gt_path, kf_path, output_dir='./'):
    """
    Automatically select the best solution based on trajectory characteristics.
    """
    print("\n" + "="*80)
    print("AUTO-SELECT: ANALYZING TRAJECTORIES")
    print("="*80)
    
    gt = np.loadtxt(gt_path)
    kf = np.loadtxt(kf_path)
    
    gt_len = len(gt)
    kf_len = len(kf)
    ratio = max(gt_len, kf_len) / min(gt_len, kf_len)
    
    print(f"Ground truth poses: {gt_len}")
    print(f"Keyframe poses: {kf_len}")
    print(f"Length ratio: {ratio:.2f}x")
    
    # Decision logic
    if abs(gt_len - kf_len) <= 10:
        print("\n✓ Trajectories are nearly equal length.")
        print("→ Recommendation: Use Solution 1B (uniform sampling) or try EVO directly")
        return solution_1b_uniform_sampling(gt_path, kf_path, output_dir)
    
    elif ratio < 1.5:
        print("\n✓ Trajectories have similar lengths (ratio < 1.5x).")
        print("→ Recommendation: Solution 1B (uniform sampling) for balanced comparison")
        return solution_1b_uniform_sampling(gt_path, kf_path, output_dir)
    
    elif ratio < 3.0:
        print("\n✓ Moderate length difference (1.5x - 3x).")
        print("→ Recommendation: Solution 1A (downsample denser trajectory)")
        return solution_1a_downsample_denser(gt_path, kf_path, output_dir)
    
    else:
        print("\n⚠ Large length difference (>3x ratio).")
        print("→ Recommendation: Solution 2 (interpolate sparser trajectory)")
        print("→ Alternative: Consider if data alignment is correct")
        return solution_2_interpolate_sparser(gt_path, kf_path, output_dir)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("\n" + "="*80)
    print("ENHANCED POSE MISMATCH FIX - Universal Solution")
    print("="*80)
    print("""
This enhanced version handles ALL cases:
  ✓ Ground truth denser than keyframes (GT > KF)
  ✓ Keyframes denser than ground truth (KF > GT)  ← YOUR CASE
  ✓ Roughly equal lengths (GT ≈ KF)

CURRENT SCENARIO:
  - Ground truth: 1,790 poses
  - Keyframes: 1,801 poses
  - Issue: Keyframes outnumber ground truth (uncommon but valid)

WHY THIS HAPPENS:
  - Very smooth camera motion → more keyframes selected
  - Short sequence with detailed tracking
  - Loop closure creating additional keyframes
""")
    
    # Default paths (modify as needed)
    default_gt = './data/data_odometry_poses/poses/08_cleaned.txt'
    default_kf = './third_party/ORB_SLAM3/Examples/Monocular/KeyFrameTrajectory.txt'
    
    print("\n" + "="*80)
    print("SOLUTION OPTIONS:")
    print("="*80)
    print("  0) Auto-select best solution (recommended)")
    print("  1a) Downsample denser trajectory (universal)")
    print("  1b) Uniform sampling (balanced)")
    print("  2) Interpolate sparser trajectory (preserves dense data)")
    print("  3) Timestamp-based alignment (if timestamps available)")
    print("  4) Advanced EVO options (try first)")
    print("  5) Show all solutions")
    
    choice = input("\nEnter choice (0-5, or press Enter for 0): ").strip() or "0"
    
    # Get file paths
    if len(sys.argv) >= 3:
        gt_path = sys.argv[1]
        kf_path = sys.argv[2]
        output_dir = sys.argv[3] if len(sys.argv) >= 4 else './'
    else:
        use_default = input(f"\nUse default paths? (y/n, default=y): ").strip().lower()
        if use_default != 'n':
            gt_path = default_gt
            kf_path = default_kf
            output_dir = './'
        else:
            gt_path = input("Ground truth path: ").strip()
            kf_path = input("Keyframe path: ").strip()
            output_dir = input("Output directory (default=./): ").strip() or './'
    
    # Execute chosen solution
    if choice == "0":
        auto_select_solution(gt_path, kf_path, output_dir)
    elif choice == "1a":
        solution_1a_downsample_denser(gt_path, kf_path, output_dir)
    elif choice == "1b":
        solution_1b_uniform_sampling(gt_path, kf_path, output_dir)
    elif choice == "2":
        solution_2_interpolate_sparser(gt_path, kf_path, output_dir)
    elif choice == "3":
        solution_3_timestamp_alignment()
    elif choice == "4":
        solution_4_evo_advanced()
    elif choice == "5":
        print("\n" + "="*80)
        print("SHOWING ALL SOLUTIONS")
        print("="*80)
        auto_select_solution(gt_path, kf_path, output_dir)
        solution_3_timestamp_alignment()
        solution_4_evo_advanced()
    else:
        print("Invalid choice. Using auto-select.")
        auto_select_solution(gt_path, kf_path, output_dir)
    
    print("\n" + "="*80)
    print("✓ PROCESSING COMPLETE")
    print("="*80)
    print("\n💡 TIPS:")
    print("  - For research papers, document which alignment method you used")
    print("  - Downsampling is conservative (loses data)")
    print("  - Interpolation assumes smooth motion between poses")
    print("  - Uniform sampling provides balanced comparison")
    print("  - Try EVO's built-in options first (Solution 4)")
    print("\n")


if __name__ == '__main__':
    main()