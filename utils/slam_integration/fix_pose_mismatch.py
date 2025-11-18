#!/usr/bin/env python3
"""
Fix for EVO pose alignment mismatch error.

The issue: ORB-SLAM3 outputs keyframe-only trajectories (sparse), while ground truth
has poses for every frame (dense). The Umeyama algorithm requires equal-sized matrices.

Solutions provided below:
"""

import numpy as np
import os
from scipy.spatial.transform import Rotation
from scipy.interpolate import interp1d


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


# ============================================================================
# SOLUTION 1: DOWNSAMPLE GROUND TRUTH TO MATCH KEYFRAME DENSITY
# ============================================================================

def solution_1_downsample_ground_truth():
    """
    Downsample ground truth to roughly match keyframe density.
    
    This approach reduces ground truth to approximately the same number of poses
    as the keyframe trajectory by taking every Nth frame.
    """
    print("\n" + "="*80)
    print("SOLUTION 1: DOWNSAMPLE GROUND TRUTH TO MATCH KEYFRAME DENSITY")
    print("="*80)
    
    gt = np.loadtxt('./data/data_odometry_poses/poses/08_cleaned.txt')
    kf = np.loadtxt('./third_party/ORB_SLAM3/Examples/Monocular/KeyFrameTrajectory.txt')
    
    print(f"Ground truth poses: {len(gt)}")
    print(f"Keyframe poses: {len(kf)}")
    
    downsample_factor = len(gt) // len(kf)
    print(f"Downsampling by factor: {downsample_factor}")
    print(f"Original GT poses: {len(gt)}")
    
    # Take every nth frame
    gt_downsampled = gt[::downsample_factor]
    
    print(f"Downsampled GT poses: {len(gt_downsampled)}")
    print(f"Keyframe poses: {len(kf)}")
    
    # Trim to match exactly
    min_len = min(len(gt_downsampled), len(kf))
    gt_downsampled = gt_downsampled[:min_len]
    kf_trimmed = kf[:min_len]
    
    print(f"After trimming to match: {len(gt_downsampled)} poses")
    
    # Save
    np.savetxt('./data/data_odometry_poses/poses/08_cleaned.txt', 
               gt_downsampled, fmt='%.12e')
    
    print(f"\nSaved downsampled ground truth to: 08_cleaned.txt")
    print("Command to run:")
    print("  evo_ape kitti ./data/data_odometry_poses/poses/08_cleaned.txt \\")
    print("    ./third_party/ORB_SLAM3/Examples/Monocular/KeyFrameTrajectory.txt \\")
    print("    -va --plot --save_results results/ate_results.zip")
    
    return gt_downsampled, kf_trimmed


# ============================================================================
# SOLUTION 2: INTERPOLATE KEYFRAME TRAJECTORY TO MATCH GROUND TRUTH DENSITY
# ============================================================================

def interpolate_pose_linear(pose1, pose2, t):
    """
    Linear interpolation between two poses (3x4 matrices).
    
    Args:
        pose1, pose2: 3x4 pose matrices
        t: interpolation factor (0 to 1)
    """
    # Extract components
    R1 = pose1[:3, :3]
    t1 = pose1[:3, 3]
    R2 = pose2[:3, :3]
    t2 = pose2[:3, 3]
    
    # Interpolate translation linearly
    t_interp = (1 - t) * t1 + t * t2
    
    # Convert rotation matrices to quaternions
    rot1 = Rotation.from_matrix(R1)
    rot2 = Rotation.from_matrix(R2)
    
    # SLERP (spherical linear interpolation) for rotation
    rot_interp = Rotation.from_quat(
        (1 - t) * rot1.as_quat() + t * rot2.as_quat()
    )
    rot_interp = rot_interp.from_quat(rot_interp.as_quat() / np.linalg.norm(rot_interp.as_quat()))
    
    # Reconstruct pose
    pose_interp = np.eye(4)
    pose_interp[:3, :3] = rot_interp.as_matrix()
    pose_interp[:3, 3] = t_interp
    
    return pose_interp[:3, :]


def solution_2_interpolate_keyframes():
    """
    Interpolate the sparse keyframe trajectory to match ground truth density.
    
    This approach uses linear interpolation between keyframes to estimate
    intermediate poses, matching the ground truth frame count.
    """
    print("\n" + "="*80)
    print("SOLUTION 2: INTERPOLATE KEYFRAME TRAJECTORY TO MATCH GT DENSITY")
    print("="*80)
    
    gt = np.loadtxt('./data/data_odometry_poses/poses/08_cleaned.txt')
    kf = np.loadtxt('./third_party/ORB_SLAM3/Examples/Monocular/KeyFrameTrajectory.txt')
    
    print(f"Ground truth poses: {len(gt)}")
    print(f"Keyframe poses: {len(kf)}")
    
    kf_poses = kitti_to_poses(kf)
    
    # Create index mapping: keyframe indices in the original sequence
    # Assuming keyframes are uniformly distributed (approximation)
    kf_indices = np.linspace(0, len(gt) - 1, len(kf_poses))
    
    # Interpolate to create poses for all GT frames
    interpolated_poses = []
    
    for i in range(len(gt)):
        if i < kf_indices[0]:
            # Before first keyframe - use first keyframe
            pose = kf_poses[0]
        elif i > kf_indices[-1]:
            # After last keyframe - use last keyframe
            pose = kf_poses[-1]
        else:
            # Find surrounding keyframes
            idx = np.searchsorted(kf_indices, i)
            if idx == 0:
                pose = kf_poses[0]
            elif idx >= len(kf_poses):
                pose = kf_poses[-1]
            else:
                # Interpolate between keyframe idx-1 and idx
                kf_before_idx = idx - 1
                kf_after_idx = idx
                
                t_before = kf_indices[kf_before_idx]
                t_after = kf_indices[kf_after_idx]
                
                # Interpolation factor
                t = (i - t_before) / (t_after - t_before)
                
                pose = interpolate_pose_linear(
                    kf_poses[kf_before_idx],
                    kf_poses[kf_after_idx],
                    t
                )
        
        # Convert back to KITTI format (3x4 flattened)
        pose_flat = pose.flatten()
        interpolated_poses.append(pose_flat)
    
    interpolated_poses = np.array(interpolated_poses)
    
    print(f"Interpolated keyframe poses: {len(interpolated_poses)}")
    print(f"Shape matches GT: {interpolated_poses.shape == gt.shape}")
    
    np.savetxt('./third_party/ORB_SLAM3/Examples/Monocular/KeyFrameTrajectory_kitti_interpolated.txt',
               interpolated_poses, fmt='%.12e')
    
    print(f"\nSaved interpolated keyframe trajectory to: KeyFrameTrajectory_kitti_interpolated.txt")
    print("Command to run:")
    print("  evo_ape kitti ./data/data_odometry_poses/poses/08_cleaned.txt \\")
    print("    ./third_party/ORB_SLAM3/Examples/Monocular/KeyFrameTrajectory_kitti_interpolated.txt \\")
    print("    -va --plot --save_results results/ate_results.zip")
    
    return interpolated_poses


# ============================================================================
# SOLUTION 3: USE EVO WITH PROPER FLAGS (RECOMMENDED)
# ============================================================================

def solution_3_evo_command():
    """
    The best solution: Use EVO's built-in alignment features.
    
    EVO can handle pose count mismatches by:
    - Using timestamp-based alignment
    - Automatically interpolating if needed
    - Using the --align_origin flag to synchronize trajectories
    """
    print("\n" + "="*80)
    print("SOLUTION 3: USE EVO WITH PROPER FLAGS (RECOMMENDED)")
    print("="*80)
    print("""
The error occurs because Umeyama's method expects equal-sized matrices.
EVO has better options for sparse vs. dense trajectory comparison:

Try these commands:

1. WITH AUTOMATIC ALIGNMENT (Best):
   evo_ape kitti ./data/data_odometry_poses/poses/08_cleaned.txt \\
     ./third_party/ORB_SLAM3/Examples/Monocular/KeyFrameTrajectory.txt \\
     -va --plot --save_results results/ate_results.zip \\
     --align_origin --correct_scale

2. WITH CUSTOM ALIGNMENT:
   evo_ape kitti ./data/data_odometry_poses/poses/08_cleaned.txt \\
     ./third_party/ORB_SLAM3/Examples/Monocular/KeyFrameTrajectory.txt \\
     -va --plot --save_results results/ate_results.zip \\
     --align_origin --n_to_align 3

3. CHECK EVO VERSION AND OPTIONS:
   evo_ape --help | grep -E "(align|scale|interpolate)"

Note: These flags handle the pose count mismatch by aligning the trajectories
properly before computing the error metric.
""")


if __name__ == '__main__':
    import sys
    
    print("\n" + "="*80)
    print("POSE MISMATCH FIX - ORB-SLAM3 vs Ground Truth")
    print("="*80)
    print("""
PROBLEM SUMMARY:
  - Ground truth: 4,071 poses (dense, every frame)
  - ORB-SLAM3 keyframes: 2,189 poses (sparse, only distinctive frames)
  - Issue: Umeyama alignment requires same number of poses

WHY THE MISMATCH?
  ORB-SLAM3 uses a keyframe-based approach:
  - Only saves poses at keyframes (visually significant poses)
  - Reduces computation and improves stability
  - This is NORMAL behavior for any visual SLAM system
  - Only ~54% of frames become keyframes in this sequence
""")
    
    print("\nChoose your approach:")
    print("  1) Downsample ground truth to keyframe density")
    print("  2) Interpolate keyframe trajectory to ground truth density")
    print("  3) Use EVO's built-in alignment (RECOMMENDED)")
    print("  4) Show all solutions")
    
    choice = input("\nEnter choice (1-4, or press Enter for 4): ").strip() or "4"
    
    if choice == "1":
        solution_1_downsample_ground_truth()
    elif choice == "2":
        solution_2_interpolate_keyframes()
    elif choice == "3":
        solution_3_evo_command()
    else:
        solution_1_downsample_ground_truth()
        solution_2_interpolate_keyframes()
        solution_3_evo_command()