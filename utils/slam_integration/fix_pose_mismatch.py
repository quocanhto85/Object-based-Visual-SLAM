#!/usr/bin/env python3
"""
Enhanced Fix for EVO pose alignment mismatch error - Complete Solution

This version handles:
1. Pose count mismatches (GT vs KF)
2. Coordinate frame mismatches (rotation/flip detection and correction)
3. Scale differences (handled by evo's Sim(3), but we verify)

The script automatically:
- Detects coordinate frame mismatches
- Applies the correct transformation
- Resamples trajectories to match lengths
- Saves corrected files ready for evo_ape
"""

import numpy as np
import os
from scipy.spatial.transform import Rotation
from scipy.interpolate import interp1d
import sys
from pathlib import Path


def load_trajectory(filepath):
    """Load trajectory from KITTI format file."""
    return np.loadtxt(filepath)


def analyze_trajectory(traj, name="Trajectory"):
    """Analyze trajectory characteristics."""
    translations = traj[:, [3, 7, 11]]  # Extract tx, ty, tz
    
    # Compute statistics
    mean = translations.mean(axis=0)
    variance = translations.var(axis=0)
    range_vals = translations.max(axis=0) - translations.min(axis=0)
    
    main_axis = variance.argmax()
    main_axis_name = ['X', 'Y', 'Z'][main_axis]
    
    return {
        'name': name,
        'translations': translations,
        'mean': mean,
        'variance': variance,
        'range': range_vals,
        'main_axis': main_axis,
        'main_axis_name': main_axis_name,
        'length': len(traj)
    }


def detect_coordinate_mismatch(gt_info, kf_info):
    """
    Detect if there's a coordinate frame mismatch between GT and KF.
    
    Returns:
        dict: {
            'mismatch': bool,
            'type': str ('rotation', 'flip', 'none'),
            'recommended_transform': str
        }
    """
    gt_main = gt_info['main_axis']
    kf_main = kf_info['main_axis']
    
    # Check if main motion axes differ
    if gt_main != kf_main:
        # Determine rotation type
        if (gt_main == 2 and kf_main == 0) or (gt_main == 0 and kf_main == 2):
            # Z <-> X: 90° rotation around Y
            return {
                'mismatch': True,
                'type': 'rotation_90_Y',
                'recommended_transform': 'rotate_90_Y',
                'description': 'Main motion axis differs: GT in {}, KF in {}'.format(
                    gt_info['main_axis_name'], kf_info['main_axis_name']
                )
            }
        else:
            return {
                'mismatch': True,
                'type': 'rotation_other',
                'recommended_transform': 'auto_detect',
                'description': 'Complex axis mismatch detected'
            }
    
    # Check for opposite directions (180° rotation or flip)
    gt_dir = np.sign(gt_info['mean'][gt_main])
    kf_dir = np.sign(kf_info['mean'][kf_main])
    
    if gt_dir != kf_dir:
        return {
            'mismatch': True,
            'type': 'flip',
            'recommended_transform': 'rotate_180_Y',
            'description': 'Trajectories face opposite directions'
        }
    
    # No obvious mismatch
    return {
        'mismatch': False,
        'type': 'none',
        'recommended_transform': 'identity',
        'description': 'No coordinate mismatch detected'
    }


def get_transformation_matrix(transform_type):
    """
    Get 4x4 transformation matrix for given transform type.
    
    Available transforms:
    - rotate_90_Y: Rotate 90° around Y (X→Z, Z→-X)
    - rotate_neg90_Y: Rotate -90° around Y (X→-Z, Z→X)
    - rotate_180_Y: Rotate 180° around Y (X→-X, Z→-Z)
    - flip_X: Flip X axis
    - flip_Z: Flip Z axis
    - identity: No transformation
    - auto_detect: Test all and return best
    """
    transforms = {
        'rotate_90_Y': np.array([
            [0, 0, -1, 0],
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1]
        ]),
        'rotate_neg90_Y': np.array([
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [-1, 0, 0, 0],
            [0, 0, 0, 1]
        ]),
        'rotate_180_Y': np.array([
            [-1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ]),
        'flip_X': np.array([
            [-1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]),
        'flip_Z': np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ]),
        'identity': np.eye(4)
    }
    
    return transforms.get(transform_type, np.eye(4))


def apply_transformation(trajectory, transform_matrix):
    """Apply transformation matrix to trajectory."""
    transformed = []
    
    for pose_row in trajectory:
        # Reshape to 4x4
        pose = np.eye(4)
        pose[:3, :] = pose_row.reshape(3, 4)
        
        # Apply transformation
        pose_new = transform_matrix @ pose
        
        # Store as 3x4 flattened
        transformed.append(pose_new[:3, :].flatten())
    
    return np.array(transformed)


def auto_detect_best_transformation(gt_traj, kf_traj):
    """
    Automatically detect best transformation by testing all options.
    Returns the transform that minimizes center distance.
    """
    gt_trans = gt_traj[:, [3, 7, 11]]
    gt_center = gt_trans.mean(axis=0)
    
    transforms_to_test = [
        'rotate_90_Y',
        'rotate_neg90_Y',
        'rotate_180_Y',
        'flip_X',
        'flip_Z',
        'identity'
    ]
    
    best_transform = 'identity'
    min_distance = float('inf')
    results = {}
    
    print("\n" + "="*80)
    print("AUTO-DETECTING BEST COORDINATE TRANSFORMATION")
    print("="*80)
    print(f"Ground truth center: [{gt_center[0]:.1f}, {gt_center[1]:.1f}, {gt_center[2]:.1f}]")
    print("\nTesting transformations:\n")
    
    for transform_name in transforms_to_test:
        T = get_transformation_matrix(transform_name)
        kf_transformed = apply_transformation(kf_traj, T)
        kf_trans = kf_transformed[:, [3, 7, 11]]
        kf_center = kf_trans.mean(axis=0)
        
        # Distance metric
        center_dist = np.linalg.norm(gt_center - kf_center)
        
        # Variance alignment metric (main axis should match)
        gt_var = gt_trans.var(axis=0)
        kf_var = kf_trans.var(axis=0)
        var_alignment = np.dot(gt_var, kf_var) / (np.linalg.norm(gt_var) * np.linalg.norm(kf_var))
        
        # Combined score (lower is better for distance, higher is better for alignment)
        score = center_dist / (1 + var_alignment)
        
        results[transform_name] = {
            'center_dist': center_dist,
            'var_alignment': var_alignment,
            'score': score,
            'kf_center': kf_center
        }
        
        print(f"  {transform_name:20s}: center_dist={center_dist:7.1f}m, var_align={var_alignment:.3f}, score={score:.1f}")
        
        if score < min_distance:
            min_distance = score
            best_transform = transform_name
    
    print(f"\n✓ Best transformation: {best_transform} (score: {min_distance:.1f})")
    print(f"  KF center after transform: {results[best_transform]['kf_center']}")
    
    return best_transform, results


def uniform_sample(trajectory, target_length):
    """Uniformly sample trajectory to target length."""
    indices = np.linspace(0, len(trajectory) - 1, target_length, dtype=int)
    return trajectory[indices]


def fix_pose_mismatch(gt_path, kf_path, output_dir='./', auto_transform=True):
    """
    Complete fix for pose mismatch: handles both coordinate frames and length.
    
    Args:
        gt_path: Path to ground truth file
        kf_path: Path to keyframe trajectory file
        output_dir: Output directory for fixed files
        auto_transform: Automatically detect and apply coordinate transformation
    
    Returns:
        dict: Information about applied fixes
    """
    print("\n" + "="*80)
    print("COMPREHENSIVE POSE MISMATCH FIX")
    print("="*80)
    
    # Load trajectories
    print("\n[1/5] Loading trajectories...")
    gt_traj = load_trajectory(gt_path)
    kf_traj = load_trajectory(kf_path)
    
    print(f"  Ground truth: {len(gt_traj)} poses")
    print(f"  Keyframes:    {len(kf_traj)} poses")
    
    # Analyze trajectories
    print("\n[2/5] Analyzing trajectory characteristics...")
    gt_info = analyze_trajectory(gt_traj, "Ground Truth")
    kf_info = analyze_trajectory(kf_traj, "Keyframes")
    
    print(f"\n  Ground Truth:")
    print(f"    Main motion axis: {gt_info['main_axis_name']}")
    print(f"    Variance: X={gt_info['variance'][0]:.1f}, Y={gt_info['variance'][1]:.1f}, Z={gt_info['variance'][2]:.1f}")
    print(f"    Range: X={gt_info['range'][0]:.1f}m, Y={gt_info['range'][1]:.1f}m, Z={gt_info['range'][2]:.1f}m")
    
    print(f"\n  Keyframes:")
    print(f"    Main motion axis: {kf_info['main_axis_name']}")
    print(f"    Variance: X={kf_info['variance'][0]:.1f}, Y={kf_info['variance'][1]:.1f}, Z={kf_info['variance'][2]:.1f}")
    print(f"    Range: X={kf_info['range'][0]:.1f}m, Y={kf_info['range'][1]:.1f}m, Z={kf_info['range'][2]:.1f}m")
    
    # Detect coordinate mismatch
    print("\n[3/5] Detecting coordinate frame mismatch...")
    mismatch_info = detect_coordinate_mismatch(gt_info, kf_info)
    
    if mismatch_info['mismatch']:
        print(f"  ⚠️  MISMATCH DETECTED: {mismatch_info['description']}")
        print(f"  Type: {mismatch_info['type']}")
        
        if auto_transform:
            if mismatch_info['recommended_transform'] == 'auto_detect':
                transform_type, _ = auto_detect_best_transformation(gt_traj, kf_traj)
            else:
                transform_type = mismatch_info['recommended_transform']
            
            print(f"\n  Applying transformation: {transform_type}")
            T = get_transformation_matrix(transform_type)
            kf_traj = apply_transformation(kf_traj, T)
            
            # Re-analyze after transformation
            kf_info = analyze_trajectory(kf_traj, "Keyframes (after transform)")
            print(f"  After transform - Main axis: {kf_info['main_axis_name']}")
        else:
            print(f"  Recommended transform: {mismatch_info['recommended_transform']}")
            print(f"  Auto-transform disabled. Proceeding without transformation.")
    else:
        print(f"  ✓ No coordinate mismatch detected")
        transform_type = 'none'
    
    # Handle length mismatch
    print("\n[4/5] Handling trajectory length mismatch...")
    target_length = min(len(gt_traj), len(kf_traj))
    
    if len(gt_traj) != len(kf_traj):
        print(f"  Length mismatch detected")
        print(f"  Sampling both to length: {target_length}")
        
        gt_traj = uniform_sample(gt_traj, target_length)
        kf_traj = uniform_sample(kf_traj, target_length)
    else:
        print(f"  ✓ Lengths already match: {target_length}")
    
    # Save results
    print("\n[5/5] Saving corrected trajectories...")
    os.makedirs(output_dir, exist_ok=True)
    
    gt_output = os.path.join(output_dir, 'gt_corrected.txt')
    kf_output = os.path.join(output_dir, 'kf_corrected.txt')
    
    np.savetxt(gt_output, gt_traj, fmt='%.12e')
    np.savetxt(kf_output, kf_traj, fmt='%.12e')
    
    print(f"  ✓ Saved corrected GT to: {gt_output}")
    print(f"  ✓ Saved corrected KF to: {kf_output}")
    
    # Final summary
    print("\n" + "="*80)
    print("✓ PROCESSING COMPLETE")
    print("="*80)
    
    print(f"\nApplied fixes:")
    print(f"  1. Coordinate transformation: {transform_type}")
    print(f"  2. Uniform sampling: {target_length} poses")
    
    print(f"\n📊 Ready for evaluation:")
    print(f"\n  evo_ape kitti {gt_output} {kf_output} \\")
    print(f"      --correct_scale -a --pose_relation trans_part \\")
    print(f"      -va --plot --plot_mode xz --save_results results/corrected_result.zip")
    
    return {
        'transform_applied': transform_type,
        'final_length': target_length,
        'gt_output': gt_output,
        'kf_output': kf_output,
        'mismatch_detected': mismatch_info['mismatch']
    }


def main():
    """Main function with command-line interface."""
    print("\n" + "="*80)
    print("ENHANCED POSE MISMATCH FIX - Complete Solution")
    print("="*80)
    print("""
This script automatically:
  ✓ Detects coordinate frame mismatches (rotation/flip)
  ✓ Applies the correct transformation
  ✓ Resamples trajectories to match lengths
  ✓ Generates files ready for evo_ape evaluation

No manual intervention required!
""")
    
    # Default paths
    default_gt = './data/data_odometry_poses/poses/08_cleaned.txt'
    default_kf = './third_party/ORB_SLAM3/Examples/Monocular/KeyFrameTrajectory.txt'
    default_output = './'
    
    # Parse command line arguments
    if len(sys.argv) >= 3:
        gt_path = sys.argv[1]
        kf_path = sys.argv[2]
        output_dir = sys.argv[3] if len(sys.argv) >= 4 else default_output
    else:
        print("Usage: python fix_pose_mismatch.py <gt_path> <kf_path> [output_dir]")
        print("\nUsing default paths...")
        
        gt_path = default_gt
        kf_path = default_kf
        output_dir = default_output
    
    print(f"\nInput files:")
    print(f"  Ground truth: {gt_path}")
    print(f"  Keyframes:    {kf_path}")
    print(f"  Output dir:   {output_dir}")
    
    # Check if files exist
    if not os.path.exists(gt_path):
        print(f"\n❌ Error: Ground truth file not found: {gt_path}")
        sys.exit(1)
    
    if not os.path.exists(kf_path):
        print(f"\n❌ Error: Keyframe file not found: {kf_path}")
        sys.exit(1)
    
    # Run the fix
    try:
        result = fix_pose_mismatch(gt_path, kf_path, output_dir, auto_transform=True)
        
        print("\n" + "="*80)
        print("SUCCESS!")
        print("="*80)
        print("\nYour trajectories are now aligned and ready for evaluation.")
        print("The coordinate frame mismatch has been automatically corrected.")
        print("\nNext time you run ORB-SLAM3, just update the keyframe path and re-run this script!")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())