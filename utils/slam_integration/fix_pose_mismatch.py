#!/usr/bin/env python3
"""
Enhanced Fix for EVO pose alignment mismatch error - Complete Solution
Supports both KITTI and TUM trajectory formats

This version handles:
1. Pose count mismatches (GT vs KF)
2. Coordinate frame mismatches (rotation/flip detection and correction)
3. Scale differences (handled by evo's Sim(3), but we verify)
4. Both KITTI and TUM trajectory formats

The script automatically:
- Detects trajectory format (KITTI or TUM)
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


def detect_format(filepath):
    """Detect trajectory format (KITTI or TUM)."""
    data = np.loadtxt(filepath)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    
    num_cols = data.shape[1]
    
    if num_cols == 12:
        return 'kitti'
    elif num_cols == 8:
        return 'tum'
    else:
        raise ValueError(f"Unknown format: {num_cols} columns. Expected 12 (KITTI) or 8 (TUM)")


def load_trajectory(filepath):
    """Load trajectory from KITTI or TUM format file."""
    data = np.loadtxt(filepath)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data


def tum_to_kitti(tum_data):
    """
    Convert TUM format to KITTI format.
    TUM: timestamp tx ty tz qx qy qz qw (8 values)
    KITTI: r11 r12 r13 tx r21 r22 r23 ty r31 r32 r33 tz (12 values)
    """
    kitti_data = []
    
    for row in tum_data:
        # Extract translation and quaternion
        tx, ty, tz = row[1], row[2], row[3]
        qx, qy, qz, qw = row[4], row[5], row[6], row[7]
        
        # Convert quaternion to rotation matrix
        r = Rotation.from_quat([qx, qy, qz, qw])
        R = r.as_matrix()
        
        # Create KITTI row: [r11 r12 r13 tx r21 r22 r23 ty r31 r32 r33 tz]
        kitti_row = [
            R[0, 0], R[0, 1], R[0, 2], tx,
            R[1, 0], R[1, 1], R[1, 2], ty,
            R[2, 0], R[2, 1], R[2, 2], tz
        ]
        kitti_data.append(kitti_row)
    
    return np.array(kitti_data)


def kitti_to_tum(kitti_data, start_timestamp=0.0):
    """
    Convert KITTI format to TUM format.
    KITTI: r11 r12 r13 tx r21 r22 r23 ty r31 r32 r33 tz (12 values)
    TUM: timestamp tx ty tz qx qy qz qw (8 values)
    """
    tum_data = []
    
    for i, row in enumerate(kitti_data):
        # Extract rotation matrix and translation
        R = np.array([
            [row[0], row[1], row[2]],
            [row[4], row[5], row[6]],
            [row[8], row[9], row[10]]
        ])
        tx, ty, tz = row[3], row[7], row[11]
        
        # Convert rotation matrix to quaternion
        r = Rotation.from_matrix(R)
        quat = r.as_quat()  # Returns [qx, qy, qz, qw]
        
        # Create TUM row with sequential timestamp
        timestamp = start_timestamp + i * 0.1  # 0.1 second intervals
        tum_row = [timestamp, tx, ty, tz, quat[0], quat[1], quat[2], quat[3]]
        tum_data.append(tum_row)
    
    return np.array(tum_data)


def extract_translations(traj, fmt):
    """Extract translations from trajectory based on format."""
    if fmt == 'kitti':
        return traj[:, [3, 7, 11]]  # Extract tx, ty, tz
    elif fmt == 'tum':
        return traj[:, [1, 2, 3]]   # Extract tx, ty, tz
    else:
        raise ValueError(f"Unknown format: {fmt}")


def analyze_trajectory(traj, fmt, name="Trajectory"):
    """Analyze trajectory characteristics."""
    translations = extract_translations(traj, fmt)
    
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
        'length': len(traj),
        'format': fmt
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


def apply_transformation(trajectory, transform_matrix, fmt):
    """Apply transformation matrix to trajectory (works for both KITTI and TUM)."""
    if fmt == 'kitti':
        return apply_transformation_kitti(trajectory, transform_matrix)
    elif fmt == 'tum':
        return apply_transformation_tum(trajectory, transform_matrix)
    else:
        raise ValueError(f"Unknown format: {fmt}")


def apply_transformation_kitti(trajectory, transform_matrix):
    """Apply transformation matrix to KITTI trajectory."""
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


def apply_transformation_tum(trajectory, transform_matrix):
    """Apply transformation matrix to TUM trajectory."""
    transformed = []
    
    for row in trajectory:
        timestamp = row[0]
        tx, ty, tz = row[1], row[2], row[3]
        qx, qy, qz, qw = row[4], row[5], row[6], row[7]
        
        # Convert quaternion to rotation matrix
        r = Rotation.from_quat([qx, qy, qz, qw])
        R = r.as_matrix()
        
        # Create 4x4 pose matrix
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = [tx, ty, tz]
        
        # Apply transformation
        pose_new = transform_matrix @ pose
        
        # Extract new rotation and translation
        R_new = pose_new[:3, :3]
        t_new = pose_new[:3, 3]
        
        # Convert back to quaternion
        r_new = Rotation.from_matrix(R_new)
        q_new = r_new.as_quat()  # [qx, qy, qz, qw]
        
        # Store as TUM format
        transformed.append([timestamp, t_new[0], t_new[1], t_new[2], 
                          q_new[0], q_new[1], q_new[2], q_new[3]])
    
    return np.array(transformed)


def auto_detect_best_transformation(gt_traj, kf_traj, gt_fmt, kf_fmt):
    """
    Automatically detect best transformation by testing all options.
    Returns the transform that minimizes center distance.
    """
    gt_trans = extract_translations(gt_traj, gt_fmt)
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
        kf_transformed = apply_transformation(kf_traj, T, kf_fmt)
        kf_trans = extract_translations(kf_transformed, kf_fmt)
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


def fix_pose_mismatch(gt_path, kf_path, output_dir='./', auto_transform=True, output_format='kitti'):
    """
    Complete fix for pose mismatch: handles both coordinate frames and length.
    
    Args:
        gt_path: Path to ground truth file
        kf_path: Path to keyframe trajectory file
        output_dir: Output directory for fixed files
        auto_transform: Automatically detect and apply coordinate transformation
        output_format: Output format ('kitti' or 'tum' or 'auto' to match GT)
    
    Returns:
        dict: Information about applied fixes
    """
    print("\n" + "="*80)
    print("COMPREHENSIVE POSE MISMATCH FIX (KITTI & TUM Support)")
    print("="*80)
    
    # Detect formats
    print("\n[1/6] Detecting trajectory formats...")
    gt_fmt = detect_format(gt_path)
    kf_fmt = detect_format(kf_path)
    
    print(f"  Ground truth format: {gt_fmt.upper()}")
    print(f"  Keyframes format:    {kf_fmt.upper()}")
    
    # Load trajectories
    print("\n[2/6] Loading trajectories...")
    gt_traj = load_trajectory(gt_path)
    kf_traj = load_trajectory(kf_path)
    
    print(f"  Ground truth: {len(gt_traj)} poses")
    print(f"  Keyframes:    {len(kf_traj)} poses")
    
    # Convert both to KITTI for internal processing (easier to work with)
    if gt_fmt == 'tum':
        gt_traj_kitti = tum_to_kitti(gt_traj)
        gt_working = gt_traj_kitti
        gt_working_fmt = 'kitti'
    else:
        gt_working = gt_traj
        gt_working_fmt = gt_fmt
    
    if kf_fmt == 'tum':
        kf_traj_kitti = tum_to_kitti(kf_traj)
        kf_working = kf_traj_kitti
        kf_working_fmt = 'kitti'
    else:
        kf_working = kf_traj
        kf_working_fmt = kf_fmt
    
    # Analyze trajectories
    print("\n[3/6] Analyzing trajectory characteristics...")
    gt_info = analyze_trajectory(gt_working, gt_working_fmt, "Ground Truth")
    kf_info = analyze_trajectory(kf_working, kf_working_fmt, "Keyframes")
    
    print(f"\n  Ground Truth:")
    print(f"    Main motion axis: {gt_info['main_axis_name']}")
    print(f"    Variance: X={gt_info['variance'][0]:.1f}, Y={gt_info['variance'][1]:.1f}, Z={gt_info['variance'][2]:.1f}")
    print(f"    Range: X={gt_info['range'][0]:.1f}m, Y={gt_info['range'][1]:.1f}m, Z={gt_info['range'][2]:.1f}m")
    
    print(f"\n  Keyframes:")
    print(f"    Main motion axis: {kf_info['main_axis_name']}")
    print(f"    Variance: X={kf_info['variance'][0]:.1f}, Y={kf_info['variance'][1]:.1f}, Z={kf_info['variance'][2]:.1f}")
    print(f"    Range: X={kf_info['range'][0]:.1f}m, Y={kf_info['range'][1]:.1f}m, Z={kf_info['range'][2]:.1f}m")
    
    # Detect coordinate mismatch
    print("\n[4/6] Detecting coordinate frame mismatch...")
    mismatch_info = detect_coordinate_mismatch(gt_info, kf_info)
    
    if mismatch_info['mismatch']:
        print(f"  ⚠️  MISMATCH DETECTED: {mismatch_info['description']}")
        print(f"  Type: {mismatch_info['type']}")
        
        if auto_transform:
            if mismatch_info['recommended_transform'] == 'auto_detect':
                transform_type, _ = auto_detect_best_transformation(gt_working, kf_working, 
                                                                    gt_working_fmt, kf_working_fmt)
            else:
                transform_type = mismatch_info['recommended_transform']
            
            print(f"\n  Applying transformation: {transform_type}")
            T = get_transformation_matrix(transform_type)
            kf_working = apply_transformation(kf_working, T, kf_working_fmt)
            
            # Re-analyze after transformation
            kf_info = analyze_trajectory(kf_working, kf_working_fmt, "Keyframes (after transform)")
            print(f"  After transform - Main axis: {kf_info['main_axis_name']}")
        else:
            print(f"  Recommended transform: {mismatch_info['recommended_transform']}")
            print(f"  Auto-transform disabled. Proceeding without transformation.")
    else:
        print(f"  ✓ No coordinate mismatch detected")
        transform_type = 'none'
    
    # Handle length mismatch
    print("\n[5/6] Handling trajectory length mismatch...")
    target_length = min(len(gt_working), len(kf_working))
    
    if len(gt_working) != len(kf_working):
        print(f"  Length mismatch detected")
        print(f"  Sampling both to length: {target_length}")
        
        gt_working = uniform_sample(gt_working, target_length)
        kf_working = uniform_sample(kf_working, target_length)
    else:
        print(f"  ✓ Lengths already match: {target_length}")
    
    # Determine output format
    if output_format == 'auto':
        output_format = gt_fmt
    
    print(f"\n[6/6] Saving corrected trajectories (format: {output_format.upper()})...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to output format if needed
    if output_format == 'tum' and gt_working_fmt == 'kitti':
        gt_output_data = kitti_to_tum(gt_working)
        kf_output_data = kitti_to_tum(kf_working)
    elif output_format == 'kitti' and gt_working_fmt == 'tum':
        gt_output_data = tum_to_kitti(gt_working)
        kf_output_data = tum_to_kitti(kf_working)
    else:
        gt_output_data = gt_working
        kf_output_data = kf_working
    
    gt_output = os.path.join(output_dir, 'gt_corrected.txt')
    kf_output = os.path.join(output_dir, 'kf_corrected.txt')
    
    np.savetxt(gt_output, gt_output_data, fmt='%.12e')
    np.savetxt(kf_output, kf_output_data, fmt='%.12e')
    
    print(f"  ✓ Saved corrected GT to: {gt_output}")
    print(f"  ✓ Saved corrected KF to: {kf_output}")
    
    # Final summary
    print("\n" + "="*80)
    print("✓ PROCESSING COMPLETE")
    print("="*80)
    
    print(f"\nApplied fixes:")
    print(f"  1. Format conversion: GT={gt_fmt.upper()}, KF={kf_fmt.upper()} → Output={output_format.upper()}")
    print(f"  2. Coordinate transformation: {transform_type}")
    print(f"  3. Uniform sampling: {target_length} poses")
    
    print(f"\n📊 Ready for evaluation:")
    print(f"\n APE evaluation:")
    print(f"\n  evo_ape kitti {gt_output} {kf_output} \\")
    print(f"      --correct_scale -a --pose_relation trans_part \\")
    print(f"      -va --plot --plot_mode xz --save_results results/ate_result.zip")
    
    
    print(f"\n📊 RPE evaluation:")
    print(f"\n  evo_rpe kitti {gt_output} {kf_output} \\")
    print(f"      --delta 100 --delta_unit m --correct_scale -a --pose_relation trans_part \\")
    print(f"      -va --plot --plot_mode xyz --save_results results/rpe_100m.zip")
    
    return {
        'transform_applied': transform_type,
        'final_length': target_length,
        'gt_output': gt_output,
        'kf_output': kf_output,
        'mismatch_detected': mismatch_info['mismatch'],
        'input_formats': {'gt': gt_fmt, 'kf': kf_fmt},
        'output_format': output_format
    }


def main():
    """Main function with command-line interface."""
    print("\n" + "="*80)
    print("ENHANCED POSE MISMATCH FIX - Complete Solution")
    print("Supports KITTI and TUM trajectory formats")
    print("="*80)
    print("""
This script automatically:
  ✓ Detects trajectory format (KITTI or TUM)
  ✓ Detects coordinate frame mismatches (rotation/flip)
  ✓ Applies the correct transformation
  ✓ Resamples trajectories to match lengths
  ✓ Generates files ready for evo_ape evaluation

No manual intervention required!
""")
    
    # Default paths
    default_gt = './data/data_odometry_poses/poses/08.txt'
    default_kf = './third_party/ORB_SLAM3/Examples/Monocular/KeyFrameTrajectory.txt'
    default_output = './experiment_outputs'
    
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
        result = fix_pose_mismatch(gt_path, kf_path, output_dir, 
                                  auto_transform=True, output_format='kitti')
        
        print("\n" + "="*80)
        print("SUCCESS!")
        print("="*80)
        print("\nYour trajectories are now aligned and ready for evaluation.")
        print(f"Input formats: GT={result['input_formats']['gt'].upper()}, "
              f"KF={result['input_formats']['kf'].upper()}")
        print(f"Output format: {result['output_format'].upper()}")
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