"""
3D Cuboid Visualization Tool

Visualize ground truth and predicted cuboids for qualitative assessment.
Creates:
- Side-by-side 2D projections
- 3D bird's-eye view
- Individual frame comparisons
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import json
import os
from pathlib import Path
from typing import List, Dict
import argparse

from cuboid_utils import Cuboid3D, load_ground_truth_cuboids


def plot_cuboid_3d(ax, cuboid: Cuboid3D, color='blue', alpha=0.3, label=None):
    """
    Plot a 3D cuboid on a matplotlib 3D axes.
    
    Args:
        ax: Matplotlib 3D axes
        cuboid: Cuboid3D object
        color: Color for the cuboid
        alpha: Transparency (0-1)
        label: Optional label for legend
    """
    corners = cuboid.corners
    
    # Define the 12 edges of the cuboid
    edges = [
        [corners[0], corners[1]], [corners[1], corners[2]], 
        [corners[2], corners[3]], [corners[3], corners[0]],  # Bottom face
        [corners[4], corners[5]], [corners[5], corners[6]], 
        [corners[6], corners[7]], [corners[7], corners[4]],  # Top face
        [corners[0], corners[4]], [corners[1], corners[5]], 
        [corners[2], corners[6]], [corners[3], corners[7]]   # Vertical edges
    ]
    
    # Plot edges
    for edge in edges:
        xs = [edge[0][0], edge[1][0]]
        ys = [edge[0][1], edge[1][1]]
        zs = [edge[0][2], edge[1][2]]
        ax.plot(xs, ys, zs, color=color, linewidth=2, alpha=alpha*2)
    
    # Plot faces (optional, for better visibility)
    # Define 6 faces
    faces = [
        [corners[0], corners[1], corners[2], corners[3]],  # Bottom
        [corners[4], corners[5], corners[6], corners[7]],  # Top
        [corners[0], corners[1], corners[5], corners[4]],  # Front
        [corners[2], corners[3], corners[7], corners[6]],  # Back
        [corners[0], corners[3], corners[7], corners[4]],  # Left
        [corners[1], corners[2], corners[6], corners[5]]   # Right
    ]
    
    face_collection = Poly3DCollection(faces, alpha=alpha, 
                                      facecolor=color, edgecolor=color)
    ax.add_collection3d(face_collection)
    
    # Add center point
    ax.scatter(*cuboid.center, color=color, s=50, alpha=1.0)
    
    # Add label at center
    if label:
        ax.text(cuboid.center[0], cuboid.center[1], cuboid.center[2], 
               label, fontsize=8, color=color)


def plot_frame_comparison(gt_file: str, pred_file: str, output_file: str = None,
                         show_plot: bool = True):
    """
    Create comparison visualization for a single frame.
    
    Args:
        gt_file: Path to ground truth JSON file
        pred_file: Path to predicted cuboids JSON file
        output_file: Optional path to save the figure
        show_plot: Whether to display the plot
    """
    # Load cuboids
    gt_cuboids = load_ground_truth_cuboids(gt_file)
    pred_cuboids = load_ground_truth_cuboids(pred_file) if os.path.exists(pred_file) else []
    
    # Get frame ID
    frame_id = int(Path(gt_file).stem)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(18, 6))
    
    # 3D view - Ground Truth
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.set_title(f'Ground Truth - Frame {frame_id}', fontsize=12, fontweight='bold')
    
    for i, cuboid in enumerate(gt_cuboids):
        color = 'green' if cuboid.class_name == 'Car' else 'blue'
        label = f"{cuboid.class_name} {i+1}"
        plot_cuboid_3d(ax1, cuboid, color=color, alpha=0.3, label=label)
    
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.view_init(elev=20, azim=45)
    
    # 3D view - Predictions
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.set_title(f'Predictions - Frame {frame_id}', fontsize=12, fontweight='bold')
    
    for i, cuboid in enumerate(pred_cuboids):
        color = 'orange' if cuboid.class_name == 'Car' else 'purple'
        label = f"{cuboid.class_name} {i+1}"
        plot_cuboid_3d(ax2, cuboid, color=color, alpha=0.3, label=label)
    
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_zlabel('Z (m)')
    ax2.view_init(elev=20, azim=45)
    
    # Match axis limits
    all_corners = []
    for cuboid in gt_cuboids + pred_cuboids:
        all_corners.extend(cuboid.corners)
    
    if all_corners:
        all_corners = np.array(all_corners)
        x_min, y_min, z_min = np.min(all_corners, axis=0)
        x_max, y_max, z_max = np.max(all_corners, axis=0)
        
        for ax in [ax1, ax2]:
            ax.set_xlim([x_min - 5, x_max + 5])
            ax.set_ylim([y_min - 5, y_max + 5])
            ax.set_zlim([z_min - 2, z_max + 2])
    
    # Bird's-eye view - Overlay
    ax3 = fig.add_subplot(133)
    ax3.set_title(f'Bird\'s-Eye View - Frame {frame_id}', fontsize=12, fontweight='bold')
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Z (m)')
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal')
    
    # Plot ground truth (green)
    for i, cuboid in enumerate(gt_cuboids):
        corners_2d = cuboid.corners[:, [0, 2]]  # X-Z plane
        corners_2d = np.vstack([corners_2d[[0, 1, 2, 3, 0]], 
                               corners_2d[[4, 5, 6, 7, 4]]])
        ax3.plot(corners_2d[:5, 0], corners_2d[:5, 1], 
                'g-', linewidth=2, alpha=0.7, label='GT' if i == 0 else '')
        ax3.plot(corners_2d[5:, 0], corners_2d[5:, 1], 
                'g--', linewidth=1, alpha=0.5)
        ax3.scatter(cuboid.center[0], cuboid.center[2], 
                   color='green', s=100, marker='o', alpha=0.8)
    
    # Plot predictions (red)
    for i, cuboid in enumerate(pred_cuboids):
        corners_2d = cuboid.corners[:, [0, 2]]  # X-Z plane
        corners_2d = np.vstack([corners_2d[[0, 1, 2, 3, 0]], 
                               corners_2d[[4, 5, 6, 7, 4]]])
        ax3.plot(corners_2d[:5, 0], corners_2d[:5, 1], 
                'r-', linewidth=2, alpha=0.7, label='Pred' if i == 0 else '')
        ax3.plot(corners_2d[5:, 0], corners_2d[5:, 1], 
                'r--', linewidth=1, alpha=0.5)
        ax3.scatter(cuboid.center[0], cuboid.center[2], 
                   color='red', s=100, marker='x', alpha=0.8)
    
    # Add camera position
    ax3.scatter(0, 0, color='black', s=200, marker='^', 
               label='Camera', zorder=10)
    
    ax3.legend(loc='upper right')
    
    # Add statistics
    stats_text = f"GT Objects: {len(gt_cuboids)}\nPred Objects: {len(pred_cuboids)}"
    ax3.text(0.02, 0.98, stats_text, transform=ax3.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save if requested
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved visualization to: {output_file}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def create_video_visualization(gt_dir: str, pred_dir: str, output_dir: str,
                              start_frame: int = 0, end_frame: int = 100,
                              skip_frames: int = 1):
    """
    Create frame-by-frame visualizations for a sequence.
    
    Args:
        gt_dir: Ground truth directory
        pred_dir: Predictions directory
        output_dir: Output directory for images
        start_frame: Starting frame index
        end_frame: Ending frame index
        skip_frames: Skip every N frames (for faster processing)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nGenerating visualizations from frame {start_frame} to {end_frame}")
    print(f"Skip: {skip_frames}, Output: {output_dir}")
    
    frame_count = 0
    for frame_id in range(start_frame, end_frame + 1, skip_frames):
        gt_file = os.path.join(gt_dir, f"{frame_id:06d}.json")
        pred_file = os.path.join(pred_dir, f"{frame_id:06d}.json")
        output_file = os.path.join(output_dir, f"frame_{frame_id:06d}.png")
        
        if not os.path.exists(gt_file):
            continue
        
        try:
            plot_frame_comparison(gt_file, pred_file, output_file, show_plot=False)
            frame_count += 1
            
            if frame_count % 10 == 0:
                print(f"  Processed {frame_count} frames...")
        
        except Exception as e:
            print(f"  Error processing frame {frame_id}: {e}")
    
    print(f"\n✓ Generated {frame_count} visualization frames")
    print(f"  Location: {output_dir}")
    
    # Suggest ffmpeg command for creating video
    print("\nTo create a video, run:")
    print(f"  ffmpeg -framerate 10 -pattern_type glob -i '{output_dir}/frame_*.png' \\")
    print(f"         -c:v libx264 -pix_fmt yuv420p {output_dir}/visualization.mp4")


def plot_trajectory_with_cuboids(gt_dir: str, pred_dir: str, 
                                output_file: str = None,
                                max_frames: int = 500):
    """
    Plot trajectory and cuboid positions over time.
    
    Args:
        gt_dir: Ground truth directory
        pred_dir: Predictions directory
        output_file: Path to save figure
        max_frames: Maximum number of frames to plot
    """
    import glob
    
    gt_files = sorted(glob.glob(os.path.join(gt_dir, "*.json")))[:max_frames]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot 1: All cuboid centers (bird's-eye view)
    ax1.set_title('All Object Positions (Bird\'s-Eye View)', 
                 fontsize=12, fontweight='bold')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Z (m)')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Plot ground truth positions
    for gt_file in gt_files:
        gt_cuboids = load_ground_truth_cuboids(gt_file)
        for cuboid in gt_cuboids:
            color = 'green' if cuboid.class_name == 'Car' else 'blue'
            ax1.scatter(cuboid.center[0], cuboid.center[2], 
                       color=color, s=30, alpha=0.3)
    
    # Plot predicted positions
    for gt_file in gt_files:
        frame_id = int(Path(gt_file).stem)
        pred_file = os.path.join(pred_dir, f"{frame_id:06d}.json")
        
        if os.path.exists(pred_file):
            pred_cuboids = load_ground_truth_cuboids(pred_file)
            for cuboid in pred_cuboids:
                color = 'red' if cuboid.class_name == 'Car' else 'orange'
                ax1.scatter(cuboid.center[0], cuboid.center[2], 
                           color=color, s=20, alpha=0.2, marker='x')
    
    # Add camera at origin
    ax1.scatter(0, 0, color='black', s=200, marker='^', 
               label='Camera', zorder=10)
    
    # Plot 2: Object count over time
    ax2.set_title('Object Count Over Time', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Frame Index')
    ax2.set_ylabel('Number of Objects')
    ax2.grid(True, alpha=0.3)
    
    frame_ids = []
    gt_counts = []
    pred_counts = []
    
    for gt_file in gt_files:
        frame_id = int(Path(gt_file).stem)
        pred_file = os.path.join(pred_dir, f"{frame_id:06d}.json")
        
        gt_cuboids = load_ground_truth_cuboids(gt_file)
        pred_cuboids = load_ground_truth_cuboids(pred_file) if os.path.exists(pred_file) else []
        
        frame_ids.append(frame_id)
        gt_counts.append(len(gt_cuboids))
        pred_counts.append(len(pred_cuboids))
    
    ax2.plot(frame_ids, gt_counts, 'g-', linewidth=2, alpha=0.7, label='Ground Truth')
    ax2.plot(frame_ids, pred_counts, 'r-', linewidth=2, alpha=0.7, label='Predicted')
    ax2.legend()
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved trajectory plot to: {output_file}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize 3D cuboids for Object-based SLAM evaluation'
    )
    
    parser.add_argument('--mode', type=str, choices=['single', 'sequence', 'trajectory'],
                       default='single', help='Visualization mode')
    parser.add_argument('--gt_dir', type=str, required=True,
                       help='Ground truth cuboids directory')
    parser.add_argument('--pred_dir', type=str, required=True,
                       help='Predicted cuboids directory')
    parser.add_argument('--output_dir', type=str, default='./visualizations',
                       help='Output directory for saved visualizations')
    parser.add_argument('--frame', type=int, default=0,
                       help='Frame number for single frame mode')
    parser.add_argument('--start_frame', type=int, default=0,
                       help='Start frame for sequence mode')
    parser.add_argument('--end_frame', type=int, default=100,
                       help='End frame for sequence mode')
    parser.add_argument('--skip_frames', type=int, default=1,
                       help='Skip every N frames in sequence mode')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.mode == 'single':
        gt_file = os.path.join(args.gt_dir, f"{args.frame:06d}.json")
        pred_file = os.path.join(args.pred_dir, f"{args.frame:06d}.json")
        output_file = os.path.join(args.output_dir, f"frame_{args.frame:06d}.png")
        
        plot_frame_comparison(gt_file, pred_file, output_file, show_plot=True)
    
    elif args.mode == 'sequence':
        create_video_visualization(
            args.gt_dir, args.pred_dir, args.output_dir,
            args.start_frame, args.end_frame, args.skip_frames
        )
    
    elif args.mode == 'trajectory':
        output_file = os.path.join(args.output_dir, "trajectory_plot.png")
        plot_trajectory_with_cuboids(args.gt_dir, args.pred_dir, output_file)


if __name__ == "__main__":
    main()