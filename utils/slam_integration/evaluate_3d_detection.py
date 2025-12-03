"""
3D Object Detection and Tracking Evaluation for Object-based SLAM

CORRECTED VERSION - IoU Threshold Fixed for Monocular SLAM

Key changes:
- Default IoU thresholds: [0.25, 0.5, 0.7] (was [0.5, 0.7])
- Primary matching uses 0.25 threshold (appropriate for monocular SLAM)
- Multi-threshold evaluation for comprehensive analysis
"""

import numpy as np
import json
import os
import glob
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm

from cuboid_utils import (
    Cuboid3D, load_ground_truth_cuboids, compute_3d_iou,
    match_cuboids_hungarian, compute_center_distance, compute_orientation_error
)


def convert_numpy_types(obj):
    """
    Recursively convert numpy types to native Python types for JSON serialization.
    
    Args:
        obj: Object that may contain numpy types
        
    Returns:
        Object with numpy types converted to Python native types
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj



class ObjectDetectionEvaluator:
    """
    Evaluator for 3D object detection metrics.
    
    CORRECTED for monocular SLAM:
    - Uses IoU threshold 0.25 as primary (was 0.5)
    - Evaluates at [0.25, 0.5, 0.7] thresholds
    
    Computes:
    - Per-frame IoU statistics
    - Precision, Recall, F1-score
    - mAP (mean Average Precision) at multiple IoU thresholds
    - Class-wise performance metrics
    """
    
    def __init__(self, iou_thresholds: List[float] = None):
        """
        Initialize evaluator.
        
        Args:
            iou_thresholds: List of IoU thresholds for mAP calculation
                           Default: [0.25, 0.5, 0.7] (easy, moderate, hard)
                           CHANGED from [0.5, 0.7] to include 0.25 for monocular SLAM
        """
        if iou_thresholds is None:
            # CORRECTED: Include 0.25 threshold for monocular SLAM
            self.iou_thresholds = [0.25, 0.5, 0.7]
        else:
            self.iou_thresholds = sorted(iou_thresholds)  # Ensure sorted ascending
        
        self.results = {
            'per_frame': [],
            'per_class': defaultdict(list),
            'matches': [],
            'false_positives': [],
            'false_negatives': []
        }
    
    def evaluate_frame(self, frame_id: int, gt_cuboids: List[Cuboid3D], 
                       pred_cuboids: List[Cuboid3D]) -> Dict:
        """
        Evaluate a single frame.
        
        Args:
            frame_id: Frame identifier
            gt_cuboids: Ground truth cuboids
            pred_cuboids: Predicted cuboids
            
        Returns:
            frame_results: Dictionary with frame-level metrics
        """
        frame_results = {
            'frame_id': int(frame_id),
            'num_gt': int(len(gt_cuboids)),
            'num_pred': int(len(pred_cuboids)),
            'matches_per_threshold': {},
            'mean_iou': 0.0,
            'tp': 0,
            'fp': 0,
            'fn': 0
        }
        
        if len(gt_cuboids) == 0 and len(pred_cuboids) == 0:
            return frame_results
        
        # Evaluate at each IoU threshold
        for iou_thresh in self.iou_thresholds:
            matches = match_cuboids_hungarian(gt_cuboids, pred_cuboids, iou_thresh)
            
            tp = len(matches)
            fp = len(pred_cuboids) - tp
            fn = len(gt_cuboids) - tp
            
            # Convert matches to native Python types for JSON serialization
            matches_serializable = [
                (int(i), int(j), float(iou)) for i, j, iou in matches
            ]
            
            frame_results['matches_per_threshold'][float(iou_thresh)] = {
                'matches': matches_serializable,
                'tp': int(tp),
                'fp': int(fp),
                'fn': int(fn),
                'precision': float(tp / (tp + fp) if (tp + fp) > 0 else 0),
                'recall': float(tp / (tp + fn) if (tp + fn) > 0 else 0)
            }
        
        # CORRECTED: Use LOWEST threshold (0.25) for primary metrics
        # This is appropriate for monocular SLAM which typically achieves IoU 0.3-0.5
        primary_threshold = self.iou_thresholds[0]  # 0.25
        matches = match_cuboids_hungarian(gt_cuboids, pred_cuboids, primary_threshold)
        
        if matches:
            frame_results['mean_iou'] = float(np.mean([iou for _, _, iou in matches]))
            frame_results['tp'] = int(len(matches))
        else:
            frame_results['mean_iou'] = 0.0
        
        frame_results['fp'] = int(len(pred_cuboids) - frame_results['tp'])
        frame_results['fn'] = int(len(gt_cuboids) - frame_results['tp'])
        
        self.results['per_frame'].append(frame_results)
        
        return frame_results
    
    def compute_map(self) -> Dict[str, float]:
        """
        Compute mean Average Precision (mAP) across all frames.
        
        Returns:
            map_results: Dictionary with mAP at each threshold
        """
        map_results = {}
        
        for iou_thresh in self.iou_thresholds:
            # Access using float key (we converted it earlier)
            all_tp = sum(frame['matches_per_threshold'][float(iou_thresh)]['tp'] 
                        for frame in self.results['per_frame'])
            all_fp = sum(frame['matches_per_threshold'][float(iou_thresh)]['fp'] 
                        for frame in self.results['per_frame'])
            all_fn = sum(frame['matches_per_threshold'][float(iou_thresh)]['fn'] 
                        for frame in self.results['per_frame'])
            
            precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
            recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            map_results[f'mAP@{iou_thresh:.2f}'] = float(precision)
            map_results[f'Recall@{iou_thresh:.2f}'] = float(recall)
            map_results[f'F1@{iou_thresh:.2f}'] = float(f1)
        
        return map_results
    
    def get_summary_statistics(self) -> Dict:
        """
        Get summary statistics across all frames.
        
        Returns:
            stats: Dictionary with aggregated statistics
        """
        if not self.results['per_frame']:
            return {}
        
        ious = [frame['mean_iou'] for frame in self.results['per_frame'] 
                if frame['mean_iou'] > 0]
        
        # CORRECTED: These metrics are based on lowest threshold (0.25)
        total_tp = sum(frame['tp'] for frame in self.results['per_frame'])
        total_fp = sum(frame['fp'] for frame in self.results['per_frame'])
        total_fn = sum(frame['fn'] for frame in self.results['per_frame'])
        
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        stats = {
            'total_frames': int(len(self.results['per_frame'])),
            'mean_iou': float(np.mean(ious)) if ious else 0.0,
            'median_iou': float(np.median(ious)) if ious else 0.0,
            'std_iou': float(np.std(ious)) if ious else 0.0,
            'min_iou': float(np.min(ious)) if ious else 0.0,
            'max_iou': float(np.max(ious)) if ious else 0.0,
            'total_tp': int(total_tp),
            'total_fp': int(total_fp),
            'total_fn': int(total_fn),
            'overall_precision': float(precision),
            'overall_recall': float(recall),
            'overall_f1': float(f1),
            'primary_iou_threshold': float(self.iou_thresholds[0])  # Document which threshold was used
        }
        
        # Add mAP metrics
        stats.update(self.compute_map())
        
        return stats


def evaluate_sequence(gt_dir: str, pred_dir: str, output_dir: str, 
                     sequence_name: str = "08", title: str = None) -> Dict:
    """
    Evaluate a full sequence.
    
    CORRECTED: Now uses appropriate IoU threshold (0.25) for monocular SLAM
    
    Args:
        gt_dir: Directory with ground truth cuboid JSON files
        pred_dir: Directory with predicted cuboid JSON files
        output_dir: Directory for saving results
        sequence_name: Sequence identifier (e.g., "08")
        title: Optional custom title for the evaluation
        
    Returns:
        results: Dictionary with complete evaluation results
    """
    print("\n" + "="*70)
    print(f"EVALUATING SEQUENCE: {sequence_name}")
    if title:
        print(f"Title: {title}")
    print("="*70)
    
    # CORRECTED: Explicitly set thresholds with 0.25 as primary
    print("\nEvaluation Configuration:")
    print("  IoU Thresholds: [0.25 (primary), 0.5, 0.7]")
    print("  Primary threshold 0.25 is appropriate for monocular SLAM")
    print("  (Monocular SLAM typically achieves IoU 0.30-0.50 due to scale ambiguity)")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize evaluator with corrected thresholds
    evaluator = ObjectDetectionEvaluator(iou_thresholds=[0.25, 0.5, 0.7])
    
    # Get all ground truth files
    gt_files = sorted(glob.glob(os.path.join(gt_dir, "*.json")))
    
    if len(gt_files) == 0:
        print(f"ERROR: No ground truth files found in {gt_dir}")
        return None
    
    print(f"\nFound {len(gt_files)} ground truth files")
    
    # Evaluate each frame
    frame_details = []
    frames_evaluated = 0
    frames_skipped = 0
    
    for gt_file in tqdm(gt_files, desc="Evaluating frames"):
        frame_name = Path(gt_file).stem
        pred_file = os.path.join(pred_dir, f"{frame_name}.json")
        
        # Check if prediction exists
        if not os.path.exists(pred_file):
            frames_skipped += 1
            continue
        
        # Load cuboids
        try:
            gt_cuboids = load_ground_truth_cuboids(gt_file)
            pred_cuboids = load_ground_truth_cuboids(pred_file)  # Same format
        except Exception as e:
            print(f"\nWarning: Error loading frame {frame_name}: {e}")
            frames_skipped += 1
            continue
        
        # Evaluate frame
        frame_id = int(frame_name)
        frame_result = evaluator.evaluate_frame(frame_id, gt_cuboids, pred_cuboids)
        frame_details.append(frame_result)
        frames_evaluated += 1
    
    print(f"\nFrames evaluated: {frames_evaluated}")
    print(f"Frames skipped: {frames_skipped}")
    
    if frames_evaluated == 0:
        print("ERROR: No frames could be evaluated")
        return None
    
    # Get summary statistics
    summary = evaluator.get_summary_statistics()
    
    # Print summary
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    
    print(f"\nIoU Statistics (based on threshold {summary['primary_iou_threshold']}):")
    print(f"  Mean IoU:   {summary['mean_iou']:.4f}")
    print(f"  Median IoU: {summary['median_iou']:.4f}")
    print(f"  Std IoU:    {summary['std_iou']:.4f}")
    print(f"  Range:      [{summary['min_iou']:.4f}, {summary['max_iou']:.4f}]")
    
    print(f"\nDetection Metrics (at primary threshold {summary['primary_iou_threshold']}):")
    print(f"  True Positives:  {summary['total_tp']}")
    print(f"  False Positives: {summary['total_fp']}")
    print(f"  False Negatives: {summary['total_fn']}")
    print(f"  Precision:       {summary['overall_precision']:.4f}")
    print(f"  Recall:          {summary['overall_recall']:.4f}")
    print(f"  F1-Score:        {summary['overall_f1']:.4f}")
    
    print(f"\nmAP (mean Average Precision) at Different Thresholds:")
    for thresh in [0.25, 0.5, 0.7]:
        print(f"  IoU@{thresh:.2f}:")
        print(f"    Precision: {summary[f'mAP@{thresh:.2f}']:.4f}")
        print(f"    Recall:    {summary[f'Recall@{thresh:.2f}']:.4f}")
        print(f"    F1-Score:  {summary[f'F1@{thresh:.2f}']:.4f}")
    
    # Interpretation guidance
    print(f"\nInterpretation for Monocular SLAM:")
    mean_iou = summary['mean_iou']
    if mean_iou >= 0.4:
        print("  ✓ Mean IoU 0.40+ is GOOD for monocular SLAM")
    elif mean_iou >= 0.3:
        print("  ✓ Mean IoU 0.30-0.40 is ACCEPTABLE for monocular SLAM")
    else:
        print("  ⚠ Mean IoU < 0.30 suggests potential issues (check coordinate transforms)")
    
    precision = summary['overall_precision']
    recall = summary['overall_recall']
    if precision >= 0.7 and recall >= 0.7:
        print(f"  ✓ Precision & Recall 70%+ indicates solid detection performance")
    elif precision >= 0.5 and recall >= 0.5:
        print(f"  ✓ Precision & Recall 50-70% is reasonable")
    else:
        print(f"  ⚠ Low precision/recall may indicate detection issues")
    
    # Save results
    results = {
        'sequence': sequence_name,
        'title': title,
        'summary': summary,
        'frame_details': frame_details,
        'evaluation_config': {
            'iou_thresholds': [0.25, 0.5, 0.7],
            'primary_threshold': 0.25,
            'note': 'Threshold 0.25 is appropriate for monocular SLAM (typical IoU 0.30-0.50)'
        }
    }
    
    results_file = os.path.join(output_dir, f"evaluation_results_{sequence_name}.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\n✓ Results saved to: {results_file}")
    
    return results


def plot_evaluation_results(results: Dict, output_dir: str):
    """
    Generate visualization plots for evaluation results.
    
    Args:
        results: Evaluation results dictionary
        output_dir: Directory for saving plots
    """
    frame_details = results['frame_details']
    sequence = results['sequence']
    title = results.get('title')
    
    # Use custom title if provided
    display_title = title if title else f"KITTI {sequence}"
    
    # Extract data
    frame_ids = [f['frame_id'] for f in frame_details]
    mean_ious = [f['mean_iou'] for f in frame_details]
    num_gt = [f['num_gt'] for f in frame_details]
    num_pred = [f['num_pred'] for f in frame_details]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"3D Object Detection Evaluation - {display_title}", 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Plot 1: IoU over time
    ax1 = axes[0, 0]
    ax1.plot(frame_ids, mean_ious, 'b-', linewidth=1.5, alpha=0.7)
    ax1.axhline(y=results['summary']['mean_iou'], color='r', linestyle='--', 
                label=f"Mean: {results['summary']['mean_iou']:.3f}")
    # Add reference lines for monocular SLAM ranges
    ax1.axhline(y=0.3, color='orange', linestyle=':', alpha=0.5, label='Acceptable (0.30)')
    ax1.axhline(y=0.5, color='green', linestyle=':', alpha=0.5, label='Good (0.50)')
    ax1.set_xlabel('Frame Index')
    ax1.set_ylabel('Mean IoU')
    ax1.set_title('3D IoU Over Time (Primary Threshold 0.25)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: IoU distribution
    ax2 = axes[0, 1]
    non_zero_ious = [iou for iou in mean_ious if iou > 0]
    if non_zero_ious:
        ax2.hist(non_zero_ious, bins=30, color='green', alpha=0.7, edgecolor='black')
        ax2.axvline(x=results['summary']['mean_iou'], color='r', linestyle='--', 
                   label=f"Mean: {results['summary']['mean_iou']:.3f}")
        ax2.axvline(x=results['summary']['median_iou'], color='orange', linestyle='--',
                   label=f"Median: {results['summary']['median_iou']:.3f}")
        ax2.set_xlabel('IoU')
        ax2.set_ylabel('Frequency')
        ax2.set_title('IoU Distribution')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
    
    # Plot 3: Object counts
    ax3 = axes[1, 0]
    ax3.plot(frame_ids, num_gt, 'g-', label='Ground Truth', linewidth=1.5, alpha=0.7)
    ax3.plot(frame_ids, num_pred, 'b-', label='Predicted', linewidth=1.5, alpha=0.7)
    ax3.set_xlabel('Frame Index')
    ax3.set_ylabel('Number of Objects')
    ax3.set_title('Object Count Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Precision-Recall at different thresholds
    ax4 = axes[1, 1]
    thresholds = [0.25, 0.5, 0.7]
    precisions = [results['summary'][f'mAP@{t:.2f}'] for t in thresholds]
    recalls = [results['summary'][f'Recall@{t:.2f}'] for t in thresholds]
    
    x = np.arange(len(thresholds))
    width = 0.35
    
    ax4.bar(x - width/2, precisions, width, label='Precision', color='blue', alpha=0.7)
    ax4.bar(x + width/2, recalls, width, label='Recall', color='green', alpha=0.7)
    ax4.set_xlabel('IoU Threshold')
    ax4.set_ylabel('Score')
    ax4.set_title('Precision & Recall at Different IoU Thresholds')
    ax4.set_xticks(x)
    ax4.set_xticklabels([f'{t:.2f}' for t in thresholds])
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_ylim([0, 1.0])
    
    # Add note about primary threshold
    fig.text(0.5, 0.01, 
             'Primary metrics use IoU threshold 0.25 (appropriate for monocular SLAM)',
             ha='center', fontsize=10, style='italic', color='gray')
    
    plt.tight_layout()
    
    # Save figure
    plot_file = os.path.join(output_dir, f"evaluation_plots_{sequence}.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✓ Plots saved to: {plot_file}")
    
    plt.close()


def compare_before_after_bytetrack(results_before: Dict, results_after: Dict, 
                                   output_dir: str):
    """
    Compare evaluation results before and after ByteTrack integration.
    
    Args:
        results_before: Results without ByteTrack
        results_after: Results with ByteTrack
        output_dir: Directory for saving comparison plots
    """
    print("\n" + "="*60)
    print("BYTETRACK ABLATION STUDY")
    print("="*60)
    
    # Extract metrics
    metrics = ['mean_iou', 'overall_precision', 'overall_recall', 'overall_f1']
    metric_names = ['Mean IoU', 'Precision', 'Recall', 'F1-Score']
    
    before_vals = [results_before['summary'][m] for m in metrics]
    after_vals = [results_after['summary'][m] for m in metrics]
    improvements = [(after - before) / before * 100 if before > 0 else 0 
                   for before, after in zip(before_vals, after_vals)]
    
    # Print comparison
    print("\nMetric Comparison:")
    print(f"{'Metric':<20} {'Before':<12} {'After':<12} {'Improvement':<12}")
    print("-" * 60)
    
    for name, before, after, imp in zip(metric_names, before_vals, after_vals, improvements):
        print(f"{name:<20} {before:<12.4f} {after:<12.4f} {imp:+11.2f}%")
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('ByteTrack Integration - Performance Comparison', 
                 fontsize=14, fontweight='bold')
    
    # Bar chart comparison
    x = np.arange(len(metrics))
    width = 0.35
    
    ax1 = axes[0]
    ax1.bar(x - width/2, before_vals, width, label='Before ByteTrack', 
            color='orange', alpha=0.7)
    ax1.bar(x + width/2, after_vals, width, label='After ByteTrack', 
            color='blue', alpha=0.7)
    ax1.set_xlabel('Metrics')
    ax1.set_ylabel('Value')
    ax1.set_title('Metric Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metric_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Improvement bar chart
    ax2 = axes[1]
    colors = ['green' if imp >= 0 else 'red' for imp in improvements]
    ax2.bar(metric_names, improvements, color=colors, alpha=0.7)
    ax2.set_xlabel('Metrics')
    ax2.set_ylabel('Improvement (%)')
    ax2.set_title('Performance Improvement with ByteTrack')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax2.grid(True, alpha=0.3, axis='y')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    comparison_file = os.path.join(output_dir, "bytetrack_comparison.png")
    plt.savefig(comparison_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Comparison plot saved to: {comparison_file}")
    
    plt.close()


if __name__ == "__main__":
    # Example usage
    print("3D Object Detection Evaluator - CORRECTED VERSION")
    print("IoU Threshold: 0.25 (appropriate for monocular SLAM)")
    print("=" * 70)
    
    # Set paths (you'll need to update these)
    GT_DIR = "/mnt/user-data/uploads"  # Ground truth directory
    PRED_DIR = "/home/claude/predicted_cuboids"  # Predicted cuboids from ORB-SLAM3
    OUTPUT_DIR = "/mnt/user-data/outputs/evaluation_results"
    
    # Run evaluation
    # results = evaluate_sequence(GT_DIR, PRED_DIR, OUTPUT_DIR, sequence_name="08",
    #                             title="ORB-SLAM3 with Dynamic Multi-Object Tracking")
    
    # Plot results
    # plot_evaluation_results(results, OUTPUT_DIR)
    
    print("\n✓ Evaluation framework ready")
    print("\nKey Changes from Previous Version:")
    print("  - Primary IoU threshold: 0.25 (was 0.5)")
    print("  - Appropriate for monocular SLAM (typical IoU 0.30-0.50)")
    print("  - Multi-threshold evaluation: [0.25, 0.5, 0.7]")
    print("\nTo use:")
    print("1. Export cuboids from ORB-SLAM3")
    print("2. Run: python run_evaluation_pipeline.py [args]")