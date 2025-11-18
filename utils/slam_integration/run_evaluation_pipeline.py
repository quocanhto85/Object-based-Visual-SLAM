#!/usr/bin/env python3
"""
Complete Evaluation Pipeline for Object-based Visual SLAM

This script runs the complete evaluation workflow:
1. Load ground truth and predicted cuboids
2. Calculate per-frame 3D IoU
3. Compute detection metrics (mAP, precision, recall)
4. Generate visualizations
5. Compare before/after ByteTrack (if available)
6. Export comprehensive report

Usage:
    python run_evaluation_pipeline.py --gt_dir /path/to/ground_truth \
                                      --pred_dir /path/to/predicted \
                                      --output_dir /path/to/output \
                                      --sequence 08
"""

import argparse
import os
import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime

from evaluate_3d_detection import (
    evaluate_sequence, plot_evaluation_results, compare_before_after_bytetrack
)


def check_directories(gt_dir, pred_dir, output_dir):
    """
    Validate input directories and create output directory.
    
    Args:
        gt_dir: Ground truth directory
        pred_dir: Predicted cuboids directory
        output_dir: Output directory
        
    Returns:
        bool: True if valid, False otherwise
    """
    if not os.path.exists(gt_dir):
        print(f"Error: Ground truth directory not found: {gt_dir}")
        return False
    
    if not os.path.exists(pred_dir):
        print(f"Error: Predicted cuboids directory not found: {pred_dir}")
        return False
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"✓ Output directory: {output_dir}")
    
    return True


def generate_markdown_report(results, output_path):
    """
    Generate a comprehensive markdown report.
    
    Args:
        results: Evaluation results dictionary
        output_path: Path to save the markdown report
    """
    summary = results['summary']
    sequence = results['sequence']
    
    report = f"""# Object-based Visual SLAM Evaluation Report

## Sequence Information
- **Sequence**: KITTI {sequence}
- **Total Frames**: {summary['total_frames']}
- **Evaluation Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 3D IoU Statistics

| Metric | Value |
|--------|-------|
| Mean IoU | {summary['mean_iou']:.4f} |
| Median IoU | {summary['median_iou']:.4f} |
| Std Deviation | {summary['std_iou']:.4f} |
| Min IoU | {summary['min_iou']:.4f} |
| Max IoU | {summary['max_iou']:.4f} |

---

## Detection Performance

### Overall Metrics

| Metric | Value |
|--------|-------|
| True Positives | {summary['total_tp']} |
| False Positives | {summary['total_fp']} |
| False Negatives | {summary['total_fn']} |
| **Precision** | **{summary['overall_precision']:.4f}** |
| **Recall** | **{summary['overall_recall']:.4f}** |
| **F1-Score** | **{summary['overall_f1']:.4f}** |

### Mean Average Precision (mAP)

Performance at different IoU thresholds:

| IoU Threshold | mAP | Recall | F1-Score |
|---------------|-----|--------|----------|
| 0.25 (Easy) | {summary['mAP@0.25']:.4f} | {summary['Recall@0.25']:.4f} | {summary['F1@0.25']:.4f} |
| 0.50 (Moderate) | {summary['mAP@0.50']:.4f} | {summary['Recall@0.50']:.4f} | {summary['F1@0.50']:.4f} |
| 0.70 (Hard) | {summary['mAP@0.70']:.4f} | {summary['Recall@0.70']:.4f} | {summary['F1@0.70']:.4f} |

---

## Interpretation

### IoU Quality Assessment
"""
    
    mean_iou = summary['mean_iou']
    if mean_iou >= 0.7:
        report += "- ✅ **Excellent**: Mean IoU > 0.70 indicates very accurate 3D localization\n"
    elif mean_iou >= 0.5:
        report += "- ✓ **Good**: Mean IoU 0.50-0.70 shows solid detection accuracy\n"
    elif mean_iou >= 0.3:
        report += "- ⚠ **Moderate**: Mean IoU 0.30-0.50 suggests room for improvement\n"
    else:
        report += "- ❌ **Poor**: Mean IoU < 0.30 indicates significant localization errors\n"
    
    report += "\n### Detection Quality Assessment\n"
    
    precision = summary['overall_precision']
    recall = summary['overall_recall']
    
    if precision >= 0.8 and recall >= 0.8:
        report += "- ✅ **Excellent**: High precision and recall indicate robust detection\n"
    elif precision >= 0.6 and recall >= 0.6:
        report += "- ✓ **Good**: Balanced precision and recall\n"
    else:
        if precision < 0.6:
            report += "- ⚠ **Low Precision**: Many false positives (over-detection)\n"
        if recall < 0.6:
            report += "- ⚠ **Low Recall**: Many false negatives (missed detections)\n"
    
    report += f"""
---

## Recommendations

"""
    
    # Add recommendations based on results
    if summary['mean_iou'] < 0.5:
        report += "1. **Improve 3D Localization**: Consider refining cuboid estimation or camera calibration\n"
    
    if summary['overall_precision'] < 0.7:
        report += "2. **Reduce False Positives**: Increase detection confidence threshold\n"
    
    if summary['overall_recall'] < 0.7:
        report += "3. **Improve Detection Coverage**: Lower confidence threshold or enhance object detection\n"
    
    if summary['std_iou'] > 0.2:
        report += "4. **Reduce IoU Variance**: Improve consistency across frames\n"
    
    report += """
---

## Files Generated

- `evaluation_results_{sequence}.json` - Complete numerical results
- `evaluation_plots_{sequence}.png` - Visualization plots
- `evaluation_report_{sequence}.md` - This report

---

*Generated by Object-based Visual SLAM Evaluation Pipeline*
"""
    
    # Save report
    with open(output_path, 'w') as f:
        f.write(report)
    
    print(f"✓ Report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Run complete evaluation pipeline for Object-based Visual SLAM'
    )
    
    parser.add_argument('--gt_dir', type=str, required=True,
                       help='Directory containing ground truth cuboid JSON files')
    parser.add_argument('--pred_dir', type=str, required=True,
                       help='Directory containing predicted cuboid JSON files from ORB-SLAM3')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory for saving evaluation results')
    parser.add_argument('--sequence', type=str, default='08',
                       help='KITTI sequence name (default: 08)')
    parser.add_argument('--pred_dir_before', type=str, default=None,
                       help='Optional: Predictions before ByteTrack for comparison')
    parser.add_argument('--generate_report', action='store_true',
                       help='Generate markdown report (default: True)')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print(" Object-based Visual SLAM - Evaluation Pipeline")
    print("="*70 + "\n")
    
    # Validate directories
    if not check_directories(args.gt_dir, args.pred_dir, args.output_dir):
        return 1
    
    print(f"\nConfiguration:")
    print(f"  Ground Truth: {args.gt_dir}")
    print(f"  Predictions:  {args.pred_dir}")
    print(f"  Output:       {args.output_dir}")
    print(f"  Sequence:     {args.sequence}")
    
    # Run main evaluation
    print("\n" + "="*70)
    print("Step 1: Running Main Evaluation")
    print("="*70)
    
    results = evaluate_sequence(
        args.gt_dir,
        args.pred_dir,
        args.output_dir,
        args.sequence
    )
    
    if not results:
        print("Error: Evaluation failed")
        return 1
    
    # Generate plots
    print("\n" + "="*70)
    print("Step 2: Generating Visualizations")
    print("="*70 + "\n")
    
    plot_evaluation_results(results, args.output_dir)
    
    # Generate report
    if args.generate_report:
        print("\n" + "="*70)
        print("Step 3: Generating Report")
        print("="*70 + "\n")
        
        report_path = os.path.join(
            args.output_dir, 
            f"evaluation_report_{args.sequence}.md"
        )
        generate_markdown_report(results, report_path)
    
    # Optional: Compare before/after ByteTrack
    if args.pred_dir_before:
        print("\n" + "="*70)
        print("Step 4: ByteTrack Ablation Study")
        print("="*70 + "\n")
        
        results_before = evaluate_sequence(
            args.gt_dir,
            args.pred_dir_before,
            args.output_dir,
            args.sequence + "_before"
        )
        
        if results_before:
            compare_before_after_bytetrack(
                results_before,
                results,
                args.output_dir
            )
    
    # Print summary
    print("\n" + "="*70)
    print("✓ EVALUATION COMPLETE")
    print("="*70)
    
    summary = results['summary']
    print(f"\nKey Metrics:")
    print(f"  Mean IoU:  {summary['mean_iou']:.4f}")
    print(f"  Precision: {summary['overall_precision']:.4f}")
    print(f"  Recall:    {summary['overall_recall']:.4f}")
    print(f"  F1-Score:  {summary['overall_f1']:.4f}")
    print(f"  mAP@0.50:  {summary['mAP@0.50']:.4f}")
    
    print(f"\nResults saved to: {args.output_dir}")
    print("\nGenerated files:")
    print(f"  - evaluation_results_{args.sequence}.json")
    print(f"  - evaluation_plots_{args.sequence}.png")
    if args.generate_report:
        print(f"  - evaluation_report_{args.sequence}.md")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())