#!/usr/bin/env python3
"""
Analyze single RPE result - 100m only.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

def analyze_rpe_results(results_dir='results', file='', title_suffix=''):
    """Analyze 100m RPE result."""
    
    if title_suffix:
        title_suffix = f' - {title_suffix}'
    
    print("\n" + "="*80)
    print(f"RPE ANALYSIS - 100m TRAJECTORY SEGMENT{title_suffix}")
    print("="*80)
    
    # Path to 100m results
    rpe_100m_dir = os.path.join(results_dir, file)
    stats_file = os.path.join(rpe_100m_dir, 'stats.json')
    error_file = os.path.join(rpe_100m_dir, 'error_array.npy')
    
    print(f"\nLooking in: {rpe_100m_dir}")
    
    if not os.path.isdir(rpe_100m_dir):
        print(f"✗ Directory not found: {rpe_100m_dir}")
        return False
    
    if not os.path.exists(stats_file):
        print(f"✗ stats.json not found: {stats_file}")
        return False
    
    # Load statistics
    with open(stats_file, 'r') as f:
        stats = json.load(f)
    
    print(f"✓ Loaded stats.json")
    
    # Display statistics
    print("\n" + "-"*80)
    print("RPE STATISTICS (100m Trajectory Segment)")
    print("-"*80)
    
    print("\nMetrics:")
    for key, value in sorted(stats.items()):
        if isinstance(value, (int, float)):
            print(f"  {key:20s}: {value:12.6f}")
        else:
            print(f"  {key:20s}: {value}")
    
    # Load error array
    if os.path.exists(error_file):
        errors = np.load(error_file)
        
        print("\n" + "-"*80)
        print("ERROR ARRAY ANALYSIS")
        print("-"*80)
        print(f"\nError array shape: {errors.shape}")
        print(f"\nStatistics:")
        print(f"  Min:    {np.min(errors):.6f} m")
        print(f"  Max:    {np.max(errors):.6f} m")
        print(f"  Mean:   {np.mean(errors):.6f} m")
        print(f"  Median: {np.median(errors):.6f} m")
        print(f"  Std:    {np.std(errors):.6f} m")
        
        # Create plots
        create_plots(errors, rpe_100m_dir, title_suffix)
    else:
        print(f"✗ error_array.npy not found")
        return False
    
    return True


def create_plots(errors, output_dir, title_suffix=''):
    """Create 4-panel visualization."""
    
    print("\n" + "-"*80)
    print("GENERATING PLOTS")
    print("-"*80)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'RPE (Relative Pose Error) - 100m Trajectory Segment{title_suffix}', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Error timeline
    ax = axes[0, 0]
    ax.plot(errors, 'b-', linewidth=1)
    ax.set_xlabel('Pose Index')
    ax.set_ylabel('RPE (m)')
    ax.set_title(f'RPE Over Pose Sequence{title_suffix}')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Histogram
    ax = axes[0, 1]
    ax.hist(errors, bins=50, color='green', alpha=0.7, edgecolor='black')
    ax.set_xlabel('RPE (m)')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Error Distribution{title_suffix}')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Cumulative distribution
    ax = axes[1, 0]
    sorted_errors = np.sort(errors)
    cumsum = np.cumsum(sorted_errors) / np.sum(sorted_errors) * 100
    ax.plot(sorted_errors, cumsum, 'r-', linewidth=2)
    ax.set_xlabel('RPE (m)')
    ax.set_ylabel('Cumulative Percentage (%)')
    ax.set_title(f'Cumulative Error Distribution{title_suffix}')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Statistics box
    ax = axes[1, 1]
    ax.axis('off')
    
    mean_error = np.mean(errors)
    if mean_error < 0.05:
        perf = "✓ EXCELLENT"
    elif mean_error < 0.1:
        perf = "✓ GOOD"
    elif mean_error < 0.5:
        perf = "⚠️ ACCEPTABLE"
    elif mean_error < 1.0:
        perf = "⚠️ POOR"
    else:
        perf = "❌ VERY POOR"
    
    stats_text = f"""
RPE Statistics Summary (100m){title_suffix}

Mean Error:        {np.mean(errors):.4f} m
Median Error:      {np.median(errors):.4f} m
Std Deviation:     {np.std(errors):.4f} m
RMSE:              {np.sqrt(np.mean(errors**2)):.4f} m

Min Error:         {np.min(errors):.4f} m
Max Error:         {np.max(errors):.4f} m
Range:             {np.max(errors) - np.min(errors):.4f} m

Total Poses:       {len(errors)}

Performance Level:
{perf}
    """
    
    ax.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
            verticalalignment='center', 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plot_path = os.path.join(output_dir, 'rpe_100m_analysis.png')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {plot_path}")
    plt.close()


def print_interpretation():
    """Print RPE interpretation."""
    
    print("\n" + "="*80)
    print("RPE INTERPRETATION GUIDE")
    print("="*80)
    
    text = """
What is RPE?
  RPE = Relative Pose Error
  Measures: Local tracking accuracy between poses 100m apart
  Different from APE: APE measures cumulative global drift
                      RPE measures short-term local accuracy

Performance Benchmarks (per 100m segment):
  ✓ < 0.05 m (5 cm):     Excellent - production quality
  ✓ 0.05 - 0.1 m:        Good - acceptable for most uses
  ⚠️ 0.1 - 0.5 m:        Acceptable - needs improvement
  ⚠️ 0.5 - 1.0 m:        Poor - significant local errors
  ❌ > 1.0 m:            Very poor - not suitable

Why RPE Matters:
  • Shows if your SLAM tracks LOCALLY well
  • Even with high APE (drift), good RPE means reliable odometry
  • Good RPE + high APE = loop closure problem (not tracking)
  • Bad RPE + bad APE = fundamental tracking issue

Comparison with Your APE:
  • Your APE (global): 234.36 m (very poor)
  • Your RPE (local):  See above ↑
  
  If RPE is much better than APE:
    → Your SLAM tracks well locally but drifts globally
    → Loop closure detection is your main issue
  
  If RPE is also poor:
    → Your SLAM has fundamental tracking problems
    → Feature quality or initialization issues
    """
    
    print(text)


if __name__ == '__main__':
    results_dir = 'results'
    title_suffix = ''
    
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    
    if len(sys.argv) > 2:
        title_suffix = sys.argv[2]
    
    success = analyze_rpe_results(results_dir, file='', title_suffix=title_suffix)
    
    if success:
        print_interpretation()
        print("\n" + "="*80)
        print("✓ RPE ANALYSIS COMPLETE")
        print("="*80 + "\n")
    else:
        print("\n✗ Analysis failed")
        sys.exit(1)