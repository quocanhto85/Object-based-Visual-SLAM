#!/usr/bin/env python3
"""
Analyze EVO APE results from already-extracted files.
NO zip extraction needed. Works with: error_array.npy, stats.json, info.json
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import os

def analyze_ate_results(results_dir='results', title_suffix=''):
    """
    Analyze APE results from already-extracted files.
    Generates plots using matplotlib (no Qt needed).
    
    Args:
        results_dir: Directory containing error_array.npy, stats.json, info.json
        title_suffix: Optional suffix for plot titles (e.g., 'Case 1')
    """
    
    if title_suffix:
        title_suffix = f' - {title_suffix}'
    
    print("\n" + "="*80)
    print(f"APE RESULTS ANALYSIS (No Zip Extraction){title_suffix}")
    print("="*80)
    
    print(f"\nUsing results directory: {results_dir}")
    
    stats_file = os.path.join(results_dir, 'stats.json')
    info_file = os.path.join(results_dir, 'info.json')
    error_file = os.path.join(results_dir, 'error_array.npy')
    
    # Check files exist
    print("\n✓ Found files:")
    for fname in [stats_file, info_file, error_file]:
        if os.path.exists(fname):
            print(f"  ✓ {os.path.basename(fname)}")
        else:
            print(f"  ✗ {os.path.basename(fname)} NOT FOUND")
    
    # Load statistics
    print("\n" + "-"*80)
    print("STATISTICS")
    print("-"*80)
    
    stats = {}
    if os.path.exists(stats_file):
        with open(stats_file, 'r') as f:
            stats = json.load(f)
        
        print("\nAPE Statistics (Absolute Pose Error):")
        for key, value in stats.items():
            if isinstance(value, (int, float)):
                print(f"  {key:20s}: {value:12.6f}")
            else:
                print(f"  {key:20s}: {value}")
    
    if os.path.exists(info_file):
        with open(info_file, 'r') as f:
            info = json.load(f)
        
        print("\nMetadata:")
        for key, value in info.items():
            print(f"  {key:20s}: {value}")
    
    # Load and analyze error array
    print("\n" + "-"*80)
    print("GENERATING PLOTS")
    print("-"*80)
    
    if os.path.exists(error_file):
        errors = np.load(error_file)
        print(f"\nError array shape: {errors.shape}")
        print(f"Error array stats:")
        print(f"  Min:    {np.min(errors):.6f} m")
        print(f"  Max:    {np.max(errors):.6f} m")
        print(f"  Mean:   {np.mean(errors):.6f} m")
        print(f"  Median: {np.median(errors):.6f} m")
        print(f"  Std:    {np.std(errors):.6f} m")
        
        # Create 4-panel plot
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Absolute Pose Error (APE) Analysis{title_suffix}', fontsize=16, fontweight='bold')
        
        # Plot 1: Error over time
        ax = axes[0, 0]
        ax.plot(errors, 'b-', linewidth=1)
        ax.set_xlabel('Frame Index')
        ax.set_ylabel('Error (m)')
        ax.set_title(f'APE Over Time')
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Histogram
        ax = axes[0, 1]
        ax.hist(errors, bins=50, color='green', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Error (m)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Error Distribution')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Cumulative distribution
        ax = axes[1, 0]
        sorted_errors = np.sort(errors)
        cumsum = np.cumsum(sorted_errors) / np.sum(sorted_errors) * 100
        ax.plot(sorted_errors, cumsum, 'r-', linewidth=2)
        ax.set_xlabel('Error (m)')
        ax.set_ylabel('Cumulative Percentage (%)')
        ax.set_title(f'Cumulative Error Distribution')
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Statistics summary
        ax = axes[1, 1]
        ax.axis('off')
        
        stats_text = f"""
APE Statistics Summary

Mean Error:        {np.mean(errors):.4f} m
Median Error:      {np.median(errors):.4f} m
Std Deviation:     {np.std(errors):.4f} m
RMSE:              {np.sqrt(np.mean(errors**2)):.4f} m

Min Error:         {np.min(errors):.4f} m
Max Error:         {np.max(errors):.4f} m
Range:             {np.max(errors) - np.min(errors):.4f} m

Total Poses:       {len(errors)}
        """
        
        ax.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                verticalalignment='center', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Save combined plot
        plot_path = os.path.join(results_dir, 'ate_analysis.png')
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Saved: {plot_path}")
        plt.close()
        
        # Save individual plots
        save_individual_plots(errors, results_dir, title_suffix)
        
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nGenerated files:")
    print(f"  • {results_dir}/ate_analysis.png - 4-panel summary")
    print(f"  • {results_dir}/ate_timeline.png - Error over time")
    print(f"  • {results_dir}/ate_histogram.png - Error distribution")
    print(f"  • {results_dir}/ate_boxplot.png - Statistical box plot")
    
    return stats


def save_individual_plots(errors, results_dir, title_suffix=''):
    """Save individual high-quality plots."""
    
    # Error timeline
    plt.figure(figsize=(14, 6))
    plt.plot(errors, 'b-', linewidth=0.8)
    plt.xlabel('Frame Index', fontsize=12)
    plt.ylabel('Absolute Pose Error (m)', fontsize=12)
    plt.title(f'APE Timeline{title_suffix}', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    timeline_path = os.path.join(results_dir, 'ate_timeline.png')
    plt.savefig(timeline_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {timeline_path}")
    plt.close()
    
    # Histogram
    plt.figure(figsize=(12, 6))
    plt.hist(errors, bins=100, color='green', alpha=0.7, edgecolor='black')
    plt.xlabel('Error (m)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(f'Error Distribution{title_suffix}', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    hist_path = os.path.join(results_dir, 'ate_histogram.png')
    plt.savefig(hist_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {hist_path}")
    plt.close()
    
    # Box plot
    plt.figure(figsize=(10, 6))
    bp = plt.boxplot([errors], labels=['APE'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    plt.ylabel('Error (m)', fontsize=12)
    plt.title(f'Error Distribution (Box Plot){title_suffix}', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    box_path = os.path.join(results_dir, 'ate_boxplot.png')
    plt.savefig(box_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {box_path}")
    plt.close()


def print_summary(stats):
    """Print detailed summary of results."""
    
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    print(f"""
Absolute Pose Error (APE) - Translation Part

  Maximum Error:     {stats.get('max', 'N/A'):>12.4f} m
  Mean Error:        {stats.get('mean', 'N/A'):>12.4f} m  ← Average
  Median Error:      {stats.get('median', 'N/A'):>12.4f} m
  Minimum Error:     {stats.get('min', 'N/A'):>12.4f} m
  RMSE:              {stats.get('rmse', 'N/A'):>12.4f} m  ← Root Mean Square
  Std Deviation:     {stats.get('std', 'N/A'):>12.4f} m

Interpretation:
  • Your SLAM system has an average error of ~{stats.get('mean', 0):.2f} meters
  • RMSE of {stats.get('rmse', 0):.2f} meters indicates typical error magnitude
  • Max error of {stats.get('max', 0):.2f} meters is the worst-case deviation
    """)


if __name__ == '__main__':
    import sys
    
    # Directory containing extracted files
    results_dir = 'results'
    title_suffix = ''
    
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    
    if len(sys.argv) > 2:
        title_suffix = sys.argv[2]
    
    if not os.path.isdir(results_dir):
        print(f"\nError: Directory '{results_dir}' not found")
        print(f"Expected to find: error_array.npy, stats.json, info.json")
        print(f"\nUsage: python3 analyze_ate_results_simple.py [results_dir] [title_suffix]")
        sys.exit(1)
    
    stats = analyze_ate_results(results_dir, title_suffix)
    print_summary(stats)