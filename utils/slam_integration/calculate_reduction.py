#!/usr/bin/env python3
"""
Calculate the reduction in cuboid entries between static and ByteTrack approaches.

Reduction = (Total_Static - Unique_TrackIDs) / Total_Static × 100%
"""

import json
import os
from pathlib import Path
from collections import defaultdict

def count_cuboids(cuboid_dir: str, use_tracking: bool = False):
    """
    Count cuboids in a directory.
    
    Args:
        cuboid_dir: Path to directory containing frame JSON files
        use_tracking: If True, count unique track_ids; if False, count total objects
    
    Returns:
        If use_tracking=False: total number of cuboid entries
        If use_tracking=True: (total_entries, unique_track_ids, max_track_id)
    """
    cuboid_path = Path(cuboid_dir)
    
    if not cuboid_path.exists():
        raise FileNotFoundError(f"Directory not found: {cuboid_dir}")
    
    total_entries = 0
    track_ids = set()
    frames_processed = 0
    
    # Get all JSON files sorted by frame number
    json_files = sorted(cuboid_path.glob("*.json"))
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            objects = data.get("objects", [])
            total_entries += len(objects)
            frames_processed += 1
            
            if use_tracking:
                for obj in objects:
                    tid = obj.get("track_id", -1)
                    if tid > 0:  # Valid track_id
                        track_ids.add(tid)
                        
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Error reading {json_file}: {e}")
            continue
    
    print(f"Processed {frames_processed} frames from {cuboid_dir}")
    
    if use_tracking:
        return total_entries, len(track_ids), max(track_ids) if track_ids else 0
    else:
        return total_entries


def calculate_reduction(static_dir: str, bytetrack_dir: str):
    """
    Calculate the reduction percentage.
    
    Args:
        static_dir: Path to static cuboid outputs (no tracking)
        bytetrack_dir: Path to ByteTrack cuboid outputs (with tracking)
    """
    print("=" * 60)
    print("CUBOID REDUCTION ANALYSIS")
    print("=" * 60)
    
    # Count static cuboids (total entries across all frames)
    print("\n[1] Counting STATIC cuboids...")
    total_static = count_cuboids(static_dir, use_tracking=False)
    print(f"    Total cuboid entries: {total_static:,}")
    
    # Count ByteTrack cuboids (unique track IDs)
    print("\n[2] Counting BYTETRACK cuboids...")
    total_bytetrack, unique_tracks, max_track_id = count_cuboids(bytetrack_dir, use_tracking=True)
    print(f"    Total cuboid entries: {total_bytetrack:,}")
    print(f"    Unique track IDs: {unique_tracks:,}")
    print(f"    Max track ID: {max_track_id}")
    
    # Calculate reduction
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    if total_static > 0:
        # Method 1: Using unique track IDs
        reduction = (total_static - unique_tracks) / total_static * 100
        
        print(f"\nTotal Static Cuboid Entries:     {total_static:,}")
        print(f"Unique Tracked Objects:          {unique_tracks:,}")
        print(f"Redundant Entries Eliminated:    {total_static - unique_tracks:,}")
        print(f"\n>>> REDUCTION: {reduction:.2f}% <<<")
        
        # Additional stats
        print(f"\nAverage detections per unique object: {total_static / unique_tracks:.1f}x" if unique_tracks > 0 else "")
        
        return {
            "total_static": total_static,
            "total_bytetrack_entries": total_bytetrack,
            "unique_track_ids": unique_tracks,
            "reduction_percent": reduction,
            "redundancy_factor": total_static / unique_tracks if unique_tracks > 0 else 0
        }
    else:
        print("ERROR: No static cuboids found!")
        return None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Calculate cuboid reduction percentage")
    parser.add_argument("static", type=str, nargs="?", default="cuboid_outputs",
                        help="Path to static cuboid directory")
    parser.add_argument("bytetrack", type=str, nargs="?", default="cuboid_outputs_bytrack",
                        help="Path to ByteTrack cuboid directory")
    
    args = parser.parse_args()
    
    results = calculate_reduction(args.static, args.bytetrack)
    
    if results:
        print("\n" + "=" * 60)
        print("For your thesis, you can write:")
        print("=" * 60)
        print(f"""
ByteTrack integration achieved a {results['reduction_percent']:.1f}% reduction 
in redundant cuboid entries, consolidating {results['total_static']:,} 
frame-level detections into {results['unique_track_ids']:,} unique tracked 
objects across the sequence.
""")