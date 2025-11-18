#!/usr/bin/env python3
"""
iPhone Video Preprocessing for ORB-SLAM3
Prepares iPhone 13 video (HD 30fps) for ORB-SLAM3 monocular mode
"""

import cv2
import os
import numpy as np
from pathlib import Path
import argparse
import json

class iPhoneVideoProcessor:
    def __init__(self, video_path, output_dir, target_fps=10):
        """
        Initialize video processor
        Args:
            video_path: Path to iPhone video
            output_dir: Directory for output frames
            target_fps: Target frame rate (reduce from 30fps for processing)
        """
        self.video_path = video_path
        self.output_dir = Path(output_dir)
        self.target_fps = target_fps
        
        # iPhone 13 camera specs (for calibration reference)
        self.iphone13_specs = {
            'sensor_width_mm': 5.6,  # Approximate
            'sensor_height_mm': 4.2,  # Approximate
            'focal_length_mm': 5.1,  # Wide camera
            'resolution': (1920, 1080)  # HD
        }
        
    def extract_frames(self, grayscale=False, undistort=False):
        """
        Extract frames from video with optional preprocessing
        """
        cap = cv2.VideoCapture(self.video_path)
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video Info:")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps}")
        print(f"  Total frames: {total_frames}")
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        rgb_dir = self.output_dir / 'rgb'
        rgb_dir.mkdir(exist_ok=True)
        
        if grayscale:
            gray_dir = self.output_dir / 'gray'
            gray_dir.mkdir(exist_ok=True)
        
        # Calculate frame skip for target FPS
        frame_skip = int(fps / self.target_fps)
        
        frame_count = 0
        saved_count = 0
        timestamps = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames to achieve target FPS
            if frame_count % frame_skip == 0:
                # Generate KITTI-style filename
                filename = f"{saved_count:06d}.png"
                
                # Apply preprocessing
                processed_frame = self.preprocess_frame(frame, undistort)
                
                # Save RGB frame
                cv2.imwrite(str(rgb_dir / filename), processed_frame)
                
                # Save grayscale if requested
                if grayscale:
                    gray = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
                    cv2.imwrite(str(gray_dir / filename), gray)
                
                # Calculate timestamp
                timestamp = frame_count / fps
                timestamps.append(timestamp)
                
                saved_count += 1
                
                if saved_count % 100 == 0:
                    print(f"Processed {saved_count} frames...")
            
            frame_count += 1
        
        cap.release()
        
        # Save timestamps
        self.save_timestamps(timestamps)
        
        print(f"\nExtraction complete:")
        print(f"  Saved {saved_count} frames")
        print(f"  Output directory: {self.output_dir}")
        
        return saved_count
    
    def preprocess_frame(self, frame, undistort=False):
        """
        Apply preprocessing to individual frame
        """
        # Reduce noise
        frame = cv2.bilateralFilter(frame, 9, 75, 75)
        
        # Enhance contrast using CLAHE
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        frame = cv2.merge([l, a, b])
        frame = cv2.cvtColor(frame, cv2.COLOR_LAB2BGR)
        
        if undistort:
            # Apply minimal undistortion for iPhone lens
            # iPhone 13 has minimal distortion but we can apply slight correction
            h, w = frame.shape[:2]
            
            # Approximate camera matrix for iPhone 13 HD
            fx = fy = w  # Approximate focal length in pixels
            cx = w / 2
            cy = h / 2
            camera_matrix = np.array([[fx, 0, cx],
                                     [0, fy, cy],
                                     [0, 0, 1]], dtype=float)
            
            # Minimal distortion coefficients
            dist_coeffs = np.array([0.05, -0.05, 0, 0, 0])
            
            # Undistort
            frame = cv2.undistort(frame, camera_matrix, dist_coeffs)
        
        return frame
    
    def save_timestamps(self, timestamps):
        """
        Save timestamps in KITTI format
        """
        times_file = self.output_dir / 'times.txt'
        with open(times_file, 'w') as f:
            for ts in timestamps:
                f.write(f"{ts:.6f}\n")
    
    def generate_calibration(self):
        """
        Generate approximate calibration file for iPhone 13
        """
        width, height = self.iphone13_specs['resolution']
        
        # Approximate focal length in pixels
        focal_length_px = (width * self.iphone13_specs['focal_length_mm']) / self.iphone13_specs['sensor_width_mm']
        
        # Principal point (image center)
        cx = width / 2
        cy = height / 2
        
        # Camera matrix
        K = np.array([[focal_length_px, 0, cx],
                     [0, focal_length_px, cy],
                     [0, 0, 1]])
        
        # Distortion coefficients (minimal for iPhone)
        D = np.array([0.05, -0.05, 0, 0, 0])
        
        # Save calibration
        calib_dir = self.output_dir / 'calibration'
        calib_dir.mkdir(exist_ok=True)
        
        # KITTI-style calibration format
        calib_file = calib_dir / 'camera.txt'
        with open(calib_file, 'w') as f:
            f.write(f"# iPhone 13 Camera Calibration (Approximate)\n")
            f.write(f"image_width: {width}\n")
            f.write(f"image_height: {height}\n")
            f.write(f"camera_matrix:\n")
            f.write(f"  fx: {K[0,0]:.6f}\n")
            f.write(f"  fy: {K[1,1]:.6f}\n")
            f.write(f"  cx: {K[0,2]:.6f}\n")
            f.write(f"  cy: {K[1,2]:.6f}\n")
            f.write(f"distortion_coefficients:\n")
            f.write(f"  k1: {D[0]:.6f}\n")
            f.write(f"  k2: {D[1]:.6f}\n")
            f.write(f"  p1: {D[2]:.6f}\n")
            f.write(f"  p2: {D[3]:.6f}\n")
            f.write(f"  k3: {D[4]:.6f}\n")
        
        # Generate KITTI-style calib.txt file
        self.generate_kitti_calibration(K)
        
        # Save as numpy arrays
        np.save(calib_dir / 'K.npy', K)
        np.save(calib_dir / 'D.npy', D)
        
        # Generate ORB-SLAM3 yaml config
        self.generate_orbslam_config(K, D, width, height)
        
        print(f"Calibration saved to: {calib_dir}")
        
        return K, D
    
    def generate_kitti_calibration(self, K):
        """
        Generate KITTI-style calib.txt file
        Format: P0, P1, P2, P3 projection matrices and Tr transformation
        """
        # Create 3x4 projection matrix from 3x3 intrinsic matrix
        # P = K[R|t] where R is identity and t is zero for monocular
        P = np.zeros((3, 4))
        P[:3, :3] = K
        
        # KITTI uses P0-P3 for different cameras
        # For monocular iPhone, we'll set P0 (grayscale) and P2 (color) to the same
        # P1 and P3 are for right camera (stereo), set to zeros for monocular
        
        calib_file = self.output_dir / 'calib.txt'
        
        with open(calib_file, 'w') as f:
            # P0: Grayscale left camera projection matrix
            f.write('P0: ')
            for val in P.flatten():
                f.write(f'{val:.12e} ')
            f.write('\n')
            
            # P1: Grayscale right camera (zeros for monocular)
            f.write('P1: ')
            for _ in range(12):
                f.write('0.000000000000e+00 ')
            f.write('\n')
            
            # P2: Color left camera projection matrix (same as P0 for iPhone)
            f.write('P2: ')
            for val in P.flatten():
                f.write(f'{val:.12e} ')
            f.write('\n')
            
            # P3: Color right camera (zeros for monocular)
            f.write('P3: ')
            for _ in range(12):
                f.write('0.000000000000e+00 ')
            f.write('\n')
            
            # R0_rect: Rectification matrix (identity for monocular)
            f.write('R0_rect: ')
            R0_rect = np.eye(3)
            for val in R0_rect.flatten():
                f.write(f'{val:.12e} ')
            f.write('\n')
            
            # Tr_velo_to_cam: Velodyne to camera transformation (identity for no LiDAR)
            f.write('Tr_velo_to_cam: ')
            Tr = np.eye(3, 4)
            for val in Tr.flatten():
                f.write(f'{val:.12e} ')
            f.write('\n')
            
            # Tr_imu_to_velo: IMU to Velodyne transformation (identity for no IMU integration)
            f.write('Tr_imu_to_velo: ')
            for val in Tr.flatten():
                f.write(f'{val:.12e} ')
            f.write('\n')
        
        print(f"KITTI calib.txt saved to: {calib_file}")
    
    def generate_orbslam_config(self, K, D, width, height):
        """
        Generate ORB-SLAM3 configuration file for iPhone video
        """
        config_file = self.output_dir / 'Adelaide.yaml'
        
        config = f"""# Camera Parameters. Adjust them!
Camera.type: "PinHole"

# Camera calibration and distortion parameters (OpenCV) 
Camera.fx: {K[0,0]}
Camera.fy: {K[1,1]}
Camera.cx: {K[0,2]}
Camera.cy: {K[1,2]}

Camera.k1: {D[0]}
Camera.k2: {D[1]}
Camera.p1: {D[2]}
Camera.p2: {D[3]}
Camera.k3: {D[4]}

# Camera resolution
Camera.width: {width}
Camera.height: {height}

# Camera frames per second 
Camera.fps: {self.target_fps}

# Color order of the images (0: BGR, 1: RGB. It is ignored if images are grayscale)
Camera.RGB: 1

#--------------------------------------------------------------------------------------------
# ORB Parameters
#--------------------------------------------------------------------------------------------

# ORB Extractor: Number of features per image
ORBextractor.nFeatures: 2000

# ORB Extractor: Scale factor between levels in the scale pyramid 	
ORBextractor.scaleFactor: 1.2

# ORB Extractor: Number of levels in the scale pyramid	
ORBextractor.nLevels: 8

# ORB Extractor: Fast threshold
# Image is divided in a grid. At each cell FAST are extracted imposing a minimum response.
# Firstly we impose iniThFAST. If no corners are detected we impose a lower value minThFAST
ORBextractor.iniThFAST: 20
ORBextractor.minThFAST: 7

#--------------------------------------------------------------------------------------------
# Viewer Parameters
#--------------------------------------------------------------------------------------------
Viewer.KeyFrameSize: 0.05
Viewer.KeyFrameLineWidth: 1.0
Viewer.GraphLineWidth: 0.9
Viewer.PointSize: 2.0
Viewer.CameraSize: 0.08
Viewer.CameraLineWidth: 3.0
Viewer.ViewpointX: 0.0
Viewer.ViewpointY: -0.7
Viewer.ViewpointZ: -1.8
Viewer.ViewpointF: 500.0
"""
        
        with open(config_file, 'w') as f:
            f.write(config)
        
        print(f"ORB-SLAM3 config saved to: {config_file}")

def main():
    parser = argparse.ArgumentParser(description='Preprocess iPhone video for ORB-SLAM3')
    parser.add_argument('video', help='Path to input video file')
    parser.add_argument('--output', default='./processed_video', help='Output directory')
    parser.add_argument('--fps', type=int, default=10, help='Target FPS (default: 10)')
    parser.add_argument('--grayscale', action='store_true', help='Also save grayscale frames')
    parser.add_argument('--undistort', action='store_true', help='Apply lens undistortion')
    
    args = parser.parse_args()
    
    # Process video
    processor = iPhoneVideoProcessor(args.video, args.output, args.fps)
    
    print("Starting video preprocessing...")
    print("-" * 50)
    
    # Extract frames
    num_frames = processor.extract_frames(
        grayscale=args.grayscale,
        undistort=args.undistort
    )
    
    # Generate calibration
    processor.generate_calibration()
    
    print("-" * 50)
    print("Preprocessing complete!")
    print(f"\nNext steps:")
    print(f"1. Run object detection on frames in: {args.output}/rgb/")
    print(f"2. Use ORB-SLAM3 config: {args.output}/Adelaide.yaml")
    print(f"3. Process with ORB-SLAM3 using the extracted frames")

if __name__ == "__main__":
    main()