"""
Video preprocessing and frame extraction module.

Handles video file loading, frame extraction at specified FPS,
and provides synchronized video and audio streams.
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
import torch
from datetime import datetime


class VideoProcessor:
    """Handles video preprocessing and frame extraction."""
    
    def __init__(self, fps: float = 6.0, use_gpu: bool = False):
        """
        Initialize video processor.
        
        Args:
            fps: Target frame rate for analysis
            use_gpu: Whether to use GPU acceleration
        """
        self.fps = fps
        self.use_gpu = use_gpu
        self.logger = logging.getLogger(__name__)
        
        # Initialize device
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")
    
    def process_video(self, video_path: str) -> Dict[str, Any]:
        """
        Process video file and extract frames at specified FPS.
        
        Args:
            video_path: Path to input video file
            
        Returns:
            Dictionary containing frames, metadata, and timing information
        """
        self.logger.info(f"Processing video: {video_path}")
        
        # Open video capture
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / original_fps
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.logger.info(f"Video properties: {width}x{height}, {original_fps:.2f} FPS, {duration:.2f}s duration")
        
        # Calculate frame interval for target FPS
        frame_interval = int(original_fps / self.fps)
        self.logger.info(f"Extracting every {frame_interval} frames for {self.fps} FPS analysis")
        
        # Extract frames
        frames = []
        frame_timestamps = []
        frame_count = 0
        extracted_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                frame_timestamps.append(frame_count / original_fps)
                extracted_count += 1
                
                if extracted_count % 100 == 0:
                    self.logger.debug(f"Extracted {extracted_count} frames...")
            
            frame_count += 1
        
        cap.release()
        
        self.logger.info(f"Extracted {len(frames)} frames from {total_frames} total frames")
        
        return {
            'frames': frames,
            'frame_timestamps': frame_timestamps,
            'duration': duration,
            'original_fps': original_fps,
            'target_fps': self.fps,
            'width': width,
            'height': height,
            'total_frames': total_frames,
            'extracted_frames': len(frames),
            'timestamp': datetime.now().isoformat()
        }
    
    def extract_frame_at_time(self, video_path: str, timestamp: float) -> np.ndarray:
        """
        Extract a single frame at specific timestamp.
        
        Args:
            video_path: Path to video file
            timestamp: Time in seconds
            
        Returns:
            Frame as numpy array
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_number = int(timestamp * fps)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise ValueError(f"Could not extract frame at timestamp {timestamp}")
        
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    def get_video_info(self, video_path: str) -> Dict[str, Any]:
        """
        Get basic video information without processing frames.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with video metadata
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        info = {
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
        }
        
        cap.release()
        return info