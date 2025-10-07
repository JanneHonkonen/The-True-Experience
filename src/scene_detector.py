"""
Scene and shot detection module.

Implements shot boundary detection using histogram delta, SSIM,
and optional PySceneDetect integration for robust scene segmentation.
"""

import cv2
import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional
from skimage.metrics import structural_similarity as ssim
from scipy import stats
import pyscenedetect
from pyscenedetect import VideoManager, SceneManager
from pyscenedetect.detectors import ContentDetector


class SceneDetector:
    """Handles scene and shot detection using multiple methods."""
    
    def __init__(self, 
                 histogram_threshold: float = 0.3,
                 ssim_threshold: float = 0.8,
                 min_shot_duration: float = 1.0,
                 min_scene_duration: float = 5.0):
        """
        Initialize scene detector.
        
        Args:
            histogram_threshold: Threshold for histogram-based shot detection
            ssim_threshold: Threshold for SSIM-based shot detection
            min_shot_duration: Minimum duration for a valid shot (seconds)
            min_scene_duration: Minimum duration for a valid scene (seconds)
        """
        self.histogram_threshold = histogram_threshold
        self.ssim_threshold = ssim_threshold
        self.min_shot_duration = min_shot_duration
        self.min_scene_duration = min_scene_duration
        self.logger = logging.getLogger(__name__)
    
    def detect_scenes(self, frames: List[np.ndarray], 
                     frame_timestamps: Optional[List[float]] = None) -> List[Dict[str, Any]]:
        """
        Detect scenes and shots in video frames.
        
        Args:
            frames: List of video frames
            frame_timestamps: Optional list of frame timestamps
            
        Returns:
            List of scene dictionaries with shots and metadata
        """
        self.logger.info(f"Detecting scenes in {len(frames)} frames")
        
        if frame_timestamps is None:
            # Generate timestamps assuming 6 FPS
            frame_timestamps = [i / 6.0 for i in range(len(frames))]
        
        # Detect shot boundaries using multiple methods
        shot_boundaries = self._detect_shot_boundaries(frames, frame_timestamps)
        
        # Merge nearby boundaries and filter by minimum duration
        filtered_boundaries = self._filter_shot_boundaries(shot_boundaries, frame_timestamps)
        
        # Group shots into scenes
        scenes = self._group_shots_into_scenes(filtered_boundaries, frames, frame_timestamps)
        
        self.logger.info(f"Detected {len(scenes)} scenes with {sum(len(scene['shots']) for scene in scenes)} shots")
        
        return scenes
    
    def _detect_shot_boundaries(self, frames: List[np.ndarray], 
                               frame_timestamps: List[float]) -> List[Dict[str, Any]]:
        """Detect shot boundaries using multiple methods."""
        self.logger.debug("Detecting shot boundaries...")
        
        boundaries = []
        
        # Method 1: Histogram-based detection
        histogram_boundaries = self._histogram_detection(frames, frame_timestamps)
        boundaries.extend(histogram_boundaries)
        
        # Method 2: SSIM-based detection
        ssim_boundaries = self._ssim_detection(frames, frame_timestamps)
        boundaries.extend(ssim_boundaries)
        
        # Method 3: PySceneDetect (if available)
        try:
            pyscene_boundaries = self._pyscene_detection(frames, frame_timestamps)
            boundaries.extend(pyscene_boundaries)
        except Exception as e:
            self.logger.warning(f"PySceneDetect failed: {e}")
        
        # Remove duplicates and sort by timestamp
        unique_boundaries = self._merge_boundaries(boundaries)
        
        return unique_boundaries
    
    def _histogram_detection(self, frames: List[np.ndarray], 
                           frame_timestamps: List[float]) -> List[Dict[str, Any]]:
        """Detect shot boundaries using histogram comparison."""
        boundaries = []
        
        if len(frames) < 2:
            return boundaries
        
        # Calculate histograms for all frames
        histograms = []
        for frame in frames:
            # Convert to HSV for better color analysis
            hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
            hist = cv2.calcHist([hsv], [0, 1, 2], None, [50, 60, 60], [0, 180, 0, 256, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            histograms.append(hist)
        
        # Compare consecutive frames
        for i in range(1, len(histograms)):
            # Calculate histogram correlation
            correlation = cv2.compareHist(histograms[i-1], histograms[i], cv2.HISTCMP_CORREL)
            
            # Calculate histogram intersection
            intersection = cv2.compareHist(histograms[i-1], histograms[i], cv2.HISTCMP_INTERSECT)
            
            # Calculate chi-square distance
            chi_square = cv2.compareHist(histograms[i-1], histograms[i], cv2.HISTCMP_CHISQR)
            
            # Combine metrics for shot detection
            # Lower correlation and intersection, higher chi-square indicate shot change
            change_score = (1 - correlation) + (1 - intersection) + (chi_square / 1000)
            
            if change_score > self.histogram_threshold:
                boundaries.append({
                    'frame_index': i,
                    'timestamp': frame_timestamps[i],
                    'method': 'histogram',
                    'confidence': min(change_score, 1.0),
                    'metrics': {
                        'correlation': float(correlation),
                        'intersection': float(intersection),
                        'chi_square': float(chi_square)
                    }
                })
        
        return boundaries
    
    def _ssim_detection(self, frames: List[np.ndarray], 
                       frame_timestamps: List[float]) -> List[Dict[str, Any]]:
        """Detect shot boundaries using Structural Similarity Index."""
        boundaries = []
        
        if len(frames) < 2:
            return boundaries
        
        # Convert frames to grayscale for SSIM calculation
        gray_frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in frames]
        
        for i in range(1, len(gray_frames)):
            # Calculate SSIM between consecutive frames
            ssim_score = ssim(gray_frames[i-1], gray_frames[i])
            
            # Lower SSIM indicates more change
            change_score = 1 - ssim_score
            
            if change_score > (1 - self.ssim_threshold):
                boundaries.append({
                    'frame_index': i,
                    'timestamp': frame_timestamps[i],
                    'method': 'ssim',
                    'confidence': change_score,
                    'metrics': {
                        'ssim_score': float(ssim_score)
                    }
                })
        
        return boundaries
    
    def _pyscene_detection(self, frames: List[np.ndarray], 
                          frame_timestamps: List[float]) -> List[Dict[str, Any]]:
        """Detect shot boundaries using PySceneDetect."""
        boundaries = []
        
        # This is a simplified implementation
        # In practice, you would save frames to a temporary video file
        # and use PySceneDetect's VideoManager
        
        # For now, we'll use a simple content-based detection
        # that mimics PySceneDetect's behavior
        
        if len(frames) < 10:
            return boundaries
        
        # Calculate frame differences
        frame_diffs = []
        for i in range(1, len(frames)):
            # Convert to grayscale
            gray1 = cv2.cvtColor(frames[i-1], cv2.COLOR_RGB2GRAY)
            gray2 = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
            
            # Calculate absolute difference
            diff = cv2.absdiff(gray1, gray2)
            mean_diff = np.mean(diff)
            frame_diffs.append(mean_diff)
        
        # Find peaks in frame differences
        from scipy.signal import find_peaks
        
        peaks, _ = find_peaks(frame_diffs, height=np.mean(frame_diffs) + np.std(frame_diffs))
        
        for peak in peaks:
            if peak < len(frame_timestamps):
                boundaries.append({
                    'frame_index': peak + 1,
                    'timestamp': frame_timestamps[peak + 1],
                    'method': 'pyscene',
                    'confidence': frame_diffs[peak] / np.max(frame_diffs),
                    'metrics': {
                        'frame_diff': float(frame_diffs[peak])
                    }
                })
        
        return boundaries
    
    def _merge_boundaries(self, boundaries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge nearby boundaries and remove duplicates."""
        if not boundaries:
            return []
        
        # Sort by frame index
        boundaries.sort(key=lambda x: x['frame_index'])
        
        merged = []
        current = boundaries[0]
        
        for boundary in boundaries[1:]:
            # If boundaries are within 3 frames of each other, merge them
            if boundary['frame_index'] - current['frame_index'] <= 3:
                # Keep the one with higher confidence
                if boundary['confidence'] > current['confidence']:
                    current = boundary
            else:
                merged.append(current)
                current = boundary
        
        merged.append(current)
        return merged
    
    def _filter_shot_boundaries(self, boundaries: List[Dict[str, Any]], 
                               frame_timestamps: List[float]) -> List[Dict[str, Any]]:
        """Filter shot boundaries by minimum duration."""
        if not boundaries:
            return []
        
        filtered = []
        last_boundary_time = 0
        
        for boundary in boundaries:
            boundary_time = boundary['timestamp']
            
            # Check minimum shot duration
            if boundary_time - last_boundary_time >= self.min_shot_duration:
                filtered.append(boundary)
                last_boundary_time = boundary_time
        
        return filtered
    
    def _group_shots_into_scenes(self, shot_boundaries: List[Dict[str, Any]], 
                                frames: List[np.ndarray], 
                                frame_timestamps: List[float]) -> List[Dict[str, Any]]:
        """Group shots into scenes based on temporal proximity and visual similarity."""
        scenes = []
        
        if not shot_boundaries:
            # Single scene with all frames
            scenes.append({
                'scene_id': 0,
                'start_time': frame_timestamps[0],
                'end_time': frame_timestamps[-1],
                'shots': [{
                    'shot_id': 0,
                    'start_frame': 0,
                    'end_frame': len(frames) - 1,
                    'start_time': frame_timestamps[0],
                    'end_time': frame_timestamps[-1],
                    'duration': frame_timestamps[-1] - frame_timestamps[0]
                }],
                'visual_summary': 'Single continuous shot',
                'narrative_function': 'establishing'
            })
            return scenes
        
        # Create shots from boundaries
        shots = []
        start_frame = 0
        
        for i, boundary in enumerate(shot_boundaries):
            end_frame = boundary['frame_index'] - 1
            shots.append({
                'shot_id': i,
                'start_frame': start_frame,
                'end_frame': end_frame,
                'start_time': frame_timestamps[start_frame],
                'end_time': frame_timestamps[end_frame],
                'duration': frame_timestamps[end_frame] - frame_timestamps[start_frame]
            })
            start_frame = boundary['frame_index']
        
        # Add final shot
        shots.append({
            'shot_id': len(shots),
            'start_frame': start_frame,
            'end_frame': len(frames) - 1,
            'start_time': frame_timestamps[start_frame],
            'end_time': frame_timestamps[-1],
            'duration': frame_timestamps[-1] - frame_timestamps[start_frame]
        })
        
        # Group shots into scenes based on temporal proximity
        current_scene_shots = [shots[0]]
        scene_id = 0
        
        for i in range(1, len(shots)):
            shot = shots[i]
            last_shot = current_scene_shots[-1]
            
            # If shots are close in time, group them in same scene
            time_gap = shot['start_time'] - last_shot['end_time']
            
            if time_gap <= 2.0:  # 2 second gap threshold
                current_scene_shots.append(shot)
            else:
                # Create new scene
                scenes.append(self._create_scene(scene_id, current_scene_shots, frames, frame_timestamps))
                current_scene_shots = [shot]
                scene_id += 1
        
        # Add final scene
        if current_scene_shots:
            scenes.append(self._create_scene(scene_id, current_scene_shots, frames, frame_timestamps))
        
        return scenes
    
    def _create_scene(self, scene_id: int, shots: List[Dict[str, Any]], 
                     frames: List[np.ndarray], frame_timestamps: List[float]) -> Dict[str, Any]:
        """Create a scene from a list of shots."""
        start_time = shots[0]['start_time']
        end_time = shots[-1]['end_time']
        duration = end_time - start_time
        
        # Analyze visual content for scene summary
        visual_summary = self._analyze_scene_visual_content(shots, frames)
        
        # Determine narrative function
        narrative_function = self._determine_narrative_function(shots, duration)
        
        return {
            'scene_id': scene_id,
            'start_time': start_time,
            'end_time': end_time,
            'duration': duration,
            'shots': shots,
            'visual_summary': visual_summary,
            'narrative_function': narrative_function,
            'shot_count': len(shots)
        }
    
    def _analyze_scene_visual_content(self, shots: List[Dict[str, Any]], 
                                    frames: List[np.ndarray]) -> str:
        """Analyze visual content of a scene."""
        if not shots:
            return "Empty scene"
        
        # Simple analysis based on shot count and duration
        shot_count = len(shots)
        avg_shot_duration = sum(shot['duration'] for shot in shots) / shot_count
        
        if shot_count == 1:
            return "Single continuous shot"
        elif avg_shot_duration < 2.0:
            return f"Fast-paced sequence with {shot_count} quick cuts"
        elif avg_shot_duration > 10.0:
            return f"Slow-paced sequence with {shot_count} long takes"
        else:
            return f"Moderate-paced sequence with {shot_count} shots"
    
    def _determine_narrative_function(self, shots: List[Dict[str, Any]], 
                                    duration: float) -> str:
        """Determine narrative function of a scene."""
        shot_count = len(shots)
        
        if shot_count == 1 and duration > 30:
            return "establishing"
        elif shot_count > 10 and duration < 30:
            return "action"
        elif shot_count <= 3:
            return "dialogue"
        else:
            return "development"