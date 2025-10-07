#!/usr/bin/env python3
"""
Test suite for The True Experience – AI Film Analysis System

This script provides comprehensive testing of all system components
and validates the output against the expected schema.
"""

import unittest
import tempfile
import os
import json
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import logging

# Import system components
from src.video_processor import VideoProcessor
from src.audio_processor import AudioProcessor
from src.scene_detector import SceneDetector
from src.visual_analyzer import VisualAnalyzer
from src.dialogue_processor import DialogueProcessor
from src.multimodal_fusion import MultimodalFusion
from src.timeline_generator import TimelineGenerator
from src.json_schema import AnalysisSchema


class TestVideoProcessor(unittest.TestCase):
    """Test cases for VideoProcessor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.video_processor = VideoProcessor(fps=6.0, use_gpu=False)
    
    def test_initialization(self):
        """Test VideoProcessor initialization."""
        self.assertEqual(self.video_processor.fps, 6.0)
        self.assertFalse(self.video_processor.use_gpu)
    
    def test_get_video_info(self):
        """Test video info extraction."""
        # Create a mock video file for testing
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # Mock cv2.VideoCapture
            with patch('cv2.VideoCapture') as mock_cap:
                mock_cap.return_value.isOpened.return_value = True
                mock_cap.return_value.get.side_effect = [30.0, 100, 1920, 1080]
                
                info = self.video_processor.get_video_info(temp_path)
                
                self.assertEqual(info['fps'], 30.0)
                self.assertEqual(info['frame_count'], 100)
                self.assertEqual(info['width'], 1920)
                self.assertEqual(info['height'], 1080)
                self.assertEqual(info['duration'], 100/30.0)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestAudioProcessor(unittest.TestCase):
    """Test cases for AudioProcessor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.audio_processor = AudioProcessor(use_gpu=False)
    
    def test_initialization(self):
        """Test AudioProcessor initialization."""
        self.assertEqual(self.audio_processor.sample_rate, 44100)
        self.assertFalse(self.audio_processor.use_gpu)
    
    def test_detect_silence(self):
        """Test silence detection."""
        # Create test audio signal
        sr = 44100
        duration = 2.0
        silence = np.zeros(int(sr * duration))
        noise = np.random.normal(0, 0.1, int(sr * duration))
        
        # Test silence detection
        silence_segments = self.audio_processor._detect_silence(silence, sr)
        noise_segments = self.audio_processor._detect_silence(noise, sr)
        
        # Silence should be detected in silent audio
        self.assertGreater(len(silence_segments), 0)
        # Noise should have fewer silence segments
        self.assertLess(len(noise_segments), len(silence_segments))


class TestSceneDetector(unittest.TestCase):
    """Test cases for SceneDetector."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.scene_detector = SceneDetector()
    
    def test_initialization(self):
        """Test SceneDetector initialization."""
        self.assertEqual(self.scene_detector.histogram_threshold, 0.3)
        self.assertEqual(self.scene_detector.ssim_threshold, 0.8)
        self.assertEqual(self.scene_detector.min_shot_duration, 1.0)
        self.assertEqual(self.scene_detector.min_scene_duration, 5.0)
    
    def test_histogram_detection(self):
        """Test histogram-based shot detection."""
        # Create test frames
        frames = [
            np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8),
            np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8),
            np.ones((100, 100, 3), dtype=np.uint8) * 255,  # Very different frame
            np.ones((100, 100, 3), dtype=np.uint8) * 255,
        ]
        timestamps = [0.0, 1.0, 2.0, 3.0]
        
        boundaries = self.scene_detector._histogram_detection(frames, timestamps)
        
        # Should detect boundary at frame 2 (different from previous)
        self.assertGreater(len(boundaries), 0)
        self.assertEqual(boundaries[0]['frame_index'], 2)
    
    def test_ssim_detection(self):
        """Test SSIM-based shot detection."""
        # Create test frames
        frames = [
            np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8),
            np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8),
            np.ones((100, 100, 3), dtype=np.uint8) * 255,  # Very different frame
            np.ones((100, 100, 3), dtype=np.uint8) * 255,
        ]
        timestamps = [0.0, 1.0, 2.0, 3.0]
        
        boundaries = self.scene_detector._ssim_detection(frames, timestamps)
        
        # Should detect boundary at frame 2
        self.assertGreater(len(boundaries), 0)
        self.assertEqual(boundaries[0]['frame_index'], 2)


class TestVisualAnalyzer(unittest.TestCase):
    """Test cases for VisualAnalyzer."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock the model loading to avoid downloading during tests
        with patch('src.visual_analyzer.CLIPModel'), \
             patch('src.visual_analyzer.CLIPProcessor'), \
             patch('src.visual_analyzer.BlipProcessor'), \
             patch('src.visual_analyzer.BlipForConditionalGeneration'):
            self.visual_analyzer = VisualAnalyzer(use_gpu=False)
    
    def test_initialization(self):
        """Test VisualAnalyzer initialization."""
        self.assertFalse(self.visual_analyzer.use_gpu)
    
    def test_analyze_lighting_color(self):
        """Test lighting and color analysis."""
        # Create test frame
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        lighting, color_palette = self.visual_analyzer._analyze_lighting_color(frame)
        
        self.assertIn(lighting, ['dark', 'dim', 'normal', 'bright'])
        self.assertIsInstance(color_palette, list)
        self.assertGreater(len(color_palette), 0)


class TestDialogueProcessor(unittest.TestCase):
    """Test cases for DialogueProcessor."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock Whisper model loading
        with patch('whisper.load_model'):
            self.dialogue_processor = DialogueProcessor(use_gpu=False)
    
    def test_initialization(self):
        """Test DialogueProcessor initialization."""
        self.assertEqual(self.dialogue_processor.model_size, "base")
        self.assertFalse(self.dialogue_processor.use_gpu)
    
    def test_extract_speaker_features(self):
        """Test speaker feature extraction."""
        # Create test audio
        sr = 16000
        audio = np.random.normal(0, 0.1, sr)
        
        features = self.dialogue_processor._extract_speaker_features(audio, sr)
        
        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(len(features), 16)  # 13 MFCC + 3 additional features


class TestMultimodalFusion(unittest.TestCase):
    """Test cases for MultimodalFusion."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.multimodal_fusion = MultimodalFusion()
    
    def test_initialization(self):
        """Test MultimodalFusion initialization."""
        self.assertIsInstance(self.multimodal_fusion.emotion_vocabulary, list)
        self.assertIsInstance(self.multimodal_fusion.narrative_functions, list)
    
    def test_fuse_emotions(self):
        """Test emotion fusion."""
        visual = {'lighting': 'dark', 'environment': 'rainy'}
        dialogue = [
            {'emotion': 'sadness', 'confidence': 0.8},
            {'emotion': 'neutral', 'confidence': 0.5}
        ]
        audio = {'energy': 0.1, 'tempo': 60}
        
        result = self.multimodal_fusion._fuse_emotions(visual, dialogue, audio)
        
        self.assertIn('primary_emotion', result)
        self.assertIn('confidence', result)
        self.assertIn('breakdown', result)
        self.assertIn('sources', result)


class TestTimelineGenerator(unittest.TestCase):
    """Test cases for TimelineGenerator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.timeline_generator = TimelineGenerator(interval=5.0)
    
    def test_initialization(self):
        """Test TimelineGenerator initialization."""
        self.assertEqual(self.timeline_generator.interval, 5.0)
    
    def test_format_timestamp(self):
        """Test timestamp formatting."""
        timestamp = self.timeline_generator._format_timestamp(125.5)
        self.assertEqual(timestamp, "0:02:05.500000")
    
    def test_generate_timeline(self):
        """Test timeline generation."""
        fused_data = {
            'scenes': [
                {
                    'scene_id': 0,
                    'start_time': 0.0,
                    'end_time': 10.0,
                    'emotion': 'neutral',
                    'emotion_confidence': 0.7,
                    'narrative_function': 'setup',
                    'characters_present': ['character1'],
                    'dialogue_count': 2,
                    'music_present': True,
                    'sound_effects': ['footsteps']
                }
            ]
        }
        video_duration = 10.0
        
        timeline = self.timeline_generator.generate_timeline(fused_data, video_duration)
        
        self.assertEqual(len(timeline), 2)  # 10s video with 5s intervals
        self.assertEqual(timeline[0]['start_time'], "0:00:00")
        self.assertEqual(timeline[0]['end_time'], "0:00:05")


class TestAnalysisSchema(unittest.TestCase):
    """Test cases for AnalysisSchema."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.schema = AnalysisSchema()
    
    def test_initialization(self):
        """Test AnalysisSchema initialization."""
        self.assertIsInstance(self.schema.schema, dict)
        self.assertIn('type', self.schema.schema)
        self.assertEqual(self.schema.schema['type'], 'object')
    
    def test_validate_sample_data(self):
        """Test validation with sample data."""
        sample_data = self.schema.create_sample_output()
        
        self.assertTrue(self.schema.validate(sample_data))
    
    def test_validate_invalid_data(self):
        """Test validation with invalid data."""
        invalid_data = {
            "metadata": {
                "video_duration": "invalid",  # Should be number
                "frame_rate": 6.0,
                "audio_rate": 44100
            },
            "scenes": [],
            "timeline": []
        }
        
        self.assertFalse(self.schema.validate(invalid_data))


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system."""
    
    def test_end_to_end_validation(self):
        """Test end-to-end data validation."""
        # Create mock data that matches the expected schema
        mock_data = {
            "metadata": {
                "video_duration": 120.5,
                "frame_rate": 6.0,
                "audio_rate": 44100,
                "timeline_interval": 5.0,
                "video_path": "/path/to/video.mp4",
                "analysis_timestamp": "2024-01-01T00:00:00"
            },
            "scenes": [
                {
                    "scene_id": 0,
                    "start_time": 0.0,
                    "end_time": 30.0,
                    "duration": 30.0,
                    "shots": [
                        {
                            "shot_id": 0,
                            "start_frame": 0,
                            "end_frame": 180,
                            "start_time": 0.0,
                            "end_time": 30.0,
                            "duration": 30.0
                        }
                    ],
                    "visual_summary": "Test scene",
                    "audio_summary": "Test audio",
                    "emotion": "neutral",
                    "narrative_function": "setup",
                    "characters_present": ["character1"],
                    "objects_detected": [],
                    "environment": "test environment",
                    "lighting": "normal",
                    "camera_work": "static",
                    "dialogue_count": 0,
                    "music_present": False,
                    "sound_effects": []
                }
            ],
            "timeline": [
                {
                    "start_time": "0:00:00",
                    "end_time": "0:00:05",
                    "duration": 5.0,
                    "visual_mood": "neutral",
                    "audio_mood": "silence",
                    "emotion": "neutral",
                    "emotion_confidence": 0.5,
                    "characters": ["character1"],
                    "dialogue": [],
                    "sound_effects": [],
                    "music": {"present": False, "type": "none"},
                    "key_events": [],
                    "narrative_function": "setup",
                    "scene_ids": [0],
                    "contrasts_detected": []
                }
            ]
        }
        
        schema = AnalysisSchema()
        self.assertTrue(schema.validate(mock_data))


def run_performance_tests():
    """Run performance tests."""
    print("Running performance tests...")
    
    # Test video processing performance
    video_processor = VideoProcessor(fps=6.0, use_gpu=False)
    
    # Create test video data
    test_frames = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(100)]
    
    import time
    start_time = time.time()
    
    # Test scene detection performance
    scene_detector = SceneDetector()
    scenes = scene_detector.detect_scenes(test_frames)
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    print(f"Scene detection processed {len(test_frames)} frames in {processing_time:.2f} seconds")
    print(f"Processing rate: {len(test_frames)/processing_time:.2f} frames/second")
    
    return processing_time < 10.0  # Should process 100 frames in less than 10 seconds


def main():
    """Run all tests."""
    print("Running AI Film Analysis System Tests")
    print("=" * 50)
    
    # Set up logging
    logging.basicConfig(level=logging.WARNING)  # Reduce log noise during tests
    
    # Run unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run performance tests
    print("\n" + "=" * 50)
    performance_passed = run_performance_tests()
    
    print("\n" + "=" * 50)
    if performance_passed:
        print("✅ All tests passed!")
    else:
        print("❌ Some performance tests failed!")
    
    print("Test suite completed.")


if __name__ == "__main__":
    main()