#!/usr/bin/env python3
"""
Example usage of The True Experience – AI Film Analysis System

This script demonstrates how to use the system programmatically
and provides examples of different analysis configurations.
"""

import json
import logging
from pathlib import Path
from src.video_processor import VideoProcessor
from src.audio_processor import AudioProcessor
from src.scene_detector import SceneDetector
from src.visual_analyzer import VisualAnalyzer
from src.dialogue_processor import DialogueProcessor
from src.multimodal_fusion import MultimodalFusion
from src.timeline_generator import TimelineGenerator
from src.json_schema import AnalysisSchema


def setup_logging():
    """Set up logging for the example."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def basic_analysis_example(video_path: str, output_path: str):
    """Example of basic video analysis."""
    print("=== Basic Analysis Example ===")
    
    # Initialize components
    video_processor = VideoProcessor(fps=6.0, use_gpu=False)
    audio_processor = AudioProcessor(use_gpu=False)
    scene_detector = SceneDetector()
    visual_analyzer = VisualAnalyzer(use_gpu=False)
    dialogue_processor = DialogueProcessor(use_gpu=False)
    multimodal_fusion = MultimodalFusion()
    timeline_generator = TimelineGenerator(interval=5.0)
    
    try:
        # Process video
        print("Processing video...")
        video_data = video_processor.process_video(video_path)
        
        # Process audio
        print("Processing audio...")
        audio_data = audio_processor.process_audio(video_path)
        
        # Detect scenes
        print("Detecting scenes...")
        scenes = scene_detector.detect_scenes(video_data['frames'])
        
        # Analyze visual content
        print("Analyzing visual content...")
        visual_analysis = visual_analyzer.analyze_frames(video_data['frames'], scenes)
        
        # Process dialogue
        print("Processing dialogue...")
        dialogue_data = dialogue_processor.process_dialogue(audio_data['audio_path'])
        
        # Fuse multimodal data
        print("Fusing multimodal data...")
        fused_data = multimodal_fusion.fuse_data(
            visual_analysis, dialogue_data, audio_data, scenes
        )
        
        # Generate timeline
        print("Generating timeline...")
        timeline = timeline_generator.generate_timeline(fused_data, video_data['duration'])
        
        # Create final output
        analysis_result = {
            "metadata": {
                "video_duration": video_data['duration'],
                "frame_rate": 6.0,
                "audio_rate": audio_data['sample_rate'],
                "timeline_interval": 5.0,
                "video_path": video_path,
                "analysis_timestamp": video_data['timestamp']
            },
            "scenes": scenes,
            "timeline": timeline,
            "fused_analysis": fused_data
        }
        
        # Validate and save
        schema = AnalysisSchema()
        if schema.validate(analysis_result):
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(analysis_result, f, indent=2, ensure_ascii=False)
            print(f"Analysis complete! Output saved to: {output_path}")
        else:
            print("Output validation failed!")
            
    except Exception as e:
        print(f"Analysis failed: {e}")
        logging.exception("Full traceback:")


def high_resolution_analysis_example(video_path: str, output_path: str):
    """Example of high-resolution analysis with GPU acceleration."""
    print("=== High-Resolution Analysis Example ===")
    
    # Initialize components with higher resolution settings
    video_processor = VideoProcessor(fps=12.0, use_gpu=True)
    audio_processor = AudioProcessor(use_gpu=True)
    scene_detector = SceneDetector(
        histogram_threshold=0.25,
        ssim_threshold=0.75,
        min_shot_duration=0.5,
        min_scene_duration=3.0
    )
    visual_analyzer = VisualAnalyzer(use_gpu=True)
    dialogue_processor = DialogueProcessor(model_size="base", use_gpu=True)
    multimodal_fusion = MultimodalFusion()
    timeline_generator = TimelineGenerator(interval=2.0)
    
    try:
        # Process video
        print("Processing video at 12 FPS...")
        video_data = video_processor.process_video(video_path)
        
        # Process audio
        print("Processing audio...")
        audio_data = audio_processor.process_audio(video_path)
        
        # Detect scenes with more sensitive settings
        print("Detecting scenes with sensitive settings...")
        scenes = scene_detector.detect_scenes(video_data['frames'])
        
        # Analyze visual content
        print("Analyzing visual content...")
        visual_analysis = visual_analyzer.analyze_frames(video_data['frames'], scenes)
        
        # Process dialogue
        print("Processing dialogue...")
        dialogue_data = dialogue_processor.process_dialogue(audio_data['audio_path'])
        
        # Fuse multimodal data
        print("Fusing multimodal data...")
        fused_data = multimodal_fusion.fuse_data(
            visual_analysis, dialogue_data, audio_data, scenes
        )
        
        # Generate detailed timeline
        print("Generating detailed timeline...")
        timeline = timeline_generator.generate_timeline(fused_data, video_data['duration'])
        
        # Create final output
        analysis_result = {
            "metadata": {
                "video_duration": video_data['duration'],
                "frame_rate": 12.0,
                "audio_rate": audio_data['sample_rate'],
                "timeline_interval": 2.0,
                "video_path": video_path,
                "analysis_timestamp": video_data['timestamp']
            },
            "scenes": scenes,
            "timeline": timeline,
            "fused_analysis": fused_data
        }
        
        # Validate and save
        schema = AnalysisSchema()
        if schema.validate(analysis_result):
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(analysis_result, f, indent=2, ensure_ascii=False)
            print(f"High-resolution analysis complete! Output saved to: {output_path}")
        else:
            print("Output validation failed!")
            
    except Exception as e:
        print(f"High-resolution analysis failed: {e}")
        logging.exception("Full traceback:")


def custom_analysis_example(video_path: str, output_path: str):
    """Example of custom analysis with specific requirements."""
    print("=== Custom Analysis Example ===")
    
    # Custom settings for specific use case
    video_processor = VideoProcessor(fps=8.0, use_gpu=True)
    audio_processor = AudioProcessor(sample_rate=48000, use_gpu=True)
    
    # Custom scene detection for action sequences
    scene_detector = SceneDetector(
        histogram_threshold=0.2,  # More sensitive to changes
        ssim_threshold=0.7,       # More sensitive to structural changes
        min_shot_duration=0.3,    # Shorter minimum shot duration
        min_scene_duration=2.0    # Shorter minimum scene duration
    )
    
    visual_analyzer = VisualAnalyzer(use_gpu=True)
    dialogue_processor = DialogueProcessor(model_size="small", use_gpu=True)
    multimodal_fusion = MultimodalFusion()
    timeline_generator = TimelineGenerator(interval=3.0)
    
    try:
        # Process video
        print("Processing video with custom settings...")
        video_data = video_processor.process_video(video_path)
        
        # Process audio
        print("Processing audio at 48kHz...")
        audio_data = audio_processor.process_audio(video_path)
        
        # Detect scenes with custom settings
        print("Detecting scenes with custom sensitivity...")
        scenes = scene_detector.detect_scenes(video_data['frames'])
        
        # Analyze visual content
        print("Analyzing visual content...")
        visual_analysis = visual_analyzer.analyze_frames(video_data['frames'], scenes)
        
        # Process dialogue
        print("Processing dialogue...")
        dialogue_data = dialogue_processor.process_dialogue(audio_data['audio_path'])
        
        # Fuse multimodal data
        print("Fusing multimodal data...")
        fused_data = multimodal_fusion.fuse_data(
            visual_analysis, dialogue_data, audio_data, scenes
        )
        
        # Generate custom timeline
        print("Generating custom timeline...")
        timeline = timeline_generator.generate_timeline(fused_data, video_data['duration'])
        
        # Add custom analysis
        custom_analysis = {
            "action_sequences": detect_action_sequences(scenes),
            "emotional_peaks": find_emotional_peaks(timeline),
            "character_interactions": analyze_character_interactions(fused_data)
        }
        
        # Create final output
        analysis_result = {
            "metadata": {
                "video_duration": video_data['duration'],
                "frame_rate": 8.0,
                "audio_rate": 48000,
                "timeline_interval": 3.0,
                "video_path": video_path,
                "analysis_timestamp": video_data['timestamp'],
                "custom_analysis": True
            },
            "scenes": scenes,
            "timeline": timeline,
            "fused_analysis": fused_data,
            "custom_analysis": custom_analysis
        }
        
        # Validate and save
        schema = AnalysisSchema()
        if schema.validate(analysis_result):
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(analysis_result, f, indent=2, ensure_ascii=False)
            print(f"Custom analysis complete! Output saved to: {output_path}")
        else:
            print("Output validation failed!")
            
    except Exception as e:
        print(f"Custom analysis failed: {e}")
        logging.exception("Full traceback:")


def detect_action_sequences(scenes):
    """Detect action sequences based on scene characteristics."""
    action_sequences = []
    
    for scene in scenes:
        if scene.get('narrative_function') == 'action':
            action_sequences.append({
                'scene_id': scene['scene_id'],
                'start_time': scene['start_time'],
                'end_time': scene['end_time'],
                'shot_count': len(scene.get('shots', [])),
                'intensity': 'high' if len(scene.get('shots', [])) > 10 else 'medium'
            })
    
    return action_sequences


def find_emotional_peaks(timeline):
    """Find emotional peaks in the timeline."""
    emotional_peaks = []
    
    for i, entry in enumerate(timeline):
        confidence = entry.get('emotion_confidence', 0)
        emotion = entry.get('emotion', 'neutral')
        
        if confidence > 0.8 and emotion not in ['neutral', 'calm']:
            emotional_peaks.append({
                'time': entry['start_time'],
                'emotion': emotion,
                'confidence': confidence,
                'intensity': 'high' if confidence > 0.9 else 'medium'
            })
    
    return emotional_peaks


def analyze_character_interactions(fused_data):
    """Analyze character interactions from fused data."""
    interactions = []
    
    scenes = fused_data.get('scenes', [])
    for scene in scenes:
        characters = scene.get('characters_present', [])
        if len(characters) > 1:
            interactions.append({
                'scene_id': scene['scene_id'],
                'characters': characters,
                'interaction_type': 'dialogue' if scene.get('dialogue_count', 0) > 0 else 'visual',
                'duration': scene.get('duration', 0)
            })
    
    return interactions


def main():
    """Main function to run examples."""
    setup_logging()
    
    # Example video path (replace with your video file)
    video_path = "example_video.mp4"
    
    # Check if video file exists
    if not Path(video_path).exists():
        print(f"Video file not found: {video_path}")
        print("Please provide a valid video file path.")
        return
    
    # Run different analysis examples
    print("Running AI Film Analysis Examples...")
    print("=" * 50)
    
    # Basic analysis
    basic_analysis_example(video_path, "basic_analysis.json")
    print()
    
    # High-resolution analysis
    high_resolution_analysis_example(video_path, "high_res_analysis.json")
    print()
    
    # Custom analysis
    custom_analysis_example(video_path, "custom_analysis.json")
    print()
    
    print("All examples completed!")


if __name__ == "__main__":
    main()