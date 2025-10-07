#!/usr/bin/env python3
"""
The True Experience – AI Film Analysis System

A comprehensive video analysis system that processes video files and generates
detailed, timestamped JSON summaries of all visual events, dialogue, characters,
sound, music, emotional content, and narrative structure.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional

from src.video_processor import VideoProcessor
from src.audio_processor import AudioProcessor
from src.scene_detector import SceneDetector
from src.visual_analyzer import VisualAnalyzer
from src.dialogue_processor import DialogueProcessor
from src.multimodal_fusion import MultimodalFusion
from src.timeline_generator import TimelineGenerator
from src.json_schema import AnalysisSchema


def setup_logging(verbose: bool = False) -> None:
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('analysis.log')
        ]
    )


def main():
    """Main entry point for the video analysis system."""
    parser = argparse.ArgumentParser(
        description="The True Experience – AI Film Analysis System"
    )
    parser.add_argument(
        "video_path",
        type=str,
        help="Path to the input video file"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="analysis_output.json",
        help="Output JSON file path (default: analysis_output.json)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=6.0,
        help="Frame rate for analysis (default: 6.0 fps)"
    )
    parser.add_argument(
        "--timeline-interval",
        type=float,
        default=5.0,
        help="Timeline interval in seconds (default: 5.0s)"
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU acceleration when available"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Validate input file
    video_path = Path(args.video_path)
    if not video_path.exists():
        logger.error(f"Video file not found: {video_path}")
        sys.exit(1)
    
    try:
        logger.info(f"Starting analysis of: {video_path}")
        logger.info(f"Output will be saved to: {args.output}")
        
        # Initialize components
        video_processor = VideoProcessor(fps=args.fps, use_gpu=args.gpu)
        audio_processor = AudioProcessor(use_gpu=args.gpu)
        scene_detector = SceneDetector()
        visual_analyzer = VisualAnalyzer(use_gpu=args.gpu)
        dialogue_processor = DialogueProcessor(use_gpu=args.gpu)
        multimodal_fusion = MultimodalFusion()
        timeline_generator = TimelineGenerator(interval=args.timeline_interval)
        
        # Process video
        logger.info("Step 1: Processing video and extracting frames...")
        video_data = video_processor.process_video(str(video_path))
        
        # Process audio
        logger.info("Step 2: Processing audio and extracting features...")
        audio_data = audio_processor.process_audio(str(video_path))
        
        # Detect scenes and shots
        logger.info("Step 3: Detecting scenes and shots...")
        scenes = scene_detector.detect_scenes(video_data['frames'])
        
        # Analyze visual content
        logger.info("Step 4: Analyzing visual content...")
        visual_analysis = visual_analyzer.analyze_frames(video_data['frames'], scenes)
        
        # Process dialogue
        logger.info("Step 5: Processing dialogue and speech...")
        dialogue_data = dialogue_processor.process_dialogue(audio_data['audio_path'])
        
        # Fuse multimodal data
        logger.info("Step 6: Fusing multimodal data...")
        fused_data = multimodal_fusion.fuse_data(
            visual_analysis, dialogue_data, audio_data, scenes
        )
        
        # Generate timeline
        logger.info("Step 7: Generating emotional and structural timeline...")
        timeline = timeline_generator.generate_timeline(fused_data, video_data['duration'])
        
        # Create final output
        logger.info("Step 8: Creating final JSON output...")
        analysis_result = {
            "metadata": {
                "video_duration": video_data['duration'],
                "frame_rate": args.fps,
                "audio_rate": audio_data['sample_rate'],
                "timeline_interval": args.timeline_interval,
                "video_path": str(video_path),
                "analysis_timestamp": video_data['timestamp']
            },
            "scenes": scenes,
            "timeline": timeline,
            "fused_analysis": fused_data
        }
        
        # Validate and save output
        schema = AnalysisSchema()
        if schema.validate(analysis_result):
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(analysis_result, f, indent=2, ensure_ascii=False)
            logger.info(f"Analysis complete! Output saved to: {args.output}")
        else:
            logger.error("Output validation failed!")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        if args.verbose:
            logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main()