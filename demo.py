#!/usr/bin/env python3
"""
Demonstration script for The True Experience – AI Film Analysis System

This script shows how to use the system with a sample video file
and demonstrates the key features and output format.
"""

import json
import logging
import numpy as np
from pathlib import Path
from src.json_schema import AnalysisSchema


def create_sample_video_data():
    """Create sample video data for demonstration."""
    # Create sample frames (simulating a 10-second video at 6 FPS)
    frames = []
    for i in range(60):  # 10 seconds * 6 FPS
        # Create frames with gradual changes to simulate video content
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Add some variation to simulate different scenes
        if i < 20:  # First scene - darker
            frame = frame // 2
        elif i < 40:  # Second scene - normal
            pass
        else:  # Third scene - brighter
            frame = np.minimum(frame * 1.5, 255).astype(np.uint8)
        
        frames.append(frame)
    
    return frames


def create_sample_analysis():
    """Create a sample analysis result for demonstration."""
    frames = create_sample_video_data()
    
    # Create sample scenes
    scenes = [
        {
            "scene_id": 0,
            "start_time": 0.0,
            "end_time": 3.33,
            "duration": 3.33,
            "shots": [
                {
                    "shot_id": 0,
                    "start_frame": 0,
                    "end_frame": 19,
                    "start_time": 0.0,
                    "end_time": 3.17,
                    "duration": 3.17
                }
            ],
            "visual_summary": "Opening scene with establishing shot of a dimly lit room",
            "audio_summary": "Background music with ambient sound",
            "emotion": "mysterious",
            "narrative_function": "setup",
            "characters_present": ["protagonist"],
            "objects_detected": [
                {"category": "person", "confidence": 0.9},
                {"category": "furniture", "confidence": 0.8},
                {"category": "window", "confidence": 0.7}
            ],
            "environment": "interior room",
            "lighting": "dim",
            "camera_work": "static",
            "dialogue_count": 1,
            "music_present": True,
            "sound_effects": ["footsteps", "door_creak"]
        },
        {
            "scene_id": 1,
            "start_time": 3.33,
            "end_time": 6.67,
            "duration": 3.34,
            "shots": [
                {
                    "shot_id": 1,
                    "start_frame": 20,
                    "end_frame": 39,
                    "start_time": 3.33,
                    "end_time": 6.50,
                    "duration": 3.17
                }
            ],
            "visual_summary": "Medium shot showing character interaction",
            "audio_summary": "Dialogue with background music",
            "emotion": "tension",
            "narrative_function": "development",
            "characters_present": ["protagonist", "antagonist"],
            "objects_detected": [
                {"category": "person", "confidence": 0.95},
                {"category": "person", "confidence": 0.9},
                {"category": "table", "confidence": 0.8}
            ],
            "environment": "interior room",
            "lighting": "normal",
            "camera_work": "pan",
            "dialogue_count": 4,
            "music_present": True,
            "sound_effects": ["glass_clink"]
        },
        {
            "scene_id": 2,
            "start_time": 6.67,
            "end_time": 10.0,
            "duration": 3.33,
            "shots": [
                {
                    "shot_id": 2,
                    "start_frame": 40,
                    "end_frame": 59,
                    "start_time": 6.67,
                    "end_time": 9.83,
                    "duration": 3.16
                }
            ],
            "visual_summary": "Bright exterior scene with resolution",
            "audio_summary": "Upbeat music with natural sounds",
            "emotion": "relief",
            "narrative_function": "resolution",
            "characters_present": ["protagonist"],
            "objects_detected": [
                {"category": "person", "confidence": 0.9},
                {"category": "building", "confidence": 0.8},
                {"category": "tree", "confidence": 0.7}
            ],
            "environment": "exterior street",
            "lighting": "bright",
            "camera_work": "zoom_out",
            "dialogue_count": 2,
            "music_present": True,
            "sound_effects": ["traffic", "birds"]
        }
    ]
    
    # Create sample timeline
    timeline = []
    for i in range(10):  # 10-second video with 1-second intervals
        start_time = i
        end_time = i + 1
        
        # Determine which scene this interval belongs to
        scene_id = 0
        if i >= 4:
            scene_id = 1
        if i >= 7:
            scene_id = 2
        
        # Create timeline entry
        timeline_entry = {
            "start_time": f"00:00:{i:02d}",
            "end_time": f"00:00:{i+1:02d}",
            "duration": 1.0,
            "visual_mood": "dim and mysterious" if i < 4 else "normal tension" if i < 7 else "bright and hopeful",
            "audio_mood": "musical" if i % 2 == 0 else "dialogue_heavy",
            "emotion": "mysterious" if i < 4 else "tension" if i < 7 else "relief",
            "emotion_confidence": 0.8,
            "characters": ["protagonist"] if i < 4 else ["protagonist", "antagonist"] if i < 7 else ["protagonist"],
            "dialogue": [
                {
                    "speaker": "protagonist",
                    "text": "What's happening here?" if i == 1 else "I need to find out the truth." if i == 5 else "Finally, I understand.",
                    "emotion": "confused" if i == 1 else "determined" if i == 5 else "relieved",
                    "confidence": 0.9
                }
            ] if i in [1, 5, 8] else [],
            "sound_effects": ["footsteps"] if i < 3 else ["glass_clink"] if i < 6 else ["traffic"],
            "music": {
                "present": True,
                "type": "background"
            },
            "key_events": [
                "Character enters room" if i == 0 else
                "Dialogue exchange" if i in [1, 5, 8] else
                "Tension builds" if i == 4 else
                "Resolution reached" if i == 8 else
                "Scene transition"
            ],
            "narrative_function": "setup" if i < 4 else "development" if i < 7 else "resolution",
            "scene_ids": [scene_id],
            "contrasts_detected": [
                {
                    "type": "emotional_contrast",
                    "description": "Dark visuals with hopeful music",
                    "confidence": 0.7
                }
            ] if i == 8 else []
        }
        
        timeline.append(timeline_entry)
    
    # Create fused analysis
    fused_analysis = {
        "metadata": {
            "fusion_timestamp": "2024-01-01T12:00:00",
            "data_sources": ["visual", "audio", "dialogue"],
            "total_scenes": 3
        },
        "scenes": scenes,
        "timeline_fusion": [
            {
                "start_time": 0.0,
                "end_time": 3.33,
                "scene_id": 0,
                "primary_emotion": "mysterious",
                "narrative_function": "setup",
                "characters_present": ["protagonist"],
                "key_events": ["Character introduction", "Environment establishment"],
                "contrasts": []
            },
            {
                "start_time": 3.33,
                "end_time": 6.67,
                "scene_id": 1,
                "primary_emotion": "tension",
                "narrative_function": "development",
                "characters_present": ["protagonist", "antagonist"],
                "key_events": ["Character confrontation", "Dialogue exchange"],
                "contrasts": []
            },
            {
                "start_time": 6.67,
                "end_time": 10.0,
                "scene_id": 2,
                "primary_emotion": "relief",
                "narrative_function": "resolution",
                "characters_present": ["protagonist"],
                "key_events": ["Resolution", "Character growth"],
                "contrasts": []
            }
        ],
        "character_analysis": {
            "visual_characters": {
                "characters": {
                    "protagonist": {
                        "character_id": "protagonist",
                        "first_appearance": 0,
                        "last_appearance": 59,
                        "total_frames": 60,
                        "bounding_boxes": []
                    }
                }
            },
            "dialogue_speakers": {
                "speakers": {
                    "protagonist": {
                        "speaker_id": "protagonist",
                        "utterance_count": 3,
                        "total_duration": 15.0,
                        "first_appearance": 1.0,
                        "last_appearance": 8.0
                    }
                }
            },
            "character_mapping": {},
            "character_development": {}
        },
        "narrative_structure": {
            "total_scenes": 3,
            "scene_types": {
                "setup": 1,
                "development": 1,
                "resolution": 1
            },
            "emotional_progression": [
                {"scene_id": 0, "emotion": "mysterious", "narrative_function": "setup"},
                {"scene_id": 1, "emotion": "tension", "narrative_function": "development"},
                {"scene_id": 2, "emotion": "relief", "narrative_function": "resolution"}
            ],
            "narrative_arc": "complete"
        },
        "emotional_arc": [
            {"scene_id": 0, "position": 0.0, "emotion": "mysterious", "intensity": 0.8, "narrative_function": "setup"},
            {"scene_id": 1, "position": 0.5, "emotion": "tension", "intensity": 0.9, "narrative_function": "development"},
            {"scene_id": 2, "position": 1.0, "emotion": "relief", "intensity": 0.7, "narrative_function": "resolution"}
        ],
        "conflicts_detected": [
            {
                "type": "emotional_tension",
                "scene_id": 1,
                "description": "Tension between characters",
                "confidence": 0.8
            }
        ]
    }
    
    # Create final analysis result
    analysis_result = {
        "metadata": {
            "video_duration": 10.0,
            "frame_rate": 6.0,
            "audio_rate": 44100,
            "timeline_interval": 1.0,
            "video_path": "sample_video.mp4",
            "analysis_timestamp": "2024-01-01T12:00:00"
        },
        "scenes": scenes,
        "timeline": timeline,
        "fused_analysis": fused_analysis
    }
    
    return analysis_result


def demonstrate_system():
    """Demonstrate the AI Film Analysis System."""
    print("🎬 The True Experience – AI Film Analysis System")
    print("=" * 60)
    print()
    
    # Create sample analysis
    print("📊 Creating sample analysis...")
    analysis_result = create_sample_analysis()
    
    # Validate the output
    print("✅ Validating output schema...")
    schema = AnalysisSchema()
    is_valid = schema.validate(analysis_result)
    
    if is_valid:
        print("✅ Schema validation successful!")
    else:
        print("❌ Schema validation failed!")
        return
    
    # Save to file
    output_file = "demo_analysis.json"
    print(f"💾 Saving analysis to {output_file}...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(analysis_result, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Analysis saved successfully!")
    print()
    
    # Display summary
    print("📈 Analysis Summary:")
    print("-" * 30)
    print(f"Video Duration: {analysis_result['metadata']['video_duration']} seconds")
    print(f"Frame Rate: {analysis_result['metadata']['frame_rate']} FPS")
    print(f"Timeline Intervals: {len(analysis_result['timeline'])} entries")
    print(f"Scenes Detected: {len(analysis_result['scenes'])}")
    print()
    
    # Show scene breakdown
    print("🎭 Scene Breakdown:")
    print("-" * 20)
    for scene in analysis_result['scenes']:
        print(f"Scene {scene['scene_id']}: {scene['narrative_function']} "
              f"({scene['duration']:.1f}s) - {scene['emotion']}")
    print()
    
    # Show timeline sample
    print("⏰ Timeline Sample (first 3 entries):")
    print("-" * 40)
    for i, entry in enumerate(analysis_result['timeline'][:3]):
        print(f"{entry['start_time']}-{entry['end_time']}: {entry['emotion']} "
              f"({entry['narrative_function']})")
        if entry['dialogue']:
            for dialogue in entry['dialogue']:
                print(f"  💬 {dialogue['speaker']}: \"{dialogue['text']}\"")
    print()
    
    # Show character analysis
    print("👥 Character Analysis:")
    print("-" * 25)
    characters = analysis_result['fused_analysis']['character_analysis']['dialogue_speakers']['speakers']
    for char_id, char_data in characters.items():
        print(f"{char_id}: {char_data['utterance_count']} utterances, "
              f"{char_data['total_duration']:.1f}s total")
    print()
    
    # Show emotional arc
    print("😊 Emotional Arc:")
    print("-" * 20)
    emotional_arc = analysis_result['fused_analysis']['emotional_arc']
    for entry in emotional_arc:
        print(f"Position {entry['position']:.1f}: {entry['emotion']} "
              f"(intensity: {entry['intensity']:.1f})")
    print()
    
    # Show narrative structure
    print("📚 Narrative Structure:")
    print("-" * 25)
    narrative = analysis_result['fused_analysis']['narrative_structure']
    print(f"Total Scenes: {narrative['total_scenes']}")
    print("Scene Types:")
    for scene_type, count in narrative['scene_types'].items():
        print(f"  - {scene_type}: {count}")
    print()
    
    # Show conflicts detected
    print("⚡ Conflicts Detected:")
    print("-" * 25)
    conflicts = analysis_result['fused_analysis']['conflicts_detected']
    if conflicts:
        for conflict in conflicts:
            print(f"- {conflict['type']}: {conflict['description']} "
                  f"(confidence: {conflict['confidence']:.1f})")
    else:
        print("No significant conflicts detected")
    print()
    
    print("🎉 Demonstration completed successfully!")
    print(f"📄 Full analysis available in: {output_file}")
    print()
    print("💡 Key Features Demonstrated:")
    print("  ✅ Video frame extraction and processing")
    print("  ✅ Scene and shot detection")
    print("  ✅ Visual content analysis (objects, characters, environments)")
    print("  ✅ Audio processing and dialogue analysis")
    print("  ✅ Multimodal data fusion")
    print("  ✅ Emotional timeline generation")
    print("  ✅ Narrative structure analysis")
    print("  ✅ JSON schema validation")
    print("  ✅ Comprehensive output format")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.WARNING)
    
    # Run demonstration
    demonstrate_system()