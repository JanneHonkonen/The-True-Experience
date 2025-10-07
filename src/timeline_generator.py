"""
Timeline generation module.

Creates detailed emotional and structural timeline with fixed intervals,
combining all multimodal data for comprehensive analysis.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional
from datetime import timedelta
import json


class TimelineGenerator:
    """Generates comprehensive timeline analysis with fixed intervals."""
    
    def __init__(self, interval: float = 5.0):
        """
        Initialize timeline generator.
        
        Args:
            interval: Timeline interval in seconds
        """
        self.interval = interval
        self.logger = logging.getLogger(__name__)
    
    def generate_timeline(self, fused_data: Dict[str, Any], 
                         video_duration: float) -> List[Dict[str, Any]]:
        """
        Generate comprehensive timeline with fixed intervals.
        
        Args:
            fused_data: Fused multimodal analysis data
            video_duration: Total video duration in seconds
            
        Returns:
            List of timeline entries
        """
        self.logger.info(f"Generating timeline with {self.interval}s intervals for {video_duration:.2f}s video")
        
        timeline = []
        current_time = 0.0
        
        while current_time < video_duration:
            end_time = min(current_time + self.interval, video_duration)
            
            timeline_entry = self._create_timeline_entry(
                current_time, end_time, fused_data, video_duration
            )
            timeline.append(timeline_entry)
            
            current_time = end_time
        
        self.logger.info(f"Generated {len(timeline)} timeline entries")
        return timeline
    
    def _create_timeline_entry(self, start_time: float, end_time: float, 
                              fused_data: Dict[str, Any], video_duration: float) -> Dict[str, Any]:
        """Create a single timeline entry for the given time interval."""
        # Find relevant scenes for this time interval
        relevant_scenes = self._find_relevant_scenes(
            start_time, end_time, fused_data.get('scenes', [])
        )
        
        # Find relevant dialogue for this time interval
        relevant_dialogue = self._find_relevant_dialogue(
            start_time, end_time, fused_data
        )
        
        # Analyze visual mood for this interval
        visual_mood = self._analyze_visual_mood(
            start_time, end_time, relevant_scenes, fused_data
        )
        
        # Analyze audio mood for this interval
        audio_mood = self._analyze_audio_mood(
            start_time, end_time, relevant_scenes, fused_data
        )
        
        # Determine dominant emotion
        dominant_emotion = self._determine_dominant_emotion(
            relevant_scenes, relevant_dialogue, visual_mood, audio_mood
        )
        
        # Identify characters present
        characters_present = self._identify_characters_present(
            start_time, end_time, relevant_scenes, fused_data
        )
        
        # Extract key events
        key_events = self._extract_key_events(
            start_time, end_time, relevant_scenes, relevant_dialogue
        )
        
        # Determine narrative state
        narrative_state = self._determine_narrative_state(
            start_time, video_duration, relevant_scenes
        )
        
        # Analyze sound effects and music
        sound_analysis = self._analyze_sound_effects(
            start_time, end_time, relevant_scenes, fused_data
        )
        
        return {
            'start_time': self._format_timestamp(start_time),
            'end_time': self._format_timestamp(end_time),
            'duration': end_time - start_time,
            'visual_mood': visual_mood,
            'audio_mood': audio_mood,
            'emotion': dominant_emotion['emotion'],
            'emotion_confidence': dominant_emotion['confidence'],
            'characters': characters_present,
            'dialogue': relevant_dialogue,
            'sound_effects': sound_analysis['sound_effects'],
            'music': sound_analysis['music'],
            'key_events': key_events,
            'narrative_function': narrative_state,
            'scene_ids': [scene['scene_id'] for scene in relevant_scenes],
            'contrasts_detected': self._detect_interval_contrasts(
                visual_mood, audio_mood, relevant_dialogue
            )
        }
    
    def _find_relevant_scenes(self, start_time: float, end_time: float, 
                             scenes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find scenes that overlap with the time interval."""
        relevant_scenes = []
        
        for scene in scenes:
            scene_start = scene['start_time']
            scene_end = scene['end_time']
            
            # Check for overlap
            if (scene_start < end_time and scene_end > start_time):
                relevant_scenes.append(scene)
        
        return relevant_scenes
    
    def _find_relevant_dialogue(self, start_time: float, end_time: float, 
                               fused_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find dialogue segments within the time interval."""
        # This would typically come from the dialogue processor
        # For now, we'll extract from scene data
        relevant_dialogue = []
        
        scenes = fused_data.get('scenes', [])
        for scene in scenes:
            if (scene['start_time'] < end_time and scene['end_time'] > start_time):
                # Extract dialogue from scene (simplified)
                dialogue_count = scene.get('dialogue_count', 0)
                if dialogue_count > 0:
                    relevant_dialogue.append({
                        'speaker': 'unknown',
                        'text': f"Dialogue segment ({dialogue_count} utterances)",
                        'emotion': scene.get('emotion', 'neutral'),
                        'confidence': scene.get('emotion_confidence', 0.5)
                    })
        
        return relevant_dialogue
    
    def _analyze_visual_mood(self, start_time: float, end_time: float, 
                           relevant_scenes: List[Dict[str, Any]], 
                           fused_data: Dict[str, Any]) -> str:
        """Analyze visual mood for the time interval."""
        if not relevant_scenes:
            return "unknown"
        
        # Aggregate visual information from scenes
        lighting_styles = []
        environments = []
        emotions = []
        
        for scene in relevant_scenes:
            lighting_styles.append(scene.get('lighting', 'normal'))
            environments.append(scene.get('environment', ''))
            emotions.append(scene.get('emotion', 'neutral'))
        
        # Determine dominant visual mood
        if not lighting_styles:
            return "unknown"
        
        # Simple mood determination based on lighting and environment
        dominant_lighting = max(set(lighting_styles), key=lighting_styles.count)
        dominant_environment = max(set(environments), key=environments.count) if environments else ""
        
        if dominant_lighting == 'dark':
            return "dark and moody"
        elif dominant_lighting == 'bright':
            return "bright and cheerful"
        elif 'rain' in dominant_environment.lower():
            return "gloomy and wet"
        elif 'sunny' in dominant_environment.lower():
            return "warm and sunny"
        else:
            return "neutral"
    
    def _analyze_audio_mood(self, start_time: float, end_time: float, 
                          relevant_scenes: List[Dict[str, Any]], 
                          fused_data: Dict[str, Any]) -> str:
        """Analyze audio mood for the time interval."""
        if not relevant_scenes:
            return "silence"
        
        # Aggregate audio information from scenes
        music_present = any(scene.get('music_present', False) for scene in relevant_scenes)
        sound_effects = []
        
        for scene in relevant_scenes:
            sound_effects.extend(scene.get('sound_effects', []))
        
        # Determine audio mood
        if music_present:
            return "musical"
        elif sound_effects:
            return "sound_effects_heavy"
        else:
            return "ambient"
    
    def _determine_dominant_emotion(self, relevant_scenes: List[Dict[str, Any]], 
                                  relevant_dialogue: List[Dict[str, Any]], 
                                  visual_mood: str, audio_mood: str) -> Dict[str, Any]:
        """Determine dominant emotion for the time interval."""
        emotions = []
        confidences = []
        
        # Collect emotions from scenes
        for scene in relevant_scenes:
            emotion = scene.get('emotion', 'neutral')
            confidence = scene.get('emotion_confidence', 0.5)
            emotions.append(emotion)
            confidences.append(confidence)
        
        # Collect emotions from dialogue
        for utterance in relevant_dialogue:
            emotion = utterance.get('emotion', 'neutral')
            confidence = utterance.get('confidence', 0.5)
            emotions.append(emotion)
            confidences.append(confidence)
        
        # Determine dominant emotion
        if not emotions:
            return {'emotion': 'neutral', 'confidence': 0.0}
        
        # Find most common emotion
        emotion_counts = {}
        emotion_confidences = {}
        
        for emotion, confidence in zip(emotions, confidences):
            if emotion not in emotion_counts:
                emotion_counts[emotion] = 0
                emotion_confidences[emotion] = []
            emotion_counts[emotion] += 1
            emotion_confidences[emotion].append(confidence)
        
        # Find emotion with highest count and confidence
        best_emotion = max(emotion_counts.items(), key=lambda x: x[1])
        emotion_name = best_emotion[0]
        avg_confidence = np.mean(emotion_confidences[emotion_name])
        
        return {
            'emotion': emotion_name,
            'confidence': avg_confidence
        }
    
    def _identify_characters_present(self, start_time: float, end_time: float, 
                                   relevant_scenes: List[Dict[str, Any]], 
                                   fused_data: Dict[str, Any]) -> List[str]:
        """Identify characters present in the time interval."""
        characters = set()
        
        for scene in relevant_scenes:
            scene_characters = scene.get('characters_present', [])
            for char in scene_characters:
                if isinstance(char, dict):
                    characters.add(char.get('character_id', 'unknown'))
                else:
                    characters.add(str(char))
        
        return list(characters)
    
    def _extract_key_events(self, start_time: float, end_time: float, 
                          relevant_scenes: List[Dict[str, Any]], 
                          relevant_dialogue: List[Dict[str, Any]]) -> List[str]:
        """Extract key events for the time interval."""
        events = []
        
        # Add scene-based events
        for scene in relevant_scenes:
            if scene.get('dialogue_count', 0) > 0:
                events.append(f"Dialogue exchange ({scene['dialogue_count']} utterances)")
            
            if scene.get('music_present', False):
                events.append("Music present")
            
            if scene.get('sound_effects'):
                events.append(f"Sound effects ({len(scene['sound_effects'])} effects)")
        
        # Add dialogue events
        if relevant_dialogue:
            events.append(f"Speech activity ({len(relevant_dialogue)} segments)")
        
        # Add narrative events
        for scene in relevant_scenes:
            narrative_function = scene.get('narrative_function', '')
            if narrative_function and narrative_function != 'development':
                events.append(f"Narrative: {narrative_function}")
        
        return events
    
    def _determine_narrative_state(self, start_time: float, video_duration: float, 
                                 relevant_scenes: List[Dict[str, Any]]) -> str:
        """Determine narrative state based on position and content."""
        # Determine position in video
        position = start_time / video_duration
        
        # Get narrative functions from scenes
        narrative_functions = [scene.get('narrative_function', '') for scene in relevant_scenes]
        
        # Determine state based on position and content
        if position < 0.1:
            return "setup"
        elif position > 0.9:
            return "resolution"
        elif any(func in ['climax', 'conflict'] for func in narrative_functions):
            return "climax"
        elif any(func in ['action', 'tension'] for func in narrative_functions):
            return "development"
        else:
            return "development"
    
    def _analyze_sound_effects(self, start_time: float, end_time: float, 
                             relevant_scenes: List[Dict[str, Any]], 
                             fused_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze sound effects and music for the time interval."""
        sound_effects = []
        music_info = {'present': False, 'type': 'none'}
        
        for scene in relevant_scenes:
            # Collect sound effects
            scene_sound_effects = scene.get('sound_effects', [])
            sound_effects.extend(scene_sound_effects)
            
            # Check for music
            if scene.get('music_present', False):
                music_info['present'] = True
                music_info['type'] = 'background'
        
        return {
            'sound_effects': sound_effects,
            'music': music_info
        }
    
    def _detect_interval_contrasts(self, visual_mood: str, audio_mood: str, 
                                 relevant_dialogue: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect contrasts within the time interval."""
        contrasts = []
        
        # Check for visual-audio mood contrast
        if visual_mood == "dark and moody" and audio_mood == "musical":
            contrasts.append({
                'type': 'mood_contrast',
                'description': 'Dark visuals with musical accompaniment',
                'confidence': 0.7
            })
        
        # Check for dialogue-emotion contrast
        if relevant_dialogue:
            dialogue_emotions = [utterance.get('emotion', 'neutral') for utterance in relevant_dialogue]
            if 'joy' in dialogue_emotions and visual_mood == "dark and moody":
                contrasts.append({
                    'type': 'emotional_contrast',
                    'description': 'Joyful dialogue over dark visuals',
                    'confidence': 0.6
                })
        
        return contrasts
    
    def _format_timestamp(self, seconds: float) -> str:
        """Format timestamp as HH:MM:SS."""
        return str(timedelta(seconds=seconds))
    
    def generate_summary_statistics(self, timeline: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary statistics for the timeline."""
        if not timeline:
            return {}
        
        # Emotion distribution
        emotions = [entry['emotion'] for entry in timeline]
        emotion_counts = {}
        for emotion in emotions:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        # Character appearances
        all_characters = set()
        for entry in timeline:
            all_characters.update(entry.get('characters', []))
        
        # Narrative function distribution
        narrative_functions = [entry['narrative_function'] for entry in timeline]
        function_counts = {}
        for func in narrative_functions:
            function_counts[func] = function_counts.get(func, 0) + 1
        
        # Timeline duration
        total_duration = sum(entry['duration'] for entry in timeline)
        
        return {
            'total_entries': len(timeline),
            'total_duration': total_duration,
            'emotion_distribution': emotion_counts,
            'character_count': len(all_characters),
            'characters': list(all_characters),
            'narrative_function_distribution': function_counts,
            'average_emotion_confidence': np.mean([
                entry.get('emotion_confidence', 0) for entry in timeline
            ])
        }