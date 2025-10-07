"""
Multimodal fusion module.

Combines visual, audio, and dialogue data to create unified analysis
with emotional content, narrative function, and conflict resolution.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import json


class MultimodalFusion:
    """Handles fusion of multimodal data for comprehensive analysis."""
    
    def __init__(self):
        """Initialize multimodal fusion system."""
        self.logger = logging.getLogger(__name__)
        
        # Emotion vocabulary
        self.emotion_vocabulary = [
            'joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust',
            'calm', 'excitement', 'tension', 'relief', 'confusion',
            'anticipation', 'suspense', 'romance', 'melancholy'
        ]
        
        # Narrative function vocabulary
        self.narrative_functions = [
            'setup', 'development', 'climax', 'resolution', 'transition',
            'conflict', 'revelation', 'character_development', 'world_building',
            'foreshadowing', 'callback', 'comic_relief', 'emotional_beat'
        ]
    
    def fuse_data(self, visual_analysis: Dict[str, Any], 
                  dialogue_data: Dict[str, Any], 
                  audio_data: Dict[str, Any], 
                  scenes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Fuse multimodal data into unified analysis.
        
        Args:
            visual_analysis: Visual content analysis results
            dialogue_data: Dialogue processing results
            audio_data: Audio processing results
            scenes: Detected scenes
            
        Returns:
            Fused multimodal analysis
        """
        self.logger.info("Fusing multimodal data...")
        
        fused_analysis = {
            'metadata': {
                'fusion_timestamp': datetime.now().isoformat(),
                'data_sources': ['visual', 'audio', 'dialogue'],
                'total_scenes': len(scenes)
            },
            'scenes': [],
            'timeline_fusion': [],
            'character_analysis': {},
            'narrative_structure': {},
            'emotional_arc': [],
            'conflicts_detected': []
        }
        
        # Fuse data for each scene
        for scene in scenes:
            scene_fusion = self._fuse_scene_data(
                scene, visual_analysis, dialogue_data, audio_data
            )
            fused_analysis['scenes'].append(scene_fusion)
        
        # Generate timeline fusion
        fused_analysis['timeline_fusion'] = self._generate_timeline_fusion(
            scenes, visual_analysis, dialogue_data, audio_data
        )
        
        # Analyze characters across modalities
        fused_analysis['character_analysis'] = self._analyze_characters_multimodal(
            visual_analysis, dialogue_data
        )
        
        # Analyze narrative structure
        fused_analysis['narrative_structure'] = self._analyze_narrative_structure(
            fused_analysis['scenes']
        )
        
        # Generate emotional arc
        fused_analysis['emotional_arc'] = self._generate_emotional_arc(
            fused_analysis['scenes']
        )
        
        # Detect conflicts
        fused_analysis['conflicts_detected'] = self._detect_conflicts(
            fused_analysis['scenes']
        )
        
        return fused_analysis
    
    def _fuse_scene_data(self, scene: Dict[str, Any], 
                        visual_analysis: Dict[str, Any], 
                        dialogue_data: Dict[str, Any], 
                        audio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fuse data for a single scene."""
        scene_id = scene['scene_id']
        start_time = scene['start_time']
        end_time = scene['end_time']
        
        # Get relevant dialogue for this scene
        scene_dialogue = self._get_scene_dialogue(dialogue_data, start_time, end_time)
        
        # Get visual analysis for this scene
        scene_visual = self._get_scene_visual(visual_analysis, scene)
        
        # Get audio analysis for this scene
        scene_audio = self._get_scene_audio(audio_data, start_time, end_time)
        
        # Fuse emotional content
        emotion_fusion = self._fuse_emotions(scene_visual, scene_dialogue, scene_audio)
        
        # Determine narrative function
        narrative_function = self._determine_narrative_function(
            scene, scene_dialogue, scene_visual, scene_audio
        )
        
        # Detect visual-audio contrasts
        contrasts = self._detect_contrasts(scene_visual, scene_audio, scene_dialogue)
        
        return {
            'scene_id': scene_id,
            'start_time': start_time,
            'end_time': end_time,
            'duration': end_time - start_time,
            'visual_summary': scene_visual.get('summary', ''),
            'audio_summary': scene_audio.get('summary', ''),
            'dialogue_summary': self._summarize_dialogue(scene_dialogue),
            'emotion': emotion_fusion['primary_emotion'],
            'emotion_confidence': emotion_fusion['confidence'],
            'emotion_breakdown': emotion_fusion['breakdown'],
            'narrative_function': narrative_function,
            'contrasts_detected': contrasts,
            'characters_present': scene_visual.get('characters', []),
            'objects_detected': scene_visual.get('objects', []),
            'environment': scene_visual.get('environment', ''),
            'lighting': scene_visual.get('lighting', 'normal'),
            'camera_work': scene_visual.get('camera_work', 'static'),
            'dialogue_count': len(scene_dialogue),
            'music_present': scene_audio.get('music_detected', False),
            'sound_effects': scene_audio.get('sound_effects', [])
        }
    
    def _get_scene_dialogue(self, dialogue_data: Dict[str, Any], 
                           start_time: float, end_time: float) -> List[Dict[str, Any]]:
        """Get dialogue segments within scene timeframe."""
        aligned_dialogue = dialogue_data.get('aligned_dialogue', [])
        
        scene_dialogue = []
        for utterance in aligned_dialogue:
            if start_time <= utterance['start_time'] <= end_time:
                scene_dialogue.append(utterance)
        
        return scene_dialogue
    
    def _get_scene_visual(self, visual_analysis: Dict[str, Any], 
                         scene: Dict[str, Any]) -> Dict[str, Any]:
        """Get visual analysis for a scene."""
        # Find scene analysis
        scene_analyses = visual_analysis.get('scene_analysis', [])
        scene_visual = next(
            (s for s in scene_analyses if s['scene_id'] == scene['scene_id']), 
            {}
        )
        
        # Aggregate frame analysis for this scene
        frame_analyses = visual_analysis.get('frame_analysis', [])
        scene_frames = []
        
        for shot in scene['shots']:
            start_frame = shot['start_frame']
            end_frame = shot['end_frame']
            
            for i in range(start_frame, min(end_frame + 1, len(frame_analyses))):
                if i < len(frame_analyses):
                    scene_frames.append(frame_analyses[i])
        
        # Aggregate visual information
        all_objects = []
        all_characters = []
        all_environments = []
        all_lighting = []
        
        for frame in scene_frames:
            all_objects.extend(frame.get('objects', []))
            all_characters.extend(frame.get('characters', []))
            all_environments.append(frame.get('environment', ''))
            all_lighting.append(frame.get('lighting', 'normal'))
        
        return {
            'summary': scene_visual.get('visual_summary', ''),
            'objects': all_objects,
            'characters': all_characters,
            'environment': max(set(all_environments), key=all_environments.count) if all_environments else '',
            'lighting': max(set(all_lighting), key=all_lighting.count) if all_lighting else 'normal',
            'camera_work': scene_visual.get('camera_work', []),
            'dominant_objects': scene_visual.get('dominant_objects', [])
        }
    
    def _get_scene_audio(self, audio_data: Dict[str, Any], 
                        start_time: float, end_time: float) -> Dict[str, Any]:
        """Get audio analysis for a scene."""
        # This is a simplified implementation
        # In practice, you would analyze audio features within the time window
        
        features = audio_data.get('features', {})
        
        return {
            'summary': 'Audio analysis for scene',
            'music_detected': True,  # Placeholder
            'sound_effects': [],  # Placeholder
            'energy': float(np.mean(features.get('rms', [0.1]))),
            'tempo': features.get('tempo', 120),
            'brightness': float(np.mean(features.get('spectral_centroid', [1000])))
        }
    
    def _fuse_emotions(self, visual: Dict[str, Any], 
                      dialogue: List[Dict[str, Any]], 
                      audio: Dict[str, Any]) -> Dict[str, Any]:
        """Fuse emotional content from all modalities."""
        # Collect emotions from each modality
        visual_emotion = self._extract_visual_emotion(visual)
        dialogue_emotions = self._extract_dialogue_emotions(dialogue)
        audio_emotion = self._extract_audio_emotion(audio)
        
        # Weight emotions by confidence and modality importance
        emotions = []
        
        if visual_emotion['confidence'] > 0.3:
            emotions.append({
                'emotion': visual_emotion['emotion'],
                'confidence': visual_emotion['confidence'] * 0.3,  # Visual weight
                'source': 'visual'
            })
        
        if dialogue_emotions:
            for emotion in dialogue_emotions:
                emotions.append({
                    'emotion': emotion['emotion'],
                    'confidence': emotion['confidence'] * 0.5,  # Dialogue weight
                    'source': 'dialogue'
                })
        
        if audio_emotion['confidence'] > 0.3:
            emotions.append({
                'emotion': audio_emotion['emotion'],
                'confidence': audio_emotion['confidence'] * 0.2,  # Audio weight
                'source': 'audio'
            })
        
        # Determine primary emotion
        if not emotions:
            primary_emotion = 'neutral'
            confidence = 0.0
        else:
            # Find emotion with highest weighted confidence
            best_emotion = max(emotions, key=lambda x: x['confidence'])
            primary_emotion = best_emotion['emotion']
            confidence = best_emotion['confidence']
        
        # Create emotion breakdown
        emotion_counts = {}
        for emotion in emotions:
            emo = emotion['emotion']
            if emo not in emotion_counts:
                emotion_counts[emo] = {'count': 0, 'total_confidence': 0}
            emotion_counts[emo]['count'] += 1
            emotion_counts[emo]['total_confidence'] += emotion['confidence']
        
        breakdown = {}
        for emotion, data in emotion_counts.items():
            breakdown[emotion] = {
                'count': data['count'],
                'average_confidence': data['total_confidence'] / data['count']
            }
        
        return {
            'primary_emotion': primary_emotion,
            'confidence': confidence,
            'breakdown': breakdown,
            'sources': emotions
        }
    
    def _extract_visual_emotion(self, visual: Dict[str, Any]) -> Dict[str, Any]:
        """Extract emotion from visual content."""
        # Simple emotion mapping based on visual features
        lighting = visual.get('lighting', 'normal')
        environment = visual.get('environment', '').lower()
        
        if lighting == 'dark' or 'dark' in environment:
            return {'emotion': 'tension', 'confidence': 0.7}
        elif lighting == 'bright' or 'sunny' in environment:
            return {'emotion': 'joy', 'confidence': 0.6}
        elif 'rain' in environment or 'storm' in environment:
            return {'emotion': 'melancholy', 'confidence': 0.6}
        else:
            return {'emotion': 'neutral', 'confidence': 0.3}
    
    def _extract_dialogue_emotions(self, dialogue: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract emotions from dialogue."""
        emotions = []
        for utterance in dialogue:
            if utterance.get('emotion') and utterance.get('emotion') != 'neutral':
                emotions.append({
                    'emotion': utterance['emotion'],
                    'confidence': utterance.get('emotion_confidence', 0.5)
                })
        return emotions
    
    def _extract_audio_emotion(self, audio: Dict[str, Any]) -> Dict[str, Any]:
        """Extract emotion from audio features."""
        energy = audio.get('energy', 0.1)
        tempo = audio.get('tempo', 120)
        
        if energy > 0.2 and tempo > 140:
            return {'emotion': 'excitement', 'confidence': 0.7}
        elif energy < 0.05 and tempo < 80:
            return {'emotion': 'calm', 'confidence': 0.6}
        elif energy > 0.15 and tempo > 120:
            return {'emotion': 'tension', 'confidence': 0.6}
        else:
            return {'emotion': 'neutral', 'confidence': 0.3}
    
    def _determine_narrative_function(self, scene: Dict[str, Any], 
                                    dialogue: List[Dict[str, Any]], 
                                    visual: Dict[str, Any], 
                                    audio: Dict[str, Any]) -> str:
        """Determine narrative function of a scene."""
        # Use existing narrative function if available
        if 'narrative_function' in scene:
            return scene['narrative_function']
        
        # Analyze based on content
        dialogue_count = len(dialogue)
        shot_count = len(scene.get('shots', []))
        duration = scene.get('duration', 0)
        
        if dialogue_count > 5 and shot_count <= 3:
            return 'dialogue'
        elif shot_count > 10 and duration < 30:
            return 'action'
        elif duration > 60 and shot_count == 1:
            return 'establishing'
        elif dialogue_count == 0 and shot_count > 5:
            return 'montage'
        else:
            return 'development'
    
    def _detect_contrasts(self, visual: Dict[str, Any], 
                         audio: Dict[str, Any], 
                         dialogue: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect visual-audio contrasts (e.g., upbeat music over sad visuals)."""
        contrasts = []
        
        # Get primary emotions
        visual_emotion = self._extract_visual_emotion(visual)
        audio_emotion = self._extract_audio_emotion(audio)
        
        # Check for emotional contrast
        if visual_emotion['emotion'] != audio_emotion['emotion']:
            if (visual_emotion['emotion'] in ['sadness', 'melancholy'] and 
                audio_emotion['emotion'] in ['joy', 'excitement']):
                contrasts.append({
                    'type': 'emotional_contrast',
                    'description': 'Upbeat music over sad visuals',
                    'visual_emotion': visual_emotion['emotion'],
                    'audio_emotion': audio_emotion['emotion'],
                    'confidence': (visual_emotion['confidence'] + audio_emotion['confidence']) / 2
                })
        
        # Check for energy contrast
        visual_energy = 0.5  # Placeholder
        audio_energy = audio.get('energy', 0.1)
        
        if abs(visual_energy - audio_energy) > 0.3:
            contrasts.append({
                'type': 'energy_contrast',
                'description': 'Visual and audio energy mismatch',
                'visual_energy': visual_energy,
                'audio_energy': audio_energy,
                'confidence': 0.6
            })
        
        return contrasts
    
    def _summarize_dialogue(self, dialogue: List[Dict[str, Any]]) -> str:
        """Create a summary of dialogue in a scene."""
        if not dialogue:
            return "No dialogue"
        
        # Group by speaker
        speaker_utterances = {}
        for utterance in dialogue:
            speaker = utterance.get('speaker_id', 'unknown')
            if speaker not in speaker_utterances:
                speaker_utterances[speaker] = []
            speaker_utterances[speaker].append(utterance['text'])
        
        # Create summary
        summary_parts = []
        for speaker, utterances in speaker_utterances.items():
            if len(utterances) == 1:
                summary_parts.append(f"{speaker}: {utterances[0]}")
            else:
                summary_parts.append(f"{speaker}: {len(utterances)} utterances")
        
        return "; ".join(summary_parts)
    
    def _generate_timeline_fusion(self, scenes: List[Dict[str, Any]], 
                                 visual_analysis: Dict[str, Any], 
                                 dialogue_data: Dict[str, Any], 
                                 audio_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate fused timeline analysis."""
        timeline = []
        
        for scene in scenes:
            scene_fusion = self._fuse_scene_data(
                scene, visual_analysis, dialogue_data, audio_data
            )
            
            timeline.append({
                'start_time': scene['start_time'],
                'end_time': scene['end_time'],
                'scene_id': scene['scene_id'],
                'primary_emotion': scene_fusion['emotion'],
                'narrative_function': scene_fusion['narrative_function'],
                'characters_present': scene_fusion['characters_present'],
                'key_events': self._extract_key_events(scene_fusion),
                'contrasts': scene_fusion['contrasts_detected']
            })
        
        return timeline
    
    def _extract_key_events(self, scene_fusion: Dict[str, Any]) -> List[str]:
        """Extract key events from scene fusion."""
        events = []
        
        if scene_fusion['dialogue_count'] > 0:
            events.append(f"Dialogue exchange ({scene_fusion['dialogue_count']} utterances)")
        
        if scene_fusion['music_present']:
            events.append("Music present")
        
        if scene_fusion['sound_effects']:
            events.append(f"Sound effects ({len(scene_fusion['sound_effects'])} effects)")
        
        if scene_fusion['characters_present']:
            events.append(f"Characters present: {len(scene_fusion['characters_present'])}")
        
        return events
    
    def _analyze_characters_multimodal(self, visual_analysis: Dict[str, Any], 
                                     dialogue_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze characters across visual and dialogue modalities."""
        character_analysis = {
            'visual_characters': visual_analysis.get('character_tracking', {}),
            'dialogue_speakers': dialogue_data.get('speaker_analysis', {}),
            'character_mapping': {},
            'character_development': {}
        }
        
        # Map visual characters to dialogue speakers
        visual_chars = character_analysis['visual_characters'].get('characters', {})
        dialogue_speakers = character_analysis['dialogue_speakers'].get('speakers', {})
        
        # Simple mapping based on temporal overlap
        # In practice, this would use more sophisticated matching
        for char_id, char_data in visual_chars.items():
            character_analysis['character_mapping'][char_id] = {
                'visual_data': char_data,
                'dialogue_data': None,  # Would be filled with matching speaker
                'confidence': 0.5
            }
        
        return character_analysis
    
    def _analyze_narrative_structure(self, scenes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze overall narrative structure."""
        narrative_structure = {
            'total_scenes': len(scenes),
            'scene_types': {},
            'emotional_progression': [],
            'narrative_arc': 'unknown'
        }
        
        # Count scene types
        for scene in scenes:
            scene_type = scene.get('narrative_function', 'unknown')
            narrative_structure['scene_types'][scene_type] = narrative_structure['scene_types'].get(scene_type, 0) + 1
        
        # Track emotional progression
        for scene in scenes:
            narrative_structure['emotional_progression'].append({
                'scene_id': scene['scene_id'],
                'emotion': scene.get('emotion', 'neutral'),
                'narrative_function': scene.get('narrative_function', 'unknown')
            })
        
        return narrative_structure
    
    def _generate_emotional_arc(self, scenes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate emotional arc across scenes."""
        emotional_arc = []
        
        for i, scene in enumerate(scenes):
            emotional_arc.append({
                'scene_id': scene['scene_id'],
                'position': i / max(1, len(scenes) - 1),  # 0 to 1
                'emotion': scene.get('emotion', 'neutral'),
                'intensity': scene.get('emotion_confidence', 0.5),
                'narrative_function': scene.get('narrative_function', 'unknown')
            })
        
        return emotional_arc
    
    def _detect_conflicts(self, scenes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect narrative conflicts and tensions."""
        conflicts = []
        
        for scene in scenes:
            # Check for emotional conflicts within scene
            contrasts = scene.get('contrasts_detected', [])
            for contrast in contrasts:
                if contrast['type'] == 'emotional_contrast':
                    conflicts.append({
                        'type': 'emotional_tension',
                        'scene_id': scene['scene_id'],
                        'description': contrast['description'],
                        'confidence': contrast['confidence']
                    })
        
        return conflicts