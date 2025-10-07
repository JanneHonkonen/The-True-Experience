"""
Dialogue processing module.

Handles speech-to-text conversion, speaker identification,
emotion analysis, and dialogue alignment with timestamps.
"""

import whisper
import torch
import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional
import librosa
from scipy.spatial.distance import cosine
from sklearn.cluster import KMeans
import re
from datetime import timedelta


class DialogueProcessor:
    """Handles dialogue processing and speech analysis."""
    
    def __init__(self, model_size: str = "base", use_gpu: bool = False):
        """
        Initialize dialogue processor.
        
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
            use_gpu: Whether to use GPU acceleration
        """
        self.model_size = model_size
        self.use_gpu = use_gpu
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"Initializing dialogue processor with {model_size} model on {self.device}")
        
        # Load Whisper model
        self.whisper_model = whisper.load_model(model_size, device=self.device)
        
        # Speaker identification parameters
        self.speaker_embeddings = {}
        self.speaker_counter = 0
    
    def process_dialogue(self, audio_path: str) -> Dict[str, Any]:
        """
        Process dialogue from audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with dialogue analysis results
        """
        self.logger.info(f"Processing dialogue from: {audio_path}")
        
        # Load audio
        audio, sr = librosa.load(audio_path, sr=16000)  # Whisper expects 16kHz
        
        # Transcribe audio
        transcription = self._transcribe_audio(audio)
        
        # Process segments
        dialogue_segments = self._process_segments(transcription['segments'])
        
        # Identify speakers
        speaker_analysis = self._identify_speakers(dialogue_segments, audio, sr)
        
        # Analyze emotions
        emotion_analysis = self._analyze_emotions(dialogue_segments, audio, sr)
        
        # Align with timestamps
        aligned_dialogue = self._align_dialogue(dialogue_segments, speaker_analysis, emotion_analysis)
        
        return {
            'transcription': transcription,
            'dialogue_segments': dialogue_segments,
            'speaker_analysis': speaker_analysis,
            'emotion_analysis': emotion_analysis,
            'aligned_dialogue': aligned_dialogue,
            'total_speakers': len(speaker_analysis['speakers']),
            'total_utterances': len(dialogue_segments)
        }
    
    def _transcribe_audio(self, audio: np.ndarray) -> Dict[str, Any]:
        """Transcribe audio using Whisper."""
        self.logger.debug("Transcribing audio with Whisper...")
        
        try:
            result = self.whisper_model.transcribe(audio, language="en")
            
            # Process segments
            processed_segments = []
            for segment in result['segments']:
                processed_segments.append({
                    'id': segment['id'],
                    'start': segment['start'],
                    'end': segment['end'],
                    'text': segment['text'].strip(),
                    'no_speech_prob': segment.get('no_speech_prob', 0.0),
                    'confidence': 1.0 - segment.get('no_speech_prob', 0.0)
                })
            
            return {
                'text': result['text'],
                'language': result.get('language', 'en'),
                'segments': processed_segments
            }
            
        except Exception as e:
            self.logger.error(f"Transcription failed: {e}")
            return {
                'text': '',
                'language': 'en',
                'segments': []
            }
    
    def _process_segments(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process and clean transcription segments."""
        processed = []
        
        for segment in segments:
            # Clean text
            text = segment['text'].strip()
            if not text or text == '':
                continue
            
            # Remove common transcription artifacts
            text = re.sub(r'\b(um|uh|er|ah)\b', '', text, flags=re.IGNORECASE)
            text = re.sub(r'\s+', ' ', text).strip()
            
            if text:
                processed.append({
                    'id': segment['id'],
                    'start_time': segment['start'],
                    'end_time': segment['end'],
                    'duration': segment['end'] - segment['start'],
                    'text': text,
                    'confidence': segment['confidence'],
                    'no_speech_prob': segment['no_speech_prob']
                })
        
        return processed
    
    def _identify_speakers(self, segments: List[Dict[str, Any]], 
                          audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """Identify speakers using audio features."""
        self.logger.debug("Identifying speakers...")
        
        if not segments:
            return {'speakers': {}, 'speaker_assignments': []}
        
        # Extract audio features for each segment
        segment_features = []
        for segment in segments:
            start_sample = int(segment['start_time'] * sr)
            end_sample = int(segment['end_time'] * sr)
            
            if start_sample < len(audio) and end_sample <= len(audio):
                segment_audio = audio[start_sample:end_sample]
                features = self._extract_speaker_features(segment_audio, sr)
                segment_features.append(features)
            else:
                segment_features.append(None)
        
        # Cluster speakers
        valid_features = [f for f in segment_features if f is not None]
        if not valid_features:
            return {'speakers': {}, 'speaker_assignments': []}
        
        # Use K-means clustering for speaker identification
        n_speakers = min(len(valid_features), 5)  # Assume max 5 speakers
        if len(valid_features) > 1:
            kmeans = KMeans(n_clusters=n_speakers, random_state=42, n_init=10)
            speaker_labels = kmeans.fit_predict(valid_features)
        else:
            speaker_labels = [0]
        
        # Create speaker assignments
        speaker_assignments = []
        feature_idx = 0
        
        for i, segment in enumerate(segments):
            if segment_features[i] is not None:
                speaker_id = f"speaker_{speaker_labels[feature_idx]}"
                speaker_assignments.append({
                    'segment_id': segment['id'],
                    'speaker_id': speaker_id,
                    'confidence': 0.8  # Placeholder confidence
                })
                feature_idx += 1
            else:
                speaker_assignments.append({
                    'segment_id': segment['id'],
                    'speaker_id': 'unknown',
                    'confidence': 0.0
                })
        
        # Create speaker profiles
        speakers = {}
        for assignment in speaker_assignments:
            speaker_id = assignment['speaker_id']
            if speaker_id not in speakers:
                speakers[speaker_id] = {
                    'speaker_id': speaker_id,
                    'utterance_count': 0,
                    'total_duration': 0.0,
                    'first_appearance': None,
                    'last_appearance': None
                }
            
            # Find corresponding segment
            segment = next((s for s in segments if s['id'] == assignment['segment_id']), None)
            if segment:
                speakers[speaker_id]['utterance_count'] += 1
                speakers[speaker_id]['total_duration'] += segment['duration']
                
                if speakers[speaker_id]['first_appearance'] is None:
                    speakers[speaker_id]['first_appearance'] = segment['start_time']
                speakers[speaker_id]['last_appearance'] = segment['end_time']
        
        return {
            'speakers': speakers,
            'speaker_assignments': speaker_assignments
        }
    
    def _extract_speaker_features(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Extract features for speaker identification."""
        if len(audio) == 0:
            return np.zeros(13)  # Return zero vector if no audio
        
        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        
        # Extract additional features
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(audio))
        rms_energy = np.mean(librosa.feature.rms(y=audio))
        
        # Combine features
        features = np.concatenate([
            mfcc_mean,
            [spectral_centroid, zero_crossing_rate, rms_energy]
        ])
        
        return features
    
    def _analyze_emotions(self, segments: List[Dict[str, Any]], 
                         audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """Analyze emotional content of dialogue."""
        self.logger.debug("Analyzing emotions in dialogue...")
        
        emotion_analysis = {
            'segment_emotions': [],
            'overall_emotion': 'neutral',
            'emotion_timeline': []
        }
        
        for segment in segments:
            start_sample = int(segment['start_time'] * sr)
            end_sample = int(segment['end_time'] * sr)
            
            if start_sample < len(audio) and end_sample <= len(audio):
                segment_audio = audio[start_sample:end_sample]
                emotion = self._classify_emotion(segment_audio, sr, segment['text'])
                
                emotion_analysis['segment_emotions'].append({
                    'segment_id': segment['id'],
                    'emotion': emotion['emotion'],
                    'confidence': emotion['confidence'],
                    'valence': emotion['valence'],
                    'arousal': emotion['arousal']
                })
            else:
                emotion_analysis['segment_emotions'].append({
                    'segment_id': segment['id'],
                    'emotion': 'neutral',
                    'confidence': 0.0,
                    'valence': 0.5,
                    'arousal': 0.5
                })
        
        # Determine overall emotion
        if emotion_analysis['segment_emotions']:
            emotions = [e['emotion'] for e in emotion_analysis['segment_emotions']]
            emotion_analysis['overall_emotion'] = max(set(emotions), key=emotions.count)
        
        return emotion_analysis
    
    def _classify_emotion(self, audio: np.ndarray, sr: int, text: str) -> Dict[str, Any]:
        """Classify emotion from audio and text features."""
        # Simple emotion classification based on audio features
        # In practice, this would use trained ML models
        
        # Analyze audio features
        pitch = np.mean(librosa.piptrack(y=audio, sr=sr)[0])
        energy = np.mean(librosa.feature.rms(y=audio))
        tempo = librosa.beat.tempo(y=audio, sr=sr)
        
        # Analyze text sentiment (simplified)
        text_lower = text.lower()
        positive_words = ['good', 'great', 'excellent', 'wonderful', 'amazing', 'love', 'happy']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'angry', 'sad', 'disappointed']
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        # Determine emotion based on features
        valence = 0.5  # Neutral by default
        arousal = 0.5  # Neutral by default
        
        if positive_count > negative_count:
            valence = 0.7
        elif negative_count > positive_count:
            valence = 0.3
        
        if energy > 0.1 and tempo > 120:
            arousal = 0.8
        elif energy < 0.05 and tempo < 80:
            arousal = 0.2
        
        # Map to emotion categories
        if valence > 0.6 and arousal > 0.6:
            emotion = 'joy'
        elif valence > 0.6 and arousal < 0.4:
            emotion = 'calm'
        elif valence < 0.4 and arousal > 0.6:
            emotion = 'anger'
        elif valence < 0.4 and arousal < 0.4:
            emotion = 'sadness'
        elif arousal > 0.6:
            emotion = 'excitement'
        else:
            emotion = 'neutral'
        
        return {
            'emotion': emotion,
            'confidence': 0.7,  # Placeholder confidence
            'valence': valence,
            'arousal': arousal
        }
    
    def _align_dialogue(self, segments: List[Dict[str, Any]], 
                       speaker_analysis: Dict[str, Any], 
                       emotion_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Align dialogue segments with speaker and emotion information."""
        aligned_dialogue = []
        
        # Create lookup dictionaries
        speaker_lookup = {a['segment_id']: a for a in speaker_analysis['speaker_assignments']}
        emotion_lookup = {e['segment_id']: e for e in emotion_analysis['segment_emotions']}
        
        for segment in segments:
            speaker_info = speaker_lookup.get(segment['id'], {})
            emotion_info = emotion_lookup.get(segment['id'], {})
            
            aligned_segment = {
                'segment_id': segment['id'],
                'start_time': segment['start_time'],
                'end_time': segment['end_time'],
                'duration': segment['duration'],
                'text': segment['text'],
                'speaker_id': speaker_info.get('speaker_id', 'unknown'),
                'speaker_confidence': speaker_info.get('confidence', 0.0),
                'emotion': emotion_info.get('emotion', 'neutral'),
                'emotion_confidence': emotion_info.get('confidence', 0.0),
                'valence': emotion_info.get('valence', 0.5),
                'arousal': emotion_info.get('arousal', 0.5),
                'transcription_confidence': segment['confidence']
            }
            
            aligned_dialogue.append(aligned_segment)
        
        return aligned_dialogue
    
    def format_timestamp(self, seconds: float) -> str:
        """Format timestamp as HH:MM:SS."""
        return str(timedelta(seconds=seconds))
    
    def get_speaker_summary(self, dialogue_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get summary of speaker information."""
        speakers = dialogue_data['speaker_analysis']['speakers']
        
        summary = {
            'total_speakers': len(speakers),
            'speaker_details': []
        }
        
        for speaker_id, speaker_info in speakers.items():
            summary['speaker_details'].append({
                'speaker_id': speaker_id,
                'utterance_count': speaker_info['utterance_count'],
                'total_duration': speaker_info['total_duration'],
                'first_appearance': self.format_timestamp(speaker_info['first_appearance'] or 0),
                'last_appearance': self.format_timestamp(speaker_info['last_appearance'] or 0)
            })
        
        return summary