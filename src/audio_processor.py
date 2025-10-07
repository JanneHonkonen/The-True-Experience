"""
Audio processing and feature extraction module.

Handles audio extraction, source separation, and feature analysis
for dialogue, sound effects, and music detection.
"""

import librosa
import numpy as np
import soundfile as sf
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import torch
import torchaudio
from pydub import AudioSegment
import tempfile
import os


class AudioProcessor:
    """Handles audio processing and feature extraction."""
    
    def __init__(self, sample_rate: int = 44100, use_gpu: bool = False):
        """
        Initialize audio processor.
        
        Args:
            sample_rate: Target sample rate for audio processing
            use_gpu: Whether to use GPU acceleration
        """
        self.sample_rate = sample_rate
        self.use_gpu = use_gpu
        self.logger = logging.getLogger(__name__)
        
        # Initialize device
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")
    
    def process_audio(self, video_path: str) -> Dict[str, Any]:
        """
        Extract and process audio from video file.
        
        Args:
            video_path: Path to input video file
            
        Returns:
            Dictionary containing audio data and features
        """
        self.logger.info(f"Processing audio from: {video_path}")
        
        # Extract audio using librosa
        audio, sr = librosa.load(video_path, sr=self.sample_rate, mono=False)
        
        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio)
        
        self.logger.info(f"Audio loaded: {len(audio)} samples at {sr} Hz")
        
        # Calculate duration
        duration = len(audio) / sr
        
        # Extract audio features
        features = self._extract_audio_features(audio, sr)
        
        # Save audio to temporary file for further processing
        temp_audio_path = self._save_temp_audio(audio, sr)
        
        return {
            'audio': audio,
            'sample_rate': sr,
            'duration': duration,
            'audio_path': temp_audio_path,
            'features': features
        }
    
    def _extract_audio_features(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """Extract various audio features."""
        self.logger.debug("Extracting audio features...")
        
        features = {}
        
        # Spectral features
        features['spectral_centroid'] = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        features['spectral_rolloff'] = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
        features['zero_crossing_rate'] = librosa.feature.zero_crossing_rate(audio)[0]
        
        # MFCC features
        features['mfcc'] = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        
        # Rhythm features
        features['tempo'], features['beats'] = librosa.beat.beat_track(y=audio, sr=sr)
        features['onset_frames'] = librosa.onset.onset_detect(y=audio, sr=sr)
        
        # Energy and RMS
        features['rms'] = librosa.feature.rms(y=audio)[0]
        features['spectral_bandwidth'] = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
        
        # Harmonic and percussive components
        features['harmonic'], features['percussive'] = librosa.effects.hpss(audio)
        
        # Silence detection
        features['silence_frames'] = self._detect_silence(audio, sr)
        
        return features
    
    def _detect_silence(self, audio: np.ndarray, sr: int, threshold: float = 0.01) -> List[Tuple[float, float]]:
        """Detect silent segments in audio."""
        # Calculate RMS energy
        frame_length = int(0.1 * sr)  # 100ms frames
        hop_length = frame_length // 2
        
        rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Find silent frames
        silent_frames = rms < threshold
        
        # Convert to time segments
        silence_segments = []
        in_silence = False
        start_time = 0
        
        for i, is_silent in enumerate(silent_frames):
            time = i * hop_length / sr
            
            if is_silent and not in_silence:
                start_time = time
                in_silence = True
            elif not is_silent and in_silence:
                silence_segments.append((start_time, time))
                in_silence = False
        
        # Handle case where audio ends in silence
        if in_silence:
            silence_segments.append((start_time, len(audio) / sr))
        
        return silence_segments
    
    def _save_temp_audio(self, audio: np.ndarray, sr: int) -> str:
        """Save audio to temporary file for further processing."""
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, f"temp_audio_{id(audio)}.wav")
        
        sf.write(temp_path, audio, sr)
        self.logger.debug(f"Audio saved to temporary file: {temp_path}")
        
        return temp_path
    
    def separate_sources(self, audio_path: str) -> Dict[str, str]:
        """
        Separate audio into different sources (vocals, drums, bass, other).
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with paths to separated audio files
        """
        self.logger.info("Separating audio sources...")
        
        try:
            # This would use Demucs for source separation
            # For now, we'll create placeholder separated files
            temp_dir = tempfile.gettempdir()
            base_name = Path(audio_path).stem
            
            separated = {}
            for source in ['vocals', 'drums', 'bass', 'other']:
                # In a real implementation, this would use Demucs
                # For now, we'll copy the original audio
                separated_path = os.path.join(temp_dir, f"{base_name}_{source}.wav")
                audio, sr = librosa.load(audio_path, sr=self.sample_rate)
                sf.write(separated_path, audio, sr)
                separated[source] = separated_path
            
            return separated
            
        except Exception as e:
            self.logger.warning(f"Source separation failed: {e}")
            return {'vocals': audio_path, 'drums': audio_path, 'bass': audio_path, 'other': audio_path}
    
    def detect_music_segments(self, audio: np.ndarray, sr: int) -> List[Dict[str, Any]]:
        """
        Detect music segments in audio.
        
        Args:
            audio: Audio signal
            sr: Sample rate
            
        Returns:
            List of music segments with metadata
        """
        self.logger.debug("Detecting music segments...")
        
        # Simple music detection based on spectral features
        # In practice, this would use more sophisticated ML models
        
        music_segments = []
        
        # Calculate features over time
        hop_length = 512
        frame_length = 2048
        
        # Spectral centroid (brightness)
        spectral_centroid = librosa.feature.spectral_centroid(
            y=audio, sr=sr, hop_length=hop_length
        )[0]
        
        # Zero crossing rate (speech vs music indicator)
        zcr = librosa.feature.zero_crossing_rate(audio, hop_length=hop_length)[0]
        
        # RMS energy
        rms = librosa.feature.rms(y=audio, hop_length=hop_length)[0]
        
        # Simple heuristic: music has higher spectral centroid and lower ZCR
        music_threshold = np.median(spectral_centroid) * 1.2
        zcr_threshold = np.median(zcr) * 0.8
        
        music_frames = (spectral_centroid > music_threshold) & (zcr < zcr_threshold)
        
        # Convert frames to time segments
        in_music = False
        start_time = 0
        
        for i, is_music in enumerate(music_frames):
            time = i * hop_length / sr
            
            if is_music and not in_music:
                start_time = time
                in_music = True
            elif not is_music and in_music:
                music_segments.append({
                    'start_time': start_time,
                    'end_time': time,
                    'confidence': np.mean(spectral_centroid[i-10:i+10]) / np.max(spectral_centroid)
                })
                in_music = False
        
        return music_segments
    
    def analyze_emotion_from_audio(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Analyze emotional content from audio features.
        
        Args:
            audio: Audio signal
            sr: Sample rate
            
        Returns:
            Dictionary with emotional analysis
        """
        self.logger.debug("Analyzing emotional content from audio...")
        
        # Extract features for emotion analysis
        features = self._extract_audio_features(audio, sr)
        
        # Simple emotion classification based on audio features
        # In practice, this would use trained ML models
        
        emotion_scores = {
            'valence': 0.5,  # Positive/negative
            'arousal': 0.5,  # Calm/excited
            'dominance': 0.5  # Submissive/dominant
        }
        
        # Analyze tempo for arousal
        tempo = features['tempo']
        if tempo > 120:
            emotion_scores['arousal'] = 0.8
        elif tempo < 80:
            emotion_scores['arousal'] = 0.2
        
        # Analyze spectral centroid for valence
        avg_spectral_centroid = np.mean(features['spectral_centroid'])
        if avg_spectral_centroid > np.median(features['spectral_centroid']):
            emotion_scores['valence'] = 0.7
        else:
            emotion_scores['valence'] = 0.3
        
        # Analyze RMS for dominance
        avg_rms = np.mean(features['rms'])
        if avg_rms > np.median(features['rms']):
            emotion_scores['dominance'] = 0.7
        else:
            emotion_scores['dominance'] = 0.3
        
        return {
            'emotion_scores': emotion_scores,
            'tempo': tempo,
            'energy': float(avg_rms),
            'brightness': float(avg_spectral_centroid)
        }