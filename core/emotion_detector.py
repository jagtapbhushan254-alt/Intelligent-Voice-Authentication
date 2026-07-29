"""
SecureX-Assist - Emotion Detection from Voice
Detects emotional state from voice audio features
"""

import numpy as np
import logging
from typing import Dict, Optional, Tuple
import librosa
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class EmotionResult:
    """Result of emotion detection"""
    primary_emotion: str
    confidence: float
    all_emotions: Dict[str, float]
    features: Dict[str, float]


class EmotionDetector:
    """
    Detects emotions from voice audio using acoustic features
    
    Emotions detected:
    - Happy: High pitch, fast tempo, high energy
    - Sad: Low pitch, slow tempo, low energy
    - Angry: High energy, loud, fast tempo, harsh
    - Calm: Moderate pitch, slow tempo, low energy
    - Excited: High pitch, fast tempo, very high energy
    - Stressed: High pitch variation, moderate energy
    """
    
    def __init__(self):
        self.emotion_labels = ['happy', 'sad', 'angry', 'calm', 'excited', 'stressed', 'neutral']
        
        # Feature thresholds (calibrated)
        self.thresholds = {
            'pitch_mean': 150,  # Hz
            'pitch_std': 50,    # Hz
            'energy_mean': 0.05,
            'tempo': 120,       # BPM
            'spectral_centroid': 2000  # Hz
        }
        
        logger.info("EmotionDetector initialized")
    
    def detect(self, audio_data: np.ndarray, sample_rate: int = 16000) -> EmotionResult:
        """
        Detect emotion from audio data
        
        Args:
            audio_data: Audio samples as numpy array
            sample_rate: Sample rate in Hz
            
        Returns:
            EmotionResult with detected emotion and confidence
        """
        try:
            # Extract features
            features = self._extract_features(audio_data, sample_rate)
            
            # Calculate emotion scores
            emotion_scores = self._calculate_emotion_scores(features)
            
            # Get primary emotion
            primary_emotion = max(emotion_scores.items(), key=lambda x: x[1])
            
            return EmotionResult(
                primary_emotion=primary_emotion[0],
                confidence=primary_emotion[1],
                all_emotions=emotion_scores,
                features=features
            )
            
        except Exception as e:
            logger.error(f"Emotion detection failed: {e}")
            return EmotionResult(
                primary_emotion='neutral',
                confidence=0.5,
                all_emotions={'neutral': 0.5},
                features={}
            )
    
    def _extract_features(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, float]:
        """Extract acoustic features from audio"""
        try:
            # Convert to float32 if needed
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # Normalize
            if np.max(np.abs(audio_data)) > 0:
                audio_data = audio_data / np.max(np.abs(audio_data))
            
            features = {}
            
            # Pitch (F0) - fundamental frequency
            try:
                pitches, magnitudes = librosa.piptrack(y=audio_data, sr=sample_rate)
                pitch_values = []
                for t in range(pitches.shape[1]):
                    index = magnitudes[:, t].argmax()
                    pitch = pitches[index, t]
                    if pitch > 0:
                        pitch_values.append(pitch)
                
                if pitch_values:
                    features['pitch_mean'] = float(np.mean(pitch_values))
                    features['pitch_std'] = float(np.std(pitch_values))
                    features['pitch_max'] = float(np.max(pitch_values))
                else:
                    features['pitch_mean'] = 150.0
                    features['pitch_std'] = 20.0
                    features['pitch_max'] = 200.0
            except Exception as e:
                logger.debug(f"Pitch extraction failed: {e}")
                features['pitch_mean'] = 150.0
                features['pitch_std'] = 20.0
                features['pitch_max'] = 200.0
            
            # Energy (RMS)
            energy = librosa.feature.rms(y=audio_data)[0]
            features['energy_mean'] = float(np.mean(energy))
            features['energy_std'] = float(np.std(energy))
            features['energy_max'] = float(np.max(energy))
            
            # Tempo
            try:
                tempo, _ = librosa.beat.beat_track(y=audio_data, sr=sample_rate)
                features['tempo'] = float(tempo)
            except Exception as e:
                logger.debug(f"Tempo extraction failed: {e}")
                features['tempo'] = 120.0
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)[0]
            features['spectral_centroid'] = float(np.mean(spectral_centroids))
            
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sample_rate)[0]
            features['spectral_rolloff'] = float(np.mean(spectral_rolloff))
            
            # Zero crossing rate (indicates noisiness)
            zcr = librosa.feature.zero_crossing_rate(audio_data)[0]
            features['zcr_mean'] = float(np.mean(zcr))
            
            # MFCCs (timbre)
            mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
            for i in range(5):  # Use first 5 MFCCs
                features[f'mfcc_{i}'] = float(np.mean(mfccs[i]))
            
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return {
                'pitch_mean': 150.0,
                'energy_mean': 0.05,
                'tempo': 120.0,
                'spectral_centroid': 2000.0
            }
    
    def _calculate_emotion_scores(self, features: Dict[str, float]) -> Dict[str, float]:
        """Calculate emotion scores based on features"""
        scores = {}
        
        # Extract key features
        pitch_mean = features.get('pitch_mean', 150.0)
        pitch_std = features.get('pitch_std', 20.0)
        energy_mean = features.get('energy_mean', 0.05)
        tempo = features.get('tempo', 120.0)
        spectral_centroid = features.get('spectral_centroid', 2000.0)
        zcr = features.get('zcr_mean', 0.1)
        
        # Normalize features (0-1 scale)
        pitch_norm = min(pitch_mean / 300.0, 1.0)
        pitch_var_norm = min(pitch_std / 100.0, 1.0)
        energy_norm = min(energy_mean / 0.2, 1.0)
        tempo_norm = min(tempo / 200.0, 1.0)
        spectral_norm = min(spectral_centroid / 4000.0, 1.0)
        zcr_norm = min(zcr / 0.3, 1.0)
        
        # Happy: High pitch, fast tempo, high energy
        scores['happy'] = (
            pitch_norm * 0.3 +
            tempo_norm * 0.3 +
            energy_norm * 0.3 +
            spectral_norm * 0.1
        )
        
        # Sad: Low pitch, slow tempo, low energy
        scores['sad'] = (
            (1 - pitch_norm) * 0.3 +
            (1 - tempo_norm) * 0.3 +
            (1 - energy_norm) * 0.3 +
            (1 - spectral_norm) * 0.1
        )
        
        # Angry: High energy, high pitch variance, fast tempo
        scores['angry'] = (
            energy_norm * 0.35 +
            pitch_var_norm * 0.3 +
            tempo_norm * 0.2 +
            zcr_norm * 0.15
        )
        
        # Calm: Low energy, stable pitch, slow tempo
        scores['calm'] = (
            (1 - energy_norm) * 0.35 +
            (1 - pitch_var_norm) * 0.35 +
            (1 - tempo_norm) * 0.2 +
            (1 - zcr_norm) * 0.1
        )
        
        # Excited: Very high energy, high pitch, fast tempo
        scores['excited'] = (
            energy_norm * 0.4 +
            pitch_norm * 0.3 +
            tempo_norm * 0.3
        )
        
        # Stressed: High pitch variance, moderate energy
        scores['stressed'] = (
            pitch_var_norm * 0.4 +
            energy_norm * 0.3 +
            zcr_norm * 0.2 +
            tempo_norm * 0.1
        )
        
        # Neutral: Moderate everything
        neutral_score = 1.0 - max(scores.values()) if scores else 0.5
        scores['neutral'] = max(neutral_score, 0.3)
        
        # Normalize scores to sum to 1.0
        total = sum(scores.values())
        if total > 0:
            scores = {k: v / total for k, v in scores.items()}
        
        return scores
    
    def get_emotion_response(self, emotion: str) -> str:
        """Get appropriate response based on detected emotion"""
        responses = {
            'happy': [
                "It's great to hear you in such a good mood!",
                "Your happiness is contagious!",
                "I'm glad you're feeling wonderful!",
                "That's awesome! Keep that positive energy!"
            ],
            'sad': [
                "I'm here for you. How can I help?",
                "I'm sorry you're feeling down. Want to talk about it?",
                "It's okay to feel sad sometimes. I'm here to listen.",
                "Would you like me to do something to cheer you up?"
            ],
            'angry': [
                "I understand you're frustrated. Let me help.",
                "Take a deep breath. I'm here to assist you.",
                "I hear your frustration. What can I do to help?",
                "Let's work through this together calmly."
            ],
            'calm': [
                "You seem very relaxed. How can I help you?",
                "Nice to see you in a peaceful state.",
                "You're very calm today. What can I do for you?",
                "I appreciate your composed demeanor."
            ],
            'excited': [
                "Your excitement is contagious! What's going on?",
                "You sound really energized! Tell me more!",
                "I can feel your enthusiasm! What's happening?",
                "Wow, you're really pumped up! That's great!"
            ],
            'stressed': [
                "You seem stressed. Want me to help lighten the load?",
                "Take a moment to breathe. I'm here to help.",
                "I notice you're under pressure. How can I assist?",
                "Let me help reduce your stress. What do you need?"
            ],
            'neutral': [
                "How can I help you today?",
                "What can I do for you?",
                "I'm ready to assist you.",
                "How may I be of service?"
            ]
        }
        
        import random
        return random.choice(responses.get(emotion, responses['neutral']))


# Global instance
_emotion_detector: Optional[EmotionDetector] = None


def get_emotion_detector() -> EmotionDetector:
    """Get or create global emotion detector"""
    global _emotion_detector
    
    if _emotion_detector is None:
        _emotion_detector = EmotionDetector()
    
    return _emotion_detector


if __name__ == "__main__":
    # Test emotion detector
    detector = EmotionDetector()
    
    # Create test audio (sine wave)
    sample_rate = 16000
    duration = 2.0
    frequency = 200  # Hz
    
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Test 1: Happy (high pitch, high energy)
    audio_happy = np.sin(2 * np.pi * 300 * t) * 0.8
    result = detector.detect(audio_happy, sample_rate)
    print(f"Happy test: {result.primary_emotion} ({result.confidence:.2f})")
    
    # Test 2: Sad (low pitch, low energy)
    audio_sad = np.sin(2 * np.pi * 100 * t) * 0.2
    result = detector.detect(audio_sad, sample_rate)
    print(f"Sad test: {result.primary_emotion} ({result.confidence:.2f})")
    
    # Test 3: Angry (high energy, variable pitch)
    audio_angry = np.sin(2 * np.pi * 250 * t) * 0.9
    audio_angry += np.random.normal(0, 0.1, len(audio_angry))
    result = detector.detect(audio_angry, sample_rate)
    print(f"Angry test: {result.primary_emotion} ({result.confidence:.2f})")
    
    print("\nEmotion detector test complete!")
