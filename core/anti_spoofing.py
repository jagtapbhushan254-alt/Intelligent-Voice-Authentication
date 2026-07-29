"""
SecureX-Assist - Anti-Spoofing & Replay Attack Detection
Advanced security features for voice biometric authentication
"""

import numpy as np
import torch
import torchaudio
from scipy import signal
from scipy.io import wavfile
from typing import Dict, Tuple, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class AntiSpoofingEngine:
    """
    Advanced anti-spoofing engine with multiple detection methods:
    1. Replay Attack Detection (spectral analysis)
    2. Voice Liveness Detection (dynamic features)
    3. Background Noise Analysis (environmental robustness)
    4. Speech Quality Assessment (naturalness check)
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        security_cfg = config.get('security', {}) if isinstance(config, dict) else {}
        system_cfg = config.get('system', {}) if isinstance(config, dict) else {}

        # Replay score is a "genuine-ness" score (higher is more likely live voice).
        # Keep these configurable because different microphones/environments shift scores.
        self.replay_live_min = float(
            security_cfg.get('replay_live_min', system_cfg.get('replay_live_min', 0.20))
        )
        self.liveness_live_min = float(
            security_cfg.get('liveness_live_min', system_cfg.get('liveness_live_min', 0.50))
        )
        self.quality_genuine_min = float(
            security_cfg.get('quality_genuine_min', system_cfg.get('quality_genuine_min', 0.40))
        )
        logger.info("Anti-Spoofing Engine initialized")
    
    def analyze_audio_security(self, audio_path: str) -> Dict:
        """
        Comprehensive security analysis of audio sample
        
        Returns dict with:
        - is_live: Boolean indicating if voice is live (not replayed)
        - is_genuine: Boolean indicating if voice is natural (not synthesized)
        - confidence: Overall confidence score (0-1)
        - details: Dictionary with detailed analysis
        """
        try:
            # Load audio
            sample_rate, audio_data = self._load_audio(audio_path)
            
            # Run all detection methods
            replay_score = self._detect_replay_attack(audio_data, sample_rate)
            liveness_score = self._detect_liveness(audio_data, sample_rate)
            quality_score = self._assess_speech_quality(audio_data, sample_rate)
            noise_score = self._analyze_background_noise(audio_data, sample_rate)
            
            # Calculate overall scores using configurable thresholds.
            # replay_score is "higher is more genuine/live".
            is_live = replay_score >= self.replay_live_min and liveness_score >= self.liveness_live_min
            # Keep genuine gate permissive enough for normal mic/noise variation.
            is_genuine = quality_score >= self.quality_genuine_min
            
            logger.info(
                f"Security decision: is_live={is_live} "
                f"(replay_score={replay_score:.3f} >= {self.replay_live_min:.2f}: {replay_score >= self.replay_live_min}, "
                f"liveness_score={liveness_score:.3f} >= {self.liveness_live_min:.2f}: {liveness_score >= self.liveness_live_min})"
            )
            logger.info(
                f"Security decision: is_genuine={is_genuine} "
                f"(quality_score={quality_score:.3f} >= {self.quality_genuine_min:.2f}: {quality_score >= self.quality_genuine_min})"
            )
            
            # Overall confidence (weighted average)
            confidence = (
                replay_score * 0.35 +
                liveness_score * 0.30 +
                quality_score * 0.20 +
                noise_score * 0.15
            )
            
            result = {
                'is_live': is_live,
                'is_genuine': is_genuine,
                'confidence': float(confidence),
                'spoof_score': float(1 - confidence),  # Higher score = more likely spoofed
                'details': {
                    'replay_detection': float(replay_score),
                    'liveness_detection': float(liveness_score),
                    'quality_assessment': float(quality_score),
                    'noise_analysis': float(noise_score),
                    'synthesis_detection': float(1 - (quality_score * 0.6 + liveness_score * 0.4)),  # Lower = more likely synthetic
                    'liveness_check': float(liveness_score),  # Alias for liveness_detection
                    'spectral_artifacts': float(1 - quality_score)  # Higher = more spectral artifacts (lower quality)
                }
            }
            
            logger.info(f"Anti-spoofing analysis: live={is_live}, genuine={is_genuine}, confidence={confidence:.2%}")
            return result
            
        except Exception as e:
            logger.error(f"Anti-spoofing analysis failed: {e}")
            # Return permissive result on error to avoid blocking legitimate users
            return {
                'is_live': True,
                'is_genuine': True,
                'confidence': 0.5,
                'spoof_score': 0.5,
                'details': {
                    'error': str(e),
                    'synthesis_detection': 0.5,  # Neutral score on error
                    'liveness_check': 0.5,  # Neutral score on error
                    'spectral_artifacts': 0.5  # Neutral score on error
                }
            }
    
    def _load_audio(self, audio_path: str) -> Tuple[int, np.ndarray]:
        """Load audio file and return (sample_rate, audio_data)"""
        try:
            sample_rate, audio_data = wavfile.read(audio_path)
            
            # Convert to float32 and normalize
            if audio_data.dtype == np.int16:
                audio_data = audio_data.astype(np.float32) / 32768.0
            elif audio_data.dtype == np.int32:
                audio_data = audio_data.astype(np.float32) / 2147483648.0
            
            # Convert stereo to mono
            if audio_data.ndim == 2:
                audio_data = np.mean(audio_data, axis=1)
            
            return sample_rate, audio_data
            
        except Exception as e:
            logger.error(f"Audio loading failed: {e}")
            raise
    
    def _detect_replay_attack(self, audio_data: np.ndarray, sample_rate: int) -> float:
        """
        Detect replay attacks using spectral analysis
        
        Replay attacks show:
        - Reduced high-frequency content (speaker distortion)
        - Abnormal spectral envelope
        - Linear phase response
        - Reduced dynamic range
        
        Returns: Score 0-1 (1 = likely genuine, 0 = likely replay)
        """
        try:
            # Compute spectrogram
            f, t, Sxx = signal.spectrogram(audio_data, sample_rate, nperseg=512)
            
            # 1. High-frequency content analysis
            # Replay attacks typically lose high frequencies due to speaker limitations
            high_freq_idx = f > 4000  # Above 4kHz
            low_freq_idx = f < 1000   # Below 1kHz
            
            high_energy = np.mean(Sxx[high_freq_idx, :])
            low_energy = np.mean(Sxx[low_freq_idx, :])
            
            # Genuine voice maintains better high-freq/low-freq ratio
            freq_ratio = high_energy / (low_energy + 1e-10)
            freq_score = min(freq_ratio * 5, 1.0)  # Normalize to 0-1
            
            # 2. Dynamic range analysis
            # Replay attacks have compressed dynamic range
            dynamic_range = np.max(np.abs(audio_data)) - np.min(np.abs(audio_data))
            dynamic_score = min(dynamic_range * 2, 1.0)
            
            # 3. Zero-crossing rate variation
            # Genuine speech has more variation in zero crossings
            zcr = np.array([
                np.sum(np.diff(np.sign(audio_data[i:i+512])) != 0)
                for i in range(0, len(audio_data) - 512, 256)
            ])
            zcr_variance = np.var(zcr) / (np.mean(zcr) + 1e-10)
            zcr_score = min(zcr_variance * 0.5, 1.0)
            
            # 4. Spectral flux (temporal variation)
            # Replay attacks show less spectral variation over time
            spectral_flux = np.mean([
                np.sum(np.abs(Sxx[:, i+1] - Sxx[:, i]))
                for i in range(Sxx.shape[1] - 1)
            ])
            flux_score = min(spectral_flux * 100, 1.0)
            
            # Combined replay detection score
            replay_score = (
                freq_score * 0.35 +
                dynamic_score * 0.25 +
                zcr_score * 0.20 +
                flux_score * 0.20
            )
            
            logger.debug(f"Replay detection: {replay_score:.2%} (freq={freq_score:.2f}, dyn={dynamic_score:.2f}, zcr={zcr_score:.2f}, flux={flux_score:.2f})")
            return replay_score
            
        except Exception as e:
            logger.warning(f"Replay detection failed: {e}")
            return 0.7  # Neutral score on error
    
    def _detect_liveness(self, audio_data: np.ndarray, sample_rate: int) -> float:
        """
        Detect voice liveness using dynamic acoustic features
        
        Live voices show:
        - Natural pitch variations
        - Breathing patterns
        - Micro-variations in speech
        - Natural formant transitions
        
        Returns: Score 0-1 (1 = live voice, 0 = synthesized/static)
        """
        try:
            # 1. Pitch variation analysis
            # Extract fundamental frequency (F0) using autocorrelation
            frame_length = int(0.025 * sample_rate)  # 25ms frames
            hop_length = int(0.010 * sample_rate)    # 10ms hop
            
            f0_values = []
            for i in range(0, len(audio_data) - frame_length, hop_length):
                frame = audio_data[i:i+frame_length]
                
                # Autocorrelation for pitch detection
                autocorr = np.correlate(frame, frame, mode='full')
                autocorr = autocorr[len(autocorr)//2:]
                
                # Find first peak (fundamental frequency)
                if len(autocorr) > 1:
                    peaks = signal.find_peaks(autocorr)[0]
                    if len(peaks) > 0:
                        f0 = sample_rate / peaks[0] if peaks[0] > 0 else 0
                        if 50 < f0 < 500:  # Valid human pitch range
                            f0_values.append(f0)
            
            if len(f0_values) > 5:
                # Calculate pitch variation (jitter)
                f0_std = np.std(f0_values)
                f0_mean = np.mean(f0_values)
                pitch_variation = f0_std / (f0_mean + 1e-10)
                pitch_score = min(pitch_variation * 20, 1.0)
            else:
                pitch_score = 0.5
            
            # 2. Energy envelope variation
            # Live speech has natural energy fluctuations
            frame_energies = []
            for i in range(0, len(audio_data) - frame_length, hop_length):
                frame = audio_data[i:i+frame_length]
                energy = np.sum(frame ** 2)
                frame_energies.append(energy)
            
            energy_variation = np.std(frame_energies) / (np.mean(frame_energies) + 1e-10)
            energy_score = min(energy_variation * 5, 1.0)
            
            # 3. Spectral centroid variation
            # Natural speech shows varying spectral content
            f, t, Sxx = signal.spectrogram(audio_data, sample_rate, nperseg=512)
            spectral_centroids = []
            
            for i in range(Sxx.shape[1]):
                spectrum = Sxx[:, i]
                centroid = np.sum(f * spectrum) / (np.sum(spectrum) + 1e-10)
                spectral_centroids.append(centroid)
            
            centroid_variation = np.std(spectral_centroids) / (np.mean(spectral_centroids) + 1e-10)
            centroid_score = min(centroid_variation * 2, 1.0)
            
            # Combined liveness score
            liveness_score = (
                pitch_score * 0.4 +
                energy_score * 0.3 +
                centroid_score * 0.3
            )
            
            logger.debug(f"Liveness detection: {liveness_score:.2%} (pitch={pitch_score:.2f}, energy={energy_score:.2f}, centroid={centroid_score:.2f})")
            return liveness_score
            
        except Exception as e:
            logger.warning(f"Liveness detection failed: {e}")
            return 0.6  # Neutral score on error
    
    def _assess_speech_quality(self, audio_data: np.ndarray, sample_rate: int) -> float:
        """
        Assess speech quality and naturalness
        
        Natural speech shows:
        - Appropriate signal-to-noise ratio
        - Natural formant structure
        - Smooth spectral envelope
        - Appropriate bandwidth
        
        Returns: Score 0-1 (1 = natural quality, 0 = poor/synthetic)
        """
        try:
            # 1. Signal-to-Noise Ratio estimation
            # Use first/last 0.2s as noise estimate
            noise_frames = int(0.2 * sample_rate)
            if len(audio_data) > noise_frames * 2:
                noise = np.concatenate([audio_data[:noise_frames], audio_data[-noise_frames:]])
                signal_power = np.mean(audio_data ** 2)
                noise_power = np.mean(noise ** 2)
                snr = 10 * np.log10((signal_power / (noise_power + 1e-10)) + 1e-10)
                snr_score = min(max(snr / 30, 0), 1.0)  # Normalize to 0-1
            else:
                snr_score = 0.5
            
            # 2. Spectral smoothness
            # Natural speech has smooth spectral envelope
            f, Pxx = signal.welch(audio_data, sample_rate, nperseg=1024)
            
            # Calculate spectral derivative (smoothness)
            spectral_derivative = np.abs(np.diff(Pxx))
            smoothness = 1.0 / (1.0 + np.mean(spectral_derivative) * 1000)
            
            # 3. Bandwidth assessment
            # Human speech typically contains energy between 100Hz - 8kHz
            valid_freq_idx = (f >= 100) & (f <= 8000)
            bandwidth_energy = np.sum(Pxx[valid_freq_idx])
            total_energy = np.sum(Pxx)
            bandwidth_score = bandwidth_energy / (total_energy + 1e-10)
            
            # Combined quality score
            quality_score = (
                snr_score * 0.4 +
                smoothness * 0.3 +
                bandwidth_score * 0.3
            )
            
            logger.debug(f"Speech quality: {quality_score:.2%} (snr={snr_score:.2f}, smooth={smoothness:.2f}, bw={bandwidth_score:.2f})")
            return quality_score
            
        except Exception as e:
            logger.warning(f"Quality assessment failed: {e}")
            return 0.5  # Neutral score on error
    
    def _analyze_background_noise(self, audio_data: np.ndarray, sample_rate: int) -> float:
        """
        Analyze background noise characteristics
        
        Genuine recordings in different environments show:
        - Natural background noise patterns
        - Consistent noise floor
        - Realistic environmental acoustics
        
        Returns: Score 0-1 (1 = natural environment, 0 = suspicious)
        """
        try:
            # 1. Noise floor consistency
            # Divide audio into segments and check noise floor variation
            segment_length = int(0.5 * sample_rate)  # 0.5s segments
            num_segments = len(audio_data) // segment_length
            
            if num_segments >= 2:
                noise_floors = []
                for i in range(num_segments):
                    segment = audio_data[i * segment_length:(i + 1) * segment_length]
                    # Get bottom 10% energy frames (noise floor)
                    frame_energies = segment ** 2
                    sorted_energies = np.sort(frame_energies)
                    noise_floor = np.mean(sorted_energies[:len(sorted_energies)//10])
                    noise_floors.append(noise_floor)
                
                # Consistent noise floor indicates genuine recording
                noise_consistency = 1.0 - min(np.std(noise_floors) / (np.mean(noise_floors) + 1e-10), 1.0)
            else:
                noise_consistency = 0.5
            
            # 2. Spectral noise characteristics
            # Natural noise has broadband characteristics
            f, Pxx = signal.welch(audio_data, sample_rate, nperseg=512)
            
            # Check if noise is present across frequency bands
            freq_bands = [
                (f >= 0) & (f < 500),
                (f >= 500) & (f < 2000),
                (f >= 2000) & (f < 4000),
                (f >= 4000) & (f <= 8000)
            ]
            
            band_energies = [np.mean(Pxx[band]) for band in freq_bands]
            band_variance = np.std(band_energies) / (np.mean(band_energies) + 1e-10)
            
            # Natural noise has moderate variance across bands
            spectral_naturalness = 1.0 - min(abs(band_variance - 0.3) / 0.3, 1.0)
            
            # Combined noise analysis score
            noise_score = (
                noise_consistency * 0.6 +
                spectral_naturalness * 0.4
            )
            
            logger.debug(f"Noise analysis: {noise_score:.2%} (consistency={noise_consistency:.2f}, naturalness={spectral_naturalness:.2f})")
            return noise_score
            
        except Exception as e:
            logger.warning(f"Noise analysis failed: {e}")
            return 0.5  # Neutral score on error


class VoiceNormalizer:
    """
    Normalize voice recordings to handle different environments
    - Noise reduction
    - Volume normalization
    - Frequency equalization
    """
    
    def __init__(self):
        logger.info("Voice Normalizer initialized")
    
    def normalize_audio(self, audio_path: str, output_path: Optional[str] = None) -> str:
        """
        Normalize audio to improve recognition across different environments
        
        Args:
            audio_path: Path to input audio file
            output_path: Path to save normalized audio (optional)
            
        Returns:
            Path to normalized audio file
        """
        try:
            # Load audio
            sample_rate, audio_data = wavfile.read(audio_path)
            
            # Convert to float32
            if audio_data.dtype == np.int16:
                audio_data = audio_data.astype(np.float32) / 32768.0
            elif audio_data.dtype == np.int32:
                audio_data = audio_data.astype(np.float32) / 2147483648.0
            
            # 1. Volume normalization (RMS normalization)
            rms = np.sqrt(np.mean(audio_data ** 2))
            target_rms = 0.1  # Target RMS level
            if rms > 1e-6:
                audio_data = audio_data * (target_rms / rms)
            
            # 2. Simple noise reduction using spectral subtraction
            audio_data = self._reduce_noise(audio_data, sample_rate)
            
            # 3. High-pass filter to remove low-frequency noise
            audio_data = self._highpass_filter(audio_data, sample_rate, cutoff=80)
            
            # 4. Clip to prevent distortion
            audio_data = np.clip(audio_data, -1.0, 1.0)
            
            # Save normalized audio
            if output_path is None:
                output_path = audio_path.replace('.wav', '_normalized.wav')
            
            # Convert back to int16
            audio_int16 = (audio_data * 32767).astype(np.int16)
            wavfile.write(output_path, sample_rate, audio_int16)
            
            logger.info(f"Audio normalized: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Audio normalization failed: {e}")
            return audio_path  # Return original on error
    
    def _reduce_noise(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """Simple spectral subtraction for noise reduction"""
        try:
            # Estimate noise from first 0.3s
            noise_sample_len = int(0.3 * sample_rate)
            if len(audio_data) > noise_sample_len:
                noise_profile = audio_data[:noise_sample_len]
                
                # Apply spectral subtraction (simplified)
                from scipy.signal import stft, istft
                
                f, t, Zxx = stft(audio_data, sample_rate, nperseg=256)
                _, _, Noise = stft(noise_profile, sample_rate, nperseg=256)
                
                # Subtract noise spectrum
                noise_magnitude = np.mean(np.abs(Noise), axis=1, keepdims=True)
                clean_Zxx = Zxx - noise_magnitude * 0.5
                
                # Reconstruct signal
                _, audio_clean = istft(clean_Zxx, sample_rate)
                
                # Ensure same length as input
                if len(audio_clean) > len(audio_data):
                    audio_clean = audio_clean[:len(audio_data)]
                elif len(audio_clean) < len(audio_data):
                    audio_clean = np.pad(audio_clean, (0, len(audio_data) - len(audio_clean)))
                
                return audio_clean
            
            return audio_data
            
        except Exception as e:
            logger.warning(f"Noise reduction failed: {e}")
            return audio_data
    
    def _highpass_filter(self, audio_data: np.ndarray, sample_rate: int, cutoff: float = 80) -> np.ndarray:
        """Apply high-pass filter to remove low-frequency noise"""
        try:
            nyquist = sample_rate / 2
            normalized_cutoff = cutoff / nyquist
            
            # Design Butterworth high-pass filter
            b, a = signal.butter(4, normalized_cutoff, btype='high')
            
            # Apply filter
            filtered = signal.filtfilt(b, a, audio_data)
            
            return filtered
            
        except Exception as e:
            logger.warning(f"High-pass filter failed: {e}")
            return audio_data
