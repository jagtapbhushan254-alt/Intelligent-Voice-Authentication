"""
SecureX-Assist - Advanced Audio Preprocessing
Data augmentation for voice biometric training and anti-spoofing
"""

import numpy as np
import scipy.signal as signal
from typing import List, Tuple, Optional
import logging
from pathlib import Path
import random

logger = logging.getLogger(__name__)


class AudioAugmentationEngine:
    """
    Advanced audio data augmentation for voice biometric training
    Generates diverse voice samples from limited enrollment data
    """

    def __init__(self, config: dict):
        self.config = config
        self.augmentation_settings = config.get('augmentation', {})

        # Default augmentation parameters
        self.pitch_shift_range = self.augmentation_settings.get('pitch_shift', [-3, 3])
        self.gain_db_range = self.augmentation_settings.get('gain_db', [-6, 6])
        self.speed_factor_range = self.augmentation_settings.get('speed_factor', [0.95, 1.05])
        self.noise_level = self.augmentation_settings.get('noise_level', 0.01)

        logger.info("AudioAugmentationEngine initialized with advanced augmentation")

    def augment_audio_sample(self, audio_data: np.ndarray, sample_rate: int) -> List[np.ndarray]:
        """
        Generate augmented versions of a single audio sample

        Args:
            audio_data: Original audio numpy array (can be 1D or 2D)
            sample_rate: Audio sample rate

        Returns:
            List of augmented audio samples
        """
        # Ensure audio is 1D
        if audio_data.ndim == 2:
            audio_data = audio_data.squeeze()
        elif audio_data.ndim != 1:
            logger.warning(f"Unexpected audio dimensions: {audio_data.shape}, skipping augmentation")
            return [audio_data]

        augmented_samples = [audio_data]  # Include original

        # Generate 4 augmented versions
        for i in range(4):
            augmented = audio_data.copy()

            # Apply random augmentations in sequence
            augmented = self._apply_pitch_shift(augmented, sample_rate)
            augmented = self._apply_gain_variation(augmented)
            augmented = self._apply_speed_variation(augmented, sample_rate)
            augmented = self._add_background_noise(augmented)

            augmented_samples.append(augmented)

        return augmented_samples

    def _apply_pitch_shift(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Apply random pitch shifting (±3% by default)
        """
        try:
            # Random pitch shift in semitones
            pitch_shift = random.uniform(self.pitch_shift_range[0], self.pitch_shift_range[1])

            if abs(pitch_shift) < 0.1:  # Skip if shift is negligible
                return audio_data

            # Convert semitones to frequency ratio
            ratio = 2 ** (pitch_shift / 12.0)

            # Resample to achieve pitch shift
            new_length = int(len(audio_data) / ratio)
            if new_length > 0:
                shifted_audio = signal.resample(audio_data, new_length)
                # Pad or truncate to original length
                if len(shifted_audio) > len(audio_data):
                    shifted_audio = shifted_audio[:len(audio_data)]
                else:
                    # Pad with zeros
                    padding = np.zeros(len(audio_data) - len(shifted_audio))
                    shifted_audio = np.concatenate([shifted_audio, padding])

                return shifted_audio.astype(audio_data.dtype)
            else:
                return audio_data

        except Exception as e:
            logger.warning(f"Pitch shift failed: {e}")
            return audio_data

    def _apply_gain_variation(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Apply random gain variation (±6 dB by default)
        """
        try:
            # Random gain in dB
            gain_db = random.uniform(self.gain_db_range[0], self.gain_db_range[1])
            gain_factor = 10 ** (gain_db / 20.0)  # Convert dB to linear

            # Apply gain
            augmented = audio_data * gain_factor

            # Prevent clipping
            max_val = np.max(np.abs(augmented))
            if max_val > 1.0:
                augmented = augmented / max_val * 0.95

            return augmented

        except Exception as e:
            logger.warning(f"Gain variation failed: {e}")
            return audio_data

    def _apply_speed_variation(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Apply random speed variation (±5% by default)
        """
        try:
            # Random speed factor
            speed_factor = random.uniform(self.speed_factor_range[0], self.speed_factor_range[1])

            if abs(speed_factor - 1.0) < 0.01:  # Skip if change is negligible
                return audio_data

            # Resample to achieve speed change
            new_length = int(len(audio_data) / speed_factor)
            if new_length > 0:
                speed_changed = signal.resample(audio_data, new_length)

                # Stretch/compress to original length using interpolation
                original_indices = np.linspace(0, len(speed_changed) - 1, len(audio_data))
                stretched = np.interp(original_indices, np.arange(len(speed_changed)), speed_changed)

                return stretched.astype(audio_data.dtype)
            else:
                return audio_data

        except Exception as e:
            logger.warning(f"Speed variation failed: {e}")
            return audio_data

    def _add_background_noise(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Add subtle background noise to simulate real environments
        """
        try:
            # Generate white noise
            noise = np.random.normal(0, self.noise_level, len(audio_data))

            # Mix with original audio
            noisy_audio = audio_data + noise

            # Normalize to prevent clipping
            max_val = np.max(np.abs(noisy_audio))
            if max_val > 1.0:
                noisy_audio = noisy_audio / max_val * 0.95

            return noisy_audio

        except Exception as e:
            logger.warning(f"Background noise addition failed: {e}")
            return audio_data

    def generate_enrollment_samples(self, original_samples: List[np.ndarray], sample_rate: int) -> List[np.ndarray]:
        """
        Generate comprehensive enrollment dataset from original samples

        Args:
            original_samples: List of original audio samples (typically 3, can be 1D or 2D)
            sample_rate: Audio sample rate

        Returns:
            Extended list including original + augmented samples
        """
        enrollment_dataset = []

        for original in original_samples:
            # Ensure audio is 1D for processing
            if original.ndim == 2:
                processed_original = original.squeeze()
            else:
                processed_original = original
            
            # Add original sample
            enrollment_dataset.append(processed_original)

            # Generate augmented versions
            augmented_versions = self.augment_audio_sample(processed_original, sample_rate)
            enrollment_dataset.extend(augmented_versions[1:])  # Skip original (already added)

        logger.info(f"Generated {len(enrollment_dataset)} enrollment samples from {len(original_samples)} originals")
        return enrollment_dataset


class VoiceQualityAnalyzer:
    """
    Analyze voice quality and detect potential spoofing indicators
    """

    def __init__(self, config: dict):
        self.config = config
        self.sample_rate = config.get('audio', {}).get('sample_rate', 16000)

    def analyze_voice_quality(self, audio_data: np.ndarray) -> dict:
        """
        Analyze various voice quality metrics

        Returns:
            Dictionary with quality scores and spoofing indicators
        """
        try:
            # Ensure audio is 1D for analysis
            if audio_data.ndim == 2:
                audio_data = audio_data.squeeze()
            elif audio_data.ndim != 1:
                logger.warning(f"Unexpected audio dimensions: {audio_data.shape}")
                return {'quality_score': 0.0, 'is_live_voice': False}

            # Basic quality metrics
            rms_energy = float(np.sqrt(np.mean(audio_data ** 2)))
            zero_crossings = float(np.sum(np.abs(np.diff(np.sign(audio_data)))) / (2 * len(audio_data)))
            spectral_centroid = float(self._calculate_spectral_centroid(audio_data))

            # Robustness metrics for noisy environments
            snr_db = float(self._estimate_snr_db(audio_data))
            clipping_ratio = float(self._estimate_clipping_ratio(audio_data))

            # Spoofing indicators
            quality_score = self._calculate_quality_score(audio_data, rms_energy, zero_crossings, spectral_centroid)

            # Configurable live gate: combine legacy quality_score with SNR / clipping checks.
            security_cfg = self.config.get('security', {}) if isinstance(self.config, dict) else {}
            system_cfg = self.config.get('system', {}) if isinstance(self.config, dict) else {}
            audio_cfg = self.config.get('audio', {}) if isinstance(self.config, dict) else {}

            min_quality = float(security_cfg.get('min_voice_quality_score', system_cfg.get('min_voice_quality_score', 0.25)))
            min_snr_db = float(security_cfg.get('min_snr_db', system_cfg.get('min_snr_db', 6.0)))
            max_clip_ratio = float(security_cfg.get('max_clipping_ratio', system_cfg.get('max_clipping_ratio', 0.015)))
            # Guard against short/noisy capture: require minimum audio duration as well.
            min_verify_seconds = float(security_cfg.get('min_verify_audio_seconds', system_cfg.get('min_verify_audio_seconds', audio_cfg.get('min_speech_duration', 0.5))))
            duration_seconds = float(len(audio_data) / float(self.sample_rate)) if self.sample_rate else 0.0

            is_live_voice = (
                quality_score >= min_quality
                and snr_db >= min_snr_db
                and clipping_ratio <= max_clip_ratio
                and duration_seconds >= min_verify_seconds
            )

            # Debug logging
            logger.info(
                "Voice quality metrics - RMS: %.4f, ZCR: %.4f, Centroid: %.1f, SNR(dB): %.1f, Clip: %.3f, Score: %.3f, Live: %s",
                rms_energy,
                zero_crossings,
                spectral_centroid,
                snr_db,
                clipping_ratio,
                quality_score,
                is_live_voice,
            )

            return {
                'rms_energy': float(rms_energy),
                'zero_crossings': float(zero_crossings),
                'spectral_centroid': float(spectral_centroid),
                'snr_db': float(snr_db),
                'clipping_ratio': float(clipping_ratio),
                'quality_score': float(quality_score),
                'duration_seconds': float(duration_seconds),
                    'is_live_voice': bool(is_live_voice),
            }

        except Exception as e:
            logger.error(f"Voice quality analysis failed: {e}")
            return {'quality_score': 0.0, 'is_live_voice': False}

    def _calculate_spectral_centroid(self, audio_data: np.ndarray) -> float:
        """Calculate spectral centroid (brightness of sound)"""
        try:
            # Compute FFT
            fft = np.fft.fft(audio_data)
            freqs = np.fft.fftfreq(len(audio_data), 1/self.sample_rate)

            # Magnitude spectrum
            magnitude = np.abs(fft)

            # Spectral centroid
            centroid = np.sum(freqs * magnitude) / np.sum(magnitude)

            return abs(centroid)  # Return absolute value

        except Exception:
            return 0.0

    def _calculate_quality_score(self, audio_data: np.ndarray, rms: float, zcr: float, centroid: float) -> float:
        """Calculate overall voice quality score (0-1)."""
        try:
            score = 0.0

            # RMS energy score (reasonable loudness)
            if 0.001 < rms < 0.8:
                score += 0.3
            elif 0.0005 < rms < 1.0:
                score += 0.2

            # Zero crossing rate: typical human speech
            if 0.01 < zcr < 0.5:
                score += 0.3
            elif 0.005 < zcr < 0.8:
                score += 0.2

            # Spectral centroid: broadly within human speech
            if 300 < centroid < 10000:
                score += 0.4
            elif 100 < centroid < 15000:
                score += 0.3

            return float(min(score, 1.0))
        except Exception:
            return 0.0

    def _estimate_snr_db(self, audio_data: np.ndarray) -> float:
        """Estimate SNR in dB using a simple percentile-based noise floor.

        This is intentionally lightweight and robust: it approximates noise floor
        from low-energy regions, which works reasonably well for login phrases.
        """
        try:
            if audio_data.size < 256:
                return 0.0

            x = np.asarray(audio_data, dtype=np.float32)
            # Frame the signal (20ms @ 16k ~ 320 samples)
            frame_len = int(0.02 * self.sample_rate)
            hop = max(1, frame_len // 2)
            if frame_len <= 0 or x.size < frame_len:
                return 0.0

            energies = []
            for i in range(0, x.size - frame_len, hop):
                frame = x[i:i + frame_len]
                energies.append(float(np.mean(frame * frame)))
            if not energies:
                return 0.0

            energies_np = np.array(energies, dtype=np.float32)
            # Noise floor from 10th percentile energy, signal from 90th percentile.
            noise = float(np.percentile(energies_np, 10))
            signal = float(np.percentile(energies_np, 90))
            if noise <= 1e-12:
                return 60.0
            ratio = max(1e-6, signal / noise)
            return float(10.0 * np.log10(ratio))
        except Exception as e:
            logger.debug("SNR estimation failed: %s", e)
            return 0.0

    def _estimate_clipping_ratio(self, audio_data: np.ndarray) -> float:
        """Return fraction of samples near full-scale (indicative of clipping)."""
        try:
            x = np.asarray(audio_data, dtype=np.float32)
            if x.size == 0:
                return 0.0
            # If audio has been normalized to ~0.95 peak, treat >=0.98 as clipped.
            clipped = np.mean(np.abs(x) >= 0.98)
            return float(clipped)
        except Exception as e:
            logger.debug("Clipping estimation failed: %s", e)
            return 0.0


def denoise_and_agc(audio_data: np.ndarray, sample_rate: int = 16000, strength: float = 0.35) -> np.ndarray:
    """Lightweight denoise + AGC for noisy mic capture.

    - Denoise: spectral gating via a gentle high-pass + median-based noise floor.
    - AGC: normalize RMS toward a target while avoiding clipping.

    Keeps processing intentionally mild to avoid harming speaker embeddings.
    """
    try:
        x = np.asarray(audio_data, dtype=np.float32)
        if x.ndim > 1:
            x = x.squeeze()
        if x.size < 256:
            return x

        # High-pass at 60Hz to remove rumble/DC
        nyq = (sample_rate / 2.0) if sample_rate else 8000.0
        cutoff = 60.0 / max(1.0, nyq)
        b, a = signal.butter(2, cutoff, btype='high')
        x = signal.filtfilt(b, a, x).astype(np.float32)

        # Gentle spectral gate using STFT magnitude noise floor
        n_fft = 512
        hop = 160
        win = signal.windows.hann(n_fft, sym=False)
        f, t, Zxx = signal.stft(x, fs=sample_rate, window=win, nperseg=n_fft, noverlap=n_fft - hop)
        mag = np.abs(Zxx)
        noise_floor = np.median(mag, axis=1, keepdims=True)
        # Gate: keep bins above (1 + strength)*noise_floor
        gate = (mag >= (1.0 + float(strength)) * noise_floor).astype(np.float32)
        Zxx_f = Zxx * gate
        _, x_rec = signal.istft(Zxx_f, fs=sample_rate, window=win, nperseg=n_fft, noverlap=n_fft - hop)
        x = x_rec.astype(np.float32)

        # AGC to target RMS
        target_rms = 0.08
        rms = float(np.sqrt(np.mean(x * x))) if x.size else 0.0
        if rms > 1e-6:
            gain = min(4.0, target_rms / rms)
            x = x * gain

        # Soft clip safeguard
        peak = float(np.max(np.abs(x))) if x.size else 0.0
        if peak > 0.99:
            x = x / peak * 0.95

        return x
    except Exception as e:
        logger.debug("denoise_and_agc failed: %s", e)
        return np.asarray(audio_data, dtype=np.float32)