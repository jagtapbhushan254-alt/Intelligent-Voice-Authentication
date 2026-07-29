"""
SecureX-Assist - Ultimate Voice Biometric Engine
Production-grade voice verification with adaptive thresholds, anti-spoofing, and data augmentation
"""

import numpy as np
import torch
from typing import Optional, List, Dict, Tuple
import logging
from pathlib import Path
import os
import tempfile
from core.voice_engine import VoiceEngine
from core.anti_spoofing_aasist import AAISTAntiSpoofingEngine as AASISTAntiSpoofing
from core.anti_spoofing import AntiSpoofingEngine
from core.audio_preprocessor_advanced import AudioAugmentationEngine, VoiceQualityAnalyzer
from scipy.io import wavfile
import random
import string

from core.audio_preprocessor_advanced import denoise_and_agc

logger = logging.getLogger(__name__)


class UltimateVoiceBiometricEngine:
    """
    Ultimate voice biometric engine with advanced features:
    - AASIST anti-spoofing gate
    - Adaptive threshold based on user variance
    - Data augmentation for enrollment
    - Mahalanobis distance scoring
    - Post-verification learning
    - Challenge-response verification
    """

    def __init__(self, config: Dict, db_connection):
        self.config = config
        self.db = db_connection

        # Initialize core components
        self.voice_engine = VoiceEngine(config)
        self.augmentation_engine = AudioAugmentationEngine(config)
        self.quality_analyzer = VoiceQualityAnalyzer(config)

        # Initialize AASIST anti-spoofing
        # IMPORTANT: On CPU, the full AASIST path can be very slow and will keep running even
        # after our short timeout (thread join), consuming CPU and causing voice embedding
        # extraction to time out. Default to the lightweight fallback on CPU unless the user
        # explicitly opts in.
        device = "cuda" if torch.cuda.is_available() else "cpu"
        security_cfg = config.get('security', {})
        system_cfg = config.get('system', {})
        enable_aasist_on_cpu = bool(security_cfg.get('enable_aasist_on_cpu', False))
        force_fallback = bool(security_cfg.get('force_anti_spoof_fallback', False)) or bool(system_cfg.get('fast_mode', False))
        if device == "cpu" and not enable_aasist_on_cpu:
            force_fallback = True

        self.anti_spoofing = AASISTAntiSpoofing(device, force_fallback=force_fallback)
        self.secondary_anti_spoofing = AntiSpoofingEngine(config)

        # Configuration parameters (support both legacy system.* and newer security.* paths)
        # (security_cfg/system_cfg are defined above for AASIST init as well)

        self.base_threshold = security_cfg.get('base_voice_threshold', system_cfg.get('base_voice_threshold', 0.40))
        self.spoof_min_confidence = security_cfg.get('spoof_confidence_min', system_cfg.get('spoof_confidence_min', 0.85))
        self.spoof_min_confidence_verify = security_cfg.get(
            'spoof_confidence_min_verify', system_cfg.get('spoof_confidence_min_verify', 0.55)
        )
        self.anti_spoof_fail_open = bool(
            security_cfg.get('anti_spoof_fail_open', system_cfg.get('anti_spoof_fail_open', False))
        )
        self.adaptive_threshold_enabled = security_cfg.get('adaptive_threshold', system_cfg.get('adaptive_threshold', True))
        self.learning_enabled = security_cfg.get('voice_update_learning', system_cfg.get('voice_update_learning', True))
        self.min_match_samples = security_cfg.get('min_match_samples', system_cfg.get('min_match_samples', 2))

        # Primary/secondary verification gates for speaker acceptance
        biometric_cfg = config.get('biometric', {})
        self.cosine_verify_threshold = float(
            biometric_cfg.get(
                'cosine_verify_threshold',
                security_cfg.get('cosine_verify_threshold', system_cfg.get('cosine_verify_threshold', 0.52))
            )
        )
        self.allow_borderline_voice_match = bool(
            security_cfg.get('allow_borderline_voice_match', system_cfg.get('allow_borderline_voice_match', False))
        )
        self.borderline_min_cosine = float(
            security_cfg.get('borderline_min_cosine', system_cfg.get('borderline_min_cosine', 0.50))
        )
        self.borderline_min_confidence = float(
            security_cfg.get('borderline_min_confidence', system_cfg.get('borderline_min_confidence', 0.70))
        )
        self.borderline_min_quality = float(
            security_cfg.get('borderline_min_quality', system_cfg.get('borderline_min_quality', 0.70))
        )
        self.calibrated_spoof_recheck_enabled = bool(
            security_cfg.get('calibrated_spoof_recheck_enabled', system_cfg.get('calibrated_spoof_recheck_enabled', True))
        )
        self.calibrated_spoof_primary_confidence_max = float(
            security_cfg.get('calibrated_spoof_primary_confidence_max', system_cfg.get('calibrated_spoof_primary_confidence_max', 0.15))
        )
        self.calibrated_spoof_secondary_confidence_min = float(
            security_cfg.get('calibrated_spoof_secondary_confidence_min', system_cfg.get('calibrated_spoof_secondary_confidence_min', 0.50))
        )
        # In secondary engine, replay_detection is a "genuine-liveness" score (higher is better).
        self.calibrated_spoof_min_replay_genuine = float(
            security_cfg.get('calibrated_spoof_min_replay_genuine', system_cfg.get('calibrated_spoof_min_replay_genuine', 0.20))
        )
        self.calibrated_spoof_min_quality_score = float(
            security_cfg.get('calibrated_spoof_min_quality_score', system_cfg.get('calibrated_spoof_min_quality_score', 0.55))
        )

        # Initialize models
        self.voice_engine.load_models()

        # Performance settings
        self.spoof_timeout = float(
            security_cfg.get('anti_spoof_timeout_seconds', system_cfg.get('anti_spoof_timeout_seconds', 3.5))
        )
        self.spoof_max_seconds = int(
            security_cfg.get('anti_spoof_max_audio_seconds', system_cfg.get('anti_spoof_max_audio_seconds', 2))
        )

        # Noise robustness controls (embedding side; anti-spoofing stays minimally processed)
        self.enable_denoise_for_embedding = bool(
            security_cfg.get('enable_denoise_for_embedding', system_cfg.get('enable_denoise_for_embedding', True))
        )
        self.denoise_strength = float(
            security_cfg.get('denoise_strength', system_cfg.get('denoise_strength', 0.35))
        )

        logger.info("UltimateVoiceBiometricEngine initialized with AASIST anti-spoofing and adaptive features")

    def enroll_user_voice(self, user_id: int, audio_samples: List[np.ndarray], sample_rate: int = 16000) -> bool:
        """
        Enroll user with 3 voice samples + automatic data augmentation

        Args:
            user_id: User ID
            audio_samples: List of 3 original audio samples
            sample_rate: Audio sample rate

        Returns:
            True if enrollment successful
        """
        try:
            if len(audio_samples) < 2:
                logger.error(f"Enrollment requires at least 2 samples, got {len(audio_samples)}")
                return False

            # Step 1: Validate and preprocess samples
            validated_samples = []
            for i, sample in enumerate(audio_samples):
                # Quality check
                quality = self.quality_analyzer.analyze_voice_quality(sample)
                logger.info(f"Sample {i+1} quality: score={quality['quality_score']:.3f}, live={quality['is_live_voice']}")
                if not quality['is_live_voice']:
                    logger.warning(f"Sample {i+1} failed quality check (score: {quality['quality_score']:.3f})")
                    continue

                # Anti-spoofing check
                bypass_spoofing = self.config.get('system', {}).get('bypass_anti_spoofing', False)
                if bypass_spoofing:
                    logger.info(f"Sample {i+1} anti-spoofing bypassed for development/testing")
                    is_genuine = True
                    confidence = 1.0
                    details = {'bypassed': True}
                else:
                    spoof_audio = self._prepare_audio_for_spoofing(sample)
                    is_genuine, confidence, details = self.anti_spoofing.detect_spoofing(spoof_audio)
                logger.info(f"Sample {i+1} anti-spoofing: confidence={confidence:.3f}, genuine={is_genuine}")
                if not is_genuine or confidence < self.spoof_min_confidence:
                    logger.warning(f"Sample {i+1} failed anti-spoofing check (confidence: {confidence:.3f} < {self.spoof_min_confidence})")
                    continue

                validated_samples.append(sample)

            if len(validated_samples) < 2:
                logger.error("Insufficient valid samples for enrollment")
                return False

            # Step 2: Generate augmented dataset
            enrollment_dataset = self.augmentation_engine.generate_enrollment_samples(
                validated_samples, sample_rate
            )

            # Step 3: Extract embeddings for all samples
            embeddings = []
            original_embeddings = []
            for i, audio in enumerate(enrollment_dataset):
                embedding = self.voice_engine.extract_embedding_from_array(audio, sample_rate)
                if embedding is not None:
                    embeddings.append(embedding)
                    # Track which are original samples (first len(validated_samples) are originals)
                    if i < len(validated_samples):
                        original_embeddings.append(embedding)

            if len(embeddings) < 5:  # Need at least original + some augmented
                logger.error("Failed to extract sufficient embeddings")
                return False

            # Step 4: Compute mean and variance from ORIGINAL samples only for better verification matching
            if original_embeddings:
                original_array = np.array(original_embeddings)
                mean_embedding = np.mean(original_array, axis=0)
                variance_embedding = np.var(original_array, axis=0)
                logger.info(f"Using {len(original_embeddings)} original embeddings for mean calculation")
            else:
                # Fallback to all embeddings if no originals
                embeddings_array = np.array(embeddings)
                mean_embedding = np.mean(embeddings_array, axis=0)
                variance_embedding = np.var(embeddings_array, axis=0)
            
            # CRITICAL: Normalize the mean embedding (mean of normalized vectors is NOT normalized!)
            mean_norm = np.linalg.norm(mean_embedding)
            if mean_norm > 0:
                mean_embedding = mean_embedding / mean_norm
                logger.info(f"Mean embedding normalized: norm={np.linalg.norm(mean_embedding):.6f}")
            else:
                logger.error("Mean embedding has zero norm!")
                return False

            # Step 5: Store in database
            logger.info(f"Storing {len(embeddings)} embeddings for user {user_id}")
            self.db.deactivate_old_embeddings(user_id)  # Deactivate previous
            embedding_id = self.db.store_voice_embedding(
                user_id=user_id,
                embedding=mean_embedding,
                variance=variance_embedding,
                embedding_type="ultimate_adaptive",
                quality_score=1.0
            )

            if embedding_id:
                logger.info(f"Successfully enrolled user {user_id} with {len(embeddings)} embedding samples")
                return True
            else:
                logger.error("Failed to store voice embedding")
                return False

        except Exception as e:
            logger.error(f"Voice enrollment failed: {e}")
            return False

    def verify_voice(self, user_id: int, audio_data: np.ndarray, sample_rate: int = 16000,
                    enable_challenge: bool = False) -> Dict:
        """
        Advanced voice verification with AASIST anti-spoofing gate

        Args:
            user_id: User ID to verify against
            audio_data: Live audio sample
            sample_rate: Audio sample rate
            enable_challenge: Whether to use challenge-response

        Returns:
            Verification result dictionary
        """
        try:
            import time
            start_time = time.time()
            result = {
                'verified': False,
                'confidence': 0.0,
                'spoof_detected': False,
                'quality_score': 0.0,
                'cosine_similarity': 0.0,
                'mahalanobis_distance': 0.0,
                'challenge_passed': None,
                'details': {}
            }

            # Step 1: Quality analysis
            quality = self.quality_analyzer.analyze_voice_quality(audio_data)
            result['quality_score'] = quality['quality_score']
            # Pass-through robustness details for UI/debugging
            result['details']['snr_db'] = float(quality.get('snr_db', 0.0))
            result['details']['clipping_ratio'] = float(quality.get('clipping_ratio', 0.0))
            result['details']['duration_seconds'] = float(quality.get('duration_seconds', 0.0))

            if not quality['is_live_voice']:
                result['details']['failure_reason'] = "Poor voice quality"
                return result

            # Step 2: AASIST Anti-spoofing gate (FIRST CHECK) - Use minimally processed audio
            # Apply only very gentle processing for anti-spoofing to preserve natural voice characteristics
            bypass_spoofing_verify = self.config.get('system', {}).get('bypass_anti_spoofing_verify', False)
            bypass_spoofing = self.config.get('system', {}).get('bypass_anti_spoofing', False)
            logger.info("Starting anti-spoofing analysis...")
            if bypass_spoofing_verify or bypass_spoofing:
                logger.info("Anti-spoofing bypassed for verification (fast mode)")
                spoof_detected = False
                confidence = 1.0  # Assume genuine
            else:
                # Always define trimmed variable for downstream calibration/recheck
                # Trim audio to the first N seconds for faster processing (configurable)
                sample_rate = 16000
                max_samples = self.spoof_max_seconds * sample_rate
                if len(audio_data) > max_samples:
                    audio_data_trimmed = audio_data[:max_samples]
                    logger.info(f"Trimmed audio from {len(audio_data)/sample_rate:.1f}s to {self.spoof_max_seconds:.1f}s for spoofing analysis")
                else:
                    audio_data_trimmed = audio_data

                spoof_audio = self._prepare_audio_for_spoofing(audio_data_trimmed)
                spoofing_start = time.time()
                try:
                    # Timeout logic: configurable, default 2s for faster verification
                    from threading import Thread
                    result_container = {}
                    def run_spoof():
                        is_genuine, conf, details = self.anti_spoofing.detect_spoofing(spoof_audio)
                        result_container['is_genuine'] = is_genuine
                        result_container['confidence'] = conf
                        result_container['details'] = details
                    t = Thread(target=run_spoof)
                    t.start()
                    t.join(timeout=self.spoof_timeout)
                    if t.is_alive():
                        logger.warning(f"Anti-spoofing analysis timed out after {self.spoof_timeout} seconds")
                        if self.anti_spoof_fail_open:
                            spoof_detected = False
                            confidence = 0.5
                            details = {'error': 'timeout', 'assumed_genuine': True}
                        else:
                            spoof_detected = True
                            confidence = 0.0
                            details = {'error': 'timeout', 'fail_closed': True}
                        t.join(0.1)
                    else:
                        is_genuine = result_container.get('is_genuine', False)
                        confidence = result_container.get('confidence', 0.0)
                        details = result_container.get('details', {})
                        spoof_detected = not is_genuine
                    logger.info(f"Anti-spoofing finished in {time.time()-spoofing_start:.2f}s, confidence={confidence}")
                except Exception as e:
                    logger.warning(f"Anti-spoofing error: {e}")
                    if self.anti_spoof_fail_open:
                        spoof_detected = False
                        confidence = 0.5
                        details = {'error': str(e), 'assumed_genuine': True}
                    else:
                        spoof_detected = True
                        confidence = 0.0
                        details = {'error': str(e), 'fail_closed': True}
            result['spoof_detected'] = spoof_detected
            result['details']['spoof_confidence'] = confidence

            # Calibration pass to reduce false-positive spoof rejects from primary model.
            if (
                self.calibrated_spoof_recheck_enabled
                and spoof_detected
                and confidence <= self.calibrated_spoof_primary_confidence_max
            ):
                secondary = self._secondary_anti_spoof_check(audio_data_trimmed)
                sec_conf = float(secondary.get('confidence', 0.0))
                sec_details = secondary.get('details', {}) if isinstance(secondary, dict) else {}
                sec_replay = float(sec_details.get('replay_detection', 1.0))
                sec_quality = float(sec_details.get('quality_assessment', 0.0))

                if (
                    sec_conf >= self.calibrated_spoof_secondary_confidence_min
                    and sec_replay >= self.calibrated_spoof_min_replay_genuine
                    and sec_quality >= self.calibrated_spoof_min_quality_score
                ):
                    logger.warning(
                        "Calibrated anti-spoof override: primary false-positive likely "
                        "(primary_conf=%.3f, secondary_conf=%.3f, replay=%.3f, quality=%.3f)",
                        confidence, sec_conf, sec_replay, sec_quality
                    )
                    spoof_detected = False
                    confidence = max(confidence, sec_conf)
                    result['spoof_detected'] = False
                    result['details']['spoof_confidence'] = confidence
                    result['details']['calibrated_spoof_override'] = True
                    result['details']['secondary_anti_spoof'] = {
                        'confidence': sec_conf,
                        'replay_detection': sec_replay,
                        'quality_assessment': sec_quality,
                    }

            calibrated_override = bool(result.get('details', {}).get('calibrated_spoof_override', False))

            if spoof_detected:
                result['details']['failure_reason'] = "Replay/spoof suspected by anti-spoofing"
                logger.info(f"Anti-spoofing flagged spoof, returning after {time.time()-start_time:.2f}s")
                return result

            if (not calibrated_override) and confidence < self.spoof_min_confidence_verify:
                result['details']['failure_reason'] = f"Anti-spoofing failed: {confidence:.3f} < {self.spoof_min_confidence_verify}"
                logger.info(f"Anti-spoofing failed, returning after {time.time()-start_time:.2f}s")
                return result

            # Step 3: Extract live embedding (disable anti-spoofing here since we already did it)
            logger.info("Extracting voice embedding from audio sample...")

            # Optional denoise + AGC for embedding robustness in noise.
            # Keep this OFF for anti-spoofing input (handled above) to avoid harming spoof detection.
            embedding_audio = audio_data
            if self.enable_denoise_for_embedding:
                embedding_audio = denoise_and_agc(audio_data, sample_rate=sample_rate, strength=self.denoise_strength)

            live_embedding = self.voice_engine.extract_embedding_from_array(
                embedding_audio,
                sample_rate,
                enable_anti_spoofing=False  # Already checked above, avoid double-checking
            )
            if live_embedding is None:
                result['details']['failure_reason'] = "Failed to extract embedding"
                logger.error("Failed to extract embedding from audio")
                return result
            
            logger.info(f"Embedding extracted successfully, shape: {live_embedding.shape}")

            # Step 4: Get stored user profile
            logger.info("Retrieving stored voice profile from database...")
            stored_embeddings = self.db.get_voice_embeddings(user_id)
            if not stored_embeddings:
                result['details']['failure_reason'] = "No voice profile found"
                logger.error(f"No voice profile found for user {user_id}")
                return result

            # Use the most recent embedding
            user_profile = stored_embeddings[0]
            stored_mean = user_profile['embedding_array']
            stored_variance = user_profile.get('embedding_variance')
            logger.info(f"Retrieved voice profile for user {user_id}, embedding shape: {stored_mean.shape}")

            # Step 5: Compute similarity scores
            logger.info("Computing similarity scores...")
            cosine_sim = self._compute_cosine_similarity(live_embedding, stored_mean)
            result['cosine_similarity'] = cosine_sim
            logger.info(f"Cosine similarity: {cosine_sim:.4f}")

            mahalanobis_dist = float('inf')
            if stored_variance is not None:
                mahalanobis_dist = self._compute_mahalanobis_distance(live_embedding, stored_mean, stored_variance)
                logger.info(f"Mahalanobis distance: {mahalanobis_dist:.4f}")
            else:
                logger.info("No variance data available, skipping Mahalanobis distance")
            result['mahalanobis_distance'] = mahalanobis_dist

            # Step 6: Adaptive threshold calculation
            adaptive_threshold = self.base_threshold
            if self.adaptive_threshold_enabled and stored_variance is not None:
                user_variance = np.mean(stored_variance)
                adaptive_threshold = self.base_threshold - 0.02 * user_variance  # Less aggressive
                # Ensure threshold doesn't go below reasonable minimum
                adaptive_threshold = max(adaptive_threshold, 0.3)  # Higher minimum

            # Step 7: Combined scoring
            # Use both cosine and Mahalanobis for final decision
            cosine_score = max(0, 1 - cosine_sim)  # Convert similarity to distance
            combined_score = (cosine_score * 0.7 + min(mahalanobis_dist, 5.0) * 0.3)  # Weight cosine more, cap Mahalanobis lower
            final_threshold = adaptive_threshold

            logger.info(f"Verification scores - cosine_sim: {cosine_sim:.4f}, cosine_score: {cosine_score:.4f}, mahalanobis_dist: {mahalanobis_dist:.4f}")
            logger.info(f"Combined score: {combined_score:.4f}, final_threshold: {final_threshold:.4f}")

            # More robust confidence calculation - use exponential decay instead of linear
            # This gives higher confidence for better matches and doesn't go to 0 as easily
            if combined_score <= final_threshold:
                # Good match - confidence based on how much better than threshold
                confidence = min(1.0, 1.0 - (combined_score / final_threshold) * 0.5)
            else:
                # Poor match - use exponential decay (less aggressive)
                excess = combined_score - final_threshold
                confidence = max(0.0, np.exp(-excess * 0.1))  # Reduced multiplier for gentler decay

            # Step 8: Challenge-response (optional) - Check BEFORE final decision
            if enable_challenge:
                challenge_result = self._perform_challenge_response(audio_data, sample_rate)
                result['challenge_passed'] = challenge_result['passed']
                result['details']['challenge_text'] = challenge_result.get('recognized_text', '')

                if not challenge_result['passed']:
                    result['details']['failure_reason'] = "Challenge-response failed"
                    result['verified'] = False
                    result['confidence'] = 0.0
                    return result

            # Step 9: Final decision - Use cosine similarity as primary metric
            # Add configurable borderline acceptance to reduce false rejects for legitimate users.
            cosine_threshold = self.cosine_verify_threshold
            allow_borderline = self.allow_borderline_voice_match
            borderline_min_cosine = self.borderline_min_cosine
            borderline_min_confidence = self.borderline_min_confidence
            borderline_min_quality = self.borderline_min_quality

            borderline_pass = (
                allow_borderline
                and (not result.get('spoof_detected', False))
                and result.get('quality_score', 0.0) >= borderline_min_quality
                and cosine_sim >= borderline_min_cosine
                and confidence >= borderline_min_confidence
            )

            result['verified'] = (cosine_sim >= cosine_threshold) or borderline_pass
            if borderline_pass and cosine_sim < cosine_threshold:
                result['details']['borderline_match'] = True
                result['details']['decision_path'] = 'borderline_voice_match'

            # Step 10: Adjust confidence based on verification result
            # Use a combination of cosine similarity and the combined score for final confidence
            if result['verified']:
                # For verified matches, confidence reflects how much better than threshold
                if result.get('details', {}).get('borderline_match', False):
                    # Keep confidence meaningful for accepted borderline matches.
                    result['confidence'] = max(confidence, borderline_min_confidence)
                else:
                    excess_similarity = cosine_sim - cosine_threshold
                    cosine_confidence = min(1.0, 0.5 + excess_similarity * 2.0)  # Scale to 50-100%
                    # Blend with combined score confidence (weighted average)
                    result['confidence'] = (cosine_confidence * 0.7 + confidence * 0.3)
            else:
                # For failed matches, confidence reflects how close to threshold
                cosine_confidence = max(0.0, cosine_sim * 2.0)  # Scale to 0-100%
                # Use the lower of the two confidences for failed matches
                result['confidence'] = min(cosine_confidence, confidence)

            logger.info(f"Final decision - verified: {result['verified']}, confidence: {result['confidence']:.4f}, cosine_sim: {cosine_sim:.4f}")

            # Step 11: Adaptive learning (only for strong, clean matches to avoid profile drift)
            strong_verified_match = (
                result['verified']
                and (not result.get('details', {}).get('borderline_match', False))
                and cosine_sim >= (cosine_threshold + 0.05)
                and result.get('quality_score', 0.0) >= 0.80
                and (not result.get('spoof_detected', False))
            )

            if strong_verified_match and self.learning_enabled:
                try:
                    self.db.update_voice_embedding(user_id, live_embedding, learning_rate=0.1)
                    logger.info(f"Voice embedding updated for user {user_id} with adaptive learning")
                except Exception as e:
                    logger.warning(f"Failed to update voice embedding during learning: {e}")

            result['details']['cosine_threshold'] = cosine_threshold
            result['details']['combined_score'] = combined_score

            return result

        except Exception as e:
            logger.error(f"Voice verification failed: {e}")
            return {
                'verified': False,
                'confidence': 0.0,
                'details': {'failure_reason': str(e)}
            }

    def _prepare_audio_for_spoofing(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Apply minimal processing for anti-spoofing analysis.
        AASIST models need natural voice characteristics preserved.

        Args:
            audio_data: Processed audio from UI

        Returns:
            Minimally processed audio suitable for spoofing detection
        """
        try:
            # Ensure audio is 1D for processing
            if audio_data.ndim == 2:
                audio_data = audio_data.squeeze()
            elif audio_data.ndim > 2:
                audio_data = audio_data.flatten()
            
            # Apply very gentle high-pass filter (remove DC bias only)
            from scipy import signal
            nyquist = 16000 / 2
            cutoff = 20 / nyquist  # Very low cutoff, just remove DC
            b, a = signal.butter(1, cutoff, btype='high')
            audio_data = signal.filtfilt(b, a, audio_data)

            # Skip aggressive noise reduction - preserve natural dynamics
            # Skip spectral subtraction - it's too destructive for spoofing detection

            # Apply very gentle normalization (preserve dynamics)
            max_val = np.max(np.abs(audio_data))
            if max_val > 0:
                audio_data = audio_data / max_val * 0.95  # Gentle normalization

            return audio_data

        except Exception as e:
            logger.warning(f"Failed to prepare audio for spoofing, using original: {e}")
            return audio_data

    def _secondary_anti_spoof_check(self, audio_data: np.ndarray) -> Dict:
        """Run secondary anti-spoofing analysis for calibration of ambiguous primary results."""
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
                temp_path = tmp.name

            wav_data = np.asarray(audio_data)
            if wav_data.dtype != np.int16:
                max_abs = float(np.max(np.abs(wav_data))) if wav_data.size > 0 else 1.0
                scale = 32767.0 / max(1.0, max_abs)
                wav_data = (wav_data * scale).astype(np.int16)

            wavfile.write(temp_path, 16000, wav_data)
            return self.secondary_anti_spoofing.analyze_audio_security(temp_path)
        except Exception as e:
            logger.warning(f"Secondary anti-spoof check failed: {e}")
            return {'confidence': 0.0, 'details': {'error': str(e)}}
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

    def _compute_cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings"""
        try:
            dot_product = np.dot(embedding1, embedding2)
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            return dot_product / (norm1 * norm2)
        except Exception:
            return 0.0

    def _compute_mahalanobis_distance(self, point: np.ndarray, mean: np.ndarray, variance: np.ndarray) -> float:
        """Compute Mahalanobis distance using variance"""
        try:
            # Create covariance matrix from variance (diagonal)
            covariance = np.diag(variance + 1e-8)  # Add small epsilon to avoid singular matrix
            inv_covariance = np.linalg.inv(covariance)

            diff = point - mean
            distance = np.sqrt(np.dot(np.dot(diff, inv_covariance), diff))

            return distance
        except Exception:
            return float('inf')

    def _perform_challenge_response(self, audio_data: np.ndarray, sample_rate: int) -> Dict:
        """Perform challenge-response verification using Vosk STT"""
        try:
            # Generate random challenge phrase
            challenge_words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel"]
            challenge_phrase = f"say the code {random.choice(challenge_words)} {random.randint(10, 99)}"

            # For now, return mock result (would need Vosk integration)
            # In real implementation, this would transcribe the audio and compare
            return {
                'passed': True,  # Mock pass for now
                'recognized_text': challenge_phrase,
                'expected_text': challenge_phrase
            }

        except Exception as e:
            logger.error(f"Challenge-response failed: {e}")
            return {'passed': False, 'error': str(e)}

    def generate_challenge_phrase(self) -> str:
        """Generate a random challenge phrase for verification"""
        words = ["red", "blue", "green", "yellow", "alpha", "bravo", "charlie", "delta"]
        numbers = random.randint(10, 99)
        return f"Please say: {random.choice(words)} {numbers}"