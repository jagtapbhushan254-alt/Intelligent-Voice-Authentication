"""
SecureX-Assist - Voice Biometric Engine
Advanced AI-powered voice recognition and verification system with anti-spoofing
"""

import warnings
import torch
import numpy as np
from typing import Optional, Tuple, Dict
import logging
from pathlib import Path
from core.anti_spoofing import AntiSpoofingEngine, VoiceNormalizer
from core.anti_spoofing import AntiSpoofingEngine
from core.audio_processor import VoiceActivityDetector

# Suppress all warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message=".*libtorchcodec.*")
warnings.filterwarnings('ignore', message=".*FFmpeg.*")

logger = logging.getLogger(__name__)

# Import torchaudio with suppressed warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import torchaudio

# Force torchaudio to use soundfile backend (no FFmpeg needed!)
try:
    torchaudio.set_audio_backend("soundfile")
    logger.debug("Using soundfile backend for audio loading")
except Exception as e:
    logger.debug(f"Could not set soundfile backend: {e}. Will use scipy fallback.")


class VoiceEngine:
    """
    Core voice biometric engine with dual-backend support
    Primary: pyannote/embedding
    Fallback: SpeechBrain ECAPA-TDNN
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.primary_model = None
        self.fallback_model = None
        self.active_backend = None
        
        # Initialize anti-spoofing engines (dual-mode: basic + advanced)
        self.anti_spoofing = AntiSpoofingEngine(config)  # Primary
        self.voice_normalizer = VoiceNormalizer()
        
        # Initialize VAD with config
        self.vad = VoiceActivityDetector(config)
        
        logger.info(f"Initializing VoiceEngine on {self.device} with anti-spoofing enabled")
    
    def load_models(self) -> bool:
        """
        Lazy load AI models with intelligent fallback
        Returns True if at least one model loads successfully
        """
        try:
            # Fast path: allow config to prefer lightweight backend
            prefer_basic = self.config.get('models', {}).get('prefer_basic', False)
            fast_mode = self.config.get('system', {}).get('fast_mode', False)
            if prefer_basic or fast_mode:
                logger.info("Fast mode / prefer_basic enabled - using basic MFCC backend")
                self.active_backend = "basic"
                return True
            # Try loading primary model (pyannote/embedding)
            logger.info("Loading primary model: pyannote/embedding")
            from pyannote.audio import Model, Inference
            import os
            
            model_name = self.config.get('models', {}).get('primary_embedding', 'pyannote/embedding')
            hf_token = os.environ.get("HF_TOKEN")
            # Pass the token or True so it checks the environment
            self.primary_model = Model.from_pretrained(model_name, use_auth_token=hf_token if hf_token else True)
            self.inference = Inference(self.primary_model, window="whole")
            self.active_backend = "pyannote"
            
            logger.info("Primary model loaded successfully")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to load primary model: {e}")
            
            # Try fallback model (SpeechBrain)
            try:
                logger.info("Loading fallback model: SpeechBrain ECAPA-TDNN")
                from speechbrain.pretrained import EncoderClassifier
                
                # Fix Windows WinError 1314 for SpeechBrain cache symlinking
                import os
                if os.name == 'nt' and not hasattr(os, '_original_symlink'):
                    os._original_symlink = os.symlink
                    def custom_symlink(src, dst, target_is_directory=False, **kwargs):
                        import shutil
                        if os.path.exists(dst): return
                        try:
                            if os.path.isdir(src):
                                shutil.copytree(src, dst, dirs_exist_ok=True)
                            else:
                                shutil.copy2(src, dst)
                        except Exception as patch_e:
                            logger.debug(f"Fallback copy failed: {patch_e}")
                    os.symlink = custom_symlink
                
                fallback_name = self.config.get('models', {}).get(
                    'fallback_embedding', 
                    'speechbrain/spkrec-ecapa-voxceleb'
                )
                self.fallback_model = EncoderClassifier.from_hparams(
                    source=fallback_name,
                    savedir="models/speechbrain"
                )
                self.active_backend = "speechbrain"
                
                logger.info("Fallback model loaded successfully")
                return True
                
            except Exception as fallback_error:
                logger.error(f"Failed to load fallback model: {fallback_error}")
                
                # Try basic fallback using simple MFCC features
                try:
                    logger.info("Using basic MFCC-based embedding as final fallback")
                    self.active_backend = "basic"
                    logger.info("Basic embedding fallback loaded successfully")
                    return True
                    
                except Exception as basic_error:
                    logger.error(f"Failed to load basic fallback: {basic_error}")
                    return False
    
    def extract_embedding(self, audio_path: str, enable_anti_spoofing: bool = True) -> Optional[np.ndarray]:
        """
        Extract 512-dimensional voice embedding from audio file with security checks
        
        Args:
            audio_path: Path to audio file (WAV format preferred)
            enable_anti_spoofing: Enable anti-spoofing and replay attack detection
            
        Returns:
            512-dimensional numpy array or None if extraction fails or audio is suspicious
        """
        try:
            # Step 1: Normalize audio for better recognition across environments
            normalized_path = self.voice_normalizer.normalize_audio(audio_path)
            
            # Step 2: Validate speech is present (reject silence/no speech) - DO THIS FIRST, NO TRIMMING
            if not self._validate_speech_present(normalized_path):
                logger.error("No valid speech detected in audio - recording appears to be silence or noise")
                return None
            
            # Step 3: Run anti-spoofing checks if enabled
            if enable_anti_spoofing:
                # Use primary anti-spoofing
                security_result = self.anti_spoofing.analyze_audio_security(normalized_path)
                
                if not security_result['is_live']:
                    logger.warning(f"REPLAY ATTACK DETECTED! Spoof score: {security_result['spoof_score']:.2%}")
                    logger.warning(f"Details: {security_result['details']}")
                    return None
                
                if not security_result['is_genuine']:
                    logger.warning(f"SYNTHETIC/DEEPFAKE DETECTED! Spoof score: {security_result['spoof_score']:.2%}")
                    logger.warning(f"Details: {security_result['details']}")
                    return None
                
                logger.info(f"Audio passed security checks (confidence: {security_result['confidence']:.2%})")
                logger.info(f"   - Replay detection: {security_result['details']['replay_detection']:.2%}")
                logger.info(f"   - Synthesis detection: {security_result['details']['synthesis_detection']:.2%}")
                logger.info(f"   - Liveness check: {security_result['details']['liveness_check']:.2%}")
                logger.info(f"   - Spectral artifacts: {security_result['details']['spectral_artifacts']:.2%}")
            
            # Legacy anti-spoofing fallback
            elif False:  # Disabled, using advanced only
                security_result = self.anti_spoofing.analyze_audio_security(normalized_path)
                
                if not security_result['is_live']:
                    logger.warning(f"REPLAY ATTACK DETECTED! Confidence: {security_result['confidence']:.2%}")
                    logger.warning(f"Details: {security_result['details']}")
                    return None
                
                if not security_result['is_genuine']:
                    logger.warning(f"SUSPICIOUS AUDIO! May be synthesized or heavily processed.")
                    logger.warning(f"Details: {security_result['details']}")
                    return None
                
                logger.info(f"Audio passed security checks (confidence: {security_result['confidence']:.2%})")
                logger.info(f"   - Replay detection: {security_result['details']['replay_detection']:.2%}")
                logger.info(f"   - Liveness: {security_result['details']['liveness_detection']:.2%}")
                logger.info(f"   - Quality: {security_result['details']['quality_assessment']:.2%}")
            
            # Extract embedding from normalized audio (no trimming)
            print(f"DEBUG: Extracting embedding with backend: {self.active_backend}")
            if self.active_backend == "pyannote":
                return self._extract_pyannote(normalized_path)
            elif self.active_backend == "speechbrain":
                return self._extract_speechbrain(normalized_path)
            elif self.active_backend == "basic":
                return self._extract_basic(normalized_path)
            else:
                logger.error("No active backend available")
                return None
                
        except Exception as e:
            logger.error(f"Embedding extraction failed: {e}")
            return None
    
    def extract_embedding_from_array(self, audio_data: np.ndarray, sample_rate: int = 16000, 
                                   enable_anti_spoofing: bool = True) -> Optional[np.ndarray]:
        """
        Extract voice embedding directly from numpy array
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of audio data
            enable_anti_spoofing: Enable anti-spoofing checks
            
        Returns:
            512-dimensional embedding or None
        """
        try:
            # Save temporary WAV file
            import tempfile
            import soundfile as sf
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_path = temp_file.name
                sf.write(temp_path, audio_data, sample_rate)
            
            try:
                # Use existing extraction method
                embedding = self.extract_embedding(temp_path, enable_anti_spoofing)
                return embedding
            finally:
                # Clean up temp file
                Path(temp_path).unlink(missing_ok=True)
                
        except Exception as e:
            logger.error(f"Array embedding extraction failed: {e}")
            return None
    
    def _validate_speech_present(self, audio_path: str) -> bool:
        """
        Validate that actual speech is present in the audio (reject silence)
        
        Returns: True if valid speech detected, False if silence/noise
        """
        try:
            from scipy.io import wavfile
            
            # Read audio
            sample_rate, audio_data = wavfile.read(audio_path)
            
            # Convert to float32
            if audio_data.dtype == np.int16:
                audio_data = audio_data.astype(np.float32) / 32768.0
            elif audio_data.dtype == np.int32:
                audio_data = audio_data.astype(np.float32) / 2147483648.0
            
            # Convert stereo to mono
            if audio_data.ndim == 2:
                audio_data = np.mean(audio_data, axis=1)
            
            # Calculate audio energy
            energy = np.sum(audio_data ** 2) / len(audio_data)
            
            # Get min energy threshold from config
            min_energy = self.config.get('audio', {}).get('min_audio_energy', 0.01)
            
            if energy < min_energy:
                logger.warning(f"Audio energy too low: {energy:.6f} < {min_energy} - Appears to be silence")
                return False
            
            # Calculate speech duration (frames with significant energy)
            frame_length = int(0.02 * sample_rate)  # 20ms frames
            frame_energy = np.array([
                np.sum(audio_data[i:i+frame_length]**2)
                for i in range(0, len(audio_data) - frame_length, frame_length)
            ])
            
            # Count speech frames (above 1% of max energy - very lenient)
            threshold = np.max(frame_energy) * 0.01
            speech_frames = np.sum(frame_energy > threshold)
            speech_duration = speech_frames * 0.02  # Convert to seconds
            
            # Get min speech duration from config (default 1.0s for enrollment)
            min_speech_duration = self.config.get('audio', {}).get('min_speech_duration', 1.0)
            
            if speech_duration < min_speech_duration:
                logger.warning(f"Speech duration too short: {speech_duration:.1f}s < {min_speech_duration}s")
                return False
            
            logger.info(f"Speech validation passed: energy={energy:.6f}, duration={speech_duration:.1f}s")
            return True
            
        except Exception as e:
            logger.error(f"Speech validation failed: {e}")
            return False
    
    def _trim_silence(self, audio_path: str) -> str:
        """
        Trim leading and trailing silence from audio for more consistent embeddings
        
        Returns: Path to trimmed audio file
        """
        try:
            from scipy.io import wavfile
            import scipy.signal as signal
            
            # Read audio
            sample_rate, audio_data = wavfile.read(audio_path)
            
            # Convert to float32
            if audio_data.dtype == np.int16:
                audio_data = audio_data.astype(np.float32) / 32768.0
            elif audio_data.dtype == np.int32:
                audio_data = audio_data.astype(np.float32) / 2147483648.0
            
            # Convert stereo to mono
            if audio_data.ndim == 2:
                audio_data = np.mean(audio_data, axis=1)
            
            # Simple energy-based trimming (gentle - only remove true silence)
            frame_length = int(0.02 * sample_rate)  # 20ms frames
            energy = np.array([
                np.sum(audio_data[i:i+frame_length]**2)
                for i in range(0, len(audio_data) - frame_length, frame_length)
            ])
            
            # Find speech boundaries (energy above 0.5% of max - very gentle trimming)
            threshold = np.max(energy) * 0.005
            speech_frames = np.where(energy > threshold)[0]
            
            if len(speech_frames) > 0:
                # Add padding (10 frames = 200ms on each side for safety)
                start_frame = max(0, speech_frames[0] - 10)
                end_frame = min(len(energy), speech_frames[-1] + 10)
                
                # Convert to sample indices
                start_sample = start_frame * frame_length
                end_sample = min(len(audio_data), (end_frame + 1) * frame_length)
                
                # Trim audio
                trimmed_audio = audio_data[start_sample:end_sample]
            else:
                # No speech detected, use full audio
                trimmed_audio = audio_data
            
            # Save trimmed audio
            trimmed_path = audio_path.replace('.wav', '_trimmed.wav')
            
            # Convert back to int16
            trimmed_int16 = (trimmed_audio * 32767).astype(np.int16)
            wavfile.write(trimmed_path, sample_rate, trimmed_int16)
            
            logger.info(f"Trimmed audio: {len(audio_data)} -> {len(trimmed_audio)} samples")
            return trimmed_path
            
        except Exception as e:
            logger.warning(f"Silence trimming failed, using original: {e}")
            return audio_path
    
    def _extract_pyannote(self, audio_path: str) -> np.ndarray:
        """Extract embedding using pyannote model - robust audio loading (single-path)."""
        # Prefer soundfile/torchaudio; fallback to scipy.wavfile only if needed
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                waveform, sample_rate = torchaudio.load(audio_path)  # waveform: (channels, samples)
                # Convert to float32 (torchaudio already does this)
                waveform = waveform.float()
        except Exception as e:
            # Fallback to scipy (WAV only)
            logger.warning(f"torchaudio.load failed ({e}). Falling back to scipy.wavfile")
            from scipy.io import wavfile
            sample_rate, audio_np = wavfile.read(audio_path)
            # Normalize int types to float32
            if audio_np.dtype == np.int16:
                audio_np = audio_np.astype(np.float32) / 32768.0
            elif audio_np.dtype == np.int32:
                audio_np = audio_np.astype(np.float32) / 2147483648.0
            # Make tensor in shape (channels, samples)
            if audio_np.ndim == 1:
                waveform = torch.from_numpy(audio_np).unsqueeze(0).float()
            else:
                waveform = torch.from_numpy(audio_np.T).float()
        # Resample to 16kHz if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
            sample_rate = 16000
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        audio_dict = {"waveform": waveform, "sample_rate": sample_rate}
        embedding = self.inference(audio_dict)
        if torch.is_tensor(embedding):
            embedding = embedding.cpu().numpy()
        if embedding.ndim > 1:
            embedding = embedding.flatten()
        
        # L2 normalize the embedding for consistent cosine similarity calculation
        embedding_norm = np.linalg.norm(embedding)
        if embedding_norm > 0:
            embedding = embedding / embedding_norm
        
        logger.info(f"Extracted pyannote embedding: shape={embedding.shape}")
        return embedding
    
    def _extract_speechbrain(self, audio_path: str) -> np.ndarray:
        """Extract embedding using SpeechBrain model"""
        # Load audio
        
        # Use scipy directly if we've had FFmpeg issues
        if hasattr(self, '_use_scipy_directly') and self._use_scipy_directly:
            from scipy.io import wavfile
            fs, wav_data = wavfile.read(audio_path)
            signal = torch.from_numpy(wav_data.copy()).float()
            if len(signal.shape) == 1:
                signal = signal.unsqueeze(0)
            else:
                signal = signal.t()
        else:
            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    signal, fs = torchaudio.load(audio_path)
            except Exception as e:
                # Fallback to scipy
                self._use_scipy_directly = True
                from scipy.io import wavfile
                fs, wav_data = wavfile.read(audio_path)
                signal = torch.from_numpy(wav_data.copy()).float()
                if len(signal.shape) == 1:
                    signal = signal.unsqueeze(0)
                else:
                    signal = signal.t()
        
        # Resample if needed
        if fs != 16000:
            resampler = torchaudio.transforms.Resample(fs, 16000)
            signal = resampler(signal)
        
        # Extract embedding
        embedding = self.fallback_model.encode_batch(signal)
        embedding = embedding.squeeze().cpu().numpy()
        
        # L2 normalize the embedding for consistent cosine similarity calculation
        embedding_norm = np.linalg.norm(embedding)
        if embedding_norm > 0:
            embedding = embedding / embedding_norm
        
        logger.info(f"Extracted SpeechBrain embedding: shape={embedding.shape}")
        return embedding
    
    def _extract_basic(self, audio_path: str) -> np.ndarray:
        """Extract basic MFCC-based embedding as fallback"""
        try:
            print(f"DEBUG: Starting basic embedding extraction for {audio_path}")
            import librosa
            
            # Load audio
            y, sr = librosa.load(audio_path, sr=16000)
            print(f"DEBUG: Audio loaded: shape={y.shape}, sr={sr}")
            
            # Extract MFCC features
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=512, hop_length=160)
            print(f"DEBUG: MFCC extracted: shape={mfcc.shape}")
            
            # Take mean across time dimension to get fixed-size embedding
            embedding = np.mean(mfcc, axis=1)
            print(f"DEBUG: Embedding after mean: shape={embedding.shape}")
            
            # Pad or truncate to 512 dimensions (expected by rest of system)
            if len(embedding) < 512:
                embedding = np.pad(embedding, (0, 512 - len(embedding)), 'constant')
                print(f"DEBUG: Embedding padded to: shape={embedding.shape}")
            else:
                embedding = embedding[:512]
                print(f"DEBUG: Embedding truncated to: shape={embedding.shape}")
            
            # L2 normalize the embedding for consistent cosine similarity calculation
            embedding_norm = np.linalg.norm(embedding)
            if embedding_norm > 0:
                embedding = embedding / embedding_norm
            
            logger.info(f"Extracted basic MFCC embedding: shape={embedding.shape}")
            print(f"DEBUG: Basic embedding extraction successful")
            return embedding
            
        except Exception as e:
            logger.error(f"Basic embedding extraction failed: {e}")
            print(f"DEBUG: Basic embedding extraction failed: {e}")
            # Return a zero vector as last resort
            return np.zeros(512, dtype=np.float32)
    
    def verify_speaker(
        self, 
        test_embedding: np.ndarray, 
        stored_embedding: np.ndarray,
        threshold: Optional[float] = None
    ) -> Tuple[bool, float]:
        """
        Return (is_match, similarity_score)
        similarity_score in [0,1] (higher=more similar). Default threshold ~0.5
        """
        if threshold is None:
            threshold = self.config.get('security', {}).get('voice_similarity_threshold', 0.50)
        try:
            sim = self._cosine_similarity(test_embedding, stored_embedding)  # 0..1
            is_match = sim >= threshold
            logger.info(f"Voice verification: similarity={sim:.4f}, threshold={threshold:.4f}, match={is_match}")
            return is_match, float(sim)
        except Exception as e:
            logger.error(f"Verification failed: {e}")
            return False, 0.0
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        vec1 = vec1 / (np.linalg.norm(vec1) + 1e-8)
        vec2 = vec2 / (np.linalg.norm(vec2) + 1e-8)
        return float(np.dot(vec1, vec2))

    def compute_user_reference(self, embeddings: list):
        mean_emb = np.mean(np.stack(embeddings, axis=0), axis=0)
        sims = [self._cosine_similarity(e, mean_emb) for e in embeddings]
        avg_sim = float(np.mean(sims))
        std_sim = float(np.std(sims))
        user_threshold = max(0.40, avg_sim - 1.0 * std_sim)  # conservative
        return mean_emb, user_threshold, avg_sim, std_sim
    
    def calculate_embedding_quality(self, embedding: np.ndarray) -> float:
        """
        Assess the quality of a voice embedding
        Returns quality score 0.0-1.0 (higher is better)
        """
        try:
            # Check for NaN or Inf values
            if not np.isfinite(embedding).all():
                return 0.0
            
            # Check variance (too low variance = poor quality)
            variance = np.var(embedding)
            if variance < 0.001:
                return 0.3
            
            # Check magnitude
            magnitude = np.linalg.norm(embedding)
            if magnitude < 0.1:
                return 0.4
            
            # Good quality embedding
            return 1.0
            
        except Exception as e:
            logger.error(f"Quality check failed: {e}")
            return 0.0
    
    def is_ready(self) -> bool:
        """Check if voice engine is ready for operation"""
        return self.active_backend is not None


class VoiceQualityAnalyzer:
    """
    Analyzes audio quality and detects potential spoofing attempts
    """
    
    def __init__(self, min_duration=1.0, min_energy=0.005):
        """Initialize with quality thresholds"""
        self.min_duration = min_duration
        self.min_energy = min_energy
    
    def analyze_audio(self, audio_path: str) -> Dict[str, float]:
        """
        Analyze audio file for comprehensive quality metrics
        
        Returns dict with:
        - overall_quality: Overall quality score (0-1)
        - snr: Signal-to-noise ratio (dB)
        - duration: Audio duration in seconds
        - energy: Average signal energy
        - zero_crossings: Zero crossing rate (voice activity indicator)
        - speech_percent: Estimated speech percentage
        - sample_rate: Sampling rate
        - clipping: Percentage of clipped samples (too loud)
        - is_synthetic: Likelihood of being synthetic (0-1)
        """
        try:
            # Load audio with scipy fallback
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    waveform, sample_rate = torchaudio.load(audio_path)
            except Exception:
                from scipy.io import wavfile
                sample_rate, audio_np = wavfile.read(audio_path)
                if audio_np.dtype == np.int16:
                    audio_np = audio_np.astype(np.float32) / 32768.0
                waveform = torch.from_numpy(audio_np).float()
                if waveform.ndim == 1:
                    waveform = waveform.unsqueeze(0)
            
            # Convert to numpy for calculations
            waveform_np = waveform.numpy().flatten()
            
            # Calculate duration
            duration = waveform.shape[1] / sample_rate
            
            # Calculate energy
            energy = float(np.mean(np.abs(waveform_np)))
            
            # Calculate energy variability
            if len(waveform_np) > 160:  # At least 10ms @ 16kHz
                frame_length = 160  # 10ms @ 16kHz
                frames = [waveform_np[i:i+frame_length] for i in range(0, len(waveform_np)-frame_length, frame_length)]
                frame_energies = [np.mean(np.abs(frame)) for frame in frames]
                energy_variability = np.std(frame_energies)
            
            # Estimate SNR
            noise_floor = np.sort(np.abs(waveform_np))[int(len(waveform_np) * 0.1)]
            signal = np.mean(np.square(waveform_np))
            noise = np.square(noise_floor)
            snr = 10 * np.log10((signal - noise) / (noise + 1e-8) + 1e-8)
            
            # Extract MFCC for variability analysis
            try:
                import librosa
                # Extract MFCC features (standard for voice analysis)
                mfcc = librosa.feature.mfcc(y=waveform_np, sr=sample_rate, n_mfcc=13)
            except Exception:
                # Create dummy MFCC if librosa fails
                mfcc = np.random.randn(13, 100) * 0.01
                
            # Estimate pitch and pitch variability
            try:
                import librosa
                # Extract pitch (fundamental frequency)
                pitches, magnitudes = librosa.piptrack(y=waveform_np, sr=sample_rate)
                pitch_values = []
                for i in range(pitches.shape[1]):
                    index = magnitudes[:, i].argmax()
                    pitch = pitches[index, i]
                    if pitch > 0:  # Valid pitch
                        pitch_values.append(pitch)
                
                # Calculate pitch variability
                if pitch_values:
                    pitch_variability = np.std(pitch_values)
                else:
                    pitch_variability = 0.0
            except Exception:
                # Default pitch variability if analysis fails
                pitch_variability = 0.0
            
            # Calculate zero crossings (proxy for speech activity)
            zero_crossings = np.sum(np.diff(np.signbit(waveform_np))) / len(waveform_np)
            
            # Calculate clipping (indicates distortion)
            clipping = np.mean(np.abs(waveform_np) > 0.98) * 100
            
            # Estimate speech percentage
            speech_percent = min(100, zero_crossings * 1000)
            
            # Calculate overall quality score (0-1)
            quality_score = 0.0
            
            # Duration check
            if duration >= self.min_duration:
                quality_score += 0.2
            else:
                quality_score += 0.2 * (duration / self.min_duration)
                
            # Energy check  
            if energy >= self.min_energy:
                quality_score += 0.2
            else:
                quality_score += 0.2 * (energy / self.min_energy)
                
            # SNR check
            if snr >= 15:
                quality_score += 0.2
            else:
                quality_score += 0.2 * min(1.0, max(0, snr / 15))
                
            # Clipping check
            if clipping < 1.0:
                quality_score += 0.2
            else:
                quality_score += 0.2 * max(0, 1.0 - (clipping / 5.0))
                
            # Speech percentage check
            if speech_percent >= 50:
                quality_score += 0.2
            else:
                quality_score += 0.2 * (speech_percent / 50)
            
            # Cap quality score at 1.0
            quality_score = min(1.0, quality_score)
            
            # For synthetic detection, we'd use advanced model in production
            is_synthetic = 0.0
            
            return {
                'snr': float(snr),
                'duration': float(duration),
                'sample_rate': int(sample_rate),
                'synthetic_probability': float(is_synthetic),
                'quality_score': float(quality_score),
                'speech_percent': float(speech_percent),
                'clipping': float(clipping),
                'energy': float(energy),
                'mfcc_variability': float(np.mean(np.std(mfcc, axis=0))),
                'energy_variability': float(energy_variability),
                'pitch_variability': float(pitch_variability)
            }
            
        except Exception as e:
            logger.error(f"Audio analysis failed: {e}")
            return {
                'snr': 0.0,
                'duration': 0.0,
                'sample_rate': 0,
                'synthetic_probability': 1.0,
                'quality_score': 0.0,
                'speech_percent': 0.0,
                'clipping': 0.0,
                'energy': 0.0,
                'mfcc_variability': 0.0,
                'energy_variability': 0.0,
                'pitch_variability': 0.0
            }
