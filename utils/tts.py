"""SecureX-Assist - Text-to-Speech System.

Audio feedback and confirmation using Piper TTS with a Windows SAPI/pyttsx3 fallback.
Supports selecting a preferred voice (e.g., female) when using the system TTS engine.
"""

import logging
from typing import Optional
import threading
import sounddevice as sd
import numpy as np
import pyttsx3 # Ensure this is installed as a fallback
import time
import sys
import re
import os

logger = logging.getLogger("utils.tts") # Use conventional logger name
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s - %(name)s - INFO - %(message)s"))
    logger.addHandler(ch)
logger.setLevel(logging.INFO)
logger.propagate = False

# Keep module import quiet; use logger for diagnostics.


class TextToSpeech:
    """
    Text-to-speech engine using Piper for natural voice synthesis
    Provides voice confirmations and status updates
    """

    def __init__(self, config: dict):
        self.config = config
        self.enabled = config.get('tts', {}).get('enabled', True)

        tts_cfg = config.get('tts', {}) if isinstance(config, dict) else {}
        # Default speaking rate (words per minute). Keep this conservative; clarity > speed.
        self.fast_rate = int(tts_cfg.get('rate_wpm', 170))
        self.volume = float(tts_cfg.get('volume', 0.9))
        self.short_mode = bool(tts_cfg.get('short_mode', True))
        # Avoid delaying flows like login: status prompts should not block.
        self.default_blocking = bool(tts_cfg.get('default_blocking', False))

        # When short_mode is enabled, speak in smaller chunks rather than speeding up.
        # Allow optional overrides, but keep safe defaults.
        self._max_chunk_chars = int(tts_cfg.get('max_chunk_chars', 110 if self.short_mode else 240))
        self._chunk_pause_seconds = float(tts_cfg.get('chunk_pause_seconds', 0.12 if self.short_mode else 0.0))
        
        logger.info("TTS init (enabled=%s, short_mode=%s, rate_wpm=%s)", self.enabled, self.short_mode, self.fast_rate)

        # Piper TTS settings
        self.voice_preference = self._resolve_voice_preference(tts_cfg)

        # Piper model selection: voice (gender) is defined by the model.
        self.model_path = self._select_piper_model_path(tts_cfg)
        logger.info("TTS voice preference: %s", self.voice_preference)
        logger.info("TTS model path: %s", self.model_path)
        self.voice = None

        # Fallback TTS engine (pyttsx3)
        self._fallback_engine = None
        self._engine_lock = threading.Lock() # Lock for pyttsx3
        self._engine_ready = False

        if self.enabled:
            self._initialize_voice()
            # Initialize fallback immediately for reliability and speed
            self._init_fallback_engine()
        else:
            logger.warning("⚠️ TTS is DISABLED in configuration!")

    def _resolve_voice_preference(self, tts_cfg: dict) -> str:
        """Resolve preferred voice from config and environment.

        Supported values: default, female, male
        Env override: SECUREX_TTS_VOICE
        """
        try:
            cfg_value = str(tts_cfg.get('voice', tts_cfg.get('voice_preference', 'default'))).strip().lower()
        except Exception:
            cfg_value = 'default'

        env_value = os.getenv('SECUREX_TTS_VOICE', '').strip().lower()
        value = env_value or cfg_value or 'default'

        if value not in {'default', 'female', 'male'}:
            logger.warning("Unknown tts.voice=%r; using 'default'", value)
            return 'default'
        return value

    def _select_piper_model_path(self, tts_cfg: dict) -> str:
        """Pick the Piper model path based on preferred voice when provided.

        Config keys:
          - tts.model_path (default)
          - tts.model_path_female (optional)
          - tts.model_path_male (optional)

        Note: If the preferred model path is not set or doesn't exist, falls back to model_path.
        """
        default_path = tts_cfg.get('model_path', 'en_US-lessac-medium.onnx')
        preferred_path = None

        if self.voice_preference == 'female':
            preferred_path = tts_cfg.get('model_path_female')
        elif self.voice_preference == 'male':
            preferred_path = tts_cfg.get('model_path_male')

        # Allow env override for Piper model path if desired.
        env_model = os.getenv('SECUREX_TTS_MODEL_PATH', '').strip()
        if env_model:
            preferred_path = env_model

        candidate = preferred_path or default_path
        try:
            if candidate and os.path.exists(candidate):
                return candidate
            if preferred_path and preferred_path != default_path:
                logger.warning("Preferred Piper model not found (%s). Falling back to %s", preferred_path, default_path)
        except Exception:
            # If path checks fail for any reason, just use the configured value.
            pass
        return default_path

    def _normalize_text(self, text: str) -> str:
        """Normalize whitespace and strip control characters."""
        if not text:
            return ""
        # Collapse whitespace, keep punctuation.
        normalized = re.sub(r"\s+", " ", text).strip()
        # Remove stray non-printing characters that can confuse some TTS engines.
        normalized = "".join(ch for ch in normalized if ch.isprintable())
        return normalized

    def _chunk_text(self, text: str) -> list[str]:
        """Split text into shorter, speakable chunks (sentences/clauses)."""
        text = self._normalize_text(text)
        if not text:
            return []

        # First split by sentence boundaries.
        # Keep punctuation at the end of each chunk.
        sentences = re.split(r"(?<=[\.!\?])\s+", text)
        chunks: list[str] = []

        for sentence in (s.strip() for s in sentences if s and s.strip()):
            if len(sentence) <= self._max_chunk_chars:
                chunks.append(sentence)
                continue

            # If a sentence is still too long, split by clause separators.
            clause_parts = re.split(r"(?<=[,;:])\s+", sentence)
            for part in (p.strip() for p in clause_parts if p and p.strip()):
                if len(part) <= self._max_chunk_chars:
                    chunks.append(part)
                else:
                    # Final fallback: hard-wrap by words.
                    words = part.split(" ")
                    current = ""
                    for w in words:
                        if not current:
                            current = w
                        elif len(current) + 1 + len(w) <= self._max_chunk_chars:
                            current = f"{current} {w}"
                        else:
                            chunks.append(current)
                            current = w
                    if current:
                        chunks.append(current)

        return chunks

    def _estimate_speech_timeout_seconds(self, text: str) -> float:
        """Heuristic join timeout for blocking fallback speech."""
        try:
            words = len(re.findall(r"\w+", text or ""))
            wpm = max(80, int(self.fast_rate or 170))
            est_seconds = (words / wpm) * 60.0
            # Add overhead for engine init + pauses.
            return max(10.0, min(90.0, est_seconds + 8.0))
        except Exception:
            return 10.0

    def _piper_synthesize_to_array(self, text: str) -> tuple[np.ndarray, int]:
        """Synthesize text to a float32 numpy array via Piper.

        Piper's Python API yields `AudioChunk` objects (streaming). Different
        versions expose audio as float arrays and/or int16 bytes.

        Returns:
            (audio_float32_array, sample_rate)
        """
        float_arrays: list[np.ndarray] = []
        int16_bytes_parts: list[bytes] = []
        sample_rate: Optional[int] = None

        for chunk in self.voice.synthesize(text):
            # Newer piper-tts exposes these on AudioChunk.
            if hasattr(chunk, 'sample_rate') and sample_rate is None:
                try:
                    sample_rate = int(chunk.sample_rate)
                except Exception:
                    sample_rate = None

            if hasattr(chunk, 'audio_float_array'):
                arr = getattr(chunk, 'audio_float_array', None)
                if isinstance(arr, np.ndarray) and arr.size:
                    # Ensure float32 for sounddevice.
                    float_arrays.append(arr.astype(np.float32, copy=False))
                    continue

            if hasattr(chunk, 'audio_int16_bytes'):
                b = getattr(chunk, 'audio_int16_bytes', None)
                if isinstance(b, (bytes, bytearray)) and len(b):
                    int16_bytes_parts.append(bytes(b))
                    continue

            # Backwards/alternate shapes (best-effort).
            if isinstance(chunk, (bytes, bytearray)):
                int16_bytes_parts.append(bytes(chunk))

        if float_arrays:
            audio = np.concatenate(float_arrays)
            if audio.size == 0:
                raise ValueError("No audio data generated")
            return audio.astype(np.float32, copy=False), int(sample_rate or getattr(self.voice.config, 'sample_rate', 22050))

        audio_data = b''.join(int16_bytes_parts)
        if not audio_data:
            raise ValueError("No audio data generated")

        audio = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        return audio, int(sample_rate or getattr(self.voice.config, 'sample_rate', 22050))

    def _speak_with_piper(self, chunks: list[str], blocking: bool) -> None:
        """Speak one or more chunks using Piper, sequentially."""
        if not chunks:
            return

        # Non-blocking multi-chunk playback must run in a thread so
        # subsequent sd.play calls don't stomp each other.
        def play_sequence(wait_each: bool):
            for idx, chunk in enumerate(chunks):
                audio_array, sample_rate = self._piper_synthesize_to_array(chunk)
                sd.play(audio_array, samplerate=sample_rate)
                if wait_each:
                    sd.wait()
                if self._chunk_pause_seconds and idx < len(chunks) - 1:
                    time.sleep(self._chunk_pause_seconds)

        if blocking:
            play_sequence(wait_each=True)
        else:
            thread = threading.Thread(target=lambda: play_sequence(wait_each=True), daemon=True)
            thread.start()

    def _initialize_voice(self):
        """Initialize Piper voice"""
        try:
            # Try to load the model
            from piper import PiperVoice
            if os.path.exists(self.model_path):
                self.voice = PiperVoice.load(self.model_path)
                print(f"[TTS] ✅ Piper TTS initialized with model: {self.model_path}")
                logger.info(f"Piper TTS initialized with model: {self.model_path}")
            else:
                print(f"[TTS] ❌ Piper model not found: {self.model_path}")
                logger.error(f"Piper model file not found: {self.model_path}")
                self.voice = None
        except ImportError:
            print("[TTS] ⚠️ piper-tts library not found. Using fallback.")
            logger.warning("piper-tts library not found. pip install piper-tts")
            logger.warning("Falling back to system TTS (pyttsx3).")
            self.voice = None
        except Exception as e:
            print(f"[TTS] ❌ Failed to initialize Piper: {e}")
            logger.error(f"Failed to initialize Piper TTS model ({self.model_path}): {e}")
            logger.info("Falling back to system TTS (pyttsx3)...")
            self.voice = None
    
    def _init_fallback_engine(self):
        """Pre-initialize fallback engine for faster response"""
        if self._engine_ready:
            return
        
        try:
            with self._engine_lock:
                if sys.platform == 'win32':
                    print("[TTS] Initializing Windows SAPI engine...")
                    try:
                        self._fallback_engine = pyttsx3.init('sapi5')
                    except:
                        print("[TTS] SAPI5 failed, trying default engine...")
                        self._fallback_engine = pyttsx3.init()
                else:
                    self._fallback_engine = pyttsx3.init()
                
                # Configure engine
                self._fallback_engine.setProperty('rate', self.fast_rate)
                self._fallback_engine.setProperty('volume', self.volume)
                self._configure_fallback_voice(self._fallback_engine)
                self._engine_ready = True
                logger.info("Fallback TTS engine initialized successfully")
        except Exception as e:
            print(f"[TTS] ❌ Failed to initialize fallback: {e}")
            logger.error(f"Failed to initialize fallback engine: {e}")
            self._fallback_engine = None
            self._engine_ready = False

    def _configure_fallback_voice(self, engine) -> None:
        """Configure the fallback (pyttsx3) voice to match preference when possible."""
        if not engine:
            return
        if self.voice_preference == 'default':
            return

        try:
            voices = engine.getProperty('voices') or []
        except Exception as e:
            logger.debug("Could not query pyttsx3 voices: %s", e)
            return

        if not voices:
            return

        pref = self.voice_preference

        def voice_text(v) -> str:
            parts = []
            for attr in ('id', 'name', 'gender', 'age'):
                try:
                    val = getattr(v, attr, None)
                    if val:
                        parts.append(str(val))
                except Exception:
                    continue
            return " ".join(parts).lower()

        def score(v) -> int:
            text = voice_text(v)
            s = 0
            if pref == 'female':
                # Common Windows female voice: Microsoft Zira Desktop
                for token, weight in (('zira', 50), ('female', 40), ('woman', 15), ('girl', 10)):
                    if token in text:
                        s += weight
            elif pref == 'male':
                # Common Windows male voice: Microsoft David Desktop / Mark
                for token, weight in (('david', 50), ('mark', 30), ('male', 40), ('man', 15), ('boy', 10)):
                    if token in text:
                        s += weight
            return s

        best = max(voices, key=score)
        if score(best) <= 0:
            # No confident match; leave default.
            logger.info("No clear %s voice found for pyttsx3; keeping default", pref)
            return

        try:
            engine.setProperty('voice', best.id)
            logger.info("Selected pyttsx3 voice for %s: %s", pref, getattr(best, 'name', best.id))
        except Exception as e:
            logger.warning("Failed to set pyttsx3 voice: %s", e)

    def speak(self, text: str, blocking: Optional[bool] = None):
        """
        Speak text using the best available TTS engine.
        This function is thread-safe.

        Args:
            text: Text to speak
            blocking: If True, wait for speech to complete
        """
        if blocking is None:
            blocking = self.default_blocking
        logger.debug("TTS.speak(text=%r, blocking=%s)", text, blocking)
        
        if not self.enabled:
            print("[TTS] ⚠️ TTS is disabled")
            logger.info("DEBUG: TTS.speak early exit (disabled)")
            return
        
        if not text or not text.strip():
            print("[TTS] ⚠️ Empty text")
            logger.info("DEBUG: TTS.speak early exit (empty text)")
            return

        normalized = self._normalize_text(text)
        chunks = self._chunk_text(normalized) if self.short_mode else [normalized]

        if self.voice:
            # --- Use Piper TTS ---
            try:
                print("[TTS] Using Piper TTS...")
                # Speak sequentially (short_mode splits into smaller chunks).
                self._speak_with_piper(chunks, blocking=bool(blocking))

                logger.info("Piper spoke")
                return  # Success

            except Exception as e:
                logger.error(f"Piper TTS failed: {e}")
                self.voice = None  # Disable Piper to prevent future slow failures
                self._fallback_tts(normalized, bool(blocking))
        else:
            # --- Fallback to system TTS (pyttsx3) ---
            self._fallback_tts(normalized, bool(blocking))

    def _fallback_tts(self, text: str, blocking: bool = False):
        """Fallback TTS using Windows SAPI directly (most reliable)"""
        logger.debug("Fallback TTS (blocking=%s)", blocking)

        chunks = self._chunk_text(text) if self.short_mode else [self._normalize_text(text)]
        chunks = [c for c in chunks if c and c.strip()]
        if not chunks:
            return

        def speak_task():
            # This function runs in a separate thread
            success = False
            
            # Method 1: Try Windows SAPI direct (most reliable on Windows)
            if sys.platform == 'win32' and not success:
                try:
                    import win32com.client
                    speaker = win32com.client.Dispatch("SAPI.SpVoice")
                    try:
                        # Attempt to select a gendered voice if requested.
                        if self.voice_preference in {'female', 'male'}:
                            gender = 'Female' if self.voice_preference == 'female' else 'Male'
                            matches = speaker.GetVoices(f"Gender={gender}")
                            if matches is not None and getattr(matches, 'Count', 0) > 0:
                                speaker.Voice = matches.Item(0)
                                logger.info("Selected Windows SAPI %s voice", self.voice_preference)
                    except Exception as voice_e:
                        logger.debug("Could not select Windows SAPI voice: %s", voice_e)
                    try:
                        speaker.Rate = int((self.fast_rate - 150) / 10)  # coarse mapping
                        speaker.Volume = int(max(0, min(100, self.volume * 100)))
                    except Exception:
                        pass
                    for idx, chunk in enumerate(chunks):
                        speaker.Speak(chunk)
                        if self._chunk_pause_seconds and idx < len(chunks) - 1:
                            time.sleep(self._chunk_pause_seconds)
                    logger.info("Windows SAPI spoke")
                    success = True
                except Exception as sapi_e:
                    logger.error(f"Windows SAPI failed: {sapi_e}")
            
            # Method 2: Try pyttsx3 if SAPI failed
            if not success:
                try:
                    with self._engine_lock:
                        if self._fallback_engine is None and not self._engine_ready:
                            logger.info("Initializing pyttsx3 fallback engine...")
                            
                            if sys.platform == 'win32':
                                try:
                                    self._fallback_engine = pyttsx3.init('sapi5')
                                    print("[TTS] Using pyttsx3 with SAPI5")
                                except:
                                    self._fallback_engine = pyttsx3.init()
                            else:
                                self._fallback_engine = pyttsx3.init()
                            
                            # Optimized settings for faster speech
                            self._fallback_engine.setProperty('rate', self.fast_rate)
                            self._fallback_engine.setProperty('volume', self.volume)
                            self._engine_ready = True
                        
                        if self._fallback_engine:
                            # Ensure voice preference is applied even if engine was lazily created.
                            try:
                                self._configure_fallback_voice(self._fallback_engine)
                            except Exception:
                                pass
                            for idx, chunk in enumerate(chunks):
                                self._fallback_engine.say(chunk)
                                self._fallback_engine.runAndWait()
                                if self._chunk_pause_seconds and idx < len(chunks) - 1:
                                    time.sleep(self._chunk_pause_seconds)
                            logger.info("pyttsx3 spoke")
                            success = True
                except Exception as e:
                    logger.error(f"pyttsx3 failed: {e}")
            
            if not success:
                logger.error("All TTS methods failed")
            logger.debug("Fallback TTS thread finished")

        try:
            # Start the speak task in a new thread
            thread = threading.Thread(target=speak_task, daemon=False)
            thread.start()

            if blocking:
                # If blocking, wait for this new thread to finish
                thread.join(timeout=self._estimate_speech_timeout_seconds(text))
                logger.debug("Fallback TTS blocking done")
            else:
                # Give it a moment to start
                time.sleep(0.1)
                logger.debug("Fallback TTS started (non-blocking)")

        except Exception as e:
            print(f"[TTS] ❌ Failed to start fallback thread: {e}")
            logger.error(f"Failed to start fallback TTS thread: {e}")
            logger.info("DEBUG: Exiting _fallback_tts after exception.")

    def speak_async(self, text: str):
        """Helper function: Speak text asynchronously"""
        self.speak(text, blocking=False)

    def speak_sync(self, text: str):
        """Helper function: Speak text synchronously"""
        self.speak(text, blocking=True)

    def stop(self):
        """Stop current audio playback"""
        try:
            sd.stop()
            print("[TTS] Audio playback stopped")
            logger.info("TTS audio playback stopped")
        except Exception as e:
            logger.warning(f"Error stopping audio playback: {e}")

    def shutdown(self):
        """Shutdown TTS system"""
        logger.info("Shutting down TTS engine...")
        self.enabled = False # Stop any new requests
        
        # Clean up fallback engine
        with self._engine_lock:
            if self._fallback_engine:
                try:
                    self._fallback_engine.stop()
                except Exception as e:
                    logger.warning(f"Error stopping pyttsx3 engine: {e}")
                self._fallback_engine = None
        
        # Stop any sounddevice playback
        try:
            sd.stop()
        except Exception as e:
            logger.warning(f"Error stopping sounddevice: {e}")
            
        logger.info("TTS shutdown complete.")

    # --- Start: Standard Assistant Phrases ---
    # These are just convenient wrappers for self.speak()

    def welcome(self, username: str = "user"):
        """Welcome message"""
        if self.short_mode:
            self.speak(f"Welcome back, {username}.")
        else:
            self.speak(f"Welcome to SecureX Assist, {username}.")
    
    def authenticate_success(self):
        """Authentication success message"""
        self.speak("Authenticated.")
    
    def authenticate_failed(self):
        """Authentication failed message"""
        self.speak("Not verified.")
    
    def voice_verification_started(self):
        """Voice verification started"""
        self.speak("Checking voice.")
    
    def voice_verification_passed(self):
        """Voice verification passed"""
        self.speak("Voice verified.")
    
    def voice_verification_failed(self):
        """Voice verification failed"""
        self.speak("Voice not verified.")
    
    def liveness_challenge(self, phrase: str):
        """Announce liveness challenge phrase"""
        self.speak(f"Please say the following phrase: {phrase}")
    
    def liveness_passed(self):
        """Liveness check passed"""
        self.speak("Live voice confirmed.")
    
    def liveness_failed(self):
        """Liveness check failed"""
        self.speak("Liveness failed.")
    
    def recording_started(self):
        """Recording started"""
        self.speak("Speak now.")
    
    def recording_complete(self):
        """Recording complete"""
        self.speak("Got it.")
    
    def enrollment_started(self):
        """Enrollment started"""
        self.speak("Enrollment. Speak clearly.")
    
    def enrollment_complete(self):
        """Enrollment complete"""
        self.speak("Enrollment complete.")
    
    def error(self, message: str = "An error occurred"):
        """Error message"""
        self.speak(message)
    
    # --- End: Standard Assistant Phrases ---
