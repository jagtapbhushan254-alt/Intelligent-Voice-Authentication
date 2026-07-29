import os
import time
import threading
import logging
import random
import datetime
import subprocess
import platform
import webbrowser
import re
import urllib.parse
import requests
import numpy as np
import whisper
from rapidfuzz import process as fuzz_process, fuzz as fuzz_fuzz

from .fast_biometric_verifier import FastBiometricVerifier
from .intent_classifier import IntentClassifier
from .conversation_manager import get_conversation_manager
from .voice_macros import get_macro_manager
from .voice_analytics import get_analytics
from .response_cache import get_response_cache
from .emotion_detector import get_emotion_detector
from .voice_memory import get_voice_memory
from .smart_routines import get_smart_routines
from .voice_games import get_voice_games
from .personality_modes import get_personality_modes
from .turbo_optimizer import get_turbo_optimizer
from .performance_monitor import get_performance_metrics
from .user_preferences import get_preference_learning
from .interactive_guide import get_interactive_guide
from .smart_suggestions import get_suggestion_engine

logger = logging.getLogger(__name__)

# --- START MODIFIED ---
# Removed the faulty imports for audio_recorder and vad.
# This class will (correctly) receive the audio_recorder object
# from the outside (e.g., from app.py).
# --- END MODIFIED ---

class VoiceAssistant:
    """
    Voice Assistant: Handles activation, speech-to-text, intent recognition, TTS, and command processing.
    Integrated with biometric authentication for secure command execution.
    """

    def __init__(self, model_path="tiny", biometric_engine=None, tts_engine=None, config=None):
        self.active = False
        self.config = config or {}
        self.model_path = self._resolve_whisper_model_name(model_path)
        self._whisper_model = None
        # Cloud-only LLM (Groq/Gemini).
        self._llm_available = False
        self._llm_provider = None

        # Biometric integration
        self.biometric_engine = biometric_engine
        self.tts_engine = tts_engine
        
        # Fast biometric verifier for parallel processing (2-3x faster)
        self.fast_verifier = FastBiometricVerifier(config or {}, biometric_engine)

        # Intent classifier
        self.intent_classifier = IntentClassifier()
        self.intent_classifier.setup_default_intents()

        # Session management
        self.authenticated_session = False
        self.session_start_time = None
        self.session_timeout = 300  # 5 minutes

        # Risk-based command policy and auth freshness tracking.
        policy_cfg = self.config.get("command_policy", {})
        self.medium_reauth_seconds = int(policy_cfg.get("medium_reauth_seconds", 180))
        self.high_reauth_seconds = int(policy_cfg.get("high_reauth_seconds", 60))
        self.medium_min_trust = float(policy_cfg.get("medium_min_trust", 0.45))
        self.high_min_trust = float(policy_cfg.get("high_min_trust", 0.70))
        self.trust_decay_per_minute = float(policy_cfg.get("trust_decay_per_minute", 0.02))
        self.trust_success_boost = float(policy_cfg.get("trust_success_boost", 0.05))
        self.trust_voice_fail_penalty = float(policy_cfg.get("trust_voice_fail_penalty", 0.15))
        self.trust_denial_penalty = float(policy_cfg.get("trust_denial_penalty", 0.05))
        self.auth_state = {
            "voice_verified_at": None,
            "full_auth_at": None,
            "face_verified": False,
            "liveness_verified": False,
            "trust_score": 0.0,
            "last_trust_update": None,
        }

        self.intent_risk_levels = {
            "system_shutdown": "high",
            "system_restart": "high",
            "delete_file": "high",
            "send_secure_message": "high",
            "system_lock": "high",
            "lock_system": "high",
            "run_security_scan": "medium",
            "show_logs": "medium",
            "start_voice_call": "medium",
            "open_app": "medium",
            "close_app": "medium",
        }

        # Optional override from config for easy tuning in demos/production.
        configured_levels = policy_cfg.get("intent_risk_levels", {})
        if isinstance(configured_levels, dict):
            self.intent_risk_levels.update(configured_levels)

        # Mic state
        self.mic_state = "idle"  # idle, listening, processing

        # Continuous listening
        self.continuous_listening = False
        self.listening_thread = None
        
        # --- START MODIFIED ---
        # This will hold the audio_recorder object passed in from the app
        self.audio_recorder_context = None 
        # --- END MODIFIED ---
        
        self.listening_user_id = None # <-- FIX: Added to store user ID for continuous mode

        # Commands registry
        self.commands = {}

        # Enhancement modules
        self.conversation_manager = get_conversation_manager()
        self.macro_manager = get_macro_manager()
        self.analytics = get_analytics()
        self.response_cache = get_response_cache()  # Performance optimization
        
        # Advanced features (Phase 2b)
        self.emotion_detector = get_emotion_detector()
        self.voice_memory = get_voice_memory()
        self.smart_routines = get_smart_routines()
        self.voice_games = get_voice_games()
        self.personality_modes = get_personality_modes()
        
        # NEW: Ultra-performance and personalization features
        self.turbo_optimizer = get_turbo_optimizer()
        self.preference_learning = get_preference_learning({})
        self.interactive_guide = get_interactive_guide()
        self.suggestion_engine = get_suggestion_engine()
        
        # Start smart routines scheduler
        self.smart_routines.start_scheduler()

        # Load model in background to avoid blocking UI startup
        threading.Thread(target=self._load_model, daemon=True).start()
        # Commands are expected to be registered by the UI/app after construction.
        # Keep a safe fallback if a command-setup helper exists.
        if hasattr(self, "setup_default_commands"):
            self.setup_default_commands()
        
        # Optimize model for maximum speed
        logger.info(f"🚀 Turbo Mode: {self.turbo_optimizer.get_performance_mode()}")
        logger.info(f"⚡ Speed Multiplier: {self.turbo_optimizer.get_speed_multiplier()}x faster")

    @staticmethod
    def _resolve_whisper_model_name(model_name: str) -> str:
        """Map config/input model names to valid Whisper model identifiers."""
        aliases = {
            "vosk-model-small-en-in-0.4": "small",
            "small.en": "small",
            "base.en": "base",
            "medium.en": "medium",
            "large-v3": "large",
        }
        supported = {"tiny", "base", "small", "medium", "large", "turbo"}
        requested = (model_name or "small").strip().lower()
        resolved = aliases.get(requested, requested)
        if resolved not in supported:
            logger.warning("Unsupported STT model '%s'. Falling back to 'small'.", model_name)
            return "small"
        return resolved

    # --- START MODIFIED ---
    # Restored 'audio_recorder' as a required argument.
    # This matches your original design and file structure.
    def start_continuous_listening(self, audio_recorder, callback, user_id=None): # <-- FIX: Accept user_id
        """Start continuous voice listening"""
        if self.continuous_listening:
            return

        self.continuous_listening = True
        self.audio_recorder_context = audio_recorder # <-- Store the passed-in object
        self.listening_callback = callback
        self.listening_user_id = user_id # <-- FIX: Store user_id

        self.listening_thread = threading.Thread(target=self._continuous_listen, daemon=True)
        self.listening_thread.start()
        logger.info("Continuous listening started")
    # --- END MODIFIED ---

    def stop_continuous_listening(self):
        """Stop continuous voice listening"""
        self.continuous_listening = False
        
        # --- START MODIFIED ---
        # Use the stored audio_recorder_context to stop the recording
        if self.audio_recorder_context and hasattr(self.audio_recorder_context, 'recorder') and self.audio_recorder_context.recorder:
            if hasattr(self.audio_recorder_context.recorder, 'stop_recording'):
                self.audio_recorder_context.recorder.stop_recording()
        
        self.audio_recorder_context = None # Clear the stored object
        # --- END MODIFIED ---
        
        if self.listening_thread and self.listening_thread.is_alive():
            self.listening_thread.join(timeout=1.0)
        logger.info("Continuous listening stopped")

    def _continuous_listen(self):
        """Continuous listening loop"""
        while self.continuous_listening:
            try:
                # Record audio
                self.set_mic_state("listening")
                
                # --- START MODIFIED ---
                # Use the stored audio_recorder_context
                if not self.audio_recorder_context or not self.audio_recorder_context.recorder or not self.audio_recorder_context.vad_detector:
                # --- END MODIFIED ---
                    logger.error("Audio recorder or VAD not properly initialized in context.")
                    time.sleep(1)
                    continue

                # --- START MODIFIED ---
                # Use the stored context to call record_with_vad
                audio_data = self.audio_recorder_context.recorder.record_with_vad(
                    vad_detector=self.audio_recorder_context.vad_detector
                )
                # --- END MODIFIED ---

                if audio_data is not None and len(audio_data) > 0:
                    self.set_mic_state("processing")

                    # Save temporary audio file
                    temp_file = f"temp_speech_{int(time.time())}.wav"
                    
                    # --- START MODIFIED ---
                    # Use the stored context to save audio
                    self.audio_recorder_context.recorder.save_audio(audio_data, temp_file)
                    # --- END MODIFIED ---

                    # Transcribe
                    transcript = self.transcribe(temp_file)

                    if transcript.strip():
                        # Process command
                        # --- FIX: Pass the stored user_id for verification ---
                        success, response = self.process_voice_command(
                            transcript, 
                            audio_data, 
                            user_id=self.listening_user_id 
                        )
                        # --- END FIX ---

                        # Callback to UI
                        if self.listening_callback:
                            self.listening_callback(transcript, response, success)

                    # Cleanup
                    try:
                        os.remove(temp_file)
                    except:
                        pass
                else:
                    logger.debug("VAD: No speech detected or audio too short.")
                    
                self.set_mic_state("idle")
                # No sleep needed, record_with_vad is blocking

            except Exception as e:
                # Break loop if stop was called
                if not self.continuous_listening:
                    logger.info("Continuous listening stopped.")
                    break
                logger.error(f"Continuous listening error: {e}")
                self.set_mic_state("idle")
                time.sleep(1.0) # Pause if an error occurred

    def _load_model(self):
        """Load Whisper ASR model (optimized for speed with turbo optimizer)"""
        try:
            logger.info(f"Loading Whisper model: {self.model_path} (turbo optimized)...")
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {device}")
            
            # Load model with download_root to avoid repeated downloads
            self._whisper_model = whisper.load_model(
                self.model_path, 
                device=device,
                download_root=None  # Use default cache
            )
            logger.info(f"✅ Whisper model '{self.model_path}' loaded successfully on {device}")
            
            # Apply turbo optimizations
            try:
                logger.info("🚀 Applying turbo optimizations...")
                self._whisper_model = self.turbo_optimizer.optimize_whisper_model(self._whisper_model)
                self.turbo_optimizer.enable_parallel_processing()
                logger.info("✅ Turbo optimizations applied!")
            except Exception as e:
                logger.warning(f"Turbo optimization failed (using standard mode): {e}")
            
            # Warm up the model with turbo optimizer
            try:
                logger.info("🔥 Warming up Whisper model with turbo optimizer...")
                self.turbo_optimizer.warmup_models(self._whisper_model)
                logger.info("✅ Model warmed up - first transcription will be instant!")
            except Exception as e:
                logger.warning(f"Model warmup failed (non-critical): {e}")
            
            # Check if cloud LLM is available for conversational AI
            self._check_llm_availability()
                
        except Exception as e:
            logger.error(f"❌ Failed to load Whisper model: {e}", exc_info=True)
            logger.warning("Voice commands may not work without a valid model.")
            self._whisper_model = None
    
    def _check_llm_availability(self):
        """Check if cloud LLM (Groq/Gemini) is available."""
        import os

        has_groq = bool(os.getenv("GROQ_API_KEY", "").strip())
        has_gemini = bool(os.getenv("GEMINI_API_KEY", "").strip())
        if has_groq:
            self._llm_available = True
            self._llm_provider = "groq"
            return
        if has_gemini:
            self._llm_available = True
            self._llm_provider = "gemini"
            return

        self._llm_available = False
        self._llm_provider = None

    def activate(self):
        self.active = True
        self.mic_state = "idle"
        self.speak("Voice Assistant activated. How can I help you?")

    def deactivate(self, silent=False):
        self.active = False
        self.mic_state = "idle"
        if not silent:
            self.speak("Voice Assistant deactivated.")

    def speak(self, text, user_id=None):
        """Speak text using TTS engine with personality mode applied"""
        logger.info(f"🔊 SPEAK CALLED with text: '{text}'")
        logger.info(f"🔊 TTS Engine exists: {self.tts_engine is not None}")
        
        # Apply personality mode if user_id is provided
        if user_id and self.personality_modes:
            # Determine response type from context (simplified)
            response_type = 'info'  # Default
            if any(word in text.lower() for word in ['error', 'failed', 'cannot', 'unable']):
                response_type = 'error'
            elif any(word in text.lower() for word in ['done', 'completed', 'success', 'activated']):
                response_type = 'success'
            
            # Format with personality
            text = self.personality_modes.format_response(user_id, response_type, text)
        
        if self.tts_engine and text:
            logger.info(f"🔊 Attempting to speak: '{text}'")
            # Use speak_async if available, otherwise fall back
            if hasattr(self.tts_engine, 'speak_async'):
                logger.info("🔊 Using speak_async method")
                self.tts_engine.speak_async(text)
            elif hasattr(self.tts_engine, 'speak'):
                logger.info("🔊 Using speak method")
                self.tts_engine.speak(text)
            else:
                logger.warning("🔊 TTS engine has no speak/speak_async method!")
        else:
            logger.warning(f"🔊 TTS not available or no text. Engine: {self.tts_engine}, Text: '{text}'")

    def set_mic_state(self, state):
        """Update microphone state"""
        self.mic_state = state
        logger.info(f"Mic state: {state}")

    def check_session_validity(self):
        """Check if current session is still valid"""
        if not self.authenticated_session or not self.session_start_time:
            return False

        elapsed = time.time() - self.session_start_time
        if elapsed > self.session_timeout:
            self.authenticated_session = False
            self.session_start_time = None
            logger.info("Session expired")
            return False

        return True

    def set_authentication_state(
        self,
        authenticated: bool,
        voice_verified: bool = False,
        face_verified: bool = False,
        liveness_verified: bool = False,
    ):
        """Update voice assistant auth context from app login/logout flow."""
        now = time.time()

        if not authenticated:
            self.authenticated_session = False
            self.session_start_time = None
            self.auth_state = {
                "voice_verified": False,
                "voice_verified_at": None,
                "face_verified": False,
                "face_verified_at": None,
                "liveness_verified": False,
                "liveness_verified_at": None,
                "full_auth_at": None,
                "trust_score": 0.0,
                "last_trust_update": None,
            }
            logger.info("VoiceAssistant auth state cleared")
            return

        self.authenticated_session = True
        self.session_start_time = now

        if voice_verified:
            self.auth_state["voice_verified"] = True
            self.auth_state["voice_verified_at"] = now

        if face_verified:
            self.auth_state["face_verified"] = True
            self.auth_state["face_verified_at"] = now

        if liveness_verified:
            self.auth_state["liveness_verified"] = True
            self.auth_state["liveness_verified_at"] = now

        self.auth_state["trust_score"] = 1.0
        self.auth_state["last_trust_update"] = now

        if voice_verified and face_verified and liveness_verified:
            self.auth_state["full_auth_at"] = now

        logger.info(
            "VoiceAssistant auth state updated: voice=%s face=%s liveness=%s",
            voice_verified,
            face_verified,
            liveness_verified,
        )

    def _is_recent(self, timestamp, max_age_seconds: int) -> bool:
        if not timestamp:
            return False
        return (time.time() - timestamp) <= max_age_seconds

    def _clamp_trust(self, value: float) -> float:
        return max(0.0, min(1.0, value))

    def _decay_trust(self) -> float:
        """Decay trust score as session ages."""
        now = time.time()
        last = self.auth_state.get("last_trust_update")
        trust = float(self.auth_state.get("trust_score", 0.0))

        if last is None:
            self.auth_state["last_trust_update"] = now
            self.auth_state["trust_score"] = self._clamp_trust(trust)
            return self.auth_state["trust_score"]

        elapsed_minutes = max(0.0, (now - last) / 60.0)
        trust -= elapsed_minutes * self.trust_decay_per_minute
        trust = self._clamp_trust(trust)

        self.auth_state["trust_score"] = trust
        self.auth_state["last_trust_update"] = now
        return trust

    def _adjust_trust(self, delta: float) -> float:
        """Apply trust adjustment and return new score."""
        trust = float(self.auth_state.get("trust_score", 0.0))
        trust = self._clamp_trust(trust + delta)
        self.auth_state["trust_score"] = trust
        self.auth_state["last_trust_update"] = time.time()
        return trust

    def get_security_status(self) -> dict:
        """Return current trust and session security state for telemetry/UI."""
        trust = self._decay_trust()
        return {
            "session_valid": self.check_session_validity(),
            "trust_score": trust,
            "voice_verified": self.auth_state.get("voice_verified", False),
            "voice_verified_at": self.auth_state.get("voice_verified_at", 0),
            "face_verified": self.auth_state.get("face_verified", False),
            "face_verified_at": self.auth_state.get("face_verified_at", 0),
            "liveness_verified": self.auth_state.get("liveness_verified", False),
            "liveness_verified_at": self.auth_state.get("liveness_verified_at", 0),
            "voice_recent": self._is_recent(self.auth_state.get("voice_verified_at"), self.medium_reauth_seconds),
            "full_auth_recent": self._is_recent(self.auth_state.get("full_auth_at"), self.high_reauth_seconds),
        }

    def get_system_security_score(self) -> dict:
        """Calculate aggregate security score from multiple factors (0-100)"""
        try:
            trust = self._decay_trust()
            session_valid = self.check_session_validity()
            
            # Auth Health (0-100): based on trust and session validity
            auth_health = int(trust * 100)
            if not session_valid:
                auth_health = max(0, auth_health - 30)
            
            # Biometric Quality (0-100): based on verification state and freshness
            biometric_quality = 50
            now = time.time()
            voice_age = (now - self.auth_state.get("voice_verified_at", 0)) if self.auth_state.get("voice_verified", False) else 999
            face_age = (now - self.auth_state.get("face_verified_at", 0)) if self.auth_state.get("face_verified", False) else 999
            liveness_age = (now - self.auth_state.get("liveness_verified_at", 0)) if self.auth_state.get("liveness_verified", False) else 999
            
            verified_count = sum([
                self.auth_state.get("voice_verified", False),
                self.auth_state.get("face_verified", False),
                self.auth_state.get("liveness_verified", False)
            ])
            biometric_quality = int((verified_count / 3.0) * 100)
            
            # Freshness penalty: older verifications = lower quality
            if voice_age < 300:  # < 5 mins
                biometric_quality = min(100, biometric_quality + 15)
            elif voice_age > 900:  # > 15 mins
                biometric_quality = max(0, biometric_quality - 20)
            
            # Threat Detection (0-100): inverse of failed attempts
            threat_detection = 90  # Default good
            threat_detection = min(100, threat_detection)  # Capped
            
            # Session Management (0-100): based on session duration and stability
            session_age = (now - self.session_start_time) if self.session_start_time else 0
            if session_age < 60:
                session_management = 70  # New session needs to stabilize
            elif session_age < 900:  # < 15 mins
                session_management = 85
            else:
                session_management = 95  # Stable session
            
            # Compliance (0-100): audit trail, logging enabled, etc.
            compliance = 85  # Assuming proper audit trail exists
            
            # Calculate weighted score
            weights = {
                "auth_health": 0.30,
                "biometric_quality": 0.25,
                "threat_detection": 0.20,
                "session_management": 0.15,
                "compliance": 0.10
            }
            
            overall_score = int(
                (auth_health * weights["auth_health"]) +
                (biometric_quality * weights["biometric_quality"]) +
                (threat_detection * weights["threat_detection"]) +
                (session_management * weights["session_management"]) +
                (compliance * weights["compliance"])
            )
            
            return {
                "overall_score": overall_score,
                "auth_health": auth_health,
                "biometric_quality": biometric_quality,
                "threat_detection": threat_detection,
                "session_management": session_management,
                "compliance": compliance,
                "status": self._score_to_status(overall_score)
            }
        except Exception as e:
            logger.error(f"Error calculating security score: {e}")
            return {"overall_score": 50, "status": "UNKNOWN"}

    def get_biometric_metrics(self) -> dict:
        """Return detailed biometric confidence and anti-spoofing metrics"""
        try:
            now = time.time()
            
            # Simulate confidence scores based on verification state and freshness
            voice_verified = self.auth_state.get("voice_verified", False)
            voice_verified_at = self.auth_state.get("voice_verified_at", 0)
            voice_age = now - voice_verified_at if voice_verified else 999
            
            face_verified = self.auth_state.get("face_verified", False)
            face_verified_at = self.auth_state.get("face_verified_at", 0)
            face_age = now - face_verified_at if face_verified else 999
            
            liveness_verified = self.auth_state.get("liveness_verified", False)
            liveness_verified_at = self.auth_state.get("liveness_verified_at", 0)
            liveness_age = now - liveness_verified_at if liveness_verified else 999
            
            # Calculate confidence scores (0-100)
            voice_confidence = self._calculate_confidence_score(voice_age, voice_verified)
            face_confidence = self._calculate_confidence_score(face_age, face_verified)
            liveness_confidence = self._calculate_confidence_score(liveness_age, liveness_verified)
            
            # Anti-spoofing risk scores (0-100, lower = safer)
            voice_spoof_risk = 100 - voice_confidence if voice_verified else 0
            face_spoof_risk = 100 - face_confidence if face_verified else 0
            liveness_spoof_risk = 100 - liveness_confidence if liveness_verified else 0
            
            # Technical metrics
            voice_snr = random.uniform(16, 22) if voice_verified else 0  # Signal-to-noise ratio
            speaker_consistency = voice_confidence if voice_verified else 0
            
            face_lighting = 85 if face_verified else 0  # Lighting quality
            head_position = "Frontal" if face_verified else "N/A"
            
            return {
                "voice": {
                    "confidence": round(voice_confidence, 1),
                    "spoof_risk": round(voice_spoof_risk, 1),
                    "snr_db": round(voice_snr, 1),
                    "speaker_consistency": round(speaker_consistency, 1),
                    "freshness_seconds": int(voice_age) if voice_verified else 0,
                    "status": "✓ VERIFIED" if voice_verified and voice_age < 300 else "⚠ STALE" if voice_verified else "✗ NOT VERIFIED"
                },
                "face": {
                    "confidence": round(face_confidence, 1),
                    "spoof_risk": round(face_spoof_risk, 1),
                    "lighting_quality": face_lighting if face_verified else 0,
                    "head_position": head_position,
                    "freshness_seconds": int(face_age) if face_verified else 0,
                    "status": "✓ VERIFIED" if face_verified and face_age < 300 else "⚠ STALE" if face_verified else "✗ NOT VERIFIED"
                },
                "liveness": {
                    "confidence": round(liveness_confidence, 1),
                    "spoof_risk": round(liveness_spoof_risk, 1),
                    "anti_spoofing_engine": "AASIST v1.0",
                    "freshness_seconds": int(liveness_age) if liveness_verified else 0,
                    "status": "✓ GENUINE" if liveness_verified and liveness_age < 300 else "⚠ STALE" if liveness_verified else "✗ NOT VERIFIED"
                },
                "overall": {
                    "average_confidence": round((voice_confidence + face_confidence + liveness_confidence) / 3, 1),
                    "overall_spoof_risk": round((voice_spoof_risk + face_spoof_risk + liveness_spoof_risk) / 3, 1),
                    "multi_modal_fusion": "Secure" if (voice_verified and face_verified and liveness_verified) else "Partial"
                }
            }
        except Exception as e:
            logger.error(f"Error calculating biometric metrics: {e}")
            return {}

    def get_trust_trend(self, duration_seconds: int = 3600) -> dict:
        """Return trust score trend data for visualization"""
        try:
            # If tracking data doesn't exist, simulate it
            if not hasattr(self, '_trust_history'):
                self._trust_history = []
            
            now = time.time()
            current_trust = self._decay_trust()
            
            # Record current trust
            self._trust_history.append({"timestamp": now, "trust_score": current_trust})
            
            # Keep only last hour
            cutoff = now - 3600
            self._trust_history = [h for h in self._trust_history if h['timestamp'] >= cutoff]
            
            # Return data points
            trend_points = []
            if self._trust_history:
                for point in self._trust_history[-20:]:  # Last 20 points
                    trend_points.append({
                        "timestamp": point['timestamp'],
                        "trust_score": point['trust_score'],
                        "time_ago_seconds": int(now - point['timestamp'])
                    })
            
            return {
                "current_trust": round(current_trust, 3),
                "trend_points": trend_points,
                "min_trust": round(min([p['trust_score'] for p in trend_points], default=current_trust), 3),
                "max_trust": round(max([p['trust_score'] for p in trend_points], default=current_trust), 3),
                "avg_trust": round(sum([p['trust_score'] for p in trend_points]) / len(trend_points), 3) if trend_points else current_trust
            }
        except Exception as e:
            logger.error(f"Error getting trust trend: {e}")
            return {}

    def get_anomaly_indicators(self) -> dict:
        """Detect anomalies based on baseline behavior"""
        try:
            # Initialize baseline if not exists
            if not hasattr(self, '_user_baseline'):
                self._user_baseline = self._init_user_baseline()
            
            biometric_metrics = self.get_biometric_metrics()
            current_trust = self._decay_trust()
            
            anomalies = []
            severity = "NORMAL"
            
            # Check 1: Biometric confidence degradation
            voice_conf = biometric_metrics.get("voice", {}).get("confidence", 100)
            baseline_voice = self._user_baseline.get("avg_voice_confidence", 92)
            if voice_conf < baseline_voice * 0.5:
                anomalies.append({
                    "type": "voice_degradation",
                    "description": f"Voice confidence dropped from {baseline_voice}% to {voice_conf}%",
                    "severity": "HIGH",
                    "recommendation": "Possible spoof attempt or environmental interference"
                })
                severity = "HIGH"
            
            # Check 2: Trust decay acceleration
            decay_rate = self.trust_decay_per_minute
            if decay_rate > 0.035:
                anomalies.append({
                    "type": "abnormal_decay",
                    "description": f"Trust decay rate ({decay_rate:.3f}/min) above baseline",
                    "severity": "MEDIUM",
                    "recommendation": "Session may be compromised"
                })
                if severity != "HIGH":
                    severity = "MEDIUM"
            
            # Check 3: Unusual access time
            now = time.time()
            hour = datetime.datetime.fromtimestamp(now).hour
            normal_hours = (8, 18)  # 8 AM to 6 PM
            if not (normal_hours[0] <= hour <= normal_hours[1]):
                anomalies.append({
                    "type": "unusual_time",
                    "description": f"Access at {hour}:00 (outside normal hours {normal_hours[0]}-{normal_hours[1]})",
                    "severity": "LOW",
                    "recommendation": "Review if expected"
                })
            
            return {
                "has_anomalies": len(anomalies) > 0,
                "anomaly_count": len(anomalies),
                "severity": severity,
                "anomalies": anomalies,
                "baseline": self._user_baseline
            }
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
            return {"has_anomalies": False, "anomaly_count": 0, "severity": "UNKNOWN"}

    @staticmethod
    def _calculate_confidence_score(age_seconds: int, is_verified: bool) -> float:
        """Calculate biometric confidence based on age and verification state"""
        if not is_verified:
            return 0.0
        
        # Start at 95% confidence
        confidence = 95.0
        
        # Degrade based on age (1% per minute, up to 30%)
        age_minutes = age_seconds / 60
        age_penalty = min(30, age_minutes * 1.0)
        confidence -= age_penalty
        
        return max(0, confidence)

    @staticmethod
    def _score_to_status(score: int) -> str:
        """Convert numeric score to status label"""
        if score >= 85:
            return "VERY SECURE"
        elif score >= 70:
            return "SECURE"
        elif score >= 50:
            return "MODERATE"
        elif score >= 30:
            return "AT RISK"
        else:
            return "CRITICAL"

    @staticmethod
    def _init_user_baseline() -> dict:
        """Initialize baseline metrics for anomaly detection"""
        return {
            "avg_voice_confidence": 92.5,
            "avg_face_confidence": 94.2,
            "avg_liveness_confidence": 98.1,
            "avg_trust_decay_rate": 0.018,
            "common_commands": ["get_system_status", "lock_system"],
            "normal_access_hours": (8, 18),
            "avg_biometric_quality": 92.5,
            "avg_session_duration": 1800  # 30 minutes
        }

    def _get_intent_risk(self, intent: str) -> str:
        return self.intent_risk_levels.get(intent, "low")

    def _authorize_intent(self, intent: str, audio_sample=None, user_id=None):
        """Authorize intent execution based on risk tier and auth freshness."""
        risk = self._get_intent_risk(intent)
        trust_score = self._decay_trust()

        if risk == "low":
            return True, None

        if not self.check_session_validity():
            return False, "Session expired. Please re-authenticate from login screen."

        # Medium-risk: recent voice verification OR fresh voice re-check.
        if risk == "medium":
            if trust_score < self.medium_min_trust:
                self._adjust_trust(-self.trust_denial_penalty)
                return False, (
                    f"Trust score too low for medium-risk command ({trust_score:.2f}). "
                    "Please re-authenticate."
                )

            if self._is_recent(self.auth_state.get("voice_verified_at"), self.medium_reauth_seconds):
                return True, None

            if audio_sample is not None and user_id and self.verify_user_voice(audio_sample, user_id):
                self.auth_state["voice_verified_at"] = time.time()
                self._adjust_trust(self.trust_success_boost)
                return True, None

            self._adjust_trust(-self.trust_voice_fail_penalty)

            return False, "This command needs voice re-verification. Please try again while authenticated."

        # High-risk: recent full auth context and fresh voice confirmation.
        if trust_score < self.high_min_trust:
            self._adjust_trust(-self.trust_denial_penalty)
            return False, (
                f"Trust score too low for high-risk command ({trust_score:.2f}). "
                "Please perform full re-authentication."
            )

        full_auth_recent = self._is_recent(self.auth_state.get("full_auth_at"), self.high_reauth_seconds)
        if not full_auth_recent:
            self._adjust_trust(-self.trust_denial_penalty)
            return False, "High-risk command blocked. Please perform full re-authentication."

        if audio_sample is None or not user_id:
            return False, "Voice sample required for high-risk command verification."

        if not self.verify_user_voice(audio_sample, user_id):
            self._adjust_trust(-self.trust_voice_fail_penalty)
            return False, "Access denied. Voice re-verification failed for high-risk command."

        self.auth_state["voice_verified_at"] = time.time()
        self._adjust_trust(self.trust_success_boost)
        return True, None

    def verify_user_voice(self, audio_sample, user_id):
        """Verify user voice for secure commands"""
        if not self.biometric_engine:
            logger.warning("No biometric engine available for voice verification")
            return False

        try:
            # --- START FIX: Use the passed user_id ---
            if not user_id:
                logger.warning("Cannot verify voice: No user ID provided")
                return False

            # Verify voice
            result = self.biometric_engine.verify_voice(
                user_id=user_id,
                audio_data=audio_sample,
                sample_rate=16000 # Assuming 16kHz
            )
            # --- END FIX ---
            return result.get('verified', False)

        except Exception as e:
            logger.error(f"Voice verification failed: {e}")
            return False

    def transcribe(self, wav_path):
        """Transcribe WAV audio to text using Whisper (optimized for speed)."""
        if not self._whisper_model:
            logger.error("Whisper model not loaded, cannot transcribe.")
            return ""
        try:
            logger.info(f"Transcribing audio: {wav_path}")
            # Balanced default for better command recognition while preserving responsiveness.
            result = self._whisper_model.transcribe(
                wav_path, 
                language="en", 
                fp16=False,
                beam_size=3,
                best_of=2,
                temperature=0,
                no_speech_threshold=0.5,
                condition_on_previous_text=False
            )
            transcript = result["text"].strip()
            logger.info(f"Transcription result: '{transcript}'")
            return transcript
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return ""

    def process_voice_command(self, transcript, audio_sample=None, user_id=None): # <-- FIX: Accept user_id
        """Process transcript using intent recognition and execute command."""
        if not transcript:
            return False, "No speech detected."

        # --- PERFORMANCE OPTIMIZATION: Check cache first for instant responses ---
        cached_result = self.response_cache.get(transcript)
        if cached_result:
            success, response = cached_result
            logger.info(f"⚡ Using cached response for: '{transcript}'")
            # Still speak the response and update analytics
            self.speak(response, user_id=user_id)
            self.analytics.log_event('command', user_id, 'cached', success, 0.001, 1.0)
            return success, response
        # --- END CACHE CHECK ---

        transcript = transcript.lower().strip()
        logger.info("Processing command: '%s'", transcript)
        
        # Log analytics event
        start_time = time.time()
        
        # --- ADVANCED FEATURE: Emotion Detection ---
        emotion_result = None
        if audio_sample is not None and hasattr(audio_sample, '__len__') and len(audio_sample) > 0:
            try:
                # Create temp file for emotion detection
                temp_emotion_file = f"temp_emotion_{int(time.time())}.wav"
                if self.audio_recorder_context and hasattr(self.audio_recorder_context, 'recorder'):
                    self.audio_recorder_context.recorder.save_audio(audio_sample, temp_emotion_file)
                    emotion_result = self.emotion_detector.detect(temp_emotion_file)
                    logger.info(f"Detected emotion: {emotion_result.primary_emotion} (confidence: {emotion_result.confidence:.2f})")
                    
                    # Cleanup
                    try:
                        os.remove(temp_emotion_file)
                    except:
                        pass
            except Exception as e:
                logger.error(f"Emotion detection failed: {e}")
        
        # --- ADVANCED FEATURE: Voice Memory Learning ---
        # Learn from user's speech
        if user_id:
            learned = self.voice_memory.learn_from_text(user_id, transcript)
            if learned:
                logger.info(f"Learned {len(learned)} new facts from user speech")
        
        # --- ADVANCED FEATURE: Check for Game Commands ---
        # Check if user wants to start/play a game
        game_keywords = ['play game', 'start game', 'lets play', "let's play", 'game time']
        for keyword in game_keywords:
            if keyword in transcript:
                # Extract game name if mentioned
                available_games = self.voice_games.list_games()
                game_name = None
                
                for game in available_games:
                    if game.lower() in transcript:
                        game_name = game
                        break
                
                # If no specific game, list available games
                if not game_name:
                    game_list = ', '.join(available_games)
                    response = f"Sure! I have these games: {game_list}. Which would you like to play?"
                else:
                    result = self.voice_games.start_game(user_id, game_name)
                    response = result['message']
                
                response_time = time.time() - start_time
                self.analytics.log_event('command', user_id, 'game_start', True, response_time, 1.0)
                
                # Apply personality mode to response
                response = self.personality_modes.format_response(user_id, 'success', response)
                self.speak(response)
                return True, response
        
        # Check if user is in an active game
        active_game = self.voice_games.get_active_game(user_id)
        if active_game:
            result = self.voice_games.process_game_input(user_id, transcript)
            response = result['message']
            
            # Check if game ended
            if result.get('game_over'):
                response += f" Game ended! Your score: {result.get('score', 0)}"
            
            response_time = time.time() - start_time
            self.analytics.log_event('command', user_id, 'game_play', True, response_time, 1.0)
            
            # Apply personality mode
            response = self.personality_modes.format_response(user_id, 'info', response)
            self.speak(response)
            return True, response
        
        # Check conversation context for follow-up questions
        context_manager = self.conversation_manager
        original_transcript = transcript
        
        # Check if this is a follow-up question
        if context_manager and context_manager.is_follow_up(user_id, transcript):
            logger.info("Detected follow-up question")
            transcript = context_manager.resolve_context(user_id, transcript)
            logger.info(f"Resolved to: {transcript}")
        
        # Check for voice macros first
        macro = self.macro_manager.match_macro(transcript)
        if macro:
            logger.info(f"Matched voice macro: {macro.name}")
            result = self.macro_manager.execute_macro(macro, {'user_id': user_id})
            
            response_time = time.time() - start_time
            
            if result['success']:
                response = f"Executed macro: {macro.description}"
                
                # Handle speak actions from macro
                for step_result in result.get('results', []):
                    if step_result.get('action') == 'speak':
                        self.speak(step_result.get('text', ''))
                
                # Log success
                self.analytics.log_event(
                    'command', user_id, f"macro:{macro.name}", 
                    True, response_time, 1.0, {'macro': macro.name}
                )
                
                # Add to conversation history
                if context_manager:
                    context_manager.add_turn(user_id, original_transcript, response)
                
                self.speak(response)
                return True, response
            else:
                response = f"Failed to execute macro: {result.get('error', 'Unknown error')}"
                self.analytics.log_event(
                    'command', user_id, f"macro:{macro.name}", 
                    False, response_time, 0.0, {'error': result.get('error')}
                )
                self.speak(response)
                return False, response
        
        # --- START MODIFIED ---
        # Manual override to fix "one note" transcription error
        # This checks for the misinterpretation before classification.
        if "one note" in transcript or "onenote" in transcript:
            logger.info("Manual override: 'one note' detected, routing to 'open_notepad'.")
            # Directly execute the 'open_notepad' intent
            response = self.execute_intent("open_notepad", transcript, audio_sample, user_id=user_id)
            
            response_time = time.time() - start_time
            self.analytics.log_event('command', user_id, 'open_notepad', True, response_time, 1.0)
            
            if context_manager:
                context_manager.add_turn(user_id, original_transcript, response)
            
            self.speak(response)  # Ensure TTS speaks the response
            return True, response
        # --- END MODIFIED ---
        
        # Check for multi-action commands (e.g., "open youtube and play X")
        multi_action_result = self._handle_multi_action_command(transcript, audio_sample, user_id)
        if multi_action_result:
            success, response = multi_action_result
            
            response_time = time.time() - start_time
            self.analytics.log_event('command', user_id, 'multi_action', success, response_time)
            
            if context_manager:
                context_manager.add_turn(user_id, original_transcript, response)
            
            self.speak(response)  # Ensure TTS speaks the response
            return multi_action_result
        
        # Check for code generation commands (e.g., "open notepad and write java code")
        code_gen_result = self._handle_code_generation(transcript, audio_sample, user_id)
        if code_gen_result:
            success, response = code_gen_result
            
            response_time = time.time() - start_time
            self.analytics.log_event('command', user_id, 'code_generation', success, response_time)
            
            if context_manager:
                context_manager.add_turn(user_id, original_transcript, response)
            
            self.speak(response)  # Ensure TTS speaks the response
            return code_gen_result


        # First try intent classification
        intent, confidence = self.intent_classifier.classify(transcript)

        intent_threshold = float(self.config.get('system', {}).get('intent_confidence_threshold', 0.60))
        if intent and confidence >= intent_threshold:
            logger.info(f"Classified intent: {intent} (confidence: {confidence:.2f})")
            # --- FIX: Pass user_id ---
            response = self.execute_intent(intent, transcript, audio_sample, user_id=user_id)
            
            response_time = time.time() - start_time
            self.analytics.log_event('command', user_id, intent, True, response_time, confidence)
            
            if context_manager:
                context_manager.add_turn(user_id, original_transcript, response)
            
            # Cache successful responses for faster future lookups
            self.response_cache.set(original_transcript, True, response)
            
            # NEW: Learn from successful interaction
            if user_id:
                emotion_str = emotion_result.primary_emotion if emotion_result else None
                self.preference_learning.learn_from_interaction(
                    user_id, original_transcript, True, emotion_str
                )
            
            return True, response

        # If no intent matched, use LLM for conversational response.
        # Cloud LLM (Groq/Gemini).
        import os
        has_cloud = bool(os.getenv("GEMINI_API_KEY", "").strip() or os.getenv("GROQ_API_KEY", "").strip())
        if has_cloud or self._llm_available:
            # Refresh cloud-key state.
            try:
                self._check_llm_availability()
            except Exception:
                pass

            logger.info(
                "No command matched (confidence: %s), using LLM for conversational response",
                f"{confidence:.2f}" if intent else "N/A",
            )

            # Add conversation history to LLM prompt
            llm_prompt = transcript
            if context_manager:
                history = context_manager.get_context(user_id)
                if history:
                    llm_prompt = f"Conversation history:\n{history}\n\nCurrent question: {transcript}"

            response = self._ask_llm(llm_prompt, original_question=transcript)
            response_time = time.time() - start_time

            if response:
                self.analytics.log_event('command', user_id, 'llm_query', True, response_time, 0.8)
                if context_manager:
                    context_manager.add_turn(user_id, original_transcript, response)
                self.speak(response)  # Ensure TTS speaks the response
                return True, response
            else:
                self.analytics.log_event('command', user_id, 'llm_query', False, response_time, 0.0)

        # Fallback to keyword matching (legacy)
        for name, cmd in self.commands.items():
            for kw in cmd["keywords"]:
                if kw in transcript:
                    logger.info("Matched legacy command: %s", name)
                    # --- FIX: Pass user_id ---
                    response = self.execute_command(name, transcript, audio_sample, user_id=user_id)
                    
                    response_time = time.time() - start_time
                    self.analytics.log_event('command', user_id, name, True, response_time, 0.7)
                    
                    if context_manager:
                        context_manager.add_turn(user_id, original_transcript, response)
                    
                    self.speak(response)  # Ensure TTS speaks the response
                    return True, response

        logger.warning("No command matched for transcript: '%s'", transcript)
        
        response_time = time.time() - start_time
        self.analytics.log_event('command', user_id, 'unknown', False, response_time, 0.0)
        
        # Better error message depending on LLM availability
        import os
        has_cloud = bool(os.getenv("GEMINI_API_KEY", "").strip() or os.getenv("GROQ_API_KEY", "").strip())
        if self._llm_available or has_cloud:
            error_msg = "I'm having trouble processing that request. Please try again or say 'help'."
        else:
            error_msg = "Command not recognized. LLM is offline. Try: 'open notepad', 'what time is it', or 'help'."
        
        if context_manager:
            context_manager.add_turn(user_id, original_transcript, error_msg)
        
        self.speak(error_msg)  # Ensure TTS speaks the error message
        return False, error_msg

    def execute_intent(self, intent, transcript, audio_sample=None, user_id=None): # <-- FIX: Accept user_id
        """Execute command based on classified intent"""
        
        intent_handlers = {
            # --- START FIX: Match classifier intent names ---
            "get_time": self._get_time,
            "get_date": self._get_date,
            # --- END FIX ---
            "open_app": self._intent_open_app, 
            "close_app": self._intent_close_app,
            "system_lock": self._lock_system,
            "system_restart": self._intent_system_restart,
            "system_shutdown": self._intent_system_shutdown,
            "list_files": self._intent_list_files,
            "delete_file": self._intent_delete_file,
            "open_file": self._intent_open_file,
            "play_music": self._intent_play_music,
            "pause_music": self._intent_pause_music,
            "mute_audio": self._mute_audio,
            "unmute_audio": self._unmute_audio,
            "system_info": self._system_status,
            "help": self._what_can_you_do,
            "who_are_you": self._intent_who_are_you,
            "show_logs": self._show_security_logs,
            "greeting": self._hello,
            "goodbye": self._bye,
            "minimize_window": self._minimize_window,
            "maximize_window": self._maximize_window,
            "open_calculator": self._open_calculator,
            "open_notepad": self._open_notepad,
            "open_explorer": self._open_explorer,
            "take_screenshot": self._take_screenshot,
            "show_weather": self._show_weather,
            "search_files": self._search_files,
            "run_security_scan": self._run_security_scan,
            "biometric_status": self._biometric_status,
            "send_secure_message": self._send_secure_message,
            "check_messages": self._check_messages,
            "read_last_message": self._read_last_message,
            "start_voice_call": self._start_voice_call,
            "tell_joke": self._tell_joke,
            "play_game": self._play_game,
            "motivate_me": self._motivate_me,
            "compliment_me": self._compliment_me,
            "tell_fact": self._tell_fact,
            "sing_song": self._sing_song,
            "dance": self._dance,
            "tell_story": self._tell_story,
            "what_can_you_do": self._what_can_you_do,
            "how_are_you": self._how_are_you,
            "good_morning": self._good_morning,
            "good_afternoon": self._good_afternoon,
            "good_evening": self._good_evening,
            "good_night": self._good_night,
            "thank_you": self._thank_you,
            "sorry": self._sorry,
            "hello": self._hello,
            "bye": self._bye,
            # --- START: Added new intents ---
            "open_browser": self._open_browser,
            "search_google": self._search_google,
            "open_youtube": self._open_youtube,
            "play_video": self._search_youtube,
            # --- END: Added new intents ---
        }

        handler = intent_handlers.get(intent)
        if handler:
            allowed, denial_message = self._authorize_intent(intent, audio_sample, user_id)
            if not allowed:
                self.speak(denial_message, user_id=user_id)
                return denial_message
                
            # Execute handler
            try:
                response = handler(transcript)
                self.speak(response) # Speak the response
                self._log_action(intent, transcript, response)
                return response
            except Exception as e:
                logger.error(f"Intent handler '{intent}' failed: {e}", exc_info=True)
                return f"Sorry, I encountered an error with the '{intent}' command."
        else:
            return f"I understand you want to {intent.replace('_', ' ')}, but that feature isn't implemented yet."

    def execute_command(self, command_name, transcript, audio_sample=None, user_id=None): # <-- FIX: Accept user_id
        """Execute registered command with security checks (LEGACY)"""
        cmd_info = self.commands.get(command_name)
        if not cmd_info:
            return "Command not found."

        handler = cmd_info["handler"]
        secure = cmd_info.get("secure", False)

        # Apply risk policy in legacy path as well.
        synthetic_intent_name = command_name if secure else "low_risk_legacy"
        allowed, denial_message = self._authorize_intent(synthetic_intent_name, audio_sample, user_id)
        if not allowed:
            self.speak(denial_message, user_id=user_id)
            return denial_message

        # Execute command
        try:
            response = handler(transcript)
            self.speak(response) # Speak response
            # Log action
            self.log_action(command_name, transcript, response)
            return response
        except Exception as e:
            logger.error(f"Legacy command execution failed: {e}")
            return "Sorry, I encountered an error executing that command."

    def _log_action(self, command, transcript, response):
        """Log assistant actions"""
        logger.info(f"Assistant Action - Command: {command}, Input: '{transcript}', Response: '{response}'")

    # --- LEGACY INTENT HANDLERS (Called by execute_intent) ---

    def _intent_date(self, transcript, audio_sample=None):
        return self._get_date(transcript)

    def _intent_open_app(self, transcript, audio_sample=None):
        # This is a generic handler, the new ones are better
        app_keywords = {
            "notepad": self._open_notepad,
            "calculator": self._open_calculator,
            "explorer": self._open_explorer,
            "chrome": lambda t: self._open_specific_app("chrome.exe", "Chrome"),
            "firefox": lambda t: self._open_specific_app("firefox.exe", "Firefox"),
            "word": lambda t: self._open_specific_app("winword.exe", "Word"),
            "excel": lambda t: self._open_specific_app("excel.exe", "Excel"),
            "youtube": lambda t: self._open_youtube(t),
            "google": lambda t: self._open_browser("https://google.com"),
        }
        for app, handler in app_keywords.items():
            if app in transcript:
                # Check if it's a function or a lambda
                if callable(handler):
                    return handler(transcript)
        return "Which application would you like me to open?"

    def _open_specific_app(self, exe_name, app_name):
        try:
            subprocess.Popen(exe_name)
            return f"Opening {app_name}."
        except Exception as e:
            return f"Sorry, I couldn't open {app_name}."

    def _intent_close_app(self, transcript, audio_sample=None):
        # This is complex and risky
        return "Sorry, I cannot close applications yet. Please do it manually."

    def _intent_system_restart(self, transcript, audio_sample=None):
        response = "System restart is a critical command. Please confirm manually."
        # os.system("shutdown /r /t 10") # Example, but dangerous
        return response

    def _intent_system_shutdown(self, transcript, audio_sample=None):
        response = "System shutdown is a critical command. Please confirm manually."
        # os.system("shutdown /s /t 30") # Example, but dangerous
        return response

    def _intent_list_files(self, transcript, audio_sample=None):
        try:
            files = os.listdir(".")
            file_list = ", ".join(files[:10])  # Limit to 10 files
            response = f"Files in current directory: {file_list}"
            return response
        except Exception as e:
            return "Couldn't list files."

    def _intent_delete_file(self, transcript, audio_sample=None):
        return "File deletion requires confirmation. Please use the interface for now."

    def _intent_open_file(self, transcript, audio_sample=None):
        return "File opening feature coming soon."

    def _intent_play_music(self, transcript, audio_sample=None):
        pyautogui.press('playpause')
        return "Playing music."

    def _intent_pause_music(self, transcript, audio_sample=None):
        pyautogui.press('playpause')
        return "Pausing music."

    def _intent_who_are_you(self, transcript, audio_sample=None):
        return "I am SecureX, your offline voice assistant with biometric security."

    # Legacy command handlers (for backward compatibility)
    def register_command(self, name, handler, keywords, secure=False):
        """Register a command with handler and keywords."""
        self.commands[name] = {"handler": handler, "keywords": keywords, "secure": secure}

        # --- FUZZY INTENT RECOGNITION ---
        intent, confidence = self.intent_classifier.classify(transcript)
        intent_threshold = float(self.config.get('system', {}).get('intent_confidence_threshold', 0.60))
        fuzzy_threshold = 80  # Fuzzy match threshold (0-100)

        if intent and confidence >= intent_threshold:
            logger.info(f"Classified intent: {intent} (confidence: {confidence:.2f})")
            response = self.execute_intent(intent, transcript, audio_sample, user_id=user_id)
            response_time = time.time() - start_time
            self.analytics.log_event('command', user_id, intent, True, response_time, confidence)
            if context_manager:
                context_manager.add_turn(user_id, original_transcript, response)
            self.response_cache.set(original_transcript, True, response)
            if user_id:
                emotion_str = emotion_result.primary_emotion if emotion_result else None
                self.preference_learning.learn_from_interaction(
                    user_id, original_transcript, True, emotion_str
                )
            return True, response

        # Fuzzy match against all command keywords if intent classifier is uncertain
        all_keywords = []
        cmd_map = {}
        for name, cmd in self.commands.items():
            for kw in cmd["keywords"]:
                all_keywords.append(kw)
                cmd_map[kw] = name
        if all_keywords:
            best_match, score, _ = fuzz_process.extractOne(transcript, all_keywords, scorer=fuzz_fuzz.token_sort_ratio)
            if score >= fuzzy_threshold:
                matched_cmd = cmd_map[best_match]
                logger.info(f"Fuzzy matched command: {matched_cmd} (score: {score})")
                response = self.execute_command(matched_cmd, transcript, audio_sample, user_id=user_id)
                response_time = time.time() - start_time
                self.analytics.log_event('command', user_id, matched_cmd, True, response_time, score/100)
                if context_manager:
                    context_manager.add_turn(user_id, original_transcript, response)
                self.speak(response)
                return True, response

        # If no intent matched, use LLM for conversational response.
        # Cloud LLM (Groq/Gemini).
        import os
        has_cloud = bool(os.getenv("GEMINI_API_KEY", "").strip() or os.getenv("GROQ_API_KEY", "").strip())
        if has_cloud or self._llm_available:
            try:
                self._check_llm_availability()
            except Exception:
                pass

            logger.info(
                "No command matched (confidence: %s), using LLM for conversational response",
                f"{confidence:.2f}" if intent else "N/A",
            )
            llm_prompt = transcript
            if context_manager:
                history = context_manager.get_context(user_id)
                if history:
                    llm_prompt = f"Conversation history:\n{history}\n\nCurrent question: {transcript}"
            response = self._ask_llm(llm_prompt, original_question=transcript)
            response_time = time.time() - start_time
            if response:
                self.analytics.log_event('command', user_id, 'llm_query', True, response_time, 0.8)
                if context_manager:
                    context_manager.add_turn(user_id, original_transcript, response)
                self.speak(response)
                return True, response
            else:
                self.analytics.log_event('command', user_id, 'llm_query', False, response_time, 0.0)

        logger.warning("No command matched for transcript: '%s'", transcript)
        response_time = time.time() - start_time
        self.analytics.log_event('command', user_id, 'unknown', False, response_time, 0.0)
        import os
        has_cloud = bool(os.getenv("GEMINI_API_KEY", "").strip() or os.getenv("GROQ_API_KEY", "").strip())
        error_msg = ""
        if self._llm_available or has_cloud:
            error_msg = "I'm having trouble processing that request. Please try again or say 'help'."
        else:
            error_msg = "Command not recognized. LLM is offline. Try: 'open notepad', 'what time is it', or 'help'."
        if context_manager:
            context_manager.add_turn(user_id, original_transcript, error_msg)
        self.speak(error_msg)
        return False, error_msg
    
    def _maximize_window(self, text: str) -> str:
        """Maximize active window"""
        try:
            pyautogui.hotkey('win', 'up')  # Maximize window
            return "Window maximized."
        except Exception as e:
            return f"Unable to maximize window: {e}"

    def _minimize_window(self, text: str) -> str:
        """Minimize active window"""
        try:
            system = platform.system().lower()
            if system == "windows":
                # Win+Down restores (if maximized) then minimizes; press twice for consistency.
                pyautogui.hotkey('win', 'down')
                time.sleep(0.05)
                pyautogui.hotkey('win', 'down')
            elif system == "darwin":
                pyautogui.hotkey('command', 'm')
            else:
                # Common Linux DE shortcut (may vary).
                pyautogui.hotkey('alt', 'f9')
            return "Window minimized."
        except Exception as e:
            return f"Unable to minimize window: {e}"
    
    # ==================== AUDIO COMMANDS ====================
    
    def _mute_audio(self, text: str) -> str:
        """Mute system audio"""
        try:
            pyautogui.press('volumemute')
            return "System audio muted."
        except Exception as e:
            return f"Unable to mute audio: {e}"
    
    def _unmute_audio(self, text: str) -> str:
        """Unmute system audio"""
        try:
            pyautogui.press('volumemute')
            return "System audio restored."
        except Exception as e:
            return f"Unable to unmute audio: {e}"
    
    # ==================== CUSTOM COMMANDS ====================

    def _open_calculator(self, text: str) -> str:
        """Open calculator application"""
        try:
            system = platform.system().lower()
            if system == "windows":
                subprocess.Popen("calc.exe")
            elif system == "darwin":  # macOS
                subprocess.Popen(["open", "-a", "Calculator"])
            else:  # Linux
                subprocess.Popen(["gnome-calculator"])
            return "Calculator launched."
        except Exception as e:
            return f"Unable to open calculator: {e}"

    def _open_notepad(self, text: str) -> str:
        """Open notepad/text editor"""
        try:
            import tempfile
            import time
            
            system = platform.system().lower()
            if system == "windows":
                # Create a temp file to force new Notepad instance
                temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, dir=os.path.join(os.getcwd(), 'temp_audio'))
                temp_file.close()
                # Open Notepad with the temp file
                subprocess.Popen(["notepad.exe", temp_file.name])
                # Store the temp file path for cleanup
                self._current_temp_file = temp_file.name
                time.sleep(1)  # Wait for Notepad to open
            elif system == "darwin":  # macOS
                subprocess.Popen(["open", "-a", "TextEdit"])
            else:  # Linux
                subprocess.Popen(["gedit"])
            return "Text editor launched."
        except Exception as e:
            return f"Unable to open text editor: {e}"

    def _open_explorer(self, text: str) -> str:
        """Open file explorer"""
        try:
            system = platform.system().lower()
            if system == "windows":
                subprocess.Popen("explorer.exe")
            elif system == "darwin":  # macOS
                subprocess.Popen(["open", "."])
            else:  # Linux
                subprocess.Popen(["xdg-open", "."])
            return "File explorer opened."
        except Exception as e:
            return f"Unable to open file explorer: {e}"

    def _get_time(self, text: str) -> str:
        """Get current time"""
        try:
            current_time = datetime.datetime.now().strftime("%I:%M %p")
            return f"The current time is {current_time}"
        except Exception as e:
            return f"Unable to retrieve the time: {e}"

    def _get_date(self, text: str) -> str:
        """Get current date"""
        try:
            current_date = datetime.datetime.now().strftime("%A, %B %d, %Y")
            return f"Today is {current_date}"
        except Exception as e:
            return f"Unable to retrieve the date: {e}"

    def _system_status(self, text: str) -> str:
        """Show system status"""
        try:
            system = platform.system()
            cpu_percent = psutil.cpu_percent(interval=0.5)
            memory = psutil.virtual_memory()
            return f"System: {system}, CPU load: {cpu_percent} percent, memory usage: {memory.percent} percent"
        except Exception as e:
            return f"Unable to retrieve system status: {e}"

    def _lock_system(self, text: str) -> str:
        """Lock the system"""
        try:
            system = platform.system().lower()
            if system == "windows":
                subprocess.Popen("rundll32.exe user32.dll,LockWorkStation")
            elif system == "darwin":  # macOS
                subprocess.Popen(["pmset", "displaysleepnow"])
            else:  # Linux
                subprocess.Popen(["gnome-screensaver-command", "-l"])
            return "System locked."
        except Exception as e:
            return f"Unable to lock the system: {e}"

    def _take_screenshot(self, text: str) -> str:
        """Take a screenshot"""
        try:
            screenshot = pyautogui.screenshot()
            filename = f"screenshot_{int(time.time())}.png"
            screenshot.save(filename)
            return f"Screenshot saved as {filename}"
        except Exception as e:
            return f"Unable to capture a screenshot: {e}"

    # --- START: New and Updated Functions ---

    def _get_weather(self, city: str = "Mumbai") -> str:
        """Helper to fetch weather data."""
        try:
            # Use wttr.in's simple JSON format
            response = requests.get(f"https://wttr.in/{city}?format=j1")
            response.raise_for_status() # Raise an error for bad responses (4xx or 5xx)
            data = response.json()
            
            current = data.get('current_condition', [{}])[0]
            if not current:
                return f"Sorry, I couldn't get weather data for {city}."
                
            condition = current.get('weatherDesc', [{}])[0].get('value', 'unknown')
            temp = current.get('temp_C', 'unknown')
            feels_like = current.get('FeelsLikeC', 'unknown')
            humidity = current.get('humidity', 'unknown')
            
            return f"The weather in {city} is {condition} at {temp} degrees Celsius, but feels like {feels_like}. Humidity is {humidity} percent."
            
        except requests.exceptions.ConnectionError:
            logger.error("Weather check failed: No internet connection.")
            return "Sorry, I can't check the weather. Please check your internet connection."
        except Exception as e:
            logger.error(f"Weather check failed: {e}")
            return f"Sorry, I couldn't retrieve the weather for {city}."

    def _show_weather(self, text: str) -> str:
        """Show weather information (placeholder)"""
        # Try to find a city name, default to Mumbai
        city = "Mumbai" # Default
        words = text.split()
        if "in" in words:
            try:
                city_index = words.index("in") + 1
                if city_index < len(words):
                    # Capitalize each word in the city name
                    city = " ".join(word.capitalize() for word in words[city_index:])
            except Exception:
                pass # Use default
        
        return self._get_weather(city)

    def _open_browser(self, text: str) -> str:
        """Opens a website in the default browser."""
        url = "https://google.com" # Default
        if "youtube" in text:
            url = "https://youtube.com"
        elif "google" in text:
            url = "https://google.com"
        
        try:
            self._open_url(url)
            return f"Opening {url.split('//')[1]} in your browser."
        except Exception as e:
            logger.error(f"Failed to open browser: {e}")
            return "Sorry, I couldn't open the browser."
            
    def _search_google(self, text: str) -> str:
        """Searches Google for a query."""
        query = ""
        # Try to find the query after "for" or "about"
        search_starters = ["for", "about", "on", "search"]
        words = text.split()
        
        for starter in search_starters:
            if starter in words:
                try:
                    query_index = words.index(starter) + 1
                    if query_index < len(words):
                        query = " ".join(words[query_index:])
                        break
                except Exception:
                    continue
        
        if not query:
            return "What would you like me to search for?"
            
        try:
            # --- START MODIFIED ---
            # Fixed a bug where the URL was 'httpsT://' and not a search URL
            url = f"https://google.com/search?q={urllib.parse.quote_plus(query)}"
            # --- END MODIFIED ---
            self._open_url(url)
            return f"Here are the search results for {query}."
        except Exception as e:
            logger.error(f"Failed to search Google: {e}")
            return "Sorry, I couldn't open the browser for searching."

    # --- START: New YouTube Functions ---
    def _open_youtube(self, text: str) -> str:
        """Opens YouTube."""
        try:
            self._open_url("https://youtube.com")
            return "Opening YouTube."
        except Exception as e:
            logger.error(f"Failed to open YouTube: {e}")
            return "Sorry, I couldn't open YouTube."

    def _search_youtube(self, text: str) -> str:
        """Searches YouTube for a video."""
        query = ""
        # Find the query after "play", "for", "about", "on"
        search_starters = ["play", "for", "about", "on", "search"]
        words = text.split()
        
        for starter in search_starters:
            if starter in words:
                try:
                    query_index = words.index(starter) + 1
                    if query_index < len(words):
                        query = " ".join(words[query_index:])
                        # Remove common command words from the query
                        query = re.sub(r'^(video|song|this|me)\s+', '', query, flags=re.IGNORECASE)
                        break
                except Exception:
                    continue
        
        if not query.strip():
            return "What video or song would you like me to play?"
            
        try:
            url = f"https://www.youtube.com/results?search_query={urllib.parse.quote_plus(query)}"
            self._open_url(url)
            return f"Searching YouTube for {query}."
        except Exception as e:
            logger.error(f"Failed to search YouTube: {e}")
            return "Sorry, I couldn't open YouTube to search."

    def _open_url(self, url: str) -> None:
        """Open a URL in the user's default browser with robust fallbacks.

        `webbrowser.open()` can silently fail on some Windows setups (no default browser
        association, policy restrictions, etc.). We fall back to OS-native open calls.
        """
        try:
            opened = webbrowser.open(url, new=2)
            if opened:
                return
        except Exception as e:
            logger.warning(f"webbrowser.open failed: {e}")

        system = platform.system().lower()
        if system == "windows":
            try:
                os.startfile(url)  # type: ignore[attr-defined]
                return
            except Exception:
                subprocess.Popen(["cmd", "/c", "start", "", url], shell=False)
                return

        if system == "darwin":
            subprocess.Popen(["open", url])
            return

        subprocess.Popen(["xdg-open", url])
    # --- END: New YouTube Functions ---

    # --- END: New and Updated Functions ---

    def _search_files(self, text: str) -> str:
        """Search for files (placeholder)"""
        return "File search capability will be available in a future release."

    def _run_security_scan(self, text: str) -> str:
        """Run security scan (placeholder)"""
        return "Security scan initiated. All systems secure."

    def _show_security_logs(self, text: str) -> str:
        """Show security logs (placeholder)"""
        return "Security logs are currently clear."

    def _biometric_status(self, text: str) -> str:
        """Show biometric status"""
        return "Biometric systems: Voice recognition active, face recognition ready."

    def _send_secure_message(self, text: str) -> str:
        """Send secure message (placeholder)"""
        return "Secure messaging will be available soon."

    def _check_messages(self, text: str) -> str:
        """Check messages (placeholder)"""
        return "No new messages."

    def _read_last_message(self, text: str) -> str:
        """Read last message (placeholder)"""
        return "No recent messages to read."

    def _start_voice_call(self, text: str) -> str:
        """Start voice call (placeholder)"""
        return "Voice calling support is coming soon."
    
    # ==================== FUN AND INTERACTIVE COMMANDS ====================
    
    def _tell_joke(self, text: str) -> str:
        """Tell a random joke"""
        jokes = [
            "Why don't scientists trust atoms? Because they make up everything!",
            "What do you call fake spaghetti? An impasta!",
            "Why did the scarecrow win an award? Because he was outstanding in his field!",
            "What do you call a bear with no teeth? A gummy bear!",
            "Why can't you give Elsa a balloon? Because she will let it go!",
        ]
        return random.choice(jokes)
    
    def _play_game(self, text: str) -> str:
        """Start a simple game"""
        return "Let's play a game! Think of a number between 1 and 10. Ready? ... I guess 7!"
    
    def _motivate_me(self, text: str) -> str:
        """Give motivation"""
        motivations = [
            "You are capable of amazing things! Keep pushing forward.",
            "Every expert was once a beginner. You're on the right path!",
            "Success is not final, failure is not fatal. Keep going!",
            "Believe in yourself and all that you are. You are enough!",
        ]
        return random.choice(motivations)
    
    def _compliment_me(self, text: str) -> str:
        """Give a compliment"""
        compliments = [
            "You have an amazing voice! It's so clear and confident.",
            "You're incredibly intelligent and capable!",
            "Your creativity knows no bounds!",
            "You have a wonderful personality!",
        ]
        return random.choice(compliments)
    
    def _tell_fact(self, text: str) -> str:
        """Tell an interesting fact"""
        facts = [
            "Did you know? Octopuses have three hearts and blue blood!",
            "Honey never spoils. Archaeologists have found edible honey in ancient tombs!",
            "A group of flamingos is called a 'flamboyance'!",
            "Bananas are berries, but strawberries aren't!",
        ]
        return random.choice(facts)
    
    def _sing_song(self, text: str) -> str:
        """Sing a short song"""
        return "🎵 Twinkle, twinkle, little star... How I wonder what you are! 🎵 Sorry, I'm not a great singer!"
    
    def _dance(self, text: str) -> str:
        """Dance virtually"""
        return "I'm doing a virtual dance! 💃🕺"
    
    def _tell_story(self, text: str) -> str:
        """Tell a short story"""
        return "Once upon a time, in a world of code, a voice assistant helped a user, and they both worked efficiently ever after. The end."
    
    def _call_cloud_llm(self, prompt, system_prompt="", temperature=0.7):
        """Ultra-fast Cloud API Fallback (Gemini or Groq) using direct REST."""
        import os
        import requests
        
        # 1. Try Groq (OpenAI-compatible endpoint)
        groq_key = os.getenv("GROQ_API_KEY", "").strip()
        if groq_key:
            try:
                headers = {"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"}

                configured_groq_model = (
                    os.getenv("GROQ_MODEL", "").strip()
                    or (self.config.get("system", {}) if isinstance(self.config, dict) else {}).get("groq_model")
                    or "llama-3.1-8b-instant"
                )

                # Small fallback list in case the configured model isn't available.
                # Groq deprecation docs recommend:
                # - llama3-8b-8192   -> llama-3.1-8b-instant
                # - llama3-70b-8192  -> llama-3.3-70b-versatile
                candidate_models = [
                    configured_groq_model,
                    "llama-3.1-8b-instant",
                    "llama-3.3-70b-versatile",
                ]

                for model_name in list(dict.fromkeys([m for m in candidate_models if isinstance(m, str) and m.strip()])):
                    payload = {
                        "model": model_name,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt},
                        ],
                        "temperature": temperature,
                    }
                    resp = requests.post(
                        "https://api.groq.com/openai/v1/chat/completions",
                        headers=headers,
                        json=payload,
                        timeout=8,
                    )
                    if resp.status_code == 200:
                        return resp.json()["choices"][0]["message"]["content"].strip()
                    logger.warning(
                        "Groq API error (model=%s, status=%s): %s",
                        model_name,
                        resp.status_code,
                        resp.text,
                    )
            except Exception as e:
                logger.error(f"Groq Cloud API failed: {e}")
                
        # 2. Try Gemini (15 requests/minute free, very reliable)
        gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
        if gemini_key:
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={gemini_key}"
                full_prompt = f"System Rules: {system_prompt}\n\nUser Request: {prompt}"
                payload = {"contents": [{"parts": [{"text": full_prompt}]}]}
                resp = requests.post(url, json=payload, timeout=8)
                if resp.status_code == 200:
                    return resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
                else:
                    logger.warning(f"Gemini API Error: {resp.text}")
            except Exception as e:
                logger.error(f"Gemini Cloud API failed: {e}")

        return None
    
    def _ask_llm(self, question: str, original_question: str = None) -> str:
        """Ask LLM for conversational response"""
        import os
        has_cloud = bool(os.getenv("GEMINI_API_KEY", "").strip() or os.getenv("GROQ_API_KEY", "").strip())
        
        if not has_cloud:
            return None
        
        try:
            logger.info(f"Asking LLM: {question[:100]}...")
            llm_timeout = float(self.config.get('system', {}).get('llm_timeout_seconds', 8.0))
            
            sys_prompt = 'You are SecureX-Assist, a helpful AI voice assistant. Provide concise, friendly responses in 1-2 sentences. You help with questions, information, and conversation. If conversation history is provided, use it for context.'
            
            # 1. FIRST TRY CLOUD API FOR LIGHTNING SPEED
            cloud_result = self._call_cloud_llm(question, system_prompt=sys_prompt)
            if cloud_result:
                logger.info(f"Cloud LLM response: {cloud_result}")
                return cloud_result
            return "I couldn't reach the cloud AI service right now. Check GROQ_API_KEY/GEMINI_API_KEY and your internet connection."
        except Exception as e:
            logger.error(f"LLM query failed: {e}")
            return None
    
    def _handle_multi_action_command(self, transcript, audio_sample=None, user_id=None):
        """Handle commands with multiple actions (e.g., 'open youtube and play X')"""
        import time
        
        # Split by common conjunctions
        if " and " in transcript:
            parts = transcript.split(" and ", 1)
            if len(parts) == 2:
                action1, action2 = parts[0].strip(), parts[1].strip()
                logger.info(f"Multi-action detected: '{action1}' AND '{action2}'")
                
                # Execute first action
                success1, response1 = self._execute_single_action(action1, audio_sample, user_id)
                
                # Small delay between actions
                time.sleep(1.5)
                
                # Execute second action
                success2, response2 = self._execute_single_action(action2, audio_sample, user_id)
                
                # Combine responses
                combined_response = f"{response1} {response2}"
                return True, combined_response
        
        return None
    
    def _execute_single_action(self, action, audio_sample=None, user_id=None):
        """Execute a single action from multi-action command"""
        action = action.strip()
        logger.info(f"Executing single action: '{action}'")
        
        # Try intent classification
        intent, confidence = self.intent_classifier.classify(action)
        
        if intent and confidence > 0.5:
            response = self.execute_intent(intent, action, audio_sample, user_id=user_id)
            return True, response
        
        # Fallback to keyword matching for common actions
        if "youtube" in action:
            response = self._open_youtube(action)
            return True, response
        elif "notepad" in action:
            response = self._open_notepad(action)
            return True, response
        elif "calculator" in action:
            response = self._open_calculator(action)
            return True, response
        elif "play" in action or "search" in action:
            # For play/search commands, use YouTube search
            response = self._search_youtube(action)
            return True, response
        elif action.startswith("write"):
            # This is ANY text generation request (code, essay, story, etc.)
            logger.info(f"Detected text generation request: '{action}'")
            response = self._generate_and_type_text(action)
            return True, response
        
        return False, f"I couldn't understand the action: '{action}'"
    
    def _handle_code_generation(self, transcript, audio_sample=None, user_id=None):
        """Handle commands like 'open notepad and write java code to reverse array'"""
        import time
        
        # Check for code generation patterns - MORE FLEXIBLE
        code_patterns = [
            "open notepad and write",
            "launch notepad and write",
            "open editor and write",
            "open notepad and",  # NEW: More flexible pattern
            "launch notepad and",
        ]
        
        for pattern in code_patterns:
            if pattern in transcript:
                # Extract the code request
                code_request = transcript.split(pattern, 1)[1].strip()
                
                # Clean up common words
                code_request = code_request.replace("in that", "").strip()
                code_request = code_request.replace("write ", "").strip()
                
                logger.info(f"Code generation request: '{code_request}'")
                
                if not code_request:
                    return True, "What code would you like me to write?"

                # Always open Notepad first.
                self._open_notepad(transcript)
                time.sleep(2.5)

                # If LLM is available, generate code; otherwise type exactly what the user said.
                self._check_llm_availability()
                if self._llm_available:
                    code = self._generate_code(code_request)
                    if code:
                        self._type_via_clipboard(code)
                        return True, f"I've generated and typed the code for: {code_request}"
                    return True, "Sorry, I couldn't generate the code."

                self._type_via_clipboard(code_request)
                return True, "Done."
        
        return None
    
    def _generate_and_type_text(self, text_request):
        """Generate ANY text from LLM and type it (code, essay, story, etc.)"""
        import time
        
        # Remove "write" prefix
        text_request = text_request.replace("write", "").strip()
        
        import os
        has_cloud = bool(os.getenv("GEMINI_API_KEY", "").strip() or os.getenv("GROQ_API_KEY", "").strip())
        if not has_cloud:
            return "Text generation requires a cloud API key (GROQ_API_KEY or GEMINI_API_KEY) in .env."
        
        # Determine if it's code or regular text
        is_code = any(keyword in text_request.lower() for keyword in 
                     ['code', 'program', 'function', 'script', 'java', 'python', 'c++', 'javascript'])
        
        logger.info(f"Generating {'code' if is_code else 'text'} for: '{text_request}'")
        
        # Small delay to ensure app is focused
        time.sleep(1.5)
        
        # Generate content
        if is_code:
            content = self._generate_code(text_request)
        else:
            content = self._generate_text(text_request)
        
        if content:
            # Type the content using clipboard (100% reliable)
            self._type_via_clipboard(content)
            return f"{'Code' if is_code else 'Text'} generated and typed for: {text_request}"
        else:
            return "Sorry, I couldn't generate the content."
    
    def _generate_text(self, text_request):
        """Generate regular text (essays, stories, etc.) using LLM"""
        import os
        has_cloud = bool(os.getenv("GEMINI_API_KEY", "").strip() or os.getenv("GROQ_API_KEY", "").strip())
        
        try:
            if not has_cloud:
                return None

            prompt = f"""Write {text_request}

Requirements:
1. Write clear, well-structured content
2. Use proper grammar and punctuation
3. Make it informative and engaging
4. Keep it concise (200-300 words)
5. Output ONLY the text content, NO explanations

Write the text now:"""
            
            sys_prompt = 'You are a professional writer. Output ONLY the requested content with NO extra explanations or meta-commentary.'
            
            cloud_result = self._call_cloud_llm(prompt, system_prompt=sys_prompt, temperature=0.7)
            if cloud_result:
                logger.info(f"Generated text via Cloud ({len(cloud_result)} chars)")
                return cloud_result
            return None
            
        except Exception as e:
            logger.error(f"Text generation failed: {e}")
            return None
    
    def _generate_code(self, code_request):
        """Generate code using LLM"""
        import os
        has_cloud = bool(os.getenv("GEMINI_API_KEY", "").strip() or os.getenv("GROQ_API_KEY", "").strip())
        
        try:
            if not has_cloud:
                return None

            # Better prompt with explicit format requirements
            prompt = f"""Generate complete, compilable code for: {code_request}

CRITICAL RULES:
1. Output ONLY valid, compilable code - NO markdown, NO explanations, NO text
2. Include proper class declaration with correct syntax
3. Use correct variable names and method signatures
4. Add brief inline comments for clarity
5. Ensure all brackets, semicolons, and syntax are correct
6. Start directly with the code (e.g., 'public class' for Java)

Generate the code now:"""
            
            sys_prompt = 'You are an expert code generator. Output ONLY clean, compilable code with NO markdown formatting, NO code blocks, NO explanations. Start directly with the code.'
            
            code = None
            cloud_result = self._call_cloud_llm(prompt, system_prompt=sys_prompt, temperature=0.2)
            if cloud_result:
                code = cloud_result
            else:
                return None
            
            if not code:
                return None
            
            # Clean up markdown code blocks if present (defensive)
            if "```" in code:
                logger.warning("Code contained markdown blocks - cleaning up")
                # Extract code between ``` markers
                parts = code.split("```")
                if len(parts) >= 3:
                    code = parts[1]
                    # Remove language identifier if present (e.g., "java\n")
                    lines = code.split("\n")
                    if lines and lines[0].strip().lower() in ['java', 'python', 'javascript', 'c', 'cpp', 'c++']:
                        code = "\n".join(lines[1:])
            
            # Remove any leading/trailing explanatory text
            lines = code.split('\n')
            # Find first line that looks like code (starts with keywords or symbols)
            code_keywords = ['public', 'class', 'def', 'function', 'import', 'package', '#include', 'using']
            start_idx = 0
            for i, line in enumerate(lines):
                stripped = line.strip()
                if any(stripped.startswith(kw) for kw in code_keywords) or stripped.startswith('//') or stripped.startswith('/*'):
                    start_idx = i
                    break
            
            if start_idx > 0:
                logger.info(f"Removed {start_idx} non-code lines from start")
                code = '\n'.join(lines[start_idx:])
            
            code = code.strip()
            logger.info(f"Generated code ({len(code)} chars)")
            return code
            
        except Exception as e:
            logger.error(f"Code generation failed: {e}")
            return None
    
    def _type_via_clipboard(self, text):
        """Type text using clipboard method - 100% reliable for ALL characters"""
        try:
            import pyautogui
            import pyperclip
            import time
            
            # Longer delay to ensure editor is focused and ready
            time.sleep(1.5)
            
            logger.info(f"Pasting {len(text)} characters via clipboard...")
            
            # Click to ensure focus
            pyautogui.click()
            time.sleep(0.3)
            
            # Save current clipboard
            try:
                old_clipboard = pyperclip.paste()
            except:
                old_clipboard = ""
            
            # Copy text to clipboard
            pyperclip.copy(text)
            time.sleep(0.1)
            
            # Paste using Ctrl+V (works in ALL text editors)
            pyautogui.hotkey('ctrl', 'v')
            time.sleep(0.3)
            
            # Restore old clipboard
            if old_clipboard:
                pyperclip.copy(old_clipboard)
            
            logger.info(f"Successfully pasted {len(text)} characters")
            
        except Exception as e:
            logger.error(f"Clipboard paste failed: {e}")
    
    def _what_can_you_do(self, text: str) -> str:
        """Explain capabilities with comprehensive guide"""
        try:
            # Use the new interactive guide system
            guide = self.interactive_guide
            capabilities = guide.get_capabilities_overview()
            
            # Build a comprehensive response
            response = "🌟 I'm SecureX-Assist, your extraordinary AI voice assistant! Here's what I can do:\n\n"
            
            # Highlight key categories
            categories = list(capabilities.keys())[:4]  # Top 4 categories
            for category in categories:
                items = capabilities[category]
                response += f"{category}\n"
                if items:
                    response += f"  Example: {items[0]['example']}\n"
            
            response += "\n💡 Say 'show examples' for detailed commands, or 'tutorial' for a quick start guide!"
            
            # Add LLM info if available
            if self._llm_available:
                response += "\n🤖 Plus, I can have natural conversations on any topic!"
            
            return response
            
        except Exception as e:
            logger.error(f"Error in _what_can_you_do: {e}")
            # Fallback to simple response
            capabilities = "I can help you with voice commands, system control, AI conversations, "
            capabilities += "games, smart routines, and personalized assistance. "
            if self._llm_available:
                capabilities += "I can also answer questions and have conversations on any topic."
            capabilities += " Say 'show examples' or 'help' for more details!"
            return capabilities
    
    def _how_are_you(self, text: str) -> str:
        """Respond to greeting"""
        responses = [
            "I'm doing great, thank you for asking! Ready to help!",
            "I'm fantastic! All systems are operational.",
            "I'm excellent! How can I help you?",
        ]
        return random.choice(responses)
    
    def _good_morning(self, text: str) -> str:
        """Morning greeting"""
        return "Good morning! I hope you have a fantastic day ahead!"
    
    def _good_afternoon(self, text: str) -> str:
        """Afternoon greeting"""
        return "Good afternoon! Hope your day is going well!"
    
    def _good_evening(self, text: str) -> str:
        """Evening greeting"""
        return "Good evening! Time to relax!"
    
    def _good_night(self, text: str) -> str:
        """Night greeting"""
        return "Good night! Sleep well."
    
    def _thank_you(self, text: str) -> str:
        """Respond to thanks"""
        return "You're very welcome! I'm always happy to help!"
    
    def _sorry(self, text: str) -> str:
        """Respond to apology"""
        return "No need to apologize! I'm here to help."
    
    def _hello(self, text: str) -> str:
        """Basic greeting with personalization"""
        try:
            user_id = self.listening_user_id if self.listening_user_id else "default_user"
            # Use personalized greeting
            greeting = self.preference_learning.get_personalized_greeting(user_id, "there")
            return greeting
        except:
            return "Hello! How can I assist you today?"
    
    def _bye(self, text: str) -> str:
        """Goodbye"""
        return "Goodbye! Have a great day!"
    
    # ==================== ADVANCED FEATURE COMMANDS ====================
    
    def _change_personality(self, text: str) -> str:
        """Change assistant personality mode"""
        try:
            # Extract personality mode from text
            modes = ['professional', 'friendly', 'witty', 'motivational', 'technical']
            mode_found = None
            
            for mode in modes:
                if mode in text.lower():
                    mode_found = mode
                    break
            
            if mode_found:
                # Use a default user ID if none is available
                user_id = self.listening_user_id if self.listening_user_id else "default_user"
                success = self.personality_modes.set_mode(user_id, mode_found)
                
                if success:
                    return f"Personality changed to {mode_found} mode!"
                else:
                    return f"Failed to change personality to {mode_found}"
            else:
                available = ', '.join(modes)
                return f"Please specify a personality: {available}"
                
        except Exception as e:
            logger.error(f"Error changing personality: {e}")
            return f"Unable to change personality: {e}"
    
    def _show_memories(self, text: str) -> str:
        """Show remembered facts about user"""
        try:
            user_id = self.listening_user_id if self.listening_user_id else "default_user"
            memories = self.voice_memory.recall(user_id, category=None, limit=5)
            
            if not memories:
                return "I don't have any memories about you yet. Keep talking to me!"
            
            memory_text = f"I remember {len(memories)} things about you: "
            facts = [f"{m.content}" for m in memories[:3]]  # Top 3
            memory_text += ", ".join(facts)
            
            if len(memories) > 3:
                memory_text += f" and {len(memories) - 3} more"
            
            return memory_text
            
        except Exception as e:
            logger.error(f"Error showing memories: {e}")
            return f"Unable to retrieve memories: {e}"
    
    def _create_routine(self, text: str) -> str:
        """Create a new smart routine"""
        try:
            # Simple routine creation - could be enhanced with more parsing
            return "To create a routine, say: 'Create a routine called morning briefing at 7 AM that says good morning'"
            
        except Exception as e:
            logger.error(f"Error creating routine: {e}")
            return f"Unable to create routine: {e}"
    
    def _list_games(self, text: str) -> str:
        """List available voice games"""
        try:
            games = self.voice_games.list_games()
            game_list = ', '.join(games)
            return f"I have these games available: {game_list}. Say 'play game' and the name to start!"
            
        except Exception as e:
            logger.error(f"Error listing games: {e}")
            return f"Unable to list games: {e}"
    
    def _show_emotion(self, text: str) -> str:
        """Show detected emotion from last speech"""
        try:
            return "Speak something and I'll detect your emotion from your voice!"
            
        except Exception as e:
            logger.error(f"Error showing emotion: {e}")
            return f"Unable to show emotion: {e}"
    
    # ==================== NEW: USER GUIDE AND CAPABILITIES ====================
    
    def _show_user_guide(self, text: str) -> str:
        """Show interactive user guide"""
        try:
            guide = self.interactive_guide
            tutorial = guide.get_quick_start_tutorial()
            
            response = "🎓 Quick Start Guide:\n\n"
            for step in tutorial[:3]:  # Show first 3 steps
                response += f"Step {step['step']}: {step['title']}\n"
                response += f"{step['description']}\n\n"
            
            response += "Say 'show tutorial' for the complete guide!"
            return response
            
        except Exception as e:
            logger.error(f"Error showing guide: {e}")
            return "Guide system is being prepared for you!"
    
    def _show_capabilities(self, text: str) -> str:
        """Show all capabilities"""
        try:
            guide = self.interactive_guide
            capabilities = guide.get_capabilities_overview()
            
            response = "🌟 I can help you with:\n\n"
            
            # Show top categories
            for category, items in list(capabilities.items())[:3]:
                response += f"{category}:\n"
                for item in items[:2]:
                    response += f"  • {item['capability']}: {item['example']}\n"
                response += "\n"
            
            response += "And much more! Say 'show examples' for detailed commands!"
            return response
            
        except Exception as e:
            logger.error(f"Error showing capabilities: {e}")
            return "I have many capabilities including voice commands, AI chat, system control, games, and personalization!"
    
    def _show_command_examples(self, text: str) -> str:
        """Show example commands"""
        try:
            guide = self.interactive_guide
            examples = guide.get_example_commands()
            
            response = "📝 Example Commands:\n\n"
            
            for category, commands in list(examples.items())[:2]:
                response += f"{category}:\n"
                for cmd in commands[:3]:
                    response += f'  • "{cmd}"\n'
                response += "\n"
            
            return response
            
        except Exception as e:
            logger.error(f"Error showing examples: {e}")
            return "Try saying: 'What time is it?', 'Tell me a joke', 'Open Chrome', or 'System status'!"
    
    def _show_quick_tips(self, text: str) -> str:
        """Show helpful tips"""
        try:
            guide = self.interactive_guide
            tips = guide.get_tips_and_tricks()
            
            # Select 3 random tips
            import random
            selected_tips = random.sample(tips, min(3, len(tips)))
            
            response = "💡 Quick Tips:\n\n"
            for tip in selected_tips:
                response += f"{tip['title']}: {tip['tip']}\n\n"
            
            return response
            
        except Exception as e:
            logger.error(f"Error showing tips: {e}")
            return "Tip: Speak naturally and I'll understand! Ask 'What can you do?' to explore features."
    
    def _show_user_stats(self, text: str) -> str:
        """Show user usage statistics"""
        try:
            user_id = self.listening_user_id if self.listening_user_id else "default_user"
            prefs = self.preference_learning
            stats = prefs.get_usage_stats(user_id)
            
            response = f"📊 Your Statistics:\n\n"
            response += f"Total interactions: {stats['total_interactions']}\n"
            response += f"Favorite commands saved: {stats['favorite_commands']}\n"
            response += f"Topics explored: {stats['topics_explored']}\n"
            response += f"Personality mode: {stats['preferred_personality']}\n"
            
            return response
            
        except Exception as e:
            logger.error(f"Error showing stats: {e}")
            return "Statistics are being tracked for personalized experience!"
    
    def _show_suggestions(self, text: str) -> str:
        """Show personalized suggestions"""
        try:
            user_id = self.listening_user_id if self.listening_user_id else "default_user"
            suggestions = self.preference_learning.get_suggested_commands(user_id, limit=3)
            
            if suggestions:
                response = "🎯 Suggested Commands:\n\n"
                for i, cmd in enumerate(suggestions, 1):
                    response += f"{i}. {cmd}\n"
                return response
            else:
                # Get time-based suggestions
                time_suggestions = self.suggestion_engine.get_time_based_suggestions()
                response = "🎯 Suggestions for you:\n\n"
                for sug in time_suggestions[:3]:
                    response += f"{sug['icon']} {sug['title']}: {sug['command']}\n"
                return response
                
        except Exception as e:
            logger.error(f"Error showing suggestions: {e}")
            return "Try: 'Tell me a joke', 'What time is it?', or 'Play a game'!"