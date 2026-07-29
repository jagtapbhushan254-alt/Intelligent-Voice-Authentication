"""
SecureX-Assist - Security Framework
Multi-factor authentication and session management
"""

import secrets
import hashlib
import time
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import random
from enum import Enum
from collections import deque
import os
import subprocess

logger = logging.getLogger(__name__)


@dataclass
class SecurityContext:
    """Security state for authenticated sessions"""
    user_id: int
    username: str
    voice_verified: bool = False
    password_verified: bool = False
    liveness_verified: bool = False
    session_token: str = ""
    created_at: float = 0.0
    last_activity: float = 0.0
    
    def is_fully_authenticated(self) -> bool:
        """Check if all three factors are verified"""
        return (
            self.voice_verified and 
            self.password_verified and 
            self.liveness_verified
        )
    
    def is_expired(self, timeout: int = 3600) -> bool:
        """Check if session has expired"""
        if self.last_activity == 0:
            return True
        return (time.time() - self.last_activity) > timeout
    
    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity = time.time()


class AlertLevel(Enum):
    """Escalation levels for authentication failures."""
    NONE = "none"
    SOFT = "soft"
    ACTIVE = "active"
    HARD = "hard"


@dataclass
class AuthAlertDecision:
    """Decision returned after a failed authentication attempt."""
    level: AlertLevel
    message: str
    recent_failures: int
    cooldown_seconds: int = 0
    should_lock_workstation: bool = False


@dataclass
class AuthAlertState:
    """Per-user failure tracking for staged alerting."""
    failures: deque = field(default_factory=lambda: deque(maxlen=50))
    cooldown_until: Optional[float] = None


class AuthenticationAlertSystem:
    """
    Staged alerting for unauthenticated attempts.

    Levels:
    - SOFT: Warning + log
    - ACTIVE: Warning + cooldown
    - HARD: Extended cooldown + optional workstation lock
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        section = config.get("security_alerts", {})
        self.window_seconds = section.get("window_seconds", 120)
        self.soft_threshold = section.get("soft_threshold", 1)
        self.active_threshold = section.get("active_threshold", 3)
        self.hard_threshold = section.get("hard_threshold", 5)
        self.active_cooldown_seconds = section.get("active_cooldown_seconds", 30)
        self.hard_cooldown_seconds = section.get("hard_cooldown_seconds", 120)
        self.enable_workstation_lock = section.get("enable_workstation_lock", False)
        self.states: Dict[str, AuthAlertState] = {}

        logger.info(
            "AuthenticationAlertSystem initialized: window=%ss soft=%s active=%s hard=%s",
            self.window_seconds,
            self.soft_threshold,
            self.active_threshold,
            self.hard_threshold,
        )

    def get_remaining_cooldown(self, username: str) -> int:
        """Return remaining cooldown in seconds, or 0 if not throttled."""
        state = self.states.get(username)
        if not state or not state.cooldown_until:
            return 0
        remaining = int(max(0, state.cooldown_until - time.time()))
        if remaining == 0:
            state.cooldown_until = None
        return remaining

    def register_success(self, username: str) -> None:
        """Reset alert state after successful authentication."""
        if username in self.states:
            del self.states[username]

    def register_failure(
        self,
        username: str,
        reason: str = "authentication_failed",
        spoof_detected: bool = False,
        sensitive_action: bool = False,
    ) -> AuthAlertDecision:
        """Record failure and return escalation decision."""
        now = time.time()
        state = self.states.setdefault(username, AuthAlertState())
        state.failures.append(now)

        # Keep only failures in configured window.
        cutoff = now - self.window_seconds
        while state.failures and state.failures[0] < cutoff:
            state.failures.popleft()

        recent = len(state.failures)
        level = AlertLevel.SOFT if recent >= self.soft_threshold else AlertLevel.NONE
        cooldown_seconds = 0
        should_lock = False

        if recent >= self.hard_threshold or (spoof_detected and recent >= self.active_threshold):
            level = AlertLevel.HARD
            cooldown_seconds = self.hard_cooldown_seconds
            state.cooldown_until = now + cooldown_seconds
            should_lock = self.enable_workstation_lock
        elif recent >= self.active_threshold:
            level = AlertLevel.ACTIVE
            cooldown_seconds = self.active_cooldown_seconds
            state.cooldown_until = now + cooldown_seconds

        message = self._build_message(level, recent, reason, cooldown_seconds, spoof_detected)
        logger.warning(
            "Auth alert: user=%s level=%s recent=%s reason=%s cooldown=%ss spoof=%s",
            username,
            level.value,
            recent,
            reason,
            cooldown_seconds,
            spoof_detected,
        )

        if should_lock:
            self._lock_workstation()

        return AuthAlertDecision(
            level=level,
            message=message,
            recent_failures=recent,
            cooldown_seconds=cooldown_seconds,
            should_lock_workstation=should_lock,
        )

    @staticmethod
    def _build_message(
        level: AlertLevel,
        recent: int,
        reason: str,
        cooldown_seconds: int,
        spoof_detected: bool,
    ) -> str:
        if level == AlertLevel.HARD:
            base = "Hard security alert triggered"
        elif level == AlertLevel.ACTIVE:
            base = "Active security alert triggered"
        else:
            base = "Failed authentication attempt recorded"

        spoof_text = " (spoof suspected)" if spoof_detected else ""
        cooldown_text = f" Cooldown: {cooldown_seconds}s." if cooldown_seconds > 0 else ""
        return f"{base}{spoof_text}. Recent failures: {recent}. Reason: {reason}.{cooldown_text}"

    @staticmethod
    def _lock_workstation() -> None:
        """Lock workstation on Windows when hard alert is reached."""
        try:
            if os.name == "nt":
                subprocess.run(
                    ["rundll32.exe", "user32.dll,LockWorkStation"],
                    check=False,
                    capture_output=True,
                )
                logger.warning("Workstation lock command executed due to hard security alert")
        except Exception as e:
            logger.error(f"Failed to lock workstation: {e}")


class SecurityManager:
    """
    Manages multi-factor authentication and security operations
    Three-Factor Authentication (3FA):
    1. Voice Biometric (who you are)
    2. Password (what you know)
    3. Liveness Challenge (prove you're real)
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.sessions: Dict[str, SecurityContext] = {}
        self.failed_attempts: Dict[str, int] = {}

        # Security settings with backward-compatible fallbacks.
        security_cfg = config.get('security', {})
        system_cfg = config.get('system', {})
        self.max_attempts = security_cfg.get('max_login_attempts', system_cfg.get('max_login_attempts', 3))
        self.session_timeout = security_cfg.get('session_timeout', system_cfg.get('session_timeout', 3600))
        self.require_liveness = security_cfg.get('require_liveness', system_cfg.get('require_liveness', True))
        
        logger.info("SecurityManager initialized with 3FA")
    
    def create_session(self, user_id: int, username: str) -> SecurityContext:
        """
        Create a new security session
        
        Args:
            user_id: Database user ID
            username: Username
            
        Returns:
            New SecurityContext
        """
        session_token = secrets.token_urlsafe(32)
        
        context = SecurityContext(
            user_id=user_id,
            username=username,
            session_token=session_token,
            created_at=time.time(),
            last_activity=time.time()
        )
        
        self.sessions[session_token] = context
        logger.info(f"Created session for user: {username}")
        
        return context
    
    def verify_password(self, password: str, stored_hash: str) -> bool:
        """
        Verify password against stored hash
        
        Args:
            password: Plain text password
            stored_hash: Stored password hash
            
        Returns:
            True if password matches
        """
        password_hash = self.hash_password(password)
        return secrets.compare_digest(password_hash, stored_hash)
    
    @staticmethod
    def hash_password(password: str, salt: Optional[str] = None) -> str:
        """
        Hash password with SHA-256 and salt
        
        Args:
            password: Plain text password
            salt: Optional salt (generated if not provided)
            
        Returns:
            Hashed password string
        """
        if salt is None:
            salt = secrets.token_hex(16)
        
        # Combine password and salt
        salted = f"{password}{salt}".encode('utf-8')
        
        # Hash with SHA-256
        hashed = hashlib.sha256(salted).hexdigest()
        
        # Return format: salt$hash
        return f"{salt}${hashed}"
    
    def verify_hashed_password(self, password: str, stored_hash: str) -> bool:
        """
        Verify password against stored hash with salt
        
        Args:
            password: Plain text password
            stored_hash: Stored hash in format "salt$hash"
            
        Returns:
            True if password matches
        """
        try:
            # Split salt and hash
            if '$' not in stored_hash:
                # Legacy format without salt
                return False
            
            salt, expected_hash = stored_hash.split('$', 1)
            
            # Hash the provided password with same salt
            salted = f"{password}{salt}".encode('utf-8')
            actual_hash = hashlib.sha256(salted).hexdigest()
            
            # Constant-time comparison
            return secrets.compare_digest(actual_hash, expected_hash)
            
        except Exception as e:
            logger.error(f"Password verification failed: {e}")
            return False
    
    def generate_liveness_challenge(self) -> str:
        """
        Generate random phrase for liveness detection
        Prevents replay attacks by requiring real-time speech
        
        Returns:
            Random phrase like "Swift purple dragon flies"
        """
        adjectives = [
            "swift", "brave", "silent", "bright", "dark", "cyber", "neon",
            "quantum", "stellar", "cosmic", "electric", "digital", "chrome"
        ]
        
        nouns = [
            "dragon", "phoenix", "cipher", "matrix", "nexus", "prism",
            "vector", "vortex", "pulse", "echo", "signal", "ghost"
        ]
        
        verbs = [
            "flies", "soars", "jumps", "runs", "glows", "shines", "flows",
            "pulses", "streams", "phases", "warps", "shifts"
        ]
        
        adjective = random.choice(adjectives)
        noun = random.choice(nouns)
        verb = random.choice(verbs)
        
        phrase = f"{adjective} {noun} {verb}"
        logger.info(f"Generated liveness challenge: {phrase}")
        
        return phrase
    
    def verify_liveness(self, expected_phrase: str, transcribed_text: str) -> Tuple[bool, float]:
        """
        Verify liveness by comparing expected phrase with transcribed speech
        
        Args:
            expected_phrase: The challenge phrase generated
            transcribed_text: What the user said (from STT)
            
        Returns:
            (is_valid, similarity_score)
        """
        # Normalize text
        expected = expected_phrase.lower().strip()
        actual = transcribed_text.lower().strip()
        
        # Calculate similarity (simple word matching)
        expected_words = set(expected.split())
        actual_words = set(actual.split())
        
        if not expected_words:
            return False, 0.0
        
        # Word overlap
        overlap = len(expected_words & actual_words)
        similarity = overlap / len(expected_words)
        
        # Require at least 70% match
        is_valid = similarity >= 0.7
        
        logger.info(f"Liveness check: expected='{expected}', actual='{actual}', similarity={similarity:.2f}, valid={is_valid}")
        
        return is_valid, similarity
    
    def record_failed_attempt(self, username: str):
        """Record failed login attempt"""
        if username not in self.failed_attempts:
            self.failed_attempts[username] = 0
        
        self.failed_attempts[username] += 1
        
        if self.failed_attempts[username] >= self.max_attempts:
            logger.warning(f"Max login attempts reached for user: {username}")
    
    def reset_failed_attempts(self, username: str):
        """Reset failed attempt counter after successful login"""
        if username in self.failed_attempts:
            del self.failed_attempts[username]
    
    def is_account_locked(self, username: str) -> bool:
        """Check if account is locked due to too many failed attempts"""
        return self.failed_attempts.get(username, 0) >= self.max_attempts
    
    def get_session(self, session_token: str) -> Optional[SecurityContext]:
        """Retrieve session by token"""
        return self.sessions.get(session_token)
    
    def invalidate_session(self, session_token: str):
        """Destroy a session"""
        if session_token in self.sessions:
            username = self.sessions[session_token].username
            del self.sessions[session_token]
            logger.info(f"Session invalidated for user: {username}")
    
    def cleanup_expired_sessions(self):
        """Remove expired sessions"""
        current_time = time.time()
        expired_tokens = []
        
        for token, context in self.sessions.items():
            if (current_time - context.last_activity) > self.session_timeout:
                expired_tokens.append(token)
        
        for token in expired_tokens:
            self.invalidate_session(token)
        
        if expired_tokens:
            logger.info(f"Cleaned up {len(expired_tokens)} expired sessions")


class AntiSpoofingDetector:
    """
    Advanced anti-spoofing and anomaly detection
    Detects synthetic voices, recordings, and other attacks
    """
    
    @staticmethod
    def detect_replay_attack(audio_features: Dict) -> Tuple[bool, float]:
        """
        Detect if audio is a recording (replay attack)
        
        Returns:
            (is_replay, confidence)
        """
        # Placeholder for advanced ML-based detection
        # In production, use specialized anti-spoofing models like:
        # - ASVspoof models
        # - Deep learning classifiers trained on spoofed vs. genuine audio
        
        confidence = 0.0  # 0-1 scale
        is_replay = confidence > 0.5
        
        return is_replay, confidence
    
    @staticmethod
    def detect_synthetic_voice(audio_features: Dict) -> Tuple[bool, float]:
        """
        Detect if voice is AI-generated (deepfake)
        
        Returns:
            (is_synthetic, confidence)
        """
        # Check audio quality metrics
        snr = audio_features.get('snr', 0)
        
        # Very high SNR might indicate processed audio
        if snr > 30:
            return True, 0.6
        
        # Placeholder for ML model
        confidence = 0.0
        is_synthetic = confidence > 0.5
        
        return is_synthetic, confidence
