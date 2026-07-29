"""
SecureX-Assist - Utility Functions
Helper functions and utilities
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any
import os
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


def _get_project_root() -> Path:
    # utils/helpers.py -> project root
    return Path(__file__).resolve().parents[1]


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file with profile support
    
    Args:
        config_path: Path to config file (optional)
        
    Returns:
        Configuration dictionary
    """
    try:
        # Determine which config to load based on environment
        if config_path is None:
            env_profile = os.getenv('SECUREX_ENV', 'development').lower()
            
            if env_profile == 'production':
                config_path = "config_production.yaml"
            elif env_profile == 'development':
                config_path = "config_development.yaml"
            else:
                config_path = "config.yaml"  # Fallback to original
            
            logger.info(f"Environment profile: {env_profile}")
        
        # Resolve relative config paths from project root so launching from a
        # different working directory still works.
        cfg_path = Path(config_path)
        if not cfg_path.is_absolute():
            cfg_path = _get_project_root() / cfg_path

        # Load the configuration file
        if cfg_path.exists():
            with open(cfg_path, 'r') as f:
                config = yaml.safe_load(f)
            
            logger.info(f"[OK] Configuration loaded from {cfg_path}")
            
            # Log security mode for transparency
            dev_mode = config.get('system', {}).get('development_mode', False)
            bypass = config.get('system', {}).get('bypass_anti_spoofing', False)
            
            if dev_mode or bypass:
                logger.warning("[WARNING] DEVELOPMENT MODE - Security bypasses enabled")
            else:
                logger.info("[SECURE] PRODUCTION MODE - Full security enabled")
            
            return config
        else:
            logger.warning(f"Config file {cfg_path} not found, using default")
            return get_default_config()
        
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        # Return default config
        return get_default_config()


def get_default_config() -> Dict[str, Any]:
    """Get default configuration"""
    return {
        'app': {
            'name': 'SecureX-Assist',
            'version': '1.0.0',
            'theme': 'sci-fi'
        },
        'security': {
            'voice_threshold': 0.1,  # legacy, not used
            'voice_similarity_threshold': 0.50,  # new, for cosine similarity
            'session_timeout': 3600,
            'max_login_attempts': 3,
            'require_liveness': True,
            'multi_factor': True,
            'bypass_anti_spoofing': False
        },
        'audio': {
            'sample_rate': 16000,
            'channels': 1,
            'chunk_size': 1024,
            'recording_duration': 5,
            'min_speech_duration': 2.5,
            'min_audio_energy': 0.0005,
            'vad_aggressiveness': 1
        },
        'models': {
            'primary_embedding': 'pyannote/embedding',
            'fallback_embedding': 'speechbrain/spkrec-ecapa-voxceleb',
            'vad_model': 'silero-vad'
        },
        'database': {
            'path': 'securex_db.sqlite',
            'backup_enabled': True
        },
        'ui': {
            'window_width': 1000,
            'window_height': 700,
            'theme_color': '#00ff41',
            'animation_speed': 300
        },
        'tts': {
            'enabled': True,
            'rate': 1.0,
            'rate_wpm': 170,
            'volume': 0.9,
            'short_mode': True,
            'default_blocking': False
        }
    }


def load_environment():
    """Load environment variables from .env file"""
    env_path = _get_project_root() / '.env'
    
    if env_path.exists():
        load_dotenv(env_path)
        logger.info("Environment variables loaded from .env")
        logger.debug(
            "Cloud keys present: GROQ_API_KEY=%s GEMINI_API_KEY=%s",
            bool(os.getenv("GROQ_API_KEY", "").strip()),
            bool(os.getenv("GEMINI_API_KEY", "").strip()),
        )
    else:
        logger.warning(".env file not found. Using default settings.")
        
        # Check for .env.example
        example_path = _get_project_root() / '.env.example'
        if example_path.exists():
            logger.info("Found .env.example - copy it to .env and configure")


def get_env_variable(key: str, default: Any = None) -> Any:
    """Get environment variable with fallback"""
    return os.getenv(key, default)


def setup_logging(log_level: str = "INFO", log_file: str = "securex.log"):
    """
    Setup logging configuration
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Log file path
    """
    # Create logs directory if needed
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create handlers with UTF-8 encoding
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    
    # Console handler with UTF-8 encoding for Windows
    import sys
    stream_handler = logging.StreamHandler(sys.stdout)
    if sys.platform == 'win32':
        # Try to set console to UTF-8 mode
        try:
            import codecs
            sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        except:
            pass
    
    # Set formatter for both handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    
    logging.basicConfig(
        level=numeric_level,
        handlers=[file_handler, stream_handler]
    )

    # Keep third-party internals quiet so real app issues are visible.
    logging.getLogger("numba").setLevel(logging.WARNING)
    logging.getLogger("numba.core").setLevel(logging.WARNING)
    logging.getLogger("flet").setLevel(logging.INFO)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("fsspec").setLevel(logging.WARNING)
    
    logger.info(f"Logging initialized: level={log_level}, file={log_file}")


def create_temp_directory(base_dir: str = "temp_audio") -> Path:
    """Create temporary directory for audio files"""
    temp_dir = Path(base_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir


def cleanup_temp_files(directory: str = "temp_audio"):
    """Clean up temporary audio files"""
    try:
        temp_dir = Path(directory)
        if temp_dir.exists():
            for file in temp_dir.glob("*.wav"):
                file.unlink()
            logger.info(f"Cleaned up temporary files in {directory}")
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def format_timestamp(timestamp: float) -> str:
    """Format Unix timestamp to readable date/time"""
    from datetime import datetime
    dt = datetime.fromtimestamp(timestamp)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


class ColorTheme:
    """Modern UI color theme"""

    # Primary colors
    PRIMARY = "#2962ff"  # A vibrant blue
    PRIMARY_DARK = "#0039cb"
    PRIMARY_LIGHT = "#768fff"

    # Accent color
    ACCENT = "#ffab40"  # A warm orange

    # Neutral colors
    BACKGROUND = "#121212"  # Dark background
    SURFACE = "#1e1e1e"  # Slightly lighter surface
    TEXT_PRIMARY = "#ffffff"  # White text
    TEXT_SECONDARY = "#bdbdbd"  # Gray text
    BORDER = "#37474f"  # Dark gray border

    # Status colors
    SUCCESS = "#4caf50"  # Green
    INFO = "#29b6f6"  # Light blue
    WARNING = "#ffc107"  # Amber
    ERROR = "#f44336"  # Red

    # Matrix theme colors
    MATRIX_GREEN = "#00ff41"  # Bright matrix green
    CYBER_BLUE = "#00bfff"    # Bright cyber blue
    DARK_GREEN = "#004d1a"    # Dark green for backgrounds

    # Glow effects
    GLOW_PRIMARY = "0px 0px 10px #2962ff"
    GLOW_ACCENT = "0px 0px 10px #ffab40"
    GLOW_ERROR = "0px 0px 10px #f44336"
