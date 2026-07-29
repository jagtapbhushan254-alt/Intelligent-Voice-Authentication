"""
SecureX-Assist - Main Entry Point
Voice Biometric Authentication System
"""

import sys
import os
import logging
from pathlib import Path

# Disable HuggingFace symlinks to prevent WinError 1314 on Windows without Admin rights
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['HF_HUB_DISABLE_SYMLINKS'] = '1'

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.helpers import load_config, load_environment, setup_logging
import flet as ft


def main():
    """Main application entry point"""
    
    # Load environment variables
    load_environment()
    
    # Load configuration
    config = load_config()
    
    # Setup logging
    log_level = config.get('app', {}).get('log_level', 'INFO')
    setup_logging(log_level=log_level)
    
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("SECUREX-ASSIST - Voice Biometric Authentication System")
    logger.info(f"Version: {config.get('app', {}).get('version', '1.0.0')}")
    logger.info("=" * 60)
    
    try:
        # Import and run Flet app
        from ui.app import main as app_main
        
        logger.info("Starting Flet application...")
        
        ft.app(
            target=app_main,
            assets_dir="assets" if Path("assets").exists() else None,
        )
        
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
        print("\n⚠️  Troubleshooting:")
        print("1. Ensure all dependencies are installed: pip install -r requirements.txt")
        print("2. Check that .env file is configured with HF_TOKEN")
        print("3. Verify microphone access permissions")
        print("4. See logs for detailed error information")
        sys.exit(1)
    finally:
        logger.info("Application shutdown complete")


if __name__ == "__main__":
    main()
