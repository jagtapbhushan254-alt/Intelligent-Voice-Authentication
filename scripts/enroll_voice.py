"""
Voice Enrollment Script
Enroll a user's voice for biometric authentication
"""

import sys
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from core.database import Database
from core.voice_engine import VoiceEngine
from core.audio_processor import AudioRecorder, AudioProcessor
from utils.helpers import load_config, setup_logging, create_temp_directory

logger = logging.getLogger(__name__)


def enroll_voice():
    """Interactive voice enrollment"""
    
    print("\n" + "=" * 60)
    print("🎤 VOICE ENROLLMENT - SecureX-Assist")
    print("=" * 60)
    
    # Load config
    config = load_config()
    setup_logging(log_level="INFO")
    
    # Initialize components
    db = Database(config['database']['path'])
    db.connect()
    db.initialize_schema()
    
    voice_engine = VoiceEngine(config)
    audio_recorder = AudioRecorder(config)
    
    # Get username
    print("\n📝 Enter username to enroll:")
    username = input("> ").strip()
    
    if not username:
        print("❌ Username cannot be empty")
        return
    
    # Check if user exists
    user = db.get_user_by_username(username)
    if not user:
        print(f"❌ User '{username}' not found in database")
        print("\nCreate this user first using the database initialization script.")
        return
    
    user_id = user['id']
    print(f"✅ Found user: {username} (ID: {user_id})")
    
    # Load AI models
    print("\n🤖 Loading AI models...")
    if not voice_engine.load_models():
        print("❌ Failed to load AI models")
        return
    
    print(f"✅ Using {voice_engine.active_backend} backend")
    
    # Create temp directory
    temp_dir = create_temp_directory()
    
    # Record voice samples
    print("\n" + "=" * 60)
    print("🎙️  VOICE RECORDING")
    print("=" * 60)
    print("\n📢 Instructions:")
    print("1. Find a quiet location")
    print("2. Speak naturally and clearly")
    print("3. Say anything - read text, count numbers, etc.")
    print("4. Duration: 5 seconds")
    print("\n⚠️  Recording will start immediately after you press Enter")
    
    input("\nPress Enter when ready...")
    
    # Record audio
    print("\n🔴 RECORDING... Speak now!")
    audio_data = audio_recorder.record_audio(duration=5.0)
    
    if audio_data is None:
        print("❌ Recording failed")
        return
    
    print("✅ Recording complete")
    
    # Save audio
    audio_path = temp_dir / f"enroll_{user_id}.wav"
    if not audio_recorder.save_audio(audio_data, str(audio_path)):
        print("❌ Failed to save audio")
        return
    
    # Process audio
    print("\n🔧 Processing audio...")
    processed_audio = AudioProcessor.normalize(audio_data)
    processed_audio = AudioProcessor.trim_silence(processed_audio)

    # Extract voice embedding
    print("🧠 Extracting voice biometric signature...")
    embedding = voice_engine.extract_embedding(str(audio_path))
    
    if embedding is None:
        print("❌ Failed to extract voice embedding")
        return
    
    print(f"✅ Embedding extracted: shape={embedding.shape}")
    
    # Check quality
    quality_score = voice_engine.calculate_embedding_quality(embedding)
    print(f"📊 Quality score: {quality_score:.2%}")
    
    if quality_score < 0.5:
        print("⚠️  Low quality embedding. Consider re-recording in a quieter environment.")
    
    # Store in database
    print("\n💾 Saving to database...")
    
    # Deactivate old embeddings
    db.deactivate_old_embeddings(user_id)
    
    # Store new embedding
    embedding_id = db.store_voice_embedding(
        user_id=user_id,
        embedding=embedding,
        embedding_type=voice_engine.active_backend,
        quality_score=quality_score
    )
    
    if embedding_id:
        print(f"✅ Voice profile saved (ID: {embedding_id})")
        print("\n" + "=" * 60)
        print("🎉 ENROLLMENT COMPLETE!")
        print("=" * 60)
        print(f"\nUser '{username}' can now login with voice authentication.")
        print("\n💡 Tips:")
        print("- Speak naturally during login")
        print("- Use similar environment (noise level)")
        print("- Re-enroll if having authentication issues")
    else:
        print("❌ Failed to save voice profile")
    
    # Cleanup
    try:
        audio_path.unlink()
    except:
        pass


if __name__ == "__main__":
    try:
        enroll_voice()
    except KeyboardInterrupt:
        print("\n\n⚠️  Enrollment cancelled by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        logger.error(f"Enrollment error: {e}", exc_info=True)
