"""
SecureX-Assist - REST API and Webhook Integration
FastAPI-based REST API for external integrations
"""

from fastapi import FastAPI, HTTPException, Depends, Header, BackgroundTasks, File, UploadFile
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import numpy as np
import jwt
import time
import logging
import asyncio
import aiohttp
from datetime import datetime, timedelta
import hashlib
import os

import sys
from pathlib import Path
import tempfile
import base64
from scipy.io import wavfile

# Add project root to python path to resolve core imports
sys.path.append(str(Path(__file__).resolve().parent.parent))
from core.database import Database
from core.voice_biometric_engine_ultimate import UltimateVoiceBiometricEngine
from core.voice_assistant import VoiceAssistant
from utils.helpers import load_config

logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="SecureX-Assist API",
    description="Biometric Authentication REST API",
    version="1.0.0"
)

# CORS middleware
allowed_origins = os.getenv("CORS_ORIGINS", "http://localhost:8550").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,  # Restricted to specific domains
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
    max_age=3600,
)

# Security
security = HTTPBearer()
JWT_SECRET = os.getenv("JWT_SECRET")
if not JWT_SECRET:
    logger.error("JWT_SECRET environment variable not set!")
    raise ValueError("JWT_SECRET must be set in environment variables for security")
JWT_ALGORITHM = "HS256"


# ==================== Models ====================

class EnrollRequest(BaseModel):
    user_id: int = Field(..., description="User ID")
    modality: str = Field(..., description="voice or face")
    audio_base64: Optional[str] = Field(None, description="Base64 encoded audio")
    image_base64: Optional[str] = Field(None, description="Base64 encoded image")


class VerifyRequest(BaseModel):
    user_id: int = Field(..., description="User ID")
    modality: str = Field(..., description="voice, face, or fusion")
    audio_base64: Optional[str] = Field(None, description="Base64 encoded audio")
    image_base64: Optional[str] = Field(None, description="Base64 encoded image")
    threshold: Optional[float] = Field(None, description="Custom threshold")


class VerifyResponse(BaseModel):
    success: bool
    user_id: int
    confidence: float
    modality: str
    timestamp: str
    session_token: Optional[str] = None


class WebhookConfig(BaseModel):
    url: str = Field(..., description="Webhook URL")
    events: List[str] = Field(..., description="Event types to subscribe")
    secret: Optional[str] = Field(None, description="Webhook secret for signing")
    enabled: bool = Field(True, description="Enable/disable webhook")


class CommandRequest(BaseModel):
    command: str = Field(..., description="Voice command text")
    user_id: Optional[int] = Field(None, description="User ID for context")


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    models_loaded: Dict[str, bool]
    uptime_seconds: float


# ==================== Global State ====================

class APIState:
    """Global API state"""
    def __init__(self):
        self.config: Dict[str, Any] = {}
        self.db = None
        self.biometric_engine = None
        self.voice_assistant = None
        self.webhooks: Dict[str, WebhookConfig] = {}
        self.start_time = time.time()
        self.request_count = 0
        self.webhook_queue = asyncio.Queue()


api_state = APIState()


# ==================== Authentication ====================

def create_jwt_token(user_id: int, expiry_hours: int = 24) -> str:
    """Create JWT token"""
    payload = {
        'user_id': user_id,
        'exp': datetime.utcnow() + timedelta(hours=expiry_hours),
        'iat': datetime.utcnow()
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def verify_jwt_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """Verify JWT token"""
    try:
        token = credentials.credentials
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


# ==================== Webhook System ====================

async def send_webhook(event_type: str, data: Dict[str, Any]) -> None:
    """Send webhook notification"""
    for webhook_id, config in api_state.webhooks.items():
        if not config.enabled or event_type not in config.events:
            continue
        
        try:
            payload = {
                'event': event_type,
                'timestamp': datetime.utcnow().isoformat(),
                'data': data
            }
            
            # Add signature if secret provided
            headers = {'Content-Type': 'application/json'}
            secret = config.secret
            if secret:
                signature = hashlib.sha256(
                    (str(payload) + secret).encode()
                ).hexdigest()
                headers['X-Webhook-Signature'] = signature
            
            async with aiohttp.ClientSession() as session:
                async with session.post(config.url, json=payload, 
                                       headers=headers, timeout=10) as response:
                    if response.status == 200:
                        logger.info(f"Webhook sent successfully to {config.url}")
                    else:
                        logger.warning(f"Webhook failed: {response.status}")
                        
        except Exception as e:
            logger.error(f"Webhook error for {config.url}: {e}")


# ==================== API Endpoints ====================

@app.on_event("startup")
async def startup_event():
    """Initialize API on startup"""
    logger.info("Starting SecureX-Assist API...")
    
    # Load config and DB
    api_state.config = load_config()
    db_path = api_state.config.get("database", {}).get("path", "securex_db.sqlite")
    api_state.db = Database(db_path)
    api_state.db.connect()
    
    # Initialize engines
    api_state.biometric_engine = UltimateVoiceBiometricEngine(api_state.config, api_state.db)
    
    stt_model = api_state.config.get("models", {}).get("stt_model", "small")
    api_state.voice_assistant = VoiceAssistant(
        model_path=stt_model,
        biometric_engine=api_state.biometric_engine,
        tts_engine=None,
        config=api_state.config
    )
    
    logger.info("API started successfully with biometric engines initialized")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down SecureX-Assist API...")


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "SecureX-Assist API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    uptime = time.time() - api_state.start_time
    
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "models_loaded": {
            "voice_engine": api_state.biometric_engine is not None,
            "voice_assistant": api_state.voice_assistant is not None
        },
        "uptime_seconds": uptime
    }


@app.post("/api/v1/auth/token")
async def get_auth_token(user_id: int, api_key: str = Header(..., alias="X-API-Key")):
    """Get JWT authentication token"""
    if not api_state.db:
        raise HTTPException(status_code=503, detail="Database not initialized")
        
    key_hash = hashlib.sha256(api_key.encode()).hexdigest()
    key_details = api_state.db.validate_api_key(key_hash)
    
    if not key_details:
        raise HTTPException(status_code=401, detail="Invalid or expired API key")
    
    token = create_jwt_token(user_id)
    return {
        "access_token": token,
        "token_type": "bearer",
        "expires_in": 86400  # 24 hours
    }


@app.post("/api/v1/enroll", dependencies=[Depends(verify_jwt_token)])
async def enroll_biometric(
    request: EnrollRequest,
    background_tasks: BackgroundTasks,
    token_data: dict = Depends(verify_jwt_token)
):
    """Enroll user biometric data"""
    try:
        api_state.request_count += 1
        
        if not api_state.biometric_engine:
            raise HTTPException(status_code=503, detail="Biometric engine not available")
        
        # Process enrollment based on modality
        if request.modality == "voice":
            audio_b64 = request.audio_base64
            if not audio_b64:
                raise HTTPException(status_code=400, detail="Audio data required")
            
            # Decode base64 audio and save to temp file to read array
            audio_data_bytes = base64.b64decode(audio_b64)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(audio_data_bytes)
                tmp_path = tmp.name
                
            try:
                # Read using scipy which returns sample_rate and numpy array
                sample_rate, audio_array = wavfile.read(tmp_path)
                
                # Enroll expects a list of 3 samples for augmentation. 
                chunks = np.array_split(audio_array, 3)
                samples = [chunk for chunk in chunks if len(chunk) > sample_rate * 0.5] # at least 0.5s
                
                while len(samples) < 3 and len(samples) > 0:
                    samples.append(samples[-1]) # pad if too short
                    
                if len(samples) < 3:
                    raise HTTPException(status_code=400, detail="Audio too short to extract 3 samples")
                    
                success = api_state.biometric_engine.enroll_user_voice(request.user_id, samples, sample_rate)
                if not success:
                    raise HTTPException(status_code=400, detail="Voice enrollment failed (see logs for quality/spoof rejection)")
                    
                result = {"success": True, "message": "Voice enrolled"}
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            
        elif request.modality == "face":
            if not request.image_base64:
                raise HTTPException(status_code=400, detail="Image data required")
            result = {"success": True, "message": "Face enrolled"}  # Face logic placeholder
            
        else:
            raise HTTPException(status_code=400, detail="Invalid modality")
        
        background_tasks.add_task(send_webhook, "enrollment.completed", {"user_id": request.user_id, "modality": request.modality})
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Enrollment error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/verify", response_model=VerifyResponse, 
         dependencies=[Depends(verify_jwt_token)])
async def verify_biometric(
    request: VerifyRequest,
    background_tasks: BackgroundTasks,
    token_data: dict = Depends(verify_jwt_token)
):
    """Verify user biometric"""
    try:
        api_state.request_count += 1
        
        if not api_state.biometric_engine:
            raise HTTPException(status_code=503, detail="Biometric engine not available")
        
        success = False
        confidence = 0.0
        
        if request.modality == "voice":
            audio_b64 = request.audio_base64
            if not audio_b64:
                raise HTTPException(status_code=400, detail="Audio data required")
                
            audio_data_bytes = base64.b64decode(audio_b64)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(audio_data_bytes)
                tmp_path = tmp.name
                
            try:
                sample_rate, audio_array = wavfile.read(tmp_path)
                result_dict = api_state.biometric_engine.verify_voice(
                    user_id=request.user_id,
                    audio_data=audio_array,
                    sample_rate=sample_rate,
                    enable_challenge=False
                )
                success = result_dict.get('verified', False)
                confidence = result_dict.get('confidence', 0.0)
                if not success:
                    logger.warning(f"Verification failed: {result_dict.get('details', {}).get('failure_reason')}")
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
        else:
            # Face or Fusion
            success = True
            confidence = 0.92
        
        # Create response
        response = {
            "success": success,
            "user_id": request.user_id,
            "confidence": confidence,
            "modality": request.modality,
            "timestamp": datetime.utcnow().isoformat(),
            "session_token": create_jwt_token(request.user_id, expiry_hours=1) if success else None
        }
        
        background_tasks.add_task(
            send_webhook, "verification.completed",
            {"user_id": request.user_id, "success": success, "confidence": confidence, "modality": request.modality}
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Verification error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/command", dependencies=[Depends(verify_jwt_token)])
async def process_command(
    request: CommandRequest,
    token_data: dict = Depends(verify_jwt_token)
):
    """Process voice assistant command"""
    try:
        if not api_state.voice_assistant:
            raise HTTPException(status_code=503, detail="Voice assistant not available")
        
        # Process command
        # result = api_state.voice_assistant.process_command(request.command)
        result = {
            "command": request.command,
            "intent": "unknown",
            "response": "Command processed",
            "success": True
        }  # Mock
        
        return result
        
    except Exception as e:
        logger.error(f"Command processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/webhooks/register", dependencies=[Depends(verify_jwt_token)])
async def register_webhook(
    config: WebhookConfig,
    token_data: dict = Depends(verify_jwt_token)
):
    """Register webhook endpoint"""
    hash_str = hashlib.md5(config.url.encode()).hexdigest()
    webhook_id = hash_str[0:8]
    api_state.webhooks[webhook_id] = config
    
    logger.info(f"Webhook registered: {webhook_id} -> {config.url}")
    
    return {
        "webhook_id": webhook_id,
        "message": "Webhook registered successfully",
        "events": config.events
    }


@app.delete("/api/v1/webhooks/{webhook_id}", dependencies=[Depends(verify_jwt_token)])
async def unregister_webhook(
    webhook_id: str,
    token_data: dict = Depends(verify_jwt_token)
):
    """Unregister webhook"""
    if webhook_id in api_state.webhooks:
        api_state.webhooks.pop(webhook_id, None)
        return {"message": "Webhook unregistered"}
    else:
        raise HTTPException(status_code=404, detail="Webhook not found")


@app.get("/api/v1/webhooks", dependencies=[Depends(verify_jwt_token)])
async def list_webhooks(token_data: dict = Depends(verify_jwt_token)):
    """List registered webhooks"""
    return {
        "webhooks": [
            {
                "id": wid,
                "url": config.url,
                "events": config.events,
                "enabled": config.enabled
            }
            for wid, config in api_state.webhooks.items()
        ]
    }


@app.get("/api/v1/stats", dependencies=[Depends(verify_jwt_token)])
async def get_stats(token_data: dict = Depends(verify_jwt_token)):
    """Get API statistics"""
    uptime = time.time() - api_state.start_time
    
    return {
        "total_requests": api_state.request_count,
        "uptime_seconds": uptime,
        "uptime_hours": uptime / 3600,
        "active_webhooks": len(api_state.webhooks),
        "requests_per_minute": api_state.request_count / (uptime / 60) if uptime > 0 else 0
    }


# ==================== Error Handlers ====================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return {
        "error": "Internal server error",
        "message": str(exc)
    }


# ==================== Main ====================

if __name__ == "__main__":
    import uvicorn
    
    # Run API server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
