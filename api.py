import os
import sys
import io
import logging
from pathlib import Path
from typing import Optional
from urllib.parse import quote
import numpy as np
import soundfile as sf
import torch
import librosa
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add CosyVoice to path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/CosyVoice'.format(ROOT_DIR))
sys.path.append('{}/CosyVoice/third_party/Matcha-TTS'.format(ROOT_DIR))

from CosyVoice.cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from CosyVoice.cosyvoice.utils.file_utils import load_wav

# Initialize FastAPI app
app = FastAPI(title="CosyVoice OpenAI-Compatible TTS API", version="1.0.0")


def setup_cors(app: FastAPI, allow_origins: list = None):
    """
    Setup CORS middleware for the FastAPI app.
    
    Args:
        app: FastAPI application instance
        allow_origins: List of allowed origins. If None or ["*"], allows all origins.
    """
    if allow_origins is None:
        allow_origins = ["*"]
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    logger.info(f"CORS enabled with origins: {allow_origins}")


# Global model instance (lazy loading)
cosyvoice_model: Optional[CosyVoice] = None
VOICES_DIR = Path(__file__).parent / "voices"
MODEL_DIR = Path(__file__).parent / "CosyVoice" / "pretrained_models" / "CosyVoice2-0.5B"
MAX_VAL = 0.8
PROMPT_SR = 16000


class TTSRequest(BaseModel):
    model: Optional[str] = Field(None, description="Model name (ignored)")
    input: str = Field(..., max_length=4096, description="Text to synthesize")
    voice: str = Field(..., description="Voice name from voices directory")
    instructions: Optional[str] = Field(None, description="Additional instructions for instruct mode")
    response_format: Optional[str] = Field("wav", description="Audio format")
    speed: Optional[float] = Field(1.0, description="Speed control (0.5-2.0)")
    stream_format: Optional[str] = Field("audio", description="Stream format: audio or sse")


def postprocess(speech, top_db=60, hop_length=220, win_length=440):
    """Post-process generated audio"""
    speech, _ = librosa.effects.trim(
        speech, top_db=top_db,
        frame_length=win_length,
        hop_length=hop_length
    )
    if speech.abs().max() > MAX_VAL:
        speech = speech / speech.abs().max() * MAX_VAL
    speech = torch.concat([speech, torch.zeros(1, int(cosyvoice_model.sample_rate * 0.2))], dim=1)
    return speech


def get_cosyvoice_model() -> CosyVoice:
    """Lazy load CosyVoice model."""
    global cosyvoice_model
    if cosyvoice_model is None:
        logger.info("Loading CosyVoice model...")
        
        if not MODEL_DIR.exists():
            raise RuntimeError(f"Model directory not found: {MODEL_DIR}")
        
        # Check if it's CosyVoice2
        if 'CosyVoice2' in str(MODEL_DIR):
            cosyvoice_model = CosyVoice2(str(MODEL_DIR))
        else:
            cosyvoice_model = CosyVoice(str(MODEL_DIR))
        
        logger.info("CosyVoice model loaded successfully")
    
    return cosyvoice_model


def load_voice(voice_name: str) -> tuple[str, str]:
    """
    Load voice files (wav + txt) from voices directory.
    
    Args:
        voice_name: Name of the voice (without extension)
    
    Returns:
        Tuple of (prompt_wav_path, prompt_text)
    
    Raises:
        HTTPException: If voice files not found
    """
    wav_path = VOICES_DIR / f"{voice_name}.wav"
    txt_path = VOICES_DIR / f"{voice_name}.txt"
    
    if not wav_path.exists() or not txt_path.exists():
        available_voices = sorted(set(
            f.stem for f in VOICES_DIR.glob("*.wav")
            if (VOICES_DIR / f"{f.stem}.txt").exists()
        ))
        raise HTTPException(
            status_code=400,
            detail=f"Voice '{voice_name}' not found. Available voices: {', '.join(available_voices)}"
        )
    
    # Read prompt text
    with open(txt_path, "r", encoding="utf-8") as f:
        prompt_text = f.read().strip()
    
    return str(wav_path), prompt_text


def verify_api_key(authorization: Optional[str]) -> None:
    """
    Verify API key if OPENAI_API_KEY environment variable is set.
    
    Args:
        authorization: Authorization header value
    
    Raises:
        HTTPException: If authentication fails
    """
    expected_key = os.environ.get("OPENAI_API_KEY", "").strip()
    
    # If no key is set, skip authentication
    if not expected_key:
        return
    
    # Check authorization header
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Missing Authorization header"
        )
    
    # Extract bearer token
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Invalid Authorization header format. Expected 'Bearer <token>'"
        )
    
    provided_key = authorization[7:]  # Remove "Bearer " prefix
    
    if provided_key != expected_key:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )


def convert_audio_format(wav_data: np.ndarray, sample_rate: int, format: str) -> bytes:
    """
    Convert audio data to specified format.
    
    Args:
        wav_data: Audio waveform as numpy array
        sample_rate: Sample rate in Hz
        format: Target format (mp3, opus, aac, flac, wav, pcm)
    
    Returns:
        Audio data as bytes
    """
    buffer = io.BytesIO()
    
    # Map format to soundfile subtype
    format_map = {
        "wav": "WAV",
        "flac": "FLAC",
        "pcm": "WAV",  # PCM is just raw WAV
    }
    
    if format in format_map:
        sf.write(buffer, wav_data, sample_rate, format=format_map[format])
        buffer.seek(0)
        return buffer.read()
    elif format in ["mp3", "opus", "aac"]:
        # For MP3/OPUS/AAC, we'd need ffmpeg or similar
        # For simplicity, default to WAV
        logger.warning(f"Format '{format}' not directly supported, returning WAV")
        sf.write(buffer, wav_data, sample_rate, format="WAV")
        buffer.seek(0)
        return buffer.read()
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format: {format}"
        )


def generate_audio_chunks(wav_data: np.ndarray, sample_rate: int, format: str, chunk_size: int = 4096):
    """
    Generate audio data in chunks for streaming.
    
    Args:
        wav_data: Audio waveform as numpy array
        sample_rate: Sample rate in Hz
        format: Target format
        chunk_size: Size of each chunk in bytes
    
    Yields:
        Audio data chunks
    """
    # Convert to target format
    audio_bytes = convert_audio_format(wav_data, sample_rate, format)
    
    # Stream in chunks
    for i in range(0, len(audio_bytes), chunk_size):
        yield audio_bytes[i:i + chunk_size]


@app.post("/v1/audio/speech")
async def create_speech(
    request: TTSRequest,
    authorization: Optional[str] = Header(None),
    x_mode: Optional[str] = Header(None, alias="X-Mode"),
    x_stream_inference: Optional[str] = Header(None, alias="X-Stream-Inference")
):
    """
    Generate speech from text using CosyVoice.
    
    OpenAI-compatible endpoint for text-to-speech synthesis.
    
    Custom Headers:
    - X-Mode: Inference mode - "zero_shot" (default), "cross_lingual", or "instruct"
    - X-Stream-Inference: "True" to enable streaming inference, "False" to disable (default: False)
    """
    try:
        # Verify API key if required
        verify_api_key(authorization)
        
        # Validate input
        if not request.input or not request.input.strip():
            raise HTTPException(status_code=400, detail="Input text is required")
        
        # Parse mode
        # If X-Mode header is set, use it; otherwise auto-detect based on instructions
        if x_mode:
            mode = x_mode.lower()
        elif request.instructions and request.instructions.strip():
            # Automatically use instruct mode if instructions are provided
            mode = "instruct"
        else:
            mode = "zero_shot"
        
        if mode not in ["zero_shot", "cross_lingual", "instruct"]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid X-Mode: {mode}. Must be 'zero_shot', 'cross_lingual', or 'instruct'"
            )
        
        # Parse stream inference
        stream_inference = x_stream_inference and x_stream_inference.lower() == "true"
        
        # Validate speed
        if request.speed < 0.5 or request.speed > 2.0:
            raise HTTPException(
                status_code=400,
                detail="Speed must be between 0.5 and 2.0"
            )
        
        logger.info(f"Processing with mode={mode}, stream_inference={stream_inference}, speed={request.speed}")
        
        # Load voice files
        prompt_wav_path, prompt_text = load_voice(request.voice)
        
        # Get model
        model = get_cosyvoice_model()
        
        # Load and preprocess prompt audio
        prompt_speech_16k = postprocess(load_wav(prompt_wav_path, PROMPT_SR))
        
        # Generate audio based on mode
        logger.info(f"Generating audio for voice '{request.voice}': {request.input[:60]}...")
        
        result_audio = None
        
        if mode == "zero_shot":
            # 3s极速复刻 - Zero-shot voice cloning
            for i in model.inference_zero_shot(
                request.input, 
                prompt_text, 
                prompt_speech_16k, 
                stream=stream_inference, 
                speed=request.speed
            ):
                audio = i['tts_speech'].numpy().flatten()
                if result_audio is None:
                    result_audio = audio
                else:
                    result_audio = np.concatenate([result_audio, audio])
                    
        elif mode == "cross_lingual":
            # 跨语种复刻 - Cross-lingual voice cloning
            for i in model.inference_cross_lingual(
                request.input, 
                prompt_speech_16k, 
                stream=stream_inference, 
                speed=request.speed
            ):
                audio = i['tts_speech'].numpy().flatten()
                if result_audio is None:
                    result_audio = audio
                else:
                    result_audio = np.concatenate([result_audio, audio])
                    
        elif mode == "instruct":
            # 自然语言控制 - Instruct mode
            if not request.instructions or not request.instructions.strip():
                raise HTTPException(
                    status_code=400,
                    detail="Instructions are required for instruct mode"
                )
            
            # Check if model supports instruct2 (CosyVoice2)
            if hasattr(model, 'inference_instruct2'):
                for i in model.inference_instruct2(
                    request.input, 
                    request.instructions, 
                    prompt_speech_16k, 
                    stream=stream_inference, 
                    speed=request.speed
                ):
                    audio = i['tts_speech'].numpy().flatten()
                    if result_audio is None:
                        result_audio = audio
                    else:
                        result_audio = np.concatenate([result_audio, audio])
            else:
                raise HTTPException(
                    status_code=400,
                    detail="Instruct mode is not supported by the current model"
                )
        
        if result_audio is None:
            raise HTTPException(status_code=500, detail="Audio generation failed")
        
        # Get sample rate
        sample_rate = model.sample_rate
        
        # Determine response format
        response_format = request.response_format.lower()
        
        # Map MIME types
        mime_types = {
            "mp3": "audio/mpeg",
            "opus": "audio/opus",
            "aac": "audio/aac",
            "flac": "audio/flac",
            "wav": "audio/wav",
            "pcm": "audio/pcm",
        }
        media_type = mime_types.get(response_format, "audio/wav")
        
        # Return streaming response
        if request.stream_format == "audio" or request.stream_format is None:
            return StreamingResponse(
                generate_audio_chunks(result_audio, sample_rate, response_format),
                media_type=media_type,
                headers={
                    "Content-Disposition": f"attachment; filename=speech.{response_format}",
                    "Transfer-Encoding": "chunked"
                }
            )
        elif request.stream_format == "sse":
            # SSE format not fully implemented, return error
            raise HTTPException(
                status_code=400,
                detail="SSE stream format is not supported in this implementation"
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid stream_format: {request.stream_format}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating speech: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/v1/voices")
async def list_voices(request: Request):
    """List available voices with preview URLs."""
    available_voices = sorted(set(
        f.stem for f in VOICES_DIR.glob("*.wav")
        if (VOICES_DIR / f"{f.stem}.txt").exists()
    ))
    
    # Build base URL from request
    base_url = str(request.base_url).rstrip('/')
    
    return {
        "voices": [
            {
                "id": voice,
                "name": voice,
                "preview_url": f"{base_url}/v1/voices/{quote(voice, safe='')}/preview"
            }
            for voice in available_voices
        ]
    }


@app.get("/v1/voices/{voice_name}/preview")
async def get_voice_preview(voice_name: str):
    """
    Get preview audio for a specific voice.
    
    Args:
        voice_name: Name of the voice (without extension)
    
    Returns:
        Audio file for preview
    
    Raises:
        HTTPException 404: If voice file not found
    """
    wav_path = VOICES_DIR / f"{voice_name}.wav"
    
    if not wav_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Voice preview not found: {voice_name}"
        )
    
    return FileResponse(
        path=wav_path,
        media_type="audio/wav",
        filename=f"{voice_name}.wav"
    )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="CosyVoice OpenAI-Compatible TTS API")
    parser.add_argument("--host", type=str, default=None, help="Host to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=None, help="Port to bind (default: 8000)")
    parser.add_argument("--model-dir", type=str, default=None, help="Model directory path (default: CosyVoice/pretrained_models/CosyVoice2-0.5B)")
    parser.add_argument("--allow-cors", action="store_true", help="Enable CORS support for cross-origin requests")
    parser.add_argument("--cors-origins", type=str, default="*", help="Comma-separated list of allowed CORS origins (default: *)")
    args = parser.parse_args()
    
    # Update model directory if provided
    if args.model_dir:
        MODEL_DIR = Path(args.model_dir)
    
    # Get host and port from args or environment
    host = args.host or os.environ.get("HOST", "0.0.0.0")
    port = args.port or int(os.environ.get("PORT", 8000))
    
    # Setup CORS if requested
    if args.allow_cors:
        if args.cors_origins == "*":
            setup_cors(app, ["*"])
        else:
            origins = [origin.strip() for origin in args.cors_origins.split(",")]
            setup_cors(app, origins)
    
    logger.info(f"Starting CosyVoice OpenAI-Compatible TTS API on {host}:{port}")
    if args.allow_cors:
        logger.info(f"CORS is enabled")
    uvicorn.run(app, host=host, port=port)

