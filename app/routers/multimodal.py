"""
Multimodal Module for MCP Server

This module handles multimodal capabilities:
- Speech-to-text (using Whisper)
- Text-to-speech (using Piper or ElevenLabs)
- Image processing and generation
"""

import os
import logging
import json
import tempfile
import base64
from io import BytesIO
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException, Depends, File, UploadFile, Form
import subprocess
from PIL import Image

# Initialize router
router = APIRouter()

# Configure logging
logger = logging.getLogger("MCPServer.Multimodal")

# Check for Whisper availability
WHISPER_AVAILABLE = False
try:
    # Check if whisper is installed
    result = subprocess.run(["which", "whisper"], capture_output=True, text=True)
    WHISPER_AVAILABLE = result.returncode == 0
    if not WHISPER_AVAILABLE:
        logger.warning("Whisper not found in PATH. Speech-to-text functionality will be limited.")
except Exception as e:
    logger.warning(f"Error checking for Whisper: {e}")

# Check for Piper availability
PIPER_AVAILABLE = False
try:
    # Check if piper is installed
    result = subprocess.run(["which", "piper"], capture_output=True, text=True)
    PIPER_AVAILABLE = result.returncode == 0
    if not PIPER_AVAILABLE:
        logger.warning("Piper not found in PATH. Text-to-speech functionality will be limited.")
except Exception as e:
    logger.warning(f"Error checking for Piper: {e}")

# Check for ElevenLabs API key
ELEVENLABS_AVAILABLE = "MCP_ELEVENLABS_API_KEY" in os.environ
if not ELEVENLABS_AVAILABLE:
    logger.warning("ElevenLabs API key not found. ElevenLabs TTS will not be available.")

# Request models
class SpeechToTextRequest(BaseModel):
    """Request model for speech-to-text"""
    audio_data: str  # Base64 encoded audio data
    model: Optional[str] = "base"  # Whisper model to use

class TextToSpeechRequest(BaseModel):
    """Request model for text-to-speech"""
    text: str
    voice: Optional[str] = "default"
    provider: Optional[str] = "piper"  # piper or elevenlabs

class ImageProcessRequest(BaseModel):
    """Request model for image processing"""
    image_data: str  # Base64 encoded image data
    operation: str  # resize, crop, filter, etc.
    parameters: Optional[Dict[str, Any]] = None

# Speech-to-text function
async def speech_to_text(audio_data: bytes, model: str = "base") -> str:
    """
    Convert speech to text using Whisper
    
    Args:
        audio_data: Raw audio data
        model: Whisper model to use (tiny, base, small, medium, large)
        
    Returns:
        Transcribed text
    """
    if not WHISPER_AVAILABLE:
        raise ValueError("Whisper is not available on this system")
    
    try:
        # Save audio data to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_file.write(audio_data)
            temp_path = temp_file.name
        
        # Run Whisper on the temporary file
        result = subprocess.run(
            ["whisper", temp_path, "--model", model, "--output_format", "txt"],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Read the output file (Whisper creates a .txt file with the same name)
        with open(f"{temp_path}.txt", "r") as f:
            transcription = f.read().strip()
        
        # Clean up temporary files
        os.unlink(temp_path)
        os.unlink(f"{temp_path}.txt")
        
        return transcription
        
    except Exception as e:
        logger.error(f"Error in speech-to-text: {e}")
        raise ValueError(f"Speech-to-text error: {str(e)}")

# Text-to-speech with Piper
async def text_to_speech_piper(text: str, voice: str = "default") -> bytes:
    """
    Convert text to speech using Piper
    
    Args:
        text: Text to convert to speech
        voice: Voice to use
        
    Returns:
        Audio data as bytes
    """
    if not PIPER_AVAILABLE:
        raise ValueError("Piper is not available on this system")
    
    try:
        # Create a temporary file for the output
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name
        
        # Run Piper to generate speech
        voice_arg = [] if voice == "default" else ["--voice", voice]
        process = subprocess.run(
            ["piper", "--output_file", temp_path] + voice_arg,
            input=text.encode(),
            capture_output=True,
            check=True
        )
        
        # Read the output file
        with open(temp_path, "rb") as f:
            audio_data = f.read()
        
        # Clean up temporary file
        os.unlink(temp_path)
        
        return audio_data
        
    except Exception as e:
        logger.error(f"Error in text-to-speech with Piper: {e}")
        raise ValueError(f"Text-to-speech error: {str(e)}")

# Text-to-speech with ElevenLabs
async def text_to_speech_elevenlabs(text: str, voice: str = "default") -> bytes:
    """
    Convert text to speech using ElevenLabs
    
    Args:
        text: Text to convert to speech
        voice: Voice ID to use
        
    Returns:
        Audio data as bytes
    """
    if not ELEVENLABS_AVAILABLE:
        raise ValueError("ElevenLabs API key not found")
    
    try:
        import requests
        
        # Get API key
        api_key = os.environ.get("MCP_ELEVENLABS_API_KEY")
        
        # Use default voice if not specified
        if voice == "default":
            voice = "21m00Tcm4TlvDq8ikWAM"  # Default ElevenLabs voice
        
        # Call ElevenLabs API
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice}"
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": api_key
        }
        data = {
            "text": text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.5
            }
        }
        
        response = requests.post(url, json=data, headers=headers)
        
        if response.status_code != 200:
            raise ValueError(f"ElevenLabs API error: {response.text}")
        
        return response.content
        
    except Exception as e:
        logger.error(f"Error in text-to-speech with ElevenLabs: {e}")
        raise ValueError(f"Text-to-speech error: {str(e)}")

# Image processing function
async def process_image(image_data: bytes, operation: str, parameters: Optional[Dict[str, Any]] = None) -> bytes:
    """
    Process an image
    
    Args:
        image_data: Raw image data
        operation: Operation to perform (resize, crop, filter, etc.)
        parameters: Parameters for the operation
        
    Returns:
        Processed image data
    """
    try:
        # Open the image
        image = Image.open(BytesIO(image_data))
        
        # Default parameters
        if parameters is None:
            parameters = {}
        
        # Perform the requested operation
        if operation == "resize":
            width = parameters.get("width", image.width)
            height = parameters.get("height", image.height)
            image = image.resize((width, height))
            
        elif operation == "crop":
            left = parameters.get("left", 0)
            top = parameters.get("top", 0)
            right = parameters.get("right", image.width)
            bottom = parameters.get("bottom", image.height)
            image = image.crop((left, top, right, bottom))
            
        elif operation == "rotate":
            angle = parameters.get("angle", 0)
            image = image.rotate(angle)
            
        elif operation == "filter":
            filter_name = parameters.get("filter", "BLUR")
            if filter_name == "BLUR":
                from PIL import ImageFilter
                image = image.filter(ImageFilter.BLUR)
            elif filter_name == "CONTOUR":
                from PIL import ImageFilter
                image = image.filter(ImageFilter.CONTOUR)
            elif filter_name == "SHARPEN":
                from PIL import ImageFilter
                image = image.filter(ImageFilter.SHARPEN)
            else:
                raise ValueError(f"Unknown filter: {filter_name}")
                
        else:
            raise ValueError(f"Unknown operation: {operation}")
        
        # Convert the processed image back to bytes
        output = BytesIO()
        image.save(output, format=image.format or "PNG")
        return output.getvalue()
        
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise ValueError(f"Image processing error: {str(e)}")

# Endpoints
@router.post("/speech-to-text")
async def speech_to_text_endpoint(request: SpeechToTextRequest):
    """Convert speech to text"""
    try:
        # Decode base64 audio data
        audio_data = base64.b64decode(request.audio_data)
        
        # Convert speech to text
        text = await speech_to_text(audio_data, request.model)
        
        return {
            "text": text,
            "model": request.model
        }
    except Exception as e:
        logger.error(f"Error in speech-to-text endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/text-to-speech")
async def text_to_speech_endpoint(request: TextToSpeechRequest):
    """Convert text to speech"""
    try:
        # Choose the appropriate TTS provider
        if request.provider == "elevenlabs" and ELEVENLABS_AVAILABLE:
            audio_data = await text_to_speech_elevenlabs(request.text, request.voice)
        elif request.provider == "piper" and PIPER_AVAILABLE:
            audio_data = await text_to_speech_piper(request.text, request.voice)
        else:
            if PIPER_AVAILABLE:
                audio_data = await text_to_speech_piper(request.text, request.voice)
            elif ELEVENLABS_AVAILABLE:
                audio_data = await text_to_speech_elevenlabs(request.text, request.voice)
            else:
                raise HTTPException(status_code=500, detail="No text-to-speech providers available")
        
        # Encode audio data as base64
        audio_base64 = base64.b64encode(audio_data).decode()
        
        return {
            "audio_data": audio_base64,
            "provider": request.provider,
            "voice": request.voice
        }
    except Exception as e:
        logger.error(f"Error in text-to-speech endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/process-image")
async def process_image_endpoint(request: ImageProcessRequest):
    """Process an image"""
    try:
        # Decode base64 image data
        image_data = base64.b64decode(request.image_data)
        
        # Process the image
        processed_data = await process_image(
            image_data,
            request.operation,
            request.parameters
        )
        
        # Encode processed image data as base64
        processed_base64 = base64.b64encode(processed_data).decode()
        
        return {
            "processed_image": processed_base64,
            "operation": request.operation,
            "parameters": request.parameters
        }
    except Exception as e:
        logger.error(f"Error in process-image endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload-audio")
async def upload_audio(
    file: UploadFile = File(...),
    model: str = Form("base")
):
    """Upload an audio file for transcription"""
    try:
        # Read the file content
        audio_data = await file.read()
        
        # Convert speech to text
        text = await speech_to_text(audio_data, model)
        
        return {
            "filename": file.filename,
            "text": text,
            "model": model
        }
    except Exception as e:
        logger.error(f"Error in upload-audio endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload-image")
async def upload_image(
    file: UploadFile = File(...),
    operation: str = Form("resize"),
    parameters: str = Form("{}")
):
    """Upload an image for processing"""
    try:
        # Read the file content
        image_data = await file.read()
        
        # Parse parameters
        params = json.loads(parameters)
        
        # Process the image
        processed_data = await process_image(
            image_data,
            operation,
            params
        )
        
        # Encode processed image data as base64
        processed_base64 = base64.b64encode(processed_data).decode()
        
        return {
            "filename": file.filename,
            "processed_image": processed_base64,
            "operation": operation,
            "parameters": params
        }
    except Exception as e:
        logger.error(f"Error in upload-image endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))
