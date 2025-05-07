"""
ElevenLabs Text-to-Speech Tool for MCP Server

This tool provides access to ElevenLabs' text-to-speech API for high-quality voice synthesis.
"""

import os
import logging
import requests
import json
import base64
from app.routers.tools import Tool

# Configure logging
logger = logging.getLogger("MCPServer.Tools.ElevenLabsTTS")

class ElevenLabsTTSTool(Tool):
    """Tool for text-to-speech using ElevenLabs"""
    
    def __init__(self):
        super().__init__(
            name="elevenlabs_tts",
            description="Convert text to speech using ElevenLabs' high-quality voices",
            parameters={
                "text": "The text to convert to speech",
                "voice_id": "Optional voice ID (default: 'Rachel')",
                "model_id": "Optional model ID (default: 'eleven_monolingual_v1')",
                "stability": "Optional voice stability (0.0-1.0, default: 0.5)",
                "similarity_boost": "Optional voice similarity boost (0.0-1.0, default: 0.75)"
            }
        )
        self.api_key = os.environ.get("MCP_ELEVENLABS_API_KEY")
        if not self.api_key:
            logger.warning("ElevenLabs API key not found")
        
        # Default voices
        self.default_voices = {
            "Rachel": "21m00Tcm4TlvDq8ikWAM",
            "Domi": "AZnzlk1XvdvUeBnXmlld",
            "Bella": "EXAVITQu4vr4xnSDxMaL",
            "Antoni": "ErXwobaYiN019PkySvjV",
            "Elli": "MF3mGyEYCl7XYWbV9V6O",
            "Josh": "TxGEqnHWrfWFTfGW9XjX",
            "Arnold": "VR6AewLTigWG4xSOukaG",
            "Adam": "pNInz6obpgDQGcFmaJgB",
            "Sam": "yoZ06aMxZJJ28mfd3POQ"
        }
    
    async def execute(self, **kwargs):
        """Convert text to speech using ElevenLabs"""
        if not self.api_key:
            return {
                "error": "ElevenLabs API key not configured",
                "audio": None
            }
        
        text = kwargs.get("text", "")
        if not text:
            return {
                "error": "Text is required",
                "audio": None
            }
        
        # Get voice ID (either by name or direct ID)
        voice_id = kwargs.get("voice_id", "Rachel")
        if voice_id in self.default_voices:
            voice_id = self.default_voices[voice_id]
        
        # Get other parameters
        model_id = kwargs.get("model_id", "eleven_monolingual_v1")
        stability = float(kwargs.get("stability", 0.5))
        similarity_boost = float(kwargs.get("similarity_boost", 0.75))
        
        try:
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
            
            headers = {
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
                "xi-api-key": self.api_key
            }
            
            data = {
                "text": text,
                "model_id": model_id,
                "voice_settings": {
                    "stability": stability,
                    "similarity_boost": similarity_boost
                }
            }
            
            response = requests.post(url, headers=headers, json=data)
            
            if response.status_code == 200:
                # Save the audio to a temporary file
                import tempfile
                import uuid
                
                # Create a directory for audio files if it doesn't exist
                audio_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app", "static", "audio")
                os.makedirs(audio_dir, exist_ok=True)
                
                # Generate a unique filename
                filename = f"{uuid.uuid4()}.mp3"
                filepath = os.path.join(audio_dir, filename)
                
                # Save the audio
                with open(filepath, "wb") as f:
                    f.write(response.content)
                
                # Return the URL to the audio file
                audio_url = f"/static/audio/{filename}"
                
                return {
                    "audio_url": audio_url,
                    "text": text,
                    "voice_id": voice_id,
                    "model_id": model_id
                }
            
            else:
                return {
                    "error": f"ElevenLabs API error: {response.text}",
                    "audio": None
                }
        
        except Exception as e:
            logger.error(f"Error in ElevenLabs TTS tool: {e}")
            return {
                "error": f"Error: {str(e)}",
                "audio": None
            }
