"""
AI Router Module for MCP Server

This module handles routing requests to different AI models:
- Online APIs (OpenRouter, OpenAI, HuggingFace)
- Local models (via Ollama)

It selects the appropriate model based on:
- User preference
- Connectivity status
- Task requirements
"""

import os
import logging
import json
import asyncio
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel
import aiohttp
import requests
from fastapi import APIRouter, HTTPException, Depends

# Initialize router
router = APIRouter()

# Configure logging
logger = logging.getLogger("MCPServer.AIRouter")

# Model configuration
DEFAULT_MODELS = {
    "default": {
        "online": "gpt-3.5-turbo",
        "offline": "mistral:latest"
    },
    "poetry": {
        "online": "claude-3-opus-20240229",
        "offline": "llama3:latest"
    },
    "code": {
        "online": "gpt-4o",
        "offline": "codellama:latest"
    },
    "assistant": {
        "online": "gpt-4-turbo",
        "offline": "mistral:latest"
    },
    "research": {
        "online": "claude-3-opus-20240229",
        "offline": "llama3:latest"
    }
}

# Load model configuration from environment or config file
try:
    model_config_path = os.environ.get("MCP_MODEL_CONFIG", "config/models.json")
    if os.path.exists(model_config_path):
        with open(model_config_path, "r") as f:
            custom_models = json.load(f)
            # Merge with defaults, prioritizing custom configurations
            for mode, models in custom_models.items():
                if mode in DEFAULT_MODELS:
                    DEFAULT_MODELS[mode].update(models)
                else:
                    DEFAULT_MODELS[mode] = models
except Exception as e:
    logger.warning(f"Could not load custom model configuration: {e}")

# Request models
class AIRequest(BaseModel):
    """Request model for AI router"""
    message: str
    context: Optional[Dict[str, Any]] = None
    mode: str = "default"
    model: Optional[str] = None
    use_tools: bool = True
    max_tokens: Optional[int] = None
    temperature: Optional[float] = 0.7

# Check if online
async def is_online() -> bool:
    """Check if the system has internet connectivity"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("https://www.google.com", timeout=2) as response:
                return response.status == 200
    except:
        return False

# Get API keys
def get_api_key(provider: str) -> Optional[str]:
    """Get API key for a specific provider"""
    env_var = f"MCP_{provider.upper()}_API_KEY"
    return os.environ.get(env_var)

# Format prompt with context
def format_prompt(message: str, context: Optional[Dict[str, Any]] = None, mode: str = "default") -> str:
    """Format the prompt with context and mode-specific instructions"""
    
    # Base system prompts for different modes
    system_prompts = {
        "default": "You are a helpful AI assistant.",
        "poetry": "You are a creative poetry assistant with a deep understanding of literary forms, styles, and techniques.",
        "code": "You are a programming assistant with expertise in software development, debugging, and best practices.",
        "assistant": "You are a personal assistant who helps organize tasks, schedules, and provides helpful information.",
        "research": "You are a research assistant who provides well-structured, factual information with citations when possible."
    }
    
    # Get the appropriate system prompt
    system_prompt = system_prompts.get(mode, system_prompts["default"])
    
    # Add context if provided
    if context:
        # Add relevant memories if available
        if "memories" in context:
            memory_text = "\n\nRelevant information from your memory:\n"
            for memory in context["memories"]:
                memory_text += f"- {memory['content']}\n"
            system_prompt += memory_text
        
        # Add tool descriptions if available and tools are enabled
        if "tools" in context and context.get("use_tools", True):
            tools_text = "\n\nYou have access to the following tools:\n"
            for tool in context["tools"]:
                tools_text += f"- {tool['name']}: {tool['description']}\n"
            system_prompt += tools_text
    
    # Format as a chat completion prompt
    formatted_prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": message}
    ]
    
    return formatted_prompt

# OpenAI API call
async def call_openai(prompt, model: str = "gpt-3.5-turbo", temperature: float = 0.7, max_tokens: Optional[int] = None):
    """Call the OpenAI API"""
    api_key = get_api_key("openai")
    if not api_key:
        raise ValueError("OpenAI API key not found")
    
    try:
        import openai
        client = openai.OpenAI(api_key=api_key)
        
        # Prepare parameters
        params = {
            "model": model,
            "messages": prompt,
            "temperature": temperature,
        }
        
        if max_tokens:
            params["max_tokens"] = max_tokens
        
        # Make the API call
        response = client.chat.completions.create(**params)
        
        return {
            "message": response.choices[0].message.content,
            "model_used": model,
            "provider": "openai",
            "tool_calls": None  # We'll implement tool calling in a more advanced version
        }
    except Exception as e:
        logger.error(f"Error calling OpenAI API: {e}")
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")

# OpenRouter API call
async def call_openrouter(prompt, model: str = "gpt-3.5-turbo", temperature: float = 0.7, max_tokens: Optional[int] = None):
    """Call the OpenRouter API"""
    api_key = get_api_key("openrouter")
    if not api_key:
        raise ValueError("OpenRouter API key not found")
    
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Prepare parameters
        params = {
            "model": model,
            "messages": prompt,
            "temperature": temperature,
        }
        
        if max_tokens:
            params["max_tokens"] = max_tokens
        
        # Make the API call
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=params
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise ValueError(f"OpenRouter API error: {error_text}")
                
                result = await response.json()
                
                return {
                    "message": result["choices"][0]["message"]["content"],
                    "model_used": model,
                    "provider": "openrouter",
                    "tool_calls": None  # We'll implement tool calling in a more advanced version
                }
    except Exception as e:
        logger.error(f"Error calling OpenRouter API: {e}")
        raise HTTPException(status_code=500, detail=f"OpenRouter API error: {str(e)}")

# Ollama API call
async def call_ollama(prompt, model: str = "mistral:latest", temperature: float = 0.7, max_tokens: Optional[int] = None):
    """Call the local Ollama instance"""
    try:
        # Convert chat format to text for Ollama
        system_message = next((msg["content"] for msg in prompt if msg["role"] == "system"), "")
        user_message = next((msg["content"] for msg in prompt if msg["role"] == "user"), "")
        
        combined_prompt = f"{system_message}\n\n{user_message}"
        
        # Prepare parameters
        params = {
            "model": model,
            "prompt": combined_prompt,
            "temperature": temperature,
        }
        
        if max_tokens:
            params["max_tokens"] = max_tokens
        
        # Make the API call to local Ollama instance
        response = requests.post(
            "http://localhost:11434/api/generate",
            json=params
        )
        
        if response.status_code != 200:
            raise ValueError(f"Ollama API error: {response.text}")
        
        result = response.json()
        
        return {
            "message": result["response"],
            "model_used": model,
            "provider": "ollama",
            "tool_calls": None  # Local models don't support tool calling yet
        }
    except Exception as e:
        logger.error(f"Error calling Ollama API: {e}")
        raise HTTPException(status_code=500, detail=f"Ollama API error: {str(e)}")

# Select model based on mode and connectivity
async def select_model(mode: str, online: bool, specified_model: Optional[str] = None) -> str:
    """Select the appropriate model based on mode and connectivity"""
    if specified_model:
        return specified_model
    
    if mode in DEFAULT_MODELS:
        return DEFAULT_MODELS[mode]["online" if online else "offline"]
    else:
        return DEFAULT_MODELS["default"]["online" if online else "offline"]

# Main AI response function
async def get_ai_response(
    message: str,
    context: Optional[Dict[str, Any]] = None,
    mode: str = "default",
    use_tools: bool = True,
    use_memory: bool = True,
    model: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: float = 0.7
) -> Dict[str, Any]:
    """
    Get a response from an AI model
    
    This function:
    1. Checks connectivity
    2. Selects the appropriate model
    3. Formats the prompt with context
    4. Calls the appropriate API
    5. Processes the response
    """
    try:
        # Check connectivity
        online = await is_online()
        logger.info(f"Connectivity status: {'online' if online else 'offline'}")
        
        # Select model
        selected_model = await select_model(mode, online, model)
        logger.info(f"Selected model: {selected_model}")
        
        # Format prompt
        if context is None:
            context = {}
        
        # Add tool and memory flags to context
        context["use_tools"] = use_tools
        context["use_memory"] = use_memory
        
        formatted_prompt = format_prompt(message, context, mode)
        
        # Call the appropriate API
        if online:
            # Try to use OpenRouter first (since OpenAI has quota issues)
            try:
                # Use OpenRouter for all models
                response = await call_openrouter(
                    formatted_prompt, 
                    model=selected_model,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            except Exception as router_error:
                logger.warning(f"OpenRouter error: {router_error}, falling back to OpenAI")
                # Fall back to OpenAI if OpenRouter fails
                if selected_model.startswith("gpt-"):
                    response = await call_openai(
                        formatted_prompt, 
                        model=selected_model,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                else:
                    # Re-raise the error if we can't use OpenAI as fallback
                    raise router_error
        else:
            # Use Ollama for offline mode
            response = await call_ollama(
                formatted_prompt, 
                model=selected_model,
                temperature=temperature,
                max_tokens=max_tokens
            )
        
        return response
        
    except Exception as e:
        logger.error(f"Error in AI router: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Endpoints
@router.post("/chat")
async def route_chat(request: AIRequest):
    """Route a chat request to the appropriate AI model"""
    response = await get_ai_response(
        message=request.message,
        context=request.context,
        mode=request.mode,
        model=request.model,
        max_tokens=request.max_tokens,
        temperature=request.temperature
    )
    return response

@router.get("/models")
async def list_models():
    """List available AI models"""
    # Online models
    online_models = {mode: config["online"] for mode, config in DEFAULT_MODELS.items()}
    
    # Local models
    try:
        # Try to get available models from Ollama
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            local_models = [tag["name"] for tag in response.json()["models"]]
        else:
            local_models = [config["offline"] for _, config in DEFAULT_MODELS.items()]
    except:
        local_models = [config["offline"] for _, config in DEFAULT_MODELS.items()]
    
    return {
        "online": online_models,
        "offline": local_models
    }
