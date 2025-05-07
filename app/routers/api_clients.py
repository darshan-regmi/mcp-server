"""
API Clients Module for MCP Server

This module handles external API integrations:
- OpenRouter.ai
- HuggingFace
- Google Search
- WolframAlpha
"""

import os
import logging
import json
import asyncio
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel
import aiohttp
import requests
from fastapi import APIRouter, HTTPException, Depends, Query

# Initialize router
router = APIRouter()

# Configure logging
logger = logging.getLogger("MCPServer.APIClients")

# API configuration
API_KEYS = {
    "openrouter": os.environ.get("MCP_OPENROUTER_API_KEY"),
    "huggingface": os.environ.get("MCP_HUGGINGFACE_API_KEY"),
    "google": os.environ.get("MCP_GOOGLE_API_KEY"),
    "wolfram": os.environ.get("MCP_WOLFRAM_API_KEY")
}

# Check which APIs are available
AVAILABLE_APIS = {name: key is not None for name, key in API_KEYS.items()}
for name, available in AVAILABLE_APIS.items():
    if available:
        logger.info(f"{name.capitalize()} API is available")
    else:
        logger.warning(f"{name.capitalize()} API is not available (no API key)")

# Request models
class SearchRequest(BaseModel):
    """Request model for search"""
    query: str
    provider: str = "google"  # google or custom
    limit: int = 5

class WolframRequest(BaseModel):
    """Request model for Wolfram Alpha queries"""
    query: str
    format: str = "plaintext"  # plaintext, image, etc.

class HuggingFaceRequest(BaseModel):
    """Request model for HuggingFace API"""
    model: str
    inputs: Any
    parameters: Optional[Dict[str, Any]] = None

# Google Search function
async def google_search(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Perform a Google search
    
    Args:
        query: Search query
        limit: Maximum number of results to return
        
    Returns:
        List of search results
    """
    if not AVAILABLE_APIS["google"]:
        raise ValueError("Google API key not found")
    
    try:
        # Use Google Custom Search API
        api_key = API_KEYS["google"]
        cx = os.environ.get("MCP_GOOGLE_CX")  # Search engine ID
        
        if not cx:
            raise ValueError("Google Custom Search Engine ID not found")
        
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": api_key,
            "cx": cx,
            "q": query,
            "num": min(limit, 10)  # Google API allows max 10 results per request
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise ValueError(f"Google API error: {error_text}")
                
                result = await response.json()
                
                # Format the results
                items = result.get("items", [])
                formatted_results = []
                
                for item in items:
                    formatted_results.append({
                        "title": item.get("title"),
                        "link": item.get("link"),
                        "snippet": item.get("snippet"),
                        "source": "google"
                    })
                
                return formatted_results
                
    except Exception as e:
        logger.error(f"Error in Google search: {e}")
        raise ValueError(f"Google search error: {str(e)}")

# Wolfram Alpha function
async def wolfram_query(query: str, format: str = "plaintext") -> Dict[str, Any]:
    """
    Query Wolfram Alpha
    
    Args:
        query: Query string
        format: Response format (plaintext, image, etc.)
        
    Returns:
        Wolfram Alpha response
    """
    if not AVAILABLE_APIS["wolfram"]:
        raise ValueError("Wolfram Alpha API key not found")
    
    try:
        # Use Wolfram Alpha API
        api_key = API_KEYS["wolfram"]
        
        url = "https://api.wolframalpha.com/v2/query"
        params = {
            "appid": api_key,
            "input": query,
            "format": format,
            "output": "json"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise ValueError(f"Wolfram Alpha API error: {error_text}")
                
                result = await response.json()
                
                # Check if the query was successful
                if result.get("queryresult", {}).get("success") == "false":
                    return {
                        "success": False,
                        "error": "No results found",
                        "query": query
                    }
                
                # Extract pods (result sections)
                pods = result.get("queryresult", {}).get("pods", [])
                formatted_pods = []
                
                for pod in pods:
                    # Extract subpods (result items)
                    subpods = pod.get("subpods", [])
                    formatted_subpods = []
                    
                    for subpod in subpods:
                        formatted_subpods.append({
                            "plaintext": subpod.get("plaintext"),
                            "image": subpod.get("img", {}).get("src") if format != "plaintext" else None
                        })
                    
                    formatted_pods.append({
                        "title": pod.get("title"),
                        "subpods": formatted_subpods
                    })
                
                return {
                    "success": True,
                    "query": query,
                    "pods": formatted_pods
                }
                
    except Exception as e:
        logger.error(f"Error in Wolfram Alpha query: {e}")
        raise ValueError(f"Wolfram Alpha error: {str(e)}")

# HuggingFace API function
async def huggingface_api(model: str, inputs: Any, parameters: Optional[Dict[str, Any]] = None) -> Any:
    """
    Call HuggingFace API
    
    Args:
        model: Model ID
        inputs: Model inputs
        parameters: Model parameters
        
    Returns:
        Model output
    """
    if not AVAILABLE_APIS["huggingface"]:
        raise ValueError("HuggingFace API key not found")
    
    try:
        # Use HuggingFace Inference API
        api_key = API_KEYS["huggingface"]
        
        url = f"https://api-inference.huggingface.co/models/{model}"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Prepare payload
        payload = {
            "inputs": inputs
        }
        
        if parameters:
            payload["parameters"] = parameters
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise ValueError(f"HuggingFace API error: {error_text}")
                
                # Check if response is JSON or binary (for image models)
                content_type = response.headers.get("Content-Type", "")
                
                if "application/json" in content_type:
                    result = await response.json()
                else:
                    # Binary response (e.g., image)
                    result = {
                        "binary_data": True,
                        "content_type": content_type,
                        "data": await response.read()
                    }
                
                return result
                
    except Exception as e:
        logger.error(f"Error in HuggingFace API call: {e}")
        raise ValueError(f"HuggingFace API error: {str(e)}")

# Endpoints
@router.post("/search")
async def search_endpoint(request: SearchRequest):
    """Search the web"""
    try:
        if request.provider == "google":
            results = await google_search(request.query, request.limit)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown search provider: {request.provider}")
        
        return {
            "query": request.query,
            "provider": request.provider,
            "results": results
        }
    except Exception as e:
        logger.error(f"Error in search endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/wolfram")
async def wolfram_endpoint(request: WolframRequest):
    """Query Wolfram Alpha"""
    try:
        result = await wolfram_query(request.query, request.format)
        return result
    except Exception as e:
        logger.error(f"Error in Wolfram Alpha endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/huggingface")
async def huggingface_endpoint(request: HuggingFaceRequest):
    """Call HuggingFace API"""
    try:
        result = await huggingface_api(
            model=request.model,
            inputs=request.inputs,
            parameters=request.parameters
        )
        return result
    except Exception as e:
        logger.error(f"Error in HuggingFace endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status")
async def api_status():
    """Get the status of all API integrations"""
    return {
        "available_apis": AVAILABLE_APIS
    }
