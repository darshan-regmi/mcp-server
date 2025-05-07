"""
MCP Server - Model Context Protocol
A modular AI assistant engine that connects language models with tools and data
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import logging
import json
import time
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("mcp_server.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("MCPServer")

# Initialize FastAPI app
app = FastAPI(
    title="MCP Server",
    description="Model Context Protocol Server - A bridge between AI models and tools",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development - restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Route for web interface
@app.get("/", response_class=HTMLResponse)
async def get_web_interface():
    """Serve the web interface"""
    html_file = static_dir / "index.html"
    return HTMLResponse(content=html_file.read_text(), status_code=200)

# Import routers
try:
    from app.routers import ai_router, tools, memory, multimodal, api_clients
    app.include_router(ai_router.router, prefix="/ai", tags=["AI Models"])
    app.include_router(tools.router, prefix="/tools", tags=["Tools"])
    app.include_router(memory.router, prefix="/memory", tags=["Memory"])
    app.include_router(multimodal.router, prefix="/multimodal", tags=["Multimodal"])
    app.include_router(api_clients.router, prefix="/api", tags=["API Clients"])
except ImportError as e:
    logger.warning(f"Some routers could not be imported: {e}")
    # Continue without the missing routers

# Request models
class ChatRequest(BaseModel):
    """Request model for chat interactions"""
    message: str
    context: Optional[Dict[str, Any]] = None
    mode: Optional[str] = "default"  # default, poetry, code, assistant, research
    use_tools: Optional[bool] = True
    use_memory: Optional[bool] = True
    model: Optional[str] = None  # Specific model to use, if any

class ToolRequest(BaseModel):
    """Request model for direct tool execution"""
    tool_name: str
    parameters: Dict[str, Any]

# Response models
class ChatResponse(BaseModel):
    """Response model for chat interactions"""
    message: str
    tool_calls: Optional[List[Dict[str, Any]]] = None
    sources: Optional[List[Dict[str, Any]]] = None
    mode: str
    model_used: str
    processing_time: float

class ToolResponse(BaseModel):
    """Response model for tool execution"""
    result: Any
    success: bool
    error: Optional[str] = None

# Health check endpoint
@app.get("/health")
async def health_check():
    """Check if the MCP server is running"""
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "version": app.version
    }

# Chat endpoint
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, background_tasks: BackgroundTasks):
    """
    Main chat endpoint for interacting with the MCP server
    
    This endpoint:
    1. Routes the request to the appropriate AI model
    2. Processes the response through tools if needed
    3. Stores context in memory if enabled
    4. Returns the formatted response
    """
    start_time = time.time()
    
    try:
        # Import here to avoid circular imports
        from app.routers.ai_router import get_ai_response
        
        # Get response from AI model
        response = await get_ai_response(
            message=request.message,
            context=request.context,
            mode=request.mode,
            use_tools=request.use_tools,
            use_memory=request.use_memory,
            model=request.model
        )
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # If memory is enabled, store the interaction asynchronously
        if request.use_memory:
            from app.routers.memory import store_interaction
            background_tasks.add_task(
                store_interaction,
                user_message=request.message,
                ai_response=response["message"],
                context=request.context,
                mode=request.mode
            )
        
        return {
            "message": response["message"],
            "tool_calls": response.get("tool_calls"),
            "sources": response.get("sources"),
            "mode": request.mode,
            "model_used": response.get("model_used", "unknown"),
            "processing_time": processing_time
        }
        
    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Tool execution endpoint
@app.post("/tools/execute", response_model=ToolResponse)
async def execute_tool(request: ToolRequest):
    """
    Execute a specific tool directly
    
    This endpoint allows direct execution of tools without going through the AI model
    """
    try:
        # Import here to avoid circular imports
        from app.routers.tools import execute_tool_by_name
        
        # Execute the tool
        result = await execute_tool_by_name(
            tool_name=request.tool_name,
            parameters=request.parameters
        )
        
        return {
            "result": result,
            "success": True
        }
        
    except Exception as e:
        logger.error(f"Error executing tool {request.tool_name}: {e}")
        return {
            "result": None,
            "success": False,
            "error": str(e)
        }

# Error handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for all unhandled exceptions"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)}
    )

if __name__ == "__main__":
    import uvicorn
    
    # Get configuration from environment variables or use defaults
    host = os.environ.get("MCP_HOST", "0.0.0.0")
    port = int(os.environ.get("MCP_PORT", "8000"))
    
    logger.info(f"Starting MCP server on {host}:{port}")
    uvicorn.run("app.main:app", host=host, port=port, reload=True)
