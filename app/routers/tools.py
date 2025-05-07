"""
Tools Module for MCP Server

This module manages tool plugins that can be called by the AI models:
- Poetry generation
- Code assistance
- Mac cleanup
- Daily agent
- Memory manager
- And more custom tools
"""

import os
import logging
import importlib
import inspect
import json
import subprocess
from typing import Dict, List, Optional, Any, Callable
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks

# Initialize router
router = APIRouter()

# Configure logging
logger = logging.getLogger("MCPServer.Tools")

# Tool registry
TOOLS_REGISTRY = {}

# Tool base class
class Tool:
    """Base class for all tools"""
    name: str
    description: str
    parameters: Dict[str, Any]
    
    def __init__(self, name: str, description: str, parameters: Dict[str, Any] = None):
        self.name = name
        self.description = description
        self.parameters = parameters or {}
    
    async def execute(self, **kwargs):
        """Execute the tool with the given parameters"""
        raise NotImplementedError("Tool must implement execute method")

# Request models
class ToolExecutionRequest(BaseModel):
    """Request model for tool execution"""
    tool_name: str
    parameters: Dict[str, Any]

# Response models
class ToolExecutionResponse(BaseModel):
    """Response model for tool execution"""
    result: Any
    success: bool
    error: Optional[str] = None

# Load tools from directory
def load_tools():
    """Load all tools from the tools directory"""
    tools_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "tools")
    if not os.path.exists(tools_dir):
        logger.warning(f"Tools directory not found: {tools_dir}")
        return
    
    # Get all Python files in the tools directory
    tool_files = [f[:-3] for f in os.listdir(tools_dir) if f.endswith('.py') and not f.startswith('__')]
    
    for tool_file in tool_files:
        try:
            # Import the tool module
            module_path = f"tools.{tool_file}"
            module = importlib.import_module(module_path)
            
            # Find all Tool classes in the module
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and issubclass(obj, Tool) and obj != Tool:
                    # Create an instance of the tool
                    tool_instance = obj()
                    
                    # Register the tool
                    TOOLS_REGISTRY[tool_instance.name] = tool_instance
                    logger.info(f"Registered tool: {tool_instance.name}")
        except Exception as e:
            logger.error(f"Error loading tool {tool_file}: {e}")

# Initialize built-in tools

# Poetry Generation Tool
class PoetryGenerationTool(Tool):
    """Tool for generating poetry"""
    
    def __init__(self):
        super().__init__(
            name="poetry_gen",
            description="Generate poetry based on a topic, style, or mood",
            parameters={
                "topic": "The main topic or theme of the poem",
                "style": "Optional style (e.g., sonnet, haiku, free verse)",
                "mood": "Optional mood (e.g., melancholic, joyful, contemplative)",
                "length": "Optional approximate length in lines"
            }
        )
    
    async def execute(self, **kwargs):
        """Generate poetry using the AI model"""
        from app.routers.ai_router import get_ai_response
        
        topic = kwargs.get("topic", "")
        style = kwargs.get("style", "")
        mood = kwargs.get("mood", "")
        length = kwargs.get("length", "")
        
        # Construct a prompt for the poetry generation
        prompt = f"Write a poem about {topic}"
        if style:
            prompt += f" in the style of a {style}"
        if mood:
            prompt += f" with a {mood} mood"
        if length:
            prompt += f" that is approximately {length} lines long"
        
        # Get response from AI model in poetry mode
        response = await get_ai_response(
            message=prompt,
            mode="poetry",
            temperature=0.8
        )
        
        return {
            "poem": response["message"],
            "topic": topic,
            "style": style,
            "mood": mood
        }

# Code Assistance Tool
class CodeAssistanceTool(Tool):
    """Tool for code assistance"""
    
    def __init__(self):
        super().__init__(
            name="code_assist",
            description="Provide code assistance, including generation, explanation, and debugging",
            parameters={
                "task": "The coding task (generate, explain, debug)",
                "language": "The programming language",
                "code": "The code to explain or debug (if applicable)",
                "requirements": "Requirements for code generation"
            }
        )
    
    async def execute(self, **kwargs):
        """Provide code assistance using the AI model"""
        from app.routers.ai_router import get_ai_response
        
        task = kwargs.get("task", "generate")
        language = kwargs.get("language", "python")
        code = kwargs.get("code", "")
        requirements = kwargs.get("requirements", "")
        
        # Construct a prompt based on the task
        if task == "generate":
            prompt = f"Generate {language} code that meets these requirements: {requirements}"
        elif task == "explain":
            prompt = f"Explain the following {language} code in detail:\n\n```{language}\n{code}\n```"
        elif task == "debug":
            prompt = f"Debug the following {language} code and explain the issues:\n\n```{language}\n{code}\n```"
        else:
            prompt = f"Help with this {language} code: {code}\nRequirements: {requirements}"
        
        # Get response from AI model in code mode
        response = await get_ai_response(
            message=prompt,
            mode="code",
            temperature=0.3
        )
        
        return {
            "result": response["message"],
            "task": task,
            "language": language
        }

# Mac Cleanup Tool
class MacCleanupTool(Tool):
    """Tool for cleaning up Mac systems"""
    
    def __init__(self):
        super().__init__(
            name="mac_cleanup",
            description="Clean up Mac system, including caches, logs, and temporary files",
            parameters={
                "target": "What to clean (all, caches, logs, downloads, xcode)",
                "dry_run": "If true, only show what would be cleaned without actually deleting"
            }
        )
    
    async def execute(self, **kwargs):
        """Clean up Mac system"""
        target = kwargs.get("target", "all").lower()
        dry_run = kwargs.get("dry_run", True)
        
        # Validate we're on a Mac
        if not os.path.exists("/Applications"):
            return {
                "success": False,
                "message": "This tool only works on macOS systems"
            }
        
        results = {}
        
        # Define cleanup functions
        cleanup_actions = {
            "caches": self._clean_caches,
            "logs": self._clean_logs,
            "downloads": self._clean_downloads,
            "xcode": self._clean_xcode
        }
        
        # Execute requested cleanup
        if target == "all":
            for action_name, action_func in cleanup_actions.items():
                results[action_name] = action_func(dry_run)
        elif target in cleanup_actions:
            results[target] = cleanup_actions[target](dry_run)
        else:
            return {
                "success": False,
                "message": f"Unknown target: {target}. Valid targets are: all, caches, logs, downloads, xcode"
            }
        
        return {
            "success": True,
            "dry_run": dry_run,
            "results": results
        }
    
    def _clean_caches(self, dry_run):
        """Clean system and user caches"""
        cache_dirs = [
            "~/Library/Caches",
            "~/Library/Application Support/Google/Chrome/Default/Cache",
            "~/Library/Application Support/Firefox/Profiles/*/cache"
        ]
        
        results = {}
        for cache_dir in cache_dirs:
            expanded_path = os.path.expanduser(cache_dir)
            if os.path.exists(expanded_path):
                if dry_run:
                    # Get size of directory
                    size = self._get_dir_size(expanded_path)
                    results[cache_dir] = f"Would clean {self._format_size(size)}"
                else:
                    # Actually clean
                    try:
                        subprocess.run(["rm", "-rf", expanded_path + "/*"], check=False)
                        results[cache_dir] = "Cleaned"
                    except Exception as e:
                        results[cache_dir] = f"Error: {str(e)}"
            else:
                results[cache_dir] = "Not found"
        
        return results
    
    def _clean_logs(self, dry_run):
        """Clean log files"""
        log_dirs = [
            "~/Library/Logs",
            "/var/log"
        ]
        
        results = {}
        for log_dir in log_dirs:
            expanded_path = os.path.expanduser(log_dir)
            if os.path.exists(expanded_path):
                if dry_run:
                    # Get size of directory
                    size = self._get_dir_size(expanded_path)
                    results[log_dir] = f"Would clean {self._format_size(size)}"
                else:
                    # Actually clean
                    try:
                        # For system logs, we need sudo, so we'll skip those
                        if log_dir.startswith("/var"):
                            results[log_dir] = "Skipped (requires sudo)"
                        else:
                            subprocess.run(["rm", "-rf", expanded_path + "/*"], check=False)
                            results[log_dir] = "Cleaned"
                    except Exception as e:
                        results[log_dir] = f"Error: {str(e)}"
            else:
                results[log_dir] = "Not found"
        
        return results
    
    def _clean_downloads(self, dry_run):
        """Clean Downloads folder"""
        downloads_dir = "~/Downloads"
        expanded_path = os.path.expanduser(downloads_dir)
        
        if os.path.exists(expanded_path):
            if dry_run:
                # Get size of directory
                size = self._get_dir_size(expanded_path)
                return f"Would clean {self._format_size(size)}"
            else:
                # Actually clean
                try:
                    subprocess.run(["rm", "-rf", expanded_path + "/*"], check=False)
                    return "Cleaned"
                except Exception as e:
                    return f"Error: {str(e)}"
        else:
            return "Not found"
    
    def _clean_xcode(self, dry_run):
        """Clean Xcode derived data and archives"""
        xcode_dirs = [
            "~/Library/Developer/Xcode/DerivedData",
            "~/Library/Developer/Xcode/Archives",
            "~/Library/Developer/Xcode/iOS DeviceSupport"
        ]
        
        results = {}
        for xcode_dir in xcode_dirs:
            expanded_path = os.path.expanduser(xcode_dir)
            if os.path.exists(expanded_path):
                if dry_run:
                    # Get size of directory
                    size = self._get_dir_size(expanded_path)
                    results[xcode_dir] = f"Would clean {self._format_size(size)}"
                else:
                    # Actually clean
                    try:
                        subprocess.run(["rm", "-rf", expanded_path + "/*"], check=False)
                        results[xcode_dir] = "Cleaned"
                    except Exception as e:
                        results[xcode_dir] = f"Error: {str(e)}"
            else:
                results[xcode_dir] = "Not found"
        
        return results
    
    def _get_dir_size(self, path):
        """Get the size of a directory in bytes"""
        try:
            result = subprocess.run(
                ["du", "-s", "-k", path],
                capture_output=True,
                text=True,
                check=False
            )
            if result.returncode == 0:
                # Output is in kilobytes
                size_kb = int(result.stdout.split()[0])
                return size_kb * 1024
            return 0
        except:
            return 0
    
    def _format_size(self, size_bytes):
        """Format size in bytes to human-readable format"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.2f} PB"

# Daily Agent Tool
class DailyAgentTool(Tool):
    """Tool for daily assistant tasks"""
    
    def __init__(self):
        super().__init__(
            name="daily_agent",
            description="Perform daily assistant tasks like checking weather, schedule, and providing summaries",
            parameters={
                "task": "The task to perform (morning_briefing, weather, schedule, summary)",
                "date": "Optional date for schedule or summary (default: today)"
            }
        )
    
    async def execute(self, **kwargs):
        """Execute daily agent tasks"""
        from app.routers.ai_router import get_ai_response
        from datetime import datetime, date
        
        task = kwargs.get("task", "morning_briefing")
        date_str = kwargs.get("date", datetime.now().strftime("%Y-%m-%d"))
        
        # Handle different tasks
        if task == "morning_briefing":
            # Get weather
            weather = await self._get_weather()
            
            # Get schedule
            schedule = await self._get_schedule(date_str)
            
            # Construct morning briefing
            briefing = f"Good morning! Here's your daily briefing for {date_str}.\n\n"
            
            if weather:
                briefing += f"Weather: {weather}\n\n"
            
            if schedule:
                briefing += "Today's schedule:\n"
                for item in schedule:
                    briefing += f"- {item['time']}: {item['description']}\n"
            else:
                briefing += "You have no scheduled events for today.\n"
            
            # Get AI to format the briefing nicely
            response = await get_ai_response(
                message=f"Format this daily briefing in a friendly, motivational way:\n\n{briefing}",
                mode="assistant",
                temperature=0.7
            )
            
            return {
                "briefing": response["message"],
                "date": date_str,
                "weather": weather,
                "schedule": schedule
            }
            
        elif task == "weather":
            weather = await self._get_weather()
            return {
                "weather": weather,
                "date": date_str
            }
            
        elif task == "schedule":
            schedule = await self._get_schedule(date_str)
            return {
                "schedule": schedule,
                "date": date_str
            }
            
        elif task == "summary":
            # This would typically summarize the day's activities, notes, etc.
            # For now, we'll return a placeholder
            return {
                "summary": "Daily summary functionality is not yet implemented",
                "date": date_str
            }
            
        else:
            return {
                "error": f"Unknown task: {task}",
                "valid_tasks": ["morning_briefing", "weather", "schedule", "summary"]
            }
    
    async def _get_weather(self):
        """Get weather information (placeholder)"""
        # In a real implementation, this would call a weather API
        return "Weather information is not available in this demo version"
    
    async def _get_schedule(self, date_str):
        """Get schedule for the given date (placeholder)"""
        # In a real implementation, this would fetch from a calendar API
        # For demo purposes, return an empty schedule
        return []

# Register built-in tools
TOOLS_REGISTRY["poetry_gen"] = PoetryGenerationTool()
TOOLS_REGISTRY["code_assist"] = CodeAssistanceTool()
TOOLS_REGISTRY["mac_cleanup"] = MacCleanupTool()
TOOLS_REGISTRY["daily_agent"] = DailyAgentTool()

# Load custom tools
try:
    load_tools()
except Exception as e:
    logger.error(f"Error loading custom tools: {e}")

# Get all available tools
def get_available_tools():
    """Get all available tools"""
    return {
        name: {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters
        }
        for name, tool in TOOLS_REGISTRY.items()
    }

# Execute a tool by name
async def execute_tool_by_name(tool_name: str, parameters: Dict[str, Any]):
    """Execute a tool by name with the given parameters"""
    if tool_name not in TOOLS_REGISTRY:
        raise ValueError(f"Tool not found: {tool_name}")
    
    tool = TOOLS_REGISTRY[tool_name]
    return await tool.execute(**parameters)

# Endpoints
@router.get("/")
async def list_tools():
    """List all available tools"""
    return get_available_tools()

@router.post("/execute")
async def execute_tool(request: ToolExecutionRequest, background_tasks: BackgroundTasks):
    """Execute a tool"""
    try:
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
