"""
Memory Tool for MCP Server

This tool provides enhanced memory capabilities for the AI assistant.
"""

import os
import logging
import json
import time
import datetime
from typing import Dict, List, Any, Optional
from app.routers.tools import Tool
from app.routers.memory import store_memory, search_memory, get_recent_memories

# Configure logging
logger = logging.getLogger("MCPServer.Tools.Memory")

class MemoryTool(Tool):
    """Tool for managing AI memory"""
    
    def __init__(self):
        super().__init__(
            name="memory",
            description="Store and retrieve memories for the AI assistant",
            parameters={
                "action": "The memory action to perform (store, retrieve, summarize)",
                "content": "The content to store (for store action)",
                "query": "The query to search for (for retrieve action)",
                "category": "Optional category for the memory",
                "importance": "Optional importance level (1-10)"
            }
        )
    
    async def execute(self, **kwargs):
        """Execute memory operations"""
        action = kwargs.get("action", "")
        if not action:
            return {
                "error": "Action is required (store, retrieve, summarize)",
                "result": None
            }
        
        try:
            if action == "store":
                return await self._store_memory(**kwargs)
            elif action == "retrieve":
                return await self._retrieve_memory(**kwargs)
            elif action == "summarize":
                return await self._summarize_memories(**kwargs)
            else:
                return {
                    "error": f"Unknown action: {action}",
                    "valid_actions": ["store", "retrieve", "summarize"]
                }
        
        except Exception as e:
            logger.error(f"Error in Memory tool: {e}")
            return {
                "error": f"Error: {str(e)}",
                "result": None
            }
    
    async def _store_memory(self, **kwargs):
        """Store a new memory"""
        content = kwargs.get("content", "")
        if not content:
            return {
                "error": "Content is required for store action",
                "result": None
            }
        
        category = kwargs.get("category", "general")
        importance = kwargs.get("importance", 5)
        try:
            importance = int(importance)
            if importance < 1 or importance > 10:
                importance = 5
        except:
            importance = 5
        
        # Create memory object
        memory = {
            "content": content,
            "category": category,
            "importance": importance,
            "timestamp": time.time(),
            "date": datetime.datetime.now().isoformat()
        }
        
        # Store the memory
        memory_id = await store_memory(memory)
        
        return {
            "result": "Memory stored successfully",
            "memory_id": memory_id,
            "memory": memory
        }
    
    async def _retrieve_memory(self, **kwargs):
        """Retrieve memories based on a query"""
        query = kwargs.get("query", "")
        if not query:
            return {
                "error": "Query is required for retrieve action",
                "memories": []
            }
        
        category = kwargs.get("category", None)
        limit = kwargs.get("limit", 5)
        try:
            limit = int(limit)
            if limit < 1 or limit > 20:
                limit = 5
        except:
            limit = 5
        
        # Search memories
        memories = await search_memory(query, category=category, limit=limit)
        
        return {
            "memories": memories,
            "query": query,
            "category": category,
            "count": len(memories)
        }
    
    async def _summarize_memories(self, **kwargs):
        """Summarize recent memories"""
        from app.routers.ai_router import get_ai_response
        
        category = kwargs.get("category", None)
        days = kwargs.get("days", 7)
        try:
            days = int(days)
            if days < 1 or days > 90:
                days = 7
        except:
            days = 7
        
        # Get recent memories
        memories = await get_recent_memories(days=days, category=category)
        
        if not memories:
            return {
                "summary": "No memories found for the specified period",
                "memories": []
            }
        
        # Format memories for summarization
        memory_text = ""
        for memory in memories:
            date = datetime.datetime.fromisoformat(memory.get("date", "")).strftime("%Y-%m-%d")
            memory_text += f"- [{date}] {memory.get('content', '')}\n"
        
        # Get AI to summarize the memories
        prompt = f"Summarize the following memories in a concise way, highlighting important patterns and insights:\n\n{memory_text}"
        
        response = await get_ai_response(
            message=prompt,
            mode="assistant",
            temperature=0.7
        )
        
        return {
            "summary": response["message"],
            "memory_count": len(memories),
            "period": f"Last {days} days",
            "category": category or "all"
        }
