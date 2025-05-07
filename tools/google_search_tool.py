"""
Google Search Tool for MCP Server

This tool provides access to Google Search API for web search capabilities.
"""

import os
import logging
import requests
import json
from app.routers.tools import Tool

# Configure logging
logger = logging.getLogger("MCPServer.Tools.GoogleSearch")

class GoogleSearchTool(Tool):
    """Tool for performing Google searches"""
    
    def __init__(self):
        super().__init__(
            name="google_search",
            description="Search the web using Google Search API",
            parameters={
                "query": "The search query",
                "num_results": "Optional number of results to return (default: 5)",
                "safe_search": "Optional safe search setting (off, medium, high)"
            }
        )
        self.api_key = os.environ.get("MCP_GOOGLE_API_KEY")
        self.cx = os.environ.get("MCP_GOOGLE_CX")
        
        if not self.api_key:
            logger.warning("Google API key not found")
        if not self.cx:
            logger.warning("Google Custom Search Engine ID not found")
    
    async def execute(self, **kwargs):
        """Execute a Google search query"""
        if not self.api_key:
            return {
                "error": "Google API key not configured",
                "results": []
            }
        
        if not self.cx:
            return {
                "error": "Google Custom Search Engine ID not configured",
                "results": []
            }
        
        query = kwargs.get("query", "")
        if not query:
            return {
                "error": "Query is required",
                "results": []
            }
        
        num_results = kwargs.get("num_results", 5)
        try:
            num_results = int(num_results)
            if num_results < 1 or num_results > 10:
                num_results = 5
        except:
            num_results = 5
        
        safe_search = kwargs.get("safe_search", "medium")
        if safe_search not in ["off", "medium", "high"]:
            safe_search = "medium"
        
        try:
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                "key": self.api_key,
                "cx": self.cx,
                "q": query,
                "num": num_results,
                "safe": safe_search
            }
            
            response = requests.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                
                # Extract search results
                results = []
                if "items" in data:
                    for item in data["items"]:
                        result = {
                            "title": item.get("title", ""),
                            "link": item.get("link", ""),
                            "snippet": item.get("snippet", "")
                        }
                        results.append(result)
                
                return {
                    "results": results,
                    "query": query,
                    "total_results": data.get("searchInformation", {}).get("totalResults", "0")
                }
            
            else:
                return {
                    "error": f"Google Search API error: {response.text}",
                    "results": []
                }
        
        except Exception as e:
            logger.error(f"Error in Google Search tool: {e}")
            return {
                "error": f"Error: {str(e)}",
                "results": []
            }
