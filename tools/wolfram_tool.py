"""
Wolfram Alpha Tool for MCP Server

This tool provides access to Wolfram Alpha's computational knowledge engine.
"""

import os
import logging
import requests
import json
from app.routers.tools import Tool

# Configure logging
logger = logging.getLogger("MCPServer.Tools.Wolfram")

class WolframTool(Tool):
    """Tool for accessing Wolfram Alpha's computational knowledge"""
    
    def __init__(self):
        super().__init__(
            name="wolfram",
            description="Access Wolfram Alpha for calculations, data, and knowledge queries",
            parameters={
                "query": "The query to send to Wolfram Alpha",
                "format": "Optional format (simple, short, full)",
            }
        )
        self.api_key = os.environ.get("MCP_WOLFRAM_API_KEY")
        if not self.api_key:
            logger.warning("Wolfram Alpha API key not found")
    
    async def execute(self, **kwargs):
        """Execute a Wolfram Alpha query"""
        if not self.api_key:
            return {
                "error": "Wolfram Alpha API key not configured",
                "result": None
            }
        
        query = kwargs.get("query", "")
        if not query:
            return {
                "error": "Query is required",
                "result": None
            }
        
        format_type = kwargs.get("format", "simple")
        
        try:
            # Simple API (text results)
            if format_type == "simple":
                url = f"https://api.wolframalpha.com/v1/result"
                params = {
                    "appid": self.api_key,
                    "i": query,
                    "units": "metric"
                }
                response = requests.get(url, params=params)
                
                if response.status_code == 200:
                    return {
                        "result": response.text,
                        "query": query,
                        "format": format_type
                    }
                else:
                    return {
                        "error": f"Wolfram Alpha API error: {response.text}",
                        "result": None
                    }
            
            # Full API (pods with images and data)
            else:
                url = f"https://api.wolframalpha.com/v2/query"
                params = {
                    "appid": self.api_key,
                    "input": query,
                    "format": "plaintext",
                    "output": "json",
                    "units": "metric"
                }
                
                response = requests.get(url, params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Extract the pods (sections of results)
                    pods = []
                    if "pods" in data.get("queryresult", {}):
                        for pod in data["queryresult"]["pods"]:
                            pod_data = {
                                "title": pod.get("title", ""),
                                "subpods": []
                            }
                            
                            for subpod in pod.get("subpods", []):
                                if "plaintext" in subpod and subpod["plaintext"]:
                                    pod_data["subpods"].append(subpod["plaintext"])
                            
                            if pod_data["subpods"]:
                                pods.append(pod_data)
                    
                    # Format the results
                    if format_type == "short":
                        # Return just the input interpretation and result
                        result = "No results found"
                        for pod in pods:
                            if pod["title"] == "Result" or pod["title"] == "Results":
                                result = "\n".join(pod["subpods"])
                                break
                        
                        return {
                            "result": result,
                            "query": query,
                            "format": format_type
                        }
                    
                    # Full format with all pods
                    else:
                        formatted_result = ""
                        for pod in pods:
                            formatted_result += f"## {pod['title']}\n"
                            for text in pod["subpods"]:
                                formatted_result += f"{text}\n\n"
                        
                        return {
                            "result": formatted_result.strip(),
                            "pods": pods,
                            "query": query,
                            "format": format_type
                        }
                
                else:
                    return {
                        "error": f"Wolfram Alpha API error: {response.text}",
                        "result": None
                    }
        
        except Exception as e:
            logger.error(f"Error in Wolfram Alpha tool: {e}")
            return {
                "error": f"Error: {str(e)}",
                "result": None
            }
