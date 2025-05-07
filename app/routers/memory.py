"""
Memory Module for MCP Server

This module manages the memory system for the MCP server:
- Stores and retrieves information from vector databases
- Manages context for AI interactions
- Handles knowledge base operations
"""

import os
import logging
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException, Depends, Query, File, UploadFile
import chromadb
from chromadb.config import Settings

# Initialize router
router = APIRouter()

# Configure logging
logger = logging.getLogger("MCPServer.Memory")

# Memory configuration
MEMORY_DIR = os.environ.get("MCP_MEMORY_DIR", "memory")
os.makedirs(MEMORY_DIR, exist_ok=True)

# Initialize ChromaDB
try:
    chroma_client = chromadb.Client(Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=os.path.join(MEMORY_DIR, "chroma")
    ))
    
    # Create collections if they don't exist
    memories_collection = chroma_client.get_or_create_collection("memories")
    notes_collection = chroma_client.get_or_create_collection("notes")
    documents_collection = chroma_client.get_or_create_collection("documents")
    
    logger.info("ChromaDB initialized successfully")
except Exception as e:
    logger.error(f"Error initializing ChromaDB: {e}")
    chroma_client = None
    memories_collection = None
    notes_collection = None
    documents_collection = None

# Request models
class MemoryItem(BaseModel):
    """Model for a memory item"""
    content: str
    metadata: Optional[Dict[str, Any]] = None
    source: Optional[str] = None

class NoteItem(BaseModel):
    """Model for a note item"""
    title: str
    content: str
    tags: Optional[List[str]] = None

class SearchQuery(BaseModel):
    """Model for a search query"""
    query: str
    collection: str = "memories"
    limit: int = 5
    metadata_filter: Optional[Dict[str, Any]] = None

# Store an interaction in memory
async def store_interaction(
    user_message: str,
    ai_response: str,
    context: Optional[Dict[str, Any]] = None,
    mode: str = "default"
):
    """Store an interaction in memory"""
    if not memories_collection:
        logger.warning("Cannot store interaction: ChromaDB not initialized")
        return
    
    try:
        # Create a unique ID for the interaction
        interaction_id = f"interaction_{int(time.time())}"
        
        # Store user message
        memories_collection.add(
            ids=[f"{interaction_id}_user"],
            documents=[user_message],
            metadatas=[{
                "type": "user_message",
                "timestamp": datetime.now().isoformat(),
                "mode": mode,
                "interaction_id": interaction_id
            }]
        )
        
        # Store AI response
        memories_collection.add(
            ids=[f"{interaction_id}_ai"],
            documents=[ai_response],
            metadatas=[{
                "type": "ai_response",
                "timestamp": datetime.now().isoformat(),
                "mode": mode,
                "interaction_id": interaction_id
            }]
        )
        
        logger.info(f"Stored interaction {interaction_id}")
        
    except Exception as e:
        logger.error(f"Error storing interaction: {e}")

# Search memory for relevant information
async def search_memory(query: str, limit: int = 5, collection_name: str = "memories", metadata_filter: Optional[Dict[str, Any]] = None):
    """Search memory for relevant information"""
    if not chroma_client:
        logger.warning("Cannot search memory: ChromaDB not initialized")
        return []
    
    try:
        # Get the appropriate collection
        collection = None
        if collection_name == "memories":
            collection = memories_collection
        elif collection_name == "notes":
            collection = notes_collection
        elif collection_name == "documents":
            collection = documents_collection
        else:
            raise ValueError(f"Unknown collection: {collection_name}")
        
        if not collection:
            logger.warning(f"Collection not found: {collection_name}")
            return []
        
        # Prepare the query
        query_params = {
            "query_texts": [query],
            "n_results": limit
        }
        
        # Add metadata filter if provided
        if metadata_filter:
            where_clause = {}
            for key, value in metadata_filter.items():
                where_clause[key] = value
            query_params["where"] = where_clause
        
        # Execute the query
        results = collection.query(**query_params)
        
        # Format the results
        formatted_results = []
        for i, (doc, metadata, distance) in enumerate(zip(
            results.get("documents", [[]])[0],
            results.get("metadatas", [[]])[0],
            results.get("distances", [[]])[0]
        )):
            formatted_results.append({
                "content": doc,
                "metadata": metadata,
                "relevance": 1.0 - (distance / 2.0) if distance is not None else 0.0  # Normalize distance to relevance score
            })
        
        return formatted_results
        
    except Exception as e:
        logger.error(f"Error searching memory: {e}")
        return []

# Store a note in memory
async def store_note(title: str, content: str, tags: Optional[List[str]] = None):
    """Store a note in memory"""
    if not notes_collection:
        logger.warning("Cannot store note: ChromaDB not initialized")
        return False
    
    try:
        # Create a unique ID for the note
        note_id = f"note_{int(time.time())}"
        
        # Prepare metadata
        metadata = {
            "title": title,
            "timestamp": datetime.now().isoformat(),
            "type": "note"
        }
        
        if tags:
            metadata["tags"] = ", ".join(tags)
        
        # Store the note
        notes_collection.add(
            ids=[note_id],
            documents=[content],
            metadatas=[metadata]
        )
        
        logger.info(f"Stored note {note_id}: {title}")
        return True
        
    except Exception as e:
        logger.error(f"Error storing note: {e}")
        return False

# Store a document in memory
async def store_document(filename: str, content: str, metadata: Optional[Dict[str, Any]] = None):
    """Store a document in memory"""
    if not documents_collection:
        logger.warning("Cannot store document: ChromaDB not initialized")
        return False
    
    try:
        # Create a unique ID for the document
        document_id = f"doc_{int(time.time())}"
        
        # Prepare metadata
        doc_metadata = {
            "filename": filename,
            "timestamp": datetime.now().isoformat(),
            "type": "document"
        }
        
        if metadata:
            doc_metadata.update(metadata)
        
        # Store the document
        documents_collection.add(
            ids=[document_id],
            documents=[content],
            metadatas=[doc_metadata]
        )
        
        logger.info(f"Stored document {document_id}: {filename}")
        return True
        
    except Exception as e:
        logger.error(f"Error storing document: {e}")
        return False

# Endpoints
@router.post("/search")
async def search_endpoint(query: SearchQuery):
    """Search memory for relevant information"""
    results = await search_memory(
        query=query.query,
        limit=query.limit,
        collection_name=query.collection,
        metadata_filter=query.metadata_filter
    )
    
    return {
        "query": query.query,
        "collection": query.collection,
        "results": results
    }

@router.post("/notes")
async def create_note(note: NoteItem):
    """Create a new note"""
    success = await store_note(
        title=note.title,
        content=note.content,
        tags=note.tags
    )
    
    if success:
        return {
            "success": True,
            "message": f"Note '{note.title}' created successfully"
        }
    else:
        raise HTTPException(status_code=500, detail="Failed to create note")

@router.post("/documents")
async def upload_document(
    file: UploadFile = File(...),
    tags: Optional[str] = Query(None, description="Comma-separated tags")
):
    """Upload a document to memory"""
    try:
        # Read the file content
        content = await file.read()
        content_str = content.decode("utf-8", errors="ignore")
        
        # Prepare metadata
        metadata = {
            "filename": file.filename,
            "content_type": file.content_type
        }
        
        if tags:
            metadata["tags"] = tags
        
        # Store the document
        success = await store_document(
            filename=file.filename,
            content=content_str,
            metadata=metadata
        )
        
        if success:
            return {
                "success": True,
                "message": f"Document '{file.filename}' uploaded successfully"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to upload document")
            
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/collections")
async def list_collections():
    """List all available collections"""
    if not chroma_client:
        raise HTTPException(status_code=500, detail="ChromaDB not initialized")
    
    try:
        collections = chroma_client.list_collections()
        return {
            "collections": [collection.name for collection in collections]
        }
    except Exception as e:
        logger.error(f"Error listing collections: {e}")
        raise HTTPException(status_code=500, detail=str(e))
