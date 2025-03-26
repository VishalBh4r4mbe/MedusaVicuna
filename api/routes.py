# api/routes.py
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import time
import uuid
from typing import List, Dict, Any, Optional
import asyncio
from utils.logger import logger
import config

router = APIRouter()

class GenerationRequest(BaseModel):
    """Request model for text generation."""
    prompt: str = Field(..., description="Input text prompt")
    max_tokens: int = Field(default=100, ge=1, le=2048, description="Maximum number of tokens to generate")
    temperature: float = Field(default=0.8, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(default=0.95, ge=0.0, le=1.0, description="Top-p sampling threshold")
    stream: bool = Field(default=False, description="Whether to stream the response")

class GenerationResponse(BaseModel):
    """Response model for text generation."""
    id: str = Field(..., description="Unique ID for the request")
    text: str = Field(..., description="Generated text")
    prompt_tokens: int = Field(..., description="Number of tokens in the prompt")
    completion_tokens: int = Field(..., description="Number of tokens in the completion")
    time_taken: float = Field(..., description="Time taken to generate text in seconds")

class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(..., description="Service status")
    model: str = Field(..., description="Model name")
    version: str = Field(..., description="Service version")

def get_batcher():
    """Get the dynamic batcher from the app state."""
    from main import batcher
    return batcher

@router.post("/generate", response_model=GenerationResponse)
async def generate(request: GenerationRequest, batcher=Depends(get_batcher)):
    """
    Generate text from a prompt.
    
    Args:
        request: Generation request
        
    Returns:
        Generation response
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    logger.info(f"Request {request_id}: {request.prompt[:50]}...")
    
    try:
        # Submit request to batcher
        result = await batcher.generate(
            request_id=request_id,
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p
        )
        
        # Calculate tokens (in a real implementation, we would get this from the model)
        # Here we use a simple approximation
        prompt_tokens = len(request.prompt.split())
        completion_tokens = len(result.split())
        
        time_taken = time.time() - start_time
        
        logger.info(f"Request {request_id} completed in {time_taken:.2f}s")
        
        # Return response
        return GenerationResponse(
            id=request_id,
            text=result,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            time_taken=time_taken
        )
    
    except Exception as e:
        logger.error(f"Error processing request {request_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check the health of the service."""
    return HealthResponse(
        status="ok",
        model=config.MODEL_NAME,
        version=config.VERSION
    )