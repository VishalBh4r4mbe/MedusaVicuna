# main.py
import asyncio
import uvicorn
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
import os
import time
from typing import Dict, Any

from model.llama_cpp_model import LlamaCppModel
from model.medusa import MedusaModel
from model.dynamic_batcher import DynamicBatcher
from api.routes import router
from utils.logger import logger, Timer
import config

# Initialize the FastAPI app
app = FastAPI(
    title="LLM Medusa Service",
    description="A FastAPI service for serving an optimized LLM with a medusa head",
    version=config.VERSION
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and batcher
base_model = None
medusa_model = None
batcher = None

@app.on_event("startup")
async def startup_event():
    """Initialize models and batcher on startup."""
    global base_model, medusa_model, batcher
    
    # Print configuration
    config.print_config()
    
    # Initialize base model
    logger.info("Initializing base model...")
    with Timer("Base model initialization"):
        base_model = LlamaCppModel(
            model_path=config.MODEL_PATH,
            n_ctx=config.MODEL_CONTEXT_SIZE,
            n_gpu_layers=config.N_GPU_LAYERS,
            n_threads=config.N_THREADS
        )
    
    # Initialize Medusa model
    logger.info("Initializing Medusa model...")
    with Timer("Medusa model initialization"):
        # Check if Medusa weights exist
        weights_path = config.MEDUSA_WEIGHTS_PATH
        if os.path.exists(weights_path):
            logger.info(f"Found Medusa weights at {weights_path}")
        else:
            logger.warning(f"Medusa weights not found at {weights_path}, will use random initialization")
            weights_path = None
            
        medusa_model = MedusaModel(
            base_model=base_model,
            k_heads=config.MEDUSA_K_HEADS,
            max_tokens_per_step=config.MEDUSA_MAX_TOKENS_PER_STEP,
            weights_path=weights_path  # Pass weights path to the model
        )
    
    # Initialize dynamic batcher
    logger.info("Initializing dynamic batcher...")
    batcher = DynamicBatcher(
        model=medusa_model,
        batch_size=config.BATCH_SIZE,
        max_wait_time=config.MAX_WAIT_TIME,
        max_concurrent_batches=config.MAX_CONCURRENT_BATCHES
    )
    
    logger.info("Service started successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    logger.info("Shutting down service...")

# Include the API routes
app.include_router(router)

# Add a root endpoint
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "LLM Medusa Service",
        "version": config.VERSION,
        "status": "ok",
        "model": config.MODEL_NAME
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=config.HOST,
        port=config.PORT,
        log_level=config.LOG_LEVEL.lower(),
        workers=config.WORKERS
    )
