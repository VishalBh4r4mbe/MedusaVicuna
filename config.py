# config.py
import os
from typing import Dict, Any
import json

# Service information
VERSION = "1.0.0"
SERVICE_NAME = "llm-medusa-service"

# Model configuration
MODEL_NAME = "lmsys/vicuna-7b"
MODEL_PATH = os.environ.get("MODEL_PATH", "models/vicuna-7b-v1.3.Q4_K_M.gguf")
BASE_MODEL_PATH = MODEL_PATH  # For training Medusa heads
MODEL_CONTEXT_SIZE = int(os.environ.get("MODEL_CONTEXT_SIZE", "2048"))
N_GPU_LAYERS = int(os.environ.get("N_GPU_LAYERS", "1000")) # Set to -1 to offload all layers to GPU
N_THREADS = int(os.environ.get("N_THREADS", "8"))  # Increased for better performance
GPU_MEMORY_UTILIZATION = float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.9"))  # GPU memory utilization (0.0 to 1.0)

# Medusa configuration
MEDUSA_K_HEADS = int(os.environ.get("MEDUSA_K_HEADS", "5"))
MEDUSA_MAX_TOKENS_PER_STEP = int(os.environ.get("MEDUSA_MAX_TOKENS_PER_STEP", "3"))
MEDUSA_WEIGHTS_PATH = os.environ.get("MEDUSA_WEIGHTS_PATH", "C:\\Users\\visha\\Documents\\src\\project\\medusa_heads_weights\\medusa_heads_epoch_1_batch_140.pth")

# Dynamic batching configuration
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "4"))
MAX_WAIT_TIME = float(os.environ.get("MAX_WAIT_TIME", "0.1"))
MAX_CONCURRENT_BATCHES = int(os.environ.get("MAX_CONCURRENT_BATCHES", "4"))

# Server configuration
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8000"))
WORKERS = int(os.environ.get("WORKERS", "1"))

# Log level
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")

def get_config() -> Dict[str, Any]:
    """Get the current configuration as a dictionary."""
    return {
        "version": VERSION,
        "service_name": SERVICE_NAME,
        "model": {
            "name": MODEL_NAME,
            "path": MODEL_PATH,
            "context_size": MODEL_CONTEXT_SIZE,
            "n_gpu_layers": N_GPU_LAYERS,
            "n_threads": N_THREADS
        },
        "medusa": {
            "k_heads": MEDUSA_K_HEADS,
            "max_tokens_per_step": MEDUSA_MAX_TOKENS_PER_STEP
        },
        "batching": {
            "batch_size": BATCH_SIZE,
            "max_wait_time": MAX_WAIT_TIME,
            "max_concurrent_batches": MAX_CONCURRENT_BATCHES
        },
        "server": {
            "host": HOST,
            "port": PORT,
            "workers": WORKERS
        },
        "log_level": LOG_LEVEL
    }

def print_config():
    """Print the current configuration."""
    config = get_config()
    print(json.dumps(config, indent=2))
