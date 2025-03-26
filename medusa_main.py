import torch
import time
from utils.logger import logger
import config
from model.llama_cpp_model import LlamaCppModel
from medusa_code import  MedusaModel

def main():
    # Initialize base model
    base_model = LlamaCppModel(
        model_path=config.MODEL_PATH,
        n_ctx=config.MODEL_CONTEXT_SIZE,
        n_gpu_layers=config.N_GPU_LAYERS,
        n_threads=config.N_THREADS
    )
    
    # Initialize Medusa model
    medusa_model = MedusaModel(base_model)
    
    # Generate text
    prompt = "Write a poem about artificial intelligence."
    start_time = time.time()
    completion = medusa_model.generate(prompt, max_tokens=100)
    duration = time.time() - start_time
    
    logger.info(f"Completion: {completion}")
    logger.info(f"Generated in {duration:.2f}s")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)

