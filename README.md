FastAPI LLM Service with Medusa Head
This project implements a FastAPI service that serves a Language Model (LLM) with a medusa head architecture for optimized inference. The implementation focuses on three key optimizations:

Model Compilation - Using llama.cpp to optimize inference speed
Speculative Decoding - Implementing a medusa head architecture to improve generation speed
Dynamic Batching - Efficiently handling multiple concurrent requests

📋 Key Features

Optimized Inference with llama.cpp compilation
Speculative Decoding with medusa head architecture
Dynamic Batching for improved throughput
FastAPI Service with production-ready configuration
Comprehensive Testing with pytest
Docker Support for easy deployment

🏗️ Architecture
The service consists of several key components:

Base Model (LlamaCppModel): Wraps the llama.cpp library to provide optimized inference
Medusa Model (MedusaModel): Implements speculative decoding using multiple prediction heads
Dynamic Batcher (DynamicBatcher): Handles concurrent requests with efficient batching
FastAPI Service: Provides REST API endpoints for text generation

Medusa Head Architecture
The medusa head architecture is a speculative decoding technique that uses multiple smaller decoders (heads) to predict future tokens in parallel, improving generation speed. Here's how it works:

The base model generates the initial tokens
Multiple medusa heads predict possible future tokens
The base model verifies which predictions are correct
Correct predictions are accepted without re-computing them

This approach significantly improves generation speed by speculating on future tokens and accepting them when correct, reducing the number of model forward passes required.
🚀 Installation

Clone this repository:
bashCopygit clone https://github.com/yourusername/fastapi-llm-service.git
cd fastapi-llm-service

Install dependencies:
bashCopypip install -r requirements.txt

Download and convert the Vicuna model:
bashCopymkdir -p models
# Download model from Hugging Face
# Convert to GGUF format using llama.cpp tools


⚙️ Configuration
Configuration is managed through environment variables and the config.py file:
VariableDescriptionDefaultMODEL_PATHPath to the GGUF model filemodels/vicuna-7b-v1.3.Q4_K_M.ggufMODEL_CONTEXT_SIZEMaximum context size2048N_GPU_LAYERSNumber of layers to offload to GPU25N_THREADSNumber of threads to use4MEDUSA_K_HEADSNumber of Medusa heads5MEDUSA_MAX_TOKENS_PER_STEPMaximum tokens per step3BATCH_SIZEMaximum batch size4MAX_WAIT_TIMEMaximum wait time for batching0.1MAX_CONCURRENT_BATCHESMaximum concurrent batches4HOSTServer host0.0.0.0PORTServer port8000WORKERSNumber of Uvicorn workers1LOG_LEVELLog levelINFO
🏃 Usage
Starting the Service
bashCopypython main.py
API Endpoints

GET /: Root endpoint with service information
GET /health: Health check endpoint
POST /generate: Generate text from a prompt

Example Request
bashCopycurl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Write a poem about artificial intelligence.",
    "max_tokens": 100,
    "temperature": 0.8,
    "top_p": 0.95
  }'
🧪 Testing
Run the test suite with pytest:
bashCopypytest tests/
Performance Benchmarks
The tests include performance benchmarks that compare:

Base model vs. Medusa model inference speed
Sequential vs. Batched request processing
Different batch sizes and configurations

🔬 Technical Details
Model Compilation with llama.cpp
We use llama.cpp to compile and optimize the Vicuna-7B model, which offers several advantages:

Quantization: Reduces model size and memory usage through quantization
Optimized Kernels: Uses highly optimized C++ code for inference
GPU Acceleration: Efficiently offloads computation to the GPU

Speculative Decoding with Medusa Heads
The medusa head architecture improves generation speed by:

Parallelism: Generating multiple candidate tokens in parallel
Speculative Execution: Accepting correct predictions without re-computation
Adaptive Generation: Dynamically adjusting to different contexts

Dynamic Batching
Our dynamic batching system:

Collects multiple requests into batches
Processes batches concurrently using a thread pool
Balances latency and throughput with configurable parameters
Optimizes GPU utilization by batching similar requests together

🔮 Future Improvements

Distributed Serving: Scale to multiple machines
Quantization Tuning: Experiment with different quantization methods
Streaming Responses: Implement token-by-token streaming
Custom Heads Training: Train specialized medusa heads for different tasks
Adaptive Batching: Dynamically adjust batch parameters based on load

📚 References

llama.cpp GitHub Repository
FastAPI Documentation
Vicuna Model
Medusa: Simple Framework for Accelerating LLM Generation with Multiple Decoding Heads (Paper)