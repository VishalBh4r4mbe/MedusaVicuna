import time
import asyncio
import uuid
from typing import Dict, List, Any, Optional, Tuple
from queue import Queue, Empty
from threading import Thread
import concurrent.futures
from utils.logger import logger
import config

class DynamicBatcher:
    """
    Dynamic batching system for efficient handling of concurrent LLM requests.
    Groups requests together to improve throughput while maintaining low latency.
    """
    def __init__(
        self, 
        model,
        batch_size: int = config.BATCH_SIZE,
        max_wait_time: float = config.MAX_WAIT_TIME,
        max_concurrent_batches: int = config.MAX_CONCURRENT_BATCHES
    ):
        """
        Initialize the dynamic batcher.
        
        Args:
            model: The model to use for generation (base model or Medusa model)
            batch_size: Maximum batch size for processing
            max_wait_time: Maximum time to wait before processing a batch
            max_concurrent_batches: Maximum number of batches to process concurrently
        """
        self.model = model
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.max_concurrent_batches = max_concurrent_batches
        
        # Queue for incoming requests
        self.request_queue = asyncio.Queue()
        
        # Results dictionary for retrieving generated text
        self.results = {}
        
        # Event loop
        self.loop = asyncio.get_event_loop()
        
        # Thread pool for CPU-bound tasks
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent_batches)
        
        # Start the batch processor
        self.processor_task = asyncio.create_task(self._process_batches())
        
        logger.info(f"Dynamic batcher initialized with batch_size={batch_size}, max_wait_time={max_wait_time}s")
    
    async def generate(
        self, 
        request_id: str, 
        prompt: str, 
        max_tokens: int = 100,
        temperature: float = 0.8,
        top_p: float = 0.95
    ) -> str:
        """
        Submit a generation request to the batcher and wait for the result.
        
        Args:
            request_id: Unique identifier for the request
            prompt: Input text prompt
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling threshold
            
        Returns:
            Generated text
        """
        # Create future for retrieving result
        future = asyncio.Future()
        self.results[request_id] = future
        
        # Add request to queue
        await self.request_queue.put((
            request_id,
            prompt,
            max_tokens,
            temperature,
            top_p
        ))
        
        # Wait for result
        return await future
    
    async def _process_batches(self):
        """Process batches of requests using the model."""
        while True:
            batch = []
            start_time = time.time()
            
            # Collect requests until batch is full or max wait time is reached
            try:
                # Get the first request
                request = await asyncio.wait_for(
                    self.request_queue.get(),
                    timeout=self.max_wait_time
                )
                batch.append(request)
                
                # Try to get more requests up to batch_size
                while len(batch) < self.batch_size:
                    try:
                        request = await asyncio.wait_for(
                            self.request_queue.get(),
                            timeout=max(0, self.max_wait_time - (time.time() - start_time))
                        )
                        batch.append(request)
                    except asyncio.TimeoutError:
                        # No more requests within timeout
                        break
            
            except asyncio.TimeoutError:
                # No requests at all within timeout
                await asyncio.sleep(0.01)  # Avoid busy waiting
                continue
            
            if batch:
                # Process batch in a separate thread
                asyncio.create_task(self._execute_batch(batch))
    
    async def _execute_batch(self, batch: List[Tuple]):
        """
        Execute a batch of requests using the model.
        
        Args:
            batch: List of (request_id, prompt, max_tokens, temperature, top_p) tuples
        """
        # Extract batch components
        request_ids, prompts, max_tokens_list, temperatures, top_ps = zip(*batch)
        
        # Use the most conservative max_tokens value for the entire batch
        batch_max_tokens = min(max_tokens_list)
        
        # Log batch information
        logger.info(f"Processing batch of {len(batch)} requests")

        try:
            # Process batch in a thread pool to avoid blocking
            logger.info(f"Temperature: {temperatures[0]}, Top_p: {top_ps[0]}")
            results = await self.loop.run_in_executor(
                self.executor,
                self._generate_batch,
                prompts,
                batch_max_tokens,
                float(temperatures[0]), # Use first temperature for simplicity, ensure it's a float
                float(top_ps[0]) # Use first top_p for simplicity, ensure it's a float
            )
            # Set results for each request
            for request_id, result in zip(request_ids, results):
                logger.info(f"Result for request {request_id}: {result[:30]}...")
                if request_id in self.results:
                    self.results[request_id].set_result(result)
                    del self.results[request_id]

        except Exception as e:
            # Handle errors
            logger.error(f"Error processing batch: {str(e)}")
            for request_id in request_ids:
                if request_id in self.results:
                    self.results[request_id].set_exception(e)
                    del self.results[request_id]

    def _generate_batch(
        self,
        prompts: List[str],
        max_tokens: int,
        temperature: float,
        top_p: float
    ) -> List[str]:
        """
        Generate text for a batch of prompts.
        This runs in a separate thread.

        Args:
            prompts: List of input prompts
            max_tokens: Maximum number of tokens to generate per prompt
            temperature: Sampling temperature
            top_p: Top-p sampling threshold

        Returns:
            List of generated texts
        """
        return self.model.generate_batch(
            prompts=prompts,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
