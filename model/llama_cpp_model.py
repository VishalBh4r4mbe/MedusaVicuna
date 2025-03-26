# model/llama_cpp_model.py
import os
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from llama_cpp import Llama
from utils.logger import logger
import config

class LlamaCppModel:
    """
    Wrapper for llama.cpp model that provides inference capabilities.
    """
    def __init__(
        self,
        model_path: str = config.MODEL_PATH,
        n_ctx: int = config.MODEL_CONTEXT_SIZE,
        n_gpu_layers: int = config.N_GPU_LAYERS,
        n_threads: int = config.N_THREADS,
        verbose: bool = False
    ):
        """
        Initialize the LlamaCppModel.

        Args:
            model_path: Path to the GGUF model file
            n_ctx: Maximum context size
            n_gpu_layers: Number of layers to offload to GPU
            n_threads: Number of threads to use
            verbose: Enable verbose logging
        """
        logger.info(f"Initializing LlamaCppModel with n_gpu_layers: {n_gpu_layers}")
        if n_gpu_layers == -1:
            logger.info("Attempting to offload all layers to GPU")
        elif n_gpu_layers > 0:
            logger.info(f"Offloading {n_gpu_layers} layers to GPU")
        else:
            logger.info("Not offloading any layers to GPU")


        # Load the model using llama.cpp
        self.model = Llama.from_pretrained(
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            verbose=verbose,
            repo_id="LeyaGao/vicuna-7b-v1.5-Q4_K_M-GGUF",
            filename="vicuna-7b-v1.5-q4_k_m.gguf",
            use_mlock=True,  # Lock memory to prevent swapping
            use_mmap=True,    # Use memory mapping for faster loading
            embedding=True
        )
        logger.info(f"Model loaded successfully, n_gpu_layers: {n_gpu_layers}")
        if n_gpu_layers > 0 or n_gpu_layers == -1:
            logger.info("GPU layers are enabled.")
        else:
            logger.info("GPU layers are not enabled.")

        # Model metadata
        self.vocab_size = self.model.n_vocab()
        self.context_length = n_ctx
        
    def _format_prompt(self, prompt: str) -> str:
        """Format the prompt for the Vicuna model."""
        return f"USER: {prompt}\nASSISTANT:"
    
    def embed(self, text: str) -> np.ndarray:
        """Get embeddings for a text."""
        return self.model.embed(text)
    
    def tokenize(self, text: str) -> List[int]:
        """Tokenize a text string into token IDs."""
        return self.model.tokenize(text.encode('utf-8'))
    
    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs back to a text string."""
        return self.model.detokenize(tokens).decode('utf-8', errors='replace')
    
    def get_logits(self, tokens: List[int]) -> np.ndarray:
        """
        Get logits for the next token prediction given a sequence of tokens.
        
        Args:
            tokens: List of token IDs
            
        Returns:
            Logits array of shape (vocab_size,)
        """
        # Evaluate tokens
        self.model.eval(tokens)
        logger.info(f"YO{logits}")

        # Get logits for the last token - ensure it's a numpy array
        logits = self.model.eval_logits
        logger.info(f"{logits}")
        # Check if it's a deque or other collection type
        if hasattr(logits, 'pop'):  # Checking for a collection-like interface
            logger.info(f"Converting logits from {type(logits)} to numpy array")
            # Convert deque to list then to numpy array
            logits_list = list(logits)
            
            # Debug information
            logger.info(f"Logits list length: {len(logits_list)}")
            
            # If it's a nested structure, we might need just the first element
            if len(logits_list) == 1 and hasattr(logits_list[0], '__len__'):
                logits_array = np.array(logits_list[0])
                logger.info(f"Using nested logits with shape: {logits_array.shape}")
            else:
                logits_array = np.array(logits_list)
                logger.info(f"Using flat logits with shape: {logits_array.shape}")
                
            # Ensure we have the right shape for sampling
            if logits_array.ndim > 1:
                # If we have a 2D array, take the last row (token)
                logits_array = logits_array[-1, :]
                logger.info(f"Taking last row, new shape: {logits_array.shape}")
        else:
            logits_array = np.array(logits)
            logger.info(f"Direct logits with shape: {logits_array.shape}")
            
        # Ensure we have a 1D array for sampling
        if logits_array.ndim > 1:
            logits_array = logits_array.flatten()
            logger.info(f"Flattened to shape: {logits_array.shape}")
            
        # Ensure array is valid for sampling
        if np.isnan(logits_array).any() or np.isinf(logits_array).any():
            logger.warning("Found NaN or Inf in logits, replacing with small values")
            logits_array = np.nan_to_num(logits_array, nan=-1e10, posinf=1e10, neginf=-1e10)
            
        return logits_array
    
    def get_next_token(self, tokens: List[int], temperature: float = 0.8, top_p: float = 0.95) -> int:
        """
        Sample the next token given a sequence of tokens.
        
        Args:
            tokens: List of token IDs
            temperature: Sampling temperature (higher = more random)
            top_p: Top-p sampling threshold
            
        Returns:
            Next token ID
        """
        # Use our get_logits method to handle logits properly
        logits = self.get_logits(tokens)
        logger.info("Getting ")
        # Convert logits to probabilities with temperature and top_p sampling
        probs = self._get_probs(logits, temperature, top_p)
        
        # Sample a token from the probability distribution
        next_token = self._sample_token_from_probs(probs)
        
        logger.info(f"Sampled token: {next_token}")
        return next_token
    
    def _get_probs(self, logits: np.ndarray, temperature: float, top_p: float) -> np.ndarray:
        """Convert logits to probability distribution with temperature and top_p sampling."""
        # Apply temperature
        if temperature > 0:
            logits = logits / temperature
        
        # Convert to probabilities
        probs = np.exp(logits - np.max(logits))
        probs = probs / np.sum(probs)
        
        # Apply top_p sampling (nucleus sampling)
        if top_p < 1.0:
            sorted_probs = np.sort(probs)[::-1]
            cumulative_probs = np.cumsum(sorted_probs)
            cutoff = sorted_probs[np.argmax(cumulative_probs >= top_p)]
            probs[probs < cutoff] = 0
            probs = probs / np.sum(probs)  # Re-normalize
            
        return probs
    
    def _sample_token_from_probs(self, probs: np.ndarray) -> int:
        """Sample a token from a probability distribution."""
        try:
            return int(np.random.choice(len(probs), p=probs))
        except ValueError as e:
            # Handle case where probs don't sum to 1 due to numerical issues
            logger.warning(f"Error sampling from probs: {e}")
            return int(np.argmax(probs))
    
    def generate(
        self, 
        prompt: str, 
        max_tokens: int = 100, 
        temperature: float = 0.8, 
        top_p: float = 0.95,
        stop: Optional[List[str]] = None
    ) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input text prompt
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_p: Top-p sampling threshold
            stop: List of stop sequences
            
        Returns:
            Generated text
        """
        formatted_prompt = self._format_prompt(prompt)
        
        # Tokenize the prompt
        tokens = self.tokenize(formatted_prompt)
        
        # Generate completion
        # Generate with the model - properly handle the generator
        completion_generator = self.model.generate(
            tokens,
            temp=temperature,
            top_p=top_p,
            stop=stop
        )
        
        # Convert generator to list
        completion_tokens_list = []
        for token in completion_generator:
            completion_tokens_list.append(token)
            
        logger.info(f"Type of completion_tokens_list: {type(completion_tokens_list)}")
        logger.info(f"Collected {len(completion_tokens_list)} tokens")

        # Get only the newly generated tokens (not the prompt)
        new_tokens = completion_tokens_list[len(tokens):] if len(completion_tokens_list) > len(tokens) else []
        logger.info(f"new_tokens: {new_tokens}")
        if not isinstance(new_tokens, list):
            logger.error("new_tokens is not a list!")
        else:
            logger.info("new_tokens is a list")

        # Convert tokens back to text
        completion_text = self.decode(new_tokens)
        
        logger.info(f"generate completion_text: {completion_text}")
        return completion_text.strip()
    
    def generate_batch(
        self, 
        prompts: List[str], 
        max_tokens: int = 100,
        temperature: float = 0.8,
        top_p: float = 0.95
    ) -> List[str]:
        """
        generate_batch
        Generate text for a batch of prompts.
        
        Args:
            prompts: List of input prompts
            max_tokens: Maximum number of tokens to generate per prompt
            temperature: Sampling temperature
            top_p: Top-p sampling threshold
            
        Returns:
            List of generated texts
        """
        logger.info(f"generate_batch prompts: {prompts}, max_tokens: {max_tokens}, temperature: {temperature}, top_p: {top_p}")
        # Process each prompt sequentially (llama.cpp doesn't support true batching)
        results = []
        for prompt in prompts:
            result = self.generate(
                prompt, 
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p
            )
            results.append(result)
        
        return results
