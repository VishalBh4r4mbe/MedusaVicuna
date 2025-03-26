import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from typing import List, Tuple, Dict, Optional, Any
import time
from utils.logger import logger
import config
from model.llama_cpp_model import LlamaCppModel

class MedusaHead(nn.Module):
    """
    A single Medusa head for speculative decoding.
    Each head predicts K future tokens given the current context.
    """
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, future_pos: int):
        """
        Initialize a Medusa head.
        
        Args:
            input_dim: Dimension of input embeddings
            output_dim: Output dimension (typically vocab size)
            hidden_dim: Hidden dimension size
            future_pos: Which future position this head predicts (1=next token, 2=token after next, etc.)
        """
        super().__init__()
        self.future_pos = future_pos
        
        # Simple MLP design
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Medusa head.
        
        Args:
            hidden_states: Hidden states from the base model [batch_size, seq_len, hidden_dim]
            
        Returns:
            Logits for the predicted tokens [batch_size, seq_len, vocab_size]
        """
        return self.mlp(hidden_states)

class MedusaModel():
    """
    Implementation of the Medusa speculative decoding model.
    Uses a base LLM with multiple prediction heads to generate text faster.
    """
    def __init__(
        self, 
        base_model: LlamaCppModel,
        k_heads: int = config.MEDUSA_K_HEADS,
        max_tokens_per_step: int = config.MEDUSA_MAX_TOKENS_PER_STEP,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        weights_path: str = config.MEDUSA_WEIGHTS_PATH,
    ):
        """
        Initialize the Medusa model.
        
        Args:
            base_model: The base LLM model to use
            k_heads: Number of Medusa heads to use
            max_tokens_per_step: Maximum number of tokens to generate in a single step
            device: Device to run the model on
            weights_path: Path to pre-trained Medusa head weights, if available
        """
        self.base_model = base_model
        self.k_heads = k_heads
        self.max_tokens_per_step = max_tokens_per_step
        self.device = device
        self.hidden_dim = 4096  # Typical hidden dim for Vicuna-7b
        
        # Since we're using a custom implementation, we'll create the heads directly
        # Each head predicts tokens at different future positions
        self.heads = []
        for i in range(1, k_heads + 1):
            head = MedusaHead(
                input_dim=self.hidden_dim,
                output_dim=self.base_model.vocab_size,
                hidden_dim=self.hidden_dim // 2,
                future_pos=i
            ).to(device)
            self.heads.append(head)
            if eval:
                self.heads[-1].eval()
        
        # Load pre-trained weights or initialize from scratch
        self._initialize_heads(weights_path)
        
        logger.info(f"Initialized Medusa model with {k_heads} heads")
    def _initialize_heads(self, weights_path=config.MEDUSA_WEIGHTS_PATH):
        """
        Initialize the Medusa heads with pre-trained weights or from scratch.
        
        Args:
            weights_path: Path to pre-trained weights, if available
        """
        logger.info(f"weights_path:{weights_path}, is_avl:{os.path.exists(weights_path)}")
        if weights_path and os.path.exists(weights_path):
            try:
                logger.info(f"Loading Medusa heads weights from {weights_path}")
                saved_weights = torch.load(weights_path, map_location=self.device)
                
                if isinstance(saved_weights, list) and len(saved_weights) == len(self.heads):
                    # Load weights for each head
                    for i, (head, state_dict) in enumerate(zip(self.heads, saved_weights)):
                        head.load_state_dict(state_dict)
                        logger.info(f"Loaded pre-trained weights for Medusa head {i+1}")
                else:
                    logger.error(f"Incompatible weights file: expected {len(self.heads)} head weights, got {len(saved_weights) if isinstance(saved_weights, list) else 'unknown format'}")
                    self._initialize_random_weights()
            except Exception as e:
                logger.error(f"Error loading Medusa weights: {str(e)}")
                self._initialize_random_weights()
        else:
            logger.info("No pre-trained weights provided, using random initialization")
            self._initialize_random_weights()
            
    def _initialize_random_weights(self):
        """Initialize heads with random weights using Kaiming initialization."""
        for i, head in enumerate(self.heads):
            logger.info(f"Initializing Medusa head {i+1} with random weights")
            
            # Apply Kaiming initialization
            for m in head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def _get_hidden_states(self, tokens: List[int]) -> np.ndarray:
        """
        Get hidden states from the base model for a given sequence of tokens.
        In a real implementation, this would extract embeddings from llama.cpp.
        
        Args:
            tokens: List of token IDs
            
        Returns:
            Hidden states array
        """
        # This is a placeholder - in a real implementation we would
        # extract embeddings from the llama.cpp model
        # For this assignment, we'll simulate this with random values
        hidden_dim = self.hidden_dim
        hidden_states = np.random.randn(1, len(tokens), hidden_dim).astype(np.float32)
        return hidden_states
    
    def _predict_with_heads(self, tokens: List[int]) -> List[List[int]]:
        """
        Use Medusa heads to predict future tokens.
        
        Args:
            tokens: Current sequence of tokens
            
        Returns:
            List of candidate token sequences
        """
        # Get hidden states from base model
        hidden_states = self._get_hidden_states(tokens)
        hidden_states_tensor = torch.tensor(hidden_states).to(self.device)
        
        # Get predictions from each head
        candidate_sequences = []
        
        # Build tree of possible continuations
        for depth in range(1, self.max_tokens_per_step + 1):
            if depth > len(self.heads):
                break
                
            # Use the appropriate head for this depth
            head = self.heads[depth - 1]
            
            # Get logits from the head
            logits = head(hidden_states_tensor[:, -1:, :])  # Only use the last token's hidden state
            
            # Sample from the logits
            probs = F.softmax(logits.squeeze(1), dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            
            # Create a new candidate sequence
            candidate = tokens + [next_token]
            candidate_sequences.append(candidate)
        
        return candidate_sequences
    
    def _verify_candidates(self, tokens: List[int], candidates: List[List[int]]) -> Tuple[List[int], int]:
        """
        Verify candidate token sequences against the base model.
        
        Args:
            tokens: Original tokens
            candidates: List of candidate token sequences
            
        Returns:
            Tuple of (accepted tokens, number of accepted tokens)
        """
        # For each candidate, check if the base model would generate the same tokens
        accepted_tokens = []
        current_tokens = tokens.copy()
        
        for candidate in candidates:
            # The token to verify is the last token in the candidate
            token_to_verify = candidate[-1]
            logger.info(f"token_to_verify:{token_to_verify}")
            # Check if base model predicts the same token
            logger.info(f"{self.base_model}")
            try:
                next_token = self.base_model.get_next_token(current_tokens)
            except Exception as e:
                print(e)
                raise
            logger.info(f"get_next_token:{next_token}")

            if next_token == token_to_verify:
                # Accept this token
                accepted_tokens.append(token_to_verify)
                current_tokens.append(token_to_verify)
            else:
                # Stop verification when we hit a mismatch
                break
        
        return accepted_tokens, len(accepted_tokens)
    
    def generate(
        self, 
        prompt: str, 
        max_tokens: int = 100, 
        temperature: float = 0.8,
        top_p: float = 0.95
    ) -> str:
        """
        Generate text using Medusa speculative decoding.
        
        Args:
            prompt: Input text prompt
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling threshold
            
        Returns:
            Generated text
        """
        start_time = time.time()
        
        # Format and tokenize the prompt
        formatted_prompt = self.base_model._format_prompt(prompt)
        tokens = self.base_model.tokenize(formatted_prompt)
        
        # Track metrics
        total_tokens_generated = 0
        total_steps = 0
        total_tokens_accepted = 0
        
        # Generate until we reach max_tokens
        generated_tokens = []
        
        while len(generated_tokens) < max_tokens:
            total_steps += 1
            logger.info(f"--total_steps--{total_steps}--")
            # 1. Use Medusa heads to predict future tokens
            candidates = self._predict_with_heads(tokens + generated_tokens)
            # logger.info(f"candidates = {candidates}")
            # 2. Verify candidates with base model
            accepted_tokens, num_accepted = self._verify_candidates(
                tokens + generated_tokens, 
                candidates
            )
            logger.info(f"accepted_tokens, num_accepted:{accepted_tokens}, {num_accepted}")
            # 3. Add accepted tokens to result
            generated_tokens.extend(accepted_tokens)
            total_tokens_accepted += num_accepted
            
            # 4. If no tokens were accepted, generate one with base model
            if num_accepted == 0:
                next_token = self.base_model.get_next_token(
                    tokens + generated_tokens,
                    temperature=temperature,
                    top_p=top_p
                )
                generated_tokens.append(next_token)
                total_tokens_generated += 1
            
            # Check if we've generated enough tokens
            if len(generated_tokens) >= max_tokens:
                break
        
        # Decode generated tokens
        completion_text = self.base_model.decode(generated_tokens)
        
        # Log metrics
        duration = time.time() - start_time
        tokens_per_second = len(generated_tokens) / duration
        acceptance_rate = total_tokens_accepted / len(generated_tokens)
        
        logger.info(f"Generated {len(generated_tokens)} tokens in {duration:.2f}s ({tokens_per_second:.2f} tokens/s)")
        logger.info(f"Acceptance rate: {acceptance_rate:.2%}")
        logger.info(f"Steps: {total_steps}, Avg tokens per step: {len(generated_tokens)/total_steps:.2f}")
        
        return completion_text.strip()
    
    def generate_batch(
        self, 
        prompts: List[str], 
        max_tokens: int = 100,
        temperature: float = 0.8,
        top_p: float = 0.95
    ) -> List[str]:
        """
        Generate text for a batch of prompts using Medusa speculative decoding.
        
        Args:
            prompts: List of input prompts
            max_tokens: Maximum number of tokens to generate per prompt
            temperature: Sampling temperature
            top_p: Top-p sampling threshold
            
        Returns:
            List of generated texts
        """
        # Process each prompt sequentially
        results = []
        for prompt in prompts:
            logger.info(f"Generating for prompt:{prompt}")
            result = self.generate(
                prompt, 
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p
            )
            results.append(result)
        
        return results
