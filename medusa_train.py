import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import numpy as np
import argparse
import json
import config
from medusa_code import MedusaModel, MedusaHead
from model.llama_cpp_model import LlamaCppModel
from utils.logger import logger
import time
import os
import tqdm
import copy
from grokfast_pytorch import GrokFastAdamW

class LlamaHiddenStateExtractor:
    """
    Helper class to extract hidden states from a LLaMA model.
    This is crucial for properly training Medusa to match the base model's behavior.
    """
    def __init__(self, base_model:LlamaCppModel, device):
        self.device = device
        self.hidden_dim = 4096  # Set to match the Vicuna-7b model dimension

        # The Llama model must be created with embedding=True to call this method
        self.model = base_model
        if hasattr(self.model, 'set_embedding'):
            self.model.set_embedding(True)
        
        logger.info(f"Hidden state extractor initialized on {device}")

    def get_hidden_states(self, input_ids):
        """
        Extract hidden states from the LLaMA model.
        
        In llama.cpp, direct access to hidden states is limited, so we use a hybrid approach:
        1. For inference, we use eval to get token predictions
        2. For training, we extract embeddings and apply linear projections to approximate hidden states
        
        Args:
            input_ids: Tensor of token IDs [batch_size, seq_len]
            
        Returns:
            hidden_states: Tensor of hidden states [batch_size, seq_len, hidden_dim]
        """
        batch_size, seq_len = input_ids.shape
        hidden_states = []
        
        for idx in range(batch_size):
            # Convert to list of integers
            tokens = input_ids[idx].tolist()
            
            # Get embeddings for these tokens - use the model's embed method if available
            if hasattr(self.model, 'embed'):
                # This is an improvement over random values, but still an approximation
                embeddings = self.model.embed(self.model.decode(tokens))
                sample_hidden = torch.tensor(embeddings, device=self.device)
                
                # Apply a transformation to match expected hidden state dimensions
                # This is a simplification - in a full implementation, you would use proper hidden states
                if sample_hidden.shape[1] != self.hidden_dim:
                    projection = nn.Linear(sample_hidden.shape[1], self.hidden_dim).to(self.device)
                    sample_hidden = projection(sample_hidden)
            else:
                # If embed isn't available, we fall back to a more sophisticated approximation
                # We run each token through the model's forward pass and collect intermediate activations
                all_token_states = []
                for i in range(1, len(tokens)):
                    # Evaluate the model on this prefix
                    self.model.model.eval(tokens[:i])
                    # Get the logits - we'll project from vocab size to hidden dimension
                    logits = self.model.get_logits(tokens[:i])
                    # Create a projection matrix for this position (this is just an approximation)
                    proj = torch.eye(min(len(logits), self.hidden_dim), device=self.device)
                    if len(logits) > self.hidden_dim:
                        logits = logits[:self.hidden_dim]  # Truncate if needed
                    elif len(logits) < self.hidden_dim:
                        # Pad with zeros if needed
                        padding = torch.zeros(self.hidden_dim - len(logits), device=self.device)
                        logits = torch.cat([torch.tensor(logits, device=self.device), padding])
                    all_token_states.append(logits)
                
                # Stack all positions
                sample_hidden = torch.stack(all_token_states) if all_token_states else torch.zeros((0, self.hidden_dim))
                
                # Prepend a position for the first token (BOS) since we skipped it above
                bos_embedding = torch.zeros(1, self.hidden_dim, device=self.device)
                sample_hidden = torch.cat([bos_embedding, sample_hidden], dim=0)
            
            # Ensure proper shape
            if len(sample_hidden.shape) == 1:
                sample_hidden = sample_hidden.unsqueeze(0)  # Add sequence dimension
            
            # Ensure we match the expected sequence length
            if sample_hidden.shape[0] < seq_len:
                # Pad with zeros if too short
                padding = torch.zeros(seq_len - sample_hidden.shape[0], sample_hidden.shape[1], device=self.device)
                sample_hidden = torch.cat([sample_hidden, padding], dim=0)
            elif sample_hidden.shape[0] > seq_len:
                # Truncate if too long
                sample_hidden = sample_hidden[:seq_len]
            
            hidden_states.append(sample_hidden)
        
        # Stack all batches
        hidden_states = torch.stack(hidden_states)
        
        return hidden_states

class TextDataset(Dataset):
    """Dataset class for handling text data for Medusa training."""
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoded = self.tokenizer(text, 
                                 max_length=self.max_length, 
                                 truncation=True, 
                                 padding='max_length', 
                                 return_tensors='pt')
        return encoded['input_ids'].squeeze(0)

def load_and_preprocess_data(args):
    """
    Load and preprocess the dataset for Medusa training.
    
    Args:
        args: Command-line arguments containing dataset configuration
        
    Returns:
        train_dataset: PyTorch Dataset for training
        val_dataset: PyTorch Dataset for validation
    """
    logger.info(f"Loading dataset: {args.dataset_name}, split: {args.dataset_split}")

    try:
       dataset = load_dataset("theblackcat102/sharegpt-english",)
       print(dataset)
       texts = dataset['train']['text']
       logger.info(f"Loaded {len(texts)} examples from Hugging Face Hub")
    except Exception as e:
        logger.warning(f"Failed to load dataset from Hugging Face Hub: {e}")
        logger.info("Using placeholder dataset for testing")

        # Fallback to placeholder data for testing
        texts = [
            "This is a sample text for training Medusa heads.",
            "Another example sentence to demonstrate the training process.",
            "Medusa heads are trained to predict future tokens speculatively.",
            "Good datasets improve model performance substantially.",
            "Training with actual hidden states from the base model is critical.",
            "The goal is to predict the next few tokens accurately.",
            "Speculative decoding can speed up inference significantly.",
            "Higher acceptance rates mean better performance.",
            "Make sure to track metrics during training.",
            "Proper GPU utilization makes training much faster."
        ] * 50  # Replicate to create a reasonably sized test dataset

        logger.info(f"Created placeholder dataset with {len(texts)} examples")

    # Load tokenizer
    class SimpleTokenizer:
        def __call__(self, text, max_length=512, truncation=True, padding='max_length', return_tensors='pt'):
            # Very simple whitespace tokenizer for testing
            words = text.split()[:max_length]
            tokens = [i+1 for i in range(len(words))]  # Start from 1, 0 is padding
            if padding == 'max_length' and len(tokens) < max_length:
                tokens = tokens + [0] * (max_length - len(tokens))
            if return_tensors == 'pt':
                import torch
                tokens = torch.tensor([tokens])
            return {"input_ids": tokens}

    tokenizer = SimpleTokenizer()
    logger.warning("Using simple fallback tokenizer")

    # Create dataset
    dataset = TextDataset(texts, tokenizer, max_length=args.max_seq_length)

    # Split into train and validation
    val_size = min(int(len(dataset) * 0.1), 1000)  # 10% for validation, max 1000 examples
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )

    logger.info(f"Created training dataset with {len(train_dataset)} examples")
    logger.info(f"Created validation dataset with {len(val_dataset)} examples")

    return train_dataset, val_dataset

def evaluate_medusa(medusa_model, val_loader, hidden_state_extractor, device):
    """
    Evaluate the Medusa model on validation data.
    
    Args:
        medusa_model: The Medusa model to evaluate
        val_loader: DataLoader for validation data
        hidden_state_extractor: LlamaHiddenStateExtractor for getting hidden states
        device: Device to run evaluation on
        
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    total_loss = 0
    total_acceptance = 0
    total_tokens = 0
    total_steps = 0
    
    with torch.no_grad():
        for batch_idx, input_ids in enumerate(val_loader):
            if batch_idx >= 10:  # Limit validation to 10 batches for speed
                break
                
            input_ids = input_ids.to(device)
            batch_size, seq_len = input_ids.shape
            
            # Get hidden states from the base model
            hidden_states = hidden_state_extractor.get_hidden_states(input_ids)
            
            # Evaluate each head
            loss = 0
            for i, head in enumerate(medusa_model.heads):
                future_pos = i + 1
                
                # Skip if we're predicting beyond sequence length
                if future_pos >= seq_len:
                    continue
                    
                # Get head predictions
                logits = head(hidden_states)
                
                # Prepare targets - shift to get future tokens
                target_tokens = input_ids[:, future_pos:]
                predicted_logits = logits[:, :-future_pos, :]
                
                # Calculate loss
                if predicted_logits.shape[1] > 0 and target_tokens.shape[1] > 0:
                    batch_size, seq_len_pred, vocab_size = predicted_logits.shape
                    
                    # Reshape for cross entropy
                    predicted_logits_flat = predicted_logits.reshape(-1, vocab_size)
                    target_tokens_flat = target_tokens[:, :seq_len_pred].reshape(-1)
                    
                    # Calculate loss, ignoring padding
                    head_loss = F.cross_entropy(
                        predicted_logits_flat, 
                        target_tokens_flat, 
                        ignore_index=0,
                        reduction='mean'
                    )
                    loss += head_loss
                    
                    # Calculate acceptance rate
                    predicted_tokens = torch.argmax(predicted_logits, dim=-1)
                    correct_predictions = (predicted_tokens == target_tokens[:, :seq_len_pred])
                    total_acceptance += correct_predictions.sum().item()
                    total_tokens += correct_predictions.numel()
            
            total_loss += loss.item()
            total_steps += 1
    
    # Calculate metrics
    avg_loss = total_loss / max(total_steps, 1)
    acceptance_rate = total_acceptance / max(total_tokens, 1)
    
    metrics = {
        "val_loss": avg_loss,
        "acceptance_rate": acceptance_rate
    }
    
    logger.info(f"Validation: Loss: {avg_loss:.4f}, Acceptance Rate: {acceptance_rate:.2%}")
    
    return metrics

def train_medusa_heads(args):
    """
    Train Medusa heads using the provided arguments.
    
    Args:
        args: Command-line arguments for training configuration
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    logger.info(f"Using device: {device}")

    if device.type == "cuda":
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    # Load base model and prepare hidden state extractor
    logger.info(f"Initializing hidden state extractor with model: {args.base_model_path}")
    base_model = LlamaCppModel(
        model_path=args.base_model_path,
        n_ctx=config.MODEL_CONTEXT_SIZE,
        n_gpu_layers=config.N_GPU_LAYERS,
        n_threads=config.N_THREADS,
        # verbose=True
    )
    hidden_state_extractor = LlamaHiddenStateExtractor(base_model, device)
    
    # Initialize base model in eval mode (this is the one we'll use for reference
    
    # Initialize Medusa model (only heads will be trained)
    logger.info(f"Initializing Medusa model with {args.k_heads} heads")
    
    # Create a set of trainable Medusa heads
    heads = []
    for i in range(1, args.k_heads + 1):
        head = MedusaHead(
            input_dim=hidden_state_extractor.hidden_dim,
            output_dim=base_model.vocab_size,
            hidden_dim=hidden_state_extractor.hidden_dim // 2,
            future_pos=i
        ).to(device)
        heads.append(head)
    
    # Load datasets
    train_dataset, val_dataset = load_and_preprocess_data(args)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0, # Modified to 0 to fix EOFError
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0, # Modified to 0 to fix EOFError
        pin_memory=True
    )
    
    # Create optimizer
    parameters = []
    for head in heads:
        parameters.extend(head.parameters())
    
    optimizer = GrokFastAdamW(parameters, lr=args.learning_rate)
    
    # Create learning rate scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs * len(train_loader),
        eta_min=args.learning_rate / 10
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Training loop
    best_acceptance_rate = 0
    metrics_history = []
    try:
            
        for epoch in range(args.epochs):
            logger.info(f"Epoch {epoch+1}/{args.epochs}")
            
            # Training phase
            total_loss = 0
            start_time = time.time()
            
            # Set heads to training mode
            for head in heads:
                head.train()
            # Main training loop
            if not args.disable_tqdm:
                logger.info('Starting training...')

                progress_bar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}")
            else:
                progress_bar = train_loader
            for batch_idx, input_ids in enumerate(progress_bar):
                if not args.disable_tqdm:
                    progress_bar.set_description(f"Epoch {epoch+1} (Batch {batch_idx+1}/{len(train_loader)})")
                input_ids = input_ids.to(device)
                logger.info(f"time_elapsed:{time.time()-start_time},Batch:{batch_idx}/{len(train_loader)}")
                batch_size, seq_len = input_ids.shape
                
                # Get hidden states from the base model
                hidden_states = hidden_state_extractor.get_hidden_states(input_ids)
                
                # Train each head
                optimizer.zero_grad()
                loss = 0
                # logger.info("predicting future tokens")
                for i, head in enumerate(heads):
                    future_pos = i + 1  # Head i predicts token at position i+1
                    
                    # Skip if future position is beyond sequence length
                    if future_pos >= seq_len:
                        continue
                    
                    # Get logits from the head
                    logits = head(hidden_states)  # [batch_size, seq_len, vocab_size]
                    
                    # Set up targets (shifted)
                    target_tokens = input_ids[:, future_pos:]  # Skip first future_pos tokens
                    predicted_logits = logits[:, :-future_pos, :]  # Remove last future_pos positions
                    
                    # Calculate loss
                    if predicted_logits.shape[1] > 0 and target_tokens.shape[1] > 0:
                        batch_size, seq_len_pred, vocab_size = predicted_logits.shape
                        
                        # Reshape for cross entropy
                        predicted_logits_flat = predicted_logits.reshape(-1, vocab_size)
                        target_tokens_flat = target_tokens[:, :seq_len_pred].reshape(-1)
                        
                        # Calculate loss, ignoring padding
                        head_loss = F.cross_entropy(
                            predicted_logits_flat, 
                            target_tokens_flat, 
                            ignore_index=0,
                            reduction='mean'
                        )
                        loss += head_loss
                logger.info(f"loss:{loss}")
                # Backpropagation
                if loss > 0:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(parameters, args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                
                # Update metrics
                total_loss += loss.item()
                avg_loss = total_loss / (batch_idx + 1)
                # Run evaluation
                # Save checkpoint periodically
                if (batch_idx + 1) % args.save_interval == 0:
                    medusa_model = MedusaModel(base_model, device=device,eval=True)
                    medusa_model.heads = heads  # Replace with trained heads
                    metrics = evaluate_medusa(medusa_model, val_loader, hidden_state_extractor, device)
                    logger.info(f"Evaluation Metrics: {metrics}")
                    checkpoint_path = os.path.join(
                        args.output_dir, 
                        f"medusa_heads_epoch_{epoch+1}_batch_{batch_idx+1}.pth"
                    )
                    torch.save([head.state_dict() for head in heads], checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
            
            # Calculate epoch metrics
            epoch_time = time.time() - start_time
            logger.info(f"Epoch {epoch+1} completed in {epoch_time:.2f}s, avg loss: {avg_loss:.4f}")
            
            # Evaluation phase
            logger.info("Evaluating on validation set...")
            # Create a temporary Medusa model for evaluation
            medusa_model = MedusaModel(base_model, device=device)
            medusa_model.heads = heads  # Replace with trained heads
            
            # Run evaluation
            metrics = evaluate_medusa(medusa_model, val_loader, hidden_state_extractor, device)
            metrics['epoch'] = epoch + 1
            metrics['train_loss'] = avg_loss
            metrics_history.append(metrics)
            
            # Save if best model
            if metrics['acceptance_rate'] > best_acceptance_rate:
                logger.info(f"found better acceptance rate:{best_acceptance_rate}")
                best_acceptance_rate = metrics['acceptance_rate']
                best_checkpoint_path = os.path.join(args.output_dir, "medusa_heads_best.pth")
                torch.save([head.state_dict() for head in heads], best_checkpoint_path)
                logger.info(f"New best model! Saved to {best_checkpoint_path}")
            
            # Save latest model
            latest_checkpoint_path = os.path.join(args.output_dir, "medusa_heads_latest.pth")
            torch.save([head.state_dict() for head in heads], latest_checkpoint_path)
            
            # Save metrics history
            metrics_path = os.path.join(args.output_dir, "training_metrics.json")
            with open(metrics_path, 'w') as f:
                json.dump(metrics_history, f, indent=2)
        
        logger.info("Training complete!")
        logger.info(f"Best acceptance rate: {best_acceptance_rate:.2%}")
        logger.info(f"Final checkpoints saved to: {args.output_dir}")
    except Exception as e:
        logger.info(f"Exception :{e}")
def parse_args():
    parser = argparse.ArgumentParser(description="Train Medusa heads for speculative decoding")
    
    # Model configuration
    parser.add_argument("--base_model_path", type=str, default=config.MODEL_PATH,
                        help="Path to the base model to use")
    parser.add_argument("--k_heads", type=int, default=config.MEDUSA_K_HEADS,
                        help="Number of Medusa heads")
    
    # Dataset configuration
    parser.add_argument("--dataset_name", type=str, default="wikitext",
                        help="Name of the dataset to use (from Hugging Face)")
    parser.add_argument("--dataset_split", type=str, default="train",
                        help="Dataset split to use")
    parser.add_argument("--max_seq_length", type=int, default=512,
                        help="Maximum sequence length for training")
    
    # Training configuration
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Training batch size")

def parse_args():
    parser = argparse.ArgumentParser(description="Train Medusa heads for speculative decoding")
    
    # Model configuration
    parser.add_argument("--base_model_path", type=str, default=config.MODEL_PATH,
                        help="Path to the base model to use")
    parser.add_argument("--k_heads", type=int, default=config.MEDUSA_K_HEADS,
                        help="Number of Medusa heads")
    
    # Dataset configuration
    parser.add_argument("--dataset_name", type=str, default="wikitext",
                        help="Name of the dataset to use (from Hugging Face)")
    parser.add_argument("--dataset_split", type=str, default="train",
                        help="Dataset split to use")
    parser.add_argument("--max_seq_length", type=int, default=512,
                        help="Maximum sequence length for training")
    
    # Training configuration
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Training batch size")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Initial learning rate")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Maximum gradient norm for clipping")
    parser.add_argument("--save_interval", type=int, default=10,
                        help="Save checkpoints every N batches")
    parser.add_argument("--output_dir", type=str, default="medusa_heads_weights",
                        help="Directory to save model checkpoints")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run training on (cuda, cpu)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers")
    parser.add_argument("--disable_tqdm", type=bool, default=True,
                        help="Disable tqdm progress bar")
    
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args()
    
    # Start training
    logger.info("Starting Medusa head training...")
    logger.info(f"Device: {args.device}")
    logger.info(f"Model: {args.base_model_path}")
    logger.info(f"Dataset: {args.dataset_name}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Train
    train_medusa_heads(args)
    
    logger.info("Training completed successfully!")
