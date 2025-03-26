# Training Medusa Models for Better LLM Inference

This guide explains how to properly train Medusa models to improve inference speed and coherence with the base LLM.

## What is Medusa?

Medusa is a speculative decoding method that uses multiple prediction heads to predict several tokens at once, which are then verified against the base model. This can significantly speed up inference while maintaining the same output quality.

## Prerequisites

- PyTorch 2.0+
- A pre-trained LLM model (e.g., Llama-2, Vicuna)
- GPU with CUDA support (for efficient training)
- High-quality text dataset for training

## Training Process

### 1. Prepare Your Environment

Ensure you have the required packages installed:

```bash
pip install torch torchvision torchaudio transformers datasets
```

### 2. Prepare Your Dataset

A good training dataset should match the distribution of text you expect the model to generate. Options include:

- **General text corpus**: Wikitext, C4, or OpenWebText
- **Instruction-tuned data**: For instruction-following models
- **Domain-specific data**: For specialized applications

Example code to load a dataset:

```python
from datasets import load_dataset

# Load a dataset from Hugging Face
dataset = load_dataset("wikitext", "wikitext-103-v1", split="train")
```

### 3. Configure Training Parameters

Key parameters to configure:

- **k_heads**: Number of Medusa heads (default: 5)
- **hidden_dim**: Size of hidden layers in heads (typically half of base model's dimension)
- **batch_size**: Increase for faster training, decrease if you encounter OOM errors
- **learning_rate**: Typically 1e-4 to 5e-4 works well
- **epochs**: 2-3 epochs usually provides good results

### 4. Run Training

The `train_medusa.py` script has been updated to extract actual hidden states from the base model instead of using random values. This is critical for proper training.

To run training:

```bash
python model/train_medusa.py \
    --base_model_path models/vicuna-7b-v1.3.Q4_K_M.gguf \
    --dataset_name wikitext \
    --dataset_split train[:10%] \
    --output_dir medusa_heads_weights \
    --batch_size 8 \
    --learning_rate 1e-4 \
    --epochs 3
```

During training, the script will:
1. Extract hidden states from the base model for each input
2. Train the Medusa heads to predict future tokens
3. Save checkpoints periodically
4. Evaluate on validation data

### 5. Monitor Training

Key metrics to monitor:
- **Loss**: Should decrease steadily
- **Validation Acceptance Rate**: The percentage of Medusa predictions accepted by the base model
- **Tokens Per Second**: During evaluation, this shows inference speedup

## Using Trained Weights

After training, the weights will be saved to the specified output directory. To use them:

1. Make sure the config.py file has `MEDUSA_WEIGHTS_PATH` pointing to your weights:
   ```python
   MEDUSA_WEIGHTS_PATH = os.environ.get("MEDUSA_WEIGHTS_PATH", "medusa_heads_weights/medusa_heads_latest.pth")
   ```

2. Run the server with GPU acceleration:
   ```bash
   python main.py
   ```

3. The server will automatically load the Medusa weights if they exist at the specified path.

## Best Practices

1. **Use quality training data** that matches your expected inference distribution
2. **Train with actual hidden states** from the base model, not random values
3. **Increase `max_tokens_per_step`** as acceptance rate improves
4. **Use GPU acceleration** for both training and inference 
5. **Monitor memory usage** and adjust batch size accordingly
6. **Save checkpoints frequently** during training

## Troubleshooting

- **Low acceptance rate**: Train for more epochs or use a more representative dataset
- **CUDA out of memory**: Reduce batch size or use gradient accumulation
- **Slow training**: Increase the number of GPU layers offloaded
- **Loading errors**: Ensure the number of heads matches between training and inference

## Enhancing GPU Utilization

To maximize GPU utilization:
1. Offload all layers to GPU with `N_GPU_LAYERS=-1`
2. Use multiple threads with `N_THREADS=8` (or appropriate for your CPU)
3. Enable memory locking with `use_mlock=True`
4. Use memory mapping with `use_mmap=True` for faster loading
