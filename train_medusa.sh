#!/bin/bash
# Script to train Medusa heads with GPU acceleration

# Check if CUDA is available
if command -v nvidia-smi &> /dev/null; then
    echo "CUDA GPU detected."
    DEVICE="cuda"
else
    echo "No CUDA GPU detected, falling back to CPU."
    echo "Warning: Training on CPU will be much slower!"
    DEVICE="cpu"
fi

# Create output directory
mkdir -p medusa_heads_weights

# Run the training script
python model/train_medusa.py \
    --base_model_path models/vicuna-7b-v1.3.Q4_K_M.gguf \
    --dataset_name wikitext \
    --dataset_split wikitext-103-v1 \
    --max_seq_length 512 \
    --batch_size 8 \
    --epochs 3 \
    --learning_rate 1e-4 \
    --device "$DEVICE" \
    --num_workers 4 \
    --save_interval 500 \
    --output_dir medusa_heads_weights

echo "Training completed. Weights saved to medusa_heads_weights/"
echo "You can now use these weights for inference by setting MEDUSA_WEIGHTS_PATH in config.py"
