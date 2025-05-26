# NeuQuant - Quantization Library

A comprehensive command-line tool for quantizing models with multiple quantization strategies, supporting both inference and training-based approaches.

## Features

- Multiple Quantization Modes:
  - 8-bit and 4-bit quantization (bitsandbytes)
  - Dynamic Quantization (PyTorch)
  - Quantization-Aware Training (QAT)
  - Weight-Only Quantization
- Automatic model size reduction
- Hugging Face Hub integration
- Mixed precision support
- Detailed progress logging
- Tokenizer handling

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage (8-bit/4-bit Quantization)

```bash
python main.py --model_name "bert-base-uncased" --quant_mode "8bit"
```

### Dynamic Quantization

```bash
python main.py \
    --model_name "bert-base-uncased" \
    --quant_mode "dynamic"
```

### Quantization-Aware Training (QAT)

```bash
python main.py \
    --model_name "bert-base-uncased" \
    --quant_mode "qat" \
    --dataset_name "sst2" \
    --num_train_epochs 3 \
    --train_batch_size 8 \
    --learning_rate 2e-5
```

### Weight-Only Quantization

```bash
python main.py \
    --model_name "bert-base-uncased" \
    --quant_mode "weight_only"
```

### Upload to Hugging Face Hub

```bash
python main.py \
    --model_name "bert-base-uncased" \
    --quant_mode "8bit" \
    --push_to_hub \
    --repo_name "my-quantized-model" \
    --use_auth_token
```

## Command Line Arguments

### Required Arguments
- `--model_name`: Name or path of the pretrained model on Hugging Face
- `--quant_mode`: Type of quantization ("8bit", "4bit", "dynamic", "qat", "weight_only")

### Optional Arguments
- `--push_to_hub`: Upload the quantized model to Hugging Face Hub
- `--repo_name`: Name of the Hub repo (default: {model_name}-quantized)
- `--use_auth_token`: Use Hugging Face auth token for uploading
- `--output_dir`: Path to save the quantized model locally (default: ./quantized_model)
- `--verbose`: Enable verbose logging

### 4-bit Specific Arguments
- `--double_quant`: Enable double quantization (default: False)
- `--quant_type_4bit`: Type of 4-bit quantization ("fp4" or "nf4", default: "nf4")
- `--compute_dtype`: Compute dtype ("float16", "bfloat16", "float32", default: "float16")

### QAT Specific Arguments
- `--dataset_name`: Hugging Face dataset name for fine-tuning (required for QAT)
- `--num_train_epochs`: Number of training epochs (default: 3)
- `--train_batch_size`: Training batch size (default: 8)
- `--learning_rate`: Learning rate for fine-tuning (default: 2e-5)

### Analysis and Visualization Arguments
- `--analyze_architecture`: Analyze model architecture and provide insights
- `--save_report`: Save detailed quantization report in JSON format
- `--visualize`: Generate visualizations of model structure and quantization results
- `--generate_recommendations`: Generate recommendations for selective quantization

### Selective Quantization Arguments
- `--selective`: Enable selective quantization (preserving critical layers)
- `--exclude_patterns`: Patterns to exclude from quantization (e.g., "layernorm" "embedding")

## Examples

### Dynamic Quantization with Local Save
```bash
python main.py \
    --model_name "gpt2" \
    --quant_mode "dynamic" \
    --output_dir "./my_quantized_gpt2"
```

### QAT with Hub Upload
```bash
python main.py \
    --model_name "gpt2" \
    --quant_mode "qat" \
    --dataset_name "sst2" \
    --num_train_epochs 5 \
    --push_to_hub \
    --repo_name "gpt2-qat-quantized"
```

## Notes

- For uploading to the Hub, you need to be logged in to Hugging Face
- The tool automatically handles tokenizer saving and uploading
- Model size reduction information is displayed during quantization
- QAT mode requires a dataset for fine-tuning
- Dynamic quantization is applied to Linear and LSTM layers
- Weight-only quantization preserves activation precision
