import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Union, Dict, Any

import torch
import torch.nn as nn
from torch.quantization import quantize_dynamic, prepare_qat, convert
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from transformers.utils import logging as transformers_logging
from datasets import load_dataset
from tqdm import tqdm

def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    transformers_logging.set_verbosity_info() if verbose else transformers_logging.set_verbosity_warning()

def get_model_size(model: PreTrainedModel) -> float:
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb

def load_model_and_tokenizer(
    model_name: str,
    quant_mode: str,
    double_quant: bool = False,
    quant_type_4bit: str = "nf4",
    compute_dtype: str = "float16",
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    logging.info(f"Loading model {model_name}...")
    
    compute_dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    
    if quant_mode in ["8bit", "4bit"]:
        if quant_mode == "8bit":
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                load_in_8bit=True,
                device_map="auto",
            )
        else:  # 4bit
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=double_quant,
                bnb_4bit_quant_type=quant_type_4bit,
                bnb_4bit_compute_dtype=compute_dtype_map[compute_dtype],
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def apply_dynamic_quantization(model: PreTrainedModel) -> PreTrainedModel:
    logging.info("Applying dynamic quantization...")
    quantized_model = quantize_dynamic(
        model,
        {nn.Linear, nn.LSTM},
        dtype=torch.qint8
    )
    return quantized_model

def apply_weight_only_quantization(model: PreTrainedModel) -> PreTrainedModel:
    logging.info("Applying weight-only quantization...")
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module.weight.data = torch.quantize_per_tensor(
                module.weight.data,
                scale=1.0,
                zero_point=0,
                dtype=torch.qint8
            ).dequantize()
    return model

def prepare_qat_model(model: PreTrainedModel) -> PreTrainedModel:
    logging.info("Preparing model for QAT...")
    model.train()
    model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    model = prepare_qat(model)
    return model

def train_qat_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset_name: str,
    num_train_epochs: int,
    train_batch_size: int,
    learning_rate: float,
) -> PreTrainedModel:
    logging.info(f"Loading dataset {dataset_name}...")
    dataset = load_dataset(dataset_name)
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=128
        )
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    training_args = TrainingArguments(
        output_dir="./qat_training",
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=train_batch_size,
        learning_rate=learning_rate,
        save_strategy="epoch",
        logging_dir="./logs",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        data_collator=DataCollatorWithPadding(tokenizer),
    )
    
    logging.info("Starting QAT training...")
    trainer.train()
    
    logging.info("Converting QAT model to quantized model...")
    model.eval()
    model = convert(model)
    return model

def save_model_and_tokenizer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    output_dir: str,
    push_to_hub: bool = False,
    repo_name: Optional[str] = None,
    use_auth_token: bool = False,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Saving model and tokenizer to {output_path}...")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    if push_to_hub:
        if not repo_name:
            model_name = model.config._name_or_path
            repo_name = f"{model_name}-quantized"
        
        logging.info(f"Pushing to Hub: {repo_name}")
        model.push_to_hub(repo_name, use_auth_token=use_auth_token)
        tokenizer.push_to_hub(repo_name, use_auth_token=use_auth_token)

def main():
    parser = argparse.ArgumentParser(description="Quantize and upload Hugging Face models")
    
    parser.add_argument("--model_name", type=str, required=True, help="Name or path of the pretrained model")
    parser.add_argument("--quant_mode", type=str, required=True, 
                       choices=["8bit", "4bit", "dynamic", "qat", "weight_only"],
                       help="Type of quantization")
    parser.add_argument("--push_to_hub", action="store_true", help="Push the quantized model to Hugging Face Hub")
    parser.add_argument("--repo_name", type=str, help="Name of the Hub repo")
    parser.add_argument("--use_auth_token", action="store_true", help="Use Hugging Face auth token")
    parser.add_argument("--output_dir", type=str, default="./quantized_model", help="Path to save the quantized model")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    parser.add_argument("--double_quant", action="store_true", help="Enable double quantization for 4-bit")
    parser.add_argument("--quant_type_4bit", type=str, default="nf4", choices=["fp4", "nf4"], help="Type of 4-bit quantization")
    parser.add_argument("--compute_dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"], help="Compute dtype")
    
    parser.add_argument("--dataset_name", type=str, help="Dataset name for QAT")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs for QAT")
    parser.add_argument("--train_batch_size", type=int, default=8, help="Training batch size for QAT")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate for QAT")
    
    args = parser.parse_args()
    setup_logging(args.verbose)
    
    try:
        if args.quant_mode == "qat" and not args.dataset_name:
            raise ValueError("Dataset name is required for QAT mode")
        
        model, tokenizer = load_model_and_tokenizer(
            args.model_name,
            args.quant_mode,
            args.double_quant,
            args.quant_type_4bit,
            args.compute_dtype,
        )
        
        original_size = get_model_size(model)
        logging.info(f"Original model size: {original_size:.2f} MB")
        
        if args.quant_mode == "dynamic":
            model = apply_dynamic_quantization(model)
        elif args.quant_mode == "weight_only":
            model = apply_weight_only_quantization(model)
        elif args.quant_mode == "qat":
            model = prepare_qat_model(model)
            model = train_qat_model(
                model,
                tokenizer,
                args.dataset_name,
                args.num_train_epochs,
                args.train_batch_size,
                args.learning_rate,
            )
        
        save_model_and_tokenizer(
            model,
            tokenizer,
            args.output_dir,
            args.push_to_hub,
            args.repo_name,
            args.use_auth_token,
        )
        
        quantized_size = get_model_size(model)
        logging.info(f"Quantized model size: {quantized_size:.2f} MB")
        logging.info(f"Size reduction: {((original_size - quantized_size) / original_size * 100):.2f}%")
        
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 