import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

from datasets import load_dataset
import torch
import torch.nn as nn
from torch.quantization import quantize_dynamic, prepare_qat, convert
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from transformers.data.data_collator import DataCollatorWithPadding
from transformers.utils import logging as transformers_logging

# Try to import BitsAndBytesConfig, handle gracefully if not available
try:
    from transformers.utils.quantization_config import BitsAndBytesConfig

    HAS_BITSANDBYTES = True
except ImportError:
    BitsAndBytesConfig = None
    HAS_BITSANDBYTES = False


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    (
        transformers_logging.set_verbosity_info()
        if verbose
        else transformers_logging.set_verbosity_warning()
    )

    # Warn about BitsAndBytes availability
    if not HAS_BITSANDBYTES:
        logging.warning(
            "BitsAndBytes not available on this platform (likely macOS). "
            "4-bit and 8-bit quantization will be disabled. "
            "Available modes: dynamic, qat, weight_only"
        )


def save_quantization_report(
    model_name: str,
    quant_mode: str,
    original_size: float,
    quantized_size: float,
    output_dir: str,
    quantization_time: float,
    args: argparse.Namespace,
    model: Optional[PreTrainedModel] = None,
) -> None:
    report = {
        "model_name": model_name,
        "quantization_mode": quant_mode,
        "original_size_mb": round(original_size, 2),
        "quantized_size_mb": round(quantized_size, 2),
        "size_reduction_mb": round(original_size - quantized_size, 2),
        "size_reduction_percent": round(
            (original_size - quantized_size) / original_size * 100, 2
        ),
        "compression_ratio": round(original_size / quantized_size, 2),
        "quantization_time_seconds": round(quantization_time, 2),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "parameters": {
            "double_quant": (
                args.double_quant if hasattr(args, "double_quant") else False
            ),
            "quant_type_4bit": (
                args.quant_type_4bit if hasattr(args, "quant_type_4bit") else None
            ),
            "compute_dtype": (
                args.compute_dtype if hasattr(args, "compute_dtype") else None
            ),
            "selective": args.selective if hasattr(args, "selective") else False,
            "exclude_patterns": (
                args.exclude_patterns if hasattr(args, "exclude_patterns") else None
            ),
        },
    }

    # Include quantization info if available
    if model and hasattr(model, "_quantized_info"):
        report["_quantized_info"] = model._quantized_info

    report_path = Path(output_dir) / "quantization_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logging.info(f"Quantization report saved to {report_path}")


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
        if not HAS_BITSANDBYTES:
            raise RuntimeError(
                f"{quant_mode} quantization requires BitsAndBytes, which is not available on this platform. "
                "On macOS, BitsAndBytes is not supported. Use 'dynamic', 'qat', or 'weight_only' quantization instead."
            )

        if quant_mode == "8bit":
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
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
    quantized_model = quantize_dynamic(model, {nn.Linear, nn.LSTM}, dtype=torch.qint8)
    return quantized_model


def apply_weight_only_quantization(model: PreTrainedModel) -> PreTrainedModel:
    logging.info("Applying weight-only quantization...")
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module.weight.data = torch.quantize_per_tensor(
                module.weight.data, scale=1.0, zero_point=0, dtype=torch.qint8
            ).dequantize()
    return model


def apply_selective_quantization(
    model: PreTrainedModel, exclude_patterns: Optional[list] = None
) -> PreTrainedModel:
    if exclude_patterns is None:
        exclude_patterns = [
            "layernorm",
            "layer_norm",
            "ln_",
            "norm",
            "embedding",
            "embeddings",
            "emb",
            "lm_head",
            "head",
            "output",
        ]

    logging.info(
        f"Applying selective quantization (excluding patterns: {exclude_patterns})..."
    )

    quantized_count = 0
    preserved_count = 0
    quantized_layer_names = []
    preserved_layer_names = []

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        should_preserve = any(pattern in name.lower() for pattern in exclude_patterns)

        if should_preserve:
            preserved_count += 1
            preserved_layer_names.append(name)
            if logging.root.level <= logging.DEBUG:
                logging.debug(f"Preserving layer: {name}")
        else:
            quantized_count += 1
            quantized_layer_names.append(name)
            module.weight.data = torch.quantize_per_tensor(
                module.weight.data, scale=1.0, zero_point=0, dtype=torch.qint8
            ).dequantize()
            if logging.root.level <= logging.DEBUG:
                logging.debug(f"Quantizing layer: {name}")

    logging.info(
        f"Quantized {quantized_count} modules, preserved {preserved_count} modules"
    )

    model._quantized_info = {
        "quantized_layers": quantized_count,
        "preserved_layers": preserved_count,
        "quantized_layer_names": quantized_layer_names,
        "preserved_layer_names": preserved_layer_names,
    }

    return model


def prepare_qat_model(model: PreTrainedModel) -> PreTrainedModel:
    logging.info("Preparing model for QAT...")
    model.train()
    model.qconfig = torch.quantization.get_default_qat_qconfig("fbgemm")
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
            examples["text"], padding="max_length", truncation=True, max_length=128
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
    parser = argparse.ArgumentParser(
        description="Quantize and upload Hugging Face models"
    )

    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name or path of the pretrained model",
    )
    parser.add_argument(
        "--quant_mode",
        type=str,
        required=True,
        choices=["8bit", "4bit", "dynamic", "qat", "weight_only"],
        help="Type of quantization (Note: 8bit/4bit require BitsAndBytes, not available on macOS)",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push the quantized model to Hugging Face Hub",
    )
    parser.add_argument("--repo_name", type=str, help="Name of the Hub repo")
    parser.add_argument(
        "--use_auth_token", action="store_true", help="Use Hugging Face auth token"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./quantized_model",
        help="Path to save the quantized model",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    parser.add_argument(
        "--double_quant",
        action="store_true",
        help="Enable double quantization for 4-bit",
    )
    parser.add_argument(
        "--quant_type_4bit",
        type=str,
        default="nf4",
        choices=["fp4", "nf4"],
        help="Type of 4-bit quantization",
    )
    parser.add_argument(
        "--compute_dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Compute dtype",
    )

    parser.add_argument("--dataset_name", type=str, help="Dataset name for QAT")
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Number of training epochs for QAT",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=8, help="Training batch size for QAT"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=2e-5, help="Learning rate for QAT"
    )
    parser.add_argument(
        "--analyze_architecture", action="store_true", help="Analyze model architecture"
    )
    parser.add_argument(
        "--save_report", action="store_true", help="Save detailed quantization report"
    )
    parser.add_argument(
        "--selective",
        action="store_true",
        help="Use selective quantization to preserve critical layers",
    )
    parser.add_argument(
        "--exclude_patterns",
        type=str,
        nargs="+",
        help="Patterns to exclude from quantization when using selective mode",
    )
    parser.add_argument(
        "--generate_recommendations",
        action="store_true",
        help="Generate recommendations for selective quantization based on model analysis",
    )

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

        # Analyze model architecture
        if args.analyze_architecture or args.generate_recommendations:
            from utils import (
                get_layer_statistics,
                generate_quantization_recommendations,
            )

            stats = get_layer_statistics(model)
            logging.info(f"Model has {stats['total_params']:,} parameters")
            logging.info(
                f"Layer types: {', '.join(f'{k}: {v}' for k, v in stats['layer_types'].items())}"
            )

            if args.generate_recommendations:
                recommendations = generate_quantization_recommendations(model)

                logging.info("\n----- Quantization Recommendations -----")
                logging.info(
                    f"Recommended approach: {recommendations['recommended_approach']}"
                )
                logging.info(
                    f"Quantizable parameters: {recommendations['potential_savings']['quantizable_percent']:.2f}%"
                )
                logging.info(
                    f"Critical parameters: {recommendations['potential_savings']['critical_percent']:.2f}%"
                )
                logging.info(
                    f"Suggested exclude patterns: {recommendations['suggested_exclude_patterns']}"
                )

                # Save recommendations to file
                recommendations_path = (
                    Path(args.output_dir) / "quantization_recommendations.json"
                )
                with open(recommendations_path, "w") as f:
                    json.dump(recommendations, f, indent=2)
                logging.info(
                    f"Detailed recommendations saved to: {recommendations_path}"
                )

        # Apply quantization
        quantization_start = time.time()
        if args.quant_mode == "dynamic":
            model = apply_dynamic_quantization(model)
        elif args.quant_mode == "weight_only":
            if args.selective:
                model = apply_selective_quantization(model, args.exclude_patterns)
            else:
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
        quantization_time = time.time() - quantization_start

        # Evaluate quantized model
        quantized_size = get_model_size(model)
        size_reduction = (original_size - quantized_size) / original_size * 100

        logging.info(f"Quantized model size: {quantized_size:.2f} MB")
        logging.info(f"Size reduction: {size_reduction:.2f}%")
        logging.info(f"Quantization completed in {quantization_time:.2f} seconds")

        # Save model and tokenizer
        save_model_and_tokenizer(
            model,
            tokenizer,
            args.output_dir,
            args.push_to_hub,
            args.repo_name,
            args.use_auth_token,
        )

        # Save detailed report if requested
        if args.save_report:
            save_quantization_report(
                args.model_name,
                args.quant_mode,
                original_size,
                quantized_size,
                args.output_dir,
                quantization_time,
                args,
                model,
            )
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
