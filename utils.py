import torch.nn as nn
from typing import Dict, List, Any

from transformers.modeling_utils import PreTrainedModel


def get_layer_statistics(model: PreTrainedModel) -> Dict[str, Any]:
    """Get statistics about model layers and parameters."""
    stats = {
        "total_params": 0,
        "layer_types": {},
        "weight_sizes": {},
    }

    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf module
            module_type = module.__class__.__name__
            if module_type not in stats["layer_types"]:
                stats["layer_types"][module_type] = 0
            stats["layer_types"][module_type] += 1

            # Count parameters
            for param_name, param in module.named_parameters():
                if param.requires_grad:
                    stats["total_params"] += param.numel()

                    # Track weight tensor sizes
                    if "weight" in param_name:
                        shape_key = str(list(param.shape))
                        if shape_key not in stats["weight_sizes"]:
                            stats["weight_sizes"][shape_key] = 0
                        stats["weight_sizes"][shape_key] += 1

    return stats


def identify_quantization_candidates(model: PreTrainedModel) -> List[str]:
    """Identify layers that are good candidates for quantization."""
    candidates = []

    critical_patterns = ["layernorm", "layer_norm", "ln_", "embedding", "lm_head"]

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if module.weight.numel() > 100000:
                if not any(pattern in name.lower() for pattern in critical_patterns):
                    candidates.append(name)

    return candidates


def generate_quantization_recommendations(model: PreTrainedModel) -> Dict[str, Any]:
    stats = get_layer_statistics(model)

    critical_layers = []
    quantization_candidates = []
    largest_layers = []

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        critical_patterns = [
            "layernorm",
            "layer_norm",
            "ln_",
            "embedding",
            "embeddings",
            "emb",
            "lm_head",
            "head",
            "output",
        ]
        is_critical = any(pattern in name.lower() for pattern in critical_patterns)

        param_count = module.weight.numel()
        layer_info = {
            "name": name,
            "params": param_count,
            "shape": str(list(module.weight.shape)),
        }

        if is_critical:
            critical_layers.append(layer_info)
        else:
            if param_count > 100000:  # Arbitrary threshold
                quantization_candidates.append(layer_info)

        largest_layers.append((name, param_count, is_critical))

    largest_layers.sort(key=lambda x: x[1], reverse=True)

    total_params = stats["total_params"]
    critical_params = sum(layer["params"] for layer in critical_layers)
    quantizable_params = sum(layer["params"] for layer in quantization_candidates)

    recommendations = {
        "total_parameters": total_params,
        "critical_layers": critical_layers,
        "quantization_candidates": quantization_candidates,
        "largest_layers": [
            (name, params, is_critical)
            for name, params, is_critical in largest_layers[:20]
        ],
        "potential_savings": {
            "quantizable_percent": (quantizable_params / total_params) * 100,
            "critical_percent": (critical_params / total_params) * 100,
        },
        "suggested_exclude_patterns": critical_patterns,
        "recommended_approach": (
            "selective_quantization"
            if quantizable_params > 0.5 * total_params
            else "full_quantization"
        ),
    }

    return recommendations
