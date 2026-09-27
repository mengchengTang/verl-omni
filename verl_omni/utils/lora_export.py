# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import shutil
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors import safe_open
from safetensors.torch import save_file


def adapter_export_key(name: str) -> str:
    """Normalize FSDP wrapper names to a PEFT component checkpoint key."""
    name = name.replace("_fsdp_wrapped_module.", "")
    return name if name.startswith("base_model.model.") else f"base_model.model.{name}"


def inspect_lora_adapter(
    module: torch.nn.Module, adapter_name: str, component: str = "transformer"
) -> tuple[dict, set[str]]:
    """Validate a registered adapter without changing activation or trainability."""
    from peft.tuners.lora import LoraLayer

    if component != "transformer":
        raise ValueError("Adapter export currently supports only the transformer component")
    module = getattr(module, "_fsdp_wrapped_module", module)
    configs = getattr(module, "peft_config", {})
    if adapter_name == "reference" or adapter_name not in configs:
        raise ValueError(f"No registered LoRA adapter {adapter_name!r}; available: {list(configs)}")
    config = configs[adapter_name].to_dict()
    if config.get("peft_type") != "LORA":
        raise ValueError("Adapter export supports PEFT LoRA only")
    unsupported = ("modules_to_save", "target_parameters", "use_dora", "lora_bias")
    if config.get("bias", "none") != "none" or any(config.get(key) for key in unsupported):
        raise ValueError(f"Adapter export requires bias='none' and disables {unsupported}")
    if config.get("init_lora_weights", True) not in (True, False, "gaussian"):
        raise ValueError("Adapter export does not support initializers that may modify the base weights")
    expected_keys = set()
    for name, layer in module.named_modules():
        if not isinstance(layer, LoraLayer):
            continue
        if layer.merged_adapters:
            raise ValueError("Export requires unmerged LoRA layers")
        if adapter_name in layer.lora_embedding_A:
            raise ValueError("Embedding LoRA export is not supported yet")
        if adapter_name not in layer.lora_A:
            continue
        for matrix in ("lora_A", "lora_B"):
            key = adapter_export_key(f"{name}.{matrix}.weight")
            if key in expected_keys:
                raise ValueError(f"Duplicate adapter key: {key}")
            expected_keys.add(key)
    if not expected_keys:
        raise ValueError(f"Adapter {adapter_name!r} has no supported LoRA tensors")
    return config, expected_keys


def prepare_adapter_tensors(params: dict, expected_keys: set[str]) -> dict[str, torch.Tensor]:
    """Check complete component coverage and detach an independent CPU snapshot."""
    tensors = {}
    for name, tensor in params.items():
        key = adapter_export_key(name)
        if key in tensors:
            raise ValueError(f"Duplicate adapter key: {key}")
        if not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
            raise ValueError(f"Invalid or empty adapter tensor: {key}")
        tensors[key] = tensor.detach().cpu().contiguous().clone()
    if not tensors or tensors.keys() != expected_keys:
        missing, extra = expected_keys - tensors.keys(), tensors.keys() - expected_keys
        raise ValueError(f"Incomplete adapter export: missing={sorted(missing)}, unexpected={sorted(extra)}")
    for name, tensor in tensors.items():
        if name.endswith(".lora_A.weight"):
            other = tensors[name.replace(".lora_A.weight", ".lora_B.weight")]
            if tensor.ndim < 2 or other.ndim < 2 or tensor.shape[0] != other.shape[1]:
                raise ValueError(f"Invalid LoRA A/B shapes for {name}")
    return tensors


def _json_default(value):
    if isinstance(value, set):
        return sorted(value)
    if hasattr(value, "value"):
        return value.value
    raise TypeError(f"Cannot serialize adapter configuration value {type(value).__name__}")


def write_lora_adapter(output_dir: str | Path, tensors: dict, config: dict, metadata: dict) -> None:
    """Publish a validated local adapter directory without replacing earlier exports."""
    if not tensors:
        raise ValueError("Cannot publish an empty adapter")
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite adapter export: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    lock = output_dir.with_name(f".{output_dir.name}.lock")
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(fd)
    staging = None
    try:
        if output_dir.exists():
            raise FileExistsError(f"Refusing to overwrite adapter export: {output_dir}")
        staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
        (staging / "adapter_config.json").write_text(
            json.dumps(config, default=_json_default, indent=2) + "\n", encoding="utf-8"
        )
        (staging / "training_metadata.json").write_text(
            json.dumps(metadata, default=_json_default, indent=2) + "\n", encoding="utf-8"
        )
        save_file(tensors, staging / "adapter_model.safetensors", metadata={"format": "pt"})
        with safe_open(staging / "adapter_model.safetensors", framework="pt", device="cpu") as saved:
            if set(saved.keys()) != set(tensors):
                raise ValueError("Saved adapter keys do not match the collected tensors")
            for name, tensor in tensors.items():
                restored = saved.get_tensor(name)
                if restored.shape != tensor.shape or restored.dtype != tensor.dtype:
                    raise ValueError(f"Saved adapter tensor metadata mismatch: {name}")
        os.rename(staging, output_dir)
        staging = None
    finally:
        if staging is not None:
            shutil.rmtree(staging)
        lock.unlink()


def coordinated_export_phase(operation):
    """Report recoverable phase failures on all ranks; collectives retain their own timeout."""
    result, error = None, None
    try:
        result = operation()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    errors = [error]
    if dist.is_initialized():
        errors = [None] * dist.get_world_size()
        dist.all_gather_object(errors, error)
    failures = [f"rank {rank}: {message}" for rank, message in enumerate(errors) if message]
    if failures:
        raise RuntimeError("LoRA adapter export failed: " + "; ".join(failures))
    return result
