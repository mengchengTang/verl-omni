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

from pathlib import Path


def validate_lora_export_config(config):
    """Reject unsupported export requests before starting training workers."""
    checkpoint = config.actor_rollout_ref.actor.checkpoint
    frequency = checkpoint.get("save_lora_adapter_freq", 0)
    if isinstance(frequency, bool) or not isinstance(frequency, int) or frequency < 0:
        raise ValueError("save_lora_adapter_freq must be a non-negative integer")
    if not checkpoint.get("save_lora_adapter", False):
        return
    if config.actor_rollout_ref.actor.strategy not in ("fsdp", "fsdp2"):
        raise ValueError("LoRA adapter export currently requires the FSDP diffusion engine")
    model = config.actor_rollout_ref.model
    if model.get("lora_rank", 0) <= 0 and not model.get("lora_adapter_path"):
        raise ValueError("save_lora_adapter requires a LoRA model")
    adapter = checkpoint.get("lora_adapter_name", "default")
    if not adapter or adapter == "reference" or adapter not in model.get("policy_state_adapters", ["default"]):
        raise ValueError(f"No exportable policy adapter {adapter!r}")


def maybe_export_lora_adapter(trainer, *, is_last_step, checkpoint_saved):
    """Export once per completed step without changing training checkpoint retention."""
    config = trainer.config.actor_rollout_ref.actor.checkpoint
    if not config.get("save_lora_adapter", False):
        return
    step = trainer.global_steps
    frequency = config.get("save_lora_adapter_freq", 0)
    due = is_last_step or (frequency == 0 and checkpoint_saved) or (frequency > 0 and step % frequency == 0)
    if step <= 0 or not due or getattr(trainer, "_last_lora_export_step", None) == step:
        return
    output_dir = Path(trainer.config.trainer.default_local_dir) / "lora_adapters" / f"global_step_{step}"
    trainer.actor_rollout_wg.export_lora_adapter(
        str(output_dir), global_step=step, adapter_name=config.get("lora_adapter_name", "default")
    )
    trainer._last_lora_export_step = step
