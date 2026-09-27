# Export diffusion LoRA adapters during training

Last updated: 09/27/2026.

The FSDP diffusion engines can export one named transformer adapter directly
from the training model. Both the legacy diffusion trainers and V1 trainers
support this path. It is disabled by default.

Add these overrides to a diffusion LoRA training command:

```bash
actor_rollout_ref.actor.checkpoint.save_lora_adapter=true \
actor_rollout_ref.actor.checkpoint.save_lora_adapter_freq=100 \
actor_rollout_ref.actor.checkpoint.lora_adapter_name=default
```

A positive frequency exports every N completed training steps, independently
of `trainer.save_freq`. Zero follows full-checkpoint saves. When enabled, the
final completed step is also exported, even if full checkpoints are disabled.
Coincident triggers export once. Negative frequencies are invalid.

Exports are kept separately from resumable checkpoints:

```text
<trainer.default_local_dir>/lora_adapters/global_step_100/
  adapter_config.json
  adapter_model.safetensors
  training_metadata.json
```

The directory contains the selected adapter's PEFT configuration and weights.
Metadata records its transformer component, original adapter name, base-model
path or identifier, training step, PEFT weight layout, and format/software
version. The adapter must be loaded alongside the same base transformer.
The base-model identifier refers to the training model; for a diffusion
pipeline, load its transformer component before attaching the adapter.

## Adapter selection and training checkpoints

Training may keep multiple adapters, for example `default` and `old`. Set
`lora_adapter_name=old` to export that registered adapter, even when it is frozen
or inactive. Export does not switch the active adapter or change trainability.
`reference` is an adapter-disabled policy state and cannot be exported.

`save_lora_only` continues to control resumable training checkpoint contents.
`model.lora.merge` continues to control rollout synchronization. Neither option
changes this export: the files contain separate, unmerged LoRA weights.
Exports contain no optimizer or dataloader state and do not update the resume
pointer. Full-checkpoint retention does not delete the separate adapter root.
Automatic adapter retention is not implemented.

## Initial scope and validation

The initial implementation supports a single transformer component and one
adapter per export, using ordinary A/B LoRA with `bias=none`. Full configuration
serialization preserves rank/alpha patterns and RS-LoRA scaling. DoRA, embedding
LoRA, `modules_to_save`, `target_parameters`, adapter biases, and initializers
that may modify base weights are rejected. VeOmni and multi-component bundles
are outside this implementation.

FSDP1 uses full parameter summon to include adapters outside transformer
blocks; it may temporarily require full-model memory. FSDP2 materializes
selected adapter DTensors. Export restores manually offloaded parameters and
leaves FSDP2 CPU-offload policy placement under PyTorch's control.

All actor ranks participate in gathering; rank zero validates and publishes
files through a temporary directory on the same filesystem. Existing exports
are not overwritten. Recoverable validation/write errors are propagated to
all ranks. Failed process-group collectives remain subject to the distributed
runtime's timeout and failure handling. Output must be local to the writer or
on a shared filesystem; remote publication is not supported.

CPU tests cover PEFT reload with matching fixed-input outputs, named-adapter
selection, scheduling, writer failures, and mocked FSDP/offload paths. GPU
multi-rank FSDP/offload validation remains required before production use.

The artifact uses PEFT component keys, not a vLLM-Omni-specific fused layout.
Independent vLLM-Omni loading has not been validated here. In particular,
MiniMax H3's online tensor mapper is not automatically applied by its file
loader; this export alone does not establish H3 inference compatibility.

Run the focused CPU tests with:

```bash
TORCH_COMPILE_DISABLE=1 TORCHINDUCTOR_DISABLE=1 python -m pytest -q \
  tests/utils/test_lora_export_on_cpu.py \
  tests/trainer/diffusion/test_lora_export_on_cpu.py \
  tests/workers/test_diffusion_adapter_export_on_cpu.py
```
