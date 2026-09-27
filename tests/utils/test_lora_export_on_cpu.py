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

import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from peft import LoraConfig, PeftModel, get_peft_model, get_peft_model_state_dict
from safetensors.torch import load_file

_PATH = Path(__file__).parents[2] / "verl_omni/utils/lora_export.py"
_SPEC = importlib.util.spec_from_file_location("lora_export_under_test", _PATH)
export = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(export)


class ToyTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(4, 4, bias=False)
        self.output = torch.nn.Linear(4, 4, bias=False)

    def forward(self, x):
        return self.output(self.proj(x))


@pytest.fixture
def model():
    torch.manual_seed(7)
    config = LoraConfig(
        r=2,
        lora_alpha=4,
        target_modules=["proj", "output"],
        rank_pattern={"output": 1},
        alpha_pattern={"output": 3},
        use_rslora=True,
    )
    model = get_peft_model(ToyTransformer(), config)
    model.add_adapter("old", LoraConfig(r=1, lora_alpha=5, target_modules=["proj"]))
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_B" in name:
                parameter.fill_(0.4 if ".old." in name else 0.2)
    return model


@pytest.mark.parametrize("adapter", ["default", "old"])
def test_export_reload_preserves_selected_adapter_and_outputs(model, adapter, tmp_path):
    model.set_adapter(adapter)
    x = torch.randn(3, 4)
    expected_output = model(x).detach()
    base = ToyTransformer()
    base.proj.weight.data.copy_(model.base_model.model.proj.base_layer.weight)
    base.output.weight.data.copy_(model.base_model.model.output.base_layer.weight)
    model.set_adapter("old" if adapter == "default" else "default")
    before_active = model.active_adapter
    before_grad = {name: parameter.requires_grad for name, parameter in model.named_parameters()}
    before_config = deepcopy(model.peft_config[adapter].to_dict())
    config, keys = export.inspect_lora_adapter(model, adapter)
    tensors = export.prepare_adapter_tensors(get_peft_model_state_dict(model, adapter_name=adapter), keys)
    output_dir = tmp_path / "adapter"
    export.write_lora_adapter(output_dir, tensors, config, {"component": "transformer", "adapter_name": adapter})

    reloaded = PeftModel.from_pretrained(base, output_dir)
    torch.testing.assert_close(reloaded(x), expected_output)
    assert model.active_adapter == before_active
    assert before_grad == {name: parameter.requires_grad for name, parameter in model.named_parameters()}
    assert before_config == model.peft_config[adapter].to_dict()
    saved_config = json.loads((output_dir / "adapter_config.json").read_text())
    for key in ("r", "lora_alpha", "rank_pattern", "alpha_pattern", "use_rslora"):
        assert saved_config[key] == before_config[key]
    assert set(load_file(output_dir / "adapter_model.safetensors")) == keys
    assert not list(tmp_path.glob(".*"))


def test_missing_adapter_and_reference_are_rejected(model):
    for name in ("missing", "reference"):
        with pytest.raises(ValueError, match="No registered"):
            export.inspect_lora_adapter(model, name)
    with pytest.raises(ValueError, match="transformer component"):
        export.inspect_lora_adapter(model, "default", component="text_encoder")


@pytest.mark.parametrize(
    "field,value",
    [
        ("bias", "all"),
        ("use_dora", True),
        ("modules_to_save", ["head"]),
        ("target_parameters", ["experts"]),
        ("init_lora_weights", "pissa"),
    ],
)
def test_unsupported_config_fails_before_collection(model, field, value):
    setattr(model.peft_config["default"], field, value)
    with pytest.raises(ValueError, match="requires|initializers"):
        export.inspect_lora_adapter(model, "default")


def test_merged_layers_rejected(model):
    model.base_model.model.proj.merge()
    with pytest.raises(ValueError, match="unmerged"):
        export.inspect_lora_adapter(model, "default")


def test_incomplete_and_colliding_keys_are_rejected(model):
    _, keys = export.inspect_lora_adapter(model, "default")
    params = get_peft_model_state_dict(model)
    params.pop(next(iter(params)))
    with pytest.raises(ValueError, match="Incomplete adapter"):
        export.prepare_adapter_tensors(params, keys)
    with pytest.raises(ValueError, match="Duplicate adapter"):
        export.prepare_adapter_tensors(
            {"proj.lora_A.weight": torch.ones(2, 4), "base_model.model.proj.lora_A.weight": torch.ones(2, 4)}, set()
        )


def test_shape_mismatch_and_empty_tensors_rejected(model):
    _, keys = export.inspect_lora_adapter(model, "default")
    params = get_peft_model_state_dict(model)
    key = next(key for key in params if key.endswith(".lora_B.weight"))
    params[key] = torch.ones(4, 99)
    with pytest.raises(ValueError, match="shapes"):
        export.prepare_adapter_tensors(params, keys)
    params[key] = torch.empty(0)
    with pytest.raises(ValueError, match="empty adapter tensor"):
        export.prepare_adapter_tensors(params, keys)


def test_write_failure_cleans_staging_and_can_retry(model, tmp_path, monkeypatch):
    config, keys = export.inspect_lora_adapter(model, "default")
    tensors = export.prepare_adapter_tensors(get_peft_model_state_dict(model), keys)
    destination = tmp_path / "adapter"
    original = export.save_file

    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(export, "save_file", fail)
    with pytest.raises(OSError, match="disk full"):
        export.write_lora_adapter(destination, tensors, config, {})
    assert list(tmp_path.iterdir()) == []
    monkeypatch.setattr(export, "save_file", original)
    export.write_lora_adapter(destination, tensors, config, {})
    with pytest.raises(FileExistsError, match="overwrite"):
        export.write_lora_adapter(destination, tensors, config, {})


def test_rank_failure_is_propagated_to_successful_peer(monkeypatch):
    monkeypatch.setattr(export.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(export.dist, "get_world_size", lambda: 2)

    def gather(errors, local_error):
        errors[:] = [local_error, "OSError: writer failed"]

    monkeypatch.setattr(export.dist, "all_gather_object", gather)
    with pytest.raises(RuntimeError, match="rank 1: OSError: writer failed"):
        export.coordinated_export_phase(lambda: None)
