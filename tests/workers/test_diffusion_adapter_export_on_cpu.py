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

import ast
import importlib.util
import os
import sys
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from peft import LoraConfig, PeftModel, get_peft_model, get_peft_model_state_dict, inject_adapter_in_model

ROOT = Path(__file__).parents[2]


def _load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def helpers(monkeypatch):
    upstream = ModuleType("verl.utils.fsdp_utils")
    upstream.fsdp_version = Mock(return_value=0)
    upstream.collect_lora_params = Mock(side_effect=AssertionError("rollout collector must not be used"))
    upstream.layered_summon_lora_params = Mock()
    device = ModuleType("verl.utils.device")
    device.get_torch_device = lambda: SimpleNamespace(empty_cache=Mock())
    monkeypatch.setitem(sys.modules, upstream.__name__, upstream)
    monkeypatch.setitem(sys.modules, device.__name__, device)
    fsdp = _load_source("export_fsdp_helpers", "verl_omni/utils/fsdp_utils.py")
    export = _load_source("export_state_helpers", "verl_omni/utils/lora_export.py")
    return fsdp, export


@pytest.fixture
def model():
    model = get_peft_model(torch.nn.Sequential(torch.nn.Linear(4, 4)), LoraConfig(r=2, target_modules=["0"]))
    model.add_adapter("old", LoraConfig(r=1, target_modules=["0"]))
    return model


@pytest.mark.parametrize("version", [0, 1, 2])
def test_collector_selects_frozen_adapter_without_switching(helpers, model, monkeypatch, version):
    fsdp, export = helpers
    fsdp.fsdp_version.return_value = version
    from torch.distributed.fsdp import FullyShardedDataParallel

    monkeypatch.setattr(FullyShardedDataParallel, "summon_full_params", lambda *a, **kw: nullcontext())
    config, expected = export.inspect_lora_adapter(model, "old")
    state = export.prepare_adapter_tensors(fsdp.collect_lora_adapter_params(model, "old"), expected)
    reference = get_peft_model_state_dict(model, adapter_name="old")
    assert state.keys() == reference.keys()
    for key in state:
        torch.testing.assert_close(state[key], reference[key])
    assert model.active_adapter == "default"
    assert config["r"] == 1


def test_offloaded_dtensor_materializes_on_mesh_device(helpers):
    fsdp, _ = helpers
    shard = Mock(device=SimpleNamespace(type="cpu"), device_mesh=SimpleNamespace(device_type="cuda"))
    full = torch.randn(3, 4)
    shard.to.return_value.full_tensor.return_value = full
    torch.testing.assert_close(fsdp._param_to_cpu(shard), full)
    shard.to.assert_called_once_with("cuda")
    shard.full_tensor.assert_not_called()


def test_injected_component_exports_peft_reloadable_keys(helpers, tmp_path):
    fsdp, export = helpers
    base = torch.nn.Sequential(torch.nn.Linear(4, 4))
    original = deepcopy(base)
    component = inject_adapter_in_model(LoraConfig(r=2, target_modules=["0"]), base)
    with torch.no_grad():
        component[0].lora_B["default"].weight.fill_(0.3)
    config, expected = export.inspect_lora_adapter(component, "default")
    tensors = export.prepare_adapter_tensors(fsdp.collect_lora_adapter_params(component), expected)
    destination = tmp_path / "adapter"
    export.write_lora_adapter(destination, tensors, config, {"component": "transformer"})
    restored = PeftModel.from_pretrained(original, destination)
    x = torch.randn(3, 4)
    torch.testing.assert_close(restored(x), component(x))


def _engine_export_method(namespace):
    # Execute the production method in isolation from GPU/Ray import-time registrations.
    path = ROOT / "verl_omni/workers/engine/fsdp/diffusers_impl.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    engine = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "DiffusersFSDPEngine")
    method = next(
        node for node in engine.body if isinstance(node, ast.FunctionDef) and node.name == "export_lora_adapter"
    )
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["export_lora_adapter"]


@pytest.mark.parametrize("policy_offload", [False, True])
@pytest.mark.parametrize("write_fails", [False, True])
def test_engine_restores_placement_and_does_not_merge(
    helpers, model, monkeypatch, tmp_path, policy_offload, write_fails
):
    fsdp, export = helpers
    package = ModuleType("verl_omni")
    package.__version__ = "test"
    monkeypatch.setitem(sys.modules, "verl_omni", package)
    load, offload = Mock(), Mock()
    writer = Mock(side_effect=OSError("disk full") if write_fails else None)
    namespace = {
        "os": os,
        "inspect_lora_adapter": export.inspect_lora_adapter,
        "coordinated_export_phase": export.coordinated_export_phase,
        "prepare_adapter_tensors": export.prepare_adapter_tensors,
        "collect_lora_adapter_params": fsdp.collect_lora_adapter_params,
        "load_fsdp_model_to_gpu": load,
        "offload_fsdp_model_to_cpu": offload,
        "write_lora_adapter": writer,
    }
    engine = SimpleNamespace(
        module=model,
        rank=0,
        _uses_fsdp2_cpu_offload_policy=policy_offload,
        model_config=SimpleNamespace(path="original-model", lora={"merge": True}),
    )
    run_export = _engine_export_method(namespace)
    with pytest.raises(RuntimeError, match="disk full") if write_fails else nullcontext():
        run_export(engine, str(tmp_path / "adapter"), global_step=4, adapter_name="old")
    assert load.call_count == offload.call_count == (0 if policy_offload else 1)
    assert model.active_adapter == "default"
    args = writer.call_args.args
    assert args[2]["r"] == 1
    assert args[2]["base_model_name_or_path"] == "original-model"
    assert args[3]["adapter_name"] == "old"
    assert args[3]["global_step"] == 4
    assert all("lora_" in key for key in args[1])
