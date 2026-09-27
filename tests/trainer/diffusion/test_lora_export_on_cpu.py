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
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from omegaconf import OmegaConf

_PATH = Path(__file__).parents[3] / "verl_omni/trainer/diffusion/lora_export.py"
_SPEC = importlib.util.spec_from_file_location("lora_export_schedule_under_test", _PATH)
schedule = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(schedule)


@pytest.fixture
def trainer(tmp_path):
    return SimpleNamespace(
        config=OmegaConf.create(
            {
                "actor_rollout_ref": {
                    "actor": {
                        "strategy": "fsdp2",
                        "checkpoint": {
                            "save_lora_adapter": True,
                            "save_lora_adapter_freq": 3,
                            "lora_adapter_name": "default",
                        },
                    },
                    "model": {"lora_rank": 2, "policy_state_adapters": ["default", "old"]},
                },
                "trainer": {"default_local_dir": str(tmp_path), "save_freq": -1},
            }
        ),
        global_steps=1,
        actor_rollout_wg=SimpleNamespace(export_lora_adapter=Mock()),
    )


@pytest.mark.parametrize(
    "frequency,checkpoints,expected",
    [
        (3, set(), [3, 6, 7]),
        (0, {2, 4, 6}, [2, 4, 6, 7]),
        (0, set(), [7]),
        (3, {3, 6}, [3, 6, 7]),
        (1, set(), list(range(1, 8))),
    ],
)
def test_schedule_final_step_and_dedup(trainer, frequency, checkpoints, expected):
    trainer.config.actor_rollout_ref.actor.checkpoint.save_lora_adapter_freq = frequency
    schedule.validate_lora_export_config(trainer.config)
    for step in range(1, 8):
        trainer.global_steps = step
        for _ in range(2):
            schedule.maybe_export_lora_adapter(trainer, is_last_step=step == 7, checkpoint_saved=step in checkpoints)
    calls = trainer.actor_rollout_wg.export_lora_adapter.call_args_list
    assert [call.kwargs["global_step"] for call in calls] == expected
    for call in calls:
        assert Path(call.args[0]).parent.name == "lora_adapters"
        assert call.kwargs["adapter_name"] == "default"


def test_disabled_is_noop_and_step_zero_is_not_exported(trainer):
    trainer.global_steps = 0
    schedule.maybe_export_lora_adapter(trainer, is_last_step=True, checkpoint_saved=True)
    trainer.global_steps = 1
    trainer.config.actor_rollout_ref.actor.checkpoint.save_lora_adapter = False
    schedule.maybe_export_lora_adapter(trainer, is_last_step=True, checkpoint_saved=True)
    trainer.actor_rollout_wg.export_lora_adapter.assert_not_called()


@pytest.mark.parametrize(
    "path,value",
    [
        ("actor.checkpoint.save_lora_adapter_freq", -1),
        ("actor.checkpoint.save_lora_adapter_freq", 1.5),
        ("actor.strategy", "veomni"),
        ("model.lora_rank", 0),
        ("actor.checkpoint.lora_adapter_name", "reference"),
        ("actor.checkpoint.lora_adapter_name", "missing"),
    ],
)
def test_invalid_config_rejected(trainer, path, value):
    OmegaConf.update(trainer.config.actor_rollout_ref, path, value)
    with pytest.raises(ValueError):
        schedule.validate_lora_export_config(trainer.config)


def test_failed_export_does_not_mark_step_completed(trainer):
    trainer.global_steps = 3
    trainer.actor_rollout_wg.export_lora_adapter.side_effect = RuntimeError("disk full")
    with pytest.raises(RuntimeError, match="disk full"):
        schedule.maybe_export_lora_adapter(trainer, is_last_step=False, checkpoint_saved=False)
    assert not hasattr(trainer, "_last_lora_export_step")
    trainer.actor_rollout_wg.export_lora_adapter.side_effect = None
    trainer.config.actor_rollout_ref.actor.checkpoint.lora_adapter_name = "old"
    schedule.maybe_export_lora_adapter(trainer, is_last_step=False, checkpoint_saved=False)
    assert trainer._last_lora_export_step == 3
    assert trainer.actor_rollout_wg.export_lora_adapter.call_args.kwargs["adapter_name"] == "old"
