# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import importlib
import os
import pickle
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch

import safetensors.torch
import torch

from hybrid_llm.pair_ranker.config import NClassRankerConfig


# No pretrained model downloads or LLM-Blender installation are needed here.
blender_model_util = ModuleType("llm_blender.pair_ranker.model_util")
blender_model_util.build_pretrained_model = Mock()
with patch.dict(
    "sys.modules",
    {"llm_blender.pair_ranker.model_util": blender_model_util},
):
    model_util = importlib.import_module("hybrid_llm.pair_ranker.model_util")


class DirectoryCreatingPayload:
    def __init__(self, path):
        self.path = str(path)

    def __reduce__(self):
        return os.mkdir, (self.path,)


class CheckpointLoadingTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.checkpoint_path = Path(self.temp_dir.name)
        self.config = NClassRankerConfig()
        self.model = torch.nn.Linear(2, 1)
        self.state_dict = {
            "weight": torch.tensor([[1.0, 2.0]]),
            "bias": torch.tensor([3.0]),
        }
        self.builder = patch.object(
            model_util, "build_nclass_ranker", return_value=self.model
        )
        self.builder.start()
        self.addCleanup(self.builder.stop)

    def load_model(self):
        return model_util.build_nclass_ranker_from_checkpoint(
            str(self.checkpoint_path), self.config, None
        )

    def assert_weights_loaded(self):
        for name, value in self.model.state_dict().items():
            self.assertTrue(torch.equal(value, self.state_dict[name]))
            self.assertEqual(value.device.type, "cpu")

    def test_binary_weights_load_restricted_on_cpu(self):
        path = self.checkpoint_path / "pytorch_model.bin"
        torch.save(self.state_dict, path)
        with patch.object(model_util.torch, "load", wraps=torch.load) as load:
            self.assertIs(self.load_model(), self.model)
        load.assert_called_once_with(
            str(path), map_location="cpu", weights_only=True
        )
        self.assert_weights_loaded()

    def test_safetensors_weights_load_on_cpu(self):
        safetensors.torch.save_file(
            self.state_dict, self.checkpoint_path / "model.safetensors"
        )
        self.assertIs(self.load_model(), self.model)
        self.assert_weights_loaded()

    def test_safetensors_preferred_without_opening_pickle(self):
        marker = self.checkpoint_path / "executed"
        torch.save(
            DirectoryCreatingPayload(marker),
            self.checkpoint_path / "pytorch_model.bin",
        )
        safetensors.torch.save_file(
            self.state_dict, self.checkpoint_path / "model.safetensors"
        )
        with patch.object(model_util.torch, "load") as load:
            self.load_model()
        load.assert_not_called()
        self.assertFalse(marker.exists())
        self.assert_weights_loaded()

    def test_invalid_safetensors_does_not_fall_back_to_pickle(self):
        (self.checkpoint_path / "model.safetensors").write_bytes(b"invalid")
        torch.save(self.state_dict, self.checkpoint_path / "pytorch_model.bin")
        with patch.object(model_util.torch, "load") as load:
            with self.assertRaises(safetensors.SafetensorError):
                self.load_model()
        load.assert_not_called()

    def test_missing_checkpoint_has_explicit_error(self):
        with self.assertRaisesRegex(FileNotFoundError, "not found in path"):
            self.load_model()

    def test_missing_keys_still_rejected(self):
        torch.save(
            {"weight": self.state_dict["weight"]},
            self.checkpoint_path / "pytorch_model.bin",
        )
        with self.assertRaisesRegex(Exception, "missing keys"):
            self.load_model()

    def test_binary_payload_is_rejected_without_execution(self):
        marker = self.checkpoint_path / "executed"
        torch.save(
            DirectoryCreatingPayload(marker),
            self.checkpoint_path / "pytorch_model.bin",
        )
        with self.assertRaises(pickle.UnpicklingError):
            self.load_model()
        self.assertFalse(marker.exists())

    def test_no_unsafe_retry_when_restricted_load_fails(self):
        (self.checkpoint_path / "pytorch_model.bin").touch()
        with patch.object(
            model_util.torch, "load", side_effect=pickle.UnpicklingError("rejected")
        ) as load:
            with self.assertRaises(pickle.UnpicklingError):
                self.load_model()
        self.assertEqual(load.call_count, 1)
        self.assertTrue(load.call_args.kwargs["weights_only"])

    def test_existing_config_round_trip(self):
        self.config.n_candidates = 6
        self.config.candidate_models = ["small", "large"]
        torch.save(self.config, self.checkpoint_path / "config.bin")
        safe_globals_before = torch.serialization.get_safe_globals()
        with patch.object(model_util.torch, "load", wraps=torch.load) as load:
            config = model_util.load_ranker_config(str(self.checkpoint_path))
        self.assertEqual(config, self.config)
        self.assertEqual(
            torch.serialization.get_safe_globals(), safe_globals_before
        )
        load.assert_called_once_with(
            str(self.checkpoint_path / "config.bin"),
            map_location="cpu",
            weights_only=True,
        )

    def test_config_payload_is_rejected_without_execution(self):
        marker = self.checkpoint_path / "executed"
        torch.save(
            DirectoryCreatingPayload(marker), self.checkpoint_path / "config.bin"
        )
        safe_globals_before = torch.serialization.get_safe_globals()
        with self.assertRaises(pickle.UnpicklingError):
            model_util.load_ranker_config(str(self.checkpoint_path))
        self.assertFalse(marker.exists())
        self.assertEqual(
            torch.serialization.get_safe_globals(), safe_globals_before
        )

    def test_wrong_config_type_rejected(self):
        torch.save({"model_name": "unexpected"}, self.checkpoint_path / "config.bin")
        with self.assertRaisesRegex(ValueError, "Expected NClassRankerConfig"):
            model_util.load_ranker_config(str(self.checkpoint_path))


if __name__ == "__main__":
    unittest.main()
