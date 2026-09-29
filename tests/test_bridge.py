import importlib.util
from pathlib import Path
import sys
import types

import pytest
import torch

from anima_heatmap import CaptureConfig, CaptureSession


spec = importlib.util.spec_from_file_location("heatmap_bridge", Path(__file__).parents[1] / "comfyui_anima_heatmap/bridge.py")
bridge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bridge)


class Attention:
    def compute_qkv(self, x, context=None, rope_emb=None, transformer_options=None):
        return x, context, context


class FakeAnima:
    patch_spatial = 2
    patch_temporal = 1

    def __init__(self):
        self.blocks = [types.SimpleNamespace(cross_attn=Attention(), self_attn=Attention())]


class Model:
    def __init__(self, diffusion):
        self.diffusion = diffusion
        self.model_options = {}

    def get_model_object(self, name):
        return self.diffusion

    def clone(self):
        result = Model(self.diffusion)
        result.model_options = self.model_options.copy()
        return result

    def set_model_unet_function_wrapper(self, wrapper):
        self.model_options["model_function_wrapper"] = wrapper


def install_fake_anima(monkeypatch):
    for name in ("comfy", "comfy.ldm", "comfy.ldm.anima", "comfy.ldm.anima.model"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules["comfy.ldm.anima.model"].Anima = FakeAnima


@pytest.mark.parametrize("labels", [[0], [1], [0, 1], [1, 0]])
def test_native_grid_cfg_and_unchanged_qkv(tmp_path, monkeypatch, labels):
    install_fake_anima(monkeypatch)
    diffusion = FakeAnima()
    model = Model(diffusion)
    with CaptureSession(tmp_path, CaptureConfig(branches=("positive", "negative"))) as session:
        with bridge.attach_capture(model, session, [1, 0.5, 0]) as (patched, state):
            count = len(labels)
            q = torch.randn(count, 6, 2, 4)
            k = torch.randn(count, 5, 2, 4)

            def apply(x, timestep, transformer_options):
                return diffusion.blocks[0].cross_attn.compute_qkv(q, k, transformer_options=transformer_options)

            result = patched.model_options["model_function_wrapper"](apply, dict(
                input=torch.zeros(count, 4, 1, 4, 6), timestep=torch.tensor([0.5]),
                c={}, cond_or_uncond=labels))
            assert result[0] is q and result[1] is k
    assert "compute_qkv" not in diffusion.blocks[0].cross_attn.__dict__
    assert "model_function_wrapper" not in model.model_options
    assert [r["branch"] for r in session.records] == ["positive" if x == 0 else "negative" for x in labels]
    assert all(r["grid"] == [1, 2, 3] and r["step"] == 1 for r in session.records)


def test_hooks_restored_on_failure(tmp_path, monkeypatch):
    install_fake_anima(monkeypatch)
    diffusion = FakeAnima()
    with pytest.raises(RuntimeError, match="sampling failure"):
        with CaptureSession(tmp_path) as session:
            with bridge.attach_capture(Model(diffusion), session, [1, 0]) as unused:
                raise RuntimeError("sampling failure")
    assert "compute_qkv" not in diffusion.blocks[0].cross_attn.__dict__
    assert session.manifest["status"] == "failed"


def test_negative_only_is_not_recorded_as_positive(tmp_path, monkeypatch):
    install_fake_anima(monkeypatch)
    diffusion = FakeAnima()
    with CaptureSession(tmp_path) as session:
        with bridge.attach_capture(Model(diffusion), session, [1, 0]) as (patched, _):
            def apply(x, timestep, transformer_options):
                return diffusion.blocks[0].cross_attn.compute_qkv(
                    torch.zeros(1, 1, 2, 3, 2, 4), torch.zeros(1, 5, 2, 4),
                    transformer_options=transformer_options)
            patched.model_options["model_function_wrapper"](apply, dict(input=torch.zeros(1, 4, 4, 6),
                timestep=torch.tensor([1.0]), c={}, cond_or_uncond=[1]))
    assert session.records == []

