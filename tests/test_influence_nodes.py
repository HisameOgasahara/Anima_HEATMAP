import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch


ROOT = Path(__file__).parents[1]


@pytest.fixture
def influence(monkeypatch):
    class CFGGuider:
        def __init__(self, model):
            self.inner_model = model

        def inner_set_conds(self, conds):
            self.conds = conds

        def set_cfg(self, cfg):
            self.cfg = cfg

        def sample(self, noise, latent, sampler, sigmas, **kwargs):
            state = latent.clone()
            for sigma in sigmas[:-1]:
                prediction = self.predict_noise(state, sigma.reshape(1))
                # 비교 예측을 생성 경로에 섞으면 최종 상태가 달라진다.
                state = state + prediction
            return state

    comfy = ModuleType("comfy")
    for name in ("samplers", "sample", "model_management", "utils"):
        module = ModuleType(f"comfy.{name}")
        setattr(comfy, name, module)
        monkeypatch.setitem(sys.modules, f"comfy.{name}", module)
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    comfy.samplers.CFGGuider = CFGGuider
    comfy.samplers.calc_cond_batch = lambda model, conds, x, t, opts: [
        torch.full_like(x, cond) if cond is not None else None for cond in conds]
    comfy.samplers.KSampler = lambda model, steps, **kwargs: SimpleNamespace(
        sigmas=torch.linspace(1, 0, steps + 1))
    comfy.samplers.sampler_object = lambda name: name
    comfy.sample.fix_empty_latent_channels = lambda model, samples, *args: samples
    comfy.sample.prepare_noise = lambda latent, *args: torch.zeros_like(latent)
    comfy.model_management.intermediate_device = lambda: "cpu"
    comfy.utils.PROGRESS_BAR_ENABLED = False
    preview = ModuleType("latent_preview")
    preview.prepare_callback = lambda model, steps: None
    monkeypatch.setitem(sys.modules, "latent_preview", preview)
    package = ModuleType("influence_test_nodes")
    package.__path__ = [str(ROOT / "comfyui_anima_heatmap")]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    spec = importlib.util.spec_from_file_location(
        package.__name__ + ".influence", ROOT / "comfyui_anima_heatmap/influence.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("steps", [1, 7, 30])
def test_sampling_uses_requested_steps_and_original_path(influence, steps):
    latent = {"samples": torch.zeros(1, 2, 2, 2)}
    output, result = influence.sample_influence(
        SimpleNamespace(load_device="cpu", model_options={}), 2, 0,
        {"unchanged": 2, "removed": 1}, latent,
        seed=42, steps=steps, cfg=4, sampler_name="euler", scheduler="simple")
    assert result["total_steps"] == steps
    assert [r["step"] for r in result["records"]] == list(range(steps))
    torch.testing.assert_close(output["samples"], torch.full_like(latent["samples"], 8 * steps))
    np.testing.assert_array_equal(result["targets"]["unchanged"]["velocity_maps"], 0)
    assert result["targets"]["removed"]["velocity_maps"].min() > 0


def test_multiple_model_calls_share_schedule_interval(influence):
    guider = influence.InfluenceGuider(object(), 2, 0, 1, 4, [1.0, 0.5, 0.0])
    for sigma in (1.0, 0.75, 0.5, 0.25):
        guider.predict_noise(torch.zeros(1, 2, 2, 2), torch.tensor([sigma]))
    assert [r["step"] for r in guider.records] == [0, 0, 1, 1]
    assert [r["call"] for r in guider.records] == [0, 1, 2, 3]


def test_view_groups_calls_and_keeps_scale_when_selecting_steps(influence):
    maps = np.stack([np.full((1, 2, 2), value, np.float32) for value in (1, 3, 8)])
    result = dict(total_steps=2, velocity_maps=maps, records=[
        dict(step=0, call=0), dict(step=0, call=1), dict(step=1, call=2)])
    images = torch.zeros(1, 32, 32, 3)
    overlays, heatmaps, details = influence.AnimaTagInfluenceView().view(
        result, images, 0, 0.5, "per_step", "all")
    info = json.loads(details)
    assert overlays.shape == heatmaps.shape
    assert overlays.shape[0] == 2
    assert info["scale_max"] == 8
    assert "step 1/2" in info["outputs"][0]
    assert "step 2/2" in info["outputs"][1]
    _, _, selected_details = influence.AnimaTagInfluenceView().view(
        result, images, 0, 0.5, "per_step", "0")
    selected = json.loads(selected_details)
    assert selected["scale_max"] == 8
    assert selected["selected_steps"] == [0]
    with pytest.raises(ValueError, match="선택 범위"):
        influence.AnimaTagInfluenceView().view(result, images, 0, 0.5, "per_step", "2")


def test_colab_downloads_feature_workflow_and_requires_nodes():
    notebook = json.loads((ROOT / "notebooks/Anima_Heatmap_ComfyUI.ipynb").read_text(encoding="utf-8"))
    sources = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert "SOURCE_REF = 'feature/tag-influence'" in sources
    assert "SOURCE / 'example/tag_influence.json'" in sources
    assert "'AnimaTagInfluenceSampler', 'AnimaTagInfluenceView'" in sources
    workflow = json.loads((ROOT / "example/tag_influence.json").read_text(encoding="utf-8"))
    by_id = {node["id"]: node for node in workflow["nodes"]}
    for link_id, origin, slot, target, input_slot, kind in workflow["links"]:
        assert link_id in by_id[origin]["outputs"][slot]["links"]
        assert by_id[target]["inputs"][input_slot]["link"] == link_id
        assert by_id[origin]["outputs"][slot]["type"] == kind
        assert by_id[target]["inputs"][input_slot]["type"] == kind
