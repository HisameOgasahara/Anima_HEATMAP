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
    assert "'AnimaVAblationSampler'" in sources
    assert "SOURCE / node_name" in sources
    workflow = json.loads((ROOT / "example/tag_influence.json").read_text(encoding="utf-8"))
    by_id = {node["id"]: node for node in workflow["nodes"]}
    for link_id, origin, slot, target, input_slot, kind in workflow["links"]:
        assert link_id in by_id[origin]["outputs"][slot]["links"]
        assert by_id[target]["inputs"][input_slot]["link"] == link_id
        assert by_id[origin]["outputs"][slot]["type"] == kind
        assert by_id[target]["inputs"][input_slot]["type"] == kind


@pytest.fixture
def v_ablation(influence, monkeypatch):
    monkeypatch.setitem(sys.modules, "influence_test_nodes.influence", influence)
    parser = ModuleType("comfy.sd1_clip")
    parser.escape_important = parser.unescape_important = lambda text: text
    parser.token_weights = lambda text, weight: [(text, weight)]
    monkeypatch.setitem(sys.modules, "comfy.sd1_clip", parser)
    spec = importlib.util.spec_from_file_location(
        "influence_test_nodes.v_ablation", ROOT / "comfyui_anima_heatmap/v_ablation.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v_hook_preserves_qk_and_other_values_and_restores_on_error(v_ablation):
    class Attention:
        def compute_qkv(self, x, context=None, **kwargs):
            return x, context, context

    blocks = [SimpleNamespace(cross_attn=Attention(), self_attn=Attention()) for _ in range(2)]
    # 이미 인스턴스에 연결된 메서드도 복원해야 한다.
    saved = blocks[0].cross_attn.compute_qkv
    blocks[0].cross_attn.compute_qkv = saved
    q = torch.ones(2, 6, 2, 3)
    values = torch.ones(2, 5, 2, 3)
    with pytest.raises(RuntimeError, match="comparison failed"):
        with v_ablation.ablate_token_values(SimpleNamespace(blocks=blocks), [1, 3]):
            for block in blocks:
                observed_q, observed_k, observed_v = block.cross_attn.compute_qkv(q, values)
                assert observed_q is q and observed_k is values
                assert torch.count_nonzero(observed_v[:, [1, 3]]) == 0
                torch.testing.assert_close(observed_v[:, [0, 2, 4]], values[:, [0, 2, 4]])
                assert block.self_attn.compute_qkv(q, values)[2] is values
            assert torch.all(values == 1)
            raise RuntimeError("comparison failed")
    assert blocks[0].cross_attn.compute_qkv is saved
    assert "compute_qkv" not in blocks[1].cross_attn.__dict__


@pytest.mark.parametrize("cfg", [1, 4])
def test_v_node_changes_comparison_only_and_shares_view(v_ablation, influence, monkeypatch, cfg):
    class Attention:
        def compute_qkv(self, x, context=None, **kwargs):
            return x, context, context

    class Anima:
        def __init__(self):
            self.blocks = [SimpleNamespace(cross_attn=Attention()) for _ in range(2)]

    for name in ("comfy.ldm", "comfy.ldm.anima", "comfy.ldm.anima.model"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    sys.modules["comfy.ldm.anima.model"].Anima = Anima
    diffusion = Anima()
    model = SimpleNamespace(load_device="cpu", model_options={},
                            get_model_object=lambda name: diffusion)
    calls = []

    def predict(model, conds, x, timestep, options):
        results = []
        for cond in conds:
            if cond is None:
                results.append(None)
                continue
            context = cond[0][0]
            values = [block.cross_attn.compute_qkv(x, context)[2] for block in diffusion.blocks]
            calls.append((float(context.mean()), float(values[0].sum())))
            results.append(torch.full_like(x, sum(float(v.sum()) for v in values)))
        return results

    monkeypatch.setattr(sys.modules["comfy.samplers"], "calc_cond_batch", predict)
    token_map = dict(ids=[0, 1, 2], pieces=["▁sketch", "book", "▁city"], special_indices=[])
    positive = [[torch.ones(1, 3, 2, 2), {"anima_heatmap_tokens": token_map}]]
    negative = [[torch.zeros(1, 3, 2, 2), {}]]
    latent = {"samples": torch.zeros(1, 2, 2, 2)}
    output, result = v_ablation.AnimaVAblationSampler().sample(
        model, positive, negative, latent, "sketchbook", seed=42, steps=2, cfg=cfg,
        sampler_name="euler", scheduler="simple")
    torch.testing.assert_close(output["samples"], torch.full_like(latent["samples"], 48 * cfg))
    assert result["token_indices"] == [0, 1]
    assert result["method"] == "v_ablation"
    assert [r["step"] for r in result["records"]] == [0, 1]
    np.testing.assert_allclose(result["velocity_maps"][:, 0, 0, 0],
                               np.array([1, 2]) * 16 * cfg * np.sqrt(2))
    assert all(total == 0 for mean, total in calls if mean == 0)
    assert all("compute_qkv" not in block.cross_attn.__dict__ for block in diffusion.blocks)
    overlays, heatmaps, _ = influence.AnimaTagInfluenceView().view(
        result, torch.zeros(1, 32, 32, 3), 0, 0.5, "per_step", "all")
    assert overlays.shape == heatmaps.shape and overlays.shape[0] == 2


def test_v_node_rejects_missing_phrase_before_model_execution(v_ablation, monkeypatch):
    class Anima:
        blocks = []

    module = ModuleType("comfy.ldm.anima.model")
    module.Anima = Anima
    monkeypatch.setitem(sys.modules, module.__name__, module)
    model = SimpleNamespace(get_model_object=lambda name: Anima())
    positive = [[torch.ones(1, 1, 2, 2), {"anima_heatmap_tokens": dict(
        ids=[0], pieces=["▁city"], special_indices=[])}]]
    with pytest.raises(ValueError, match="찾지 못했습니다"):
        v_ablation.AnimaVAblationSampler().sample(model, positive, [], {}, "sketchbook",
            seed=42, steps=2, cfg=4, sampler_name="euler", scheduler="simple")
