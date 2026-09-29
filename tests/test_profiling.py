import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from anima_heatmap import CaptureConfig, CaptureSession, compute_attention
from anima_heatmap.profiling import CaptureProfile, measure


def test_capture_counts_and_output_parity(tmp_path):
    q, k = torch.randn(1, 2, 4, 3), torch.randn(1, 2, 5, 3)
    expected = compute_attention(q, k)
    profile = CaptureProfile()
    with profile.activate(), profile.measure("sampling_total", "cpu"):
        with CaptureSession(tmp_path, CaptureConfig(query_chunk=2)) as session:
            session.capture(q, k, relation="image->text", branch="positive", step=0,
                            call=0, layer=0, grid=(1, 2, 2), total_steps=1, total_layers=1)
        actual = compute_attention(q, k)
    torch.testing.assert_close(actual, expected)
    data = profile.result()
    assert data["counts"]["attention_total"] == 2
    assert data["counts"]["cpu_transfer"] == 3
    assert data["counts"]["file_save"] == 1
    times = data["seconds"]
    assert times["attention_total"] >= times["cpu_transfer"]
    assert times["attention_compute"] + times["cpu_transfer"] == pytest.approx(times["attention_total"])


def test_context_is_restored_after_failure(monkeypatch):
    profile = CaptureProfile()
    calls = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: calls.append(device))
    with pytest.raises(RuntimeError):
        with profile.activate(), measure("test", "cuda:0"):
            raise RuntimeError("test")
    assert len(calls) == 2
    with measure("inactive", "cuda:0"):
        pass
    assert len(calls) == 2
    assert profile.counts == {"test": 1}


def test_profile_counts_fallback(monkeypatch):
    def fail(*args, **kwargs):
        raise torch.OutOfMemoryError("test")
    monkeypatch.setattr(torch, "softmax", fail)
    profile = CaptureProfile()
    with profile.activate():
        result = compute_attention(torch.zeros(1, 1, 4, 2), torch.zeros(1, 1, 5, 2),
                                   query_chunk=2, key_chunk=2)
    torch.testing.assert_close(result, torch.full((1, 1, 4, 5), 0.2))
    assert profile.counts["oom_fallback"] == 1
    assert profile.counts["cpu_transfer"] == 6


@pytest.mark.parametrize("fail", [False, True])
def test_separate_node_outputs_and_failure_report(tmp_path, monkeypatch, fail):
    class Sampler:
        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {"model": ("MODEL",)}}

        def sample(self, **kwargs):
            if fail:
                raise ValueError("sampling failed")
            return ({"samples": "unchanged"}, str(tmp_path), str(tmp_path))

    nodes = ModuleType("nodes")
    nodes.NODE_CLASS_MAPPINGS = {"AnimaHeatmapSampler": Sampler}
    folders = ModuleType("folder_paths")
    folders.get_output_directory = lambda: str(tmp_path)
    monkeypatch.setitem(sys.modules, "nodes", nodes)
    monkeypatch.setitem(sys.modules, "folder_paths", folders)
    spec = importlib.util.spec_from_file_location("profiler_node", Path(__file__).parents[1] / "comfyui_anima_profiler/__init__.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    node = module.AnimaHeatmapProfileSampler()
    assert node.INPUT_TYPES() == Sampler.INPUT_TYPES()
    if fail:
        with pytest.raises(ValueError, match="sampling failed"):
            node.sample(model=SimpleNamespace(load_device="cpu"))
    else:
        result = node.sample(model=SimpleNamespace(load_device="cpu"))
        assert result[:3] == ({"samples": "unchanged"}, str(tmp_path), str(tmp_path))
        assert Path(result[4]).is_file()
    data = json.loads(next(tmp_path.rglob("profile.json")).read_text(encoding="utf-8"))
    assert data["status"] == ("failed" if fail else "complete")
