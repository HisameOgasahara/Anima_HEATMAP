import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
import numpy as np

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
    evidence = profile.evidence()
    saved = evidence["files"][0]
    torch.testing.assert_close(torch.from_numpy(np.load(saved["file"])), expected)
    assert saved["payload_bytes"] == expected.numel() * expected.element_size()
    assert saved["file_bytes"] == Path(saved["file"]).stat().st_size
    assert saved["record"]["layer"] == 0
    assert set(saved["phases"]) == {"open", "numpy_write", "close"}
    assert evidence["storage_summary"]["total_bytes"] == saved["file_bytes"]


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
            from anima_heatmap.profiling import sampling_step
            for i in range(3):
                torch.randn(4, 4).softmax(-1)
                sampling_step(i)
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
    assert node.INPUT_TYPES()["required"] == Sampler.INPUT_TYPES()["required"]
    if fail:
        with pytest.raises(ValueError, match="sampling failed"):
            node.sample(model=SimpleNamespace(load_device="cpu"), storage_probe_mib=1)
    else:
        result = node.sample(model=SimpleNamespace(load_device="cpu"), storage_probe_mib=1)
        assert result[:3] == ({"samples": "unchanged"}, str(tmp_path), str(tmp_path))
        assert Path(result[4]).is_file()
    data = json.loads(next(tmp_path.rglob("profile.json")).read_text(encoding="utf-8"))
    assert data["status"] == ("failed" if fail else "complete")
    assert data["torch_trace"]["completed_sampler_steps"] == [0, 1]
    assert data["torch_trace"]["top_cpu"]
    assert Path(data["torch_trace"]["path"]).is_file()
    assert not data["diagnostic_errors"]
    assert len(data["storage_benchmark"]["repeats"]) == 3
    assert not list(tmp_path.rglob("anima_profile_probe_*"))
    trace = json.loads(Path(data["torch_trace"]["path"]).read_text(encoding="utf-8"))
    assert trace["traceEvents"]
    assert data["diagnostic_status"] == "complete"


def test_slow_write_is_attributed_to_write_phase(tmp_path, monkeypatch):
    import time
    from anima_heatmap.profiling import save_array
    original = np.save
    def slow_save(*args, **kwargs):
        time.sleep(0.025)
        return original(*args, **kwargs)
    monkeypatch.setattr(np, "save", slow_save)
    profile = CaptureProfile()
    with profile.activate():
        save_array(tmp_path / "map.npy", np.zeros((4, 4), dtype=np.float32))
    row = profile.files[0]
    assert row["phases"]["numpy_write"] >= 0.025
    assert row["seconds"] >= row["phases"]["numpy_write"]
    assert max(row["phases"], key=row["phases"].get) == "numpy_write"


def test_storage_probe_separates_sync_and_removes_files(tmp_path, monkeypatch):
    import os
    import time
    from anima_heatmap.diagnostics import benchmark_storage
    original = os.fsync
    def slow_sync(fd):
        time.sleep(0.02)
        original(fd)
    monkeypatch.setattr(os, "fsync", slow_sync)
    result = benchmark_storage(tmp_path, 4096, repeats=2)
    assert all(row["fsync_seconds"] >= 0.02 for row in result["repeats"])
    assert all(row["bytes"] > 4096 for row in result["repeats"])
    assert not list(tmp_path.iterdir())


def test_storage_probe_uses_real_attention_and_preserves_it(tmp_path):
    from anima_heatmap.diagnostics import benchmark_storage
    path = tmp_path / "raw.npy"
    array = np.arange(2048, dtype=np.float32).reshape(32, 64)
    np.save(path, array)
    before = path.read_bytes()
    result = benchmark_storage(tmp_path, 4096, repeats=1, sample_path=path)
    assert result["payload_bytes"] == 4096
    assert result["source_file"] == str(path)
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


def test_linux_resource_evidence(monkeypatch):
    from anima_heatmap.diagnostics import snapshot, summarize_resources
    files = {"/proc/meminfo": "MemAvailable: 12345 kB\nDirty: 100 kB\nWriteback: 20 kB",
             "/proc/self/status": "VmRSS: 8000 kB\nVmSwap: 0 kB",
             "/proc/vmstat": "pswpin 2\npswpout 3\npgmajfault 10",
             "/proc/self/io": "write_bytes: 123456\nwchar: 654321",
             "/proc/self/stat": "1 (python worker) " + " ".join(["0"] * 30),
             "/proc/pressure/io": "some avg10=0.00 avg60=0.00 avg300=0.00 total=123",
             "/proc/pressure/memory": "some avg10=0.00 avg60=0.00 avg300=0.00 total=456"}
    def read(path, *args, **kwargs):
        return files[path.as_posix()]
    monkeypatch.setattr(Path, "read_text", read)
    first = snapshot()
    files["/proc/self/io"] = "write_bytes: 123556\nwchar: 654521"
    last = snapshot()
    result = summarize_resources([first, last])
    assert result["MemAvailable_kib"]["min"] == 12345
    assert result["process_io_delta"]["write_bytes"] == 100
    assert first["system_pressure"]["io_some_total_us"] == 123
