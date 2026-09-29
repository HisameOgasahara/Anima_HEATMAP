import numpy as np
import pytest
import torch

from anima_heatmap import CaptureConfig, CaptureSession, load_maps
from anima_heatmap import capture


@pytest.mark.parametrize("view", ["aggregate", "per_step", "per_layer", "per_record"])
@pytest.mark.parametrize("aggregation", ["mean", "daam"])
def test_memory_and_disk_maps_match(tmp_path, view, aggregation):
    tokens = {"positive": {"ids": [1, 2, 3], "pieces": ["▁cat", "▁dog", "</s>"], "special_indices": [2]}}
    q, k = torch.randn(1, 2, 4, 3), torch.randn(1, 2, 3, 3)
    sessions = []
    for save_raw in (False, True):
        with CaptureSession(tmp_path, CaptureConfig(save_raw=save_raw, heads="all", keep_records=True), token_maps=tokens) as session:
            for step in range(2):
                for layer in range(2):
                    session.capture(q, k, relation="image->text", branch="positive", step=step,
                                    call=step, layer=layer, grid=(1, 2, 2), total_steps=2, total_layers=2)
        sessions.append(session)
    memory, disk = sessions
    assert not (memory.path / "raw").exists()
    assert len(memory.arrays) == 4
    assert disk.arrays == {}
    a = load_maps(memory, phrase="cat", view=view, aggregation=aggregation)
    b = load_maps(disk.path, phrase="cat", view=view, aggregation=aggregation)
    assert len(a) == len(b)
    for first, second in zip(a, b):
        np.testing.assert_array_equal(first["raw"], second["raw"])
    with pytest.raises(ValueError, match="session"):
        load_maps(memory.path, phrase="cat")


def test_memory_capture_never_calls_npy_save(tmp_path, monkeypatch):
    monkeypatch.setattr(np, "save", lambda *a, **k: pytest.fail("memory capture must not save NPY"))
    with CaptureSession(tmp_path) as session:
        session.capture(torch.zeros(1, 1, 2, 3), torch.zeros(1, 1, 3, 3),
                        relation="image->text", branch="positive", step=0, call=0,
                        layer=0, grid=(1, 1, 2), total_steps=1, total_layers=1)
    assert session.capture_bytes == 24
    assert session.writer is None


def test_ram_guard_stops_before_computation(tmp_path, monkeypatch):
    monkeypatch.setattr(capture, "read_counters", lambda path: {"MemAvailable": 1})
    monkeypatch.setattr(capture, "compute_attention", lambda *a, **k: pytest.fail("must check RAM first"))
    with pytest.raises(MemoryError, match="RAM"):
        with CaptureSession(tmp_path) as session:
            session.capture(torch.zeros(1, 1, 2, 3), torch.zeros(1, 1, 3, 3),
                            relation="image->text", branch="positive", step=0, call=0,
                            layer=0, grid=(1, 1, 2), total_steps=1, total_layers=1)
    assert session.manifest["status"] == "failed"
    assert session.arrays == {}
