import json
from threading import Event, Thread

import numpy as np
import pytest
import torch

from anima_heatmap import CaptureSession
from anima_heatmap import writer
from anima_heatmap.profiling import CaptureProfile, measure


def test_queue_overlaps_and_bounds_inflight_arrays(tmp_path, monkeypatch):
    entered, release, submitted = Event(), Event(), Event()
    original = writer.save_array
    def blocked(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        original(*args, **kwargs)
    monkeypatch.setattr(writer, "save_array", blocked)
    array = np.arange(16, dtype=np.float32)
    worker = writer.ArrayWriter(array.nbytes)
    worker.submit(tmp_path / "a.npy", array)
    assert entered.wait(5)
    def submit_second():
        worker.submit(tmp_path / "b.npy", array + 1)
        submitted.set()
    producer = Thread(target=submit_second)
    producer.start()
    try:
        assert not submitted.wait(0.05)
        assert worker.pending == array.nbytes
    finally:
        release.set()
        producer.join(5)
        worker.finish()
    assert submitted.is_set()
    assert worker.peak_pending == array.nbytes
    assert worker.pending == 0
    np.testing.assert_array_equal(np.load(tmp_path / "a.npy"), array)
    np.testing.assert_array_equal(np.load(tmp_path / "b.npy"), array + 1)


def test_completion_waits_for_worker(tmp_path, monkeypatch):
    entered, release, finished = Event(), Event(), Event()
    original = writer.save_array
    def blocked(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        original(*args, **kwargs)
    monkeypatch.setattr(writer, "save_array", blocked)
    session = CaptureSession(tmp_path)
    session.capture(torch.zeros(1, 1, 2, 4), torch.zeros(1, 1, 3, 4),
                    relation="image->text", branch="positive", step=0, call=0,
                    layer=0, grid=(1, 1, 2), total_steps=1, total_layers=1)
    assert entered.wait(5)
    def finish():
        session.finish()
        finished.set()
    thread = Thread(target=finish)
    thread.start()
    try:
        assert not finished.wait(0.05)
        assert json.loads((session.path / "manifest.json").read_text(encoding="utf-8"))["status"] == "running"
    finally:
        release.set()
        thread.join(5)
    assert finished.is_set()
    assert json.loads((session.path / "manifest.json").read_text(encoding="utf-8"))["status"] == "complete"


def test_background_error_marks_session_failed(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("disk full")
    monkeypatch.setattr(writer, "save_array", fail)
    with pytest.raises(RuntimeError, match="파일 저장") as raised:
        with CaptureSession(tmp_path) as session:
            session.capture(torch.zeros(1, 1, 2, 4), torch.zeros(1, 1, 3, 4),
                            relation="image->text", branch="positive", step=0, call=0,
                            layer=0, grid=(1, 1, 2), total_steps=1, total_layers=1)
    assert isinstance(raised.value.__cause__, OSError)
    manifest = json.loads((session.path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert "disk full" in manifest["error"]
    assert session.writer.pending == 0


def test_oversize_is_saved_without_growing_queue(tmp_path):
    worker = writer.ArrayWriter(1)
    profile = CaptureProfile()
    with profile.activate():
        worker.submit(tmp_path / "large.npy", np.ones(16, dtype=np.float32))
        worker.finish()
    assert worker.peak_pending == 0
    assert (tmp_path / "large.npy").is_file()
    assert profile.counts["oversize_write"] == 1
    assert profile.result()["seconds"]["storage_blocking"] >= profile.seconds["write_oversize_sync"]


def test_transfer_does_not_add_device_synchronizations(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: calls.append(device))
    profile = CaptureProfile()
    with profile.activate():
        with measure("cpu_transfer", "cuda:0"):
            pass
        assert not calls
        with measure("attention_total", "cuda:0"):
            pass
    assert len(calls) == 2


def test_background_save_is_not_subtracted_from_sampler_wall_time():
    profile = CaptureProfile()
    profile.seconds.update(sampling_total=10, attention_total=3, cpu_transfer=1,
                           file_save=9, write_queue_wait=0.5, write_drain=1)
    result = profile.result()["seconds"]
    assert result["storage_blocking"] == 1.5
    assert result["sampling_other"] == 5.5
