"""Opt-in timings for a single sampling execution."""

from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from functools import wraps
from time import perf_counter, thread_time

import torch
import numpy as np
from .diagnostics import snapshot, distribution


_active = ContextVar("anima_heatmap_profile", default=None)


class CaptureProfile:
    def __init__(self):
        self.seconds = {}
        self.counts = {}
        self.files = []
        self.attention_records = []
        self.resources = []
        self.step_callback = None
        self.monitoring_seconds = 0.0

    @contextmanager
    def activate(self):
        token = _active.set(self)
        try:
            yield self
        finally:
            _active.reset(token)

    @contextmanager
    def measure(self, name, device=None):
        device = torch.device(device) if device is not None else None
        synchronize = device is not None and device.type == "cuda" and name != "cpu_transfer"
        if synchronize:
            torch.cuda.synchronize(device)
        started = perf_counter()
        try:
            with torch.profiler.record_function(f"anima::{name}") if name != "sampling_total" else nullcontext():
                yield
        finally:
            if synchronize:
                torch.cuda.synchronize(device)
            self.seconds[name] = self.seconds.get(name, 0.0) + perf_counter() - started
            self.counts[name] = self.counts.get(name, 0) + 1

    def result(self):
        seconds = dict(self.seconds)
        seconds["attention_compute"] = max(0.0, seconds.get("attention_total", 0.0)
                                            - seconds.get("cpu_transfer", 0.0))
        seconds["storage_blocking"] = sum(seconds.get(k, 0.0) for k in
                                          ("write_queue_wait", "write_drain", "write_oversize_sync"))
        seconds["sampling_other"] = max(0.0, seconds.get("sampling_total", 0.0)
                                         - seconds.get("attention_total", 0.0)
                                         - seconds["storage_blocking"])
        return {"seconds": seconds, "counts": dict(self.counts),
                "timing": "attention boundaries synchronize; cpu_transfer includes preceding GPU work waited by blocking cpu(); file_save overlaps sampling and is not additive; storage_blocking is producer queue/drain wait"}

    def save_array(self, path, array, metadata):
        monitoring_start = perf_counter()
        before = snapshot()
        self.monitoring_seconds += perf_counter() - monitoring_start
        started = perf_counter()
        file = None
        timings = {}
        cpu_start = thread_time()
        try:
            with torch.profiler.record_function("anima::file_open"):
                start = perf_counter()
                file = open(path, "wb")
                timings["open"] = perf_counter() - start
            with torch.profiler.record_function("anima::numpy_write"):
                start = perf_counter()
                np.save(file, array, allow_pickle=False)
                timings["numpy_write"] = perf_counter() - start
        finally:
            if file is not None:
                with torch.profiler.record_function("anima::file_close"):
                    start = perf_counter()
                    file.close()
                    timings["close"] = perf_counter() - start
            elapsed = perf_counter() - started
            cpu_elapsed = thread_time() - cpu_start
            self.seconds["file_save"] = self.seconds.get("file_save", 0) + elapsed
            self.counts["file_save"] = self.counts.get("file_save", 0) + 1
            size = path.stat().st_size if path.exists() else 0
            monitoring_start = perf_counter()
            after = snapshot()
            self.monitoring_seconds += perf_counter() - monitoring_start
            self.files.append({"file": str(path), "payload_bytes": array.nbytes, "file_bytes": size,
                               "record": metadata,
                               "shape": list(array.shape), "dtype": str(array.dtype), "seconds": elapsed,
                               "thread_cpu_seconds": cpu_elapsed,
                               "phases": timings, "mib_per_second": size / 2**20 / elapsed,
                               "resources_before": before, "resources_after": after})

    def evidence(self):
        total_bytes = sum(row["file_bytes"] for row in self.files)
        return {"files": self.files, "attention_records": self.attention_records,
                "monitoring_seconds": self.monitoring_seconds,
                "storage_summary": {"total_bytes": total_bytes,
                    "effective_mib_per_second": total_bytes / 2**20 / max(self.seconds.get("file_save", 0), 1e-12),
                    "thread_cpu_seconds": sum(r["thread_cpu_seconds"] for r in self.files),
                    "file_seconds": distribution([r["seconds"] for r in self.files]),
                    "phase_seconds": {phase: sum(r["phases"].get(phase, 0) for r in self.files)
                                      for phase in ("open", "numpy_write", "close")}},
                "resources": self.resources}


def measure(name, device=None):
    profile = _active.get()
    return profile.measure(name, device) if profile is not None else nullcontext()


def record_count(name):
    profile = _active.get()
    if profile is not None:
        profile.counts[name] = profile.counts.get(name, 0) + 1


def profile_attention(function):
    @wraps(function)
    def wrapped(query, *args, **kwargs):
        if _active.get() is None:
            return function(query, *args, **kwargs)
        profile = _active.get()
        before = profile.seconds.get("attention_total", 0)
        with measure("attention_total", query.device):
            result = function(query, *args, **kwargs)
        profile.attention_records.append({"query_shape": list(query.shape), "dtype": str(query.dtype),
            "device": str(query.device), "seconds": profile.seconds["attention_total"] - before})
        return result
    return wrapped


def transfer_to_cpu(tensor):
    with measure("cpu_transfer", tensor.device):
        return tensor.cpu()


def save_array(path, array, **metadata):
    profile = _active.get()
    if profile is None:
        np.save(path, array, allow_pickle=False)
    else:
        profile.save_array(path, array, metadata)


def sampling_step(step):
    profile = _active.get()
    if profile is not None and profile.step_callback is not None:
        profile.step_callback(step)
