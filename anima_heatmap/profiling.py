"""Opt-in timings for a single sampling execution."""

from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from functools import wraps
from time import perf_counter

import torch


_active = ContextVar("anima_heatmap_profile", default=None)


class CaptureProfile:
    def __init__(self):
        self.seconds = {}
        self.counts = {}

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
        synchronize = device is not None and device.type == "cuda"
        if synchronize:
            torch.cuda.synchronize(device)
        started = perf_counter()
        try:
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
        seconds["sampling_other"] = max(0.0, seconds.get("sampling_total", 0.0)
                                         - seconds.get("attention_total", 0.0)
                                         - seconds.get("file_save", 0.0))
        return {"seconds": seconds, "counts": dict(self.counts),
                "timing": "synchronized wall clock; attention_total includes cpu_transfer; profiling adds overhead"}


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
        with measure("attention_total", query.device):
            return function(query, *args, **kwargs)
    return wrapped


def transfer_to_cpu(tensor):
    with measure("cpu_transfer", tensor.device):
        return tensor.cpu()
