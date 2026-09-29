"""Bounded PyTorch CPU/CUDA trace for the first sampler steps."""

from time import perf_counter
import torch


def device_snapshot(device):
    device = torch.device(device)
    result = {"device": str(device), "torch": torch.__version__, "cuda_runtime": torch.version.cuda}
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info(device)
        result.update(name=torch.cuda.get_device_name(device), free_bytes=free, total_bytes=total,
                      allocated_bytes=torch.cuda.memory_allocated(device), reserved_bytes=torch.cuda.memory_reserved(device))
    return result


class TorchTrace:
    def __init__(self, device, steps):
        self.device = torch.device(device)
        self.steps = steps
        self.completed_steps = []
        self.overhead_seconds = 0.0
        self.stopped = False
        activities = [torch.profiler.ProfilerActivity.CPU]
        if self.device.type == "cuda":
            if torch.profiler.ProfilerActivity.CUDA not in torch.profiler.supported_activities():
                raise RuntimeError("CUDA profiler를 사용할 수 없습니다. PyTorch/CUPTI 설치를 확인하세요.")
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        self.profiler = torch.profiler.profile(activities=activities, profile_memory=True,
                                              record_shapes=False, with_stack=False)

    def start(self):
        start = perf_counter()
        self.profiler.start()
        self.overhead_seconds += perf_counter() - start

    def step(self, step):
        if self.stopped:
            return
        self.completed_steps.append(int(step))
        if len(self.completed_steps) >= self.steps:
            self.stop()

    def stop(self):
        if not self.stopped:
            start = perf_counter()
            self.profiler.stop()
            self.overhead_seconds += perf_counter() - start
            self.stopped = True

    def export(self, directory):
        self.stop()
        path = directory / "torch_trace.json"
        self.profiler.export_chrome_trace(str(path))
        rows = []
        for event in self.profiler.key_averages():
            rows.append({"name": event.key, "calls": event.count,
                         "self_cpu_ms": event.self_cpu_time_total / 1000,
                         "self_device_ms": event.self_device_time_total / 1000,
                         "self_cpu_memory_bytes": event.self_cpu_memory_usage,
                         "self_device_memory_bytes": event.self_device_memory_usage})
        events = self.profiler.events()
        device_events = sum(1 for e in events if str(e.device_type).endswith("CUDA"))
        tables = "CPU self time (ms)\n" + self.profiler.key_averages().table(sort_by="self_cpu_time_total", row_limit=30)
        tables += "\nDevice self time (ms)\n" + self.profiler.key_averages().table(sort_by="self_device_time_total", row_limit=30)
        (directory / "torch_operators.txt").write_text(tables, encoding="utf-8")
        return {"path": str(path), "completed_sampler_steps": self.completed_steps,
                "requested_steps": self.steps, "device": str(self.device),
                "device_events": device_events, "start_stop_overhead_seconds": self.overhead_seconds,
                "top_cpu": sorted(rows, key=lambda r: r["self_cpu_ms"], reverse=True)[:30],
                "top_device": sorted(rows, key=lambda r: r["self_device_ms"], reverse=True)[:30],
                "cuda_trace_status": "captured" if device_events else "missing" if self.device.type == "cuda" else "cpu_only",
                "scope": "First sampler callbacks including startup; operator statistics apply only to this trace window."}
