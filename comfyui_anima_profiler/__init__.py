import json
from pathlib import Path
import uuid
import copy

import folder_paths
from anima_heatmap.profiling import CaptureProfile
from anima_heatmap.diagnostics import snapshot, storage_info, benchmark_storage, summarize_resources, build_report
from anima_heatmap.torch_trace import TorchTrace, device_snapshot


def sampler_class():
    from nodes import NODE_CLASS_MAPPINGS
    sampler = NODE_CLASS_MAPPINGS.get("AnimaHeatmapSampler")
    if sampler is None:
        raise RuntimeError("Anima Heatmap 커스텀 노드를 먼저 설치하세요.")
    return sampler


class AnimaHeatmapProfileSampler:
    @classmethod
    def INPUT_TYPES(cls):
        inputs = copy.deepcopy(sampler_class().INPUT_TYPES())
        inputs.setdefault("optional", {}).update({
            "trace_steps": ("INT", {"default": 2, "min": 1, "max": 100}),
            "storage_probe_mib": ("INT", {"default": 64, "min": 1, "max": 256}),
        })
        return inputs

    RETURN_TYPES = ("LATENT", "ANIMA_HEATMAP_SESSION", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("latent", "session", "session_directory", "report", "profile_path")
    FUNCTION = "sample"
    CATEGORY = "Anima Profiler"

    def sample(self, trace_steps=2, storage_probe_mib=64, **kwargs):
        profile = CaptureProfile()
        trace = TorchTrace(kwargs["model"].load_device, trace_steps)
        profile.step_callback = trace.step
        result, error = None, None
        diagnostic_errors = []
        profile.resources.append(snapshot())
        device_before = device_snapshot(kwargs["model"].load_device)
        trace.start()
        try:
            with profile.activate(), profile.measure("sampling_total", kwargs["model"].load_device):
                result = sampler_class()().sample(**kwargs)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            try:
                trace.stop()
            except Exception as exc:
                diagnostic_errors.append(f"PyTorch stop: {exc!r}")
            profile.resources.append(snapshot())
            data = profile.result()
            data["schema_version"] = 3
            data["device_before"] = device_before
            data["device_after"] = device_snapshot(kwargs["model"].load_device)
            data.update(profile.evidence())
            data.update(status="failed" if error else "complete", error=error)
            root = Path(result[2]) if result is not None else Path(folder_paths.get_output_directory()) / "anima_profiler" / uuid.uuid4().hex
            root.mkdir(parents=True, exist_ok=True)
            path = root / "profile.json"
            data["storage"] = storage_info(root)
            snapshots = [profile.resources[0]] + [r["resources_after"] for r in profile.files] + [profile.resources[-1]]
            data["resource_summary"] = summarize_resources(snapshots)
            data["diagnostic_errors"] = diagnostic_errors
            try:
                data["torch_trace"] = trace.export(root)
                if data["torch_trace"]["cuda_trace_status"] == "missing":
                    data["diagnostic_errors"].append("CUDA kernel trace가 비었습니다. GPU trace 수집 실패로 표시합니다.")
            except Exception as exc:
                data["diagnostic_errors"].append(f"PyTorch trace: {exc!r}")
            try:
                sample_path = profile.files[0]["file"] if profile.files else None
                data["storage_benchmark"] = benchmark_storage(root, storage_probe_mib * 2**20, sample_path=sample_path)
            except Exception as exc:
                data["diagnostic_errors"].append(f"Storage probe: {exc!r}")
            report = build_report(data) + f"\n{path}"
            data["diagnostic_status"] = "partial" if data["diagnostic_errors"] else "complete"
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            (root / "profile_report.txt").write_text(report, encoding="utf-8")
            print(f"[Anima Profiler]\n{report}")
        return (*result, report, str(path))


NODE_CLASS_MAPPINGS = {"AnimaHeatmapProfileSampler": AnimaHeatmapProfileSampler}
NODE_DISPLAY_NAME_MAPPINGS = {"AnimaHeatmapProfileSampler": "Anima Profiler · Heatmap KSampler"}
