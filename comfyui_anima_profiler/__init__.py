import json
from pathlib import Path
import uuid

import folder_paths
from anima_heatmap.profiling import CaptureProfile


def sampler_class():
    from nodes import NODE_CLASS_MAPPINGS
    sampler = NODE_CLASS_MAPPINGS.get("AnimaHeatmapSampler")
    if sampler is None:
        raise RuntimeError("Anima Heatmap 커스텀 노드를 먼저 설치하세요.")
    return sampler


class AnimaHeatmapProfileSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return sampler_class().INPUT_TYPES()

    RETURN_TYPES = ("LATENT", "ANIMA_HEATMAP_SESSION", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("latent", "session", "session_directory", "report", "profile_path")
    FUNCTION = "sample"
    CATEGORY = "Anima Profiler"

    def sample(self, **kwargs):
        profile = CaptureProfile()
        result, error = None, None
        try:
            with profile.activate(), profile.measure("sampling_total", kwargs["model"].load_device):
                result = sampler_class()().sample(**kwargs)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            data = profile.result()
            data.update(status="failed" if error else "complete", error=error)
            root = Path(result[2]) if result is not None else Path(folder_paths.get_output_directory()) / "anima_profiler" / uuid.uuid4().hex
            root.mkdir(parents=True, exist_ok=True)
            path = root / "profile.json"
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            labels = {"sampling_total": "샘플러 전체", "attention_compute": "attention 계산·준비",
                      "cpu_transfer": "CPU 전송", "file_save": "attention NPY 저장",
                      "sampling_other": "나머지 생성·실행"}
            report = "\n".join(f"{label}: {data['seconds'].get(key, 0):.3f}초" for key, label in labels.items())
            report += f"\n수집 호출: {data['counts'].get('attention_total', 0)}"
            report += f"\nCPU 전송 횟수: {data['counts'].get('cpu_transfer', 0)}"
            report += f"\n메모리 부족 분할 전환: {data['counts'].get('oom_fallback', 0)}"
            report += f"\nGPU 동기화를 포함한 측정입니다. 측정 자체의 추가 비용이 있습니다.\n{path}"
            print(f"[Anima Profiler]\n{report}")
        return (*result, report, str(path))


NODE_CLASS_MAPPINGS = {"AnimaHeatmapProfileSampler": AnimaHeatmapProfileSampler}
NODE_DISPLAY_NAME_MAPPINGS = {"AnimaHeatmapProfileSampler": "Anima Profiler · Heatmap KSampler"}
