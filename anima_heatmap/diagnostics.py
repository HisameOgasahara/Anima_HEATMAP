"""Storage and Linux memory evidence for attention profiling."""

import io
import os
from pathlib import Path
import platform
import shutil
import tempfile
from time import perf_counter

import numpy as np


def read_counters(path):
    try:
        result = {}
        for line in Path(path).read_text().splitlines():
            parts = line.replace(":", "").split()
            if len(parts) >= 2 and parts[1].isdigit():
                result[parts[0]] = int(parts[1])
        return result
    except OSError:
        return {}


def snapshot():
    memory = read_counters("/proc/meminfo")
    process = read_counters("/proc/self/status")
    vm = read_counters("/proc/vmstat")
    io_counters = read_counters("/proc/self/io")
    faults = {}
    try:
        fields = Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()
        faults = {"minor": int(fields[7]), "major": int(fields[9])}
    except OSError:
        pass
    pressure = {}
    for kind in ("memory", "io"):
        try:
            for line in Path(f"/proc/pressure/{kind}").read_text().splitlines():
                fields = line.split()
                pressure[f"{kind}_{fields[0]}_total_us"] = int(dict(x.split("=") for x in fields[1:])["total"])
        except OSError:
            pass
    return {"memory_kib": {k: memory[k] for k in ("MemAvailable", "Dirty", "Writeback", "SwapTotal", "SwapFree") if k in memory},
            "process_kib": {k: process[k] for k in ("VmRSS", "VmHWM", "VmSwap") if k in process},
            "vm_counters": {k: vm[k] for k in ("pswpin", "pswpout", "pgmajfault") if k in vm},
            "process_io": io_counters, "process_faults": faults, "system_pressure": pressure}


def storage_info(directory):
    path = Path(directory).resolve()
    usage = shutil.disk_usage(path)
    matches = []
    try:
        for line in Path("/proc/self/mountinfo").read_text().splitlines():
            left, right = line.split(" - ", 1)
            fields, details = left.split(), right.split()
            mount = fields[4].replace(r"\040", " ").replace(r"\134", "\\")
            if path.is_relative_to(mount):
                matches.append((len(mount), {"mount": mount, "filesystem": details[0], "source": details[1]}))
    except OSError:
        pass
    return {"resolved_path": str(path), "disk_total_bytes": usage.total, "disk_free_bytes": usage.free,
            "mount_info": max(matches, key=lambda x: x[0])[1] if matches else None,
            "platform": platform.platform()}


def distribution(values):
    if not values:
        return {"count": 0}
    return {"count": len(values), "sum": float(sum(values)), "min": float(min(values)),
            "p50": float(np.percentile(values, 50)), "p95": float(np.percentile(values, 95)),
            "max": float(max(values))}


def benchmark_storage(directory, payload_bytes, repeats=3, sample_path=None):
    # Same float32 payload for memory serialization, buffered file writes and fsync.
    if sample_path is not None:
        source = np.load(sample_path, mmap_mode="r", allow_pickle=False)
        array = source.reshape(-1)[:max(1, payload_bytes // source.dtype.itemsize)].copy()
        del source
    else:
        array = np.random.default_rng(0).random(max(1, payload_bytes // 4), dtype=np.float32)
    rows = []
    for _ in range(repeats):
        start = perf_counter()
        with io.BytesIO() as memory:
            np.save(memory, array, allow_pickle=False)
            serialized_bytes = memory.tell()
        memory_seconds = perf_counter() - start
        fd, name = tempfile.mkstemp(prefix="anima_profile_probe_", suffix=".npy", dir=directory)
        try:
            with os.fdopen(fd, "wb") as file:
                start = perf_counter()
                np.save(file, array, allow_pickle=False)
                file.flush()
                buffered_seconds = perf_counter() - start
                start = perf_counter()
                os.fsync(file.fileno())
                sync_seconds = perf_counter() - start
            rows.append({"bytes": serialized_bytes, "memory_seconds": memory_seconds,
                         "buffered_seconds": buffered_seconds, "fsync_seconds": sync_seconds,
                         "durable_mib_per_second": serialized_bytes / 2**20 / (buffered_seconds + sync_seconds)})
        finally:
            Path(name).unlink(missing_ok=True)
    return {"payload_bytes": array.nbytes, "repeats": rows, "source_file": str(sample_path) if sample_path else None,
            "note": "Post-sampling identical payload probe; sampled from saved attention when available, otherwise synthetic; buffered write and fsync reported separately."}


def summarize_resources(snapshots):
    memories = [s["memory_kib"] for s in snapshots if s.get("memory_kib")]
    processes = [s["process_kib"] for s in snapshots if s.get("process_kib")]
    result = {}
    for key in ("MemAvailable", "Dirty", "Writeback", "SwapFree"):
        values = [m[key] for m in memories if key in m]
        if values:
            result[key + "_kib"] = {"min": min(values), "max": max(values)}
    for key in ("VmRSS", "VmSwap"):
        values = [m[key] for m in processes if key in m]
        if values:
            result[key + "_peak_kib"] = max(values)
    if snapshots:
        for section in ("vm_counters", "process_io", "process_faults", "system_pressure"):
            first, last = snapshots[0].get(section, {}), snapshots[-1].get(section, {})
            result[section + "_delta"] = {k: last[k] - first[k] for k in first.keys() & last.keys()}
    return result


def build_report(data):
    seconds = data["seconds"]
    labels = {"attention_compute": "attention 준비·계산(CPU 전송 호출 제외)", "cpu_transfer": "CPU 전송 호출·선행 GPU 대기",
              "storage_blocking": "저장으로 생성이 대기한 시간", "sampling_other": "나머지 생성·실행"}
    ranked = sorted(labels, key=lambda k: seconds.get(k, 0), reverse=True)
    total = seconds.get("sampling_total", 0)
    overhead = max((k for k in labels if k != "sampling_other"), key=lambda k: seconds.get(k, 0))
    lines = [f"히트맵 추가 작업의 최대 시간 구간: {labels[overhead]}", f"샘플러 전체: {total:.3f}초"]
    for key in ranked:
        value = seconds.get(key, 0)
        lines.append(f"{labels[key]}: {value:.3f}초 ({value / total * 100 if total else 0:.1f}%)")
    storage = data["storage_summary"]
    lines.append(f"백그라운드 저장 작업: {seconds.get('file_save', 0):.3f}초 (생성과 중첩; 전체에 더하지 않음)")
    lines.append("생성 대기 세부: " + " / ".join(f"{k}={seconds.get(k, 0):.3f}초" for k in ("write_queue_wait", "write_drain", "write_oversize_sync")))
    lines.append(f"저장량: {storage['total_bytes'] / 2**30:.3f} GiB / {len(data['files'])}파일 / {storage['effective_mib_per_second']:.2f} MiB/s")
    phases = storage["phase_seconds"]
    lines.append("저장 세부: " + " / ".join(f"{k}={v:.3f}초" for k, v in phases.items()))
    cpu = storage["thread_cpu_seconds"]
    lines.append(f"저장 스레드 CPU 실행: {cpu:.3f}초 / CPU 비실행 경과: {max(0, seconds.get('file_save', 0) - cpu):.3f}초 (I/O 대기·스케줄링 등)")
    if data["files"]:
        dominant = max(phases, key=phases.get)
        lines.append(f"저장 내부 최대 구간: {dominant}")
        d = storage["file_seconds"]
        lines.append(f"파일당 p50={d['p50']:.4f}초 / p95={d['p95']:.4f}초 / 최대={d['max']:.4f}초")
    mount = data["storage"]["mount_info"]
    lines.append(f"저장 위치: {data['storage']['resolved_path']} / 마운트: {mount}")
    bench = data.get("storage_benchmark", {})
    if bench.get("repeats"):
        rows = bench["repeats"]
        lines.append(f"동일 데이터 저장 비교 ({bench['payload_bytes'] / 2**20:.1f} MiB, {len(rows)}회 중앙값):")
        for key in ("memory_seconds", "buffered_seconds", "fsync_seconds", "durable_mib_per_second"):
            lines.append(f"  {key}: {float(np.median([r[key] for r in rows])):.4f}")
        lines.append("memory=RAM 직렬화 / buffered=파일 쓰기·flush / fsync=추가 영구 저장 대기")
        memory = float(np.median([r["memory_seconds"] for r in rows]))
        buffered = float(np.median([r["buffered_seconds"] for r in rows]))
        lines.append(f"파일 쓰기 / RAM 직렬화 시간비: {buffered / max(memory, 1e-12):.2f}배")
        lines.append(f"저장 장치 여유: {data['storage']['disk_free_bytes'] / 2**30:.2f} GiB")
    lines.append("메모리·I/O 근거: " + str(data["resource_summary"]))
    trace = data.get("torch_trace", {})
    lines.append(f"PyTorch trace: {trace.get('cuda_trace_status', 'failed')} / 스텝 {trace.get('completed_sampler_steps', [])}")
    for kind in ("top_cpu", "top_device"):
        key = "self_cpu_ms" if kind == "top_cpu" else "self_device_ms"
        top = [r for r in trace.get(kind, []) if r[key] > 0][:5]
        lines.append(kind + ": " + "; ".join(f"{r['name']}={r[key]:.2f}ms" for r in top))
    lines.append(f"CPU 전송 {data['counts'].get('cpu_transfer', 0)}회 / OOM 분할 전환 {data['counts'].get('oom_fallback', 0)}회")
    actions = {"storage_blocking": "우선 대상: 저장 대기. queue wait와 종료 drain, 저장 처리량을 비교하세요.",
               "attention_compute": "우선 대상: attention 계산. torch_operators.txt와 trace의 anima::attention_total 내부 연산을 확인하세요.",
               "cpu_transfer": "우선 대상: GPU→CPU 복사. trace의 anima::cpu_transfer와 복사 커널·동기화 대기를 확인하세요.",
               "sampling_other": "우선 대상: 나머지 모델 생성·실행. trace의 모델 연산과 메모리 복사를 확인하세요."}
    lines.append(actions[overhead])
    lines.append("측정에는 동기화·trace 비용이 포함됩니다. trace의 연산 통계는 표시된 스텝 범위입니다.")
    for error in data.get("diagnostic_errors", []):
        lines.append("진단 수집 실패: " + error)
    return "\n".join(lines)
