"""생성 실행 단위로 attention 원본과 메타데이터를 보관한다."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import threading
import uuid

import numpy as np

from .compute import compute_attention
from .writer import ArrayWriter
from .diagnostics import read_counters


def select_indices(spec, count, *, grid=None):
    if spec == "all":
        return list(range(count))
    if spec == "last":
        return [count - 1]
    if spec == "center":
        if grid is None:
            return [count // 2]
        t, h, w = grid
        return [(t // 2 * h + h // 2) * w + w // 2]
    if isinstance(spec, str) and spec.startswith("sample:"):
        n = int(spec.split(":")[1])
        if n < 1:
            raise ValueError("sample 개수는 양수여야 합니다.")
        return sorted(set(round(i * (count - 1) / max(1, n - 1)) for i in range(n)))
    values = []
    for item in spec.split(",") if isinstance(spec, str) else spec:
        if isinstance(item, str) and "-" in item:
            start, stop = map(int, item.split("-"))
            if start > stop:
                raise ValueError("인덱스 범위는 작은 번호부터 입력하세요.")
            values.extend(range(start, stop + 1))
        else:
            values.append(int(item))
    if not values or min(values) < 0 or max(values) >= count:
        raise ValueError(f"선택 범위는 0~{count - 1}입니다: {spec}")
    return list(dict.fromkeys(values))


@dataclass(frozen=True)
class CaptureConfig:
    relations: tuple = ("image->text",)
    steps: str = "all"
    layers: str = "all"
    heads: str = "mean"
    branches: tuple = ("positive",)
    image_queries: str = "center"
    text_keys: str = "all"
    query_chunk: int = 128
    key_chunk: int = 256
    max_capture_bytes: int = 8 * 1024**3
    max_pending_bytes: int = 256 * 1024**2
    save_raw: bool = False
    memory_reserve_bytes: int = 256 * 1024**2

    def __post_init__(self):
        if not self.relations or set(self.relations) - {"image->text", "image->image"}:
            raise ValueError("Anima 본체는 image->text / image->image를 지원합니다.")
        if not self.branches or set(self.branches) - {"positive", "negative"}:
            raise ValueError("branch는 positive / negative입니다.")
        if min(self.query_chunk, self.key_chunk, self.max_capture_bytes, self.max_pending_bytes) <= 0:
            raise ValueError("chunk 크기와 수집 용량은 양수여야 합니다.")


class CaptureSession:
    def __init__(self, output_root, config=None, *, token_maps=None, metadata=None):
        self.config = config or CaptureConfig()
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.path = Path(output_root).resolve() / f"{stamp}_{uuid.uuid4().hex}"
        self.path.mkdir(parents=True, exist_ok=False)
        if self.config.save_raw:
            (self.path / "raw").mkdir()
        self.token_maps = token_maps or {}
        self.records = []
        self.capture_bytes = 0
        self.arrays = {}
        self.lock = threading.RLock()
        self.manifest = dict(schema_version=1, status="running", storage="disk" if self.config.save_raw else "memory", config=asdict(self.config),
                             token_maps=self.token_maps, metadata=metadata or {}, records=self.records)
        self._write_manifest()
        self.writer = ArrayWriter(self.config.max_pending_bytes) if self.config.save_raw else None

    def _write_manifest(self):
        temporary = self.path / "manifest.partial.json"
        temporary.write_text(json.dumps(self.manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.path / "manifest.json")

    def wants(self, relation, step, layer, total_steps, total_layers):
        return (relation in self.config.relations
                and step in select_indices(self.config.steps, total_steps)
                and layer in select_indices(self.config.layers, total_layers))

    def capture(self, query, key, *, relation, branch, step, call, layer, grid,
                total_steps, total_layers, sigma=None, mask=None, valid_grid=None):
        """Q/K의 입력 규약은 [B,H,Q,D]. grid는 실제 patch의 (T,H,W)."""
        if branch not in self.config.branches or not self.wants(relation, step, layer, total_steps, total_layers):
            return
        if math.prod(grid) != query.shape[2]:
            raise ValueError("query 길이와 이미지 patch 격자가 일치하지 않습니다.")
        qids = select_indices(self.config.image_queries, query.shape[2], grid=grid) if relation == "image->image" else list(range(query.shape[2]))
        kids = list(range(key.shape[2]))
        if relation == "image->image" and key.shape[2] != math.prod(grid):
            raise ValueError("self-attention key 격자가 query 격자와 다릅니다.")
        if relation == "image->text":
            kids = select_indices(self.config.text_keys, key.shape[2])
        heads = self.config.heads
        if heads not in ("all", "mean"):
            heads = select_indices(heads, query.shape[1])
        head_ids = ["mean"] if heads == "mean" else list(range(query.shape[1])) if heads == "all" else heads
        size = query.shape[0] * len(head_ids) * len(qids) * len(kids) * 4
        with self.lock:
            if self.manifest["status"] != "running":
                raise RuntimeError("종료된 세션에는 기록할 수 없습니다.")
            if self.capture_bytes + size > self.config.max_capture_bytes:
                raise MemoryError("수집 용량 한도를 초과합니다. steps/layers/heads/image_queries를 줄이세요.")
            if not self.config.save_raw:
                available = read_counters("/proc/meminfo").get("MemAvailable")
                if available is not None and available * 1024 < size + self.config.memory_reserve_bytes:
                    raise MemoryError("attention 보관용 RAM이 부족합니다. 고용량 RAM 런타임 또는 save_raw=True를 사용하세요.")
            values = compute_attention(query, key, query_indices=qids, key_indices=kids,
                heads=heads, query_chunk=self.config.query_chunk, key_chunk=self.config.key_chunk, mask=mask)
            filename = f"raw/{len(self.records):06d}.npy"
            if self.writer is not None:
                self.writer.submit(self.path / filename, values.numpy(), step=int(step), call=int(call),
                           layer=int(layer), relation=relation, branch=branch)
            else:
                self.arrays[filename] = values.numpy()
            self.capture_bytes += size
            self.records.append(dict(file=filename, relation=relation, branch=branch,
                step=int(step), call=int(call), layer=int(layer), sigma=sigma,
                grid=list(grid), valid_grid=list(valid_grid or grid), query_indices=qids,
                key_indices=kids, head_ids=head_ids, key_count=int(key.shape[2]),
                total_heads=int(query.shape[1]), shape=list(values.shape)))

    def finish(self, error=None):
        with self.lock:
            write_error = None
            try:
                if self.writer is not None:
                    self.writer.finish()
            except Exception as exc:
                write_error = exc
            failure = error or write_error
            self.manifest.update(status="failed" if failure else "complete", error=str(failure) if failure else None,
                                 capture_bytes=self.capture_bytes, peak_pending_bytes=self.writer.peak_pending if self.writer else 0)
            if failure:
                self.arrays.clear()
            self._write_manifest()
            if write_error is not None and error is None:
                raise write_error

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.finish(exc)
        return False
