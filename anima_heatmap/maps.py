"""저장된 확률에서 단어·공간 지도를 만들고 PNG/NPY/GIF로 내보낸다."""

import json
from pathlib import Path
import re
import uuid

import matplotlib
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from .capture import CaptureSession, select_indices


def find_token_spans(token_map, phrase, occurrence="all"):
    """토큰 경계를 보존한다. 같은 단어의 반복 출현을 모두/번호로 선택한다."""
    pieces = token_map["pieces"]
    specials = set(token_map.get("special_indices", []))
    text, ranges = "", []
    for index, piece in enumerate(pieces):
        if index in specials:
            ranges.append((len(text), len(text)))
            continue
        decoded = piece.replace("▁", " ").replace("Ġ", " ").casefold()
        start = len(text)
        text += decoded
        ranges.append((start, len(text)))
    needle = phrase.strip().casefold()
    if not needle:
        raise ValueError("찾을 단어를 입력하세요.")
    pattern = r"(?<!\w)" + r"\s+".join(re.escape(x) for x in needle.split()) + r"(?!\w)"
    spans = []
    for match in re.finditer(pattern, text):
        ids = [i for i, (start, stop) in enumerate(ranges)
               if stop > match.start() and start < match.end() and i not in specials]
        if ids:
            spans.append(ids)
    if not spans:
        raise ValueError(f"토큰에서 {phrase!r}을 찾지 못했습니다. token_indices로 직접 선택할 수 있습니다.")
    if occurrence != "all":
        index = int(occurrence)
        if index < 0 or index >= len(spans):
            raise ValueError(f"출현 번호는 0~{len(spans) - 1}입니다.")
        spans = [spans[index]]
    return list(dict.fromkeys(i for span in spans for i in span))


def _selected(records, field, spec):
    if spec == "all":
        return records
    values = sorted(set(r[field] for r in records))
    chosen = [values[-1]] if spec == "last" else select_indices(spec, max(values) + 1)
    return [r for r in records if r[field] in chosen]


def load_maps(session, *, relation="image->text", branch="positive", batch=0,
              phrase="", token_indices=None, occurrence="all", query_index=None,
              steps="all", layers="all", heads="all", calls="all", view="aggregate",
              aggregation="mean"):
    """선택된 기록을 반환한다. aggregate/per_step/per_layer/per_record를 지원한다.

    mean은 선택 기록 평균, daam은 head 합→call 내 layer 평균→call 합이다.
    저장된 head 평균에서 개별 head를 복원할 수는 없다.
    """
    memory = isinstance(session, CaptureSession) and not session.config.save_raw
    root = session.path if isinstance(session, CaptureSession) else Path(session).resolve()
    manifest = session.manifest if isinstance(session, CaptureSession) else json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete":
        raise ValueError(f"완료되지 않은 세션입니다: {manifest['status']} / {manifest.get('error')}")
    if manifest.get("storage") == "memory" and not memory:
        raise ValueError("메모리 세션은 Sampler의 session 출력을 직접 연결하세요. 폴더 재분석에는 save_raw=True가 필요합니다.")
    records = [r for r in manifest["records"] if r["relation"] == relation and r["branch"] == branch]
    if not records:
        raise ValueError("선택한 관계/분기의 기록이 없습니다. CFG=1이면 부정 분기가 실행되지 않을 수 있습니다.")
    for field, spec in (("step", steps), ("layer", layers), ("call", calls)):
        records = _selected(records, field, spec)
        if not records:
            raise ValueError(f"{field} 선택에 해당하는 기록이 없습니다.")
    if view not in ("aggregate", "per_step", "per_layer", "per_record"):
        raise ValueError("지원하지 않는 view입니다.")
    if aggregation not in ("mean", "daam"):
        raise ValueError("aggregation은 mean 또는 daam입니다.")
    ids = token_indices
    if relation == "image->text" and ids is None:
        ids = find_token_spans(manifest["token_maps"][branch], phrase, occurrence)
    groups = {}
    for r in records:
        path = (root / r["file"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("세션 밖의 원본 파일은 읽을 수 없습니다.")
        data = session.arrays[r["file"]] if memory else np.load(path, mmap_mode="r", allow_pickle=False)
        if not 0 <= batch < data.shape[0]:
            raise ValueError("batch 번호가 기록 범위를 벗어났습니다.")
        hids = r["head_ids"]
        selected_heads = list(range(len(hids)))
        if heads not in ("all", "mean"):
            if hids == ["mean"]:
                raise ValueError("head 평균으로 저장된 세션입니다. 수집할 때 heads=all을 사용하세요.")
            requested = select_indices(heads, max(hids) + 1)
            if any(i not in hids for i in requested):
                raise ValueError("선택한 head가 저장되지 않았습니다.")
            selected_heads = [hids.index(i) for i in requested]
        a = np.asarray(data[batch, selected_heads], dtype=np.float32)
        a = a.sum(axis=0) if aggregation == "daam" else a.mean(axis=0)
        if aggregation == "daam" and hids == ["mean"]:
            a = a * r["total_heads"]
        if relation == "image->text":
            if not ids or any(i not in r["key_indices"] for i in ids):
                raise ValueError("선택한 텍스트 토큰이 모두 저장되어 있지 않습니다.")
            a = a[:, [r["key_indices"].index(i) for i in ids]].mean(axis=-1)
            label = phrase or f"tokens {ids}"
        else:
            chosen = r["query_indices"][0] if query_index is None else int(query_index)
            if chosen not in r["query_indices"]:
                raise ValueError(f"query {chosen}가 저장되지 않았습니다: {r['query_indices']}")
            a = a[r["query_indices"].index(chosen)]
            label = f"query {chosen}"
        grid = tuple(r["grid"])
        if grid[0] != 1:
            raise ValueError("이 Anima 이미지 시각화는 시간축 크기 1을 요구합니다.")
        a = a.reshape(grid)[0]
        key = "aggregate" if view == "aggregate" else f"step {r['step']}" if view == "per_step" else f"layer {r['layer']}" if view == "per_layer" else f"call {r['call']} layer {r['layer']}"
        groups.setdefault(key, []).append((r, a))
    output = []
    for group, items in groups.items():
        if len({(tuple(r["grid"]), tuple(r["valid_grid"])) for r, _ in items}) != 1:
            raise ValueError("다른 공간 격자의 기록은 한 지도에 합칠 수 없습니다.")
        if aggregation == "daam":
            by_call = {}
            for r, a in items:
                by_call.setdefault(r["call"], []).append(a)
            raw = sum(np.mean(values, axis=0) for values in by_call.values())
        else:
            raw = np.mean([a for _, a in items], axis=0)
        output.append(dict(label=f"{branch} {relation} {label} {group}", raw=raw,
                           grid=items[0][0]["grid"], valid_grid=items[0][0]["valid_grid"],
                           record_count=len(items), group=group))
    return output


def render_map(item, *, image=None, size=None, alpha=0.5, colormap="turbo",
               normalization="relative", value_range=None):
    if not 0 <= alpha <= 1:
        raise ValueError("alpha는 0~1입니다.")
    base = None
    if image is not None:
        base = np.asarray(image, dtype=np.float32)
        if base.ndim != 3 or base.shape[-1] not in (3, 4):
            raise ValueError("이미지는 H×W×3/4, float 0~1 형식입니다.")
        base = base[..., :3]
        size = base.shape[:2]
    size = size or tuple(item["raw"].shape)
    _, gh, gw = item["grid"]
    _, vh, vw = item["valid_grid"]
    target_h, target_w = size
    # 패딩 patch도 모델의 softmax에는 포함한다. 표시할 때만 원본 범위를 자른다.
    padded = (max(target_h, round(target_h * gh / vh)), max(target_w, round(target_w * gw / vw)))
    tensor = torch.from_numpy(np.array(item["raw"], dtype=np.float32))[None, None]
    raw = F.interpolate(tensor, size=padded, mode="bilinear", align_corners=False)[0, 0, :target_h, :target_w].numpy()
    if normalization == "relative":
        lo, hi = float(raw.min()), float(raw.max())
    elif normalization == "shared" and value_range is not None:
        lo, hi = value_range
    elif normalization == "absolute":
        lo, hi = 0.0, 1.0
    else:
        raise ValueError("normalization은 relative/shared/absolute입니다.")
    norm = np.clip((raw - lo) / max(hi - lo, 1e-12), 0, 1)
    heat = matplotlib.colormaps[colormap](norm)[..., :3].astype(np.float32)
    overlay = heat if base is None else (1 - alpha) * np.clip(base, 0, 1) + alpha * heat
    return dict(raw=raw, heatmap=heat, overlay=overlay, value_range=[float(lo), float(hi)])


def save_views(items, output_root, *, image=None, size=None, alpha=0.5,
               colormap="turbo", normalization="relative", fps=4):
    if not items or fps <= 0:
        raise ValueError("저장할 지도와 양수 fps가 필요합니다.")
    destination = Path(output_root) / f"view_{uuid.uuid4().hex}"
    destination.mkdir(parents=True, exist_ok=False)
    value_range = (min(float(x["raw"].min()) for x in items), max(float(x["raw"].max()) for x in items))
    frames, metadata = [], []
    for index, item in enumerate(items):
        result = render_map(item, image=image, size=size, alpha=alpha, colormap=colormap,
                            normalization=normalization, value_range=value_range)
        np.save(destination / f"{index:04d}_raw.npy", item["raw"], allow_pickle=False)
        for kind in ("heatmap", "overlay"):
            frame = Image.fromarray((np.clip(result[kind], 0, 1) * 255).astype(np.uint8))
            frame.save(destination / f"{index:04d}_{kind}.png")
            if kind == "overlay":
                frames.append(frame)
        metadata.append({k: v for k, v in item.items() if k != "raw"} | {"value_range": result["value_range"]})
    if len(frames) > 1:
        frames[0].save(destination / "sequence.gif", save_all=True, append_images=frames[1:],
                       duration=round(1000 / fps), loop=0)
    (destination / "views.json").write_text(json.dumps(dict(normalization=normalization,
        colormap=colormap, alpha=alpha, maps=metadata), ensure_ascii=False, indent=2), encoding="utf-8")
    return destination
