import json

import numpy as np
import pytest
import torch

from anima_heatmap import CaptureConfig, CaptureSession, find_token_spans, load_maps, render_map, save_views


TOKENS = {"prompt": "cat dog cat", "ids": [1, 2, 1, 3],
          "pieces": ["▁cat", "▁dog", "▁cat", "</s>"], "special_indices": [3]}


def record(session, *, branch="positive", call=0, step=0, layer=0, relation="image->text"):
    q = torch.arange(24, dtype=torch.float32).reshape(1, 2, 6, 2) / 20
    k = torch.randn(1, 2, 6 if relation == "image->image" else 4, 2)
    session.capture(q, k, relation=relation, branch=branch, step=step, call=call, layer=layer,
                    grid=(1, 2, 3), total_steps=3, total_layers=2, sigma=0.5)


def test_fresh_sessions_and_branch_isolation(tmp_path):
    paths = []
    for _ in range(2):
        with CaptureSession(tmp_path, token_maps={"positive": TOKENS}) as session:
            record(session)
            record(session, branch="negative", call=1)
        paths.append(session.path)
        manifest = json.loads((session.path / "manifest.json").read_text(encoding="utf-8"))
        assert len(manifest["records"]) == 1
        assert manifest["status"] == "complete"
    assert paths[0] != paths[1]


def test_failure_persists_and_budget_prevents_allocation(tmp_path):
    with pytest.raises(MemoryError):
        with CaptureSession(tmp_path, CaptureConfig(max_capture_bytes=1)) as session:
            record(session)
    manifest = json.loads((session.path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert not list((session.path / "raw").iterdir())
    with pytest.raises(ValueError, match="완료"):
        load_maps(session.path)


def test_token_boundaries_and_repetitions():
    assert find_token_spans(TOKENS, "cat") == [0, 2]
    assert find_token_spans(TOKENS, "cat", "1") == [2]
    assert find_token_spans(TOKENS, "cat dog") == [0, 1]
    with pytest.raises(ValueError):
        find_token_spans(TOKENS, "at")
    split = dict(pieces=["▁blue", "▁ha", "ir", "</s>"], special_indices=[3])
    assert find_token_spans(split, "blue hair") == [0, 1, 2]


def test_views_and_reanalysis(tmp_path):
    config = CaptureConfig(relations=("image->text", "image->image"), heads="all")
    with CaptureSession(tmp_path, config, token_maps={"positive": TOKENS}) as session:
        for step in range(2):
            for layer in range(2):
                record(session, call=step, step=step, layer=layer)
                record(session, call=step, step=step, layer=layer, relation="image->image")
    before = sorted((session.path / "raw").iterdir())
    items = load_maps(session.path, phrase="cat", view="per_step", heads="1")
    assert len(items) == 2 and items[0]["raw"].shape == (2, 3)
    self_maps = load_maps(session.path, relation="image->image")
    assert self_maps[0]["raw"].shape == (2, 3)
    image = np.zeros((20, 30, 3), dtype=np.float32)
    result = render_map(items[0], image=image)
    assert result["overlay"].shape == image.shape
    out = save_views(items, tmp_path, image=image, normalization="shared")
    assert (out / "sequence.gif").exists()
    assert len(list(out.glob("*_raw.npy"))) == 2
    assert sorted((session.path / "raw").iterdir()) == before


def test_daam_head_sum_and_layer_call_aggregation(tmp_path):
    with CaptureSession(tmp_path, CaptureConfig(heads="all"), token_maps={"positive": TOKENS}) as session:
        record(session, call=0, layer=0)
        record(session, call=0, layer=1)
        record(session, call=1, layer=0)
    expected = []
    for r in session.records:
        raw = np.load(session.path / r["file"])[0].sum(0)[:, [0, 2]].mean(-1).reshape(2, 3)
        expected.append(raw)
    result = load_maps(session.path, phrase="cat", aggregation="daam")[0]["raw"]
    np.testing.assert_allclose(result, (expected[0] + expected[1]) / 2 + expected[2], rtol=1e-6)


def test_mean_capture_cannot_select_individual_head(tmp_path):
    with CaptureSession(tmp_path, token_maps={"positive": TOKENS}) as session:
        record(session)
    with pytest.raises(ValueError, match="head 평균"):
        load_maps(session.path, phrase="cat", heads="0")

