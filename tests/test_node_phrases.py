import importlib.util
from pathlib import Path
import sys
from types import ModuleType

import pytest


@pytest.fixture
def phrases(monkeypatch):
    # ComfyUI 파서의 반환 계약을 대역으로 제공한다. 모듈은 ComfyUI 없이 테스트한다.
    parser = ModuleType("comfy.sd1_clip")
    parser.escape_important = lambda text: text.replace(r"\(", "\0\2").replace(r"\)", "\0\1")
    parser.unescape_important = lambda text: text.replace("\0\2", "(").replace("\0\1", ")")
    parsed = {
        "plana (blue archive)": [("plana ", 1.0), ("blue archive", 1.1)],
        "(plana (blue archive):1.2)": [("plana ", 1.2), ("blue archive", 1.32)],
    }
    parser.token_weights = lambda text, weight: parsed.get(text, [(text, weight)])
    monkeypatch.setitem(sys.modules, "comfy", ModuleType("comfy"))
    monkeypatch.setitem(sys.modules, "comfy.sd1_clip", parser)
    spec = importlib.util.spec_from_file_location("node_phrases", Path(__file__).parents[1] / "comfyui_anima_heatmap/phrases.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def token_map(pieces):
    return dict(ids=list(range(len(pieces))), pieces=pieces, special_indices=[])


@pytest.mark.parametrize("query", ["plana (blue archive)", "(plana (blue archive):1.2)"])
def test_weighted_tag_maps_to_saved_positions(phrases, query):
    tm = token_map(["▁plan", "a", "▁blue", "▁archive", "▁", ","])
    ids, matched = phrases.resolve_phrase(tm, query)
    assert ids == [0, 1, 2, 3]
    assert matched == "plana blue archive"
    assert "선택 위치: 0,1,2,3" in phrases.describe_selection(tm, query, ids, matched)


@pytest.mark.parametrize("query", ["plana (blue archive)", r"plana \(blue archive\)"])
def test_literal_parentheses_preserved(phrases, query):
    tm = token_map(["▁plan", "a", "▁(", "blue", "▁archive", ")"])
    assert phrases.resolve_phrase(tm, query)[0] == [0, 1, 2, 3, 4, 5]


def test_multiple_queries_preserve_parenthesized_commas(phrases):
    assert phrases.split_phrases("plana (blue archive), blue_ribbon\n(red, blue:1.2)") == [
        "plana (blue archive)", "blue_ribbon", "(red, blue:1.2)"]


def test_occurrence_and_no_partial_word_match(phrases):
    tm = token_map(["▁plan", "a", ",", "▁plan", "a"])
    assert phrases.resolve_phrase(tm, "plana", "1")[0] == [3, 4]
    with pytest.raises(ValueError):
        phrases.resolve_phrase(tm, "plan")
    with pytest.raises(ValueError, match="출현 번호"):
        phrases.resolve_phrase(tm, "plana", "2")
