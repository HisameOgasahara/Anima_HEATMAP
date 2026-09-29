import ast
import importlib.util
import json
from pathlib import Path
import urllib.error

import pytest


ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location("notebook_runtime", ROOT / "notebooks/runtime.py")
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


class Process:
    def __init__(self, status=None):
        self.status = status

    def poll(self):
        return self.status


def test_readiness_checks_server_and_node_registration(tmp_path, monkeypatch):
    requested = []
    def read(url):
        requested.append(url)
        return {"AnimaHeatmapSampler": {}} if url.endswith("object_info") else {}
    monkeypatch.setattr(runtime, "read_json", read)
    runtime.wait_for_server(Process(), "http://localhost", tmp_path / "log", ["AnimaHeatmapSampler"])
    assert requested == ["http://localhost/system_stats", "http://localhost/object_info"]


def test_dead_server_never_starts_tunnel(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime.subprocess, "Popen", lambda *a, **kw: pytest.fail("터널을 실행하면 안 됩니다"))
    with pytest.raises(RuntimeError, match="실행 중"):
        runtime.start_tunnel("cloudflared", "http://localhost", tmp_path / "log", Process(1))


def test_missing_nodes_report_log(tmp_path, monkeypatch):
    log = tmp_path / "log"
    log.write_text("node import failed")
    monkeypatch.setattr(runtime, "read_json", lambda url: {})
    with pytest.raises(RuntimeError, match="node import failed"):
        runtime.wait_for_server(Process(), "http://localhost", log, ["AnimaHeatmapSampler"])


def test_notebook_code_and_no_capture_implementation():
    notebook = json.loads((ROOT / "notebooks/Anima_Heatmap_ComfyUI.ipynb").read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            ast.parse(source)
            assert "compute_attention" not in source and "compute_qkv" not in source
            assert cell["outputs"] == []
