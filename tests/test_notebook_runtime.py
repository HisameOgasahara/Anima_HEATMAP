import ast
import importlib.util
import json
from pathlib import Path
import urllib.error
from types import SimpleNamespace

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


def test_tunnel_uses_http2_and_returns_without_public_probe(tmp_path, monkeypatch):
    log = tmp_path / "tunnel.log"
    commands, requests = [], []

    def launch(command, **kwargs):
        commands.append(command)
        log.write_text("https://example.trycloudflare.com\nRegistered tunnel connection\n", encoding="utf-8")
        return Process()

    monkeypatch.setattr(runtime.subprocess, "Popen", launch)
    monkeypatch.setattr(runtime, "read_json", lambda url: requests.append(url))
    process, url = runtime.start_tunnel("cloudflared", "http://localhost", log, Process())
    assert url == "https://example.trycloudflare.com"
    command = commands[0]
    assert command[command.index("--protocol") + 1] == "http2"
    assert requests == ["http://localhost/system_stats"]


def test_follow_logs_keeps_running_until_interrupt(tmp_path, monkeypatch, capsys):
    log = tmp_path / "server.log"
    log.write_text("server running\n", encoding="utf-8")

    def interrupt(seconds):
        raise KeyboardInterrupt

    monkeypatch.setattr(runtime.time, "sleep", interrupt)
    with pytest.raises(KeyboardInterrupt):
        runtime.follow_logs(Process(), Process(), log, tmp_path / "tunnel.log")
    assert "server running" in capsys.readouterr().out


@pytest.mark.parametrize("error", [KeyboardInterrupt, RuntimeError])
def test_notebook_interrupt_or_failure_stops_both_processes(tmp_path, error):
    notebook = json.loads((ROOT / "notebooks/Anima_Heatmap_ComfyUI.ipynb").read_text(encoding="utf-8"))
    cell = next(c for c in notebook["cells"] if c["id"] == "anima-heatmap-05")
    tree = ast.parse("".join(cell["source"]))
    block = next(node for node in tree.body if isinstance(node, ast.Try))
    server, tunnel = Process(), Process()
    stopped = []

    def follow(*args):
        raise error()

    fake_runtime = SimpleNamespace(
        start_server=lambda *a, **kw: (server, "http://localhost", tmp_path / "server.log"),
        start_tunnel=lambda *a, **kw: (tunnel, "https://example.trycloudflare.com"),
        follow_logs=follow, stop_process=stopped.append)
    env = dict(runtime=fake_runtime, COMFY=tmp_path, PYTHON="python", PORT=8188,
               required=[], CLOUDFLARED="cloudflared", tunnel_log=tmp_path / "tunnel.log",
               server=None, tunnel=None, display=lambda value: None, Markdown=lambda value: value)
    with pytest.raises(error):
        exec(compile(ast.Module(body=[block], type_ignores=[]), "notebook", "exec"), env)
    assert stopped == [tunnel, server]
    assert env["server"] is None and env["tunnel"] is None
    assert not any(c["id"] in ("anima-heatmap-06", "anima-heatmap-08") for c in notebook["cells"])
