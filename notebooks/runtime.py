"""Colab 서버와 터널의 수명만 관리한다. attention 모듈에 의존하지 않는다."""

import json
import os
from pathlib import Path
import re
import subprocess
import time
import urllib.error
import urllib.request


def read_json(url, timeout=3):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


def tail(path, limit=8000):
    return Path(path).read_text(encoding="utf-8", errors="replace")[-limit:]


def wait_for_server(process, base_url, log_path, required_nodes, timeout=240):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"ComfyUI가 종료되었습니다.\n{tail(log_path)}")
        try:
            read_json(base_url + "/system_stats")
            info = read_json(base_url + "/object_info")
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            time.sleep(1)
            continue
        missing = set(required_nodes) - set(info)
        if missing:
            raise RuntimeError(f"커스텀 노드를 불러오지 못했습니다: {sorted(missing)}\n{tail(log_path)}")
        return
    raise TimeoutError(f"ComfyUI 준비 시간이 초과되었습니다.\n{tail(log_path)}")


def start_server(comfy_root, python, port=8188, required_nodes=(), timeout=240):
    root = Path(comfy_root)
    base = f"http://127.0.0.1:{port}"
    log_path = root / "anima_heatmap_server.log"
    executable = Path(python)
    env = {**os.environ, "VIRTUAL_ENV": str(executable.parent.parent),
           "PYTHONUNBUFFERED": "1", "PATH": str(executable.parent) + os.pathsep + os.environ.get("PATH", "")}
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen([str(python), "main.py", "--listen", "127.0.0.1", "--port", str(port)],
                                   cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
    try:
        wait_for_server(process, base, log_path, required_nodes, timeout)
    except BaseException:
        stop_process(process)
        raise
    return process, base, log_path


def start_tunnel(cloudflared, base_url, log_path, server, timeout=120):
    if server.poll() is not None:
        raise RuntimeError("ComfyUI가 실행 중이 아닙니다.")
    read_json(base_url + "/system_stats")
    with Path(log_path).open("w", encoding="utf-8") as log:
        process = subprocess.Popen([str(cloudflared), "tunnel", "--url", base_url,
                                    "--protocol", "http2", "--no-autoupdate"],
                                   stdout=log, stderr=subprocess.STDOUT)
    deadline = time.monotonic() + timeout
    try:
        while time.monotonic() < deadline:
            if server.poll() is not None or process.poll() is not None:
                raise RuntimeError(f"서버 또는 터널이 종료되었습니다.\n{tail(log_path)}")
            content = tail(log_path)
            match = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", content)
            if match and "Registered tunnel connection" in content:
                return process, match.group(0)
            time.sleep(1)
        raise TimeoutError(f"터널 연결 준비 시간이 초과되었습니다.\n{tail(log_path)}")
    except BaseException:
        stop_process(process)
        raise


def follow_logs(server, tunnel, server_log, tunnel_log):
    """셀을 실행 상태로 유지하면서 새 서버 로그를 출력한다."""
    with Path(server_log).open(encoding="utf-8", errors="replace") as log:
        while True:
            output = log.read()
            if output:
                print(output, end="", flush=True)
            if server.poll() is not None:
                print(log.read(), end="", flush=True)
                return
            if tunnel.poll() is not None:
                raise RuntimeError(f"터널이 종료되었습니다.\n{tail(tunnel_log)}")
            time.sleep(0.2)


def stop_process(process):
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
