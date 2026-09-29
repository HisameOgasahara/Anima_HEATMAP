# Anima Profiler

Anima Heatmap 샘플러의 계산·전송·저장 병목을 측정한다.

## 사용

1. 최신 소스·설치 셀을 실행하고 ComfyUI를 재시작한다.
2. 기존 Heatmap KSampler 대신 `Anima Profiler · Heatmap KSampler`를 연결한다.
3. `report`를 `Preview Any`에 연결한다. 실행 후 `profile_path`의 JSON과 같은 폴더의 보고서를 확인한다.

| 설정 | 기본값 | 의미 |
| --- | --- | --- |
| `trace_steps` | 2 | 시작부터 PyTorch CPU·CUDA trace를 수집할 샘플러 스텝 수 |
| `storage_probe_mib` | 64 | 실제 attention 파일에서 추출해 RAM·저장 폴더에 3회 저장할 데이터의 최대 MiB |

전체 실행의 저장·메모리 통계는 끝까지 수집한다. GPU 구간별 동기화와 PyTorch trace는 실행 시간을 늘리므로, 측정 실행의 병목 비중을 비교한다.

기본은 단계·layer 지도를 생성 중 통합해 메모리에 보관한다. `keep_records=True`는 전체 기록을 메모리에 유지한다. 이 모드에서는 NPY 쓰기와 저장 비교 실험을 생략한다. Settings의 `save_raw=True`일 때 원본 NPY를 단일 백그라운드 작업에서 저장한다. 대기 중·저장 중인 배열 합계는 기본 256 MiB(`CaptureConfig.max_pending_bytes`)로 제한하며, 한도를 넘는 단일 배열은 앞선 저장 완료 후 직접 저장한다. 계산 중인 현재 배열은 이 대기열 한도와 별도다. 세션은 모든 저장이 끝난 뒤 완료된다.

`file_save`는 생성과 겹치는 저장 작업 시간이다. 생성 지연은 `storage_blocking`의 대기열 대기·종료 대기·대형 배열 직접 저장 합계로 확인한다. CPU 전송마다 넣었던 장치 동기화는 제거했으며, `cpu_transfer`에는 전송 호출이 기다린 선행 GPU 작업이 포함된다. GPU 연산·복사 자체의 시간은 PyTorch trace로 구분한다.

## 결과

| 파일 | 내용 |
| --- | --- |
| `profile_report.txt` | 병목 순위, 저장 세부 시간·처리량, 비교 실험, 메모리 지표, 주요 연산 |
| `profile.json` | 파일별 크기·단계·layer·시간·스레드 CPU 시간, 메모리·I/O 상태, trace 요약 |
| `torch_trace.json` | CPU 연산·CUDA 커널·메모리 할당·사용자 구간 타임라인 |
| `torch_operators.txt` | trace 구간의 CPU·GPU 연산별 시간 순위 |

`torch_trace.json`은 [Perfetto](https://ui.perfetto.dev/)에서 열 수 있다.

저장 측정은 `open`, `numpy_write`(NPY 헤더·데이터 쓰기), `close`로 구분한다. RAM·파일 비교 실험은 생성 종료 후 실행하며, 버퍼 쓰기와 `fsync` 대기를 따로 기록한다. 실험 파일은 완료 후 삭제한다. Linux에서는 RAM 여유·dirty/writeback·프로세스 RSS·swap·page fault·I/O pressure도 기록한다. 시스템 지표는 다른 프로세스의 영향도 포함한다.

`diagnostic_status=partial`이면 `diagnostic_errors`에 누락된 측정과 실패 원인이 표시된다. CUDA kernel 수집이 비어 있으면 CPU 측정만으로 GPU trace 성공을 표시하지 않는다.

## 로컬 설치

저장소 루트에서 실행한다. Colab 설치 셀은 이 노드를 함께 설치한다.

```bat
uv pip install --python "<ComfyUI Python 경로>" -e .
xcopy comfyui_anima_profiler "<ComfyUI 경로>\custom_nodes\comfyui_anima_profiler\" /E /I
```

[PyTorch Profiler](https://docs.pytorch.org/docs/stable/profiler)
