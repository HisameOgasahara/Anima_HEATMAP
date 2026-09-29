# Anima Profiler

Anima Heatmap 샘플러 내부의 attention 계산·CPU 전송·NPY 저장 시간을 측정한다.

## 설치

Anima Heatmap 최신 소스와 공통 Python 모듈이 필요하다. Colab 노트북의 설치 셀에서 함께 설치된다.

로컬에서는 저장소 루트에서 실행하고 ComfyUI를 재시작한다.

```bat
uv pip install --python "<ComfyUI Python 경로>" -e .
xcopy comfyui_anima_profiler "<ComfyUI 경로>\custom_nodes\comfyui_anima_profiler\" /E /I
```

## 사용

1. 기존 Heatmap KSampler를 `Anima Profiler · Heatmap KSampler`로 교체하고 같은 입력을 연결한다.
2. `latent`·`session`은 기존처럼 연결하고, `report`는 `Preview Any`에 연결한다.
3. 실행 후 표기된 시간과 세션 폴더의 `profile.json`을 확인한다.

| 항목 | 측정 범위 |
| --- | --- |
| 샘플러 전체 | 생성·attention 수집·세션 마무리 |
| attention 계산·준비 | Q/K 처리·확률 계산·결과 검사 등, CPU 전송 시간 제외 |
| CPU 전송 | attention 결과의 `.cpu()` 호출 |
| attention NPY 저장 | 수집 기록의 `np.save` 호출 |
| 나머지 생성·실행 | 전체에서 attention 계산·전송·NPY 저장을 뺀 시간 |

CPU 전송 횟수와 메모리 부족 분할 전환 횟수도 기록한다. GPU 동기화를 사용하는 구간별 경과 시간이며, 측정 자체의 추가 비용이 포함된다. `attention_total`은 계산과 전송의 합이다. View의 후처리 시간은 별도다.
