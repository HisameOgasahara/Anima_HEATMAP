# Anima Heatmap

Anima의 이미지–텍스트 attention과 이미지 내부 attention을 수집하고, 단어·공간 위치·noise 단계·layer·head별로 분석한다. 원본 확률을 보관하므로 이미지 생성 후 단어나 집계 설정을 바꿔 다시 볼 수 있다.

## 구성

| 위치 | 역할 |
| --- | --- |
| `anima_heatmap/` | ComfyUI 의존성이 없는 계산·수집·집계·시각화 패키지 |
| `comfyui_anima_heatmap/` | Anima Q/K와 생성 수명 연결, ComfyUI 노드 |
| `notebooks/Anima_Heatmap_ComfyUI.ipynb` | 설치 → 모델 다운로드 → ComfyUI 준비 확인 → 터널 연결 |
| `notebooks/runtime.py` | 서버·터널 시작, 준비 확인, 종료 |
| `tests/` | 확률 계산, 수집 분리, 원상복구, 시각화, 서버 준비 검증 |

공통 패키지는 모델을 다운로드하거나 로드하지 않는다. Q/K 텐서와 공간·실행 정보를 받는다. `attention-map-diffusers`를 호출하거나 설치하지 않는다.

## 지원 기능

| 기능 | 동작 |
| --- | --- |
| `image->text` | 각 이미지 patch가 T5 위치의 텍스트 조건을 참고한 확률 |
| `image->image` | 선택한 이미지 query patch가 다른 이미지 key patch를 참고한 확률 |
| 수집 선택 | noise 단계, layer, head, 긍정/부정, 텍스트 key, 이미지 query |
| 재분석 | 단어/구, 반복 출현 번호, 직접 토큰 번호, 이미지 query, batch, 단계, layer, head, 호출 |
| 보기 | 통합, 단계별, layer별, 호출·layer별 |
| 집계 | 기록 평균 또는 DAAM 방식 |
| 색상 | 지도별 상대값, 선택한 지도들의 공통 범위, 0~1 고정 범위 |
| 파일 | 원본 NPY, 메타데이터 JSON, 히트맵 PNG, overlay PNG, 여러 지도의 GIF |

Anima 본체에는 이미지 self-attention과 이미지→텍스트 cross-attention이 있다. FLUX식 joint attention의 `text->image`, `text->text`는 본체의 지원 관계에 포함되지 않는다. LLM adapter 내부 분석은 포함하지 않는다.

## ComfyUI 설치

Python 3.10 이상을 사용한다. ComfyUI가 사용하는 Python에 공통 패키지를 설치하고, 커스텀 노드 폴더를 `custom_nodes`에 복사하거나 연결한다. `uv`가 설치된 환경에서 실행한다.

```bat
uv pip install --python "<ComfyUI의 python.exe 경로>" -e D:\portfolio\Anima_Heatmap
xcopy D:\portfolio\Anima_Heatmap\comfyui_anima_heatmap "<ComfyUI 경로>\custom_nodes\comfyui_anima_heatmap\" /E /I
```

노드 설치 후 ComfyUI를 재시작한다. 공통 패키지 설치와 커스텀 노드 폴더 배치가 모두 필요하다. 코드 변경 후에도 재시작한다.

## 노드 연결

```text
CLIPLoader (type=anima)
  ├─ Text Encode (긍정) ─────────────┐
  └─ Text Encode (부정) ─────────────┤
UNETLoader (Anima) ──────────────────┤
EmptyLatentImage ────────────────────┤
Anima Heatmap · Settings ────────────┤
                                    ▼
                         Anima Heatmap · KSampler
                           ├─ latent → VAE Decode ──┐
                           └─ session ──────────────┤
                                                   ▼
                                         Anima Heatmap · View
                                           ├─ overlays → Preview Image
                                           └─ heatmaps → Preview Image
```

긍정·부정 모두 `Anima Heatmap · Text Encode`로 인코딩한다. 토큰 표의 번호는 T5 vocabulary ID가 아닌 attention key의 **위치 번호**다. Anima는 Qwen 출력을 어댑터를 통해 T5 위치의 조건으로 바꾼다.

첫 실행은 Settings의 `image->text`, `heads=mean`, `branches=positive`로 시작할 수 있다. View에서 `phrase`에 프롬프트의 단어 또는 구를 입력한다. 토큰 표에서 직접 고르려면 `token_indices`에 `3,4`처럼 입력한다. 직접 지정한 번호가 phrase보다 우선한다.

View의 `phrase`에는 `plana (blue archive)`처럼 프롬프트에 쓴 표현을 그대로 입력한다. 커스텀 노드가 문자 그대로의 표현을 먼저 찾고, 없으면 ComfyUI의 괄호·가중치 문법을 처리한 표현으로 찾는다. 생성 프롬프트나 저장된 attention은 변경하지 않는다. 이 처리는 기존 저장 세션에도 적용된다.

여러 키워드는 줄바꿈이나 쉼표로 구분한다. 예를 들어 `plana (blue archive), blue_ribbon, white school uniform`은 각각 별도의 지도로 출력한다. 괄호 안의 쉼표는 구분자로 사용하지 않는다. `details`를 `Preview Any`에 연결하면 입력 표현, 실제 검색 표현, 선택된 토큰 위치·사전 ID·문자 조각과 출력 이미지 순서를 확인할 수 있다. `token_indices`는 비워 두고 사용하며, 직접 입력하면 자동 검색보다 우선한다.

같은 단어가 여러 번 나오면 `occurrence=all`은 모든 출현을 모으고, `0`, `1` 등은 해당 출현만 선택한다. 단어 매칭은 토큰 경계를 보존한다. SentencePiece의 특수 문자/미지 토큰 때문에 자동 매칭이 되지 않으면 Text Encode의 `token_table`을 `Preview Any`로 확인할 수 있다.

이미지 내부 관계를 보려면 Settings에서 `both` 또는 `image->image`를 고른다. `image_queries=center`는 가운데 patch, `sample:9`는 전체 인덱스에서 균등하게 고른 9개, `all`은 모든 patch를 저장한다. View의 `query_index=-1`은 저장한 첫 query를 사용한다. 실제 저장한 번호는 세션 manifest에 기록된다. 인덱스는 위에서 아래, 왼쪽에서 오른쪽 순서다.

View만 바꾸면 sampler의 캐시된 결과로 다시 분석한다. 새 seed로 sampler를 실행하면 새 세션 폴더를 만든다. 같은 입력으로 sampler 자체가 캐시되면 기존 이미지와 해당 세션을 함께 재사용한다.

## 단계·head·집계

- `steps`, `layers`: `all`, `last`, `0,3,8`, `0-9`. 번호는 0부터 시작한다.
- `heads`: 수집 시 `mean`, `all`, `0,2,4`. 평균만 저장했다면 나중에 개별 head를 복원할 수 없다.
- `text_keys`: 기본 `all`. 선택한 key만 저장하더라도 확률의 분모는 전체 key로 계산한다. 어댑터 출력의 padding도 실제 모델과 동일하게 포함한다.
- `step`: sampler noise schedule의 구간 번호다. **solver callback 횟수가 아니다.** Heun 등의 보정 호출이 다음 noise 값에서 실행되면 그 noise 구간에 기록한다. 정확한 `sigma`와 `call`도 함께 저장한다.
- `mean`: 선택한 기록들의 head 평균 및 기록 평균이다. 모델 호출이 많은 noise 구간에는 더 많은 기록이 있을 수 있다.
- `daam`: head 합 → 각 call 내 layer 평균 → call 합 → 선택 subtoken 평균이다. head 평균으로 저장한 경우 저장된 전체 head 수를 곱해 head 합을 복원한다.
- 원본 지도는 실제 patch 해상도다. 이미지 크기로 표시할 때 보간하고, 모델 입력의 padding 영역은 표시에서 제외한다.
- `relative`: 각 지도의 최솟값~최댓값을 색상 범위로 쓴다. 서로 다른 지도의 색 강도를 절대값처럼 비교하지 않는다.
- `shared`: 이번에 선택한 모든 지도의 공통 범위를 사용한다. 단계 변화 비교에 쓸 수 있다.
- `absolute`: 0~1 범위를 사용한다. DAAM 합산값은 1을 넘을 수 있으므로 DAAM의 절대 비교에는 raw NPY 또는 shared를 쓴다.

## 저장과 ComfyUI 없는 재분석

```text
ComfyUI/output/anima_heatmap/<생성별 고유 ID>/
  manifest.json        # 상태, 설정, prompt/token, seed, sigma, 격자, 기록 인덱스
  raw/000000.npy       # [batch, head 또는 평균 1, query 선택, key 선택]
  views/view_<ID>/     # View에서 save_files=True일 때 생성
```

모든 파일 경로는 세션 기준 상대 경로라 폴더 전체를 이동해도 재분석할 수 있다. 생성 실패 시 `status=failed`와 오류를 남기고, 부분 결과를 완료 결과처럼 표시하지 않는다.

세션을 직접 읽으려면 `Anima Heatmap · Load Session`을 사용한다. 공통 모듈만 설치된 Python에서도 실행할 수 있다.

```bat
python -m anima_heatmap "<세션 폴더>" --phrase "blue hair" --image "<생성 이미지.png>" --view per_step --normalization shared --output "<지도 출력 폴더>"
```

Python에서 직접 수집할 때는 이미 projection·Q/K 정규화·필요한 RoPE가 적용된 Q/K를 전달한다. 입력 규약은 `[batch, heads, tokens, head_dim]`이다.

```python
from anima_heatmap import CaptureConfig, CaptureSession

config = CaptureConfig(relations=("image->text",), heads="all")
with CaptureSession(output_root, config, token_maps=token_maps) as capture:
    # 실행 기반의 연결부에서 실제 Q/K와 메타데이터를 전달한다.
    capture.capture(q, k, relation="image->text", branch="positive",
        step=step, call=call, layer=layer, grid=(1, patch_h, patch_w),
        total_steps=total_steps, total_layers=total_layers, sigma=sigma)
```

`token_maps`의 분기별 값에는 `prompt`, `ids`, `pieces`, `special_indices`를 넣는다. 숫자 토큰 선택만 사용하는 경우에도 모델이 사용한 토큰 순서를 저장하는 것이 좋다.

## Colab

[Colab에서 노트북 열기](https://colab.research.google.com/github/HisameOgasahara/Anima_HEATMAP/blob/main/notebooks/Anima_Heatmap_ComfyUI.ipynb)

Colab 메뉴의 `런타임 → 런타임 유형 변경`에서 GPU를 선택한다.

GPU 런타임을 선택하고 셀을 순서대로 실행한다. 기본 설정은 `HisameOgasahara/Anima_HEATMAP`의 `main` 브랜치를 clone하여 저장소 루트의 모듈과 노드를 설치한다. ZIP 업로드는 필요하지 않다.

다른 저장소를 사용하려면 설정 셀의 `SOURCE_REPO`, `SOURCE_REF`, `SOURCE_SUBDIR`를 변경한다. ZIP으로 설치하려면 `SOURCE_REPO`를 비운다. ZIP에는 `anima_heatmap`, `comfyui_anima_heatmap`, `notebooks`, `pyproject.toml`이 필요하며 `.venv`, `.tools`, 출력 데이터는 넣지 않는다.

노트북은 Anima Base·Qwen 텍스트 인코더·Qwen Image VAE를 준비한다. 설정 셀에서 모델 파일명과 revision을 바꿀 수 있다. ComfyUI는 별도 Python 환경을 사용하고 Colab의 설치된 CUDA 라이브러리를 활용한다.

서버 프로세스 생존, `/system_stats`, `/object_info`의 노드 등록을 확인한 후 HTTP/2 터널을 시작한다. 터널 URL 발급과 연결 등록을 확인하면 바로 접속 링크를 표시한다. 그 뒤에도 5번 실행 셀은 서버 로그를 출력하며 계속 실행된다. 사용을 마치면 셀의 중지 버튼을 누른다. 중지·오류 시 서버와 터널을 함께 종료하며, 실행 셀을 다시 실행하면 이전 프로세스를 먼저 종료한다. 별도 종료 셀은 없다.

소스는 최신 이력만 받는 shallow clone을 사용하고, 다운로드는 진행률과 재시도를 제공하는 curl로 실행한다. 실행 셀이 유지되어도 Colab 자체의 런타임 시간 제한이나 연결 종료를 방지한다고 보장하지 않는다.

Quick Tunnel은 개인 실험용 공개 URL이다. URL을 공유하면 다른 사람도 ComfyUI를 조작할 수 있다. 다중 사용자 서비스용 인증·권한 시스템은 포함하지 않는다. 공통 분석 모듈과 커스텀 노드는 외부 통신을 하지 않으며, 노트북이 명시적으로 다운로드와 터널을 실행한다.

## 호환 범위와 비용

ComfyUI 기본 Anima 구현의 `comfy.ldm.anima.model.Anima` 및 Cosmos `compute_qkv` 연결을 사용한다. 긍정/부정 조건 각각 한 개, 일반 KSampler 경로를 지원한다. 영역·시간 조건, ControlNet, torch.compile, attention 연산 교체, 다중 GPU 실행은 이 연결부의 검증 범위에 포함되지 않는다. 별도의 DiffSynth 연결부는 제공하지 않는다.

생성 중 Q/K/V 반환값은 그대로 유지한다. 확률은 float32로 다시 계산하므로 최적화된 저정밀 attention 커널과 비트 단위 동일성을 보장하지 않는다. 생성 결과를 바꾸는 연산을 추가하지 않는다.

전체 self-attention은 patch 수의 제곱에 비례한다. 기본값은 중앙 query 하나이며, 전체 query를 원하면 명시적으로 `all`을 선택한다. `max_capture_gib`는 원본 확률의 누적 저장량 및 개별 결과 크기의 상한이다. 전체 GPU 사용량이나 임시 텐서 메모리의 상한은 아니다. head 평균, 단계/layer 선택, chunk 크기로 수집 비용을 조절한다. 한도를 넘으면 자동 축소하지 않고 명확한 오류로 종료한다.

히트맵은 생성 중 어떤 텍스트 조건/이미지 위치를 참고했는지 보여준다. 단어 제거의 인과적 효과나 최종 픽셀의 기여도 점수를 직접 계산하지 않는다.

## 검증

```bat
uv venv .venv
uv pip install --python .venv\Scripts\python.exe -e ".[test]"
.venv\Scripts\python.exe -m pytest -q
```

테스트는 작은 텐서와 ComfyUI 인터페이스 대역을 사용한다. 실제 Anima 가중치 추론, GPU 속도/메모리, Colab의 외부 터널 접속은 별도 실행 확인이 필요하다.

## 참고 구현

- [Anima](https://huggingface.co/circlestone-labs/Anima)
- [ComfyUI Anima](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/ldm/anima/model.py)
- [attention-map-diffusers](https://github.com/wooyeolBaek/attention-map-diffusers)
- [ComfyUI DAAM](https://github.com/nisaruj/comfyui-daam)
- [Cloudflare Quick Tunnel](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/do-more-with-tunnels/trycloudflare/)
