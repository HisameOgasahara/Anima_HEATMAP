# Anima Tag Influence

[English](README.md) | [한국어](README_KR.md) | [Attention 히트맵](README__HEATMAP_KR.md)

프롬프트에서 태그를 뺐을 때, 생성 중 어느 위치의 예측이 얼마나 달라지는지 보여주는 ComfyUI 커스텀 노드.

[Colab](https://colab.research.google.com/github/HisameOgasahara/Anima_HEATMAP/blob/feature/tag-influence/notebooks/Anima_Heatmap_ComfyUI.ipynb) · [예제 워크플로](example/tag_influence.json?raw=true) · [모델](https://huggingface.co/circlestone-labs/Anima)

## 예시

빛나를 생성한 원본 이미지:

![빛나 원본 이미지](example/tag_influence/generated.png)

아래 행은 각각 캐릭터·작가·스케치북·의상 태그를 제거한 영향이다. 왼쪽부터 생성 1·4·8·12·16·20스텝이며, 모든 지도에 같은 색 눈금을 사용했다. 빨강·노랑일수록 예측 변화가 크다.

![빛나의 태그별·단계별 영향 지도](example/tag_influence/comparison.png)

| 행 | 제거한 태그 |
| --- | --- |
| 캐릭터 | `dawn (pokemon)` |
| 작가 | `@ogipote` |
| 소품 | `sketchbook` |
| 의상 | `red t-shirt`, `denim jeans` |

각 단계의 같은 생성 상태에서 태그가 있는 예측과 태그를 제거한 예측을 비교한다. 다음 단계는 원본 프롬프트로 이어지며, 영향 지도는 완성 이미지 위에 겹친다.

## 측정 원리

지도 값은 같은 생성 상태와 sigma에서 두 CFG 예측의 차이를 sigma로 나눈 뒤, latent 채널 방향의 L2 크기를 구한 값이다. 태그를 제거한 효과에는 달라진 프롬프트 문맥의 영향도 포함된다.

## 사용

1. 원본 긍정 프롬프트를 `positive`에, 비교할 태그를 제거한 프롬프트를 `ablated_positive`에 연결한다.
2. `Tag Influence Sampler`의 `latent`를 VAE Decode로 보낸다.
3. 완성 이미지와 `influence`를 `Tag Influence View`에 연결하고, `overlays` 또는 `heatmaps`를 Preview Image로 보낸다.

Sampler의 `steps`에 맞춰 단계별 지도를 만든다. 한 단계에서 모델을 여러 번 호출하는 샘플러는 noise schedule 구간별로 측정을 평균한다.

| View 설정 | 표시 |
| --- | --- |
| `view=per_step` | 단계별 지도 |
| `view=aggregate` | 선택한 단계의 전체 측정 평균 |
| `steps=all` / `last` / `0,4,8` / `0-9` | 전체 / 마지막 / 특정 단계 / 범위. 입력 번호는 0부터 시작 |
| `scale_max=0` | 전체 측정의 최대값으로 공통 색 눈금 설정 |
| `scale_max>0` | 지정한 최대값으로 색 눈금 고정 |
| `alpha` | 오버레이 불투명도 |

<details>
<summary>로컬 설치 · Windows cmd</summary>

```bat
git clone --branch feature/tag-influence --single-branch https://github.com/HisameOgasahara/Anima_HEATMAP.git
cd Anima_HEATMAP
uv pip install --python "<ComfyUI Python 경로>" -e .
xcopy comfyui_anima_heatmap "<ComfyUI 경로>\custom_nodes\comfyui_anima_heatmap\" /E /I
```

</details>
