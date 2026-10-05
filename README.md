# Anima Tag Influence

[English](README.md) | [한국어](README_KR.md) | [Attention heatmaps](README__HEATMAP.md)

ComfyUI custom nodes that show where predictions change during generation when a tag is removed from the prompt.

[Colab](https://colab.research.google.com/github/HisameOgasahara/Anima_HEATMAP/blob/feature/tag-influence/notebooks/Anima_Heatmap_ComfyUI.ipynb) · [Example workflow](example/tag_influence.json?raw=true) · [Models](https://huggingface.co/circlestone-labs/Anima) · [Research note (Korean)](docs/tag_influence_research.md)

## Example

Original image generated with Dawn:

![Original Dawn image](example/tag_influence/generated.png)

The rows show the effects of removing the character, artist, sketchbook, and clothing tags. Columns are generation steps 1, 4, 8, 12, 16, and 20. All maps share the same color scale; red and yellow indicate larger prediction changes.

![Dawn tag influence across generation steps](example/tag_influence/comparison.png)

| Row | Removed tags |
| --- | --- |
| Character | `dawn (pokemon)` |
| Artist | `@ogipote` |
| Object | `sketchbook` |
| Clothing | `red t-shirt`, `denim jeans` |

At each step, predictions with and without the selected tags are compared at the same generation state. The original prompt advances generation to the next step, and influence maps are overlaid on the final image.

## Measurement

$$
\begin{aligned}
\Delta v_t &= \frac{D_{\mathrm{full}}(x_t,\sigma_t)-D_{\mathrm{removed}}(x_t,\sigma_t)}{\sigma_t} \\
H_t(h,w) &= \left\|\Delta v_t(:,h,w)\right\|_2
= \sqrt{\sum_{c=1}^{C}\left(\Delta v_t(c,h,w)\right)^2}
\end{aligned}
$$

| Symbol | Meaning |
| --- | --- |
| $x_t$, $\sigma_t$ | Generation state and noise level at step $t$ |
| $D_{\mathrm{full}}$, $D_{\mathrm{removed}}$ | Denoised predictions for the original and tag-removed prompts, using the same negative conditioning and CFG |
| $\Delta v_t$ | Prediction difference divided by sigma |
| $c$, $C$ | Latent channel index and channel count |
| $H_t(h,w)$ | Influence map value at latent position $(h,w)$ |

The effect of removing a tag includes changes to the prompt's context.

## Usage

1. Connect the original positive prompt to `positive` and the prompt with the selected tags removed to `ablated_positive`.
2. Connect the `latent` output of `Tag Influence Sampler` to VAE Decode.
3. Connect the final image and `influence` to `Tag Influence View`, then send `overlays` or `heatmaps` to Preview Image.

Per-step maps follow the Sampler's `steps` setting. For samplers that evaluate the model several times per step, measurements are averaged within each noise schedule interval.

| View setting | Display |
| --- | --- |
| `view=per_step` | Per-step maps |
| `view=aggregate` | Mean of all measurements in the selected steps |
| `steps=all` / `last` / `0,4,8` / `0-9` | All / last / selected steps / range. Input indices start at 0 |
| `scale_max=0` | Use the maximum across all measurements as the shared color scale |
| `scale_max>0` | Fix the color scale to the specified maximum |
| `alpha` | Overlay opacity |

<details>
<summary>Local installation · Windows cmd</summary>

```bat
git clone --branch feature/tag-influence --single-branch https://github.com/HisameOgasahara/Anima_HEATMAP.git
cd Anima_HEATMAP
uv pip install --python "<ComfyUI Python path>" -e .
xcopy comfyui_anima_heatmap "<ComfyUI path>\custom_nodes\comfyui_anima_heatmap\" /E /I
```

</details>
