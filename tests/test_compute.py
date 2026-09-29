import pytest
import torch

from anima_heatmap import compute_attention


@pytest.mark.parametrize("heads", ["mean", "all", [1, 3]])
@pytest.mark.parametrize("masked", [False, True])
def test_chunked_matches_full_softmax(heads, masked):
    torch.manual_seed(17)
    q, k = torch.randn(2, 4, 7, 8), torch.randn(2, 4, 11, 8)
    mask = torch.rand(7, 11) > 0.3 if masked else None
    if masked:
        mask[0] = False
    scores = q @ k.transpose(-1, -2) / 8**0.5
    if mask is not None:
        scores = scores.masked_fill(~mask, -torch.inf)
    expected = scores.softmax(-1).nan_to_num()[:, :, [0, 3, 6]][:, :, :, [2, 7]]
    if heads == "mean":
        expected = expected.mean(1, keepdim=True)
    elif isinstance(heads, list):
        expected = expected[:, heads]
    actual = compute_attention(q, k, query_indices=[0, 3, 6], key_indices=[2, 7],
                               heads=heads, mask=mask, query_chunk=2, key_chunk=3)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=1e-7)


def test_matches_sdpa_output_without_modifying_inputs():
    q, k = torch.randn(1, 2, 5, 4), torch.randn(1, 2, 9, 4)
    v = torch.randn(1, 2, 9, 3)
    before = q.clone(), k.clone()
    mask = torch.randn(5, 9) * 0.1
    actual = compute_attention(q, k, heads="all", mask=mask, key_chunk=2) @ v
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    torch.testing.assert_close(actual, expected)
    assert torch.equal(q, before[0]) and torch.equal(k, before[1])


def test_selected_keys_use_full_denominator():
    q, k = torch.zeros(1, 1, 2, 4), torch.zeros(1, 1, 10, 4)
    actual = compute_attention(q, k, key_indices=[3])
    torch.testing.assert_close(actual, torch.full((1, 1, 2, 1), 0.1))

