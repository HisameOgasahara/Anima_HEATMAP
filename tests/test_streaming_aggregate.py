import numpy as np
import pytest
import torch

from anima_heatmap import CaptureConfig, CaptureSession, load_maps


@pytest.mark.parametrize("aggregation", ["mean", "daam"])
@pytest.mark.parametrize("heads", ["mean", "all", "1"])
def test_online_aggregation_matches_records_with_unequal_layer_counts(tmp_path, aggregation, heads):
    tokens = {b: {"ids": [1, 2, 3, 0], "pieces": ["▁cat", "▁dog", "</s>", "<pad>"],
                  "special_indices": [2, 3]} for b in ("positive", "negative")}
    torch.manual_seed(4)
    inputs = [(call, layer, torch.randn(2, 2, 4, 3), torch.randn(2, 2, 4, 3))
              for call, layers in [(0, [0]), (1, [0, 1, 2])]
              for layer in layers]
    sessions = []
    for keep in (False, True):
        config = CaptureConfig(keep_records=keep, aggregation=aggregation, heads=heads,
                               branches=("positive", "negative"), relations=("image->text", "image->image"))
        with CaptureSession(tmp_path, config, token_maps=tokens) as session:
            for call, layer, q, k in inputs:
                for branch in config.branches:
                    for relation in config.relations:
                        session.capture(q, k, relation=relation, branch=branch, step=call, call=call,
                                        layer=layer, grid=(1, 2, 2), total_steps=2, total_layers=3)
        sessions.append(session)
    aggregate, detailed = sessions
    assert len(aggregate.records) == 4
    assert aggregate.records[0]["key_indices"] == [0, 1]
    assert not (aggregate.path / "raw").exists()
    for relation in config.relations:
        for branch in config.branches:
            for token_ids in ([0], [1], [0, 1]):
                arguments = dict(relation=relation, branch=branch, batch=1,
                                 token_indices=token_ids, aggregation=aggregation)
                actual = load_maps(aggregate, **arguments)[0]
                expected = load_maps(detailed, **arguments)[0]
                np.testing.assert_allclose(actual["raw"], expected["raw"], rtol=2e-6, atol=1e-7)
                assert actual["record_count"] == expected["record_count"] == 4
    with pytest.raises(ValueError, match="keep_records"):
        load_maps(aggregate, token_indices=[0], view="per_step", aggregation=aggregation)
    with pytest.raises(ValueError, match="aggregation"):
        load_maps(aggregate, token_indices=[0], aggregation="mean" if aggregation == "daam" else "daam")


def test_retained_memory_does_not_grow_with_record_count(tmp_path):
    # One complete map is 1 * 1 * 4 * 3 * 4 = 48 bytes.
    config = CaptureConfig(max_capture_bytes=48)
    with CaptureSession(tmp_path, config) as session:
        for call in range(30):
            session.capture(torch.zeros(1, 1, 4, 2), torch.zeros(1, 1, 3, 2),
                            relation="image->text", branch="positive", step=call, call=call,
                            layer=0, grid=(1, 2, 2), total_steps=30, total_layers=1)
            assert session.capture_bytes == 48
    assert session.capture_bytes == 48
    assert len(session.arrays) == 1
    assert session.records[0]["source_records"] == 30
    np.testing.assert_allclose(load_maps(session, token_indices=[0])[0]["raw"], 1 / 3)
