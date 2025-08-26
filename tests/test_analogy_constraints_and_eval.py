import numpy as np
from core.analogy_engine import AnalogyEngine


def test_analogy_evaluate_monotonic_mapping():
    eng = AnalogyEngine()
    data = {"a": [1, 2], "b": [1, 2], "c": [3, 4]}
    src = eng.build_task_structure(data, task_id="src", domain="test")
    tgt = eng.build_task_structure(data, task_id="tgt", domain="test")
    # perfect identity mapping
    mapping = {"a": "a", "b": "b", "c": "c"}
    score_perfect = eng.evaluate(mapping, src, tgt)
    # random mapping (swap two)
    mapping_bad = {"a": "b", "b": "a", "c": "c"}
    score_bad = eng.evaluate(mapping_bad, src, tgt)
    assert score_perfect >= score_bad


def test_analogy_topk_edges_param_present():
    eng = AnalogyEngine()
    # should be configurable via manifold attribute
    assert hasattr(eng.manifold, "top_k_edges")
