from core.analogy_engine import AnalogyEngine, TaskStructure
from core.causal_engine import CausalEngine, CausalRelation, CausalEvent


def test_analogy_engine_smoke():
    ae = AnalogyEngine(seed_dim=2, max_seeds=10)
    data = {"a": [0.0, 1.0], "b": [1.0, 0.0]}
    # Build a TaskStructure manually for smoke test
    struct = TaskStructure(task_id="t1", domain="d", entities=list(data.keys()), relations=[("a","r","b")], constraints=[], goal_pattern="g", complexity_metrics={})
    assert isinstance(struct, TaskStructure)


def test_causal_engine_smoke():
    ce = CausalEngine()
    rel = CausalRelation(cause="x", effect="y", strength=0.5, confidence=0.9, contexts=["default"])
    ce.scm.add_relation(rel)
    ev = CausalEvent(timestamp=0.0, variables={"x": 1.0, "y": 0.0}, context="default")
    ce.add_event(ev)
    res = ce.scm.simulate_intervention({"x": 1.0}, context="default")
    assert isinstance(res, dict)
