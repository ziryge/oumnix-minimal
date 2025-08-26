from core.causal_engine import StructuralCausalModel


def test_causal_chain_simulation_and_intervention():
    scm = StructuralCausalModel()
    # A -> B -> C with weights 2.0
    scm.add_relation_simple("A", "B", strength=2.0, confidence=1.0, contexts=["default"])
    scm.add_relation_simple("B", "C", strength=2.0, confidence=1.0, contexts=["default"])
    # simulate baseline with intervention A=1.0
    out = scm.simulate_intervention({"A": 1.0}, context="default")
    # Expect B ~ 2, C ~ 4
    assert abs(out.get("B", 0.0) - 2.0) < 1e-5
    assert abs(out.get("C", 0.0) - 4.0) < 1e-5
    # Intervene on B (clamp to 1), expect C = 2 regardless of A
    out2 = scm.simulate_intervention({"A": 5.0, "B": 1.0}, context="default")
    assert abs(out2.get("C", 0.0) - 2.0) < 1e-5


def test_causal_reject_cycle():
    scm = StructuralCausalModel()
    scm.add_relation_simple("X", "Y", 1.0, 1.0, ["default"])
    scm.add_relation_simple("Y", "X", 1.0, 1.0, ["default"])  # cycle
    try:
        scm.to_dag()
        ok = True
    except ValueError:
        ok = False
    assert not ok
