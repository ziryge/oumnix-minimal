from ui.advanced_cli import format_neuro_state, format_system_status, format_memory_stats, format_response_analysis


def test_advanced_ui_formatters():
    neuro = {}
    assert "disabled" in format_neuro_state(neuro).lower()
    status = {"background_threads": 2, "model_params": 123456}
    mem = {"used": 3, "capacity": 10}
    resp = {"confidence": 0.7}
    s1 = format_system_status(status)
    s2 = format_memory_stats(mem)
    s3 = format_response_analysis(resp)
    assert all(isinstance(s, str) and len(s) > 0 for s in [s1, s2, s3])
