import os
from memory.persistence import PersistenceManager
from core.oumnix_ai import create_oumnix_ai, OumnixAIConfig
from utils.tokenizer import tokenizer


def test_persistence_fullstate_roundtrip(tmp_path):
    pm = PersistenceManager(base_dir=str(tmp_path), password="pass1")
    cfg = OumnixAIConfig(vocab_size=tokenizer.vocab_size, model_dim=64, n_layers=2, n_heads=2)
    ai = create_oumnix_ai(cfg)
    # simulate some state
    model_state = ai.core_model.state_dict()
    # For portability in this environment, persist a lightweight memory summary
    mem_state = {'stats': ai.memory_system.get_stats()}
    pm.save_complete_state(model_state=model_state, memory_state=mem_state, neuro_state={}, metacognition_state={}, config={'model_dim': 64})
    state = pm.load_complete_state()
    assert 'model_weights' in state and 'memory_state' in state
    # password rotation
    pm.change_password("pass2")
    state2 = pm.load_complete_state()
    assert 'model_weights' in state2
