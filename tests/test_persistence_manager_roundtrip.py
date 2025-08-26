import os
import torch
import tempfile
from utils.tokenizer import tokenizer
from core.model import OumnixSimpleAI
from memory.persistence import PersistenceManager

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test_persistence_manager_roundtrip(tmp_path):
    model = OumnixSimpleAI(vocab_size=tokenizer.vocab_size, dim=32, n_layers=1).to(DEVICE)
    out_dir = tmp_path / "state"
    pm = PersistenceManager(base_dir=str(out_dir))
    model_state = model.state_dict()
    memory_state = {"episodic": {"path": str(tmp_path / "episodic")}}
    neuro_state = {"dopamine": 0.0}
    metacognition_state = {"confidence": 0.5}
    config = {"dim": 32, "layers": 1}
    pm.save_complete_state(model_state, memory_state, neuro_state, metacognition_state, config)
    state = pm.load_complete_state()
    assert isinstance(state, dict) and "metadata" in state
