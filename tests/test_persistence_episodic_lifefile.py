import numpy as np
from memory.episodic import EpisodicMemory
from memory.persistence import PersistenceManager


def test_persistence_episodic_segments_roundtrip(tmp_path):
    pm = PersistenceManager(base_dir=str(tmp_path), password="pw")
    ep = EpisodicMemory(dim=8, normalize=True, store_vectors=True, metric="l2")
    vecs = np.random.randn(5, 8).astype('float32')
    texts = [f"t{i}" for i in range(5)]
    ep.add(vecs, texts)
    # Save embedding episodic under memory_state
    pm.save_complete_state(model_state={}, memory_state={'episodic': ep}, neuro_state={}, metacognition_state={}, config={})
    st = pm.load_complete_state()
    assert 'episodic' in st
    ep2 = st['episodic']
    res = ep2.search(vecs[0], k=1)
    assert len(res) == 1 and res[0][0] == texts[0]
