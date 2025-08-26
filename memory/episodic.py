import os
import json
import faiss
import numpy as np

class EpisodicMemory:
    """FAISS-backed episodic memory with optional vector persistence and metric toggle (L2/IP)."""
    def __init__(self, dim: int = 256, normalize: bool = False, store_vectors: bool = True, metric: str = "l2"):
        self.dim = dim
        self.normalize = normalize
        self.metric = metric.lower()
        if self.metric not in ("l2", "ip"):
            self.metric = "l2"
        self.index = faiss.IndexFlatL2(dim) if self.metric == "l2" else faiss.IndexFlatIP(dim)
        self.texts = []
        self.store_vectors = store_vectors
        self._vectors = []
        self.vectors = []

    def _maybe_normalize(self, v: np.ndarray) -> np.ndarray:
        if not self.normalize:
            return v
        norms = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
        return v / norms

    def add(self, vectors: np.ndarray, texts: list[str]):
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError("vectors must have shape [n, dim]")
        if vectors.shape[0] != len(texts):
            raise ValueError("vectors and texts must have same length")
        v = vectors.astype("float32")
        v = self._maybe_normalize(v)
        self.index.add(v)
        self.texts.extend(texts)
        if self.store_vectors:
            self._vectors.extend([row.copy() for row in v])
        self.vectors.append(v)

    def search(self, query_vec: np.ndarray, k: int = 5):
        if query_vec.ndim == 1:
            query_vec = query_vec[None, :]
        if query_vec.shape[1] != self.dim:
            raise ValueError("query_vec must have shape [1, dim] or [n, dim]")
        q = query_vec.astype("float32")
        q = self._maybe_normalize(q)
        D, I = self.index.search(q, k)
        results = []
        for dist, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(self.texts):
                continue
            results.append((self.texts[idx], float(dist)))
        return results

    def save(self, dir_path: str):
        os.makedirs(dir_path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(dir_path, "index.faiss"))
        meta = {"dim": self.dim, "normalize": self.normalize, "texts": self.texts, "store_vectors": self.store_vectors, "metric": self.metric}
        with open(os.path.join(dir_path, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f)
        if self.store_vectors and len(self._vectors) > 0:
            import numpy as _np
            _np.save(os.path.join(dir_path, "vectors.npy"), _np.stack(self._vectors, axis=0))

    def load(self, dir_path: str):
        index_path = os.path.join(dir_path, "index.faiss")
        meta_path = os.path.join(dir_path, "meta.json")
        if not os.path.exists(index_path) or not os.path.exists(meta_path):
            raise FileNotFoundError("index.faiss or meta.json not found")
        self.index = faiss.read_index(index_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.dim = int(meta.get("dim", self.dim))
        self.normalize = bool(meta.get("normalize", self.normalize))
        self.texts = list(meta.get("texts", []))
        self.store_vectors = bool(meta.get("store_vectors", self.store_vectors))
        self.metric = str(meta.get("metric", self.metric)).lower()
        vecs_path = os.path.join(dir_path, "vectors.npy")
        if self.store_vectors and os.path.exists(vecs_path):
            import numpy as _np
            arr = _np.load(vecs_path)
            self._vectors = [arr[i].copy() for i in range(arr.shape[0])]
