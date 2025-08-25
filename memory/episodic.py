"""
"""
import faiss
import numpy as np

class EpisodicMemory:
    def __init__(self, dim: int = 256):
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)  
        self.texts = []  

    def add(self, vectors: np.ndarray, texts: list[str]):
        """
"""
        if vectors.shape[0] != len(texts):
            raise ValueError("vectors and texts must have same length")
        self.index.add(vectors.astype('float32'))
        self.texts.extend(texts)

    def search(self, query_vec: np.ndarray, k: int = 5):
        """
"""
        D, I = self.index.search(query_vec.astype('float32'), k)
        results = []
        for dist, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(self.texts):
                continue
            results.append((self.texts[idx], float(dist)))
        return results
