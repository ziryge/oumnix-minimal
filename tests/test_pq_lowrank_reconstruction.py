import numpy as np
from memory.infinity_window import ProductQuantizer, LowRankCompressor


def test_pq_encode_decode_reconstruction_bound():
    dim = 16
    n = 256
    X = np.random.randn(n, dim).astype('float32')
    pq = ProductQuantizer(dim=dim, n_clusters=16, n_subvectors=4)
    pq.train(X)
    codes, codebooks = pq.encode(X)
    Xr = pq.decode(codes, codebooks)
    err = np.linalg.norm(X - Xr) / max(1.0, np.linalg.norm(X))
    assert err < 0.7  # loose bound for PQ in small-sample setting


def test_lowrank_reconstruction_bound():
    rng = np.random.default_rng(1337)
    M = rng.standard_normal((128, 16)).astype('float32')
    lr = LowRankCompressor(rank=8)
    comp = lr.compress(M)
    Mr = lr.decompress(comp)
    err = np.linalg.norm(M - Mr) / max(1.0, np.linalg.norm(M))
    assert err < 0.7
