from utils.tokenizer import SimpleTokenizer


def test_wordpiece_encode_unknown_and_known(tmp_path, monkeypatch):
    tok = SimpleTokenizer(dataset_dir=str(tmp_path), vocab_path=str(tmp_path / "vocab.pkl"), max_files=0, use_wordpiece=True)
    # manually craft a vocab with wordpiece entries
    tokens = [tok.pad_token, tok.unk_token, 'he', '##llo', '!']
    tok.id2token = tokens
    tok.token2id = {t: i for i, t in enumerate(tokens)}
    text = "Hello!"
    ids = tok.encode(text)
    # expect to map to 'he', '##llo', '!'
    assert len(ids) == 3
    dec = tok.decode(ids)
    # decode will not merge wordpieces; punctuation should attach
    assert '##' in dec and '!' in dec


def test_wordpiece_fallback_to_unk(tmp_path):
    tok = SimpleTokenizer(dataset_dir=str(tmp_path), vocab_path=str(tmp_path / "vocab.pkl"), max_files=0, use_wordpiece=True)
    # vocab without matching pieces
    tokens = [tok.pad_token, tok.unk_token]
    tok.id2token = tokens
    tok.token2id = {t: i for i, t in enumerate(tokens)}
    ids = tok.encode("xyz")
    # should emit unk
    assert ids and all(i == tok.token2id[tok.unk_token] for i in ids)
