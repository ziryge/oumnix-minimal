from utils.tokenizer import tokenizer

def test_tokenizer_punctuation_and_ellipsis():
    text = "Wait... really?! (Yes.)"
    ids = tokenizer.encode(text)
    out = tokenizer.decode(ids)
    assert len(out) > 0
