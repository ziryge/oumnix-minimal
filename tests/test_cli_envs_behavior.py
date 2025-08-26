import os
import torch
from ui import cli as cli_mod
from utils.tokenizer import tokenizer

def test_cli_envs_sampling_paths(monkeypatch):
    monkeypatch.setenv("OUMNIX_TEMPERATURE", "1.5")
    monkeypatch.setenv("OUMNIX_TOPK", "10")
    monkeypatch.setenv("OUMNIX_MAX_NEW_TOKENS", "2")
    assert callable(cli_mod._env_float)
    assert callable(cli_mod._env_int)
