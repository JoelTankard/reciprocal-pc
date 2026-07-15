from __future__ import annotations

from pathlib import Path
import sys
from types import ModuleType

import torch

import pfc.llm_data as llm_data
from pfc.llm_data import ByteTokenizer, build_text_tokenizer


def test_byte_tokenizer_rounds_to_byte_vocab() -> None:
    tokenizer = ByteTokenizer()
    ids = tokenizer.encode("A")
    lengths = tokenizer.token_byte_lengths(torch.device("cpu"))

    assert ids == [256, 65]
    assert tokenizer.decode(ids) == "A"
    assert tokenizer.get_vocab_size() == 257
    assert lengths[256].item() == 0.0
    assert lengths[65].item() == 1.0


def test_nanochat_tokenizer_requires_pickle(tmp_path) -> None:
    try:
        build_text_tokenizer("nanochat", tmp_path)
    except FileNotFoundError as exc:
        assert "tokenizer.pkl" in str(exc)
    else:
        raise AssertionError("expected missing tokenizer error")


def test_token_batcher_can_mix_train_shards(monkeypatch) -> None:
    docs_by_path = {
        "a": ["A"],
        "b": ["B"],
    }

    def fake_iter(paths):
        for path in paths:
            yield from docs_by_path[path.name]

    monkeypatch.setattr(llm_data, "iter_parquet_texts", fake_iter)
    batcher = llm_data.TokenBatcher(
        [Path("a"), Path("b")],
        batch_size=1,
        seq_len=3,
        device=torch.device("cpu"),
        tokenizer=ByteTokenizer(),
        shuffle_docs=True,
    )

    x, y = batcher.next()

    assert x.tolist()[0] == [256, 65, 256]
    assert y.tolist()[0] == [65, 256, 66]


def test_round_robin_text_iterator(monkeypatch) -> None:
    docs_by_path = {
        "a": ["a1", "a2"],
        "b": ["b1"],
    }

    def fake_iter(paths):
        for path in paths:
            yield from docs_by_path[path.name]

    monkeypatch.setattr(llm_data, "iter_parquet_texts", fake_iter)

    assert list(llm_data.iter_parquet_texts_round_robin([Path("a"), Path("b")])) == ["a1", "b1", "a2"]


def test_nanochat_bestfit_batcher_wraps_official_loader(monkeypatch, tmp_path) -> None:
    calls = {}

    class FakeNanochatTokenizer:
        def get_vocab_size(self):
            return 257

        def get_bos_token_id(self):
            return 256

    class FakeRustBPETokenizer:
        @classmethod
        def from_directory(cls, path):
            calls["tokenizer_dir"] = path
            return FakeNanochatTokenizer()

    def fake_loader(tokenizer, batch_size, seq_len, *, split, device):
        calls.update(
            tokenizer=tokenizer,
            batch_size=batch_size,
            seq_len=seq_len,
            split=split,
            device=device,
        )
        yield torch.tensor([[1, 2]]), torch.tensor([[2, 3]])

    package = ModuleType("nanochat")
    dataloader = ModuleType("nanochat.dataloader")
    tokenizer_module = ModuleType("nanochat.tokenizer")
    dataloader.tokenizing_distributed_data_loader_bos_bestfit = fake_loader
    tokenizer_module.RustBPETokenizer = FakeRustBPETokenizer
    monkeypatch.setitem(sys.modules, "nanochat", package)
    monkeypatch.setitem(sys.modules, "nanochat.dataloader", dataloader)
    monkeypatch.setitem(sys.modules, "nanochat.tokenizer", tokenizer_module)

    batcher = llm_data.NanochatBestfitBatcher(
        tokenizer_dir=tmp_path,
        batch_size=1,
        seq_len=2,
        device=torch.device("cpu"),
        tokenizer=ByteTokenizer(),
        split="val",
    )
    x, y, context = batcher.next_with_context()

    assert x.tolist() == [[1, 2]]
    assert y.tolist() == [[2, 3]]
    assert context.tolist() == [[0, 0]]
    assert calls["tokenizer_dir"] == tmp_path
    assert isinstance(calls["tokenizer"], FakeNanochatTokenizer)
    assert calls["batch_size"] == 1
    assert calls["seq_len"] == 2
    assert calls["split"] == "val"
    assert calls["device"] == "cpu"
