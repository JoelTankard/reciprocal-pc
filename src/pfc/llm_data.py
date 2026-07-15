from __future__ import annotations

import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol

import torch


BASE_URL = "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main"
MAX_SHARD = 6542
BOS_TOKEN_ID = 256
VOCAB_SIZE = 257
SPECIAL_TOKENS = [
    "<|bos|>",
    "<|user_start|>",
    "<|user_end|>",
    "<|assistant_start|>",
    "<|assistant_end|>",
    "<|python_start|>",
    "<|python_end|>",
    "<|output_start|>",
    "<|output_end|>",
]
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


def shard_name(index: int) -> str:
    return f"shard_{index:05d}.parquet"


def download_climbmix_shards(data_dir: Path, *, train_shards: int, workers: int = 1) -> list[Path]:
    try:
        import requests
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("requests is required for --download-shards; run pip install -e .") from exc
    if train_shards < 0:
        raise ValueError("train_shards must be nonnegative")
    # Keep sequential download. Shards are large; simple retry beats pool noise for local iteration.
    data_dir.mkdir(parents=True, exist_ok=True)
    indices = list(range(train_shards)) + [MAX_SHARD]
    paths = [data_dir / shard_name(index) for index in indices]

    def download_one(index: int) -> Path:
        path = data_dir / shard_name(index)
        if path.exists():
            return path
        tmp = path.with_suffix(path.suffix + ".tmp")
        url = f"{BASE_URL}/{path.name}"
        with requests.get(url, stream=True, timeout=60) as response:
            response.raise_for_status()
            with tmp.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
        tmp.replace(path)
        return path

    missing_indices = [index for index in indices if not (data_dir / shard_name(index)).exists()]
    if workers <= 1 or len(missing_indices) <= 1:
        for index in missing_indices:
            download_one(index)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(download_one, missing_indices))
    return paths


def list_climbmix_shards(data_dir: Path) -> tuple[list[Path], Path]:
    paths = sorted(path for path in data_dir.glob("shard_*.parquet") if not path.name.endswith(".tmp"))
    if not paths:
        raise FileNotFoundError(f"no ClimbMix parquet shards found in {data_dir}")
    val = data_dir / shard_name(MAX_SHARD)
    if val not in paths:
        raise FileNotFoundError(f"missing validation shard {val}; run with --download-shards first")
    train = [path for path in paths if path != val]
    if not train:
        raise FileNotFoundError(f"missing train shards in {data_dir}; run with --download-shards 1")
    return train, val


def iter_parquet_texts(paths: Iterable[Path]) -> Iterable[str]:
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("pyarrow is required to read ClimbMix parquet shards; run pip install -e .") from exc
    for path in paths:
        parquet = pq.ParquetFile(path)
        for row_group in range(parquet.num_row_groups):
            table = parquet.read_row_group(row_group, columns=["text"])
            yield from table.column("text").to_pylist()


def iter_parquet_texts_round_robin(paths: list[Path]) -> Iterable[str]:
    if not paths:
        return
    iterators = [iter_parquet_texts([path]) for path in paths]
    active = [True] * len(iterators)
    active_count = len(iterators)
    index = 0
    while active_count > 0:
        if not active[index]:
            index = (index + 1) % len(iterators)
            continue
        try:
            yield next(iterators[index])
        except StopIteration:
            active[index] = False
            active_count -= 1
        index = (index + 1) % len(iterators)


def encode_bytes(text: str) -> list[int]:
    return [BOS_TOKEN_ID, *text.encode("utf-8", errors="ignore")]


class TextTokenizer(Protocol):
    name: str

    def get_vocab_size(self) -> int:
        ...

    def get_bos_token_id(self) -> int:
        ...

    def encode(self, text: str) -> list[int]:
        ...

    def decode(self, tokens: list[int]) -> str:
        ...

    def token_byte_lengths(self, device: torch.device) -> torch.Tensor:
        ...


class ByteTokenizer:
    name = "byte"

    def get_vocab_size(self) -> int:
        return VOCAB_SIZE

    def get_bos_token_id(self) -> int:
        return BOS_TOKEN_ID

    def encode(self, text: str) -> list[int]:
        return encode_bytes(text)

    def decode(self, tokens: list[int]) -> str:
        byte_values = [token for token in tokens if 0 <= token < 256]
        return bytes(byte_values).decode("utf-8", errors="ignore")

    def token_byte_lengths(self, device: torch.device) -> torch.Tensor:
        return byte_token_lengths(device)


class NanochatBPETokenizer:
    name = "nanochat-bpe"

    def __init__(self, tokenizer_dir: Path) -> None:
        try:
            import pickle
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("pickle is required to load nanochat tokenizer") from exc
        path = tokenizer_dir / "tokenizer.pkl"
        if not path.exists():
            raise FileNotFoundError(f"missing nanochat tokenizer: {path}")
        with path.open("rb") as handle:
            self.enc = pickle.load(handle)
        self.bos_token_id = self.enc.encode_single_token("<|bos|>")
        self.token_bytes_path = tokenizer_dir / "token_bytes.pt"

    def get_vocab_size(self) -> int:
        return int(self.enc.n_vocab)

    def get_bos_token_id(self) -> int:
        return int(self.bos_token_id)

    def encode(self, text: str) -> list[int]:
        return [self.bos_token_id, *self.enc.encode_ordinary(text)]

    def decode(self, tokens: list[int]) -> str:
        return self.enc.decode([token for token in tokens if token != self.bos_token_id])

    def token_byte_lengths(self, device: torch.device) -> torch.Tensor:
        if self.token_bytes_path.exists():
            return torch.load(self.token_bytes_path, map_location=device).to(device=device, dtype=torch.float32)
        special_ids = {self.enc.encode_single_token(token) for token in SPECIAL_TOKENS}
        lengths = []
        for token_id in range(self.get_vocab_size()):
            if token_id in special_ids:
                lengths.append(0)
            else:
                lengths.append(len(self.enc.decode_single_token_bytes(token_id)))
        return torch.tensor(lengths, dtype=torch.float32, device=device)


def train_nanochat_bpe_tokenizer(
    paths: list[Path],
    tokenizer_dir: Path,
    *,
    vocab_size: int,
    max_chars: int,
    doc_cap: int,
    mix_shards: bool = False,
) -> None:
    try:
        import pickle
        import rustbpe
        import tiktoken
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "rustbpe and tiktoken are required for --train-tokenizer; run pip install -e ."
        ) from exc
    if vocab_size - len(SPECIAL_TOKENS) < 256:
        raise ValueError("tokenizer vocab size must leave at least 256 ordinary byte tokens")

    def texts() -> Iterable[str]:
        seen = 0
        source = iter_parquet_texts_round_robin(paths) if mix_shards else iter_parquet_texts(paths)
        for doc in source:
            text = doc[:doc_cap]
            seen += len(text)
            yield text
            if seen >= max_chars:
                return

    tokenizer = rustbpe.Tokenizer()
    tokenizer.train_from_iterator(texts(), vocab_size - len(SPECIAL_TOKENS), pattern=SPLIT_PATTERN)
    mergeable_ranks = {bytes(key): value for key, value in tokenizer.get_mergeable_ranks()}
    offset = len(mergeable_ranks)
    special_tokens = {name: offset + idx for idx, name in enumerate(SPECIAL_TOKENS)}
    enc = tiktoken.Encoding(
        name="nanochat-rustbpe",
        pat_str=tokenizer.get_pattern(),
        mergeable_ranks=mergeable_ranks,
        special_tokens=special_tokens,
    )
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    with (tokenizer_dir / "tokenizer.pkl").open("wb") as handle:
        pickle.dump(enc, handle)
    special_ids = {enc.encode_single_token(token) for token in SPECIAL_TOKENS}
    token_bytes = []
    for token_id in range(enc.n_vocab):
        token_bytes.append(0 if token_id in special_ids else len(enc.decode_single_token_bytes(token_id)))
    torch.save(torch.tensor(token_bytes, dtype=torch.int32), tokenizer_dir / "token_bytes.pt")


def build_text_tokenizer(mode: str, tokenizer_dir: Path) -> TextTokenizer:
    if mode == "byte":
        return ByteTokenizer()
    if mode == "nanochat":
        return NanochatBPETokenizer(tokenizer_dir)
    raise ValueError("tokenizer must be one of: byte, nanochat")


@dataclass
class TokenBatcher:
    paths: list[Path]
    batch_size: int
    seq_len: int
    device: torch.device
    tokenizer: TextTokenizer
    shuffle_docs: bool = False
    seed: int = 0

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.seq_len <= 0:
            raise ValueError("seq_len must be positive")
        self._rng = random.Random(self.seed)
        self._tokens: list[int] = []
        self._token_contexts: list[int] = []
        self._docs = iter_parquet_texts(self.paths)
        self._path_docs = [iter_parquet_texts([path]) for path in self.paths]
        self._path_index = 0

    def next(self) -> tuple[torch.Tensor, torch.Tensor]:
        x, y, _ = self.next_with_context()
        return x, y

    def next_with_context(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        needed = self.batch_size * (self.seq_len + 1)
        while len(self._tokens) < needed:
            text, context_index = self._next_text_with_context()
            tokens = self.tokenizer.encode(text)
            self._tokens.extend(tokens)
            self._token_contexts.extend([context_index] * len(tokens))
        chunk = self._tokens[:needed]
        context_chunk = self._token_contexts[:needed]
        del self._tokens[:needed]
        del self._token_contexts[:needed]
        rows = torch.tensor(
            chunk,
            dtype=torch.long,
            device=self.device,
        ).view(self.batch_size, self.seq_len + 1)
        contexts = torch.tensor(
            context_chunk,
            dtype=torch.long,
            device=self.device,
        ).view(self.batch_size, self.seq_len + 1)
        return rows[:, :-1], rows[:, 1:], contexts[:, :-1]

    def _next_text(self) -> str:
        text, _ = self._next_text_with_context()
        return text

    def _next_text_with_context(self) -> tuple[str, int]:
        if not self.shuffle_docs or len(self.paths) <= 1:
            try:
                return next(self._docs), 0
            except StopIteration:
                self._docs = iter_parquet_texts(self.paths)
                return next(self._docs), 0
        for _ in range(len(self.paths)):
            path_index = self._path_index
            self._path_index = (self._path_index + 1) % len(self.paths)
            try:
                return next(self._path_docs[path_index]), path_index
            except StopIteration:
                self._path_docs[path_index] = iter_parquet_texts([self.paths[path_index]])
                return next(self._path_docs[path_index]), path_index
        raise FileNotFoundError("no documents available from train shards")


@dataclass
class NanochatBestfitBatcher:
    tokenizer_dir: Path
    batch_size: int
    seq_len: int
    device: torch.device
    tokenizer: TextTokenizer
    split: str

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.seq_len <= 0:
            raise ValueError("seq_len must be positive")
        if self.split not in {"train", "val"}:
            raise ValueError("split must be train or val")
        try:
            from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
            from nanochat.tokenizer import RustBPETokenizer
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "nanochat-bestfit requires nanochat on PYTHONPATH"
            ) from exc
        nanochat_tokenizer = RustBPETokenizer.from_directory(self.tokenizer_dir)
        if nanochat_tokenizer.get_vocab_size() != self.tokenizer.get_vocab_size():
            raise ValueError("nanochat and metric tokenizer vocabularies differ")
        if nanochat_tokenizer.get_bos_token_id() != self.tokenizer.get_bos_token_id():
            raise ValueError("nanochat and metric tokenizer BOS ids differ")
        loader_device = "cuda" if self.device.type == "cuda" else str(self.device)
        self._loader = iter(
            tokenizing_distributed_data_loader_bos_bestfit(
                nanochat_tokenizer,
                self.batch_size,
                self.seq_len,
                split=self.split,
                device=loader_device,
            )
        )

    def next(self) -> tuple[torch.Tensor, torch.Tensor]:
        return next(self._loader)

    def next_with_context(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y = self.next()
        return x, y, torch.zeros_like(x)


ByteBatcher = TokenBatcher


def byte_token_lengths(device: torch.device) -> torch.Tensor:
    lengths = torch.ones(VOCAB_SIZE, dtype=torch.float32, device=device)
    lengths[BOS_TOKEN_ID] = 0.0
    return lengths
