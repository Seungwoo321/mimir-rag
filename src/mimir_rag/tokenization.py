from __future__ import annotations

import base64
import hashlib
from functools import lru_cache
from importlib.resources import files

import tiktoken

from .errors import ConfigurationError

VOCABULARY_SHA256 = "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7"


@lru_cache(maxsize=1)
def get_encoding() -> tiktoken.Encoding:
    resource = files("mimir_rag.data").joinpath("cl100k_base.tiktoken")
    try:
        data = resource.read_bytes()
    except OSError:
        raise ConfigurationError("Bundled tokenizer is missing; reinstall Mimir-RAG.") from None
    if hashlib.sha256(data).hexdigest() != VOCABULARY_SHA256:
        raise ConfigurationError("Bundled tokenizer checksum failed; reinstall Mimir-RAG.")
    ranks = {
        base64.b64decode(token): int(rank)
        for token, rank in (line.split() for line in data.splitlines() if line)
    }
    return tiktoken.Encoding(
        name="cl100k_base",
        pat_str=(
            r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+|"
            r" ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s"
        ),
        mergeable_ranks=ranks,
        special_tokens={
            "<|endoftext|>": 100257,
            "<|fim_prefix|>": 100258,
            "<|fim_middle|>": 100259,
            "<|fim_suffix|>": 100260,
            "<|endofprompt|>": 100276,
        },
    )
