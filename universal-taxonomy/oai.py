"""Thin OpenAI helpers (client, batched embeddings, chat) shared across steps.

Reads ``OPENAI_API_KEY`` from the environment / ``.env`` (python-dotenv), the
same convention the rest of the repo uses.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List, Optional, Sequence, TypeVar

import numpy as np

T = TypeVar("T")
R = TypeVar("R")

logger = logging.getLogger(__name__)

_CLIENT = None


def get_client():
    """Return a cached OpenAI client (loads .env on first call)."""
    global _CLIENT
    if _CLIENT is None:
        from dotenv import load_dotenv
        from openai import OpenAI

        load_dotenv()
        _CLIENT = OpenAI(timeout=45.0, max_retries=3)
    return _CLIENT


def embed(
    texts: Sequence[str],
    model: str = "text-embedding-3-large",
    dim: int = 3072,
    batch_size: int = 128,
    max_retries: int = 5,
) -> np.ndarray:
    """Embed ``texts``; empty strings get a zero vector without an API call.

    Returns an ``(len(texts), dim)`` float32 array in input order.
    """
    client = get_client()
    out = np.zeros((len(texts), dim), dtype=np.float32)

    # Only non-empty strings are sent to the API.
    idxs = [i for i, t in enumerate(texts) if t and t.strip()]
    for start in range(0, len(idxs), batch_size):
        chunk_idx = idxs[start : start + batch_size]
        chunk = [texts[i] for i in chunk_idx]
        for attempt in range(max_retries):
            try:
                resp = client.embeddings.create(model=model, input=chunk, dimensions=dim)
                break
            except Exception as e:  # noqa: BLE001 - retry transient API errors
                wait = 2**attempt
                logger.warning("embed batch failed (attempt %d): %s; retry in %ds", attempt + 1, e, wait)
                time.sleep(wait)
        else:
            raise RuntimeError(f"embedding failed after {max_retries} retries")
        for j, item in zip(chunk_idx, resp.data):
            out[j] = np.asarray(item.embedding, dtype=np.float32)
    return out


def map_threaded(fn: Callable[[T], R], items: Sequence[T], workers: int = 8) -> List[R]:
    """Run ``fn`` over ``items`` concurrently, returning results in input order."""
    if not items:
        return []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(fn, items))


def chat(prompt: str, model: str, *, response_format: Optional[dict] = None,
         max_retries: int = 5, seed: Optional[int] = None) -> str:
    """Single-shot chat completion returning the message content string."""
    client = get_client()
    kwargs = {"model": model, "messages": [{"role": "user", "content": prompt}]}
    if response_format is not None:
        kwargs["response_format"] = response_format
    if seed is not None:
        kwargs["seed"] = seed
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content or ""
        except Exception as e:  # noqa: BLE001
            wait = 2**attempt
            logger.warning("chat failed (attempt %d): %s; retry in %ds", attempt + 1, e, wait)
            time.sleep(wait)
    raise RuntimeError(f"chat failed after {max_retries} retries")
