"""Single-tensor access to safetensors checkpoints: fetch one tensor (e.g. the input embedding table) from a local
checkpoint directory or a Hugging Face repo without downloading the rest, and gather rows from it via a memmap.

A decoder's embedding table is all the client needs to build prompt embeddings, and all ``align`` needs to map one
decoder's embedding space onto another's, so this avoids pulling 50+ GB of weights to get 1-3 GB.
"""

from collections.abc import Callable
import json
from pathlib import Path
import struct

import numpy as np
import requests
import torch

INDEX_FILE = "model.safetensors.index.json"
SINGLE_FILE = "model.safetensors"
EMBED_NAMES = (
    "model.language_model.embed_tokens.weight",  # Qwen3.5 / Qwen3.x (multimodal checkpoints)
    "model.embed_tokens.weight",  # Qwen3, Llama, Mistral
    "embed_tokens.weight",
    "transformer.wte.weight",
)
_NUMPY = {"BF16": np.uint16, "F16": np.float16, "F32": np.float32}
_CHUNK = 64 << 20

Reader = Callable[[int, int], bytes]  # (offset, length) -> bytes


def _local_reader(path: Path) -> Reader:
    def read(offset: int, length: int) -> bytes:
        with open(path, "rb") as handle:
            handle.seek(offset)
            return handle.read(length)

    return read


def _http_reader(url: str, token: str | None = None) -> Reader:
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    def read(offset: int, length: int) -> bytes:
        response = requests.get(url, headers={**headers, "Range": f"bytes={offset}-{offset + length - 1}"}, timeout=600)
        response.raise_for_status()
        if response.status_code != 206 and length < int(response.headers.get("Content-Length", length)):
            raise OSError(f"{url} ignored the Range header")
        return response.content

    return read


def _source(source: str, filename: str, token: str | None) -> Reader:
    """A reader for ``filename`` inside ``source``: a local directory or a Hugging Face repo id."""
    local = Path(source).expanduser()
    if local.is_dir():
        return _local_reader(local / filename)
    return _http_reader(f"https://huggingface.co/{source}/resolve/main/{filename}", token)


def read_header(read: Reader) -> tuple[dict, int]:
    """Return a safetensors header and the byte offset where tensor data starts."""
    (size,) = struct.unpack("<Q", read(0, 8))
    return json.loads(read(8, size)), 8 + size


def locate(source: str, name: str | None = None, token: str | None = None) -> tuple[str, str]:
    """Find the shard holding ``name`` (default: the first known embedding-table name present).

    Returns:
        ``(tensor name, shard file name)``.
    """
    index = None
    local = Path(source).expanduser()
    try:
        if local.is_dir():
            raw = (local / INDEX_FILE).read_bytes()
        else:
            headers = {"Authorization": f"Bearer {token}"} if token else {}
            response = requests.get(f"https://huggingface.co/{source}/resolve/main/{INDEX_FILE}", headers=headers,
                                    timeout=60)
            response.raise_for_status()
            raw = response.content
        index = json.loads(raw)["weight_map"]
    except (OSError, requests.HTTPError, json.JSONDecodeError, KeyError):
        index = None
    if index is None:
        header, _ = read_header(_source(source, SINGLE_FILE, token))
        index = {key: SINGLE_FILE for key in header if key != "__metadata__"}
    for candidate in [name] if name else EMBED_NAMES:
        if candidate in index:
            return candidate, index[candidate]
    raise KeyError(f"None of {[name] if name else EMBED_NAMES} found in {source}")


def fetch_tensor(source: str, out: str | Path, name: str | None = None, token: str | None = None) -> Path:
    """Copy one tensor from a checkpoint (local dir or HF repo) into a standalone ``.safetensors`` file.

    Only the bytes of that tensor are read (HTTP range requests for remote repos).
    """
    name, shard = locate(source, name, token)
    read = _source(source, shard, token)
    header, data_start = read_header(read)
    info = header[name]
    begin, end = info["data_offsets"]
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    new_header = json.dumps(
        {name: {"dtype": info["dtype"], "shape": info["shape"], "data_offsets": [0, end - begin]},
         "__metadata__": {"source": source, "shard": shard}}
    ).encode()
    new_header += b" " * (-len(new_header) % 8)
    with open(out, "wb") as handle:
        handle.write(struct.pack("<Q", len(new_header)) + new_header)
        for offset in range(begin, end, _CHUNK):
            handle.write(read(data_start + offset, min(_CHUNK, end - offset)))
    return out


class EmbeddingTable:
    """Memory-mapped rows of a 2-D tensor in a safetensors file; reads only the rows asked for."""

    def __init__(self, path: str | Path, name: str | None = None) -> None:
        path = Path(path)
        header, data_start = read_header(_local_reader(path))
        names = [key for key in header if key != "__metadata__"]
        self.name = name or next((n for n in EMBED_NAMES if n in names), names[0])
        info = header[self.name]
        if info["dtype"] not in _NUMPY or len(info["shape"]) != 2:
            raise ValueError(f"Unsupported tensor {self.name}: {info['dtype']} {info['shape']}")
        self.dtype = info["dtype"]
        self.shape = tuple(info["shape"])
        begin, _ = info["data_offsets"]
        self._array = np.memmap(path, dtype=_NUMPY[self.dtype], mode="r", offset=data_start + begin, shape=self.shape)

    @property
    def num_rows(self) -> int:
        return self.shape[0]

    @property
    def dim(self) -> int:
        return self.shape[1]

    def rows(self, ids: np.ndarray | list[int] | torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Gather rows ``ids`` as a ``[len(ids), dim]`` tensor."""
        ids = np.asarray(ids.cpu() if isinstance(ids, torch.Tensor) else ids, dtype=np.int64)
        raw = np.ascontiguousarray(self._array[ids])
        tensor = torch.from_numpy(raw)
        if self.dtype == "BF16":
            tensor = tensor.view(torch.bfloat16)
        return tensor.to(dtype)
