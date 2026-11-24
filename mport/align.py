"""Map one decoder's input-embedding space onto another's, from their embedding tables alone.

Two decoders that share token strings give a free set of paired points: row ``i`` of decoder A's table and row
``j`` of decoder B's table embed the same string. A ridge-regression affine map fitted on those pairs carries a memory
head trained for A over to B without ever running B. Same-tokenizer families (Qwen3.5-2B -> a Qwen 27B) share every
token; Qwen3-8B -> a Qwen 27B still shares ~131k strings.

CLI:
    python -m mport.align --src-table a.safetensors --src-tokenizer a/tokenizer.json \
        --dst-table b.safetensors --dst-tokenizer b/tokenizer.json --out maps/a_to_b.safetensors
"""

import argparse
from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np
from safetensors.torch import load_file, save_file
import torch

from mport.tables import EmbeddingTable

CHUNK = 8192


def load_vocab(tokenizer_json: str | Path, include_added: bool = False) -> dict[str, int]:
    """Token string -> id from a Hugging Face ``tokenizer.json`` (base vocab; added/special tokens optional)."""
    data = json.loads(Path(tokenizer_json).read_text())
    vocab = dict(data["model"]["vocab"])
    if include_added:
        vocab.update({token["content"]: token["id"] for token in data.get("added_tokens", [])})
    return vocab


def shared_ids(src_vocab: dict[str, int], dst_vocab: dict[str, int], src_rows: int, dst_rows: int
               ) -> tuple[np.ndarray, np.ndarray]:
    """Row ids of the token strings present in both vocabularies (and inside both tables)."""
    pairs = sorted(
        (src_vocab[token], dst_vocab[token])
        for token in src_vocab.keys() & dst_vocab.keys()
        if src_vocab[token] < src_rows and dst_vocab[token] < dst_rows
    )
    array = np.array(pairs, dtype=np.int64).reshape(-1, 2)
    return array[:, 0], array[:, 1]


@dataclass
class LinearMap:
    """``y = x @ weight.T + bias``: decoder-A embedding space -> decoder-B embedding space."""

    weight: torch.Tensor  # [dst_dim, src_dim]
    bias: torch.Tensor  # [dst_dim]
    stats: dict = field(default_factory=dict)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T.to(x.dtype) + self.bias.to(x.dtype)

    def then(self, other: "LinearMap") -> "LinearMap":
        """Compose: apply ``self`` first, then ``other`` (A->B then B->C gives A->C)."""
        return LinearMap(other.weight @ self.weight, other.weight @ self.bias + other.bias,
                         {"composed": [self.stats, other.stats]})

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_file({"weight": self.weight.contiguous(), "bias": self.bias.contiguous()}, str(path),
                  metadata={"stats": json.dumps(self.stats)})
        return path

    @classmethod
    def load(cls, path: str | Path) -> "LinearMap":
        from safetensors import safe_open

        tensors = load_file(str(path))
        with safe_open(str(path), framework="pt") as handle:
            stats = json.loads((handle.metadata() or {}).get("stats", "{}"))
        return cls(tensors["weight"], tensors["bias"], stats)


def _keep_trained(table: EmbeddingTable, ids: np.ndarray) -> np.ndarray:
    """Mask of rows whose norm is not degenerate (reserved/untrained ids sit near zero or at one shared value)."""
    norms = np.concatenate([table.rows(ids[i:i + CHUNK]).norm(dim=-1).numpy() for i in range(0, len(ids), CHUNK)])
    return norms > 0.1 * np.median(norms)


def fit_map(src: EmbeddingTable, dst: EmbeddingTable, src_ids: np.ndarray, dst_ids: np.ndarray,
            ridge: float = 1e-3, holdout: float = 0.02, seed: int = 0, method: str = "ridge") -> LinearMap:
    """Fit an affine map from ``src`` rows to ``dst`` rows over paired ids, streaming in chunks.

    ``method="ridge"``: unconstrained least squares; ``ridge`` is relative to the mean eigenvalue of the source
    covariance. ``method="procrustes"``: a scaled semi-orthogonal map (rotation/projection times one scalar), which
    cannot stretch directions and so extrapolates more conservatively to vectors off the token manifold.
    A random ``holdout`` fraction is kept out of the fit and scored (R^2, cosine, nearest-neighbour hit rate among
    held-out rows).
    """
    keep = _keep_trained(src, src_ids) & _keep_trained(dst, dst_ids)
    src_ids, dst_ids = src_ids[keep], dst_ids[keep]
    order = np.random.default_rng(seed).permutation(len(src_ids))
    n_test = max(1, int(holdout * len(order))) if holdout > 0 else 0
    test, train = order[:n_test], order[n_test:]

    d_src, d_dst = src.dim, dst.dim
    sx, sy = torch.zeros(d_src, dtype=torch.float64), torch.zeros(d_dst, dtype=torch.float64)
    xtx = torch.zeros(d_src, d_src, dtype=torch.float64)
    xty = torch.zeros(d_src, d_dst, dtype=torch.float64)
    train = np.sort(train)
    for i in range(0, len(train), CHUNK):
        rows = train[i:i + CHUNK]
        x = src.rows(src_ids[rows], torch.float64)
        y = dst.rows(dst_ids[rows], torch.float64)
        sx += x.sum(0)
        sy += y.sum(0)
        xtx += x.T @ x
        xty += x.T @ y
    n = len(train)
    mx, my = sx / n, sy / n
    cov = xtx / n - torch.outer(mx, mx)
    cross = xty / n - torch.outer(mx, my)
    if method == "ridge":
        lam = ridge * torch.trace(cov) / d_src
        w_t = torch.linalg.solve(cov + lam * torch.eye(d_src, dtype=torch.float64), cross)  # [d_src, d_dst]
    elif method == "procrustes":
        u, sigma, vh = torch.linalg.svd(cross, full_matrices=False)  # cross [d_src, d_dst]
        w_t = (sigma.sum() / torch.trace(cov)) * (u @ vh)
    else:
        raise ValueError(f"Unknown method {method!r}")
    bias = my - mx @ w_t
    result = LinearMap(w_t.T.float().contiguous(), bias.float(),
                       {"pairs": int(len(src_ids)), "train": int(n), "test": int(n_test), "ridge": ridge,
                        "method": method})
    if n_test:
        result.stats.update(evaluate_map(result, src, dst, src_ids[test], dst_ids[test]))
    return result


def evaluate_map(mapping: LinearMap, src: EmbeddingTable, dst: EmbeddingTable, src_ids: np.ndarray,
                 dst_ids: np.ndarray) -> dict:
    """R^2, mean cosine, and top-1 nearest-neighbour accuracy (within the given dst rows) of mapped src rows."""
    x = src.rows(src_ids)
    y = dst.rows(dst_ids)
    pred = mapping(x)
    sse = ((pred - y) ** 2).sum()
    sst = ((y - y.mean(0)) ** 2).sum()
    cosine = torch.nn.functional.cosine_similarity(pred, y, dim=-1)
    pn = torch.nn.functional.normalize(pred, dim=-1)
    yn = torch.nn.functional.normalize(y, dim=-1)
    hits = sum(int(((pn[i:i + 1024] @ yn.T).argmax(-1) == torch.arange(i, min(i + 1024, len(y)))).sum())
               for i in range(0, len(y), 1024))
    return {"r2": float(1 - sse / sst), "cosine": float(cosine.mean()), "nn_top1": hits / len(y),
            "norm_ratio": float(pred.norm(dim=-1).mean() / y.norm(dim=-1).mean())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src-table", required=True)
    parser.add_argument("--src-tokenizer", required=True)
    parser.add_argument("--dst-table", required=True)
    parser.add_argument("--dst-tokenizer", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--holdout", type=float, default=0.02)
    args = parser.parse_args()
    src, dst = EmbeddingTable(args.src_table), EmbeddingTable(args.dst_table)
    src_ids, dst_ids = shared_ids(load_vocab(args.src_tokenizer), load_vocab(args.dst_tokenizer), src.num_rows,
                                  dst.num_rows)
    mapping = fit_map(src, dst, src_ids, dst_ids, ridge=args.ridge, holdout=args.holdout)
    mapping.stats.update({"src": args.src_table, "dst": args.dst_table})
    mapping.save(args.out)
    print(json.dumps(mapping.stats, indent=2))


if __name__ == "__main__":
    main()
