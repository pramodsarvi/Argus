"""Frame filter stage: drop irrelevant frames before they cost VLM tokens.

FrameFilter.filter(paths) -> (kept_paths, stats). First and last frames are always kept so the
VLM keeps temporal context. Heavy deps (torch, open_clip / transformers) import lazily.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol, Sequence

import numpy as np


@dataclass
class FilterStats:
    name: str
    input: int
    kept: int
    scores: list[float] | None = None

    @property
    def keep_ratio(self) -> float:
        return self.kept / self.input if self.input else 0.0


class FrameFilter(Protocol):
    def filter(self, paths: Sequence[Path]) -> tuple[list[Path], FilterStats]: ...


class PassThrough:
    def filter(self, paths):
        paths = list(paths)
        return paths, FilterStats("passthrough", len(paths), len(paths))


# Embedder: (images: list[Path]) -> (N,D) array, (texts: list[str]) -> (M,D) array.
ImageEmbed = Callable[[list[Path]], np.ndarray]
TextEmbed = Callable[[list[str]], np.ndarray]


def _l2(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12, None)


def open_clip_embedders(model: str = "ViT-B-32", pretrained: str = "laion2b_s34b_b79k",
                        device: str | None = None, batch: int = 64) -> tuple[ImageEmbed, TextEmbed]:
    import open_clip  # lazy
    import torch
    from PIL import Image

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    m, _, preprocess = open_clip.create_model_and_transforms(model, pretrained=pretrained, device=device)
    m.eval()
    tok = open_clip.get_tokenizer(model)

    @torch.no_grad()
    def img(paths):
        out = []
        for i in range(0, len(paths), batch):
            x = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in paths[i:i + batch]]).to(device)
            out.append(m.encode_image(x).float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 1))

    @torch.no_grad()
    def txt(texts):
        return m.encode_text(tok(texts).to(device)).float().cpu().numpy()

    return img, txt


class CLIPRelevanceFilter:
    """Keep frames whose max cosine similarity to any query >= threshold.

    Thresholds are model-specific (ViT-B-32 relevant pairs often score ~0.2-0.3); calibrate on
    labelled footage. min_keep guarantees the VLM still gets context when nothing matches.
    """

    def __init__(self, queries: list[str], threshold: float = 0.22, min_keep: int = 2,
                 embedders: tuple[ImageEmbed, TextEmbed] | None = None, **clip_kw):
        if not queries:
            raise ValueError("need at least one query")
        self.queries = queries
        self.threshold = threshold
        self.min_keep = min_keep
        self._emb = embedders
        self._clip_kw = clip_kw
        self._q = None

    def _embedders(self):
        if self._emb is None:
            self._emb = open_clip_embedders(**self._clip_kw)
        return self._emb

    def scores(self, paths: Sequence[Path]) -> np.ndarray:
        img, txt = self._embedders()
        if self._q is None:
            self._q = _l2(np.asarray(txt(self.queries), dtype=np.float32))
        f = _l2(np.asarray(img(list(paths)), dtype=np.float32))
        return (f @ self._q.T).max(axis=1)

    def filter(self, paths):
        paths = list(paths)
        if len(paths) <= 2:
            return paths, FilterStats("clip", len(paths), len(paths))
        s = self.scores(paths)
        keep = s >= self.threshold
        keep[0] = keep[-1] = True
        for i in np.argsort(-s):            # top up with best-scoring frames to reach min_keep
            if keep.sum() >= self.min_keep:
                break
            keep[i] = True
        kept = [p for p, k in zip(paths, keep) if k]
        return kept, FilterStats("clip", len(paths), len(kept), [round(float(x), 4) for x in s])


class Chain:
    def __init__(self, *filters: FrameFilter):
        self.filters = filters

    def filter(self, paths):
        paths = list(paths)
        stats = []
        for f in self.filters:
            paths, st = f.filter(paths)
            stats.append(st)
        return paths, stats
