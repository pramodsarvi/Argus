"""Prompt registry + canonical message builder.

Invariant: for a given frame set, every request's message list is byte-identical up to the
last image, so vLLM's prefix cache (keyed on tokens + image content hashes) can reuse it.
"""
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Prompt:
    id: str
    version: int
    text: str
    max_tokens: int
    schema: dict = field(default_factory=dict, hash=False, compare=False)


@dataclass(frozen=True)
class Registry:
    system_text: str
    system_version: int
    prompts: tuple[Prompt, ...]

    @classmethod
    def load(cls, path: str | Path) -> "Registry":
        d = yaml.safe_load(Path(path).read_text())
        ps = tuple(Prompt(p["id"], p["version"], p["text"].strip(), p["max_tokens"], p.get("schema", {}))
                   for p in d["prompts"])
        if len({p.id for p in ps}) != len(ps):
            raise ValueError("duplicate prompt ids")
        return cls(d["shared_system"]["text"].strip(), d["shared_system"]["version"], ps)


def media_parts(frame_paths: list[str | Path]) -> list[dict]:
    # Fixed order, file:// URLs; vLLM must be started with --allowed-local-media-path covering these.
    # Each image is preceded by an explicit 1-based "Frame N:" label so frame numbers in model
    # output are well defined. Labels are identical across prompts, so the prefix stays shared.
    parts: list[dict] = []
    for i, p in enumerate(frame_paths, start=1):
        parts.append({"type": "text", "text": f"Frame {i}:"})
        parts.append({"type": "image_url", "image_url": {"url": f"file://{Path(p).resolve()}"}})
    return parts


def frames_to_video(frame_paths: list[str | Path], out_dir: str | Path | None = None,
                    fps: float = 1.0) -> Path:
    """Pack already-sampled/deduped frames into a small high-quality mp4 for video input mode.

    Content-addressed (hash of frame-content hashes + fps), so every prompt in a bundle gets the
    byte-identical file and the prefix cache can hit. Idempotent: reuses an existing file.
    Qwen2.5-VL merges frames in temporal pairs, so an odd count is padded by repeating the last
    frame. Frame dims must already be multiples of 28 (vlmpipe.frames guarantees this).
    """
    import hashlib
    import shutil
    import subprocess
    import tempfile

    paths = [Path(p) for p in frame_paths]
    if not paths:
        raise ValueError("no frames")
    h = hashlib.sha256(f"fps={fps}".encode())
    for p in paths:
        h.update(hashlib.sha256(p.read_bytes()).digest())
    out_dir = Path(out_dir) if out_dir else paths[0].parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"vid_{h.hexdigest()[:16]}.mp4"
    if out.exists():
        return out
    seq = paths + ([paths[-1]] if len(paths) % 2 else [])
    with tempfile.TemporaryDirectory(dir=out_dir) as td:
        for i, p in enumerate(seq, start=1):
            shutil.copyfile(p, Path(td) / f"{i:05d}{p.suffix}")
        tmp = Path(td) / "out.mp4"
        # yuv420p for decoder compatibility (vLLM's video loader); crf 12 ~ visually lossless.
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(fps),
                        "-i", str(Path(td) / f"%05d{seq[0].suffix}"), "-c:v", "libx264",
                        "-crf", "12", "-preset", "medium", "-g", "1", "-pix_fmt", "yuv420p",
                        "-r", str(fps), str(tmp)], check=True)
        tmp.replace(out)          # atomic: concurrent builders never see a partial file
    return out


def video_parts(frame_paths: list[str | Path], fps: float = 1.0,
                out_dir: str | Path | None = None) -> list[dict]:
    n = len(frame_paths)
    v = frames_to_video(frame_paths, out_dir, fps)
    label = (f"The video has {n} frames, numbered 1..{n} in order; "
             f"frame k is at {(1 / fps):g}*(k-1) seconds.")
    return [{"type": "text", "text": label},
            {"type": "video_url", "video_url": {"url": f"file://{v.resolve()}"}}]


def build_messages(system_text: str, frame_paths: list[str | Path], question: str | None,
                   input_mode: str = "images", fps: float = 1.0,
                   video_dir: str | Path | None = None) -> list[dict]:
    """Shared prefix = system + media. Question (if any) is appended in the SAME user turn,
    after the media, so only suffix tokens differ.

    input_mode="images": one labelled image part per frame.
    input_mode="video":  frames packed into one mp4 (≈half the visual tokens on Qwen2.5-VL via
                         temporal patch merging); one text label describes frame numbering.
    """
    if input_mode == "images":
        content = media_parts(frame_paths)
    elif input_mode == "video":
        content = video_parts(frame_paths, fps, video_dir)
    else:
        raise ValueError(f"unknown input_mode {input_mode!r}")
    if question is not None:
        content = content + [{"type": "text", "text": question}]
    return [{"role": "system", "content": system_text},
            {"role": "user", "content": content}]


def merged_question(prompts: tuple[Prompt, ...]) -> tuple[str, dict]:
    """Single-prompt alternative: ask everything at once, one JSON object keyed by prompt id."""
    q = "Answer each task below. Return one JSON object with one key per task id.\n" + \
        "\n".join(f"- {p.id}: {p.text}" for p in prompts)
    schema = {"type": "object", "properties": {p.id: p.schema for p in prompts},
              "required": [p.id for p in prompts]}
    return q, schema
