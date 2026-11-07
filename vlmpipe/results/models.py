import json
import time
from dataclasses import asdict, dataclass, field, fields

STATUSES = ("ok", "invalid", "failed", "timed_out", "stale", "skipped")


@dataclass
class ResultRecord:
    video_id: str
    segment_idx: int
    prompt_id: str
    prompt_version: int
    model: str
    ok: bool
    output: dict | str | None
    segment_start_s: float = 0.0
    # segment-local timestamp (s) of each frame shown to the model, in prompt order
    frame_times: list[float] = field(default_factory=list)
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    # --- camera-stream fields (all optional; absent for plain file/video jobs) ---
    camera_id: str | None = None
    chunk_start_ts: float | None = None       # absolute UTC epoch seconds of the chunk start
    prompt_set: str | None = None
    prompt_set_version: int | None = None
    status: str | None = None                 # ok | invalid | failed | timed_out | stale

    def __post_init__(self) -> None:
        if self.status is not None and self.status not in STATUSES:
            raise ValueError(f"bad status {self.status!r}")

    @property
    def is_camera(self) -> bool:
        return self.camera_id is not None

    def key(self) -> tuple:
        """Idempotency key: re-running the same unit/prompt/model overwrites, never duplicates.

        Camera records: (camera_id, chunk_start_ts, segment_idx, prompt_id, prompt_version, model).
        segment_idx is kept (always 0 when the chunk is the unit) so a chunk that is ever split
        into sub-segments cannot collapse results. Plain records: the original video key."""
        if self.is_camera:
            return (self.camera_id, _ts(self.chunk_start_ts), self.segment_idx, self.prompt_id,
                    self.prompt_version, self.model)
        return (self.video_id, self.segment_idx, self.prompt_id, self.prompt_version, self.model)

    def key_str(self) -> str:
        return "|".join(map(str, self.key()))

    @property
    def partition_key(self) -> str:
        """Kafka key: camera_id for camera streams (per-camera ordering), else video_id."""
        return self.camera_id if self.is_camera else self.video_id

    @property
    def base_ts(self) -> float:
        """Absolute (camera) or video-relative time origin of segment-local frame times."""
        return (self.chunk_start_ts or 0.0) + (self.segment_start_s or 0.0)

    def effective_status(self) -> str:
        return self.status or ("ok" if self.ok else "failed")

    def to_json(self) -> bytes:
        return json.dumps(asdict(self)).encode()

    @classmethod
    def from_json(cls, data: bytes | str) -> "ResultRecord":
        d = json.loads(data)
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})   # forward-compatible


def _ts(ts: float | None) -> str:
    return "None" if ts is None else f"{float(ts):.3f}"
