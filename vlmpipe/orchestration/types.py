"""Workflow/activity payloads. Plain dataclasses only (safe inside the workflow sandbox)."""
from dataclasses import dataclass, field


@dataclass
class VideoInput:
    video_uri: str                       # local path on the node (or already-downloaded file)
    video_id: str
    node_queue: str                      # f"node-{hostname}": all work for this video stays on that node
    model: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    base_url: str = "http://localhost:8000/v1"
    prompts_path: str = "configs/prompts.yaml"
    workdir: str = "/mnt/nvme/frames"
    segment_seconds: int = 60
    fps: float = 1.0
    # ~256 visual tokens/frame; with max_frames=48 a segment stays ~12k tokens (< 32k max-model-len)
    max_pixels: int = 256 * 28 * 28
    hwaccel: str | None = None
    max_parallel_segments: int = 4       # segments in analysis at once (VLM backpressure)
    max_parallel_decode: int = 2         # segments being sampled at once (runs ahead of analysis)
    input_mode: str = "images"           # "images" | "video" (Qwen2.5-VL temporal merge, ~half tokens)
    sampling: str = "fixed"              # "fixed" | "scene" (vlmpipe.frames strategy)
    max_frames: int | None = 48
    analysis_mode: str = "bundle"        # "bundle" (warm + fan-out) | "two_tier" (vlmpipe.twotier)


@dataclass
class SegmentInput:
    video: VideoInput
    segment_index: int
    segment_path: str


@dataclass
class PromptSpec:
    id: str
    version: int
    text: str
    max_tokens: int
    schema: dict = field(default_factory=dict)
    required: bool = True
    deadline_s: float = 180.0


@dataclass
class PromptPlan:
    system_text: str
    system_version: int
    prompts: list[PromptSpec]


@dataclass
class FramesOut:
    frame_paths: list[str]
    stats: dict


@dataclass
class AnalyzeInput:
    segment: SegmentInput
    plan: "PromptPlan"
    frames: "FramesOut"


@dataclass
class PromptResult:
    idempotency_key: str
    video_id: str
    segment_index: int
    prompt_id: str
    prompt_version: int
    model: str
    status: str                          # ok | invalid | failed | timed_out
    output: dict | str | None = None
    error: str | None = None
    metrics: dict = field(default_factory=dict)


@dataclass
class SegmentOutcome:
    segment_index: int
    status: str                          # complete | partial | failed
    results: list[PromptResult]
    frame_stats: dict = field(default_factory=dict)


@dataclass
class VideoOutcome:
    video_id: str
    status: str
    segments: list[SegmentOutcome]


def idempotency_key(video_id: str, segment_index: int, prompt_id: str, prompt_version: int,
                    model: str) -> str:
    return f"{video_id}:{segment_index}:{prompt_id}:v{prompt_version}:{model}"


# ----------------------------------------------------------------------------------------------
# Camera entity workflow (one ~3-min chunk per camera every ~3 min)
# ----------------------------------------------------------------------------------------------

# Defaults for 3-minute chunks: scene-adaptive, <= 32 frames x ~256 tokens ~= 8k visual tokens.
DEFAULT_CHUNK_SAMPLING = {"strategy": "scene", "fps": 1.0, "min_fps": 0.2, "max_fps": 1.0,
                          "max_frames": 32, "max_pixels": 256 * 28 * 28}
# Minimum-latency profile (measured on Qwen2.5-VL-7B FP8 / L40S: ~1.05 s per new video for 5
# prompts at 8 frames x ~128 tokens, vs ~1.5 s at 16 x 256). Camera-set sampling overrides this.
LATENCY_SAMPLING = {"strategy": "scene", "fps": 1.0, "min_fps": 0.1, "max_fps": 1.0,
                    "max_frames": 8, "max_pixels": 128 * 28 * 28}
PROFILES = {"balanced": DEFAULT_CHUNK_SAMPLING, "latency": LATENCY_SAMPLING}


@dataclass
class CameraConfig:
    camera_id: str
    node_queue: str
    model: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    base_url: str = "http://localhost:8000/v1"
    promptsets_path: str = "configs/promptsets.yaml"
    workdir: str = "/mnt/nvme/frames"
    chunk_seconds: float = 180.0
    analysis_mode: str = "bundle"        # "bundle" (default: concurrent prompts, measured fastest and
                                         # most reliable on 7B) | "two_tier" (describe once, answer
                                         # in text; only worth it with many prompts per camera)
    profile: str = "balanced"            # "balanced" | "latency" (see PROFILES)
    warm: str = "auto"                   # bundle: warm the prefix first? "auto" | "always" | "never"
    cleanup_frames: bool = True          # delete sampled frames once the chunk is analysed
    input_mode: str = "images"
    hwaccel: str | None = None
    degrade_after_chunks: float = 1.0    # lag > this many chunk durations -> degrade
    stale_after_chunks: float = 2.0      # lag > this many chunk durations -> mark stale, skip
    reorder_window_s: float = 5.0        # wait this long after wake-up so near-simultaneous
                                         # arrivals are sorted before the earliest is processed
    continue_as_new_after: int = 50      # chunks per run before continue-as-new
    max_chunks: int | None = None        # stop after N chunks (tests / bounded runs); None = forever
    idle_timeout_s: float | None = 1800.0  # no chunk for this long -> workflow completes; the next
                                           # submit_chunk (signal-with-start) starts a fresh run
    max_description_bytes: int = 16_000  # cap on carried previous description (history size)


@dataclass
class ChunkInput:
    camera_id: str
    chunk_path: str
    chunk_start_ts: float                # absolute UTC epoch seconds of the chunk's first frame
    duration_s: float | None = None      # defaults to CameraConfig.chunk_seconds


@dataclass
class ChunkRecord:
    chunk_start_ts: float
    status: str                          # complete | partial | failed | stale | empty
    degraded: bool = False
    lag_s: float = 0.0


@dataclass
class CameraState:
    config: CameraConfig
    pending: list[ChunkInput] = field(default_factory=list)
    seen_ts: list[float] = field(default_factory=list)      # bounded dedupe window
    previous_description: dict | None = None
    last_ts: float | None = None
    total_chunks: int = 0
    recent: list[ChunkRecord] = field(default_factory=list)  # bounded, for the status query


@dataclass
class ChunkSampleInput:
    config: CameraConfig
    chunk: ChunkInput
    degraded: bool = False


@dataclass
class ChunkAnalyzeInput:
    config: CameraConfig
    chunk: ChunkInput
    frames: FramesOut
    previous_description: dict | None = None
    degraded: bool = False


@dataclass
class ChunkOutcome:
    chunk_start_ts: float
    status: str
    results: list[PromptResult]
    frame_stats: dict = field(default_factory=dict)
    description: dict | None = None
    prompt_set: str | None = None        # "set_id@version"


@dataclass
class CameraSummary:
    camera_id: str
    total_chunks: int
    recent: list[ChunkRecord]
    ended: str = "max_chunks"            # "max_chunks" | "idle"


def chunk_idempotency_key(camera_id: str, chunk_start_ts: float, prompt_id: str, prompt_version: int,
                          set_id: str, set_version, model: str) -> str:
    return (f"{camera_id}:{chunk_start_ts:.3f}:{prompt_id}:v{prompt_version}:"
            f"ps-{set_id}@{set_version}:{model}")
