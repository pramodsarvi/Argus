# vlmpipe: v2 video → VLM batch pipeline

Design: `video_inference_pipeline_architecture_v2.md`. Current stage: **week-1 go/no-go spike**.

## Layout
| Path | What |
|---|---|
| `vlmpipe/segment.py` | ffprobe + keyframe-aligned stream-copy segmentation |
| `vlmpipe/frames.py` | fps sampling, processor-exact resize, dHash dedup, content-addressed PNGs (ffmpeg CPU or `-hwaccel cuda`) |
| `vlmpipe/dali_decode.py` | optional DALI/NVDEC path for the decode benchmark |
| `vlmpipe/resize.py` | Qwen2/2.5-VL `smart_resize` + token count |
| `vlmpipe/prompts.py` | versioned registry, canonical shared-prefix message builder, merged-prompt variant |
| `vlmpipe/bundle.py` | `concurrent` / `warm` / `merged` bundle modes; TTFT, cached tokens, JSON validation |
| `bench/spike.py` | runs the modes, resets the prefix cache between runs, writes JSONL |
| `configs/prompts.yaml` | 5 example use cases with JSON schemas |
| `scripts/serve_vllm.sh` | one vLLM replica per GPU, FP8, prefix caching |
| `vlmpipe/orchestration/` | Temporal workflows (video → segments → prompts), node-pinned task queues, deadlines, partial completion, worker, submit client |
| `vlmpipe/results/` | ResultRecord, Kafka/in-memory publishers + async adapter (`from_env`), schema-validating consumer, SQLite/Postgres idempotent store, per-video reducer + rules, webhook alerts |
| `vlmpipe/routing.py` | rendezvous-hash replica routing, health/failover, vLLM metrics scrape, admission control |
| `vlmpipe/filters.py` | CLIP relevance frame filter (lazy torch), pass-through, chain |
| `vlmpipe/telemetry.py` | Prometheus metrics + OTel spans (no-op without libs) |
| `eval/` | per-use-case scorers, eval runner, regression gate |
| `bench/` | spike, analyze_spike, capacity, decode_bench |
| `deploy/` | Dockerfile, single-node docker-compose, Prometheus, Grafana dashboard, k8s manifests |
| `configs/` | prompts, rules, replicas |

## Run the spike on the GPU box
```bash
python -m venv .venv && .venv/bin/pip install -e '.[dev]' vllm
mkdir -p /mnt/nvme/frames
GPU=0 FRAMES=/mnt/nvme/frames scripts/serve_vllm.sh            # terminal 1
.venv/bin/python -m bench.spike --video your_sample.mp4 \
   --workdir /mnt/nvme/frames/spike --repeats 3 --hwaccel cuda  # terminal 2
```
Repeat with `QUANT=none` (BF16), `MODEL=Qwen/Qwen2.5-VL-3B-Instruct`, and on the L40S as well as the RTX PRO 6000.

**What to look for:** `cache_hit` should be clearly higher for `warm` than for `concurrent`. Fan-out TTFT should drop in `warm` mode. Compare `merged` and `warm` on both wall time and output quality.

## Full stack on one GPU node
```bash
cd deploy && docker compose up -d
.venv/bin/python -m vlmpipe.orchestration.client --help   # submit a video
```

## Tests (no GPU needed)
`.venv/bin/pytest -q`
