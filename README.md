# vlmpipe: camera-chunk / video → VLM pipeline

Frames are sampled once with ffmpeg and sent to a local vLLM replica. Every prompt reuses the same
cached media prefix, and results are validated, stored and turned into alerts.
Design and measurements: `video_inference_pipeline_architecture_v2.md` (§11 has the GPU numbers).

## Quick start: one video, minimum latency
```bash
pip install -e .                                   # + a running vLLM, see below
python -m vlmpipe.fast clip.mp4                    # 8 frames x ~128 tokens, all prompts concurrently
python -m vlmpipe.fast clip.mp4 --frames 16 --tokens 256 --only safety ocr
```
Measured on an L40S with Qwen2.5-VL-7B FP8 and a cold cache: ~1.0 s for 8×128, ~1.5 s for 16×256
(5 prompts). Frames must live under the server's `--allowed-local-media-path`
(`$FRAMES_DIR`, default `/mnt/nvme/frames`).

## Serve the model (one replica per GPU)
```bash
GPU=0 FRAMES=/mnt/nvme/frames scripts/serve_vllm.sh      # binds 127.0.0.1, no dev endpoints
```
Defaults are production-safe. Optional: `VLLM_API_KEY` (clients read the same variable),
`MAX_SEQS`, `MM_CACHE_GB`, `MAX_LEN`, `GPU_UTIL`. `DEV_MODE=1` is needed only by `bench/spike.py` and
`bench/latency.py` (they reset the prefix cache between runs). Structured output uses the schema in
the prompt plus tolerant parsing and `jsonschema` validation (`STRUCTURED_MODE`); guided decoding
degraded this model, see §11.

## Camera streams (the full pipeline)
Cameras drop `<camera>_<UTC ts>.mp4` chunks (write under a temp name, then rename) into an inbox.
```
ingest watcher → Temporal CameraWorkflow (per camera) → sample → analyze → Kafka → results service → alerts
```
| Component | Run |
|---|---|
| vLLM | `scripts/serve_vllm.sh` |
| Worker (per GPU node) | `python -m vlmpipe.orchestration.worker` |
| Ingest | `python -m vlmpipe.ingest --inbox /mnt/nvme/inbox` |
| Results | `python -m vlmpipe.results.service` |

Worker and ingest must resolve the same node name: `NODE_NAME`, else the hostname.
Deployment: `deploy/systemd/` (plain VMs, with `vlmpipe.env.example`), `deploy/docker-compose.yml`
(single node), `deploy/k8s/`.

### Tuning knobs (env, see `deploy/systemd/vlmpipe.env.example`)
| Variable | Default | Effect |
|---|---|---|
| `PROFILE` | `balanced` | `latency` = 8 frames × ~128 tokens; per-camera `sampling` in `configs/` overrides |
| `ANALYSIS_MODE` | `bundle` | `two_tier` = describe once, answer in text (not faster at 5 prompts on 7B) |
| `SAMPLE_CONCURRENCY` | `min(4, cpus/2)` | parallel ffmpeg per worker |
| `VLM_MAX_INFLIGHT` | `4` | chunks analysed at once per replica |
| `STRUCTURED_MODE` | `prompt` | `grammar` = vLLM guided JSON (not recommended) |

Resource behaviour: pooled HTTP client per base URL, sampled frames deleted after a successful
analysis (retention sweep is the fallback), warm request only when the shared prefix is large
(≥3000 visual tokens and ≥3 prompts).

## Layout
| Path | What |
|---|---|
| `vlmpipe/fast.py` | single-video minimum-latency CLI |
| `vlmpipe/frames.py`, `resize.py`, `segment.py` | ffmpeg sampling, processor-exact resize, dHash dedup, segmenting |
| `vlmpipe/prompts.py`, `promptsets.py`, `bundle.py`, `twotier.py` | prompts, per-camera sets, request builder + runner, two-tier |
| `vlmpipe/clients.py`, `routing.py`, `telemetry.py` | pooled client, replica routing/admission, metrics |
| `vlmpipe/ingest/` | inbox watcher, SQLite ledger, retention |
| `vlmpipe/orchestration/` | Temporal workflows, activities, worker |
| `vlmpipe/results/` | schema validation, stores, assembler, camera rules (state persisted), alerts |
| `eval/`, `bench/` | accuracy scorers/gate; spike, latency, capacity, decode benchmarks |
| `configs/` | prompts, prompt sets, cameras, rules, ingest |

## Tests
```bash
pip install -e '.[dev]'
pytest -q                                   # no GPU; downloads Temporal's test server on first run
python scripts/smoke_e2e.py [bundle|two_tier]   # real ffmpeg + Temporal + worker + store, fake VLM
python -m bench.latency --video clip.mp4 --workdir $FRAMES_DIR/lat   # needs DEV_MODE=1 server
```
