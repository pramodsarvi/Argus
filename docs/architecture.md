# vlmpipe architecture

Mermaid diagrams (rendered by GitHub). Numbers are measured on one L40S with Qwen2.5-VL-7B FP8,
vLLM 0.33, cold prefix cache. Details and caveats: `video_inference_pipeline_architecture_v2.md` §11.

## 1. Camera-stream pipeline

```mermaid
flowchart TD
    CAM["Cameras<br/>1 chunk / camera / ~3 min<br/>write tmp name, then rename"] --> INBOX[("Inbox<br/>node-local NVMe")]

    subgraph INGEST["Ingest watcher"]
        INBOX --> STABLE["Stable size+mtime<br/>and ffprobe OK"]
        STABLE -->|"bad / unparsable"| REJ[("rejected/")]
        STABLE --> LEDGER["SQLite ledger<br/>dedupe (camera, start_ts)"]
        LEDGER --> SUBMIT["Signal-with-start<br/>CameraWorkflow"]
    end

    subgraph TEMPORAL["Temporal: one CameraWorkflow per camera"]
        SUBMIT --> Q["Ordered chunk queue"]
        Q --> LAG{"Lag behind<br/>real time?"}
        LAG -->|"> 2 chunks"| STALE["Publish stale records, skip"]
        LAG -->|"> 1 chunk"| DEG["Degraded: half the frames,<br/>required prompts only<br/>(optional prompts published as skipped)"]
        LAG -->|on time| NORM["Normal"]
        DEG --> SAMPLE
        NORM --> SAMPLE["sample_chunk<br/>ffmpeg scene sampling, dHash dedup<br/>capped by SAMPLE_CONCURRENCY"]
        SAMPLE --> ANALYZE["analyze_chunk<br/>capped by VLM_MAX_INFLIGHT<br/>heartbeats, resumable"]
        ANALYZE -->|"frames deleted<br/>after success"| DONE(( ))
    end

    PS[("Prompt sets + cameras.yaml<br/>hot reload")] --> ANALYZE

    subgraph VLLM["vLLM replica, same node (127.0.0.1)"]
        ANALYZE --> MODE{"analysis_mode"}
        MODE -->|"bundle (default)"| BUNDLE["All prompts concurrently<br/>warm request only if prefix ≥ 3000 tokens<br/>and ≥ 3 prompts"]
        MODE -->|two_tier| TT["Describe once, answer in text,<br/>escalate to visual on low confidence"]
        BUNDLE --> CACHE[("Prefix cache +<br/>multimodal cache")]
        TT --> CACHE
    end

    BUNDLE --> PUB["Publish per prompt<br/>(idempotent key)"]
    TT --> PUB
    STALE --> PUB
    PUB --> KAFKA[["Kafka vlm.results.*"]]

    subgraph RESULTS["Results service"]
        KAFKA --> VAL["Validate against prompt schema"]
        VAL -->|poison| DLQ[["DLQ"]]
        VAL --> DB[("Postgres / SQLite<br/>idempotent upsert")]
        VAL --> ASM["Chunk assembler<br/>all prompts or timeout (swept when idle)"]
        ASM --> RULES["Camera rules: consecutive k, window N in T<br/>cooldown; state persisted in the DB"]
        RULES --> HOOK["Webhook alerts<br/>(Idempotency-Key)"]
    end

    ANALYZE -. metrics .-> PROM["Prometheus / Grafana"]
```

## 2. Single video, minimum latency (`python -m vlmpipe.fast`)

```mermaid
flowchart LR
    V["video file"] --> P["ffprobe<br/>~0.06 s"]
    P --> S["ffmpeg: N frames,<br/>~128 tokens each, resized to the<br/>model's grid<br/>~0.35 s"]
    S --> F[("frames on local NVMe<br/>inside --allowed-local-media-path")]
    F --> R["Build requests:<br/>system + labelled frames, then question<br/>+ JSON schema in the prompt"]
    R -->|"all prompts at once"| M["vLLM 7B FP8<br/>prefill ~0.15 s<br/>decode ~80 tok/s per sequence<br/>~0.7 s"]
    M --> J["Tolerant JSON parse<br/>+ jsonschema validation"]
    J --> OUT["results + timings<br/>~1.0 s end to end (8 × 128)<br/>~1.5 s (16 × 256)"]
    S -. "removed after use" .-> F
```

Why these choices (measured): guided JSON degraded the 7B's output, so the schema goes in the
prompt; a warm request costs about what it saves below ~3000 visual tokens; `-hwaccel cuda` is
slower on short clips; speculative n-gram decoding and the 3B model gave no gain.
