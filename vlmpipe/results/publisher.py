from typing import Protocol

from .models import STATUSES, ResultRecord


class Publisher(Protocol):
    def publish(self, rec: ResultRecord) -> None: ...
    def flush(self, timeout: float = 10.0) -> int: ...


class InMemoryPublisher:
    def __init__(self) -> None:
        self.messages: list[tuple[str, bytes, bytes]] = []   # (topic, key, value)

    def topic_for(self, prompt_id: str) -> str:
        return f"vlm.results.{prompt_id}"

    def publish(self, rec: ResultRecord) -> None:
        self.messages.append((self.topic_for(rec.prompt_id), rec.partition_key.encode(),
                              rec.to_json()))

    def flush(self, timeout: float = 10.0) -> int:
        return 0


class KafkaPublisher:
    """key=camera_id (camera streams) or video_id, so results are ordered per camera/video."""

    def __init__(self, bootstrap: str, topic_prefix: str = "vlm.results",
                 topic_map: dict[str, str] | None = None, extra: dict | None = None) -> None:
        from confluent_kafka import Producer
        conf = {"bootstrap.servers": bootstrap, "enable.idempotence": True, "acks": "all",
                "max.in.flight.requests.per.connection": 5, "compression.type": "zstd",
                "linger.ms": 20}
        conf.update(extra or {})
        self._p = Producer(conf)
        self.prefix, self.topic_map = topic_prefix, topic_map or {}
        self.errors: list[str] = []

    def topic_for(self, prompt_id: str) -> str:
        return self.topic_map.get(prompt_id, f"{self.prefix}.{prompt_id}")

    def _cb(self, err, msg) -> None:
        if err is not None:
            self.errors.append(str(err))

    def publish(self, rec: ResultRecord) -> None:
        self._p.produce(self.topic_for(rec.prompt_id), key=rec.partition_key.encode(),
                        value=rec.to_json(), headers={"idempotency-key": rec.key_str()},
                        on_delivery=self._cb)
        self._p.poll(0)

    def flush(self, timeout: float = 10.0) -> int:
        return self._p.flush(timeout)


class AsyncRecordPublisher:
    """Adapter: orchestration publishes PromptResult dicts asynchronously; results layer
    publishes ResultRecord synchronously. Maps one to the other."""

    def __init__(self, inner: "Publisher") -> None:
        self.inner = inner

    @staticmethod
    def to_record(d: dict) -> ResultRecord:
        m = d.get("metrics") or {}

        def get(k, default=None):     # field may be top-level or inside metrics
            v = d.get(k)
            return m.get(k, default) if v is None else v

        status = d.get("status")
        ts = get("chunk_start_ts")
        psv = get("prompt_set_version")
        return ResultRecord(
            video_id=d["video_id"], segment_idx=d["segment_index"], prompt_id=d["prompt_id"],
            prompt_version=d["prompt_version"], model=d["model"], ok=status == "ok",
            output=d.get("output"), segment_start_s=float(get("segment_start_s", 0.0) or 0.0),
            frame_times=list(get("frame_times", []) or []),
            error=d.get("error") or (None if status == "ok" else status),
            camera_id=get("camera_id"), chunk_start_ts=float(ts) if ts is not None else None,
            prompt_set=get("prompt_set"),
            prompt_set_version=int(psv) if psv is not None else None,
            status=status if status in STATUSES else None)

    async def publish(self, result: dict) -> None:
        import asyncio
        await asyncio.to_thread(self.inner.publish, self.to_record(result))


def from_env() -> AsyncRecordPublisher:
    """KAFKA_BOOTSTRAP set -> Kafka; otherwise in-memory (dev only)."""
    import os
    bootstrap = os.environ.get("KAFKA_BOOTSTRAP")
    inner = KafkaPublisher(bootstrap) if bootstrap else InMemoryPublisher()
    return AsyncRecordPublisher(inner)
