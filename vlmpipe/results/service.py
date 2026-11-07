"""Results service for camera streams: Kafka -> validate -> store -> per-chunk assembly ->
camera temporal rules -> webhooks.

A chunk is evaluated once every prompt in its camera's prompt set has a result (any status),
or after `chunk_timeout_s` (late/missing prompts are treated as failed for rule purposes).
Per-camera workflows process chunks in order, so chunks normally complete in order; the rule
engine ignores any chunk older than the last one it saw for that camera.
"""
import logging
import time
from collections import defaultdict

from .camera_rules import CameraRuleEngine, load_camera_rules
from .models import ResultRecord

log = logging.getLogger(__name__)


class ChunkAssembler:
    def __init__(self, expected_prompts, engine: CameraRuleEngine, on_alert,
                 chunk_timeout_s: float = 600.0, clock=time.time) -> None:
        """expected_prompts(camera_id) -> set of prompt ids for that camera's prompt set."""
        self.expected = expected_prompts
        self.engine = engine
        self.on_alert = on_alert
        self.timeout = chunk_timeout_s
        self.clock = clock
        self.pending: dict[tuple[str, float], dict[str, ResultRecord]] = defaultdict(dict)
        self.first_seen: dict[tuple[str, float], float] = {}

    def add(self, rec: ResultRecord) -> list[dict]:
        if not rec.camera_id or rec.chunk_start_ts is None:
            return []                                   # not a camera-stream record
        k = (rec.camera_id, rec.chunk_start_ts)
        self.pending[k][rec.prompt_id] = rec            # latest wins (redelivery-safe)
        self.first_seen.setdefault(k, self.clock())
        try:
            need = set(self.expected(rec.camera_id))
        except KeyError:
            need = set()
        if need and need <= set(self.pending[k]):
            return self._flush(k)
        return self.sweep()

    def sweep(self) -> list[dict]:
        now = self.clock()
        alerts = []
        for k in sorted(k for k, t in self.first_seen.items() if now - t >= self.timeout):
            alerts += self._flush(k)
        return alerts

    def _flush(self, k) -> list[dict]:
        recs = list(self.pending.pop(k, {}).values())
        self.first_seen.pop(k, None)
        alerts = self.engine.observe_records(recs)
        for a in alerts:
            self.on_alert(a)
        return alerts


def build_default(rules_path: str = "configs/rules.yaml", webhook_url: str | None = None,
                  promptsets_path: str = "configs/promptsets.yaml",
                  cameras_path: str = "configs/cameras.yaml") -> ChunkAssembler:
    from vlmpipe.promptsets import PromptSetRegistry
    reg = PromptSetRegistry.load(promptsets_path, cameras_path)

    def expected(camera_id: str) -> set[str]:
        reg.reload_if_changed()
        return {p.id for p in reg.for_camera(camera_id).registry.prompts}

    if webhook_url:
        from .alerts import WebhookDispatcher
        wd = WebhookDispatcher(webhook_url)
        on_alert = lambda a: wd.send(a["camera_id"], a)          # noqa: E731
    else:
        on_alert = lambda a: log.warning("ALERT %s", a)           # noqa: E731
    return ChunkAssembler(expected, CameraRuleEngine(load_camera_rules(rules_path)), on_alert)


def main() -> None:
    import argparse
    import os
    from .consumer import default_schemas, run_kafka_consumer
    from .store import PostgresStore, SQLiteStore
    ap = argparse.ArgumentParser(description="camera results service")
    ap.add_argument("--bootstrap", default=os.environ.get("KAFKA_BOOTSTRAP", "localhost:9092"))
    ap.add_argument("--topics", nargs="+", default=["^vlm\\.results\\.(?!dlq$).*"])
    ap.add_argument("--db", default=os.environ.get("DATABASE_URL", "results.db"))
    ap.add_argument("--webhook", default=os.environ.get("ALERT_WEBHOOK_URL"))
    ap.add_argument("--rules", default="configs/rules.yaml")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    store = PostgresStore(a.db) if a.db.startswith("postgres") else SQLiteStore(a.db)
    asm = build_default(a.rules, a.webhook)
    run_kafka_consumer(a.bootstrap, a.topics, store, default_schemas(), on_record=asm.add)


if __name__ == "__main__":
    main()
