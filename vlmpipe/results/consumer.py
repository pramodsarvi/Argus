"""Kafka -> JSON-schema validation -> idempotent upsert. Commits offsets only after upsert
(at-least-once delivery; idempotent key makes redelivery harmless)."""
import logging
from pathlib import Path

import yaml

from .models import ResultRecord

log = logging.getLogger(__name__)


def _schemas_from(d) -> dict[tuple[str, int], dict]:
    """Reads both prompts.yaml (`prompts:`) and promptsets.yaml (`library:`) shapes,
    defensively: entries without id/version are skipped."""
    out: dict[tuple[str, int], dict] = {}
    if not isinstance(d, dict):
        return out
    groups = [d.get("prompts"), d.get("library")]
    sets = d.get("sets")                       # promptsets.yaml: set-specific inline prompts
    if isinstance(sets, dict):
        groups += [s.get("inline") for s in sets.values() if isinstance(s, dict)]
    for items in groups:
        items = items or []
        if isinstance(items, dict):            # library may be a mapping id -> spec
            items = [{"id": k, **v} for k, v in items.items() if isinstance(v, dict)]
        for p in items:
            if isinstance(p, dict) and "id" in p and "version" in p:
                out[(str(p["id"]), int(p["version"]))] = p.get("schema") or {}
    return out


def load_schemas(prompts_yaml: str | Path, *extra: str | Path) -> dict[tuple[str, int], dict]:
    """Schemas keyed by (prompt_id, version). Later files win on conflicts. Missing extra files
    are ignored, so callers can always pass configs/promptsets.yaml."""
    out: dict[tuple[str, int], dict] = {}
    for path in (prompts_yaml, *extra):
        p = Path(path)
        if p.exists():
            out.update(_schemas_from(yaml.safe_load(p.read_text())))
    return out


def default_schemas(config_dir: str | Path = Path(__file__).resolve().parents[2] / "configs"
                    ) -> dict[tuple[str, int], dict]:
    """prompts.yaml + camera prompt-set library (configs/promptsets.yaml) if present."""
    d = Path(config_dir)
    return load_schemas(d / "prompts.yaml", d / "promptsets.yaml")


def validate(rec: ResultRecord, schemas: dict[tuple[str, int], dict]) -> ResultRecord:
    import jsonschema
    if not rec.ok:
        if rec.status is None:
            rec.status = "failed"
        return rec
    schema = schemas.get((rec.prompt_id, rec.prompt_version))
    if schema is None:
        rec.ok, rec.error = False, f"unknown prompt {rec.prompt_id} v{rec.prompt_version}"
        rec.status = "invalid"
        return rec
    try:
        jsonschema.validate(rec.output, schema)
    except jsonschema.ValidationError as e:
        rec.ok, rec.error, rec.status = False, f"schema: {e.message}", "invalid"
        return rec
    rec.status = rec.status or "ok"
    return rec


def process_message(value: bytes, store, schemas) -> ResultRecord:
    rec = validate(ResultRecord.from_json(value), schemas)
    store.upsert(rec)
    return rec


def run_kafka_consumer(bootstrap: str, topics: list[str], store, schemas,
                       group_id: str = "vlm-results-store", on_record=None,
                       stop=lambda: False, dlq_topic: str | None = "vlm.results.dlq") -> None:
    """At-least-once: commit only after upsert, or after a poison message is parked in the DLQ."""
    from confluent_kafka import Consumer
    c = Consumer({"bootstrap.servers": bootstrap, "group.id": group_id,
                  "enable.auto.commit": False, "auto.offset.reset": "earliest"})
    c.subscribe(topics)
    dlq = None
    if dlq_topic:
        from confluent_kafka import Producer
        dlq = Producer({"bootstrap.servers": bootstrap, "enable.idempotence": True, "acks": "all"})
    try:
        while not stop():
            msg = c.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                log.warning("kafka error: %s", msg.error())
                continue
            try:
                rec = process_message(msg.value(), store, schemas)
                if on_record:
                    on_record(rec)
            except Exception as e:  # noqa: BLE001 - poison message -> DLQ, then commit
                log.exception("failed to process message at %s/%s", msg.topic(), msg.offset())
                if dlq is not None:
                    dlq.produce(dlq_topic, key=msg.key(), value=msg.value(), headers=[
                        ("src-topic", msg.topic().encode()), ("src-offset", str(msg.offset()).encode()),
                        ("error", repr(e)[:500].encode())])
                    dlq.flush(10)
            c.commit(message=msg, asynchronous=False)
    finally:
        c.close()
