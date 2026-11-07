"""python -m vlmpipe.ingest --inbox /mnt/nvme/inbox --temporal host:7233 --config configs/ingest.yaml"""
import argparse
import asyncio
import logging
from pathlib import Path

import yaml

from .watcher import Watcher


async def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m vlmpipe.ingest")
    ap.add_argument("--inbox", required=True)
    ap.add_argument("--work-root", default=None, help="default: parent of --inbox (same filesystem!)")
    ap.add_argument("--temporal", default="localhost:7233")
    ap.add_argument("--namespace", default="default")
    ap.add_argument("--config", default="configs/ingest.yaml")
    ap.add_argument("--node", default=None, help="default: hostname")
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text()) if Path(a.config).exists() else {}
    logging.basicConfig(level=cfg.get("log_level", "INFO"))

    from temporalio.client import Client
    from vlmpipe.orchestration.camera import submit_chunk   # provided by orchestration
    client = await Client.connect(a.temporal, namespace=a.namespace)

    async def submit(camera_id: str, path: str, start_ts: float, node: str) -> None:
        await submit_chunk(client, camera_id, path, start_ts, node)

    kw = {k: cfg[k] for k in ("stable_s", "pattern", "fallback", "max_attempts",
                              "backoff_base_s", "backoff_max_s") if k in cfg}
    w = Watcher(a.inbox, submit, work_root=a.work_root, node=a.node, **kw)
    await w.run_forever(cfg.get("poll_s", 2.0))


if __name__ == "__main__":
    asyncio.run(main())
