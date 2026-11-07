"""Run one worker per GPU node. Task queue = node-<hostname> so frames never leave the node.

  python -m vlmpipe.orchestration.worker --temporal localhost:7233 --max-activities 16
"""
import argparse
import os
import asyncio
import socket

from temporalio.client import Client
from temporalio.worker import Worker

from .activities import Activities
from .workflows import CameraWorkflow, VideoAnalysisWorkflow


def node_queue(hostname: str | None = None) -> str:
    # NODE_NAME (k8s downward API, spec.nodeName) is stable across pod restarts; the pod hostname is not.
    return f"node-{hostname or os.environ.get('NODE_NAME') or socket.gethostname()}"


def build_worker(client: Client, task_queue: str, activities: Activities,
                 max_concurrent_activities: int = 16) -> Worker:
    # Each analyze_segment activity fans out all prompts in-process; per-replica VLM load is bounded
    # by VideoInput.max_parallel_segments, this caps total activities (decode + analyze) per worker.
    return Worker(client, task_queue=task_queue,
                  workflows=[VideoAnalysisWorkflow, CameraWorkflow],
                  activities=activities.all(),
                  max_concurrent_activities=max_concurrent_activities)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--temporal", default=os.environ.get("TEMPORAL_ADDRESS", "localhost:7233"))
    ap.add_argument("--namespace", default="default")
    ap.add_argument("--task-queue", default=os.environ.get("TASK_QUEUE"))
    ap.add_argument("--max-activities", type=int, default=16)
    a = ap.parse_args()
    if os.environ.get("METRICS_PORT"):
        from vlmpipe.telemetry import start_metrics_server
        start_metrics_server(int(os.environ["METRICS_PORT"]))
    client = await Client.connect(a.temporal, namespace=a.namespace)
    q = a.task_queue or node_queue()
    print(f"worker on {q}")
    await build_worker(client, q, Activities(), a.max_activities).run()


if __name__ == "__main__":
    asyncio.run(main())
