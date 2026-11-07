"""Submit a video. Workflow id = video_id, so resubmitting the same video is deduplicated.

  python -m vlmpipe.orchestration.client /mnt/nvme/in/cam1.mp4 --video-id cam1-2026-11-06 --node gpu01
"""
import argparse
import asyncio
from dataclasses import asdict

from temporalio.client import Client
from temporalio.common import WorkflowIDReusePolicy

from .types import VideoInput, VideoOutcome
from .workflows import VideoAnalysisWorkflow


async def submit(client: Client, v: VideoInput, wait: bool = False):
    h = await client.start_workflow(
        VideoAnalysisWorkflow.run, v, id=v.video_id, task_queue=v.node_queue,
        id_reuse_policy=WorkflowIDReusePolicy.ALLOW_DUPLICATE_FAILED_ONLY)
    return await h.result() if wait else h


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video_uri")
    ap.add_argument("--video-id", required=True)
    ap.add_argument("--node", required=True, help="hostname of the GPU node holding the file")
    ap.add_argument("--temporal", default="localhost:7233")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--mode", choices=["bundle", "two_tier"], default="bundle")
    ap.add_argument("--max-parallel-segments", type=int, default=4)
    ap.add_argument("--max-parallel-decode", type=int, default=2)
    ap.add_argument("--input-mode", choices=["images", "video"], default="images")
    ap.add_argument("--sampling", choices=["fixed", "scene"], default="fixed")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--wait", action="store_true")
    a = ap.parse_args()
    client = await Client.connect(a.temporal)
    v = VideoInput(a.video_uri, a.video_id, f"node-{a.node}", a.model, a.base_url,
                   max_parallel_segments=a.max_parallel_segments,
                   max_parallel_decode=a.max_parallel_decode, analysis_mode=a.mode,
                   input_mode=a.input_mode, sampling=a.sampling, max_frames=a.max_frames)
    r = await submit(client, v, a.wait)
    print(asdict(r) if isinstance(r, VideoOutcome) else f"started {r.id}")


if __name__ == "__main__":
    asyncio.run(main())
