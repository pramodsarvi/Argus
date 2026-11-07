import hashlib
import json
import time

import httpx


class WebhookDispatcher:
    """POST alerts with exponential-backoff retry. Retries on network errors, 429 and 5xx;
    4xx (other than 429) is permanent. Sends an Idempotency-Key so receivers can dedupe."""

    def __init__(self, url: str, max_retries: int = 5, backoff_s: float = 0.5,
                 timeout_s: float = 10.0, client: httpx.Client | None = None,
                 sleep=time.sleep) -> None:
        self.url, self.max_retries, self.backoff_s = url, max_retries, backoff_s
        self.client = client or httpx.Client(timeout=timeout_s)
        self.sleep = sleep

    @staticmethod
    def idem_key(video_id: str, alert: dict) -> str:
        # segment alerts carry segment_idx; camera-rule alerts carry chunk_start_ts
        where = alert.get("segment_idx", alert.get("chunk_start_ts"))
        raw = f"{video_id}|{alert['rule_id']}|{where}"
        return hashlib.sha256(raw.encode()).hexdigest()[:32]

    def send(self, video_id: str, alert: dict) -> bool:
        body = json.dumps({"video_id": video_id, **alert})
        headers = {"Content-Type": "application/json",
                   "Idempotency-Key": self.idem_key(video_id, alert)}
        for attempt in range(self.max_retries + 1):
            try:
                r = self.client.post(self.url, content=body, headers=headers)
                if r.status_code < 300:
                    return True
                if r.status_code != 429 and r.status_code < 500:
                    return False
            except httpx.HTTPError:
                pass
            if attempt < self.max_retries:
                self.sleep(self.backoff_s * 2 ** attempt)
        return False

    def send_all(self, reduced: dict) -> dict[str, bool]:
        return {self.idem_key(reduced["video_id"], a): self.send(reduced["video_id"], a)
                for a in reduced["alerts"]}
