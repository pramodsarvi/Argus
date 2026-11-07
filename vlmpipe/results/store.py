"""Idempotent result store. Conflict target is `rkey` = ResultRecord.key_str(), which covers
both plain video keys and camera keys. Existing SQLite DBs (pre-camera schema) are migrated
in place: missing columns added, rkey backfilled for old rows, unique index created."""
import json
import sqlite3

from .models import ResultRecord

_COLS = ("video_id", "segment_idx", "prompt_id", "prompt_version", "model", "ok", "output",
         "segment_start_s", "frame_times", "error", "created_at",
         "camera_id", "chunk_start_ts", "prompt_set", "prompt_set_version", "status")
_NEW_COLS = {"rkey": "TEXT", "camera_id": "TEXT", "chunk_start_ts": "DOUBLE PRECISION",
             "prompt_set": "TEXT", "prompt_set_version": "INTEGER", "status": "TEXT"}
_DDL = """CREATE TABLE IF NOT EXISTS vlm_results (
  rkey TEXT PRIMARY KEY,
  video_id TEXT NOT NULL, segment_idx INTEGER NOT NULL, prompt_id TEXT NOT NULL,
  prompt_version INTEGER NOT NULL, model TEXT NOT NULL, ok BOOLEAN NOT NULL, output TEXT,
  segment_start_s DOUBLE PRECISION, frame_times TEXT, error TEXT, created_at DOUBLE PRECISION,
  camera_id TEXT, chunk_start_ts DOUBLE PRECISION, prompt_set TEXT, prompt_set_version INTEGER,
  status TEXT)"""
_INDEXES = (
    "CREATE UNIQUE INDEX IF NOT EXISTS vlm_results_rkey ON vlm_results (rkey)",
    "CREATE INDEX IF NOT EXISTS vlm_results_camera_ts ON vlm_results (camera_id, chunk_start_ts)",
    "CREATE INDEX IF NOT EXISTS vlm_results_video ON vlm_results (video_id)",
)
_UPDATE = ", ".join(f"{c}=excluded.{c}" for c in _COLS)


def _row(r: ResultRecord) -> tuple:
    return (r.key_str(), r.video_id, r.segment_idx, r.prompt_id, r.prompt_version, r.model,
            bool(r.ok), json.dumps(r.output), r.segment_start_s, json.dumps(r.frame_times),
            r.error, r.created_at, r.camera_id, r.chunk_start_ts, r.prompt_set,
            r.prompt_set_version, r.status)


def _rec(row) -> ResultRecord:
    d = dict(zip(_COLS, row))
    d["ok"] = bool(d["ok"])
    d["output"] = json.loads(d["output"]) if d["output"] is not None else None
    d["frame_times"] = json.loads(d["frame_times"] or "[]")
    return ResultRecord(**d)


class _SQLStore:
    ph = "?"

    def _upsert_sql(self) -> str:
        cols = ("rkey",) + _COLS
        vals = ", ".join([self.ph] * len(cols))
        return (f"INSERT INTO vlm_results ({', '.join(cols)}) VALUES ({vals}) "
                f"ON CONFLICT (rkey) DO UPDATE SET {_UPDATE}")

    def _select_sql(self) -> str:
        return (f"SELECT {', '.join(_COLS)} FROM vlm_results WHERE video_id={self.ph} "
                f"ORDER BY segment_idx, prompt_id")

    def _camera_sql(self, with_range: bool) -> str:
        q = f"SELECT {', '.join(_COLS)} FROM vlm_results WHERE camera_id={self.ph}"
        if with_range:
            q += f" AND chunk_start_ts >= {self.ph} AND chunk_start_ts < {self.ph}"
        return q + " ORDER BY chunk_start_ts, segment_idx, prompt_id"


class SQLiteStore(_SQLStore):
    def __init__(self, path: str = ":memory:") -> None:
        self.conn = sqlite3.connect(path)
        self._migrate()

    def _migrate(self) -> None:
        c = self.conn
        c.execute(_DDL)
        have = {r[1] for r in c.execute("PRAGMA table_info(vlm_results)")}
        for col, typ in _NEW_COLS.items():
            if col not in have:
                c.execute(f"ALTER TABLE vlm_results ADD COLUMN {col} {typ}")
        if "rkey" not in have:   # old table: backfill keys exactly as ResultRecord.key_str()
            c.execute("UPDATE vlm_results SET rkey = video_id || '|' || segment_idx || '|' || "
                      "prompt_id || '|' || prompt_version || '|' || model WHERE rkey IS NULL")
        for ddl in _INDEXES:
            c.execute(ddl)
        c.commit()

    def upsert(self, rec: ResultRecord) -> None:
        self.conn.execute(self._upsert_sql(), _row(rec))
        self.conn.commit()

    def get_video(self, video_id: str) -> list[ResultRecord]:
        return [_rec(r) for r in self.conn.execute(self._select_sql(), (video_id,))]

    def get_camera(self, camera_id: str, since: float | None = None,
                   until: float | None = None) -> list[ResultRecord]:
        rng = since is not None or until is not None
        args = (camera_id,) + ((since if since is not None else float("-inf"),
                                until if until is not None else float("inf")) if rng else ())
        return [_rec(r) for r in self.conn.execute(self._camera_sql(rng), args)]


class PostgresStore(_SQLStore):
    ph = "%s"

    def __init__(self, dsn: str) -> None:
        import psycopg
        self.conn = psycopg.connect(dsn, autocommit=True)
        self.conn.execute(_DDL)
        for col, typ in _NEW_COLS.items():
            self.conn.execute(f"ALTER TABLE vlm_results ADD COLUMN IF NOT EXISTS {col} {typ}")
        self.conn.execute("UPDATE vlm_results SET rkey = concat_ws('|', video_id, segment_idx, "
                          "prompt_id, prompt_version, model) WHERE rkey IS NULL")
        for ddl in _INDEXES:
            self.conn.execute(ddl)

    def upsert(self, rec: ResultRecord) -> None:
        self.conn.execute(self._upsert_sql(), _row(rec))

    def get_video(self, video_id: str) -> list[ResultRecord]:
        return [_rec(r) for r in self.conn.execute(self._select_sql(), (video_id,)).fetchall()]

    def get_camera(self, camera_id: str, since: float | None = None,
                   until: float | None = None) -> list[ResultRecord]:
        rng = since is not None or until is not None
        args = (camera_id,) + ((since if since is not None else float("-inf"),
                                until if until is not None else float("inf")) if rng else ())
        return [_rec(r) for r in self.conn.execute(self._camera_sql(rng), args).fetchall()]
