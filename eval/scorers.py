"""Per-prompt-type scorers. Each returns a dict with at least {"score": float in [0,1]}."""
import re
from typing import Callable


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(pred: str, ref: str) -> float:
    pred, ref = _norm(pred), _norm(ref)
    if not ref:
        return 0.0 if not pred else 1.0
    return levenshtein(pred, ref) / len(ref)


def score_ocr(pred: dict, exp: dict) -> dict:
    def join(d):
        items = sorted(d.get("texts", []), key=lambda t: (t.get("frame", 0), _norm(t.get("text", ""))))
        return " ".join(t.get("text", "") for t in items)
    c = cer(join(pred), join(exp))
    return {"score": max(0.0, 1.0 - min(1.0, c)), "cer": c}


def _prf(pred: list, exp: list, key: str, frame_key: str, tol: int) -> dict:
    used = set()
    tp = 0
    for p in pred:
        for i, e in enumerate(exp):
            if i in used:
                continue
            if _norm(str(p.get(key, ""))) == _norm(str(e.get(key, ""))) and \
                    abs(int(p.get(frame_key, -10**9)) - int(e.get(frame_key, 0))) <= tol:
                used.add(i)
                tp += 1
                break
    if not pred and not exp:
        return {"score": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0}
    prec = tp / len(pred) if pred else 0.0
    rec = tp / len(exp) if exp else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"score": f1, "precision": prec, "recall": rec, "f1": f1}


def score_safety(pred: dict, exp: dict, frame_tol: int = 1) -> dict:
    return _prf(pred.get("violations", []), exp.get("violations", []), "type", "frame", frame_tol)


def score_actions(pred: dict, exp: dict, frame_tol: int = 2) -> dict:
    return _prf(pred.get("actions", []), exp.get("actions", []), "action", "start_frame", frame_tol)


def score_anomaly(pred: dict, exp: dict) -> dict:
    return {"score": float(bool(pred.get("anomaly")) == bool(exp.get("anomaly")))}


# Optional LLM judge: callable(pred_summary, expected_dict) -> float in [0,1]. Not wired by default.
JudgeFn = Callable[[str, dict], float]


def score_summary(pred: dict, exp: dict, judge: JudgeFn | None = None) -> dict:
    text = _norm(pred.get("summary", ""))
    kws = [_norm(k) for k in exp.get("keywords", [])]
    recall = sum(k in text for k in kws) / len(kws) if kws else 1.0
    out = {"score": recall, "keyword_recall": recall}
    if judge is not None:
        j = float(judge(pred.get("summary", ""), exp))
        out["judge"] = j
        out["score"] = (recall + j) / 2
    return out


SCORERS = {"ocr": score_ocr, "safety": score_safety, "actions": score_actions,
           "anomaly": score_anomaly, "summary": score_summary}


def score(prompt_id: str, pred, exp: dict) -> dict:
    if not isinstance(pred, dict):
        return {"score": 0.0, "error": "no valid json output"}
    fn = SCORERS.get(prompt_id)
    if fn is None:
        raise KeyError(f"no scorer for prompt id {prompt_id!r}")
    return fn(pred, exp)
