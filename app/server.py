"""Live demo server: Hinglish phone-scam detector (fine-tuned HingRoBERTa ensemble).

Run from the repo root:
    python -m uvicorn app.server:app --port 8000
then open http://localhost:8000 in Chrome or Edge (needed for the browser speech-to-text).

The model runs on the CPU (~0.3 s per 3-second chunk for the 3-model ensemble).
"""
import json
import os
import sys
import threading
import time
import uuid
import warnings
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")          # CPU only: never use the laptop GPU
warnings.filterwarnings("ignore", message=".*InconsistentVersionWarning.*")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fraud_detector import CallSession, FraudDetector  # noqa: E402

CONFIG = ROOT / "outputs" / "models" / "hinglish_transformer_ensemble" / "finetuned_transformer_config.json"
DATA = ROOT / "data" / "icfd-31k" / "streaming_chunks"
STATIC = Path(__file__).resolve().parent / "static"
NORMAL_CASES = {"Clear Normal", "Ambiguous but Ultimately Normal"}

# Experimental meaning-based intent check: OFF by default (measured: warned on 22.5 % of normal calls and still
# missed keyword-free paraphrases, see outputs/experiments/). Enable with environment variable ENABLE_INTENT_CHECK=1.
INTENTS_CAL = ROOT / "outputs" / "experiments" / "intent_checker_calibration.json"
INTENTS_ON = os.environ.get("ENABLE_INTENT_CHECK") == "1"
STATE = {"detector": None, "intents": None, "error": None, "loaded_at": None, "load_seconds": None}
SESSIONS = {}
LOCK = threading.Lock()   # one inference at a time (CPU)


def load_detector():
    t0 = time.time()
    try:
        STATE["detector"] = FraudDetector(str(CONFIG), device="cpu")
        if INTENTS_ON and INTENTS_CAL.exists():                    # optional experimental intent checker
            from intent_checker import IntentChecker
            STATE["intents"] = IntentChecker(str(INTENTS_CAL), device="cpu")
        STATE["load_seconds"] = round(time.time() - t0, 1)
        STATE["loaded_at"] = time.strftime("%H:%M:%S")
    except Exception as e:  # surfaced through /api/health
        STATE["error"] = f"{type(e).__name__}: {e}"


@asynccontextmanager
async def lifespan(_app):
    threading.Thread(target=load_detector, daemon=True).start()
    yield


app = FastAPI(title="Hinglish Scam-Call Detector", lifespan=lifespan)


def detector():
    if STATE["error"]:
        raise HTTPException(500, STATE["error"])
    if STATE["detector"] is None:
        raise HTTPException(503, "Model is still loading, try again in a few seconds")
    return STATE["detector"]


def policy_for(name):
    cfg = detector().config
    return cfg["alert_policy_strict"] if name == "strict" else cfg["alert_policy"]


# ------------------------------------------------------------------ live sessions
class NewSession(BaseModel):
    policy: str = "strict"


class Update(BaseModel):
    transcript: str
    timestamp: float
    audio: Optional[dict] = None


@app.get("/api/health")
def health():
    d = STATE["detector"]
    return {"ready": d is not None, "error": STATE["error"], "device": getattr(d, "device", None),
            "members": [m["seed"] for m in d.config["members"]] if d else None, "intent_checker": STATE["intents"] is not None,
            "load_seconds": STATE["load_seconds"], "samples_available": DATA.exists()}


@app.post("/api/sessions")
def new_session(req: NewSession):
    pol = policy_for(req.policy)
    sid = uuid.uuid4().hex[:12]
    SESSIONS[sid] = {"session": CallSession(detector(), policy=pol, intent_checker=STATE["intents"]), "policy": pol}
    if len(SESSIONS) > 50:                      # keep memory bounded
        SESSIONS.pop(next(iter(SESSIONS)))
    return {"session_id": sid, "policy": req.policy, "tau_alert": pol["tau_alert"], "tau_review": pol["tau_review"]}


@app.post("/api/sessions/{sid}/update")
def update(sid: str, req: Update):
    s = SESSIONS.get(sid)
    if s is None:
        raise HTTPException(404, "Unknown session")
    t0 = time.perf_counter()
    with LOCK:
        st = s["session"].update(req.transcript, json.dumps(req.audio or {}), req.timestamp)
    pol = s["policy"]
    return {**st, "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            "risk_index": st["smoothed_score"] / pol["tau_alert"], "review_index": pol["tau_review"] / pol["tau_alert"]}


# ------------------------------------------------------------------ replay of held-out calls (from the local dataset only)
SAMPLES = {}


def _split_holdout_normals(train_meta):
    # Re-creates the notebook's held-out normal test calls (never used for training or fitting).
    uids = np.array(sorted(train_meta["conversation_uid"].unique()))
    perm = np.random.default_rng(42).permutation(uids)
    head = set(perm[: int(round(0.10 * len(perm)))])
    normal = set(train_meta.loc[train_meta["case"].isin(NORMAL_CASES), "conversation_uid"])
    rest = np.array(sorted(u for u in normal if u not in head))
    hold = np.random.default_rng(43).permutation(rest)[: int(round(0.15 * len(rest)))]
    return set(hold[len(hold) // 2:])        # second half = test


def _read(files, uids=None, cols=None):
    df = pq.read_table(files, columns=cols).to_pandas()
    if uids is not None:
        df = df[df["conversation_uid"].isin(uids)]
    return df


def build_samples():
    if SAMPLES or not DATA.exists():
        return
    meta_cols = ["conversation_uid", "case_type"]
    tr = _read(sorted(DATA.glob("train-*.parquet")), cols=meta_cols)
    tr["case"] = tr["case_type"].fillna("").astype(str).str.replace("_", " ").str.strip()
    test_normals = _split_holdout_normals(tr)
    calls = tr[tr["conversation_uid"].isin(test_normals)].groupby("conversation_uid")["case"].first()
    picks = [(u, "train") for u in sorted(calls[calls == "Clear Normal"].index)[:3]]
    picks += [(u, "train") for u in sorted(calls[calls == "Ambiguous but Ultimately Normal"].index)[:3]]
    te = _read(sorted(DATA.glob("test-*.parquet")), cols=meta_cols)
    tcase = te.groupby("conversation_uid")["case_type"].first().str.replace("_", " ")
    picks += [(u, "test") for u in sorted(tcase[tcase == "Clear Fraud"].index)[:3]]
    cd = sorted(DATA.glob("cross_domain-*.parquet"))
    if cd:
        cdm = _read(cd, cols=meta_cols)
        ccase = cdm.groupby("conversation_uid")["case_type"].first().str.replace("_", " ")
        picks += [(u, "cross_domain") for u in sorted(ccase.index)[:3]]
    cols = ["conversation_uid", "domain", "case_type", "chunk_timestamp", "final_verdict",
            "cumulative_text", "cumulative_audio_analysis_json"]
    for split in ("train", "test", "cross_domain"):
        want = {u for u, s in picks if s == split}
        if not want:
            continue
        df = _read(sorted(DATA.glob(f"{split}-*.parquet")), want, cols).sort_values(["conversation_uid", "chunk_timestamp"])
        for uid, g in df.groupby("conversation_uid", sort=True):
            case = str(g["case_type"].iloc[0]).replace("_", " ")
            y = (g["final_verdict"].fillna("NO").astype(str).str.strip().str.upper() == "YES").to_numpy()
            flip = float(g["chunk_timestamp"].to_numpy()[y.argmax()]) if y.any() else None
            sid = f"s{len(SAMPLES) + 1:02d}"
            SAMPLES[sid] = {
                "id": sid, "case_type": case, "is_scam_scenario": case not in NORMAL_CASES,
                "source": {"train": "held-out normal call (not used in training)", "test": "official test split",
                           "cross_domain": "unseen scam type (cross-domain)"}[split],
                "domain": str(g["domain"].iloc[0]), "n_chunks": len(g), "duration_s": float(g["chunk_timestamp"].max()),
                "label_flip_s": flip,
                "chunks": [{"timestamp": float(t), "transcript": str(x or ""), "audio": a}
                           for t, x, a in zip(g["chunk_timestamp"], g["cumulative_text"], g["cumulative_audio_analysis_json"])]}


@app.get("/api/samples")
def samples():
    build_samples()
    return [{k: v for k, v in s.items() if k != "chunks"} for s in SAMPLES.values()]


@app.get("/api/samples/{sid}")
def sample(sid: str):
    build_samples()
    if sid not in SAMPLES:
        raise HTTPException(404, "Unknown sample")
    s = SAMPLES[sid]
    chunks = [{**c, "audio": json.loads(c["audio"]) if isinstance(c["audio"], str) and c["audio"].strip().startswith("{") else {}}
              for c in s["chunks"]]
    return {**{k: v for k, v in s.items() if k != "chunks"}, "chunks": chunks}


# ------------------------------------------------------------------ page
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")
