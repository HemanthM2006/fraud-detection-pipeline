"""Calibrate the zero-shot intent checker on VALIDATION calls and report on TEST calls (CPU).

Validation: 1,014 held-out normal-scenario calls + 400 validation scam calls  -> choose thresholds
Test:       1,014 held-out normal-scenario calls + 400 test scam calls + 300 cross-domain scam calls -> report once
Run from the repo root:  python scripts/calibrate_intents.py
Writes outputs/experiments/intent_checker_calibration.json
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from intent_checker import DEFAULT_MODEL, INTENTS, IntentChecker, split_sentences  # noqa: E402
from scam_rules import caller_lines, evaluate, level_of  # noqa: E402

DATA = ROOT / "data" / "icfd-31k" / "streaming_chunks"
OUT = ROOT / "outputs" / "experiments" / "intent_checker_calibration.json"
NORMAL = {"Clear Normal", "Ambiguous but Ultimately Normal"}
TARGET_FA = {"ALERT": 0.01, "REVIEW": 0.05}          # call-level false-alarm budget per intent (validation); all intents are REVIEW
N_SCAM, N_CROSS, SEED = 400, 300, 42
torch.set_num_threads(max(1, (os.cpu_count() or 4)))


def holdout_normals():
    tr = pq.read_table(sorted(DATA.glob("train-*.parquet")), columns=["conversation_uid", "case_type"]).to_pandas()
    tr["case"] = tr["case_type"].fillna("").astype(str).str.replace("_", " ").str.strip()
    uids = np.array(sorted(tr.conversation_uid.unique()))
    perm = np.random.default_rng(42).permutation(uids)
    head = set(perm[: int(round(0.10 * len(perm)))])
    normal = set(tr.loc[tr.case.isin(NORMAL), "conversation_uid"])
    rest = np.array(sorted(u for u in normal if u not in head))
    hold = np.random.default_rng(43).permutation(rest)[: int(round(0.15 * len(rest)))]
    return set(hold[: len(hold) // 2]), set(hold[len(hold) // 2:])   # same split as the notebook (val, test)


def load_calls(split, uids=None, n=None):
    cols = ["conversation_uid", "case_type", "chunk_timestamp", "final_verdict", "cumulative_text"]
    t = pq.read_table(sorted(DATA.glob(f"{split}-*.parquet")), columns=cols).to_pandas()
    if uids is not None:
        t = t[t.conversation_uid.isin(uids)]
    if n is not None:
        keep = np.random.default_rng(SEED).permutation(np.array(sorted(t.conversation_uid.unique())))[:n]
        t = t[t.conversation_uid.isin(set(keep))]
    calls = []
    for uid, g in t.sort_values(["conversation_uid", "chunk_timestamp"]).groupby("conversation_uid", sort=True):
        case = str(g.case_type.iloc[0]).replace("_", " ").strip()
        y = (g.final_verdict.fillna("NO").astype(str).str.strip().str.upper() == "YES").to_numpy()
        first_seen = {}                                   # k-th caller turn -> first timestamp it appears
        for ts, txt in zip(g.chunk_timestamp, g.cumulative_text):
            for k, _ in enumerate(caller_lines(txt)):
                first_seen.setdefault(k, float(ts))
        final_turns = caller_lines(g.cumulative_text.iloc[-1])
        calls.append({"uid": uid, "case": case, "scam": case not in NORMAL,
                      "flip": float(g.chunk_timestamp.to_numpy()[y.argmax()]) if y.any() else None,
                      "turns": [(final_turns[k], first_seen.get(k, float(g.chunk_timestamp.iloc[-1]))) for k in range(len(final_turns))],
                      "rule_hits_ts": first_rule_alert(g)})
    return calls


def first_rule_alert(g):
    for ts, txt in zip(g.chunk_timestamp, g.cumulative_text):
        if level_of(evaluate(txt)) == "ALERT":
            return float(ts)
    return None


def turn_score(S, turn, j):
    return max((S[x][j] for x in split_sentences(turn)), default=-9.0)


def call_hit(call, S, thr, j):
    times = [ts for t, ts in call["turns"] if turn_score(S, t, j) >= thr]
    return min(times) if times else None


def main():
    t0 = time.time()
    val_norm, test_norm = holdout_normals()
    sets = {"val_normal": load_calls("train", val_norm), "val_scam": load_calls("validation", n=N_SCAM),
            "test_normal": load_calls("train", test_norm), "test_scam": load_calls("test", n=N_SCAM),
            "cross_scam": load_calls("cross_domain", n=N_CROSS)}
    turns = sorted({x for calls in sets.values() for c in calls for t, _ in c["turns"] for x in split_sentences(t)})
    print({k: len(v) for k, v in sets.items()}, "| unique caller sentences:", len(turns), f"| data {time.time()-t0:.0f}s", flush=True)

    chk = IntentChecker(calibration_path=None, device="cpu")
    t1 = time.time()
    S = {}
    for s in range(0, len(turns), 4096):
        part = turns[s:s + 4096]
        for t, row in zip(part, chk.scores(part)):
            S[t] = row
        print(f"  scored {min(s + 4096, len(turns))}/{len(turns)} sentences ({time.time()-t1:.0f}s)", flush=True)

    grid = np.round(np.arange(-0.20, 0.801, 0.01), 3)
    thresholds, cal = {}, {}
    for j, (iid, level, _, _) in enumerate(INTENTS):
        best = None
        for thr in grid:
            fa = np.mean([call_hit(c, S, thr, j) is not None for c in sets["val_normal"]])
            if fa <= TARGET_FA[level]:
                best = float(thr)
                break
        best = best if best is not None else 0.8
        thresholds[iid] = best
        cal[iid] = {"level": level, "threshold": best, "target_val_false_alarm": TARGET_FA[level],
                    "val_false_alarm": round(float(np.mean([call_hit(c, S, best, j) is not None for c in sets["val_normal"]])), 4),
                    "val_scam_hit_rate": round(float(np.mean([call_hit(c, S, best, j) is not None for c in sets["val_scam"]])), 4)}
        print(f"{iid:<18} {level:<6} threshold {best:.3f} | val normal FA {cal[iid]['val_false_alarm']*100:.1f}% | val scam hits {cal[iid]['val_scam_hit_rate']*100:.1f}%", flush=True)

    def system_alert(c, use_rules=True, use_intents=True):
        times = []
        if use_rules and c["rule_hits_ts"] is not None:
            times.append(c["rule_hits_ts"])
        if use_intents:
            for j, (iid, level, _, _) in enumerate(INTENTS):
                if level in ("ALERT", "REVIEW"):          # report: any intent warning (all intents are REVIEW-level)
                    t = call_hit(c, S, thresholds[iid], j)
                    if t is not None:
                        times.append(t)
        return min(times) if times else None

    report = {}
    for name in ("test_normal", "test_scam", "cross_scam"):
        calls = sets[name]
        r = {}
        for label, kw in (("rules_only", dict(use_intents=False)), ("intents_only", dict(use_rules=False)), ("rules_plus_intents", {})):
            alerts = [system_alert(c, **kw) for c in calls]
            r[label] = round(float(np.mean([a is not None for a in alerts])), 4)
            if name != "test_normal":
                lead = [c["flip"] - a for c, a in zip(calls, alerts) if a is not None and c["flip"] is not None]
                r[label + "_median_alert_lead_vs_flip_s"] = float(np.median(lead)) if lead else None
        if name == "test_normal":
            for case in sorted(NORMAL):
                sub = [c for c in calls if c["case"] == case]
                r[f"rules_plus_intents_{case.split()[0].lower()}_normal"] = round(float(np.mean([system_alert(c) is not None for c in sub])), 4)
        report[name] = r
        print(name, r, flush=True)

    examples = ["Hello Can I get your bank account number please I'm a scam caller so can I get credit card details",
                "woh 6 digit wala number jo SMS mein aaya hai, mujhe bol do jaldi",
                "Sir aapke card ke peeche jo teen number likhe hain woh padh ke sunaiye",
                "Please install this app from the link I sent, then I can see your screen and fix it",
                "Sir aapka parcel kal deliver hoga, delivery boy ko OTP dikha dena",
                "Your order has been shipped and will arrive on Monday",
                "Agar abhi payment nahi kiya toh police aapke ghar aa jayegi"]
    chk.thresholds = thresholds
    ex = [{"text": e, "hits": [(h["rule"], h["score"]) for h in chk.check([e])]} for e in examples]
    for e in ex:
        print("  example:", e["text"][:70], "->", e["hits"], flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"model": DEFAULT_MODEL, "method": "max cosine similarity to intent examples minus max similarity to normal examples",
               "thresholds": thresholds, "calibration_validation": cal,
               "test_report": report, "examples": ex, "sizes": {k: len(v) for k, v in sets.items()},
               "note": "Thresholds chosen on validation calls only; call-level false alarm = any caller turn above threshold."},
              open(OUT, "w", encoding="utf-8"), indent=2)
    print(f"saved {OUT} | total {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
