# fraud_detector.py - real-time Hinglish phone-scam detector
# Fine-tuned HingRoBERTa features (768-d) + audio/time features -> StandardScaler -> PCA (<= 50) -> Logistic Regression,
# plus a call-level alert policy (EMA smoothing, k consecutive high-risk chunks, latched ALERT, REVIEW band).
#
# Usage:
#   from fraud_detector import FraudDetector, CallSession
#   det = FraudDetector("outputs/models/finetuned_transformer_config.json")   # loads every ensemble member listed in the config
#   call = CallSession(det)
#   state = call.update(cumulative_text, audio_analysis_json, timestamp_seconds)   # every ~3 s
#   state["level"] -> "NORMAL" | "REVIEW" | "ALERT"; state["score"] -> P(fraud) of this chunk
import json
import math
import re
from collections import deque

import joblib
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PACE_MAP = {"slow": 0.0, "medium": 1.0, "fast": 2.0}


def _is_missing(v):
    return v is None or (isinstance(v, float) and math.isnan(v))


def _parse_audio(val):
    if isinstance(val, dict):
        return val
    if _is_missing(val):
        return {}
    try:
        out = json.loads(str(val))
        return out if isinstance(out, dict) else {}
    except (ValueError, TypeError):
        return {}


def _clean_text(text):
    return "" if _is_missing(text) else re.sub(r"[ \t]+", " ", str(text)).strip()


def _emotion(v):
    if _is_missing(v):
        return "Unknown"
    s = str(v).strip()
    return s if s else "Unknown"


def _number(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return x


class FraudDetector:
    # Ensemble: P(fraud) = mean over members of each member's PCA(<=50)+LR probability.
    def __init__(self, config_path, device=None, batch_size=32):
        import os
        with open(config_path, encoding="utf-8") as f:
            self.config = json.load(f)
        base = os.path.dirname(os.path.abspath(config_path))
        self.members = [_Member(os.path.join(base, m["weights_dir"]), os.path.join(base, m["head"]), self.config, device, batch_size)
                        for m in self.config["members"]]
        self.device = self.members[0].device

    def score(self, texts, audio_jsons, timestamps):
        return np.mean([m.score(texts, audio_jsons, timestamps) for m in self.members], axis=0)


class _Member:
    def __init__(self, model_dir, head_path, config, device=None, batch_size=32):
        self.config = config
        self.head = joblib.load(head_path)
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        model = model.half() if self.device.startswith("cuda") else model.float()
        self.model = model.to(self.device).eval()
        cls = self.model.classifier
        self._final = cls.out_proj if hasattr(cls, "out_proj") else cls
        probe = "namaste hello world"
        with_sp, without = self.tok(probe)["input_ids"], self.tok(probe, add_special_tokens=False)["input_ids"]
        for i in range(len(with_sp) - len(without) + 1):
            if with_sp[i:i + len(without)] == without:
                self._prefix, self._suffix = with_sp[:i], with_sp[i + len(without):]
                break
        else:
            raise ValueError("Could not infer the special-token template")
        self.max_len, self.head_tokens = int(self.config["max_len"]), int(self.config["head_tokens"])

    def _encode(self, texts):
        body = self.max_len - len(self._prefix) - len(self._suffix)
        out = []
        for ids in self.tok(list(texts), add_special_tokens=False, truncation=False)["input_ids"]:
            if len(ids) > body:
                ids = ids[:self.head_tokens] + ids[-(body - self.head_tokens):]
            out.append(self._prefix + list(ids) + self._suffix)
        return out

    @torch.no_grad()
    def text_features(self, texts):
        ids = self._encode(texts)
        feats = np.zeros((len(ids), self._final.in_features), dtype=np.float32)
        cap = {}
        hook = self._final.register_forward_hook(lambda mod, inp, out: cap.__setitem__("x", inp[0]))
        try:
            order = np.argsort([len(x) for x in ids], kind="stable")
            for s in range(0, len(order), self.batch_size):
                b = order[s:s + self.batch_size]
                enc = self.tok.pad({"input_ids": [ids[j] for j in b]}, return_tensors="pt")
                self.model(input_ids=enc["input_ids"].to(self.device), attention_mask=enc["attention_mask"].to(self.device))
                feats[b] = cap["x"].float().cpu().numpy()
        finally:
            hook.remove()
        return feats

    def features(self, texts, audio_jsons, timestamps):
        h = self.head
        texts = [_clean_text(t) for t in texts]
        audio = [_parse_audio(a) for a in audio_jsons]
        dom = [_emotion(a.get("dominant_emotion")) for a in audio]
        sec = [_emotion(a.get("secondary_emotion")) for a in audio]
        emo = pd.DataFrame({"dominant_emotion": [d if d in h["top_dominant_emotions"] else "Other" for d in dom],
                            "secondary_emotion": [d if d in h["top_secondary_emotions"] else "Other" for d in sec]})
        pace = [PACE_MAP.get(str(a.get("pace")).strip().lower(), float("nan")) for a in audio]
        conf = [_number(a.get("confidence_score")) for a in audio]
        conf = [min(max(c, 0.0), 1.0) if not math.isnan(c) else float("nan") for c in conf]
        ts = [_number(t) for t in timestamps]
        num = np.array([[h["pace_median"] if math.isnan(p) else p, h["conf_median"] if math.isnan(c) else c,
                         0.0 if math.isnan(t) else max(t, 0.0)] for p, c, t in zip(pace, conf, ts)], dtype=np.float32)
        return np.hstack([self.text_features(texts), h["ohe"].transform(emo), num]).astype(np.float32)

    def score(self, texts, audio_jsons, timestamps):
        h = self.head
        xs = h["scaler"].transform(self.features(texts, audio_jsons, timestamps)).astype(np.float32)
        return h["classifier"].predict_proba(h["pca"].transform(xs))[:, 1]


class CallSession:
    def __init__(self, detector, policy=None):
        p = dict(policy or detector.config["alert_policy"])
        self.detector = detector
        self.alpha, self.k = float(p["alpha"]), int(p["k"])
        self.tau_alert, self.tau_review = float(p["tau_alert"]), float(p["tau_review"])
        self.reset()

    def reset(self):
        self.ema, self.window = None, deque(maxlen=self.k)
        self.alerted, self.first_alert_ts, self.history = False, None, []

    def update_score(self, score, timestamp):
        score = float(score)
        prev = self.ema
        self.ema = score if prev is None else self.alpha * score + (1.0 - self.alpha) * prev
        self.window.append(self.ema)
        if not self.alerted and len(self.window) == self.k and min(self.window) >= self.tau_alert:
            self.alerted, self.first_alert_ts = True, float(timestamp)
        level = "ALERT" if self.alerted else ("REVIEW" if self.ema >= self.tau_review else "NORMAL")
        state = {"timestamp": float(timestamp), "score": score, "smoothed_score": self.ema,
                 "risk_velocity": 0.0 if prev is None else self.ema - prev, "level": level,
                 "alerted": self.alerted, "first_alert_ts": self.first_alert_ts}
        self.history.append(state)
        return state

    def update(self, cumulative_text, audio_json, timestamp):
        return self.update_score(self.detector.score([cumulative_text], [audio_json], [timestamp])[0], timestamp)
