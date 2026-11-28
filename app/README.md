# Scam Call Guard — live demo

A simple web app around the final model: speak (or type, or replay a real held-out call) and get a live
**NORMAL / REVIEW / ALERT** verdict every 3 seconds, with the reasons.

- **Speech-to-text:** the browser's built-in Web Speech API (online, Google-backed in Chrome/Edge, `en-IN`). No API key.
  The audio is sent to the browser vendor's speech service for transcription.
- **Model:** the 3-model fine-tuned HingRoBERTa ensemble (`src/fraud_detector.py`) on the **CPU**
  (~0.15–0.4 s per 3-second chunk).
- **Safety rules** (`src/scam_rules.py`): transparent IF-THEN rules on the caller's words. A request for an
  OTP / PIN / CVV / password / card number, a remote-access app (AnyDesk, TeamViewer) or gift cards / crypto
  raises **ALERT** immediately; requests for account / Aadhaar / PAN details, payment demands and threats raise
  **REVIEW**. Rules can only raise the level. Negations ("never share your OTP", "OTP mat batana"),
  postal "pin code", delivery-person OTPs and in-app login instructions are excluded.
  On 1,014 held-out *normal* calls the ALERT rules fire on 0.9 %.

## Run

```bash
pip install fastapi "uvicorn[standard]" transformers safetensors torch scikit-learn pandas pyarrow joblib
python -m uvicorn app.server:app --port 8000
```
Open http://localhost:8000 in **Chrome or Edge**, wait for "Model ready" (~20 s), then:
- **Live mic** – press *Start call*, toggle *Caller speaking* / *I'm speaking* as the conversation goes.
- **Type lines** – add lines as caller or as you; each line is the next 3-second chunk.
- **Replay real call** – held-out ICFD-31k calls (from your local `data/icfd-31k`, never used in training),
  replayed 5× faster, with the ground truth shown at the end.

Needs the model weights in `outputs/models/hinglish_transformer_ensemble/weights/seed{1234,777,3407}/`
(private Kaggle dataset `hemanthm2006/icfd31k-hinglish-fraud-detector`). The PCA heads were saved with
scikit-learn 1.6.1; newer versions load them with a warning.

## Measured on held-out calls
- Replaying 12 held-out calls through the app (strict policy, model + rules): 10/12 correct - 5/6 normal calls with no
  alert, 5/6 scam calls alerted (incl. all 3 unseen scam types). One ambiguous-normal call got a false ALERT; one short
  scam call only reached REVIEW.
- A meaning-based "intent check" was tried and is **off** by default (see `outputs/experiments/README.md`).

## Known limits
- The ML model learned from synthetic, scripted Hinglish call transcripts. Unusual phrasing (e.g. one blunt English
  sentence) can look "normal" to it — the rules cover the most dangerous requests in that case.
- Live audio features (emotion, pace, confidence) are approximated in the browser; emotion is unknown.
- Research prototype: it flags suspicious behaviour, it is not proof of fraud.
