# Trained models (ICFD-31k, Review 2)

```
outputs/models/
├── hinglish_transformer_ensemble/      FINAL MODEL (supervised) - used by src/fraud_detector.py
│   ├── finetuned_transformer_config.json      members, alert policies (default + strict), all selection results
│   ├── finetuned_transformer_metrics.json     team-protocol test metrics
│   ├── finetuned_transformer_deployment.json  call-level false alarms / detection / time-to-alert
│   ├── pca_heads/                             StandardScaler + PCA (<= 50) + LogisticRegression per member
│   ├── plots/                                 ROC/PR curves, training curves, alert trade-off
│   └── weights/seed1234|seed777|seed3407/     fine-tuned HingRoBERTa (fp16, ~557 MB each) - NOT in git
└── baselines_unsupervised/             the 4 team anomaly detectors (shared PCA-40 features)
    ├── deep_autoencoder/      (Hemanth)   .pth + metrics + curves + training loss
    ├── isolation_forest/      (Harshitha) .joblib + metrics + curves
    ├── mahalanobis_distance/  (Harsh)     .joblib + metrics + curves
    └── one_class_svm/         (Harshini)  .joblib + metrics + curves
```

## Results (official test split, team evaluation protocol)

| Model | Chunk ROC-AUC | Chunk PR-AUC | Chunk F1 | Chunk FPR | Call F1 |
|---|---|---|---|---|---|
| **HingRoBERTa ensemble (3 models) + PCA(<=50) + LR** | **0.936** | **0.964** | **0.894** | **0.226** | 0.980 |
| One-Class SVM | 0.524 | 0.635 | 0.774 | 0.950 | 0.993 |
| Deep Autoencoder | 0.512 | 0.637 | 0.768 | 0.940 | 0.989 |
| Mahalanobis Distance | 0.511 | 0.628 | 0.781 | 1.000 | 0.995 |
| Isolation Forest | 0.506 | 0.626 | 0.771 | 0.951 | 0.993 |

The unsupervised models reach F1 ~0.77 / call F1 ~0.99 only by flagging almost everything (FPR 0.94-1.0; the official
test split is 99 % scam calls), so judge models by ROC-AUC / PR-AUC and by the deployment numbers below.

Deployment (ensemble, strict policy, 1,014 normal + 4,499 scam test calls): **5.6 %** of normal calls false-alerted,
**97.0 %** of scams caught, median alert after 45 s; 93.6 % of unseen scam types caught. Latency per 3-second chunk:
~36 ms (GPU) / ~265 ms (CPU).

## Using the final model

```python
import sys; sys.path.append("src")
from fraud_detector import FraudDetector, CallSession
det = FraudDetector("outputs/models/hinglish_transformer_ensemble/finetuned_transformer_config.json")
call = CallSession(det, policy=det.config["alert_policy_strict"])
state = call.update(cumulative_text, audio_analysis_json, timestamp_seconds)   # every ~3 s -> state["level"]
```

**Weights are not in git** (over GitHub's 100 MB limit). On a fresh clone, download the private Kaggle dataset
`hemanthm2006/icfd31k-hinglish-fraud-detector` and copy its `finetuned_transformer/seed*` folders into
`hinglish_transformer_ensemble/weights/`.

**Library versions:** the PCA heads were saved with scikit-learn 1.6.1 (Kaggle). Use `scikit-learn==1.6.1` in the app
environment to avoid unpickling warnings or subtle differences; the transformer side needs `torch` and `transformers`.

Note: the notebooks write their outputs flat into `outputs/models/` when re-run; move new files into these folders.
