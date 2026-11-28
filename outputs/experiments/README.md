# Experiments (not used by the live app)

## Meaning-based intent check (`src/intent_checker.py`, `scripts/calibrate_intents.py`)
Goal: catch scam requests phrased in ways the keyword rules miss (e.g. "woh 6 digit wala number jo SMS mein aaya, bol do").

1. **Zero-shot NLI** (`MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`) - rejected after spot checks: on Hinglish
   it scored "Your order has been shipped" (0.19) above real OTP / CVV requests (0.04-0.18), and ran at ~29 pairs/s on CPU.
2. **Sentence-embedding similarity** (`paraphrase-multilingual-MiniLM-L12-v2`, caller sentence vs. example scam requests
   minus vs. example normal phrases). Calibrated on validation (1,014 normal + 400 scam calls), reported on test
   (1,014 normal + 400 scam + 300 cross-domain calls):
   - as **ALERT** trigger (~1 % FA per intent on validation): rules + intents alerted on 3.3 % of normal test calls vs 0.9 %
     for rules alone, catching 25.5 % vs 20.0 % of test scam calls - the main model already catches ~97 %.
   - as **REVIEW** warning (~5 % FA per intent): warned on 22.5 % of normal calls and still missed the keyword-free
     paraphrases above.
   Decision: not enabled in the app (opt-in with `ENABLE_INTENT_CHECK=1`). Results: `intent_checker_calibration.json`.

Next step that would actually address unusual phrasing: fine-tune a turn-level classifier (labels from the chunk where
each scam call turns fraudulent) with paraphrase / short-utterance augmentation.
