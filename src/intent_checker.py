# intent_checker.py - meaning-based "what is the caller asking for?" check on each new caller sentence.
# A multilingual sentence-embedding model (paraphrase-multilingual-MiniLM-L12-v2, the same family the team
# pipeline uses) compares each caller sentence with example phrases of scam requests (English + Hinglish) and
# with example phrases of normal calls. score = best similarity to the intent's examples - best similarity to
# the normal examples. It understands paraphrases ("woh 6 digit wala number jo SMS mein aaya, bol do") that
# keyword rules miss. Thresholds are calibrated on validation calls (scripts/calibrate_intents.py).
# All intents raise REVIEW (a visible warning), never ALERT: as an ALERT trigger it added ~2.5 points of false
# alarms on normal calls for ~5 points more scam catches, while the main model already catches ~97 % of scams.
# (A zero-shot NLI model was tried first and rejected: on Hinglish it ranked normal sentences above OTP requests.)
import json
import os
import re

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

INTENTS = [
    # (id, level, reason shown to the user, example phrases)
    ("I1_secret_intent", "REVIEW", "AI intent check: the caller seems to ask for a secret code or credential (OTP / PIN / password / card)", [
        "please share the OTP you received", "tell me the OTP", "OTP batao", "OTP bata dijiye sir", "jo code aaya hai woh bata do",
        "SMS mein jo number aaya hai woh bolo", "read out the 6 digit code you got", "aapke phone pe aaya verification code bataiye",
        "give me your card number and CVV", "card ke peeche ke teen number bataiye", "tell me the expiry date and CVV of your card",
        "debit card ki details bata dijiye", "can I get your credit card details", "tell me your ATM PIN", "apna UPI PIN daalo",
        "UPI PIN enter karke confirm kijiye", "what is your net banking password", "apna login ID aur password bataiye",
        "forward the message you just received to me", "aapke mobile pe aaya message mujhe bhej dijiye"]),
    ("I2_remote_intent", "REVIEW", "AI intent check: the caller seems to ask you to install an app or give remote access", [
        "install AnyDesk so I can help you", "download TeamViewer from the play store", "screen share kijiye", "share your screen with me",
        "please install this app from the link I sent", "link pe click karke app download kariye", "give me remote access to your phone",
        "yeh app install karo main sab theek kar dunga", "open the link and allow all permissions", "QuickSupport app download kijiye"]),
    ("I3_unusual_payment_intent", "REVIEW", "AI intent check: the caller seems to ask for gift cards, crypto or a 'safe account' transfer", [
        "buy a Google Play gift card and send me the code", "gift card kharid ke code bhejo", "send the money in bitcoin",
        "transfer all your money to this safe account", "paise safe account mein transfer kar dijiye", "pay through crypto to this wallet",
        "Amazon gift card ka code share kariye", "scan this QR code to receive the refund"]),
    ("I4_money_intent", "REVIEW", "AI intent check: the caller seems to ask for money or a payment", [
        "pay the processing fee now", "processing fee ke liye 500 rupees bhejiye", "transfer the amount immediately",
        "registration fee pay karna hoga", "paise abhi bhejo", "deposit the security amount to release your prize",
        "you need to pay the customs charges", "refund ke liye pehle chhota payment karna hoga", "pay the fine to avoid arrest"]),
    ("I5_threat_intent", "REVIEW", "AI intent check: the caller seems to threaten you (block, police, legal action)", [
        "your account will be blocked today", "aapka account block ho jayega", "police case file ho jayega", "there is an arrest warrant against you",
        "we will take legal action", "aapka number band ho jayega", "your parcel contains drugs, customs has seized it", "CBI officer bol raha hoon",
        "fine dena padega warna jail", "your SIM will be deactivated in two hours"]),
    ("I6_details_intent", "REVIEW", "AI intent check: the caller seems to ask for personal or account details", [
        "tell me your Aadhaar number", "PAN card number bataiye", "what is your date of birth", "apna account number bataiye",
        "confirm your full name and address", "mother's maiden name kya hai", "share your bank account details"]),
]
NORMAL_EXAMPLES = [
    "your order will be delivered tomorrow", "how can I help you today", "thank you for calling", "your appointment is confirmed",
    "please visit the branch with your documents", "never share your OTP with anyone", "OTP kisi ko mat batana",
    "share the OTP with the delivery person at the door", "delivery boy ko OTP dikha dena", "your bill amount is due next week",
    "the bank will never ask for your PIN", "please log in to the app to see your statement", "you will receive an OTP, enter it in the app yourself",
    "aapka order kal tak aa jayega", "main aapki kya madad kar sakta hoon", "aapka complaint register ho gaya hai",
    "your new cheque book has been dispatched", "is this a good time to talk", "we are calling to confirm your reservation",
    "please rate our service", "the interest rate on your loan is eight percent", "your policy renewal is due next month",
    "hello sir, how are you", "haan ji bataiye", "okay, thank you, have a nice day", "kya aap baad mein call kar sakte hain",
]


def split_sentences(text):
    parts = re.split(r"(?<=[.!?|])\s+", str(text or "").strip())
    return [p.strip() for p in parts if len(p.strip()) >= 3]


class IntentChecker:
    def __init__(self, calibration_path=None, model_name=DEFAULT_MODEL, device="cpu", batch_size=64):
        self.cal = {}
        if calibration_path and os.path.exists(calibration_path):
            with open(calibration_path, encoding="utf-8") as f:
                self.cal = json.load(f)
            model_name = self.cal.get("model", model_name)
        self.thresholds = self.cal.get("thresholds", {})
        self.device, self.batch_size = device, batch_size
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).float().to(device).eval()
        self.protos = [self.embed(ex) for (_, _, _, ex) in INTENTS]
        self.normal = self.embed(NORMAL_EXAMPLES)

    @torch.no_grad()
    def embed(self, texts):
        out = []
        for s in range(0, len(texts), self.batch_size):
            enc = self.tok(list(texts[s:s + self.batch_size]), padding=True, truncation=True, max_length=128, return_tensors="pt").to(self.device)
            h = self.model(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).float()
            v = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
            out.append(torch.nn.functional.normalize(v, dim=-1).cpu().numpy())
        return np.vstack(out) if out else np.zeros((0, 384), dtype=np.float32)

    def scores(self, sentences):
        """Margin score per (sentence, intent): best similarity to the intent's examples minus best to normal examples."""
        if not len(sentences):
            return np.zeros((0, len(INTENTS)), dtype=np.float32)
        e = self.embed(sentences)
        normal = (e @ self.normal.T).max(1)
        return np.stack([(e @ p.T).max(1) - normal for p in self.protos], axis=1).astype(np.float32)

    def check(self, utterances):
        """Hits for intents whose best sentence passes its calibrated threshold (one hit per intent)."""
        if not self.thresholds:
            raise RuntimeError("IntentChecker has no calibrated thresholds")
        sents = [s for u in utterances for s in split_sentences(u)]
        if not sents:
            return []
        sc = self.scores(sents)
        hits = []
        for j, (iid, level, reason, _) in enumerate(INTENTS):
            thr = self.thresholds.get(iid)
            i = int(np.argmax(sc[:, j]))
            if thr is not None and sc[i, j] >= thr:
                hits.append({"rule": iid, "level": level, "reason": reason, "evidence": sents[i][:80], "score": round(float(sc[i, j]), 3)})
        return hits
