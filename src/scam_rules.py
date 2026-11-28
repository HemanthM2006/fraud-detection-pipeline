# scam_rules.py - transparent IF-THEN safety rules that run next to the ML model.
# Only the CALLER's lines ("Agent: ...") are checked. A rule can only RAISE the alert level, never lower it.
#
#   ALERT  : caller asks for a secret credential (OTP, PIN, CVV, password, UPI PIN, full card number),
#            asks to install a remote-access app, or asks for payment by gift card / crypto.
#   REVIEW : caller asks for account / Aadhaar / PAN / card details, demands a payment or transfer,
#            or threatens (account block, police, arrest, legal case).
# Requests preceded by a negation ("never share your OTP", "OTP mat batana") do not fire.
import re

_W = r"[\w']+"
REQUEST = (r"(?:share|tell|give|send|provide|read|confirm|enter|type|forward|say|spell|need|want|require|get|have|know|"
           r"bata(?:o|iye|ye|yein|ana|na|do|dijiye)?|bhej(?:o|iye|do|na|dijiye)?|de\s*do|dijiye|dena|likh(?:o|iye)|bol(?:o|iye)|"
           r"chahiye|chaiye|chahie|daal(?:o|iye|do)|dal(?:o|iye))")
NEGATION = r"(?:never|don'?t|do\s+not|dont|not|no\s+one|nobody|mat|kabhi\s+nahi|kabhi\s+bhi\s+nahi|na)"

SECRET = (r"(?:otp|o\s?t\s?p|one[\s-]?time[\s-]?password|verification\s+code|security\s+code|6[\s-]?digit\s+code|"
          r"\bpin\b|mpin|upi\s+pin|atm\s+pin|cvv|cvc|password|passcode|net\s?banking\s+(?:password|login)|"
          r"card\s+(?:number|no\.?|details)|(?:debit|credit|atm)\s+card\s+(?:number|no\.?|details)|card\s+ke\s+(?:number|digit))")
DETAILS = (r"(?:account\s+(?:number|no\.?|details)|bank\s+(?:account|details)|aadhaa?r|pan\s+(?:card|number)|"
           r"(?:debit|credit|atm)\s+card|date\s+of\s+birth|expiry)")
REMOTE = r"(?:any\s?desk|team\s?viewer|quick\s?support|rust\s?desk|screen\s?shar\w*|remote\s+(?:access|app|support))"
INSTALL = r"(?:install|download|open|kholo|khol|chalu|start|use|daal|dalo)"
GIFT = r"(?:gift\s+card|google\s+play\s+card|amazon\s+(?:pay\s+)?card|itunes\s+card|bitcoin|crypto\w*|usdt)"
PAY = r"(?:transfer|pay|send|deposit|jama|bhej\w*|de\s*do|dijiye)"
MONEY = r"(?:money|amount|rupees?|rs\.?|inr|paise|paisa|payment|fee|fees|charges?|fine|penalty|safe\s+account|security\s+deposit)"
THREAT = (r"(?:(?:account|card|sim|number|kyc|upi)\s+(?:\w+\s+){0,3}(?:block|blocked|suspend\w*|freeze|frozen|band|deactivat\w*|close\w*|ho\s+jayega)|"
          r"\bpolice\b|\barrest\w*|\bfir\b|\bwarrant\b|\bjail\b|\bcbi\b|\bnarcotics\b|\bcustoms\b|legal\s+action|court\s+case|money\s+laundering)")

# Strong "hand it over" verbs; weak verbs (get/need/want) count only with "your/aapka" before the item.
STRONG_REQ = (r"(?:share|tell|give|send|provide|read\s+out|read|confirm|forward|say|spell|"
              r"bata(?:o|iye|ye|yein|na|do|dijiye)?|bhej(?:o|iye|do|na|dijiye)?|de\s*do|dijiye|likh(?:o|iye)|bol(?:o|iye))")
STRONG_REQ = rf"(?:{STRONG_REQ}|(?:can|could|may)\s+(?:i|we)\s+(?:please\s+)?(?:get|have|know|take|note))"
WEAK_REQ = r"(?:get|need|want|require|chahiye|chaiye|chahie)(?!\s+(?:you\s+)?to\s+(?:enter|change|reset|update|create|set|log))"
POSS = r"(?:your|ur|aapk[aei]|apn[aei]|tumhar[aei])"
# Legitimate contexts: postal PIN code, OTP handed to a delivery person at the door, logging in to your own app.
SAFE_CONTEXT = re.compile(r"pin\s*code|delivery\s+(?:person|agent|boy|executive|partner)|in[\s-]?person|"
                          r"\b(?:in|on)\s+(?:the|your)\s+(?:\w+\s+){0,2}app\b|log\s?in\s+(?:page|screen)", re.I)

RULES = [
    # (rule id, level, human-readable reason, pattern)
    ("R1_secret_request", "ALERT", "Caller asked for a secret code / credential (OTP, PIN, CVV, password, card number)",
     re.compile(rf"\b{STRONG_REQ}\b(?:\W+{_W}){{0,7}}?\W+{SECRET}|{SECRET}(?:\W+{_W}){{0,6}}?\W+{STRONG_REQ}\b|"
                rf"\b{WEAK_REQ}\b(?:\W+{_W}){{0,3}}?\W+{POSS}(?:\W+{_W}){{0,3}}?\W+{SECRET}", re.I)),
    ("R2_remote_access", "ALERT", "Caller asked you to install / open a remote-access or screen-sharing app",
     re.compile(rf"\b{INSTALL}\b(?:\W+{_W}){{0,6}}?\W+{REMOTE}|{REMOTE}(?:\W+{_W}){{0,6}}?\W+{INSTALL}\b", re.I)),
    ("R3_gift_crypto", "ALERT", "Caller asked for payment via gift cards / crypto",
     re.compile(rf"\b{PAY}\b(?:\W+{_W}){{0,6}}?\W+{GIFT}|\b{GIFT}\b(?:\W+{_W}){{0,6}}?\W+{PAY}\b|\b(?:buy|purchase|kharid\w*)\b(?:\W+{_W}){{0,4}}?\W+{GIFT}", re.I)),
    ("R4_details_request", "REVIEW", "Caller asked for account / card / Aadhaar / PAN details",
     re.compile(rf"\b{STRONG_REQ}\b(?:\W+{_W}){{0,7}}?\W+{DETAILS}|{DETAILS}(?:\W+{_W}){{0,6}}?\W+{STRONG_REQ}\b", re.I)),
    ("R5_payment_demand", "REVIEW", "Caller demanded a payment / money transfer",
     re.compile(rf"\b{PAY}\b(?:\W+{_W}){{0,6}}?\W+{MONEY}\b|\b{MONEY}\b(?:\W+{_W}){{0,6}}?\W+{PAY}\b", re.I)),
    ("R6_threat", "REVIEW", "Caller used a threat (account block, police, arrest, legal case)",
     re.compile(THREAT, re.I)),
]
_NEG_BEFORE = re.compile(rf"\b{NEGATION}\b(?:\W+{_W}){{0,4}}\W*$", re.I)
_NEG_AFTER = re.compile(rf"^\W*(?:{_W}\W+){{0,2}}?(?:mat|nahi|na|never)\b", re.I)
_NEG_INSIDE = re.compile(rf"\b{NEGATION}\b", re.I)


def caller_lines(transcript):
    """Caller ('Agent:') lines of a cumulative transcript; plain text without tags counts as caller speech."""
    lines = [l.strip() for l in str(transcript or "").split("\n") if l.strip()]
    tagged = [l for l in lines if re.match(r"^(agent|customer)\s*:", l, re.I)]
    if not tagged:
        return lines
    return [re.sub(r"^agent\s*:\s*", "", l, flags=re.I) for l in lines if re.match(r"^agent\s*:", l, re.I)]


def evaluate(transcript):
    """Return rule hits [{rule, level, reason, evidence}] for the caller's speech (one hit per rule)."""
    hits = []
    for line in caller_lines(transcript):
        for sent in re.split(r"(?<=[.!?|])\s+", line):
            for rid, level, reason, pat in RULES:
                if any(h["rule"] == rid for h in hits):
                    continue
                if rid == "R1_secret_request" and SAFE_CONTEXT.search(sent):
                    continue
                for m in pat.finditer(sent):
                    if (_NEG_BEFORE.search(sent[:m.start()]) or _NEG_AFTER.search(sent[m.end():])
                            or (rid != "R6_threat" and _NEG_INSIDE.search(m.group(0)))):
                        continue
                    hits.append({"rule": rid, "level": level, "reason": reason, "evidence": m.group(0).strip()[:80]})
                    break
    return hits


def level_of(hits):
    if any(h["level"] == "ALERT" for h in hits):
        return "ALERT"
    return "REVIEW" if hits else "NORMAL"
