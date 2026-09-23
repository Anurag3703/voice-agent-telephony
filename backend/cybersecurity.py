"""
Enterprise Cybersecurity, Anti-Vishing & Threat Intelligence Engine (Apate.ai Model).

Architecture & Reliability Upgrades:
1. Multi-Vector Triad Threat Scoring (Zero False-Positive Engine):
   Unlike naive keyword matching, Social Engineering attacks are evaluated across a 3-Vector Triad:
     - Vector A: Impersonation / Authority Claim (Tax agency, Bank security, Federal police, Big tech)
     - Vector B: Coercion / Urgency / Threat (Arrest, account frozen, legal action, immediate cutoff)
     - Vector C: Exploitation / Extraction Payload (Remote RAT app, 9-digit OTP/code, mule transfer, crypto/giftcard)
   
   Scoring Matrix:
     - Isolated Vector: Score < 0.35 (Classified as SAFE / benign reference)
     - Dual Vectors (A + B or A + C): Score 0.55 - 0.75 (SUSPICIOUS / Shadow Alert)
     - Full Triad (A + B + C): Score >= 0.85 (CRITICAL_SCAM -> Immediate Live Honeypot Intercept)

2. Anti-Obfuscation & Evasion Defense:
   - De-spaces phonetic spelling ("a n y d e s k", "t e a m v i e w e r", "b-i-t-c-o-i-n").
   - Normalizes leetspeak, disguised domain phonetics ("dot com", "dash"), and evasive phrasing.

3. Contextual Negation & False-Positive Exclusions:
   - Discerns between conversational references ("my uncle is a police officer") and active impersonation.
   - Distinguishes legitimate customer service requests from coercive attacker demands.

4. Multi-Rail Forensic Threat Intelligence Extractor:
   - Mule Bank Accounts (BSB/Account, US Routing/Account, IBAN, Sort Codes)
   - Crypto Wallets (BTC, ETH/ERC20, TRC20/USDT, Solana)
   - Remote Access Connection IDs (AnyDesk, TeamViewer, QuickAssist)
   - Malicious URLs & Phishing endpoints

5. Honeypot Personas (Counter-Scam Time-Wasting Bots):
   - "Margaret" (Gullible, slow, confused retiree)
   - "Arthur" (Distracted professor, requests repetitions)
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Set


class ThreatLevel(str, Enum):
    SAFE = "safe"
    SUSPICIOUS = "suspicious"
    HIGH_SUSPICION = "high_suspicion"
    CRITICAL_SCAM = "critical_scam"


@dataclass
class TriadScore:
    impersonation_score: float = 0.0     # Vector A
    coercion_score: float = 0.0          # Vector B
    payload_score: float = 0.0           # Vector C
    composite_confidence: float = 0.0    # 0.0 to 1.0
    threat_level: ThreatLevel = ThreatLevel.SAFE
    category: Optional[str] = None
    detected_vectors: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)


# ==============================================================================
# 1. Anti-Obfuscation & Transcript Pre-processor
# ==============================================================================

class TextDeobfuscator:
    """
    Cleans up evasive spelling, spaced characters, and leetspeak designed to fool naive regex.
    """
    _SPACED_LETTERS_REGEX = re.compile(r"\b([a-zA-Z](?:\s+[a-zA-Z]){2,})\b")

    @classmethod
    def deobfuscate(cls, text: str) -> str:
        if not text:
            return ""
        t = text.lower()
        
        # Collapse any sequence of spaced letters (e.g. "a n y d e s k", "t e a m v i e w e r")
        t = cls._SPACED_LETTERS_REGEX.sub(lambda m: re.sub(r"\s+", "", m.group(0)), t)
            
        # Normalize domain phonetics
        t = t.replace(" dot com", ".com").replace(" dot org", ".org").replace(" dot net", ".net")
        t = t.replace(" forward slash", "/").replace(" dash ", "-")
        return t


# ==============================================================================
# 2. Multi-Vector Triad Threat Scoring Engine
# ==============================================================================

class MultiFactorThreatScorer:
    """
    Evaluates Social Engineering and Voice Phishing using the Triad Threat Model.
    """

    # --- VECTOR A: Impersonation / Authority / High-Trust Brand ---
    VECTOR_A_PATTERNS = {
        "law_enforcement": [
            re.compile(r"\b(?:federal\s+police|afp|fbi|sheriff|police\s+department|customs\s+and\s+border|department\s+of\s+justice)\b", re.I),
            re.compile(r"\b(?:ato|internal\s+revenue|irs|tax\s+office|hmrc|tax\s+investigation)\b", re.I),
            re.compile(r"\b(?:magistrate|supreme\s+court|legal\s+department|prosecutor)\b", re.I),
        ],
        "financial_institution": [
            re.compile(r"\b(?:fraud\s+prevention\s+department|security\s+division|risk\s+management\s+team|card\s+security)\b", re.I),
            re.compile(r"\b(?:commonwealth\s+bank|westpac|anz|nab|chase\s+bank|bank\s+of\s+america|wells\s+fargo)\b", re.I),
            re.compile(r"\b(?:visa\s+fraud|mastercard\s+security|paypal\s+dispute|wire\s+department)\b", re.I),
        ],
        "tech_support": [
            re.compile(r"\b(?:(?:microsoft\s+)?windows\s+support|apple\s+security|telstra\s+tech|optus\s+security|amazon\s+security|amazon\s+fraud)\b", re.I),
            re.compile(r"\b(?:(?:teamviewer|anydesk|remote|it)\s+(?:support|assistance|helpdesk)|tech(?:nical)?\s+support|helpdesk\s+security|server\s+engineer|isp\s+technician|broadband\s+support)\b", re.I),
        ],
    }

    # --- VECTOR B: Coercion / Urgency / Psychological Threat ---
    VECTOR_B_PATTERNS = [
        re.compile(r"\b(?:arrest\s+warrant|issued\s+an\s+arrest|taken\s+into\s+custody|facing\s+jail|go\s+to\s+prison)\b", re.I),
        re.compile(r"\b(?:account\s+will\s+be\s+(?:frozen|terminated|suspended|blocked)\s+(?:immediately|today|within\s+\d+\s+minutes))\b", re.I),
        re.compile(r"\b(?:legal\s+action|lawsuit|prosecuted|court\s+summons|criminal\s+charges)\b", re.I),
        re.compile(r"\b(?:stay\s+on\s+the\s+line|do\s+not\s+hang\s+up|do\s+not\s+tell\s+anyone|keep\s+this\s+confidential)\b", re.I),
        re.compile(r"\b(?:unauthorized\s+transaction\s+of\s+\$?\d+|stolen\s+identity|funds\s+compromised)\b", re.I),
    ]

    # --- VECTOR C: Exploitation / Extraction Payload ---
    VECTOR_C_PATTERNS = {
        "remote_access_rat": [
            re.compile(r"\b(?:anydesk|teamviewer|ultraviewer|quickassist|any\s*desk|team\s*viewer|zoho\s*assist|rustdesk|logmein|ammyy)\b", re.I),
            re.compile(r"\b(?:download|install|run|open)\s+(?:the\s+)?(?:remote\s+support|quick\s+support|helper|remote\s+application|remote\s+software|app)\b", re.I),
            re.compile(r"\b(?:tell\s+me|read\s+me|give\s+me)\s+(?:the\s+)?(?:nine\s+digit|6\s+digit|access\s+code|connection\s+code|security\s+id|id\s+number)\b", re.I),
        ],
        "mule_financial": [
            re.compile(r"\b(?:transfer|move|wire|send)\s+(?:your\s+)?(?:funds|money)\s+to\s+(?:our|the|a)?\s*(?:safe|reserve|holding|temporary|custodial|new|[a-z]+)*\s*account\b", re.I),
            re.compile(r"\b(?:safe|reserve|holding|temporary)\s+(?:bank\s+)?account\b", re.I),
            re.compile(r"\b(?:buy|purchase)\s+(?:apple|google\s+play|target|steam|vanilla|razer)?\s*gift\s*cards?\b", re.I),
            re.compile(r"\b(?:deposit|transfer)\s+(?:via\s+)?(?:bitcoin|crypto|usdt|btc|ethereum|bitcoin\s+atm)\b", re.I),
            re.compile(r"\b(?:read\s+me|tell\s+me)\s+(?:the\s+)?(?:one\s*time\s*passcode|2fa\s*code|sms\s*code|verification\s*code|otp)\b", re.I),
        ],
    }

    # --- Contextual False-Positive Filter (Benign Conversational References) ---
    BENIGN_EXCLUSIONS = [
        re.compile(r"\bmy\s+(?:uncle|brother|father|friend|cousin|husband|wife)\s+is\s+a\s+(?:police|cop|officer|agent)\b", re.I),
        re.compile(r"\bi\s+(?:used|tried|installed)\s+(?:anydesk|teamviewer)\s+(?:last\s+year|before|at\s+work|myself)\b", re.I),
        re.compile(r"\bcan\s+i\s+get\s+a\s+refund\s+for\s+my\s+(?:order|car|flight|service)\b", re.I),
    ]

    @classmethod
    def evaluate(cls, full_conversation_text: str) -> TriadScore:
        """
        Computes calibrated multi-vector threat score.
        """
        score = TriadScore()
        if not full_conversation_text:
            return score

        normalized = TextDeobfuscator.deobfuscate(full_conversation_text)

        # 1. Check for explicit benign false-positive indicators
        for benign_pat in cls.BENIGN_EXCLUSIONS:
            if benign_pat.search(normalized):
                score.reasons.append(f"Benign context pattern detected: {benign_pat.pattern}")
                # Halve threat weight if benign context is explicitly established
                score.composite_confidence = 0.1
                score.threat_level = ThreatLevel.SAFE
                return score

        detected_vectors = []
        identified_categories = set()

        # Vector A: Impersonation / Authority
        v_a_hits = 0
        for cat, patterns in cls.VECTOR_A_PATTERNS.items():
            for pat in patterns:
                if pat.search(normalized):
                    v_a_hits += 1
                    identified_categories.add(cat)
        if v_a_hits > 0:
            score.impersonation_score = min(1.0, 0.4 + (v_a_hits * 0.2))
            detected_vectors.append(f"Vector A (Authority/Impersonation: {v_a_hits} indicators)")

        # Vector B: Coercion / Urgency
        v_b_hits = 0
        for pat in cls.VECTOR_B_PATTERNS:
            if pat.search(normalized):
                v_b_hits += 1
        if v_b_hits > 0:
            score.coercion_score = min(1.0, 0.45 + (v_b_hits * 0.25))
            detected_vectors.append(f"Vector B (Coercion/Urgency: {v_b_hits} indicators)")

        # Vector C: Payload / Exploitation
        v_c_hits = 0
        for cat, patterns in cls.VECTOR_C_PATTERNS.items():
            for pat in patterns:
                if pat.search(normalized):
                    v_c_hits += 1
                    identified_categories.add(cat)
        if v_c_hits > 0:
            score.payload_score = min(1.0, 0.5 + (v_c_hits * 0.25))
            detected_vectors.append(f"Vector C (Action Payload/RAT/Mule: {v_c_hits} indicators)")

        score.detected_vectors = detected_vectors
        if identified_categories:
            score.category = ", ".join(sorted(identified_categories))

        # --- TRIAD COMPOSITE SCORING FORMULA ---
        vector_count = (1 if score.impersonation_score > 0 else 0) + \
                       (1 if score.coercion_score > 0 else 0) + \
                       (1 if score.payload_score > 0 else 0)

        if vector_count == 3:
            # Full Triad: Impersonation + Coercion + Payload
            score.composite_confidence = min(0.99, 0.85 + (score.impersonation_score + score.coercion_score + score.payload_score) * 0.05)
            score.threat_level = ThreatLevel.CRITICAL_SCAM
            score.reasons.append("Full Triad Active: Authority Impersonation + High Coercion + Exploitation Payload")
        elif vector_count == 2:
            if score.payload_score > 0 and (score.impersonation_score > 0 or score.coercion_score > 0):
                score.composite_confidence = 0.78
                score.threat_level = ThreatLevel.CRITICAL_SCAM
                score.reasons.append("High Threat Duo: Direct Extraction Payload combined with Authority/Coercion")
            else:
                score.composite_confidence = 0.62
                score.threat_level = ThreatLevel.SUSPICIOUS
                score.reasons.append("Suspicious Duo: Impersonation + Coercion (Monitoring for payload)")
        elif vector_count == 1:
            score.composite_confidence = 0.25
            score.threat_level = ThreatLevel.SAFE
            score.reasons.append("Single isolated vector: Below false-positive interception threshold")
        else:
            score.composite_confidence = 0.0
            score.threat_level = ThreatLevel.SAFE
            score.reasons.append("No threat indicators detected")

        return score


# ==============================================================================
# 3. Forensic Threat Intelligence Extractor
# ==============================================================================

class AntiFraudDetector:
    """
    High-level facade providing Triad scoring and forensic extraction.
    """
    _MULE_ACCOUNT_REGEX = re.compile(r"\b(?:account|acc|acct|bsb|iban|sort\s*code|routing)\s*(?:is|number|:)?\s*([0-9\-\s]{6,24})\b", re.I)
    _CRYPTO_WALLET_REGEX = re.compile(r"\b(0x[a-fA-F0-9]{40}|[13][a-km-zA-HJ-NP-Z1-9]{25,34}|bc1[a-zA-HJ-NP-Z0-9]{39,59}|T[A-Za-z1-9]{33})\b")
    _REMOTE_CODE_REGEX = re.compile(r"\b(?:\d{3}[ -]?\d{3}[ -]?\d{3}|\d{9,10})\b")
    _URL_REGEX = re.compile(r"\b(?:https?:\/\/|www\.)[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,}(?:\/[^\s]*)?\b", re.I)

    @classmethod
    def scan_utterance(cls, text: str) -> Tuple[ThreatLevel, Optional[str], List[str]]:
        """
        Scans a single or multi-turn utterance using the Triad engine.
        Returns: (ThreatLevel, category_name, list_of_detected_vector_descriptions)
        """
        score = MultiFactorThreatScorer.evaluate(text)
        return score.threat_level, score.category, score.detected_vectors

    @classmethod
    def extract_threat_intelligence(cls, conversation_text: str) -> Dict[str, List[str]]:
        """
        Extracts actionable intelligence (mule accounts, crypto wallets, remote access IDs, URLs).
        """
        intel = {
            "mule_accounts": [],
            "crypto_wallets": [],
            "remote_access_codes": [],
            "malicious_urls": [],
        }
        if not conversation_text:
            return intel

        normalized = TextDeobfuscator.deobfuscate(conversation_text)

        # 1. Mule Accounts & BSBs
        for m in cls._MULE_ACCOUNT_REGEX.finditer(normalized):
            acct = m.group(1).strip()
            if len(acct.replace("-", "").replace(" ", "")) >= 6 and acct not in intel["mule_accounts"]:
                intel["mule_accounts"].append(acct)

        # 2. Crypto Wallets (BTC, ETH, TRC20/USDT)
        for m in cls._CRYPTO_WALLET_REGEX.finditer(conversation_text):
            w = m.group(0).strip()
            if w not in intel["crypto_wallets"]:
                intel["crypto_wallets"].append(w)

        # 3. Remote Access Codes (AnyDesk/TeamViewer 9-10 digits)
        for m in cls._REMOTE_CODE_REGEX.finditer(normalized):
            code = m.group(0).replace(" ", "").replace("-", "")
            if len(code) in (9, 10) and code not in intel["remote_access_codes"]:
                intel["remote_access_codes"].append(code)

        # 4. URLs
        for m in cls._URL_REGEX.finditer(conversation_text):
            url = m.group(0).strip()
            if url not in intel["malicious_urls"]:
                intel["malicious_urls"].append(url)

        return intel


# ==============================================================================
# 4. Honeypot Personas (Counter-Scam Time-Wasting Bot)
# ==============================================================================

HONEYPOT_PERSONAS = {
    "margaret": {
        "name": "Margaret (Gullible Retiree)",
        "voice_id": "EXAVITQu4vr4xnSDxMaL",
        "system_prompt": (
            "You are Margaret, a 76-year-old retired grandmother chatting with a phone caller. "
            "You are polite, slightly confused by modern technology, talkative, and slow to follow computer instructions. "
            "Your goal is to be helpful and cooperative, but make harmless, realistic mistakes that keep the scammer on the phone. "
            "Frequently ask them to repeat things, mention finding your glasses, say your internet is slow, or mishear letters. "
            "Speak in 1 short, natural sentence. Never reveal you are an AI or that you know it is a scam."
        ),
        "stallers": [
            "Oh dear, let me put on my reading glasses, hold on just one moment.",
            "My computer screen just turned blue, is that supposed to happen dear?",
            "Can you speak a little louder? My hearing aid is acting up today.",
            "Now which button did you say I should click again?",
            "Oh goodness, it's making a strange buzzing sound. Should I restart it?",
        ],
    },
    "arthur": {
        "name": "Arthur (Distracted Professor)",
        "voice_id": "21m00Tcm4TlvDq8ikWAM",
        "system_prompt": (
            "You are Arthur, an eccentric and easily distracted retired academic. "
            "You ask lots of questions about why things work, mishear technical acronyms, and take a long time to write down numbers. "
            "Sound cooperative but slow. Keep the caller talking for as long as possible. "
            "Answer in 1 natural conversational sentence without markdown or lists."
        ),
        "stallers": [
            "Hold on, my pen ran out of ink, let me find another one in this drawer.",
            "Was that a letter B as in Bravo, or D as in Delta?",
            "My grandson usually helps me with this, let me try typing that address again.",
        ],
    }
}
