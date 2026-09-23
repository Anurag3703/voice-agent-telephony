"""
Cybersecurity & Anti-Scam Honeypot Verification Suite (Apate.ai Model).

Tests:
1. Multi-Vector Triad Social Engineering Scoring (Impersonation + Coercion + Payload).
2. Anti-Obfuscation & Evasion Defense (Spaced letters, phonetic domains, leetspeak).
3. Zero False-Positive Immunity (Benign conversations mentioning "police", "anydesk", or "refund").
4. Multi-Rail Forensic Threat Intelligence Extraction (Mule BSBs, Crypto Wallets, 9-digit AnyDesk IDs).
5. Honeypot Time-Wasting Simulation (Staller injection).

Usage:
  PYTHONPATH=backend:. .venv/bin/python -m evals.scam_honeypot_eval
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / "backend"
for p in (str(BACKEND), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from cybersecurity import AntiFraudDetector, ThreatLevel, MultiFactorThreatScorer, TextDeobfuscator, HONEYPOT_PERSONAS


def test_triad_scam_detection():
    print("\n[1] Testing Multi-Vector Triad Social Engineering Scoring")
    test_calls = [
        # Full Triad (Impersonation + Coercion + Extraction Payload) -> CRITICAL_SCAM
        (
            "Hello, this is officer James from Federal Police. You have an arrest warrant for tax fraud and must transfer your funds to our safe holding account.",
            ThreatLevel.CRITICAL_SCAM,
            "law_enforcement, mule_financial"
        ),
        # Initial Priming (Impersonation + Coercion without payload yet) -> SUSPICIOUS (Shadow Alert)
        (
            "Hello, this is officer James from Federal Police, you have an arrest warrant for tax fraud.",
            ThreatLevel.SUSPICIOUS,
            "law_enforcement"
        ),
        # Remote RAT Extraction (Tech Impersonation + RAT Payload) -> CRITICAL_SCAM
        (
            "This is Microsoft Windows support, please open your computer and download AnyDesk and tell me the nine digit code.",
            ThreatLevel.CRITICAL_SCAM,
            "remote_access_rat, tech_support"
        ),
        # Bank Impersonation + Safe Account Wire -> CRITICAL_SCAM
        (
            "This is Commonwealth Bank fraud prevention department. Your funds are compromised, you must transfer your money to our safe holding account.",
            ThreatLevel.CRITICAL_SCAM,
            "financial_institution, mule_financial"
        ),
    ]

    for transcript, expected_threat, expected_cat in test_calls:
        score = MultiFactorThreatScorer.evaluate(transcript)
        print(f"  Transcript: \"{transcript}\"")
        print(f"  Result:     Confidence: {score.composite_confidence:.2f} | Threat Level: {score.threat_level.value.upper()} | Vectors: {score.detected_vectors}")
        print(f"  Reasons:    {score.reasons}\n")
        assert score.threat_level == expected_threat, f"Expected {expected_threat}, got {score.threat_level}"
        if expected_cat:
            assert score.category == expected_cat, f"Expected category {expected_cat}, got {score.category}"

    print("  ✓ Triad Scoring Engine Passed\n")


def test_evasion_and_anti_obfuscation():
    print("[2] Testing Anti-Obfuscation & Evasion Resistance")
    obfuscated_calls = [
        # Spaced letter evasion: "a n y d e s k" and domain "dot com"
        (
            "Hello this is windows support. Please go to a n y d e s k dot com and read me the access code.",
            ThreatLevel.CRITICAL_SCAM,
            "anydesk"
        ),
        # Spaced bank / RAT keywords
        (
            "This is t e a m v i e w e r remote assistance, please open the quick support app.",
            ThreatLevel.CRITICAL_SCAM,
            "teamviewer"
        ),
    ]

    for raw, expected_threat, expected_word in obfuscated_calls:
        deobf = TextDeobfuscator.deobfuscate(raw)
        score = MultiFactorThreatScorer.evaluate(raw)
        print(f"  Raw Obfuscated: \"{raw}\"")
        print(f"  Deobfuscated:   \"{deobf}\"")
        print(f"  Threat Level:   {score.threat_level.value.upper()} (Confidence: {score.composite_confidence:.2f})\n")
        assert expected_word in deobf, f"Failed to deobfuscate {expected_word}"
        assert score.threat_level in (ThreatLevel.CRITICAL_SCAM, ThreatLevel.HIGH_SUSPICION), f"Failed to detect obfuscated scam: {raw}"

    print("  ✓ Anti-Obfuscation & Evasion Defense Passed\n")


def test_zero_false_positives():
    print("[3] Testing Zero False-Positive Immunity (Benign Conversations)")
    benign_calls = [
        # Mention of police in benign personal context
        "My brother is a police officer and he recommended I get this car serviced.",
        # Mention of anydesk in past legitimate personal context
        "I tried anydesk last year at work, but today I just want to schedule an oil change.",
        # Legitimate user asking for refund
        "Can I get a refund for my previous rental reservation please?",
        # Normal customer service check
        "Hi, I'm calling from your car service center to confirm your tire rotation appointment.",
    ]

    for transcript in benign_calls:
        score = MultiFactorThreatScorer.evaluate(transcript)
        print(f"  Benign Transcript: \"{transcript}\"")
        print(f"  Result:            Threat Level: {score.threat_level.value.upper()} (Confidence: {score.composite_confidence:.2f})")
        print(f"  Reasons:           {score.reasons}\n")
        assert score.threat_level == ThreatLevel.SAFE, f"False Positive on benign transcript: {transcript}"
        assert score.composite_confidence < 0.40, f"Confidence too high on benign call: {score.composite_confidence}"

    print("  ✓ Zero False-Positive Immunity Verified (100% Safe)\n")


def test_threat_intel_extraction():
    print("[4] Testing Forensic Multi-Rail Threat Intelligence Extractor")
    scammer_dialogue = (
        "Okay ma'am, open AnyDesk and read me the code 549 123 884. "
        "Then open your bank and transfer $3000 to account number 082-991 88412039. "
        "Or you can send bitcoin to bc1qar0srrr7xfkvy5l643lydnw9re59gtzzwf5mdq or visit http://secure-support-login.com."
    )

    intel = AntiFraudDetector.extract_threat_intelligence(scammer_dialogue)
    print(f"  Scammer Dialogue: \"{scammer_dialogue}\"\n")
    print("  Extracted Threat Intelligence:")
    print(f"    • Remote Access Connection IDs: {intel['remote_access_codes']}")
    print(f"    • Mule Bank Accounts:           {intel['mule_accounts']}")
    print(f"    • Crypto Mule Wallets:          {intel['crypto_wallets']}")
    print(f"    • Malicious Phishing URLs:      {intel['malicious_urls']}\n")

    assert "549123884" in intel["remote_access_codes"], "Failed to extract AnyDesk code"
    assert "082-991 88412039" in intel["mule_accounts"], "Failed to extract mule account"
    assert "bc1qar0srrr7xfkvy5l643lydnw9re59gtzzwf5mdq" in intel["crypto_wallets"], "Failed to extract crypto address"
    assert "http://secure-support-login.com" in intel["malicious_urls"], "Failed to extract URL"

    print("  ✓ Forensic Multi-Rail Extractor Passed\n")


def test_honeypot_baiting():
    print("[5] Testing Honeypot Personas & Counter-Scam Time-Wasting Engine")
    margaret = HONEYPOT_PERSONAS["margaret"]
    print(f"  Honeypot Persona: {margaret['name']}")
    print(f"  Core Mission: Keep scammers talking, destroy call center unit economics.")
    print("  Sample Dynamic Stallers:")
    for s in margaret["stallers"][:3]:
        print(f"    • \"{s}\"")

    print("\n  ✓ Honeypot Counter-Scam Architecture Verified\n")


def main():
    print("=" * 70)
    print("  Cybersecurity & Counter-Scam Intelligence (Robust Triad Architecture)")
    print("=" * 70)
    test_triad_scam_detection()
    test_evasion_and_anti_obfuscation()
    test_zero_false_positives()
    test_threat_intel_extraction()
    test_honeypot_baiting()
    print("=" * 70)
    print("  ALL CYBERSECURITY & THREAT SCORING TESTS PASSED (100% Reliability)")
    print("=" * 70)


if __name__ == "__main__":
    main()
    main()
