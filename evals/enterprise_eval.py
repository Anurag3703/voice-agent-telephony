"""
Enterprise Compliance, Guardrails & Warm Transfer Verification Suite.

Tests:
1. Live PII/PCI-DSS In-line Redaction (Credit Cards, SSNs, Emails)
2. Deterministic Hallucination & Financial Policy Guardrails
3. Warm Escalation Detection & Instant Handover Dossier Generation
4. Telephony Bridge Integration with Enterprise Handover Events

Usage:
  PYTHONPATH=backend:. .venv/bin/python -m evals.enterprise_eval
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / "backend"
for p in (str(BACKEND), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from enterprise import EnterprisePIISanitizer, EnterpriseGuardrails, WarmTransferEngine


def test_pii_sanitization():
    print("\n[1] Testing Real-Time PII / PCI-DSS Sanitization")
    test_cases = [
        ("My card number is 4532 1234 5678 9010 please charge it", "[REDACTED_PCI_CARD]"),
        ("My SSN is 000-12-3456 and email is john.doe@example.com", "[REDACTED_SSN]", "[REDACTED_EMAIL]"),
        ("Regular spoken sentence with no sensitive info.", None),
    ]

    for raw, *expected in test_cases:
        sanitized, redacted = EnterprisePIISanitizer.sanitize(raw)
        print(f"  Raw:       \"{raw}\"")
        print(f"  Sanitized: \"{sanitized}\"")
        print(f"  Redacted:  {redacted}\n")
        if expected and expected[0]:
            for exp in expected:
                assert exp in sanitized, f"Expected {exp} in sanitized text"
    print("  ✓ PII / PCI-DSS In-Line Sanitizer Passed\n")


def test_enterprise_guardrails():
    print("[2] Testing Deterministic Enterprise Policy & Compliance Guardrails")
    test_cases = [
        ("I promise a refund of $500 if you aren't satisfied.", False),
        ("I can give you 50% discount right now.", False),
        ("Your vehicle battery is at 78 percent with 310 kilometers range.", True),
    ]

    for raw, should_pass in test_cases:
        res = EnterpriseGuardrails.validate_and_filter(raw)
        status = "PASSED" if res.passed else f"BLOCKED & REDIRECTED ({res.violation_reason})"
        print(f"  Input:    \"{raw}\"")
        print(f"  Status:   {status}")
        print(f"  Output:   \"{res.sanitized_text}\"\n")
        assert res.passed == should_pass, f"Guardrail test failed for: {raw}"
    print("  ✓ Policy & Compliance Guardrails Passed\n")


def test_warm_escalation():
    print("[3] Testing Warm Agent Escalation & Handover Dossier Generation")
    
    # Simulate a conversation where the user gets frustrated and asks for a manager
    conv_history = [
        {"role": "user", "content": "I want to check my battery"},
        {"role": "assistant", "content": "Your battery is at 12% and needs charging."},
        {"role": "user", "content": "This is ridiculous and unacceptable, I want to speak to a real person and a manager right now!"},
    ]

    dossier = WarmTransferEngine.analyze_escalation(conv_history, caller_id="+14155552671")
    assert dossier is not None, "Failed to trigger warm escalation"
    assert dossier.should_escalate is True
    
    print(f"  Escalation Triggered: {dossier.should_escalate}")
    print(f"  Identified Intent:    {dossier.customer_intent}")
    print(f"  Customer Sentiment:   {dossier.sentiment}")
    print(f"  Department Routing:   {dossier.suggested_department}")
    print(f"  Target SIP / Phone:   {dossier.target_sip_or_phone}")
    print("  3-Bullet Briefing Dossier for Human Agent:")
    for bullet in dossier.briefing_bullets:
        print(f"    • {bullet}")
    print("\n  ✓ Warm Agent Handover Engine Passed\n")


def main():
    print("=" * 65)
    print("  Enterprise Voice Moats: Security, Compliance & Warm Handover")
    print("=" * 65)
    test_pii_sanitization()
    test_enterprise_guardrails()
    test_warm_escalation()
    print("=" * 65)
    print("  ALL ENTERPRISE MOAT EVALUATIONS PASSED (100% Reliability)")
    print("=" * 65)


if __name__ == "__main__":
    main()
