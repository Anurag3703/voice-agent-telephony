"""
Cybercrime Syndicate Attribution Graph & Dynamic Cognitive Trap Engine.

Research Standard:
1. Multi-Entity Bipartite Attribution Graph:
   - Connects disparate caller phone numbers, mule accounts (BSBs, IBANs), AnyDesk IDs,
     and crypto wallets to clustered transnational fraud syndicates (e.g., "Kolkata Tech Support Gang #4", "Phnom Penh Pig-Butchering Cell").
   - Identifies co-occurrence matrices and shared banking cash-out corridors.

2. Dynamic Cognitive Traps (Adversarial Information Extraction):
   - Generates contextual conversational baits tailored to scam scripts.
   - Compels the scammer to disclose actionable forensic evidence:
     * "Which branch is your central safe account located at?" (Forces BSB / Swift code disclosure)
     * "What is your official supervisor badge number?" (Captures syndicate script alias)
     * "Can you read me the full AnyDesk session link?" (Captures malicious infrastructure URL)
"""

from __future__ import annotations

import collections
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any, Tuple


@dataclass
class SyndicateCluster:
    syndicate_id: str
    name: str
    primary_category: str
    estimated_members: int
    confidence_score: float
    associated_phone_numbers: Set[str] = field(default_factory=set)
    associated_mule_accounts: Set[str] = field(default_factory=set)
    associated_crypto_wallets: Set[str] = field(default_factory=set)
    associated_remote_ids: Set[str] = field(default_factory=set)
    total_calls_attributed: int = 0
    total_time_wasted_seconds: float = 0.0


class SyndicateAttributionGraph:
    """
    Graph correlation engine linking live scam calls to global organized crime syndicates.
    """
    _instance: Optional[SyndicateAttributionGraph] = None

    def __init__(self):
        self.syndicates: Dict[str, SyndicateCluster] = {}
        # Entity lookup inverted indexes
        self._mule_to_syndicate: Dict[str, str] = {}
        self._crypto_to_syndicate: Dict[str, str] = {}
        self._remote_to_syndicate: Dict[str, str] = {}
        self._phone_to_syndicate: Dict[str, str] = {}
        
        self._seed_known_syndicates()

    @classmethod
    def get_instance(cls) -> SyndicateAttributionGraph:
        if cls._instance is None:
            cls._instance = SyndicateAttributionGraph()
        return cls._instance

    def _seed_known_syndicates(self):
        """Seed graph with identified international cybercrime clusters."""
        syn_1 = SyndicateCluster(
            syndicate_id="SYN-APAC-04",
            name="Apex Remote-RAT Tech Support Syndicate",
            primary_category="remote_access",
            estimated_members=45,
            confidence_score=0.94,
            associated_phone_numbers={"+61 488 921 445", "+61 488 921 446", "+1 (415) 890-2193"},
            associated_mule_accounts={"062-115 88294102", "082-991 88412039"},
            associated_crypto_wallets={"bc1qar0srrr7xfkvy5l643lydnw9re59gtzzwf5mdq"},
            associated_remote_ids={"549123884", "992140881"},
            total_calls_attributed=14,
            total_time_wasted_seconds=4820.0,
        )

        syn_2 = SyndicateCluster(
            syndicate_id="SYN-EU-09",
            name="Global Tax Impersonation & Arrest Coercion Cell",
            primary_category="law_enforcement_coercion",
            estimated_members=28,
            confidence_score=0.91,
            associated_phone_numbers={"+44 7700 900123", "+1 (800) 555-0144"},
            associated_mule_accounts={"GB29NWBK60161331926819"},
            associated_crypto_wallets={"0x71C7656EC7ab88b098defB751B7401B5f6d8976F"},
            associated_remote_ids=set(),
            total_calls_attributed=9,
            total_time_wasted_seconds=3120.0,
        )

        self.syndicates[syn_1.syndicate_id] = syn_1
        self.syndicates[syn_2.syndicate_id] = syn_2

        # Build index
        for syn in (syn_1, syn_2):
            for m in syn.associated_mule_accounts:
                self._mule_to_syndicate[m] = syn.syndicate_id
            for c in syn.associated_crypto_wallets:
                self._crypto_to_syndicate[c] = syn.syndicate_id
            for r in syn.associated_remote_ids:
                self._remote_to_syndicate[r] = syn.syndicate_id
            for p in syn.associated_phone_numbers:
                self._phone_to_syndicate[p] = syn.syndicate_id

    def attribute_threat(
        self,
        caller_id: Optional[str],
        mules: List[str],
        wallets: List[str],
        remote_ids: List[str],
        scam_category: Optional[str] = None
    ) -> Tuple[Optional[SyndicateCluster], float, List[str]]:
        """
        Calculates graph match probability between live extracted indicators and known syndicates.
        Returns (MatchedSyndicate, ConfidenceScore, ListOfEvidenceLinks)
        """
        matches: collections.Counter[str] = collections.Counter()
        evidence_links: List[str] = []

        if caller_id and caller_id in self._phone_to_syndicate:
            syn_id = self._phone_to_syndicate[caller_id]
            matches[syn_id] += 3
            evidence_links.append(f"Matching Inbound ANI/Phone: {caller_id}")

        for m in mules:
            if m in self._mule_to_syndicate:
                syn_id = self._mule_to_syndicate[m]
                matches[syn_id] += 5
                evidence_links.append(f"Linked Mule Bank Account: {m}")

        for c in wallets:
            if c in self._crypto_to_syndicate:
                syn_id = self._crypto_to_syndicate[c]
                matches[syn_id] += 5
                evidence_links.append(f"Linked Crypto Cash-out Wallet: {c}")

        for r in remote_ids:
            if r in self._remote_to_syndicate:
                syn_id = self._remote_to_syndicate[r]
                matches[syn_id] += 4
                evidence_links.append(f"Shared Remote Access ID: {r}")

        if not matches:
            # If no direct link exists, check category cluster
            return None, 0.0, ["Unattributed Novel Threat Signature"]

        top_syn_id, hit_count = matches.most_common(1)[0]
        cluster = self.syndicates.get(top_syn_id)
        confidence = min(0.99, 0.45 + hit_count * 0.12)

        return cluster, confidence, evidence_links


class DynamicCognitiveTrapEngine:
    """
    Adversarially steers scam conversations to force the attacker to reveal maximum forensic entropy.
    """
    TRAP_PROMPTS = {
        "remote_access": [
            "My grandson wrote down a 9-digit code last time... is that the same as the AnyDesk number you need dear?",
            "Which website should I type in my browser to connect with you directly?",
            "What is your technician employee ID so I can write it in my notebook?",
        ],
        "financial_mule_transfer": [
            "Can you give me the exact BSB and account number so I can write it down for the teller?",
            "Which bank is this safe reserve account held with, so I know which branch to visit?",
            "Can you spell out the account name and reference code very slowly for me?",
        ],
        "law_enforcement_coercion": [
            "What is your badge number and station address so I can tell my local officer?",
            "Can you give me the official case file reference number to verify?",
        ],
    }

    @classmethod
    def get_adversarial_trap(cls, category: Optional[str], turn_count: int = 1) -> Optional[str]:
        if not category or category not in cls.TRAP_PROMPTS:
            return None

        traps = cls.TRAP_PROMPTS[category]
        idx = (turn_count - 1) % len(traps)
        return traps[idx]
