# Enterprise Voice Agent & Cyber Honeypot (Apate.ai Model) – Production Deployment

A low-latency, full-duplex Telephony & Voice AI platform designed for:
1. **Enterprise Conversational Voice Agents** (P50 TTFA < 285ms, <20ms instant barge-in carrier clearing).
2. **Cybersecurity Anti-Scam Honeypot & Threat Intelligence Engine** (Baiter personas that waste scammer time and extract mule bank accounts, crypto wallets, and AnyDesk IDs).

---

## 1. Quick Start (Docker & Docker-Compose)

### Step 1: Clone and Configure Environment
```bash
cp .env.example .env
# Edit .env with your live keys (Deepgram, Groq, ElevenLabs)
```

### Step 2: Launch Production Container
```bash
docker-compose up -d --build
```
The server will start on port `8080` with built-in health checks at `http://localhost:8080/health`.

---

## 2. Telephony & Carrier Configuration (Twilio / Telnyx / SIP)

### Option A: Standard Voice AI Phone Number (Twilio)
Point your Twilio Inbound Voice Webhook URL to:
```
https://your-domain.com/twilio/voice
```
Twilio will automatically receive TwiML and stream bi-directional 8kHz G.711 $\mu$-law audio over `wss://your-domain.com/ws/twilio`.

### Option B: Cyber Scam Honeypot Inbound Route (e.g. Margaret Baiter Bot)
Point suspected scam lines or honeypot numbers to:
```
https://your-domain.com/twilio/honeypot
```
Incoming scam calls will be automatically connected to the **Margaret** / **Arthur** counter-scam baiter bot.

---

## 3. Threat Intelligence REST API & SOC Feed

| Endpoint | Method | Description |
|---|---|---|
| `/health` | `GET` | Health check & active provider status |
| `/api/threats/stats` | `GET` | Live metrics (Scammer time wasted, $$$ lost, mule accounts count) |
| `/api/threats/feed` | `GET` | Real-time JSON list of extracted mule accounts, crypto wallets, & scam transcripts |
| `/twilio/voice` | `POST` | TwiML generator for standard voice calls |
| `/twilio/honeypot` | `POST` | TwiML generator for counter-scam honeypot routing |

---

## 4. Unique Selling Propositions (USPs) for Enterprise & Cybersecurity

1. **Dual-Brain Sub-Conscious Reflex Engine**:
   - Perceived TTFA $< 100\text{ms}$ on complex queries (Brain-1 speaks natural fillers like *"Gotcha, checking that now..."* while Brain-2 resolves database tools).
2. **Instant Carrier Buffer Purge on Barge-In**:
   - Dispatches Twilio `clear` packets in $< 20\text{ms}$, stopping playback on caller handsets without half-second delay.
3. **Real-Time PII & PCI-DSS Wire Sanitization**:
   - In-line redaction of credit cards, SSNs, and emails before saving or logging.
4. **Deterministic Policy & Hallucination Guardrails**:
   - Blocks unauthorized refund commitments or discounts before speech synthesis.
5. **Apate.ai Counter-Scam Model**:
   - Destroys scam syndicate economics ($0 ROI per outbound call) while extracting actionable threat intelligence for banks, telcos, and law enforcement.

---

## 5. Verification & Test Suite

Run the full evaluation matrix:
```bash
# Telephony E2E Call Simulator
PYTHONPATH=backend:. .venv/bin/python -m evals.telephony_eval --turns 3

# Dual-Brain Reflex Latency Benchmark
PYTHONPATH=backend:. .venv/bin/python -m evals.dual_brain_eval

# Cybersecurity & Anti-Scam Intelligence Test
PYTHONPATH=backend:. .venv/bin/python -m evals.scam_honeypot_eval

# Enterprise Guardrails & PII Sanitizer Test
PYTHONPATH=backend:. .venv/bin/python -m evals.enterprise_eval

# 100-Turn Core Pipeline Benchmark
PYTHONPATH=backend:. .venv/bin/python -m evals.benchmark --turns 50
```
