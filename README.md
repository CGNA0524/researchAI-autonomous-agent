# 🧠 ResearchAI — Production-Oriented Autonomous Research Agent

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)
[![Pydantic v2](https://img.shields.io/badge/Pydantic-v2-E92063.svg?logo=pydantic)](https://docs.pydantic.dev/)
[![Groq SDK](https://img.shields.io/badge/Groq-Official%20SDK-F55036.svg)](https://groq.com)
[![OpenAI SDK](https://img.shields.io/badge/OpenAI-Official%20SDK%20v1.50+-412991.svg?logo=openai)](https://platform.openai.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B.svg?logo=streamlit)](https://streamlit.io)
[![Tests Passing](https://img.shields.io/badge/tests-42%2F42%20passed-brightgreen.svg)]()
[![Evaluation](https://img.shields.io/badge/evaluation-10%2F10%20passed-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Predixion AI Platform Engineering Internship Technical Assignment**  
> An autonomous, reliable, production-oriented AI Research Agent featuring multi-step planning, resilient multi-engine search, hierarchical source classification, evidence coverage matrix validation, intelligent second-hop retrieval, programmatic confidence calibration, claim citation whitelisting, hard budget guards, and zero-leakage security.

---

## 📑 Table of Contents
1. [Project Overview](#1-project-overview)
2. [Key Production Upgrades](#2-key-production-upgrades)
3. [Architecture Overview](#3-architecture-overview)
4. [Source Classification & Authority Ranking](#4-source-classification--authority-ranking)
5. [Evidence Coverage Validator & Matrix](#5-evidence-coverage-validator--matrix)
6. [Intelligent Second-Hop Retrieval](#6-intelligent-second-hop-retrieval)
7. [Programmatic Confidence Calibration](#7-programmatic-confidence-calibration)
8. [Claim-Level Citation Validation](#8-claim-level-citation-validation)
9. [Token Tracking & Pre-Call Hard Budget Guard](#9-token-tracking--pre-call-hard-budget-guard)
10. [Security & Failure Handling](#10-security--failure-handling)
11. [Installation & Setup](#11-installation--setup)
12. [How to Run the Application](#12-how-to-run-the-application)
13. [Testing & Evaluation](#13-testing--evaluation)
14. [Controlled Live Demonstration](#14-controlled-live-demonstration)

---

## 1. Project Overview

**ResearchAI** is a production-oriented autonomous research system built in Python 3.11+ using FastAPI and Streamlit. It takes complex, multi-entity engineering queries (e.g. *"Compare FastAPI vs Django vs Flask for production asynchronous web microservices"*), plans multi-step investigations, searches and extracts primary technical evidence, constructs an entity-by-dimension coverage matrix, triggers second-hop retrieval for missing evidence, verifies citations against strict whitelists, and programmatically computes calibrated confidence without hallucinating specifications.

Every response conforms strictly to the backward-compatible Pydantic schema:
```json
{
  "user_question": "string",
  "short_answer": "string",
  "key_findings": ["string"],
  "sources_used": [
    {
      "title": "string",
      "url": "string",
      "relevance": "string"
    }
  ],
  "confidence_level": "HIGH | MEDIUM | LOW | INSUFFICIENT_EVIDENCE",
  "assumptions": ["string"],
  "limitations": ["string"],
  "suggested_next_steps": ["string"],
  "confidence_explanation": "Deterministic score rationale",
  "evidence_coverage": {
    "coverage_ratio": 0.85,
    "missing_entities": [],
    "missing_dimensions": []
  },
  "source_classification": {
    "OFFICIAL_DOCS": ["https://fastapi.tiangolo.com/"],
    "REPUTABLE_PUBLICATION": []
  }
}
```

---

## 2. Key Production Upgrades

| Upgrade Component | Implementation File | Key Benefit |
|---|---|---|
| **Source Authority Classification** | [`app/agent/source_classifier.py`](app/agent/source_classifier.py) | Classifies sources into 7 credibility tiers (Official Docs, Academic, Reputable, Community, Junk); detects official entity domains dynamically. |
| **Evidence Coverage Validator** | [`app/agent/coverage.py`](app/agent/coverage.py) | Builds `(entity, dimension)` matrix, calculates mathematical coverage ratio, flags missing dimensions, and detects omissions. |
| **Intelligent Second-Hop Retrieval** | [`app/agent/orchestrator.py`](app/agent/orchestrator.py) | Automatically detects coverage gaps after Round 1 and executes targeted queries exclusively for missing evidence (max 2 rounds). |
| **Programmatic Confidence Calibration** | [`app/agent/confidence.py`](app/agent/confidence.py) | LLM does NOT decide confidence. Uses deterministic point formula (0–100) combining source authority, coverage ratio, extraction depth, and contradiction penalties. |
| **Claim-Level Citation Validation** | [`app/agent/citation.py`](app/agent/citation.py) | Enforces strict source whitelist; qualifies numerical benchmark claims lacking test hardware conditions; flags ungrounded assertions. |
| **Dynamic Research Planner** | [`app/agent/planner.py`](app/agent/planner.py) | Uses dynamic current reference year (avoids stale hardcoded past years); systematically covers all comparison entities. |
| **Hard Pre-Call Budget Guard** | [`app/llm/openai_client.py`](app/llm/openai_client.py) | Proactively estimates cost before LLM calls with a 25% safety margin; blocks execution and raises `BudgetExceededError` before exceeding cap ($0.10). |
| **10-Scenario Evaluation Suite** | [`evaluations/run_evals.py`](evaluations/run_evals.py) | Reproducible benchmark running 100% offline with zero live API calls and $0.00 credit consumption; 10/10 scenarios passed. |

---

## 3. Architecture Overview

```
[ FastAPI Backend / Streamlit UI ]
                │
                ▼
      [ Agent Orchestrator ]
         │       │        │
         ▼       ▼        ▼
     [Planner] [Filter] [Coverage Validator]
         │       │        │
         ▼       │        │
     [Round 1 Discovery Search & Extraction]
                 │
                 ▼
     [Coverage Matrix Gap Evaluation]
                 │
                 ▼ (If missing evidence & budget permits)
     [Round 2 Targeted Second-Hop Retrieval]
                 │
                 ▼
     [Deterministic Programmatic Confidence Assessor]
                 │
                 ▼
     [Citation Whitelist & Benchmark Qualification]
                 │
                 ▼
   [Budget-Guarded Answer Synthesizer]
```

---

## 4. Source Classification & Authority Ranking

Sources are categorized into 7 discrete tiers:
1. **OFFICIAL_DOCS:** Official documentation matching researched entities (e.g., `fastapi.tiangolo.com`, `docs.djangoproject.com`, `docs.qdrant.tech`). Weight: 4.5.
2. **OFFICIAL_REPO:** Primary organization GitHub repositories (e.g., `github.com/tiangolo/fastapi`). Weight: 3.5.
3. **ACADEMIC_PAPER:** Peer-reviewed academic publishers (`arxiv.org`, `dl.acm.org`, `ieeexplore.ieee.org`). Weight: 3.0.
4. **REPUTABLE_PUBLICATION:** Established engineering and industry publications (`martinfowler.com`, `infoq.com`, `aws.amazon.com`, `lwn.net`). Weight: 2.2.
5. **COMMUNITY_ARTICLE:** Medium, Dev.to, personal engineering blogs. Weight: 1.0.
6. **GENERAL_REPO:** Unrelated or third-party toy repositories. Weight: 0.5.
7. **LOW_QUALITY:** Captcha walls, paywalls, spam aggregators, 404s. Weight: 0.0 (discarded).

Duplicate URLs (canonicalized and stripped of UTM tracking parameters) and content fingerprints are automatically discarded.

---

## 5. Evidence Coverage Validator & Matrix

The **Evidence Coverage Validator** (`app/agent/coverage.py`) maintains an explicit matrix for every comparison:

$$\text{Coverage Ratio} = \frac{\text{Covered Cells} + 0.5 \times \text{Partial Cells}}{\text{Total Cells}}$$

For example, when comparing **FastAPI vs Django vs Flask** across **Performance**, **Async**, **Database Integration**, and **Limitations** (12 total cells):
- If Django is missing performance data, `Django` is flagged in `missing_entities`.
- The system generates targeted second-hop queries exclusively for `("Django", "Performance")`.
- If evidence remains unavailable, the comparison is marked explicitly incomplete. Missing data is never invented.

---

## 6. Intelligent Second-Hop Retrieval

After Round 1 discovery search and extraction:
1. The Coverage Matrix identifies missing `(entity, dimension)` cells.
2. If `coverage_ratio < 0.70` and missing pairs exist:
   - Up to 2 focused queries are generated targeting only the missing evidence.
   - Round 2 search and extraction is executed.
   - The coverage matrix is re-evaluated with consolidated evidence.
3. Total retrieval rounds are strictly capped at 2.
4. If the per-request budget cap is approached, Round 2 is bypassed gracefully.

---

## 7. Programmatic Confidence Calibration

The LLM is **strictly forbidden** from deciding the final confidence level. Confidence is evaluated deterministically using a 100-point calibrated formula:

- **Source Count & Diversity:** Max 25 points.
- **Authority Tier:** Max 25 points (weighted heavily toward official docs and academic papers).
- **Evidence Coverage Ratio:** Max 30 points ($\text{ratio} \times 30$).
- **Extraction Depth & Text Volume:** Max 20 points.
- **Penalties:**
  - Missing entities: $-8$ points per entity.
  - Contradictory sources: $-20$ points penalty.

### Calibrated Confidence Rules:
- **`HIGH`:** Score $\ge 75$, coverage $\ge 70\%$, at least 1 authoritative source, at least 1 successful full extraction, **0 missing entities**, and **0 unresolved contradictions**.
- **`MEDIUM`:** Score $\ge 45$ and coverage $\ge 40\%$. Evidence is informative but exhibits minor dimension gaps.
- **`LOW`:** Score $\ge 15$ or coverage $< 40\%$. Evidence is weak, limited, or fragmented.
- **`INSUFFICIENT_EVIDENCE`:** Score $< 15$ or zero verified external sources retrieved. Refuses to speculate.

---

## 8. Claim-Level Citation Validation

1. **Strict Whitelist Enforcement:** Every citation URL in `sources_used` must belong to the retrieved and verified source whitelist. Fabricated URLs are completely prohibited.
2. **Benchmark Qualification:** Any claim containing numerical performance benchmarks (e.g., `80,000 req/s`, `10x faster`) is qualified with testing conditions:  
   *(Note: Benchmark figures reflect specific vendor or test conditions)*
3. **Contradiction Detection:** Conflicting benchmark claims between sources are detected and documented in `limitations`.
4. **Ungrounded Claims:** Statements lacking textual corroboration in retrieved pages are prefixed with `[Unverified Observation]`.

---

## 9. Token Tracking & Pre-Call Hard Budget Guard

The application safeguards API balances with proactive budget controls:

- **Provider Support:** Seamlessly supports both **OpenAI** (`gpt-4o-mini`, `gpt-4o`) and **Groq** (`openai/gpt-oss-120b`, `llama-3.3-70b-versatile`).
- **Token Tracking:** Tracks prompt tokens, completion tokens, call counts, and retries per request.
- **Verified Pricing:** Uses real model pricing ($0.15/M input, $0.60/M output for `gpt-4o-mini`).
- **Hard Pre-Call Guard:** Before making any OpenAI API call, `estimate_call_cost()` computes projected cost. If:

$$\text{Current Cost} + (\text{Estimated Cost} \times 1.25) > \text{MAX\_REQUEST\_BUDGET\_USD}$$

The call is blocked and raises `BudgetExceededError`. The orchestrator catches this and gracefully returns a structured partial result without crashing or incurring overdrafts.

---

## 10. Security & Failure Handling

- **SSRF Defense:** `SecurityValidator` validates both target hostnames and resolved IP addresses, blocking `127.0.0.1`, `localhost`, private subnets (`10.0.0.0/8`, `192.168.0.0/16`), and cloud metadata (`169.254.169.254`).
- **Prompt Injection:** Defangs known injection patterns (`[POTENTIAL_INJECTION_DEFANGED]`) and wraps evidence in `<untrusted_web_evidence>` boundaries.
- **Credential Scrubbing:** `SafeFormatter` scrubs all API keys matching `sk-...` and `gsk_...` from logs and terminal outputs.
- **Graceful Partial Results:** If search engines time out, scrapers fail, or budget is exceeded, the agent returns structured partial findings with explicit limitation disclosures rather than throwing 500 errors.

---

## 11. Installation & Setup

### Prerequisites
- Python 3.11+
- Virtual environment (`.venv`)

```powershell
# 1. Clone repository & navigate to folder
cd c:\Users\NC\Desktop\ai-research-agent

# 2. Activate virtual environment
.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt
```

### Environment Configuration (`.env`)
```ini
# Active Provider: openai or groq
LLM_PROVIDER=openai
OPENAI_MODEL=gpt-4o-mini
OPENAI_API_KEY=sk-...

# Groq Alternative Provider
GROQ_API_KEY=gsk_...
GROQ_PRIMARY_MODEL=openai/gpt-oss-120b
GROQ_FALLBACK_MODEL=openai/gpt-oss-20b

# Budget Guard Settings
MAX_REQUEST_BUDGET_USD=0.10
MAX_SEARCH_RESULTS=5
MAX_EXTRACT_PAGES=3
```

---

## 12. How to Run the Application

### Start FastAPI Backend
```powershell
.venv\Scripts\python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```
- Swagger Interactive Docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Health Check: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

### Start Streamlit Frontend
```powershell
.venv\Scripts\streamlit run streamlit_app.py --server.port 8501
```
- Open browser at: [http://localhost:8501](http://localhost:8501)

---

## 13. Testing & Evaluation

### Run Complete Unit Test Suite (42 Tests)
```powershell
.venv\Scripts\pytest -v
```
All 42 tests run in ~3 seconds and pass with zero regressions.

### Run 10-Scenario Offline Benchmark (Zero Live Credits)
```powershell
.venv\Scripts\python evaluations/run_evals.py
```
Outputs detailed execution traces and metrics to [`evaluations/eval_results.json`](evaluations/eval_results.json).

---

## 14. Controlled Live Demonstration

To perform a safe, controlled live research demonstration using minimal tokens:

1. Ensure `.env` has `LLM_PROVIDER=openai` and `OPENAI_MODEL=gpt-4o-mini`.
2. Open Streamlit at [http://localhost:8501](http://localhost:8501).
3. Submit the recommended test query:
   ```text
   Compare FastAPI vs Flask for building production async microservices
   ```
4. Observe the live execution:
   - **Step 1:** Planner extracts entities (`FastAPI`, `Flask`) and dimensions (`Performance`, `Async Support`, `Architecture`).
   - **Step 2:** Round 1 searches and classifies official docs (`fastapi.tiangolo.com`, `flask.palletsprojects.com`).
   - **Step 3:** Evidence coverage matrix evaluates covered vs missing cells.
   - **Step 4:** Programmatic confidence calibration assigns calibrated rating with mathematical explanation.
   - **Step 5:** Token usage and cost tracking reports ~$0.0004 actual cost against the $0.10 request cap.
