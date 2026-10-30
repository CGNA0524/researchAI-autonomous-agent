# 🤖 AI Tool Usage Disclosure

In compliance with engineering integrity guidelines and Section 13 of the Predixion AI Platform Engineering Internship Assignment, this document transparently details the use of AI tools throughout the conception, architecture, code generation, and testing of this project.

---

## 1. AI Tools Utilized

| AI Tool | Model / Version | Primary Purpose |
|---|---|---|
| **Antigravity IDE Agent** | Gemini 3.8 Flash | System architecture design, implementation of modular Python backend, automated test writing, security hardening, and documentation authoring. |
| **Groq Cloud API** | Llama 3.3 70B Versatile / Llama 3.1 8B Instant | In-agent reasoning engine used for question decomposition (planner), information synthesis, structured JSON formatting, and grounded citation generation. |

---

## 2. What Each Tool Helped With

### Antigravity AI Coding Assistant
1. **Architecture & Project Scaffolding:**
   - Designed modular single-agent architecture separating planning, tool execution, filtering, synthesis, and confidence evaluation.
   - Designed Pydantic v2 schemas (`ResearchRequest`, `ResearchResponse`, `ResearchPlan`, `SourceItem`).
2. **Tool Engineering & Security Hardening:**
   - Implemented dual-engine DuckDuckGo search (`html.duckduckgo.com` and `lite.duckduckgo.com`) with true destination URL unquoting.
   - Built SSRF validation logic in `app/utils/security.py` using Python's `ipaddress` library to block private IPs, loopback, and cloud metadata endpoints.
   - Designed prompt injection defanging to sanitize untrusted web content before LLM context ingestion.
3. **Automated Testing & Evaluations:**
   - Authored 19 comprehensive unit and integration tests with Pytest (`test_security.py`, `test_tools.py`, `test_orchestrator.py`, `test_api.py`).
   - Built automated test framework in `evaluations/run_evals.py` verifying all five required evaluation scenarios.

### Groq LLM API
- Decomposes ambiguous human questions into technical keywords and sub-tasks.
- Synthesizes dense web evidence into concise executive summaries and granular key findings.
- Generates structured Pydantic-compliant JSON outputs with zero hallucinations of external URLs.

---

## 3. Which Implementation Components Were AI-Assisted

- **Data Models:** `app/models/response.py`, `app/models/request.py`, `app/models/plan.py` (Structured final schema adhering strictly to Section 5).
- **Core Orchestrator:** `app/agent/orchestrator.py` (Step-by-step observable agent pipeline).
- **Agent Sub-Modules:**
  - `app/agent/planner.py` (Keyword extraction and multi-task planning).
  - `app/agent/filter.py` (Deduplication, keyword overlap calculation, authority weighting).
  - `app/agent/confidence.py` (Calibrated confidence levels based on evidence depth).
  - `app/agent/synthesizer.py` (Citation enforcement against genuine URLs).
- **Tool Suite:** `app/tools/web_search.py`, `app/tools/webpage_extractor.py`, `app/tools/registry.py`.
- **API & Presentation:** `app/main.py` (FastAPI routes) and `streamlit_app.py` (Streamlit dashboard).

---

## 4. Manual Review, Engineering Decisions & Validation

The following architectural decisions, inspections, and debugging were explicitly verified and validated:
1. **DuckDuckGo Library Deprecation & Rate Limiting Debugging:**
   - Identified that `duckduckgo_search` v8 triggers runtime warnings and returns empty results for long queries.
   - Replaced with direct async HTTP requests to DuckDuckGo HTML and Lite endpoints, with automatic fallback and URL parameter extraction (`uddg=`).
2. **SSRF Prevention Verification:**
   - Manually tested that loopback (`127.0.0.1`), metadata IP (`169.254.169.254`), and non-HTTP protocols (`file:///`, `ftp://`) are rejected before opening sockets.
3. **Citation Whitelisting & Grounding Logic:**
   - Verified that `AnswerSynthesizer` programmatically strips any citation whose URL was not in the actually retrieved sources list.
4. **Offline Resilience & Test Execution:**
   - Engineered deterministic fallbacks so the entire pipeline and test suite remain 100% testable and runnable even when offline or in environments without active Groq API credits.
5. **Windows Encoding & Compatibility:**
   - Debugged and resolved Windows `cp1252` terminal encoding issues to guarantee cross-platform execution on Windows, Linux, and macOS.
