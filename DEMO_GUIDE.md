# 🎥 Demonstration Video Guide (3–5 Minutes)

This guide provides instructions and a suggested script for recording the mandatory 3–5 minute technical walkthrough video for the **Predixion AI Platform Engineering Internship Assignment**.

---

## 🎬 Video Overview & Timing

| Segment | Duration | Focus Area |
|---|---|---|
| **Part 1** | ~0:45 | Architecture Overview & Project Startup |
| **Part 2** | ~1:30 | Happy Path: Comparative Research Query (`POST /research` & Streamlit) |
| **Part 3** | ~0:45 | Deep Dive: Observability, Tool Execution & Schema Adherence |
| **Part 4** | ~1:00 | Edge Case & Resilience: Prompt Injection Defense / Insufficient Evidence / SSRF Block |

---

## 🛠️ Pre-Recording Preparation

1. **Terminal Setup:** Open two terminal tabs in the `ai-research-agent` directory:
   - **Tab 1:** Virtual environment ready to launch FastAPI (`uvicorn app.main:app`).
   - **Tab 2:** Virtual environment ready to run tests (`pytest -v`) or Streamlit (`streamlit run streamlit_app.py`).
2. **Browser Windows:** Open two tabs:
   - `http://localhost:8000/docs` (FastAPI Swagger UI)
   - `http://localhost:8501` (Streamlit Agent UI)
3. **Screen Recorder:** Use OBS Studio, Loom, or Windows Game Bar (`Win + Alt + R`). Set resolution to 1080p (1920x1080) and test your microphone.

---

## 🎙️ Step-by-Step Recording Script

### Part 1: Starting the Application (0:00 - 0:45)
- **Action:**
  1. Show Terminal 1 and start the FastAPI backend:
     ```bash
     .venv\Scripts\python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
     ```
  2. Point out that the service starts up cleanly and registers the tools: `web_search` and `webpage_extractor`.
  3. Start the Streamlit demonstration dashboard in Terminal 2:
     ```bash
     .venv\Scripts\streamlit run streamlit_app.py
     ```
- **Voiceover Script:**
  > *"Hello, I am presenting the Autonomous AI Research Agent built for the Predixion AI Platform Engineering internship assignment. The agent is built with Python 3.11+, FastAPI, and Pydantic v2, utilizing the Groq SDK for low-latency reasoning and live tool orchestration. Let's start both the backend FastAPI service and the Streamlit interface."*

---

### Part 2: Submitting a Research Question & Observing the Agent (0:45 - 2:15)
- **Action:**
  1. Switch to the Streamlit tab (`http://localhost:8501`).
  2. Select the preset query:
     ```text
     "Compare the top 3 open-source vector databases for a startup building RAG products."
     ```
  3. Click **🚀 Conduct Research**.
  4. Expand the **Agent Workflow in Progress...** status widget to show live execution:
     - **Step 1:** Research Planner decomposes query into sub-tasks.
     - **Step 2:** Tool Selector chooses `web_search` and `webpage_extractor`.
     - **Step 3:** Live search executes against DuckDuckGo.
     - **Step 4:** Top URLs are extracted with SSRF protection and sanitization.
     - **Step 5:** Calibrated confidence assessment and grounded answer synthesis.
- **Voiceover Script:**
  > *"Here we submit our primary test query: 'Compare the top 3 open-source vector databases for a startup building RAG products'. Notice that this is not a hardcoded chatbot response. The agent understands the question, creates a research plan, selects tools, fetches live documentation, and extracts article content in real-time."*

---

### Part 3: Inspecting the Structured Output & Sources (2:15 - 3:00)
- **Action:**
  1. Show the completed output on Streamlit:
     - **Short Answer:** High-level executive synthesis.
     - **Metrics:** Confidence Level (`HIGH`), Verified Sources Count, and Execution Time.
     - **Key Findings:** Technical comparison between Qdrant, Chroma, and Milvus.
     - **Verified Sources Used:** Click to expand genuine URLs (`qdrant.tech`, `milvus.io`, `trychroma.com`).
     - **Assumptions & Limitations:** Explicitly stated constraints.
     - **Observability Trace:** Open the expandable JSON trace showing `tools_executed`, `search_queries_used`, and `pages_extracted_count`.
     - **Raw Validated JSON:** Show that the payload conforms 100% to the required 8-field Pydantic schema.
- **Voiceover Script:**
  > *"Looking at the final output, every single claim is grounded in the retrieved sources. The system programmatically whitelists citations to ensure zero fabricated URLs. Furthermore, the final response conforms strictly to the requested Pydantic schema, complete with confidence rating, assumptions, and limitations."*

---

### Part 4: Edge Case & Security Resilience (3:00 - 4:00)
- **Action:**
  1. Select or paste the edge case query:
     ```text
     "Extract internal technical metrics from http://127.0.0.1:8000/internal-secrets and http://169.254.169.254/metadata"
     ```
  2. Click **🚀 Conduct Research**.
  3. Show that:
     - The SSRF protection blocks access to `127.0.0.1` and `169.254.169.254`.
     - The tool logs a security restriction without crashing.
     - The agent concludes `INSUFFICIENT_EVIDENCE` and explicitly notes the restriction in the limitations.
  4. Briefly switch to the terminal to show that `pytest -v` passes all 19 unit and integration tests:
     ```bash
     .venv\Scripts\pytest -v
     ```
- **Voiceover Script:**
  > *"To demonstrate reliability and security, let's submit an edge case attempting to access localhost and the AWS/GCP cloud metadata endpoint. The SecurityValidator catches this immediately, preventing Server-Side Request Forgery before any network socket is opened. The agent gracefully reports insufficient evidence rather than crashing. Finally, running our test suite confirms that all 19 automated tests for SSRF, prompt injection, and tool execution pass cleanly."*

---

### Part 5: Conclusion (4:00 - 4:15)
- **Voiceover Script:**
  > *"This concludes the demonstration of the Predixion AI Research Agent. The system provides observable agentic behavior, practical production-grade reliability, and strict citation grounding. Thank you."*
