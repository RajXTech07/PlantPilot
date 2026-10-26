# PlantPilot / Operations HQ & Status Cockpit

> **Mission:** Build a human-supervised, evidence-first industrial operations AI copilot that investigates equipment anomalies, presents verified empirical findings, and recommends human-gated actions for plant reliability.

---

## 1. Executive Operations Cockpit

```mermaid
graph LR
    subgraph Status ["System Readiness Status"]
        A["FastAPI Backend\n🟢 Online (:8000)"]
        B["Streamlit Frontend\n🟢 Online (:8501)"]
        C["AI Agent Engine\n🟢 Live (NVIDIA Nemotron)"]
        D["Automated Tests\n🟢 100/100 Passed"]
        E["Safety Policy\n🟢 Enforcing Human Gate"]
        F["Dataset Adapter\n🟢 10k Rows Isolated"]
    end

    A --- B --- C --- D --- E --- F

    style A fill:#064e3b,stroke:#10b981,stroke-width:2px,color:#ecfdf5
    style B fill:#064e3b,stroke:#10b981,stroke-width:2px,color:#ecfdf5
    style C fill:#064e3b,stroke:#10b981,stroke-width:2px,color:#ecfdf5
    style D fill:#064e3b,stroke:#10b981,stroke-width:2px,color:#ecfdf5
    style E fill:#064e3b,stroke:#10b981,stroke-width:2px,color:#ecfdf5
    style F fill:#064e3b,stroke:#10b981,stroke-width:2px,color:#ecfdf5
```

### System Health & Readiness Metrics

| Component | Status | Metric / Evidence | Production Verification |
|:---|:---:|:---|:---|
| **FastAPI Backend** | 🟢 **STABLE** | HTTP 200 on `/health`, sub-50ms REST dispatch | `backend/app/plantpilot/main.py` |
| **Streamlit Dashboard** | 🟢 **STABLE** | 8 modular operational views running on port 8501 | `frontend/app.py` |
| **Investigation Loop** | 🟢 **ACTIVE** | Bounded max 8 steps, Step 0 forced diagnostic call | `backend/app/plantpilot/agent.py` |
| **Safety Policy** | 🟢 **ACTIVE** | 100% human sign-off required on recommendations | `backend/app/plantpilot/safety.py` |
| **Telemetry Adapter** | 🟢 **VERIFIED** | 10,000 records loaded, zero label leakage | `backend/app/plantpilot/data_adapter.py` |
| **Automated Test Suite** | 🟢 **100% PASS** | 100 passed tests in 4.69 seconds | `backend/tests/` |
| **Evaluation Baseline** | 🟢 **BENCHMARKED** | Stratified split (`seed=42`), Baseline F1 = 0.177 | `backend/app/plantpilot/eval/` |

---

## 2. Operator Dashboard Navigation Hierarchy

```mermaid
mindmap
  root((PlantPilot HQ))
    Operations HQ
      Active Fleet KPIs
      Open Incident Alerts
      Pending Sign-Off Counters
      Quick Action Launchers
    Incident Management
      Severity Filter (Low to Critical)
      Incident Timeline Log
      New Machine Incident Intake Form
    AI Investigation Center
      1-Click AI Run Trigger
      Step-by-Step Tool Trace Logs
      Verified Observed Evidence Cards
      Model Hypotheses & Evidence Links
    Human Operator Sign-Off
      Gated Recommendations Review
      Digital Approve / Reject
      Operator ID & Audit Notes
    Fleet & Asset Registry
      Machine Classifications (L, M, H)
      Operational Status Tracking
      Asset Profile Registration
    Dataset & Anomaly Detector
      AI4I 2020 Telemetry Explorer
      Interactive IQR Bounds Scoring
      Multi-Signal Stress Detection
    Evaluation & Benchmarks
      Stratified Split Distribution (70/15/15)
      Precision, Recall, F1 Baseline
      Label Quarantine Verification
    System Health & Diagnostics
      FastAPI Backend Ping
      NVIDIA LLM Connectivity Ping
      Database Integrity & Table Counts
```

---

## 3. Development Roadmap & Completion Checklist

- [x] **FastAPI Core Foundation**
  - [x] Decoupled backend architecture in `backend/`
  - [x] SQLite relational storage with `PRAGMA foreign_keys = ON`
  - [x] Incident registration and pagination endpoints (`/api/v1/incidents`)
  - [x] Investigation trigger and retrieval endpoints (`/api/v1/investigations`)
  - [x] Machine fleet asset management endpoints (`/api/v1/machines`)
  - [x] Human recommendation sign-off endpoint (`/api/v1/recommendations/{id}/review`)

- [x] **Evidence-First AI Agent & Safety**
  - [x] NVIDIA Nemotron-3.5-super-vl-120b integration via OpenAI SDK
  - [x] Step 0 forced diagnostic tool call (`tool_choice="required"`)
  - [x] Schema-validated tool registry with input sanitation (`ToolRegistry.validate_arguments`)
  - [x] Python safety gatekeeper with 5-tier risk grading (`safety.py`)
  - [x] Strict human-in-the-loop requirement (`requires_approval=True`)
  - [x] Bounded retry for transient upstream 500 provider errors

- [x] **Sensor Data & Evaluation Engine**
  - [x] AI4I 2020 predictive maintenance CSV adapter (`10,000` records)
  - [x] Complete label quarantine isolating failure labels from the agent
  - [x] Reproducible stratified dataset split (`seed=42`, 70% train / 15% dev / 15% eval)
  - [x] Deterministic baseline benchmark metrics (`F1 = 0.177`)

- [x] **Streamlit Operator Presentation Layer**
  - [x] Dark-mode industrial interface with custom CSS styling (`styles.css`)
  - [x] Live backend connectivity pill indicator
  - [x] 8 dedicated operational views covering operations, investigations, approvals, telemetry, and evaluation

- [x] **Quality Assurance & Verification**
  - [x] 100 automated pytest cases across unit, schema, safety, storage, and E2E integration suites
  - [x] Zero-leakage label tests validating physical telemetry isolation
  - [x] Comprehensive documentation knowledge base in `docs/` and root `GEMINI.md`

---

## 4. Core Operating Principles

```mermaid
flowchart TD
    P1["1. Evidence First\nNo diagnosis without verified observed telemetry"]
    P2["2. Strict Label Quarantine\nRaw physics for AI; failure labels for evaluation only"]
    P3["3. Human in the Loop\nHigh-consequence actions require operator sign-off"]
    P4["4. Deterministic Math\nCalculations performed by Python functions, not LLMs"]
    P5["5. Transparent Failure\nProvider outages flagged as INCOMPLETE, not hallucinated"]

    P1 --> P2 --> P3 --> P4 --> P5

    style P1 fill:#1e1b4b,stroke:#6366f1,stroke-width:2px,color:#e0e7ff
    style P2 fill:#1e1b4b,stroke:#6366f1,stroke-width:2px,color:#e0e7ff
    style P3 fill:#1e1b4b,stroke:#6366f1,stroke-width:2px,color:#e0e7ff
    style P4 fill:#1e1b4b,stroke:#6366f1,stroke-width:2px,color:#e0e7ff
    style P5 fill:#1e1b4b,stroke:#6366f1,stroke-width:2px,color:#e0e7ff
```

---

## 5. Knowledge Base Cross-References

- **System Architecture & Detailed Data Flows:** [[ARCHITECTURE]]
- **Chronological Engineering Logs & Milestones:** [[DEVELOPMENT]]
- **Scientific Experiments & Benchmark Evaluation:** [[EXPERIMENT & TEST RESULTS]]
- **Complete AI Pair-Programming Specification:** [GEMINI.md](file:///c:/Warning/hackathon/PlantPilot/GEMINI.md)