# PlantPilot — Chronological Engineering & Development Log

> **Context:** Complete chronological record of engineering decisions, technical challenges, refactoring milestones, and test suite progression for the PlantPilot framework.  
> **Related Documents:** [[ARCHITECTURE]] | [[PLANTpilot HQ]] | [[EXPERIMENT & TEST RESULTS]]

---

## 1. Milestone Timeline & Engineering Progression

```mermaid
timeline
    title PlantPilot Engineering Progression
    2026-10-24 : Core Architecture & Inception
               : FastAPI REST endpoints initialized
               : SQLite schema & Pydantic models designed
               : Initial test suite (40 tests passing)
    2026-10-25 : Evidence-First Enforcement & Diagnostics
               : Step 0 forced diagnostic tool call implemented
               : Schema validation added to ToolRegistry
               : AI4I 2020 dataset adapter with label quarantine built
               : Test suite expanded (64 -> 94 tests)
    2026-10-26 : Streamlit Operator Dashboard & 100% Verification
               : Monorepo decoupled into backend/, frontend/, docs/
               : 8 modular Streamlit operational views built
               : Foreign key cascades & junction table tests added
               : 100/100 automated test suite passing
```

---

## 2. Chronological Engineering Logs

### Phase 1: Core Framework & API Foundation (2026-10-24)
- **FastAPI Backend Setup:** Implemented modular REST routing under `/api/v1/` for machines, incidents, investigations, and operator reviews.
- **Relational Storage:** Created `backend/app/plantpilot/storage.py` managing SQLite persistence with parameterized SQL queries, UUID prefixing (`inc_`, `inv_`, `ev_`, `hyp_`, `rec_`), and JSON metadata fallback.
- **Initial Verification:** Passed 40 unit and API tests validating status codes, pagination, and data serialization.

### Phase 2: Evidence-First Enforcement & Input Sanitation (2026-10-25)
- **The Empty Value Failure:** During initial live testing, the agent invoked `diagnostic_math` with an empty array (`values=[]`). The diagnostic tool raised an unhandled exception, causing the investigation to abruptly abort.
- **Resolution:**
  - Implemented strict input validation via `ToolRegistry.validate_arguments()`.
  - Added array length constraints (`minItems=1`) and rejected boolean values in numeric arrays.
  - Implemented graceful failure handling: invalid tool invocations now produce a persisted `ToolExecution(status=FAILED)` rather than crashing the investigation loop.
- **Forced Diagnostic Step 0:** Enforced `tool_choice="required"` on the first iteration of `AgentController.investigate()`, preventing the model from outputting speculative text before executing diagnostic tools.

### Phase 3: Telemetry Adapter & Label Isolation (2026-10-25)
- **Dataset Integration:** Linked `backend/app/plantpilot/data_adapter.py` to the AI4I 2020 Predictive Maintenance dataset (10,000 records).
- **Label Quarantine Verification:** Formally separated the agent-facing sensor readings (`get_sensor_readings`) from the ground-truth failure labels (`get_evaluation_only`). The agent interface was audited to ensure zero exposure of `Machine failure`, `TWF`, `HDF`, `PWF`, `OSF`, or `RNF`.
- **Test Suite Growth:** Increased coverage from 64 to 94 passing tests.

### Phase 4: Monorepo Decoupling & Streamlit Dashboard (2026-10-26)
- **Architectural Separation:** Refactored project structure into a clean, decoupled layout:
  - `backend/` — FastAPI API, SQLite database, pytest suite.
  - `frontend/` — Streamlit dashboard and visual analytics.
  - `docs/` — Obsidian knowledge base and experiment reports.
- **Dashboard Implementation:** Built 8 production-ready Streamlit views with custom dark-mode styling (`styles.css`):
  1. *Operations HQ* (`views/overview.py`)
  2. *Incident Management* (`views/incidents.py`)
  3. *AI Investigation Center* (`views/investigations.py`)
  4. *Human Operator Sign-Off* (`views/approvals.py`)
  5. *Fleet & Asset Registry* (`views/fleet.py`)
  6. *Dataset & Anomaly Detector* (`views/detector_view.py`)
  7. *Evaluation & Benchmarks* (`views/eval_view.py`)
  8. *System Health & Tests* (`views/system_health.py`)

### Phase 5: Test Suite Expansion to 100 Tests (2026-10-26)
- **Storage Layer Hardening:**
  - Added tests validating `PRAGMA foreign_keys = ON` cascade deletions (deleting an incident automatically deletes all child investigations, evidence, and recommendations).
  - Added unit tests for the `hypothesis_evidence` junction table ensuring bidirectional mapping.
  - Added operator review auditing tests verifying timestamp, notes, and reviewer tracking.
- **Milestone Reached:** **100/100 tests passing in 4.69 seconds.**

---

## 3. Key Problem-Solving Case Studies

```mermaid
flowchart TD
    subgraph Case1 ["Case 1: Empty Array Handling"]
        A1["Problem: LLM passed values=[]\nto diagnostic_math"] --> B1["Root Cause: No minItems\nor schema checking"]
        B1 --> C1["Solution: ToolRegistry.validate_arguments\nenforces minItems=1 & numeric types"]
        C1 --> D1["Result: Graceful failure persisted\nas ToolExecution(status=FAILED)"]
    end

    subgraph Case2 ["Case 2: Label Leakage Prevention"]
        A2["Problem: Risk of LLM cheating\nusing ground-truth failure flags"] --> B2["Root Cause: Raw CSV contains\nfailure columns alongside telemetry"]
        B2 --> C2["Solution: Strict isolation in data_adapter.py;\nget_sensor_readings drops all label columns"]
        C2 --> D2["Result: Agent sees only physics;\neval engine gets isolated ground truth"]
    end

    subgraph Case3 ["Case 3: Upstream Provider Resilience"]
        A3["Problem: NVIDIA NIM endpoint\nintermittent HTTP 500 errors"] --> B3["Root Cause: External network/model\ninfrastructure latency"]
        B3 --> C3["Solution: 1-shot bounded retry\nfor qualifying transient 500s only"]
        C3 --> D3["Result: Transparent INCOMPLETE status\nrecorded with full audit trail"]
    end

    style Case1 fill:#1e293b,stroke:#38bdf8,stroke-width:1px,color:#f8fafc
    style Case2 fill:#1e293b,stroke:#10b981,stroke-width:1px,color:#f8fafc
    style Case3 fill:#1e293b,stroke:#f59e0b,stroke-width:1px,color:#f8fafc
```

---

## 4. Test Suite Progression & Velocity

```mermaid
xychart-beta
    title "Automated Test Suite Expansion"
    x-axis ["Phase 1: Core", "Phase 2: Validation", "Phase 3: Dataset", "Phase 4: Decoupling", "Phase 5: 100% Hardened"]
    y-axis "Tests Passing" 0 --> 120
    bar [40, 52, 64, 94, 100]
    line [40, 52, 64, 94, 100]
```

### Detailed Test Distribution

| Category | Test File | Test Count | Key Invariants Verified |
|:---|:---|:---:|:---|
| **Evidence-First** | `test_evidence_first.py` | 21 | Mandatory tool calls, empty input rejection, recovery |
| **Storage & Schema** | `test_storage.py` | 19 | Relational CRUD, FK cascades, junction table mapping |
| **API Endpoints** | `test_api.py` | 12 | Status codes, pagination, payload validation |
| **E2E Integration** | `test_integration_e2e.py` | 10 | Real CSV + real tools + mocked NVIDIA completions |
| **Tool Registry** | `test_tools.py` | 9 | Schema validation, statistical math calculations |
| **Pydantic Schemas** | `test_schemas.py` | 8 | Defaults, serialization, enum bounds |
| **Agent Controller** | `test_agent.py` | 7 | Bounded loop steps, tool call limits, provider handling |
| **Evaluation Engine** | `test_eval_engine.py` | 6 | Stratified split (seed=42), label isolation |
| **Safety Policy** | `test_safety.py` | 6 | Risk level grading, human approval enforcement |
| **Total** | | **100** | **100% Pass Rate in 4.69 seconds** |

---

## 5. Next Steps & Ongoing Engineering Focus

1. **Live Industrial Connector:** Create an OPC-UA / MQTT industrial bridge inheriting from `data_adapter.py` to stream live factory PLC data.
2. **Visual Anomaly Waveforms:** Add interactive time-series sensor charts using Plotly within the Streamlit Anomaly Detector view.
3. **Multi-Model Evaluator:** Benchmark alternate models (e.g. Llama-3.3-70B, DeepSeek-R1) against the identical evaluation split.