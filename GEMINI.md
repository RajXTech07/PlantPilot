# PlantPilot — Codebase Reference (GEMINI.md)

> **Purpose:** Persistent reference for AI-assisted development sessions. Consult this document first before inspecting source files. Last verified against the codebase on **2026-10-27**.

---

## 1. Project Overview

PlantPilot is an **evidence-first, human-supervised industrial operations AI copilot**. It investigates machine incidents using historical dataset evidence, structured tool calling via the NVIDIA API, and isolated evaluation. The system enforces that no diagnosis is produced without verified observations.

- **Framework:** FastAPI + Pydantic v2 + SQLite (`sqlite3`) + OpenAI SDK (pointed at NVIDIA `https://integrate.api.nvidia.com/v1`)
- **Dataset:** AI4I 2020 Predictive Maintenance (10,000 records)
- **LLM:** `nvidia/nemotron-3.5-super-vl-120b-a12b` (configurable)
- **Python:** ≥ 3.13
- **Package Manager:** `uv`
- **Status:** Fully stabilized across all 4 phases (**183 tests passing**, 100% pass rate); frontend implemented (Streamlit operator dashboard with Digital Twin & What-If Simulator, Workforce Clearance Compliance, and Diagnostics Audit)

---

## 2. Project Structure

```
PlantPilot/
├── backend/                          # All backend source, data, tests
│   ├── app/plantpilot/               # Main Python package
│   │   ├── __init__.py               # Package exports
│   │   ├── main.py                   # FastAPI application & routes
│   │   ├── agent.py                  # Agent controller (bounded investigation loop & state tracking)
│   │   ├── config.py                 # Environment-based configuration
│   │   ├── schemas.py                # Pydantic data models
│   │   ├── storage.py                # SQLite persistence layer (894 lines)
│   │   ├── tools.py                  # Tool registry & handlers (telemetry, ML risk, twin, workforce)
│   │   ├── safety.py                 # Safety policy enforcement
│   │   ├── data_adapter.py           # CSV-backed sensor data adapter
│   │   ├── detector.py               # Multi-signal anomaly detector (IQR + heuristic)
│   │   ├── detection.py              # Supervised ML failure risk inference (HistGradientBoosting + RandomForest)
│   │   ├── digital_twin.py           # Digital twin store & isolated what-if simulation records
│   │   ├── diagnostics.py            # Proof-or-refuse validator, self-auditing reports & next-best-test ranking
│   │   ├── workforce.py              # Backward-compatible workforce facade
│   │   ├── workforce/                # Modular Workforce Intelligence Subsystem
│   │   │   ├── __init__.py           # Subsystem package exports
│   │   │   ├── models.py             # Pydantic domain models (WorkerProfile, Attendance, Shifts, WorkOrders)
│   │   │   ├── registry.py           # Worker registry & L1-L4 clearance gating
│   │   │   ├── attendance.py         # Shift clocking, break deductions & overnight shift calculations
│   │   │   ├── scheduling.py         # Shift schedules, coverage verification & handovers
│   │   │   ├── maintenance.py        # Work order lifecycle & inspector evidence verification
│   │   │   ├── analytics.py          # Operational anomaly detection & explainable rule findings
│   │   │   ├── correlation.py        # Incident correlation across workers, maintenance & handovers
│   │   │   └── service.py           # Unified service orchestrator & SQLite migrations
│   │   └── eval/                     # Evaluation engine (isolated from agent)
│   │       ├── __init__.py
│   │       ├── split.py              # Reproducible stratified dataset split
│   │       ├── metrics.py            # Binary classification metrics
│   │       └── baseline.py           # Deterministic baseline predictor
│   ├── tests/                        # 183 automated tests (20 test modules)
│   │   ├── conftest.py               # Global fixtures (state reset & test config)
│   │   ├── test_agent.py             # Agent controller unit tests
│   │   ├── test_api.py               # FastAPI endpoint tests
│   │   ├── test_detection.py         # ML prediction & normalization tests
│   │   ├── test_diagnostics.py       # Evidence sufficiency, report audit & ranking tests
│   │   ├── test_digital_twin.py      # Digital twin store & isolation tests
│   │   ├── test_eval_engine.py       # Evaluation engine tests
│   │   ├── test_evidence_first.py    # Evidence-first workflow tests (largest)
│   │   ├── test_integration_e2e.py   # End-to-end integration tests
│   │   ├── test_investigation_state.py # Investigation state, budget tracking & deduplication tests
│   │   ├── test_next_best_test.py    # Next-Best-Test ranking, cost/risk penalty tests
│   │   ├── test_proof_or_refuse.py   # Strict proof-or-refuse validator tests
│   │   ├── test_report_auditing.py   # Self-auditing report grounding & causality tests
│   │   ├── test_safety.py            # Safety policy tests
│   │   ├── test_schemas.py           # Pydantic schema validation tests
│   │   ├── test_storage.py           # SQLite storage layer tests
│   │   ├── test_tools.py             # Tool registry & handler tests
│   │   ├── test_workforce.py         # Legacy workforce facade tests
│   │   ├── test_workforce_api.py     # Workforce & diagnostic REST endpoint tests
│   │   ├── test_workforce_integration.py # Incident-workforce correlation & clearance gating tests
│   │   └── test_workforce_modular.py # Modular workforce subsystem unit tests
│   ├── data/                         # Persistent data (gitignored DB)
│   │   ├── plantpilot.db             # SQLite database
│   │   ├── ai4i2020.csv              # Dataset (also at backend root)
│   │   └── README.md                 # Data governance notes
│   ├── ai4i2020.csv                  # Dataset (fallback location)
│   ├── test_nemotron.py              # Standalone NVIDIA API integration script
│   ├── pyproject.toml                # Dependencies & config
│   ├── uv.lock                       # Lock file
│   ├── .env                          # Secrets (gitignored)
│   └── .env.example                  # Template with all env vars
├── docs/                             # Obsidian knowledge base
│   ├── Architecture/ARCHITECTURE.md
│   ├── Dashboard/PLANTpilot HQ.md
│   ├── Development/DEVELOPMENT.md
│   └── Experiments/EXPERIMENT & TEST RESULTS.md
├── frontend/                         # Streamlit operator dashboard (8 views)
│   ├── app.py                        # Streamlit main entrypoint & navigation
│   ├── api_client.py                 # Backend API client
│   ├── styles.css                    # Dark-mode styling
│   └── views/                        # Overview, Incidents, Investigations, Approvals, Fleet, Detector, Eval, System Health
├── README.md                         # Project README
├── .gitignore                        # Git exclusions
└── GEMINI.md                         # This file
```

---

## 3. Architecture & Data Flow

```mermaid
graph TD
    A["Operator / API Client"] -->|POST /api/v1/incidents| B["FastAPI (main.py)"]
    B -->|create_incident| C["SQLite Storage (storage.py)"]
    B -->|POST .../investigations| D["AgentController (agent.py)"]
    D -->|chat.completions.create| E["NVIDIA API (Nemotron)"]
    E -->|tool_calls| D
    D -->|validate + execute| F["ToolRegistry (tools.py)"]
    F -->|get_sensor_readings| G["DataAdapter (data_adapter.py)"]
    G -->|reads| H["ai4i2020.csv"]
    F -->|diagnostic_math| I["Deterministic Math"]
    D -->|check_tool_call| J["SafetyPolicy (safety.py)"]
    D -->|persist evidence, hypotheses, recommendations| C
    D -->|InvestigationResult| B
    B -->|JSON response| A

    K["Detector (detector.py)"] -->|IQR + heuristic| H
    L["Eval Engine (eval/)"] -->|stratified split, metrics| H

    style D fill:#4a90d9,color:white
    style J fill:#e74c3c,color:white
    style C fill:#27ae60,color:white
    style G fill:#f39c12,color:white
```

### Investigation Lifecycle

1. **Incident created** via API → persisted with status `PENDING`
2. **Investigation started** → AgentController runs bounded loop (max 8 steps)
3. **Step 0:** `tool_choice="required"` forces the model to call a diagnostic tool
4. **Tool execution:** Validated by ToolRegistry → checked by SafetyPolicy → executed
5. **Evidence persisted** as `EvidenceType.OBSERVED` with `verified=True` only on successful execution
6. **Model reasons** over evidence → generates hypotheses and recommendations
7. **Completion rules:**
   - `COMPLETED` — tools executed + evidence gathered + reasoning provided
   - `INCOMPLETE` — no tools executed, max steps/calls exceeded, or provider failure
8. **Recommendations** always require human approval (`requires_approval=True`)

---

## 4. Key Modules

### 4.1 `config.py` — Configuration

| Variable | Default | Purpose |
|---|---|---|
| `NVIDIA_API_KEY` | *(required)* | NVIDIA API authentication |
| `NVIDIA_BASE_URL` | `https://integrate.api.nvidia.com/v1` | API endpoint |
| `PLANTPILOT_MODEL` | `nvidia/nemotron-3.5-super-vl-120b-a12b` | Model identifier |
| `PLANTPILOT_DATABASE_PATH` | `data/plantpilot.db` | SQLite path |
| `PLANTPILOT_MAX_STEPS` | `8` | Max agent loop iterations |
| `PLANTPILOT_MAX_TOOL_CALLS` | `20` | Max tool calls per investigation |
| `PLANTPILOT_MAX_INPUT_SIZE` | `8000` | Max input size (bytes) |
| `PLANTPILOT_MAX_OUTPUT_TOKENS` | `4000` | Max LLM output tokens |
| `PLANTPILOT_API_TIMEOUT_SECONDS` | `60.0` | API request timeout |
| `PLANTPILOT_TOOL_RESULT_MAX_BYTES` | `4000` | Tool result byte limit |
| `PLANTPILOT_TOOL_RESULT_MAX_CHARS` | `1200` | Tool result char limit |
| `PLANTPILOT_EVIDENCE_MAX_CHARS` | `2000` | Evidence description limit |
| `PLANTPILOT_EVIDENCE_MAX_BYTES` | `8000` | Evidence byte limit |
| `PLANTPILOT_MAX_MESSAGE_CHARS` | `12000` | Max message chars |

- Configuration is loaded once via `get_config()` (singleton pattern with `_config` module-level cache)
- `.env` is loaded via `python-dotenv`; process environment takes precedence
- `reset_config()` exists for test teardown

### 4.2 `schemas.py` — Data Models

All models use Pydantic v2 with `ConfigDict(populate_by_name=True)`.

| Model | Purpose | Key Fields |
|---|---|---|
| `Machine` / `MachineCreate` | Industrial asset registration | `id`, `product_id`, `type`, `status` |
| `Incident` / `IncidentCreate` | Operational incident | `description`, `severity`, `category`, `machine_id`, `metadata` |
| `Investigation` / `InvestigationCreate` | Investigation record | `incident_id`, `status`, `findings`, `evidence[]`, `hypotheses[]`, `tool_executions[]`, `recommendations[]` |
| `EvidenceItem` | Provenance-tracked observation | `evidence_type` (enum), `source`, `tool_execution_id` (FK), `parameter_name`, `numeric_value`, `unit`, `verified` |
| `Hypothesis` | Model-generated explanation | `title`, `explanation`, `confidence`, `supporting_evidence_ids[]`, `contradicting_evidence_ids[]` |
| `Recommendation` | Human-gated action proposal | `requires_approval` (always True), `approval_status`, `reviewed_by`, `review_notes`, `reviewed_at` |
| `ToolExecution` | Tool call audit record | `tool_name`, `arguments`, `result`, `status`, `step_index`, `input_size_bytes`, `output_size_bytes` |
| `InvestigationResult` | Agent return type | `investigation_id`, `status`, `evidence_ids[]`, `hypothesis_ids[]`, `recommendation_ids[]`, `steps_taken`, `completed` |

**Enums:**
- `Status`: `PENDING`, `IN_PROGRESS`, `COMPLETED`, `FAILED`, `INCOMPLETE`
- `EvidenceType`: `OBSERVED`, `DERIVED`, `HYPOTHESIS`, `SIMULATED`
- `ApprovalStatus`: `PENDING_APPROVAL`, `APPROVED`, `REJECTED`

### 4.3 `agent.py` — Agent Controller & Investigation State

- **Class:** `AgentController(db, tool_registry, safety_policy)`
- **Factory:** `create_agent_controller(db=None)` — uses defaults
- **Core method:** `investigate(investigation, incident) → InvestigationResult`
- **State Model:** `InvestigationState` in-memory lifecycle manager:
  - Tracks `active_hypotheses`, `verified_evidence_ids`, `contradicted_evidence_ids`, and `tool_call_count`
  - Maintains `executed_signatures: Set[str]` to deduplicate redundant tool calls across reasoning steps
  - Tracks strict step and tool call budgets (`has_budget()`)
  - Supports dynamic hypothesis revision (`_revise_hypotheses_with_evidence`) as newly verified observations arrive

**Investigation Loop Mechanics:**
- Bounded to `max_steps` (default 8) iterations
- Step 0 forces `tool_choice="required"` to ensure evidence gathering
- Subsequent steps use `tool_choice="auto"`
- Tool calls are validated, safety-checked, deduplicated, then executed
- **Evidence-first rule:** If no tool executes successfully → `INCOMPLETE`
- **Retry:** Exactly 1 retry for transient `InternalServerError` (provider `500`), only in early steps with no tool calls yet
- On successful completion: auto-generates a `Recommendation` (pending approval) and `Hypothesis` linked to evidence

**Error Handling:**
- Invalid JSON arguments → `ToolExecution(status=FAILED)`, error appended
- Safety rejection → `ToolExecution(status=FAILED)`, error appended
- `ToolError` → `FAILED` execution, does NOT set `tools_executed=True`
- Max tool calls exceeded → `INCOMPLETE`
- Max steps exceeded → `INCOMPLETE`

### 4.4 `tools.py` — Tool Registry

**Registered Tools:**

| Tool | Handler | Risk | Description |
|---|---|---|---|
| `get_sensor_readings` | `_sensor_data_handler` | `NONE` | Retrieves historical observation from AI4I 2020 by numeric UDI |
| `diagnostic_math` | `_diagnostic_math_handler` | `NONE` | Computes summary stats (count, sum, mean, min, max, std_dev, variance) |
| `predict_failure_risk` | `_predict_failure_risk_handler` | `NONE` | Supervised ML failure risk inference (HistGradientBoosting + RandomForest) |
| `get_twin_telemetry` | `_twin_telemetry_handler` | `NONE` | Retrieves status and telemetry snapshots from Digital Twin store |
| `assess_evidence_sufficiency` | `_assess_evidence_sufficiency_handler` | `NONE` | Evaluates evidence sufficiency and hypothesis verification deterministically |
| `rank_diagnostic_tests` | `_rank_diagnostic_tests_handler` | `NONE` | Multi-criteria ranking of candidate physical and sensor diagnostic tests |
| `verify_operator_clearance` | `_verify_operator_clearance_handler` | `NONE` | Validates plant personnel operational clearance levels (L1, L2, L3, L4) |
| `correlate_workforce_incident` | `_correlate_workforce_incident_handler` | `NONE` | Correlates machine incidents with active workers, unverified maintenance & shift handovers |
| `get_work_orders_for_machine` | `_get_work_orders_for_machine_handler` | `NONE` | Retrieves maintenance work orders and inspection verification status for a machine |



**`ToolRegistry` Features:**
- `validate_arguments()` — enforces JSON schema: required fields, type checks, `enum` constraints, `minItems`, `additionalProperties`, and input size limits
- `execute()` — validates then calls handler
- `to_openai_tools()` — exports as OpenAI-compatible tool definitions

**`_sensor_data_handler` Validation:**
- Rejects non-integer UDI, text/description passed as UDI, negative/zero UDI
- Maps `sensor_type` to agent-facing parameters
- Delegates to `data_adapter.get_sensor_readings()`

**`_diagnostic_math_handler` Validation:**
- Rejects empty `values` arrays
- Rejects boolean values in array
- Requires all values to be numeric (int or float)

### 4.5 `data_adapter.py` — Dataset Adapter

- **Source:** `ai4i2020.csv` (checks `data/` dir first, then backend root)
- **Agent-facing columns:** `air_temperature`, `process_temperature`, `rotational_speed`, `torque`, `tool_wear`
- **Quarantined labels:** `Machine failure`, `TWF`, `HDF`, `PWF`, `OSF`, `RNF` — **never exposed** to agent

**Key Functions:**
- `get_sensor_readings(station_id, parameter?, csv_path?)` — Returns measurements with units, provenance, and source metadata. Lookup by UDI (int) or Product ID (string).
- `get_evaluation_only(station_id, csv_path?)` — Separate function returning ground-truth labels. For evaluation engine only.
- `_load_csv(path)` — Validates CSV header and numeric columns

**Parameter Aliases:** `air_temp` → `air_temperature`, `speed`/`rpm` → `rotational_speed`, `wear` → `tool_wear`, etc.

### 4.6 `detector.py` — Anomaly Detector

- **Detection methods:**
  - **IQR-based:** Computes Q1/Q3/IQR bounds from training data; flags values outside `[Q1 - 1.5×IQR, Q3 + 1.5×IQR]`
  - **Heuristic rule:** `tool_wear > 200` (documented experimental heuristic, not validated industrial limit)
- **Multi-signal detection:** Flags when ≥ 2 parameters are anomalous
- **Functions:**
  - `fit_references(train_rows)` — Computes reference statistics from training split
  - `detect(row, references)` → `DetectorResult`
  - `compute_iqr_bounds(values)` → `(lower, upper)`
- **Does NOT use** ground-truth failure labels as inputs

### 4.7 `safety.py` — Safety Policy

- **`SafetyPolicy` class:** Pure Python gatekeeper, independent of LLM
- **Risk levels:** `NONE`, `LOW`, `MEDIUM`, `HIGH`, `CRITICAL`
- **Permitted risks:** Only `NONE` and `LOW` are auto-approved
- **`check_tool_call(name, args)`** — Rejects unregistered tools and tools above permitted risk
- **`check_recommendation(rec)`** — Enforces `requires_approval=True` and `PENDING_APPROVAL` status
- **Audit trail:** All decisions recorded in `_audit_events` list
- **`default_safety_policy()`** — Pre-registers both tools (`get_sensor_readings`, `diagnostic_math`) at `RiskLevel.NONE`

### 4.8 `storage.py` — SQLite Persistence

- **Database class:** Manages connection, schema, and all CRUD operations
- **Singleton:** `get_database()` / `reset_database()` for caching
- **FK enforcement:** `PRAGMA foreign_keys = ON`
- **Auto-migration:** `_migrate_columns()` adds missing columns to existing tables safely

**CRUD Operations:**
- Machines: `create_machine`, `get_machine`, `list_machines`
- Incidents: `create_incident`, `get_incident`, `list_incidents`
- Investigations: `create_investigation`, `get_investigation`, `update_investigation`
- Evidence: `save_evidence`, `_load_evidence`
- Hypotheses: `save_hypothesis`, `_load_hypotheses` (uses junction table `hypothesis_evidence`)
- Recommendations: `save_recommendation`, `review_recommendation`, `_load_recommendations`
- Tool Executions: `save_tool_execution`, `_load_tool_executions`
- Audit Events: `save_audit_event`

### 4.9 `eval/` — Evaluation Engine

Completely isolated from agent and investigation code.

- **`split.py`:** Reproducible stratified split (`seed=42`, 70/15/15 train/dev/eval). Asserts zero overlap between splits.
- **`metrics.py`:** `compute_metrics(predictions, actual)` → precision, recall, F1, confusion matrix
- **`baseline.py`:** `baseline_predict(rows)` — deterministic rule: `tool_wear > 200 → 1, else 0`. Reports `F1 = 0.177`.

### 4.10 `diagnostics.py` — Proof-or-Refuse, Report Audit & Next-Best-Test Engine

- **Proof-or-Refuse Diagnosis (`proof_or_refuse_diagnosis`)**:
  - Deterministic evaluation enforcing that no diagnosis is accepted without verified empirical observations.
  - Distinguishes mere hypotheses from confirmed conclusions.
  - Flags contradictory evidence across sensors (e.g. conflicting thermal readings).
  - Flags ambiguous root causes when multiple mutually exclusive failure modes share identical symptoms without distinguishing measurements.
  - Returns explicit refusal (`refused=True`) alongside actionable candidate next diagnostic tests.
- **Self-Auditing Reports (`audit_report`, `audit_investigation_record`)**:
  - Validates investigation findings and report claims directly against real SQLite database records.
  - Cross-references machine IDs, numerical values, units, and executed tool call logs.
  - Flags hallucinated sensor readings, claims citing failed tools as successful, unverified interventions, and unsupported causal assertions.
- **Next-Best-Test Engine (`rank_next_tests`)**:
  - Evaluates candidate diagnostic procedures based on information gain / diagnostic value, financial cost, duration, and safety risk.
  - Penalizes already performed tests and previously failed attempts.
  - Flags each test as `executable_by_system` / `is_executable_tool` (automated software tool call) vs. physical operator inspection requiring human clearance.

### 4.11 `workforce/` — Modular Workforce Intelligence Subsystem

Decoupled subsystem located in `backend/app/plantpilot/workforce/`:
- **`models.py`**: Pydantic models for `WorkerProfile`, `AttendanceRecord`, `ShiftSchedule`, `ShiftHandover`, `WorkOrder`, `WorkforceFinding`, `WorkforceIncidentCorrelation`.
- **`registry.py` (`WorkerRegistry`)**: Worker onboarding, certification validity checks, and strict `L1`-`L4` clearance level gating.
- **`attendance.py` (`AttendanceTracker`)**: Clock-in/out, automatic break deduction, net working hours calculation, overnight shift calculation across midnight, and duplicate/overlapping clock-in prevention.
- **`scheduling.py` (`ShiftScheduler`)**: Shift scheduling, staffing coverage calculation against actual attendance, and shift handover logging with safety notes.
- **`maintenance.py` (`MaintenanceManager`)**: Work order lifecycle (`PENDING` -> `IN_PROGRESS` -> `COMPLETED`), overdue maintenance flagging, and unverified completion detection (mandating L2+ inspector sign-off).
- **`analytics.py` (`WorkforceAnalytics`)**: Explainable, rule-based operational anomaly detection (excessive hours/fatigue, clearance mismatches, understaffing, overdue work orders).
- **`correlation.py` (`IncidentCorrelator`)**: Correlates machine incidents with active workers, unverified maintenance orders, and shift handovers within configurable operational time windows, generating investigation evidence items.
- **`service.py` (`WorkforceService`)**: Unified service orchestrator managing SQLite tables (`workforce_workers`, `workforce_attendance`, `workforce_schedules`, `workforce_handovers`, `workforce_orders`, `workforce_findings`, `workforce_correlations`) with automatic column migrations.
- **`workforce.py`**: Backward-compatible public facade exporting `WorkforceStore`, `get_workforce_store`, `reset_workforce_store`, and `model_assess_workforce`.

---

## 5. API Endpoints

All endpoints are prefixed with `/api/v1/` (except root and health).

| Method | Path | Request Body | Response | Description |
|---|---|---|---|---|
| `GET` | `/` | — | `{application, version, status}` | Root health check |
| `GET` | `/health` | `?check_provider=bool` | `{status, backend, provider?}` | Backend + optional provider health |
| `POST` | `/api/v1/incidents` | `IncidentCreate` | `Incident` (201) | Create incident |
| `GET` | `/api/v1/incidents` | `?limit=50&offset=0` | `List[Incident]` | List incidents (newest first) |
| `GET` | `/api/v1/incidents/{id}` | — | `Incident` | Get single incident |
| `POST` | `/api/v1/incidents/{id}/investigations` | `InvestigationCreate` | `InvestigationResult` | Start synchronous AI investigation |
| `GET` | `/api/v1/investigations` | `?limit=50&offset=0` | `List[Investigation]` | List investigations |
| `GET` | `/api/v1/investigations/{id}` | — | `Investigation` | Get investigation with full trace |
| `POST` | `/api/v1/investigations/{id}/audit` | — | `InvestigationAuditResponse` | Audit investigation findings vs DB records |
| `POST` | `/api/v1/machines` | `MachineCreate` | `Machine` (201) | Register machine |
| `GET` | `/api/v1/machines` | `?limit=50&offset=0` | `List[Machine]` | List machines |
| `GET` | `/api/v1/machines/{id}` | — | `Machine` | Get single machine |
| `GET` | `/api/v1/recommendations` | `?limit=50&offset=0` | `List[Recommendation]` | List recommendations |
| `POST` | `/api/v1/recommendations/{id}/review` | `RecommendationReviewRequest` | `Recommendation` | Operator approve/reject |
| `POST` | `/api/v1/twins` | `DigitalTwinCreate` | `DigitalTwin` (201) | Register digital twin |
| `GET` | `/api/v1/twins` | `?limit=50` | `List[DigitalTwin]` | List digital twins |
| `GET` | `/api/v1/twins/{id}` | — | `DigitalTwin` | Get twin with latest reading |
| `GET` | `/api/v1/twins/{id}/history` | `?limit=100` | `List[TwinReadingItem]` | Get twin telemetry history |
| `POST` | `/api/v1/twins/{id}/readings` | `TwinReadingCreate` | `TwinReadingItem` (201) | Push observed telemetry with auto-risk |
| `POST` | `/api/v1/twins/{id}/simulate` | `TwinSimulationRequest` | `TwinReadingItem` | Run isolated what-if simulation |
| `POST` | `/api/v1/predict-risk` | `PredictRiskRequest` | `FailureRiskPrediction` | Standalone ML failure risk inference |
| `POST` | `/api/v1/diagnostics/assess` | `DiagnosisAssessRequest` | `DiagnosisAssessResponse` | Assess evidence sufficiency |
| `POST` | `/api/v1/diagnostics/audit-report` | `ReportAuditRequest` | `ReportAuditResponse` | Audit claims in diagnostic report |
| `POST` | `/api/v1/diagnostics/rank-tests` | `RankTestsRequest` | `RankTestsResponse` | Rank candidate next diagnostic tests |
| `POST` | `/api/v1/diagnostics/proof-or-refuse` | `ProofOrRefuseRequest` | `ProofOrRefuseResponse` | Strict proof-or-refuse diagnosis validator |
| `POST` | `/api/v1/workforce/workers` | `WorkerCreate` | `WorkerItem` (201) | Register plant worker |
| `GET` | `/api/v1/workforce/workers` | `?limit=50&offset=0` | `List[WorkerItem]` | List registered workers |
| `GET` | `/api/v1/workforce/workers/{id}` | — | `WorkerItem` | Get worker profile |
| `POST` | `/api/v1/workforce/clock-in` | `ClockInRequest` | `AttendanceRecordItem` (201) | Clock worker into shift |
| `POST` | `/api/v1/workforce/clock-out` | `ClockOutRequest` | `AttendanceRecordItem` | Clock worker out of shift |
| `GET` | `/api/v1/workforce/workers/{id}/attendance` | — | `List[AttendanceRecordItem]` | Get worker attendance records |
| `POST` | `/api/v1/workforce/clearance-check` | `ClearanceCheckRequest` | `ClearanceCheckResponse` | Verify operator clearance level |
| `POST` | `/api/v1/workforce/detect-anomalies` | — | `WorkforceFindingsResponse` | Rule-based workforce fatigue/clearance anomalies |
| `POST` | `/api/v1/workforce/model-review` | `WorkforceReviewRequest` | `WorkforceReviewResponse` | AI-assisted workforce review |
| `POST` | `/api/v1/workforce/shifts/schedules` | `ShiftScheduleCreate` | `ShiftSchedule` (201) | Create planned shift schedule |
| `GET` | `/api/v1/workforce/shifts/schedules` | `?limit=50&offset=0` | `List[ShiftSchedule]` | List planned shift schedules |
| `GET` | `/api/v1/workforce/shifts/schedules/{id}/coverage` | — | `Dict[str, Any]` | Verify staffing coverage vs attendance |
| `POST` | `/api/v1/workforce/shifts/handovers` | `ShiftHandoverCreate` | `ShiftHandover` (201) | Log shift handover notes |
| `GET` | `/api/v1/workforce/shifts/handovers` | `?limit=50&offset=0` | `List[ShiftHandover]` | List shift handover logs |
| `POST` | `/api/v1/workforce/work-orders` | `WorkOrderCreate` | `WorkOrder` (201) | Create maintenance work order |
| `GET` | `/api/v1/workforce/work-orders` | `?limit=50&offset=0` | `List[WorkOrder]` | List maintenance work orders |
| `GET` | `/api/v1/workforce/work-orders/{id}` | — | `WorkOrder` | Get maintenance work order by ID |
| `POST` | `/api/v1/workforce/work-orders/{id}/complete` | `WorkOrderCompleteRequest` | `WorkOrder` | Complete work order with inspector verification |
| `POST` | `/api/v1/workforce/correlate` | `WorkforceCorrelateRequest` | `WorkforceIncidentCorrelation` | Correlate incident with workforce events |

**Notable:**
- Investigation endpoint is **synchronous** — runs the bounded agent loop before responding
- All unit and regression tests run **100% deterministically without external API dependencies**
- Limit range for list endpoints: `1–200`
- `ConfigError` has a dedicated exception handler returning 500

---

## 6. Database Schema

```mermaid
erDiagram
    machines {
        TEXT id PK
        TEXT product_id UK
        TEXT type
        TEXT status
        TEXT created_at
    }

    incidents {
        TEXT id PK
        TEXT machine_id FK
        TEXT description
        TEXT severity
        TEXT category
        TEXT status
        TEXT created_at
        TEXT updated_at
        TEXT metadata_json
    }

    investigations {
        TEXT id PK
        TEXT incident_id FK
        TEXT status
        TEXT findings_json
        TEXT started_at
        TEXT completed_at
        TEXT created_at
        TEXT notes
    }

    evidence {
        TEXT id PK
        TEXT investigation_id FK
        TEXT tool_execution_id FK
        TEXT evidence_type
        TEXT parameter_name
        REAL numeric_value
        TEXT unit
        TEXT description
        TEXT source
        TEXT value_json
        TEXT timestamp
        INTEGER verified
    }

    hypotheses {
        TEXT id PK
        TEXT investigation_id FK
        TEXT title
        TEXT explanation
        TEXT confidence
        TEXT supporting_evidence_json
        TEXT contradicting_evidence_json
        TEXT created_at
    }

    hypothesis_evidence {
        TEXT hypothesis_id PK_FK
        TEXT evidence_id PK_FK
        TEXT relationship_type PK
    }

    recommendations {
        TEXT id PK
        TEXT investigation_id FK
        TEXT title
        TEXT description
        INTEGER requires_approval
        TEXT approval_status
        TEXT priority
        TEXT reviewed_by
        TEXT review_notes
        TEXT reviewed_at
        TEXT created_at
        TEXT executed_at
    }

    tool_executions {
        TEXT id PK
        TEXT investigation_id FK
        TEXT tool_name
        TEXT arguments_json
        TEXT result
        TEXT status
        TEXT error_message
        TEXT executed_at
        INTEGER input_size_bytes
        INTEGER output_size_bytes
        INTEGER step_index
    }

    audit_events {
        TEXT id PK
        TEXT investigation_id FK
        TEXT event_type
        TEXT message
        TEXT details_json
        TEXT created_at
    }

    machines ||--o{ incidents : "machine_id"
    incidents ||--o{ investigations : "incident_id (CASCADE)"
    investigations ||--o{ evidence : "investigation_id (CASCADE)"
    investigations ||--o{ hypotheses : "investigation_id (CASCADE)"
    investigations ||--o{ recommendations : "investigation_id (CASCADE)"
    investigations ||--o{ tool_executions : "investigation_id (CASCADE)"
    tool_executions ||--o{ evidence : "tool_execution_id (SET NULL)"
    hypotheses ||--o{ hypothesis_evidence : "hypothesis_id (CASCADE)"
    evidence ||--o{ hypothesis_evidence : "evidence_id (CASCADE)"
```

**Indexes:** Created on `incidents(status)`, `incidents(machine_id)`, `investigations(incident_id)`, `evidence(investigation_id)`, `evidence(tool_execution_id)`, `hypotheses(investigation_id)`, `hypothesis_evidence(hypothesis_id)`, `hypothesis_evidence(evidence_id)`, `recommendations(investigation_id)`, `tool_executions(investigation_id)`.

**ID conventions:** `inc_<hex12>`, `inv_<hex12>`, `ev_<hex12>`, `hyp_<hex12>`, `rec_<hex12>`, `exec_<hex12>`, `evt_<hex12>`, `mach_<hex12>` — all UUID-based prefixed strings.

---

## 7. Setup & Development Commands

### Prerequisites
- Python ≥ 3.13
- `uv` package manager

### Installation
```bash
cd backend
uv sync    # Install dependencies from uv.lock
```

### Environment Setup
```bash
cp backend/.env.example backend/.env
# Edit .env and set NVIDIA_API_KEY
```

### Start Backend
```bash
cd backend
uv run uvicorn app.plantpilot.main:app --reload --port 8000
```

Alternative (as seen in running terminal):
```bash
cd backend
..\.venv\Scripts\python.exe -m uvicorn --app-dir app plantpilot.main:app --reload --port 8000
```

### Run Tests
```bash
cd backend
uv run pytest tests/ -v
```

### Run NVIDIA Integration Test (standalone)
```bash
cd backend
uv run python test_nemotron.py
```

---

## 8. Testing

### Test Suite Overview (183 tests across 20 modules)

| File | Tests | Category |
|---|---|---|
| `test_agent.py` | 9 | Agent controller unit tests (mocked provider, loop invariants, evidence verification) |
| `test_api.py` | 21 | FastAPI endpoints (Incidents, Machines, Approvals, Twins, Risk, Diagnostics, Workforce, Clearance) |
| `test_detection.py` | 8 | ML failure risk inference (HistGradientBoosting + RandomForest normalization) |
| `test_diagnostics.py` | 6 | Evidence sufficiency assessment, report claim audits, next-test ranking |
| `test_digital_twin.py` | 5 | Digital twin store, isolated what-if simulation, state rollback |
| `test_eval_engine.py` | 6 | Split reproducibility, metrics computation, deterministic baseline |
| `test_evidence_first.py` | 21 | Evidence-first workflow invariants, label quarantine & strict governance |
| `test_integration_e2e.py` | 11 | E2E integration: real CSV + real tools + real SQLite, mocked provider only |
| `test_investigation_state.py` | 5 | In-memory investigation state tracking, budget limits & call deduplication |
| `test_next_best_test.py` | 3 | Next-Best-Test ranking, information gain, cost/risk penalty & tool executable flags |
| `test_proof_or_refuse.py` | 5 | Strict proof-or-refuse validator, evidence sufficiency & contradiction detection |
| `test_report_auditing.py` | 4 | Self-auditing report claims grounding vs DB records & causality validation |
| `test_safety.py` | 6 | Pure Python safety policy, risk levels, auto-approval whitelisting |
| `test_schemas.py` | 13 | Pydantic v2 schemas validation, enums, field constraints |
| `test_storage.py` | 20 | SQLite storage CRUD, foreign key enforcement, migrations |
| `test_tools.py` | 16 | Tool registry, argument validation, ML and diagnostic tool handlers |
| `test_workforce.py` | 7 | Legacy workforce store facade, shift tracking, clearance gating & review |
| `test_workforce_api.py` | 6 | Workforce REST API endpoints (schedules, coverage, handovers, work orders, complete, correlate) |
| `test_workforce_integration.py` | 2 | Workforce incident correlation tool & clearance enforcement on critical actions |
| `test_workforce_modular.py` | 9 | Modular workforce subsystem (registry, attendance, overnight shifts, maintenance, analytics) |

### Test Fixtures
- **`conftest.py`:** Auto-use fixture `reset_state()` clears cached `_config` and `_db` singletons before/after each test
- Tests use `tmp_path` for isolated SQLite databases
- OpenAI client is **mocked** at the `client.chat.completions.create` boundary

### Running Subset
```bash
uv run pytest tests/test_tools.py -v          # Only tool tests
uv run pytest tests/ -k "test_evidence" -v    # Pattern match
```

---

## 9. Conventions & Patterns

### Naming
- **IDs:** Prefixed UUID hex fragments (`inc_`, `inv_`, `ev_`, `hyp_`, `rec_`, `exec_`, `evt_`, `mach_`)
- **Modules:** Lowercase snake_case filenames
- **Classes:** PascalCase (`AgentController`, `ToolRegistry`, `SafetyPolicy`)
- **Error classes:** Suffix with `Error` (`ToolError`, `ConfigError`, `AgentError`, `DatabaseError`, `DataAdapterError`)

### Architectural Decisions
1. **Evidence-first:** No investigation completes without at least one successful tool execution producing `OBSERVED` evidence
2. **Label quarantine:** Ground-truth failure labels (`Machine failure`, `TWF`, etc.) are strictly isolated from agent-facing interfaces; only accessible via `get_evaluation_only()`
3. **Safety is code, not LLM:** The `SafetyPolicy` is a Python gatekeeper — never asks the model whether an action is safe
4. **Human-in-the-loop:** All `Recommendation` objects require human approval; `requires_approval=True` and `approval_status=PENDING_APPROVAL` are enforced
5. **Bounded loops:** Step limit and tool call limit prevent infinite agent loops
6. **Audit everything:** Tool executions, errors, retries, and completion events are persisted as audit records
7. **Singleton caching:** `get_config()` and `get_database()` use module-level singletons with corresponding `reset_*()` for tests
8. **Provider-independent tools:** The `ToolRegistry` exports OpenAI format but is independent of the model provider

### Error Handling
- **Tool errors:** Caught and recorded as `ToolExecution(status=FAILED)` — investigation continues
- **API/provider errors:** Transient 500s get 1 retry; other errors break the loop
- **Config errors:** Raise `ConfigError`; caught by FastAPI exception handler
- **DB errors:** Raise `DatabaseError`

### Code Organization
- Every module has a docstring explaining purpose and boundaries
- JSON columns in SQLite suffixed with `_json` (e.g., `findings_json`, `metadata_json`, `arguments_json`)
- Datetime values stored as ISO 8601 strings in SQLite, parsed on read

---

## 10. Dataset: AI4I 2020

| Property | Value |
|---|---|
| Records | 10,000 |
| File | `backend/ai4i2020.csv` (also in `backend/data/`) |
| Source | AI4I 2020 Predictive Maintenance Dataset |

### Agent-Facing Columns (Measurements)

| Key | CSV Column | Unit |
|---|---|---|
| `air_temperature` | `Air temperature [K]` | K |
| `process_temperature` | `Process temperature [K]` | K |
| `rotational_speed` | `Rotational speed [rpm]` | rpm |
| `torque` | `Torque [Nm]` | Nm |
| `tool_wear` | `Tool wear [min]` | min |

### Evaluation-Only Columns (Ground Truth)

| Key | CSV Column | Values |
|---|---|---|
| `machine_failure` | `Machine failure` | 0/1 |
| `twf` | `TWF` (Tool Wear Failure) | 0/1 |
| `hdf` | `HDF` (Heat Dissipation Failure) | 0/1 |
| `pwf` | `PWF` (Power Failure) | 0/1 |
| `osf` | `OSF` (Overstrain Failure) | 0/1 |
| `rnf` | `RNF` (Random Failure) | 0/1 |

### Identifier Columns

| Column | Description |
|---|---|
| `UDI` | Unique Dataset Identifier (1-indexed integer) |
| `Product ID` | Product identifier string |
| `Type` | Quality type: `L`, `M`, or `H` |

### Evaluation Split
- **Method:** Stratified by `Machine failure` (0/1), `seed=42`
- **Proportions:** 70% train / 15% dev / 15% eval
- **Integrity:** Asserted zero UDI overlap between splits
- **Baseline F1:** `0.177` (rule: `tool_wear > 200 → failure`)

---

## 11. Known Issues & Technical Debt

### Provider Reliability
- Full live `COMPLETED` investigations require a stable NVIDIA API endpoint
- Intermittent provider `500` errors produce honest `INCOMPLETE` results (not manufactured successes)
- The system retries transient errors at most once in early steps

### Dataset Limitations
- Anomaly detector thresholds (IQR) are dataset-derived, not validated industrial limits
- The `tool_wear > 200` heuristic is documented as experimental, not a safety standard
- Dataset is reloaded per call in `data_adapter.py` (no in-memory caching)

### Architectural Gaps
- **No frontend:** `frontend/` directory is empty — operator dashboard is planned but unbuilt
- **No real-time data:** No simulator or real-time factory integration (historical CSV only)
- **No async investigations:** The investigation endpoint is synchronous; long-running investigations block the request
- **No authentication/authorization:** API endpoints have no auth layer
- **No rate limiting:** No request rate limiting on API endpoints

### Code Quality Items
- `test_nemotron.py` is at the backend root level (not in `tests/`) — standalone integration script
- `AGENT_PARAMS` is duplicated in both `tools.py` (line 20) and `data_adapter.py` (line 23)
- The `hypothesis_evidence` junction table coexists with JSON columns (`supporting_evidence_json`, `contradicting_evidence_json`) in the `hypotheses` table — loading falls back to JSON when junction table has no rows
- `storage.py` is the largest module (859 lines) and could benefit from splitting into sub-modules
- Some test files include `sys.path.insert(0, "app")` — possibly unnecessary given `pythonpath` in pytest config

### Missing Features (from Roadmap)
- [ ] Build operator dashboard (`frontend/`)
- [ ] Prepare final hackathon presentation
- [ ] Dataset caching in data adapter
- [ ] Async investigation support

---

## 12. Quick Reference for Common Tasks

### Adding a New Tool
1. Write handler function `_my_handler(arguments: Dict) → Any` in `tools.py`
2. Register in `_build_default_registry()` with `Tool(name=..., handler=..., argument_schema=..., risk_level=...)`
3. Register in `default_safety_policy()` in `safety.py` with matching `ToolRisk`
4. Add tests in `tests/test_tools.py`

### Adding a New API Endpoint
1. Add route in `main.py`
2. Add Pydantic schemas in `schemas.py` if needed
3. Add storage methods in `storage.py` if needed
4. Add tests in `tests/test_api.py`

### Modifying the Database Schema
1. Add columns to the `CREATE TABLE` statement in `storage.py` `_initialize_schema()`
2. Add migration entry in `_migrate_columns()` for backward compatibility
3. Update corresponding Pydantic models in `schemas.py`
4. Update CRUD methods in `storage.py`
5. Add/update tests in `tests/test_storage.py`

### Running a Single Investigation (via API)
```bash
# 1. Create an incident
curl -X POST http://localhost:8000/api/v1/incidents \
  -H "Content-Type: application/json" \
  -d '{"description": "High tool wear on UDI 1", "severity": "high"}'

# 2. Start investigation (use incident_id from response)
curl -X POST http://localhost:8000/api/v1/incidents/{incident_id}/investigations \
  -H "Content-Type: application/json" \
  -d '{"incident_id": "{incident_id}"}'
```

---

## 13. Verification Summary

| Area | Verified |
|---|---|
| Project structure & file paths | ✅ All paths match filesystem |
| Dependencies in `pyproject.toml` | ✅ fastapi, openai, pydantic, pytest, python-dotenv, uvicorn |
| Environment variables in `.env.example` | ✅ All documented |
| API endpoints in `main.py` | ✅ 11 endpoints verified |
| Database schema in `storage.py` | ✅ 9 tables, indexes, FK constraints |
| Tool registry (`get_sensor_readings`, `diagnostic_math`) | ✅ Both registered with schemas |
| Safety policy (risk levels, approval enforcement) | ✅ Both tools at `NONE` risk |
| Evidence-first enforcement in `agent.py` | ✅ `tool_choice="required"` on step 0, `INCOMPLETE` without evidence |
| Label quarantine in `data_adapter.py` | ✅ `get_sensor_readings` excludes failure labels |
| Eval engine isolation | ✅ Separate `eval/` module, stratified split, baseline F1=0.177 |
| Test suite structure | ✅ 10 test files, conftest with auto-reset fixtures |
| Documentation in `docs/` | ✅ Architecture, Dashboard, Development, Experiments |
| Frontend status | ✅ Empty directory (planned) |
