# PlantPilot — System Architecture & Data Flow

> **Status:** Stabilized & Production-Ready  
> **Last Updated:** 2026-10-26  
> **Related Documents:** [[PLANTpilot HQ]] | [[DEVELOPMENT]] | [[EXPERIMENT & TEST RESULTS]]

---

## 1. High-Level Product Architecture

PlantPilot is designed as a modular, human-supervised industrial operations AI framework with strict decoupling between the presentation layer, the application/investigation controller, the external intelligence provider, and the underlying telemetry persistence.

```mermaid
graph TB
    subgraph Client ["Operator Presentation Layer"]
        UI["Streamlit Dashboard (frontend/app.py)"]
        V1["Operations HQ"]
        V2["Incident Mgmt"]
        V3["Investigation Center"]
        V4["Human Sign-Off"]
        V5["Telemetry & Detector"]
        V6["Eval Benchmarks"]
        UI --- V1 & V2 & V3 & V4 & V5 & V6
    end

    subgraph Backend ["Application Gateway & Controller (FastAPI)"]
        API["REST Router (/api/v1)"]
        Agent["Agent Controller (agent.py)"]
        Safety["Python Safety Policy (safety.py)"]
        Tools["Tool Registry (tools.py)"]
        Adapter["Data Adapter (data_adapter.py)"]
        Detector["Anomaly Detector (detector.py)"]
    end

    subgraph LLM ["AI Intelligence Provider (NVIDIA NIM)"]
        NvidiaAPI["NVIDIA Nemotron 3.5 Super VL 120B\n(integrate.api.nvidia.com)"]
    end

    subgraph Persistence ["Persistence & Dataset"]
        DB[("SQLite Database\nplantpilot.db (9 tables)")]
        CSV[("AI4I 2020 Dataset\n(10,000 Records)")]
        Eval["Evaluation Engine (eval/)"]
    end

    %% Wiring
    UI <-->|HTTP REST / JSON| API
    API <--> DB
    API -->|Synchronous Run| Agent
    Agent <-->|Bounded Tool Loop| NvidiaAPI
    Agent -->|Validate Tool Calls| Safety
    Safety -->|Authorized Calls| Tools
    Tools -->|Physical Telemetry| Adapter
    Adapter -->|Quarantined Sensor Data| CSV
    Tools -->|Summary Stats| Tools
    Tools -->|Persist Verified Evidence| DB
    Agent -->|Persist Hypotheses & Recs| DB
    Detector -->|Fit IQR References| CSV
    Eval -.->|Strict Label Evaluation Only| CSV

    style Client fill:#1e293b,stroke:#38bdf8,stroke-width:2px,color:#f8fafc
    style Backend fill:#0f172a,stroke:#3b82f6,stroke-width:2px,color:#f8fafc
    style LLM fill:#14532d,stroke:#22c55e,stroke-width:2px,color:#f8fafc
    style Persistence fill:#3b0764,stroke:#c084fc,stroke-width:2px,color:#f8fafc
```

---

## 2. Core Subsystems & Component Responsibilities

### 2.1 Sensor Data Adapter (`backend/app/plantpilot/data_adapter.py`)
- **Primary Responsibility:** Interfaces with historical machine telemetry (`ai4i2020.csv`, 10,000 milling machine records).
- **Strict Label Quarantine:** Exposes *only* agent-facing physical measurements:
  - `air_temperature` $[K]$
  - `process_temperature` $[K]$
  - `rotational_speed` $[rpm]$
  - `torque` $[Nm]$
  - `tool_wear` $[min]$
- **Zero Label Leakage:** Completely strips ground-truth failure labels (`Machine failure`, `TWF`, `HDF`, `PWF`, `OSF`, `RNF`) from the agent. Ground-truth failure labels are exclusively accessible by `get_evaluation_only()` in the isolated evaluation engine.
- **Lookup Support:** Supports dual lookup via integer `UDI` ($1 \dots 10,000$) or string `Product ID` (e.g. `L47181`). Supports parameter aliases (e.g. `air_temp` $\to$ `air_temperature`, `rpm` $\to$ `rotational_speed`).

### 2.2 Anomaly Detection Engine (`backend/app/plantpilot/detector.py`)
- **Primary Responsibility:** Evaluates real-time or historical telemetry for multi-signal equipment stress without utilizing failure labels.
- **IQR Thresholding:** Calculates distribution quartiles ($Q1$, $Q3$, $\text{IQR} = Q3 - Q1$) across the 7,000-record training split:
  $$\text{Lower Bound} = Q1 - 1.5 \times \text{IQR}, \quad \text{Upper Bound} = Q3 + 1.5 \times \text{IQR}$$
- **Multi-Signal Anomaly Flag:** Flags an anomaly when $\ge 2$ physical parameters fall outside IQR bounds simultaneously, or when cumulative `tool_wear > 200 min` (a documented experimental heuristic).

### 2.3 AI Agent Controller (`backend/app/plantpilot/agent.py`)
- **Primary Responsibility:** Drives the bounded diagnostic investigation loop using the NVIDIA Nemotron model.
- **Bounded Investigation Loop:**
  - Maximum 8 reasoning steps and maximum 20 tool executions per incident.
  - **Step 0 Mandatory Diagnostics:** Forces `tool_choice="required"` on the initial step, ensuring the LLM cannot conclude without empirical tool execution.
  - Subsequent steps switch to `tool_choice="auto"`.
- **Evidence-First Persistence:** Telemetry returned by tools is validated and stored as `EvidenceType.OBSERVED` with `verified=True`. If no tools execute or no observations are recorded, the investigation cannot be marked `COMPLETED` and terminates as `INCOMPLETE`.
- **Fault-Tolerant Provider Handling:** Implements bounded 1-shot retry exclusively for qualifying transient upstream HTTP 500 errors.

### 2.4 Safety Policy Gatekeeper (`backend/app/plantpilot/safety.py`)
- **Primary Responsibility:** Independent, pure-Python security boundary protecting operational assets from unverified or risky LLM actions.
- **Risk Level Hierarchy:**
  - `RiskLevel.NONE` (Read-only data retrieval & statistical math — Auto-approved)
  - `RiskLevel.LOW` (Status inspection & logging — Auto-approved)
  - `RiskLevel.MEDIUM` (Diagnostic configuration adjustments — Operator approval required)
  - `RiskLevel.HIGH` (Setpoint modifications & machine recalibration — Operator approval required)
  - `RiskLevel.CRITICAL` (Emergency shutdowns, spindle halts — Operator approval required)
- **Human-in-the-Loop Enforcement:** Intercepts model-proposed recommendations and forces `requires_approval=True` and `approval_status=PENDING_APPROVAL`.

### 2.5 Storage Layer (`backend/app/plantpilot/storage.py`)
- **Primary Responsibility:** Relational SQLite persistence (`data/plantpilot.db`) with `PRAGMA foreign_keys = ON`.
- **Relational Tables:** Manages 9 tables: `machines`, `incidents`, `investigations`, `evidence`, `hypotheses`, `hypothesis_evidence`, `recommendations`, `tool_executions`, `audit_events`.
- **Cascading Deletions:** Deleting an incident cleanly cascades to all child investigations, evidence, and audit logs.
- **Dual Hypothesis-Evidence Junction:** Maintains a dedicated junction table (`hypothesis_evidence`) for granular relationships with automatic fallback to JSON fields.

### 2.6 Operator Dashboard (`frontend/app.py`)
- **Primary Responsibility:** Streamlit presentation layer connecting plant operators to the backend.
- **8 Dedicated Modules:** Operations HQ, Incident Management, AI Investigation Center, Human Sign-Off, Fleet Registry, Anomaly Detector Explorer, Evaluation Benchmarks, and System Health Diagnostics.

---

## 3. End-to-End Investigation Sequence

```mermaid
sequenceDiagram
    autonumber
    actor Operator as Plant Operator
    participant UI as Streamlit Dashboard
    participant API as FastAPI (/api/v1)
    participant Agent as AgentController
    participant Safety as SafetyPolicy
    participant Registry as ToolRegistry
    participant Adapter as DataAdapter (ai4i2020.csv)
    participant NVIDIA as NVIDIA Nemotron API
    participant DB as SQLite Storage

    Operator->>UI: Submit Incident ("Spindle overheating on Station 14")
    UI->>API: POST /api/v1/incidents
    API->>DB: Save Incident (status=PENDING)
    API-->>UI: Return Incident (inc_xxxx)

    Operator->>UI: Click "Start AI Investigation"
    UI->>API: POST /api/v1/incidents/{id}/investigations
    API->>Agent: investigate(investigation, incident)
    
    rect rgb(30, 41, 59)
        Note over Agent,NVIDIA: Step 0: Forced Diagnostic Tool Call
        Agent->>NVIDIA: chat.completions.create(tool_choice="required")
        NVIDIA-->>Agent: tool_calls: [get_sensor_readings(udi=14)]
        Agent->>Safety: check_tool_call("get_sensor_readings", args)
        Safety-->>Agent: Approved (RiskLevel.NONE)
        Agent->>Registry: execute("get_sensor_readings", args)
        Registry->>Adapter: get_sensor_readings(station_id=14)
        Adapter-->>Registry: Telemetry (air_temp, speed, torque, wear)
        Registry-->>Agent: Tool Result (JSON string)
        Agent->>DB: Save ToolExecution & Verified EvidenceItem
    end

    rect rgb(15, 23, 42)
        Note over Agent,NVIDIA: Step 1: Statistical Analysis & Hypothesis
        Agent->>NVIDIA: chat.completions.create(tool_choice="auto")
        NVIDIA-->>Agent: tool_calls: [diagnostic_math(values=[...], metric="mean")]
        Agent->>Registry: execute("diagnostic_math", args)
        Registry-->>Agent: Statistical Summary
        Agent->>DB: Save ToolExecution & EvidenceItem
    end

    rect rgb(20, 83, 45)
        Note over Agent,DB: Step 2: Synthesis & Human Gate
        Agent->>NVIDIA: chat.completions.create(tool_choice="auto")
        NVIDIA-->>Agent: Final Text: Findings + Hypotheses + Recommendations
        Agent->>Safety: check_recommendation(rec)
        Safety-->>Agent: Validated (requires_approval=True)
        Agent->>DB: Save Hypothesis, Recommendations, Audit Events
        Agent->>DB: Update Investigation (status=COMPLETED)
    end

    Agent-->>API: InvestigationResult (completed=True, steps=3)
    API-->>UI: Full Investigation Payload
    UI-->>Operator: Display Verified Evidence, Hypotheses & Action Approval Modal
```

---

## 4. Investigation Lifecycle State Machine

```mermaid
stateDiagram-v2
    [*] --> PENDING: Incident Created
    PENDING --> IN_PROGRESS: POST /api/v1/incidents/{id}/investigations
    
    state IN_PROGRESS {
        [*] --> Step0_ForcedTool: tool_choice="required"
        Step0_ForcedTool --> ToolExecution: LLM outputs tool_calls
        Step0_ForcedTool --> Step0_Failed: LLM outputs prose only
        
        ToolExecution --> ValidateArgs: SafetyPolicy + Schema Check
        ValidateArgs --> ExecuteTool: Valid args & risk permitted
        ValidateArgs --> ToolFailed: Invalid UDI / empty values
        
        ExecuteTool --> PersistEvidence: Success (EvidenceType.OBSERVED)
        ExecuteTool --> ToolFailed: Tool runtime error
        
        PersistEvidence --> StepN_Loop: tool_choice="auto" (max 8 steps)
        StepN_Loop --> ToolExecution: Additional diagnostics needed
        StepN_Loop --> ReasonAndConclude: Model delivers diagnosis
    }

    PersistEvidence --> COMPLETED: Evidence gathered + reasoning provided + recs queued
    Step0_Failed --> INCOMPLETE: No tool execution attempted
    ToolFailed --> INCOMPLETE: Max steps exceeded without verified evidence
    IN_PROGRESS --> INCOMPLETE: Upstream Provider 500 (after 1 retry)
    IN_PROGRESS --> FAILED: Unrecoverable internal system error

    COMPLETED --> [*]
    INCOMPLETE --> [*]
    FAILED --> [*]
```

---

## 5. Database Schema & Relational Entity Model

```mermaid
erDiagram
    MACHINES ||--o{ INCIDENTS : "mach_id"
    INCIDENTS ||--o{ INVESTIGATIONS : "incident_id (CASCADE)"
    INVESTIGATIONS ||--o{ EVIDENCE : "investigation_id (CASCADE)"
    INVESTIGATIONS ||--o{ HYPOTHESES : "investigation_id (CASCADE)"
    INVESTIGATIONS ||--o{ RECOMMENDATIONS : "investigation_id (CASCADE)"
    INVESTIGATIONS ||--o{ TOOL_EXECUTIONS : "investigation_id (CASCADE)"
    INVESTIGATIONS ||--o{ AUDIT_EVENTS : "investigation_id (CASCADE)"
    TOOL_EXECUTIONS ||--o{ EVIDENCE : "tool_execution_id (SET NULL)"
    HYPOTHESES ||--o{ HYPOTHESIS_EVIDENCE : "hypothesis_id (CASCADE)"
    EVIDENCE ||--o{ HYPOTHESIS_EVIDENCE : "evidence_id (CASCADE)"

    MACHINES {
        string id PK "mach_<hex12>"
        string product_id UK "e.g. L47181"
        string type "L, M, H"
        string status "OPERATIONAL, DEGRADED, OFFLINE"
        string created_at
    }

    INCIDENTS {
        string id PK "inc_<hex12>"
        string machine_id FK
        string description
        string severity "LOW, MEDIUM, HIGH, CRITICAL"
        string category
        string status "PENDING, INVESTIGATING, RESOLVED"
        string metadata_json
        string created_at
        string updated_at
    }

    INVESTIGATIONS {
        string id PK "inv_<hex12>"
        string incident_id FK
        string status "PENDING, IN_PROGRESS, COMPLETED, INCOMPLETE"
        string findings_json
        string started_at
        string completed_at
        string notes
    }

    EVIDENCE {
        string id PK "ev_<hex12>"
        string investigation_id FK
        string tool_execution_id FK
        string evidence_type "OBSERVED, DERIVED, HYPOTHESIS"
        string parameter_name "air_temp, torque, wear, etc."
        float numeric_value
        string unit "K, rpm, Nm, min"
        int verified "1 = True, 0 = False"
        string description
        string source "ai4i2020.csv"
        string timestamp
    }

    HYPOTHESES {
        string id PK "hyp_<hex12>"
        string investigation_id FK
        string title
        string explanation
        string confidence "LOW, MEDIUM, HIGH"
        string supporting_evidence_json
        string contradicting_evidence_json
        string created_at
    }

    RECOMMENDATIONS {
        string id PK "rec_<hex12>"
        string investigation_id FK
        string title
        string description
        int requires_approval "Always 1 (True)"
        string approval_status "PENDING_APPROVAL, APPROVED, REJECTED"
        string priority "LOW, MEDIUM, HIGH, URGENT"
        string reviewed_by "Operator ID / Name"
        string review_notes
        string reviewed_at
    }

    TOOL_EXECUTIONS {
        string id PK "exec_<hex12>"
        string investigation_id FK
        string tool_name "get_sensor_readings, diagnostic_math"
        string arguments_json
        string result
        string status "SUCCESS, FAILED"
        string error_message
        int step_index
        string executed_at
    }
```

---

## 6. Architectural Invariants & Guarantees

1. **Decoupled Data Source:** The core agent controller depends solely on the abstract `ToolRegistry` and `SafetyPolicy`. Swapping `ai4i2020.csv` for live OPC-UA or MQTT factory feeds requires zero changes to the LLM agent prompt or bounded loop logic.
2. **Deterministic Fallbacks:** The framework never depends on LLM reasoning for arithmetic or statistical calculation; all means, standard deviations, and IQR outlier boundaries are calculated by deterministic Python functions.
3. **No Unsupervised Actuation:** Recommendations are strictly proposals. The backend contains no automated actuation endpoint; every corrective intervention requires explicit human operator sign-off via `POST /api/v1/recommendations/{id}/review`.