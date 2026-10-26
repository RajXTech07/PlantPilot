# PLANTPILOT / ARCHITECTURE

## Product Architecture

PlantPilot is designed as a modular, human-supervised industrial operations AI framework with decoupled backend and presentation layers.

```
PLANTpilot/
├── backend/          # FastAPI, Agent Controller, SQLite, Tools, Dataset
├── docs/             # Knowledge base, Architecture, Experiments, Roadmaps
└── frontend/         # Planned Operator Dashboard & Telemetry UI
```

## Core Components

### 1. Data Adapter (`backend/app/plantpilot/data_adapter.py`)

Provides operational readings to the framework. Currently reads from the AI4I 2020 Predictive Maintenance dataset (10,000 records). 
- Strictly quarantines ground-truth failure labels (`Machine failure`, `TWF`, `HDF`, etc.) from the agent.
- Exposes only agent-facing telemetry (`air_temperature`, `process_temperature`, `rotational_speed`, `torque`, `tool_wear`).

### 2. Anomaly Detection & Monitoring Engine (`backend/app/plantpilot/detector.py`)

Evaluates operational readings using dataset-derived reference statistics (IQR bounds fit on training split) and documented heuristics (`tool_wear > 200 min`). Never uses ground-truth failure labels as inputs.

### 3. AI Investigation Agent (`backend/app/plantpilot/agent.py`)

Orchestrates a bounded tool loop using the NVIDIA API endpoint.
- Enforces **evidence-first** investigation: requires diagnostic tool calls before concluding.
- Rejects ungrounded conclusions without verified evidence.
- Distinguishes observed facts from model hypotheses.

### 4. Safety Layer (`backend/app/plantpilot/safety.py`)

Pure Python gatekeeper independent of the LLM.
- Inspects and validates tool calls by risk level (`NONE`, `LOW`, `MEDIUM`, `HIGH`, `CRITICAL`).
- Enforces that all consequential recommendations require human approval (`requires_approval=True`, `pending_approval`).

### 5. Storage Layer (`backend/app/plantpilot/storage.py`)

Persists incidents, investigations, evidence, tool executions, and audit records in SQLite (`plantpilot.db`).

### 6. Operator Dashboard (Planned - `frontend/`)

Displays machine status, investigation evidence, diagnostic charts, and recommendations for human review.

## Intended Data Flow

Data Source (ai4i2020.csv)  
→ Monitoring / Detector  
→ Incident Intake (`/api/v1/incidents`)  
→ AI Investigation Agent (`agent.py`)  
→ Tool Registry Execution (`tools.py`)  
→ Evidence Collection (`EvidenceType.OBSERVED`)  
→ Hypotheses (unverified model claims)  
→ Recommendations (human-gated)  
→ Human Operator Review  
→ Audit Record Persisted in SQLite

## Resolved Gaps & Current State

- **Sensor-data retrieval path**: Implemented and verified via `data_adapter.py` and the `get_sensor_readings` tool.
- **Evidence-first enforcement**: Investigations without verified diagnostic tool executions terminate as `INCOMPLETE`.
- **Decoupled Architecture**: Backend isolated in `backend/`, ready for independent frontend integration.

## Architectural Principle

The investigation framework must remain independent of the data source or simulator so different data sources can be connected without rewriting the agent.