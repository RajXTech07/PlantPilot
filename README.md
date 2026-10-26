# PlantPilot — Evidence-First Industrial Operations AI Copilot

A human-supervised backend framework for investigating operational incidents using historical dataset evidence, structured tool calling, and isolated evaluation.

## Project Structure

```text
PLANTpilot/
├── backend/              # FastAPI, AI agent controller, SQLite storage & test suite
│   ├── app/plantpilot/   # Application source code
│   ├── tests/            # Automated test suite (94 tests)
│   ├── data/             # Persistent SQLite database (plantpilot.db)
│   ├── ai4i2020.csv      # AI4I 2020 predictive maintenance dataset
│   └── pyproject.toml    # Backend dependencies & configuration
├── docs/                 # Obsidian knowledge base, architecture, and experiment logs
│   ├── Architecture/     # Architecture breakdown & data flow
│   ├── Dashboard/        # Project status & roadmap
│   ├── Development/      # Development logs & milestones
│   └── Experiments/      # Experiment results (001 - 003)
└── frontend/             # Operator Dashboard (planned)
```

## Verified State

- **Framework:** FastAPI + Pydantic + SQLite (`sqlite3`) + NVIDIA-compatible OpenAI SDK (`https://integrate.api.nvidia.com/v1`)
- **Dataset adapter:** `backend/app/plantpilot/data_adapter.py` — AI4I 2020 predictive maintenance CSV (`ai4i2020.csv`, 10,000 records)
- **Evidence-first agent:** `backend/app/plantpilot/agent.py` — bounded loop with evidence-only persistence; no diagnosis without verified observations
- **Anomaly detector:** `backend/app/plantpilot/detector.py` — train-fit IQR + multi-signal + documented experimental heuristic (`wear > 200`); never uses ground-truth labels
- **Evaluation:** `backend/app/plantpilot/eval/` — reproducible split (`seed=42`, stratified, no overlap); baseline `F1=0.177` reported as deterministic-only
- **Labels isolated:** Agent interface (`get_sensor_readings`) excludes `machine_failure`/`TWF`/`HDF`/`PWF`/`OSF`/`RNF`; evaluation-only via `get_evaluation_only`
- **Live NVIDIA:** Direct call `finish=tool_calls` with valid `udi=1`; full loop reaches stage 3 (evidence persisted) then stops at provider `500` — reported honestly, not manufactured
- **Tests:** 94 passed (includes 10 integration + eval + existing)

## Start Backend

```bash
cd backend
uv run uvicorn app.plantpilot.main:app --reload --port 8000
```

## Run Tests

```bash
cd backend
uv run pytest tests/ -v
```

## Key Design Rules (Enforced)

- Evidence (`EvidenceType.OBSERVED`) persisted only on successful tool execution; failed executions become `ToolExecution` with `status=FAILED`
- Tool arguments validated before execution (`ToolRegistry.validate_arguments`); invalid `udi` (text/description) rejected; empty `values` rejected
- Investigation `COMPLETED` only with evidence + reasoning; `INCOMPLETE` when model finishes without tools or provider fails
- Retry bounded to 1 attempt for qualifying transient `InternalServerError` only; no retry on invalid arguments
- `.env` loaded from environment; never printed, logged, or committed; `.env`/`.venv`/data files ignored

## Dataset

`backend/ai4i2020.csv` — AI4I 2020 Predictive Maintenance (10,000 records). Measurement columns (`Air temperature [K]`, `Process temperature [K]`, `Rotational speed [rpm]`, `Torque [Nm]`, `Tool wear [min]`) are agent-facing. Ground-truth failure labels (`Machine failure`, `TWF`, `HDF`, `PWF`, `OSF`, `RNF`) are evaluation-only.

## Documentation

Full project notes, architecture diagrams, development history, and experiment logs are available in the [`docs/`](file:///c:/Warning/hackathon/PLANTpilot/docs) directory (compatible with Obsidian).

## Limitations (Documented)

- Full live `COMPLETED` requires stable NVIDIA provider; intermittent `500` produces honest `INCOMPLETE`
- Detector thresholds are dataset-derived (IQR) + documented heuristic; not validated industrial limits
- No simulator or real-time factory integration (out of scope)
