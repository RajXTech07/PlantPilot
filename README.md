# PlantPilot — Evidence-First Industrial Operations AI Copilot

A human-supervised backend framework for investigating operational incidents using historical dataset evidence, structured tool calling, and isolated evaluation. No simulator, dashboard, or real-time factory integration.

## Verified State

- **Framework:** FastAPI + Pydantic + SQLite (`sqlite3`) + NVIDIA-compatible OpenAI SDK (`https://integrate.api.nvidia.com/v1`)
- **Dataset adapter:** `app/plantpilot/data_adapter.py` — AI4I 2020 predictive maintenance CSV (`ai4i2020.csv`, 10,000 records)
- **Evidence-first agent:** `app/plantpilot/agent.py` — bounded loop with evidence-only persistence; no diagnosis without verified observations
- **Anomaly detector:** `app/plantpilot/detector.py` — train-fit IQR + multi-signal + documented experimental heuristic (`wear > 200`); never uses ground-truth labels
- **Evaluation:** `app/plantpilot/eval/` — reproducible split (`seed=42`, stratified, no overlap); baseline `F1=0.177` reported as deterministic-only
- **Labels isolated:** Agent interface (`get_sensor_readings`) excludes `machine_failure`/`TWF`/`HDF`/`PWF`/`OSF`/`RNF`; evaluation-only via `get_evaluation_only`
- **Live NVIDIA:** Direct call `finish=tool_calls` with valid `udi=1`; full loop reaches stage 3 (evidence persisted) then stops at provider `500` — reported honestly, not manufactured
- **Tests:** 94 passed (includes 10 integration + eval + existing)

## Start

```bash
uv run uvicorn app.plantpilot.main:app --reload --port 8000
```

## Tests

```bash
uv run pytest tests/ -v
```

## Key Design Rules (Enforced)

- Evidence (`EvidenceType.OBSERVED`) persisted only on successful tool execution; failed executions become `ToolExecution` with `status=FAILED`
- Tool arguments validated before execution (`ToolRegistry.validate_arguments`); invalid `udi` (text/description) rejected; empty `values` rejected
- Investigation `COMPLETED` only with evidence + reasoning; `INCOMPLETE` when model finishes without tools or provider fails
- Retry bounded to 1 attempt for qualifying transient `InternalServerError` only; no retry on invalid arguments
- `.env` loaded from environment; never printed, logged, or committed; `.env`/`.venv`/data files ignored

## Dataset

`ai4i2020.csv` — AI4I 2020 Predictive Maintenance (10,000 records). Measurement columns (`Air temperature [K]`, `Process temperature [K]`, `Rotational speed [rpm]`, `Torque [Nm]`, `Tool wear [min]`) are agent-facing. Ground-truth failure labels (`Machine failure`, `TWF`, `HDF`, `PWF`, `OSF`, `RNF`) are evaluation-only.

## Files Created/Modified (Summary)

- `app/plantpilot/` (`__init__.py`, `main.py`, `config.py`, `schemas.py`, `agent.py`, `tools.py`, `storage.py`, `safety.py`, `detector.py`, `data_adapter.py`, `eval/`)
- `tests/` (10 files including `test_evidence_first.py`, `test_integration_e2e.py`, `test_eval_engine.py`)
- `pyproject.toml` (updated `pythonpath`)
- `.env.example`, `.gitignore`

## Limitations (Documented)

- Full live `COMPLETED` requires stable NVIDIA provider; intermittent `500` produces honest `INCOMPLETE`
- Detector thresholds are dataset-derived (IQR) + documented heuristic; not validated industrial limits
- No simulator, dashboard, or real-time factory integration (out of scope)

Co-Authored-By: Claude Code <noreply@anthropic.com>
