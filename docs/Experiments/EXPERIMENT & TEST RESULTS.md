# PLANTPILOT / EXPERIMENTS

## Experiment 001 — Placeholder Incident

**Input:** Incident description was `string`.

**Observed:**

- The investigation produced a textual finding.
    
- No evidence, hypotheses, or recommendations were stored.
    
- The earlier implementation marked the investigation completed.
    

**Conclusion:** The earlier workflow could complete without diagnostic evidence.

---

## Experiment 002 — Evidence-First Enforcement

**Input:** An incident with no numerical sensor readings.

**Observed:**

- `diagnostic_math` received `values=[]`.
    
- The tool rejected the arguments.
    
- A failed tool execution was persisted.
    
- The investigation status was `incomplete`.
    

**Conclusion:** Failure handling improved, but a successful diagnostic has not yet been demonstrated.

---

## Experiment 003 — Valid Sensor Readings & Evidence-First Workflow
 
**Status:** Completed & Verified (100% Passing).

**Objective:** Provide valid numerical readings through an approved data source (`data_adapter.py`), execute diagnostic tools, persist verified evidence, and generate a grounded investigation result.

### Results

- The agent was supplied with incident context referencing dataset UDI `1`.
- Step 0 forced tool call `get_sensor_readings(udi=1)`.
- Physical telemetry was retrieved from `ai4i2020.csv` (excluding ground-truth failure labels).
- Observations were validated and persisted in SQLite as `EvidenceType.OBSERVED` with `verified=True`.
- `diagnostic_math` was invoked with valid non-empty measurement arrays, successfully calculating `count`, `mean`, and `std_dev`.
- The model generated hypotheses grounded in the evidence and queued a high-priority recommendation (`requires_approval=True`, `status=pending_approval`).
- Investigation concluded with status `COMPLETED`.
- All 94 automated test cases passed across unit, schema, safety, storage, and E2E integration suites.

### Verified Audit Metrics

- **Unit & Integration Tests:** 94 passed in 5.8s
- **Ground-Truth Isolation:** Zero leakage of labels (`Machine failure`, `TWF`, `HDF`, `PWF`, `OSF`, `RNF`) to the agent
- **Evaluation Baseline:** Stratified split (`seed=42`, train/dev/eval) with deterministic baseline reporting `F1 = 0.177`