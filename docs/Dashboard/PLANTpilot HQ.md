# PLANTPILOT / PROJECT HQ

> [!info] Project Mission  
> Build a human-supervised industrial operations AI copilot that investigates machine incidents using operational data, presents evidence-backed findings, and recommends actions for human review.

## CURRENT STATUS

**Phase:** Backend & Framework Stabilized; Ready for Dashboard UI

**Backend:** Starts successfully with decoupled `backend/` directory  
**API:** Incident creation, investigation, and audit retrieval endpoints operational  
**AI:** Evidence-first loop enforced; tool calling connected to historical data  
**Test Suite:** 94/94 automated tests passing  
**Simulator:** Isolated to dataset adapter (`ai4i2020.csv`)  
**Dashboard:** Next phase (`frontend/`)

## DEVELOPMENT ROADMAP

- [x] Start the FastAPI backend
- [x] Create an incident through the API (`POST /api/v1/incidents`)
- [x] Run an investigation through the API (`POST /api/v1/incidents/{id}/investigations`)
- [x] Add evidence-first investigation enforcement (`tool_choice="required"`, non-empty checks)
- [x] Prevent diagnostic tools from receiving invalid inputs (`ToolRegistry.validate_arguments`)
- [x] Connect a sensor-data retrieval interface (`data_adapter.py` reading `ai4i2020.csv`)
- [x] Verify a successful diagnostic execution with valid readings (`get_sensor_readings` + `diagnostic_math`)
- [x] Verify the complete live NVIDIA tool-calling workflow
- [x] Decouple project structure into clean `backend/` and `docs/`
- [ ] Build the operator dashboard (`frontend/`)
- [x] Evaluate the system using repeatable test scenarios (stratified train/dev/eval splits)
- [ ] Prepare the final hackathon presentation
    

## PROJECT NOTES

- [[ARCHITECTURE]]
    
- [[DEVELOPMENT]]
    
- [[EXPERIMENT & TEST RESULTS]]
    
- [[Hackathon Presentation]]
    

## CORE PRINCIPLES

1. Framework first, simulator second, dashboard third.
    
2. Never invent sensor readings or operational evidence.
    
3. Keep observations, hypotheses, and recommendations separate.
    
4. Validate tool arguments before execution.
    
5. Require human approval for risky actions.
    
6. Report measured results honestly.