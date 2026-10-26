"""PlantPilot API Client for Streamlit Frontend.

Provides typed, error-handled HTTP requests to the FastAPI backend.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
import urllib.request
import urllib.error


class APIClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self.base_url = base_url.rstrip("/")

    def _request(
        self,
        method: str,
        path: str,
        payload: Optional[Dict[str, Any]] = None,
        timeout: float = 30.0,
    ) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        data = None
        headers = {
            "Accept": "application/json",
            "User-Agent": "PlantPilot-Streamlit-Client/1.0",
        }

        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

        req = urllib.request.Request(url, data=data, headers=headers, method=method)

        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                status_code = response.getcode()
                raw_body = response.read().decode("utf-8")
                parsed = json.loads(raw_body) if raw_body else {}
                return {"success": True, "status_code": status_code, "data": parsed}
        except urllib.error.HTTPError as exc:
            raw_err = exc.read().decode("utf-8")
            try:
                err_json = json.loads(raw_err)
            except Exception:
                err_json = {"detail": raw_err or str(exc)}
            return {
                "success": False,
                "status_code": exc.code,
                "error": err_json.get("detail", str(exc)),
                "raw": raw_err,
            }
        except urllib.error.URLError as exc:
            return {
                "success": False,
                "status_code": 0,
                "error": f"Cannot connect to PlantPilot backend at {self.base_url}. Reason: {exc.reason}",
            }
        except Exception as exc:
            return {
                "success": False,
                "status_code": 0,
                "error": f"Unexpected request failure: {type(exc).__name__}: {exc}",
            }

    # --- System / Health ---
    def get_root(self) -> Dict[str, Any]:
        return self._request("GET", "/")

    def get_health(self, check_provider: bool = False) -> Dict[str, Any]:
        path = f"/health?check_provider=true" if check_provider else "/health"
        return self._request("GET", path)

    # --- Machines / Assets ---
    def list_machines(self, limit: int = 50, offset: int = 0) -> Dict[str, Any]:
        return self._request("GET", f"/api/v1/machines?limit={limit}&offset={offset}")

    def get_machine(self, machine_id: str) -> Dict[str, Any]:
        return self._request("GET", f"/api/v1/machines/{machine_id}")

    def create_machine(
        self,
        product_id: str,
        asset_type: str,
        machine_id: Optional[str] = None,
        status: str = "active",
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "product_id": product_id,
            "type": asset_type,
            "status": status,
        }
        if machine_id:
            payload["id"] = machine_id
        return self._request("POST", "/api/v1/machines", payload=payload)

    # --- Incidents ---
    def list_incidents(self, limit: int = 50, offset: int = 0) -> Dict[str, Any]:
        return self._request("GET", f"/api/v1/incidents?limit={limit}&offset={offset}")

    def get_incident(self, incident_id: str) -> Dict[str, Any]:
        return self._request("GET", f"/api/v1/incidents/{incident_id}")

    def create_incident(
        self,
        description: str,
        severity: str = "medium",
        category: str = "operational",
        machine_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "description": description,
            "severity": severity,
            "category": category,
            "metadata": metadata or {},
        }
        if machine_id:
            payload["machine_id"] = machine_id
        return self._request("POST", "/api/v1/incidents", payload=payload)

    # --- Investigations ---
    def start_investigation(
        self,
        incident_id: str,
        notes: Optional[str] = None,
        timeout: float = 90.0,
    ) -> Dict[str, Any]:
        payload = {
            "incident_id": incident_id,
            "notes": notes or f"Streamlit operator triggered investigation for {incident_id}",
        }
        return self._request(
            "POST",
            f"/api/v1/incidents/{incident_id}/investigations",
            payload=payload,
            timeout=timeout,
        )

    def get_investigation(self, investigation_id: str) -> Dict[str, Any]:
        return self._request("GET", f"/api/v1/investigations/{investigation_id}")

    def list_investigations(self, limit: int = 50, offset: int = 0) -> Dict[str, Any]:
        return self._request("GET", f"/api/v1/investigations?limit={limit}&offset={offset}")

    # --- Human-in-the-Loop Reviews ---
    def list_recommendations(
        self,
        approval_status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Dict[str, Any]:
        path = f"/api/v1/recommendations?limit={limit}&offset={offset}"
        if approval_status:
            path += f"&approval_status={approval_status}"
        return self._request("GET", path)
    def review_recommendation(
        self,
        recommendation_id: str,
        status_choice: str,  # 'approved' or 'rejected'
        reviewed_by: str,
        notes: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload = {
            "status_choice": status_choice,
            "reviewed_by": reviewed_by,
            "notes": notes,
        }
        return self._request(
            "POST",
            f"/api/v1/recommendations/{recommendation_id}/review",
            payload=payload,
        )
