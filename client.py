"""
Client for CarbonSchedulerEnv.
Can connect to a remote HuggingFace Space or a local uvicorn server.
Never imports server internals — clean client/server separation.
"""

import requests
import json
from typing import Any, Dict, Optional, Tuple


class CarbonSchedulerClient:
    """
    Thin HTTP client wrapping the FastAPI server.

    Usage:
        client = CarbonSchedulerClient("http://localhost:7860")
        obs    = client.reset(seed=42)
        obs, reward, done, info = client.step(action_json)
    """

    def __init__(self, base_url: str = "http://localhost:7860", session_id: str = "default"):
        self.base_url   = base_url.rstrip("/")
        self.session_id = session_id

    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        resp = requests.post(f"{self.base_url}/reset", json={
            "session_id": self.session_id,
            "seed":       seed,
        })
        resp.raise_for_status()
        return resp.json()["observation"]

    def step(self, action_text: str) -> Tuple[Dict, float, bool, Dict]:
        resp = requests.post(f"{self.base_url}/step", json={
            "session_id":  self.session_id,
            "action_text": action_text,
        })
        resp.raise_for_status()
        data = resp.json()
        return data["observation"], data["reward"], data["done"], data["info"]

    def state(self) -> Dict[str, Any]:
        resp = requests.get(f"{self.base_url}/state", params={"session_id": self.session_id})
        resp.raise_for_status()
        return resp.json()

    def close(self):
        requests.post(f"{self.base_url}/close", json={"session_id": self.session_id})

    def health(self) -> bool:
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=3)
            return resp.status_code == 200
        except Exception:
            return False


# ── Local convenience wrapper (no HTTP, for fast iteration) ──────────────────

class LocalCarbonSchedulerClient:
    """
    Direct Python wrapper — same interface as HTTP client but runs env in-process.
    Use this for local training loops to avoid HTTP overhead.
    """
    def __init__(self, seed: Optional[int] = None):
        from environment import CarbonSchedulerEnv
        self.env = CarbonSchedulerEnv(seed=seed)

    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        return self.env.reset(seed=seed)

    def step(self, action_text: str) -> Tuple[Dict, float, bool, Dict]:
        return self.env.step(action_text)

    def state(self) -> Dict[str, Any]:
        return self.env.state()

    def close(self):
        self.env.close()
