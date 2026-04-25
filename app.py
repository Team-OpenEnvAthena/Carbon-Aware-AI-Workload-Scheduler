"""
FastAPI server for CarbonSchedulerEnv.
Follows OpenEnv server conventions:
  POST /reset
  POST /step
  GET  /state
  POST /close
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, Optional
import uvicorn

from environment import CarbonSchedulerEnv

app  = FastAPI(title="Carbon-Aware Workload Scheduler", version="1.0.0")
envs: Dict[str, CarbonSchedulerEnv] = {}   # session_id → env instance


# ── Request / response models ─────────────────────────────────────────────────

class ResetRequest(BaseModel):
    session_id: Optional[str] = "default"
    seed:       Optional[int] = None


class StepRequest(BaseModel):
    session_id:  Optional[str] = "default"
    action_text: str             # raw LLM output (JSON string)


class StateRequest(BaseModel):
    session_id: Optional[str] = "default"


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "env": "CarbonSchedulerEnv"}


@app.post("/reset")
def reset(req: ResetRequest) -> Dict[str, Any]:
    """Start a new episode. Returns initial observation."""
    env = CarbonSchedulerEnv(seed=req.seed)
    envs[req.session_id] = env
    obs = env.reset(seed=req.seed)
    return {"observation": obs, "session_id": req.session_id}


@app.post("/step")
def step(req: StepRequest) -> Dict[str, Any]:
    """Execute one scheduling step with the agent's action text."""
    env = envs.get(req.session_id)
    if env is None:
        raise HTTPException(status_code=404, detail=f"Session {req.session_id!r} not found. Call /reset first.")

    obs, reward, done, info = env.step(req.action_text)
    return {
        "observation": obs,
        "reward":      reward,
        "done":        done,
        "info":        info,
        "session_id":  req.session_id,
    }


@app.get("/state")
def state(session_id: str = "default") -> Dict[str, Any]:
    """Return full internal state — for debugging and logging."""
    env = envs.get(session_id)
    if env is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id!r} not found.")
    return env.state()


@app.post("/close")
def close(req: StateRequest) -> Dict[str, Any]:
    """Clean up session."""
    env = envs.pop(req.session_id, None)
    if env:
        env.close()
    return {"closed": req.session_id}


@app.get("/info")
def info() -> Dict[str, Any]:
    """Environment metadata — used by OpenEnv registry."""
    return {
        "name":        "carbon-scheduler",
        "version":     "1.0.0",
        "description": "Carbon-aware AI workload scheduler across 6 global data centres",
        "action_type": "json_text",
        "obs_type":    "text_prompt",
        "max_steps":   24,
        "regions":     list(CarbonSchedulerEnv.REGIONS),
        "themes":      ["sustainability", "world-modeling", "long-horizon-planning"],
    }


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)
