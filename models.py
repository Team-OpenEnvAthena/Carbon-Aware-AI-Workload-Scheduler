"""
CarbonSchedulerEnv — Action and Observation models.

Both inherit from OpenEnv base types so the framework can
auto-generate schemas, WebSocket handlers, and the web UI.

Action:
    Structured list of scheduling decisions — one per pending job.
    Each decision is either an assignment (region + start_hour) or a defer.

Observation:
    Full state of the scheduling problem — text prompt included so
    the LLM can consume it directly via `obs.prompt`.
"""

from typing import Dict, List, Optional
from pydantic import Field
from openenv.core.env_server.types import Action, Observation


class ScheduleDecision(Action):
    """One scheduling decision for a single job."""
    job_id: str = Field(..., description="ID of the job being scheduled")
    region: str = Field(
        default="",
        description="Target data centre region (e.g. 'us-west-2'). Empty if deferring.",
    )
    start_hour: int = Field(
        default=-1,
        description="UTC hour to start the job (0–23). -1 if deferring.",
    )
    defer: bool = Field(
        default=False,
        description="True to defer this job to the next scheduling step.",
    )


class CarbonSchedulerAction(Action):
    """
    Agent's full scheduling action for one step.

    The agent outputs a list of decisions — one per pending job.
    Each job must be either assigned (region + start_hour) or deferred.

    Example:
        {
          "assignments": [
            {"job_id": "job_01", "region": "us-west-2", "start_hour": 14},
            {"job_id": "job_02", "defer": true},
            {"job_id": "job_03", "region": "eu-west-1", "start_hour": 16}
          ],
          "reasoning": "Deferring job_02 to the solar window at 14:00 in Oregon"
        }
    """
    assignments: List[ScheduleDecision] = Field(
        default_factory=list,
        description="List of scheduling decisions — one per pending job.",
    )
    reasoning: str = Field(
        default="",
        description="Optional: agent's reasoning for this schedule (for interpretability).",
    )


class CarbonSchedulerObservation(Observation):
    """
    Full observation of the scheduling environment.

    Key fields for the LLM agent:
        prompt          — formatted text prompt ready for LLM consumption
        current_hour    — current UTC hour (0–23)
        jobs_pending    — list of jobs awaiting scheduling
        datacenters     — list of data centres with carbon forecasts
        carbon_saved_so_far — gCO2 saved vs naive baseline this episode

    Key fields for training:
        reward          — reward from the last step
        done            — True if episode is over
        reward_breakdown — per-component reward for interpretability
    """

    # ── Episode metadata ───────────────────────────────────────────────────
    current_hour: int = Field(default=0, description="Current UTC hour (0–23)")
    step_number: int = Field(default=0, description="Step index within episode")
    done: bool = Field(default=False)
    reward: float = Field(default=0.0)

    # ── Scheduling state ───────────────────────────────────────────────────
    jobs_pending: List[dict] = Field(
        default_factory=list,
        description="Jobs awaiting a scheduling decision this step.",
    )
    jobs_completed: int = Field(default=0)
    jobs_failed: int = Field(default=0)
    total_jobs: int = Field(default=0)

    # ── Data centres ───────────────────────────────────────────────────────
    datacenters: List[dict] = Field(
        default_factory=list,
        description="Data centre state including carbon forecast for next 12 hours.",
    )

    # ── Progress metrics ───────────────────────────────────────────────────
    carbon_saved_so_far: float = Field(
        default=0.0,
        description="gCO2 saved vs naive run-immediately baseline this episode.",
    )
    naive_carbon_so_far: float = Field(
        default=0.0,
        description="Carbon that naive scheduler would have emitted.",
    )
    actual_carbon_so_far: float = Field(
        default=0.0,
        description="Actual carbon emitted by agent's schedule so far.",
    )
    completion_rate: float = Field(
        default=0.0,
        description="Fraction of jobs completed (0.0–1.0).",
    )

    # ── Curriculum ─────────────────────────────────────────────────────────
    curriculum_stage: int = Field(
        default=1,
        description="Current curriculum level (1=easy, 2=medium, 3=hard).",
    )

    # ── Reward breakdown (for training interpretability) ───────────────────
    reward_breakdown: Dict[str, float] = Field(
        default_factory=dict,
        description="Per-component reward breakdown for this step.",
    )
    episode_summary: Dict[str, float] = Field(
        default_factory=dict,
        description="End-of-episode metrics (populated when done=True).",
    )

    # ── LLM prompt ─────────────────────────────────────────────────────────
    prompt: str = Field(
        default="",
        description="Formatted text prompt for LLM consumption — ready to use directly.",
    )
