"""
CarbonSchedulerEnv Client.

Clean client/server separation — never imports server internals.
Uses OpenEnv EnvClient base with WebSocket for low-latency multi-step episodes.

Example:
    >>> with CarbonSchedulerClient(base_url="http://localhost:8000") as client:
    ...     result = client.reset()
    ...     print(result.observation.prompt)   # feed to LLM
    ...     print(result.observation.current_hour)
    ...
    ...     action = CarbonSchedulerAction(assignments=[
    ...         ScheduleDecision(job_id="job_01", region="us-west-2", start_hour=14),
    ...         ScheduleDecision(job_id="job_02", defer=True),
    ...     ])
    ...     result = client.step(action)
    ...     print(result.reward)
    ...     print(result.observation.carbon_saved_so_far)
"""

from typing import Dict
from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import (
    CarbonSchedulerAction,
    CarbonSchedulerObservation,
    ScheduleDecision,
)


class CarbonSchedulerClient(
    EnvClient[CarbonSchedulerAction, CarbonSchedulerObservation, State]
):
    """Client for the Carbon-Aware AI Workload Scheduler environment."""

    def _step_payload(self, action: CarbonSchedulerAction) -> Dict:
        return action.model_dump()

    def _parse_result(self, payload: Dict) -> StepResult[CarbonSchedulerObservation]:
        obs_data = payload.get("observation", payload)

        # Parse ScheduleDecision list if present in observation
        observation = CarbonSchedulerObservation(
            current_hour         = obs_data.get("current_hour", 0),
            step_number          = obs_data.get("step_number", 0),
            done                 = payload.get("done", obs_data.get("done", False)),
            reward               = payload.get("reward", obs_data.get("reward", 0.0)),
            jobs_pending         = obs_data.get("jobs_pending", []),
            jobs_completed       = obs_data.get("jobs_completed", 0),
            jobs_failed          = obs_data.get("jobs_failed", 0),
            total_jobs           = obs_data.get("total_jobs", 0),
            datacenters          = obs_data.get("datacenters", []),
            carbon_saved_so_far  = obs_data.get("carbon_saved_so_far", 0.0),
            naive_carbon_so_far  = obs_data.get("naive_carbon_so_far", 0.0),
            actual_carbon_so_far = obs_data.get("actual_carbon_so_far", 0.0),
            completion_rate      = obs_data.get("completion_rate", 0.0),
            curriculum_stage     = obs_data.get("curriculum_stage", 1),
            reward_breakdown     = obs_data.get("reward_breakdown", {}),
            episode_summary      = obs_data.get("episode_summary", {}),
            prompt               = obs_data.get("prompt", ""),
        )
        return StepResult(
            observation = observation,
            reward      = payload.get("reward", obs_data.get("reward", 0.0)),
            done        = payload.get("done", obs_data.get("done", False)),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id = payload.get("episode_id"),
            step_count = payload.get("step_count", 0),
        )
