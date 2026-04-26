"""
CarbonSchedulerEnv — OpenEnv-compliant RL environment.

Carbon-Aware AI Workload Scheduler:
  An LLM agent schedules AI training jobs across 6 global data centres
  to minimise carbon emissions while meeting SLA deadlines.

Episode:
  - 24-step horizon (one step = one hour)
  - 8–14 jobs per episode, mixed priorities
  - Agent sees carbon forecasts and must plan temporally

Curriculum:
  Stage 1 — 3 jobs, 2 cleanest regions, no urgent (learn basic carbon routing)
  Stage 2 — 6 jobs, 4 regions, some urgent (learn priority handling)
  Stage 3 — 14 jobs, all 6 regions (learn temporal optimisation)

Reward (non-overlapping components):
  carbon_score    0.40 — gCO2 saved vs naive baseline
  sla_score       0.30 — shaped deadline compliance (not binary)
  deferral        0.15 — did agent defer the RIGHT jobs?
  grid_stability  0.15 — no region overloaded

FIX vs original:
  - Inherits from openenv.core.env_server.interfaces.Environment
  - step() takes CarbonSchedulerAction (Pydantic), not raw string
  - state property returns openenv.core.env_server.types.State
  - All internal imports use server-relative paths
  - SLA reward is shaped (continuous), not binary cutoff
"""

import random
from typing import Dict, List, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

# Absolute imports only — server is the top-level package when uvicorn
# runs `server.app:app`, so relative imports like `from ..models` raise:
#   ImportError: attempted relative import beyond top-level package
# PYTHONPATH=/app/env (set in Dockerfile) makes all of these resolvable.
from models import CarbonSchedulerAction, CarbonSchedulerObservation, ScheduleDecision
from server.data_models import Job, DataCenter, InternalAssignment, Priority
from server.carbon_data import (
    BASE_PROFILES, get_carbon_now, get_carbon_forecast,
    get_renewable_pct, naive_carbon_for_job,
)
from server.rewards import compute_step_reward, compute_episode_reward


# ── Curriculum config ──────────────────────────────────────────────────────────

CURRICULUM = {
    1: {
        "description":  "3 jobs, 2 cleanest regions, no urgent — learn basic carbon routing",
        "n_jobs":       (2, 3),
        "regions":      ["us-west-2", "eu-west-1"],
        "no_urgent":    True,
        "max_steps":    8,
    },
    2: {
        "description":  "6 jobs, 4 regions, some urgent — learn priority handling",
        "n_jobs":       (4, 6),
        "regions":      ["us-west-2", "us-west-1", "eu-west-1", "us-east-1"],
        "no_urgent":    False,
        "max_steps":    16,
    },
    3: {
        "description":  "Full episode — 14 jobs, all 6 regions, temporal optimisation",
        "n_jobs":       (8, 14),
        "regions":      list(BASE_PROFILES.keys()),
        "no_urgent":    False,
        "max_steps":    24,
    },
}

STAGE_REWARD_THRESHOLD = 0.70   # FIX [5]: was 0.55 — barely above naive baseline   # avg reward to advance stage
STAGE_EPISODES_NEEDED  = 5      # consecutive episodes above threshold

# ── Job templates (realistic AI workloads) ─────────────────────────────────────

JOB_TEMPLATES = [
    # (name, gpu_hours, energy_kwh, priority, deadline_offset, deferrable)
    ("LLM fine-tune large",     80,  240, Priority.HIGH,   10, True),
    ("LLM fine-tune small",     20,   60, Priority.NORMAL,  8, True),
    ("Image classifier train",  15,   45, Priority.NORMAL,  6, True),
    ("Embedding batch job",      8,   24, Priority.LOW,    12, True),
    ("Nightly eval suite",      12,   36, Priority.LOW,     8, True),
    ("Safety filter retrain",   25,   75, Priority.URGENT,  4, False),
    ("Inference cache warm",     5,   15, Priority.URGENT,  2, False),
    ("Diffusion model train",   60,  180, Priority.HIGH,   12, True),
    ("RL policy update",        30,   90, Priority.NORMAL,  8, True),
    ("Data preprocessing",       6,   18, Priority.LOW,    16, True),
    ("A/B model comparison",    10,   30, Priority.NORMAL,  6, True),
    ("Checkpoint conversion",    3,    9, Priority.URGENT,  3, False),
    ("Multilingual finetune",   50,  150, Priority.HIGH,   10, True),
    ("Vision-language align",   45,  135, Priority.HIGH,    8, True),
    ("Reward model train",      35,  105, Priority.NORMAL,  6, True),
]


class CarbonSchedulerEnvEnvironment(Environment):
    """
    Carbon-Aware AI Workload Scheduler RL Environment.

    The agent is an LLM that reads carbon intensity forecasts and
    schedules AI training jobs to minimise emissions while meeting deadlines.

    This is self-referential: we train AI to make AI training greener.

    Example:
        >>> env = CarbonSchedulerEnvEnvironment()
        >>> obs = env.reset()
        >>> print(obs.prompt)           # formatted text for LLM
        >>> print(obs.current_hour)     # e.g. 3 (3am UTC)
        >>>
        >>> action = CarbonSchedulerAction(assignments=[
        ...     ScheduleDecision(job_id="job_01", region="us-west-2", start_hour=14),
        ...     ScheduleDecision(job_id="job_02", defer=True),
        ... ], reasoning="Deferring job_02 to the solar window")
        >>>
        >>> obs = env.step(action)
        >>> print(obs.reward)
        >>> print(obs.reward_breakdown)
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, curriculum_stage: int = 1, **kwargs):
        self._curriculum_stage = max(1, min(3, curriculum_stage))
        self._stage_successes  = 0
        self._stage_episodes   = 0
        self._recent_rewards: List[float] = []
        self._reset_episode_state()

    def _reset_episode_state(self):
        self._episode_id    = str(uuid4())
        self._step_count    = 0
        self._current_hour  = 0
        self._jobs: Dict[str, Job] = {}
        self._datacenters: Dict[str, DataCenter] = {}
        self._pending_ids: List[str] = []
        self._total_carbon  = 0.0
        self._naive_carbon  = 0.0
        self._episode_done  = False
        self._last_reward_breakdown: dict = {}
        self._last_episode_summary: dict = {}
        self._rng = random.Random()

    # ── OpenEnv interface ──────────────────────────────────────────────────────

    def reset(self, seed: int = None, **kwargs) -> CarbonSchedulerObservation:
        """
        Start a new episode.
        Returns the initial observation — the agent's first view of the problem.
        """
        if seed is not None:
            self._rng = random.Random(seed)
        else:
            self._rng = random.Random()

        self._reset_episode_state()
        self._current_hour = self._rng.randint(0, 6)  # start early morning

        cfg = CURRICULUM[self._curriculum_stage]
        self._active_regions = cfg["regions"]
        self._max_steps = cfg["max_steps"]

        self._generate_jobs(cfg)
        self._init_datacenters()

        return self._make_observation(reward=0.0, done=False)

    def step(self, action: CarbonSchedulerAction, **kwargs) -> CarbonSchedulerObservation:
        """
        Execute one scheduling step.

        Args:
            action: CarbonSchedulerAction with list of ScheduleDecision objects.

        Returns:
            Updated CarbonSchedulerObservation with reward and metrics.
        """
        if self._episode_done:
            return self._make_observation(reward=0.0, done=True)

        # Convert Pydantic action to internal assignments
        internal_assignments = []
        parse_error = False

        for decision in action.assignments:
            if not decision.job_id:
                parse_error = True
                continue
            internal_assignments.append(InternalAssignment(
                job_id     = decision.job_id,
                region     = decision.region or "",
                start_hour = decision.start_hour if decision.start_hour >= 0 else self._current_hour,
                deferred   = decision.defer,
            ))

        # Apply assignments
        applied = []
        for asgn in internal_assignments:
            if self._apply_assignment(asgn):
                applied.append(asgn)

        # Expire overdue jobs
        self._expire_overdue_jobs()

        # Compute reward
        reward_result = compute_step_reward(
            assignments   = applied,
            jobs_map      = self._jobs,
            datacenters   = self._datacenters,
            current_hour  = self._current_hour,
            noise_seed    = 0,
            valid_regions = self._active_regions,
            parse_error   = parse_error,
        )

        # Track carbon
        for asgn in applied:
            if not asgn.deferred and asgn.region in self._datacenters:
                job = self._jobs.get(asgn.job_id)
                if job:
                    forecast = get_carbon_forecast(asgn.region, self._current_hour, 0)
                    offset   = max(0, asgn.start_hour - self._current_hour)
                    ci       = forecast[min(offset, len(forecast) - 1)]
                    self._total_carbon += job.energy_kwh * ci
                    self._naive_carbon += naive_carbon_for_job(
                        job.energy_kwh, self._current_hour, 0
                    )

        # Advance time
        self._current_hour = (self._current_hour + 1) % 24
        self._step_count  += 1
        self._refresh_datacenters()

        # Check termination
        all_done = all(j.completed or j.failed for j in self._jobs.values())
        time_up  = self._step_count >= self._max_steps
        self._episode_done = all_done or time_up

        step_reward = reward_result.total
        self._last_reward_breakdown = reward_result.to_dict()
        self._last_episode_summary  = {}

        if self._episode_done:
            ep_reward, ep_info = compute_episode_reward(
                all_jobs     = list(self._jobs.values()),
                total_carbon = self._total_carbon,
                naive_carbon = self._naive_carbon,
            )
            # Blend step and episode reward
            step_reward = 0.5 * step_reward + 0.5 * ep_reward
            self._last_episode_summary = ep_info
            self._check_stage_advancement(step_reward)

        return self._make_observation(
            reward = step_reward,
            done   = self._episode_done,
        )

    @property
    def state(self) -> State:
        return State(
            episode_id = self._episode_id,
            step_count = self._step_count,
        )

    # ── Private helpers ────────────────────────────────────────────────────────

    def _generate_jobs(self, cfg: dict):
        self._jobs = {}
        self._pending_ids = []
        n_jobs = self._rng.randint(*cfg["n_jobs"])

        templates = [t for t in JOB_TEMPLATES
                     if not (cfg["no_urgent"] and t[3] == Priority.URGENT)]

        for i in range(n_jobs):
            tmpl = self._rng.choice(templates)
            name, gpu_h, energy, priority, dl_offset, deferrable = tmpl
            job_id   = f"job_{i+1:02d}"
            deadline = min(23, self._current_hour + dl_offset + self._rng.randint(-1, 2))
            job = Job(
                id            = job_id,
                name          = name,
                gpu_hours     = gpu_h * self._rng.uniform(0.8, 1.2),
                energy_kwh    = energy * self._rng.uniform(0.8, 1.2),
                priority      = priority,
                deadline_hour = max(self._current_hour + 1, deadline),
                created_hour  = self._current_hour,
                deferrable    = deferrable,
            )
            self._jobs[job_id] = job
            self._pending_ids.append(job_id)

    def _init_datacenters(self):
        self._datacenters = {}
        for region in self._active_regions:
            meta = BASE_PROFILES[region]
            self._datacenters[region] = DataCenter(
                region          = region,
                name            = meta["name"],
                capacity_total  = meta["capacity"],
                capacity_used   = self._rng.uniform(0, meta["capacity"] * 0.3),
                carbon_now      = get_carbon_now(region, self._current_hour, 0),
                carbon_forecast = get_carbon_forecast(region, self._current_hour, 0),
                renewable_pct   = get_renewable_pct(region, self._current_hour),
            )

    def _refresh_datacenters(self):
        for region, dc in self._datacenters.items():
            dc.carbon_now      = get_carbon_now(region, self._current_hour, 0)
            dc.carbon_forecast = get_carbon_forecast(region, self._current_hour, 0)
            dc.renewable_pct   = get_renewable_pct(region, self._current_hour)
            dc.capacity_used   = max(0.0, dc.capacity_used * 0.7)

    def _apply_assignment(self, asgn: InternalAssignment) -> bool:
        job = self._jobs.get(asgn.job_id)
        if job is None or job.completed or job.failed:
            return False
        if asgn.deferred:
            return True
        if asgn.region not in self._datacenters:
            return False
        dc = self._datacenters[asgn.region]
        if job.gpu_hours > dc.capacity_available:
            return False
        job.assigned_region = asgn.region
        job.assigned_hour   = asgn.start_hour
        job.completed       = True
        dc.capacity_used   += job.gpu_hours
        if asgn.job_id in self._pending_ids:
            self._pending_ids.remove(asgn.job_id)
        return True

    def _expire_overdue_jobs(self):
        for job_id in list(self._pending_ids):
            job = self._jobs[job_id]
            if self._current_hour > job.deadline_hour:
                job.failed = True
                self._pending_ids.remove(job_id)

    def _make_observation(self, reward: float, done: bool) -> CarbonSchedulerObservation:
        pending = [
            self._jobs[jid].to_dict() for jid in self._pending_ids
            if not self._jobs[jid].completed and not self._jobs[jid].failed
        ]
        completed = sum(1 for j in self._jobs.values() if j.completed)
        failed    = sum(1 for j in self._jobs.values() if j.failed)
        total     = len(self._jobs)
        carbon_saved = max(0.0, self._naive_carbon - self._total_carbon)

        return CarbonSchedulerObservation(
            current_hour         = self._current_hour,
            step_number          = self._step_count,
            done                 = done,
            reward               = round(reward, 4),
            jobs_pending         = pending,
            jobs_completed       = completed,
            jobs_failed          = failed,
            total_jobs           = total,
            datacenters          = [dc.to_dict() for dc in self._datacenters.values()],
            carbon_saved_so_far  = round(carbon_saved, 1),
            naive_carbon_so_far  = round(self._naive_carbon, 1),
            actual_carbon_so_far = round(self._total_carbon, 1),
            completion_rate      = round(completed / max(1, total), 3),
            curriculum_stage     = self._curriculum_stage,
            reward_breakdown     = self._last_reward_breakdown,
            episode_summary      = self._last_episode_summary,
            prompt               = self._build_prompt(pending),
        )

    def _build_prompt(self, pending: List[dict]) -> str:
        """Format observation as LLM-ready text prompt."""
        lines = [
            f"=== Carbon-Aware Scheduler | Hour {self._current_hour:02d}:00 UTC "
            f"| Step {self._step_count} | Stage {self._curriculum_stage}/3 ===",
            "",
            "PENDING JOBS (you must schedule or defer each one):",
        ]
        for j in pending:
            tag = "(deferrable)" if j["deferrable"] else "(NOT deferrable — must assign now)"
            lines.append(
                f"  {j['id']}: {j['name']} | {j['gpu_hours']:.0f} GPU-hrs | "
                f"{j['energy_kwh']:.0f} kWh | priority={j['priority']} | "
                f"deadline=hour {j['deadline_hour']} {tag}"
            )

        lines += ["", "DATA CENTRES (carbon gCO2/kWh — lower = cleaner):"]
        for dc_obj in self._datacenters.values():
            dc = dc_obj.to_dict()
            forecast = " → ".join(
                str(int(c)) for c in dc["carbon_forecast"][:6]
            )
            lines.append(
                f"  {dc['region']}: {dc['name']} | "
                f"now={dc['carbon_now']} | next 6h: {forecast} | "
                f"available={dc['capacity_available']:.0f} GPU-hrs | "
                f"renewables={dc['renewable_pct']}%"
            )

        lines += [
            "",
            f"Progress: {sum(1 for j in self._jobs.values() if j.completed)} completed, "
            f"{sum(1 for j in self._jobs.values() if j.failed)} failed | "
            f"Carbon saved so far: {max(0.0, self._naive_carbon - self._total_carbon):.0f} gCO2",
            "",
            "RULES: 1) URGENT jobs must be assigned NOW (never defer)  "
            "2) LOW priority can defer to cleaner windows  "
            "3) Oregon (us-west-2) and Ireland (eu-west-1) are typically cleanest",
            "",
            "Respond with JSON only:",
            '{"assignments": [',
            '  {"job_id": "job_01", "region": "us-west-2", "start_hour": 14},',
            '  {"job_id": "job_02", "defer": true}',
            "]}",
        ]
        return "\n".join(lines)

    def _check_stage_advancement(self, episode_reward: float):
        """Advance curriculum stage when agent sustains good performance."""
        self._recent_rewards.append(episode_reward)
        if len(self._recent_rewards) > STAGE_EPISODES_NEEDED:
            self._recent_rewards.pop(0)

        if (len(self._recent_rewards) == STAGE_EPISODES_NEEDED
                and sum(self._recent_rewards) / STAGE_EPISODES_NEEDED >= STAGE_REWARD_THRESHOLD
                and self._curriculum_stage < 3):
            self._curriculum_stage += 1
            self._recent_rewards = []