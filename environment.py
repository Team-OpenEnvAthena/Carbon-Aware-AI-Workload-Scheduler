"""
CarbonSchedulerEnv — main OpenEnv environment class.

Follows the Gym-style API: reset() → step() → ... → done
Compatible with OpenEnv's Environment base class pattern.
"""

import json
import random
import uuid
from typing import Any, Dict, List, Optional, Tuple

from .models import (
    Job, DataCenter, Assignment, Action,
    Observation, Priority,
)
from .carbon_data import (
    BASE_PROFILES, get_carbon_now,
    get_carbon_forecast, get_renewable_pct,
    naive_carbon_for_job,
)
from .rewards import compute_step_reward, compute_episode_reward, RewardBreakdown


# ── Job templates — realistic AI workload names ───────────────────────────────

JOB_TEMPLATES = [
    # (name, gpu_hours, energy_kwh, priority, deadline_offset, deferrable)
    ("LLM fine-tune large",      80,  240, Priority.HIGH,   10, True),
    ("LLM fine-tune small",      20,   60, Priority.NORMAL,  8, True),
    ("Image classifier train",   15,   45, Priority.NORMAL,  6, True),
    ("Embedding batch job",       8,   24, Priority.LOW,    12, True),
    ("Nightly eval suite",       12,   36, Priority.LOW,     8, True),
    ("Safety filter retrain",    25,   75, Priority.URGENT,  4, False),
    ("Inference cache warm",      5,   15, Priority.URGENT,  2, False),
    ("Diffusion model train",    60,  180, Priority.HIGH,   12, True),
    ("RL policy update",         30,   90, Priority.NORMAL,  8, True),
    ("Data preprocessing",        6,   18, Priority.LOW,    16, True),
    ("A/B model comparison",     10,   30, Priority.NORMAL,  6, True),
    ("Checkpoint conversion",     3,    9, Priority.URGENT,  3, False),
    ("Multilingual finetune",    50,  150, Priority.HIGH,   10, True),
    ("Vision-language align",    45,  135, Priority.HIGH,    8, True),
    ("Reward model train",       35,  105, Priority.NORMAL,  6, True),
]


class CarbonSchedulerEnv:
    """
    Carbon-Aware AI Workload Scheduler Environment.

    Episode:
      - 24-step horizon (one step = one hour of the day)
      - Agent schedules AI training jobs across 6 real data centre regions
      - Reward: carbon saved vs naive baseline, SLA compliance, grid stability
      - Done when: all 24 hours elapsed OR all jobs completed/failed

    Observation:
      - Current hour, pending jobs, data centre carbon forecasts
      - Text prompt ready for LLM consumption

    Action:
      - JSON: {"assignments": [{job_id, region, start_hour} | {job_id, defer}]}
    """

    REGIONS = list(BASE_PROFILES.keys())
    MAX_STEPS = 24   # one per hour
    JOBS_PER_EPISODE = (8, 14)   # random range

    def __init__(self, seed: Optional[int] = None):
        self.seed = seed or random.randint(0, 10_000)
        self.rng  = random.Random(self.seed)

        # episode state (populated by reset)
        self.current_hour:   int = 0
        self.step_count:     int = 0
        self.jobs:           Dict[str, Job] = {}
        self.datacenters:    Dict[str, DataCenter] = {}
        self.pending_ids:    List[str] = []
        self.total_carbon:   float = 0.0
        self.naive_carbon:   float = 0.0
        self.episode_done:   bool  = False
        self.reward_history: List[dict] = []

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        if seed is not None:
            self.seed = seed
            self.rng  = random.Random(seed)

        self.current_hour  = self.rng.randint(0, 6)   # start early morning
        self.step_count    = 0
        self.total_carbon  = 0.0
        self.naive_carbon  = 0.0
        self.episode_done  = False
        self.reward_history = []

        self._generate_jobs()
        self._init_datacenters()

        obs = self._make_observation()
        return obs.to_dict()

    def step(self, action_text: str) -> Tuple[Dict, float, bool, Dict]:
        """
        Execute one scheduling step.

        Args:
            action_text: raw LLM output (JSON string)

        Returns:
            observation, reward, done, info
        """
        if self.episode_done:
            raise RuntimeError("Episode is done. Call reset() first.")

        # ── Parse action ──────────────────────────────────────────────────
        action, parse_error = self._parse_action(action_text)

        # ── Apply assignments ─────────────────────────────────────────────
        applied_assignments = []
        if action:
            for asgn in action.assignments:
                applied = self._apply_assignment(asgn)
                if applied:
                    applied_assignments.append(asgn)

        # ── Fail jobs that have hit their deadline with no assignment ─────
        self._expire_overdue_jobs()

        # ── Compute step reward ───────────────────────────────────────────
        reward_info = compute_step_reward(
            assignments  = applied_assignments,
            jobs_map     = self.jobs,
            datacenters  = self.datacenters,
            current_hour = self.current_hour,
            noise_seed   = self.seed,
            valid_regions = self.REGIONS,
        )
        step_reward = reward_info.total
        if parse_error:
            step_reward = min(step_reward - 0.3, -0.1)   # malformed JSON always negative

        # Track carbon
        for asgn in applied_assignments:
            if not asgn.deferred and asgn.region in self.datacenters:
                job = self.jobs.get(asgn.job_id)
                if job:
                    from .carbon_data import get_carbon_forecast
                    forecast = get_carbon_forecast(asgn.region, self.current_hour, self.seed)
                    offset   = max(0, asgn.start_hour - self.current_hour)
                    ci       = forecast[min(offset, len(forecast) - 1)]
                    self.total_carbon += job.energy_kwh * ci
                    self.naive_carbon += naive_carbon_for_job(
                        job.energy_kwh, self.current_hour, self.seed
                    )

        # ── Advance time ──────────────────────────────────────────────────
        self.current_hour = (self.current_hour + 1) % 24
        self.step_count  += 1
        self._refresh_datacenters()

        # ── Check termination ─────────────────────────────────────────────
        all_done = all(j.completed or j.failed for j in self.jobs.values())
        time_up  = self.step_count >= self.MAX_STEPS
        self.episode_done = all_done or time_up

        obs = self._make_observation()
        info = {
            "step_reward":   reward_info.to_dict(),
            "parse_error":   parse_error,
            "step_count":    self.step_count,
            "current_hour":  self.current_hour,
            "pending_count": len(self.pending_ids),
        }

        if self.episode_done:
            ep_reward, ep_info = compute_episode_reward(
                all_jobs      = list(self.jobs.values()),
                total_carbon  = self.total_carbon,
                naive_carbon  = self.naive_carbon,
            )
            info["episode_summary"] = ep_info
            step_reward = 0.5 * step_reward + 0.5 * ep_reward   # blend

        self.reward_history.append(info)
        return obs.to_dict(), float(step_reward), self.episode_done, info

    def state(self) -> Dict[str, Any]:
        """Full internal state — for logging and debugging."""
        return {
            "current_hour":  self.current_hour,
            "step_count":    self.step_count,
            "jobs":          {k: v.to_dict() for k, v in self.jobs.items()},
            "datacenters":   {k: v.to_dict() for k, v in self.datacenters.items()},
            "pending_ids":   self.pending_ids,
            "total_carbon":  round(self.total_carbon, 2),
            "naive_carbon":  round(self.naive_carbon, 2),
            "carbon_saved":  round(self.naive_carbon - self.total_carbon, 2),
            "episode_done":  self.episode_done,
            "seed":          self.seed,
        }

    def close(self):
        pass

    # ── Private helpers ───────────────────────────────────────────────────────

    def _generate_jobs(self):
        """Randomly generate a batch of AI workload jobs for this episode."""
        self.jobs = {}
        self.pending_ids = []
        n_jobs = self.rng.randint(*self.JOBS_PER_EPISODE)

        for i in range(n_jobs):
            tmpl = self.rng.choice(JOB_TEMPLATES)
            name, gpu_h, energy, priority, dl_offset, deferrable = tmpl

            job_id   = f"job_{i+1:02d}"
            deadline = min(23, self.current_hour + dl_offset + self.rng.randint(-1, 2))

            job = Job(
                id           = job_id,
                name         = name,
                gpu_hours    = gpu_h * self.rng.uniform(0.8, 1.2),
                energy_kwh   = energy * self.rng.uniform(0.8, 1.2),
                priority     = priority,
                deadline_hour = max(self.current_hour + 1, deadline),
                created_hour = self.current_hour,
                deferrable   = deferrable,
            )
            self.jobs[job_id]  = job
            self.pending_ids.append(job_id)

    def _init_datacenters(self):
        """Initialise all 6 data centres with current carbon data."""
        self.datacenters = {}
        for region, meta in BASE_PROFILES.items():
            self.datacenters[region] = DataCenter(
                region        = region,
                name          = meta["name"],
                capacity_total = meta["capacity"],
                capacity_used  = self.rng.uniform(0, meta["capacity"] * 0.4),
                carbon_now     = get_carbon_now(region, self.current_hour, self.seed),
                carbon_forecast = get_carbon_forecast(region, self.current_hour, self.seed),
                renewable_pct  = get_renewable_pct(region, self.current_hour),
            )

    def _refresh_datacenters(self):
        """Update carbon data after time advances one hour."""
        for region, dc in self.datacenters.items():
            dc.carbon_now     = get_carbon_now(region, self.current_hour, self.seed)
            dc.carbon_forecast = get_carbon_forecast(region, self.current_hour, self.seed)
            dc.renewable_pct  = get_renewable_pct(region, self.current_hour)
            # capacity partially recovers each hour
            dc.capacity_used  = max(0.0, dc.capacity_used * 0.7)

    def _apply_assignment(self, asgn: Assignment) -> bool:
        """Apply a single job assignment. Returns True if valid."""
        job = self.jobs.get(asgn.job_id)
        if job is None or job.completed or job.failed:
            return False

        if asgn.deferred:
            return True   # deferred — stays pending

        if asgn.region not in self.datacenters:
            return False

        dc = self.datacenters[asgn.region]
        if job.gpu_hours > dc.capacity_available:
            return False   # not enough capacity

        # Commit assignment
        job.assigned_region = asgn.region
        job.assigned_hour   = asgn.start_hour
        job.completed       = True
        dc.capacity_used   += job.gpu_hours

        if asgn.job_id in self.pending_ids:
            self.pending_ids.remove(asgn.job_id)

        return True

    def _expire_overdue_jobs(self):
        """Fail any jobs whose deadline has passed without assignment."""
        for job_id in list(self.pending_ids):
            job = self.jobs[job_id]
            if self.current_hour > job.deadline_hour:
                job.failed = True
                self.pending_ids.remove(job_id)

    def _parse_action(self, action_text: str) -> Tuple[Optional[Action], Optional[str]]:
        """Parse LLM JSON output into an Action. Returns (action, error_msg)."""
        try:
            # Extract JSON from possible surrounding text
            start = action_text.find("{")
            end   = action_text.rfind("}") + 1
            if start == -1 or end == 0:
                return None, "No JSON object found in response"

            raw  = action_text[start:end]
            data = json.loads(raw)
            action = Action.from_dict(data)
            return action, None

        except json.JSONDecodeError as e:
            return None, f"JSON parse error: {e}"
        except Exception as e:
            return None, f"Action parse error: {e}"

    def _make_observation(self) -> Observation:
        pending_jobs = [
            self.jobs[jid].to_dict()
            for jid in self.pending_ids
            if not self.jobs[jid].completed and not self.jobs[jid].failed
        ]
        completed = sum(1 for j in self.jobs.values() if j.completed)
        failed    = sum(1 for j in self.jobs.values() if j.failed)
        remaining = max(1, len(self.jobs))

        return Observation(
            current_hour        = self.current_hour,
            step_number         = self.step_count,
            jobs_pending        = pending_jobs,
            jobs_completed      = completed,
            jobs_failed         = failed,
            datacenters         = [dc.to_dict() for dc in self.datacenters.values()],
            carbon_saved_so_far = max(0.0, self.naive_carbon - self.total_carbon),
            budget_remaining    = len(self.pending_ids) / remaining,
        )
