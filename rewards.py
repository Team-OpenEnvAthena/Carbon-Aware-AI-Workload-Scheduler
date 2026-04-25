"""
Reward functions for the Carbon-Aware Workload Scheduler.

Design principles from the guide:
  - Multiple independent reward functions (harder to hack any single one)
  - Objective, deterministic — no LLM-as-judge
  - Each component is bounded and interpretable
  - Anti-gaming checks built in
"""

from typing import List, Dict, Tuple
from dataclasses import dataclass

from .models import Job, DataCenter, Assignment, Priority
from .carbon_data import naive_carbon_for_job, get_carbon_forecast


@dataclass
class RewardBreakdown:
    """Full reward breakdown — logged during training for inspection."""
    total:              float
    carbon_score:       float   # 0–1: carbon saved vs naive baseline
    sla_score:          float   # 0–1: fraction of jobs meeting deadlines
    deferral_quality:   float   # 0–1: did agent defer the right jobs?
    grid_stability:     float   # 0–1: no region overloaded
    urgency_penalty:    float   # ≤0: penalty for deferring urgent jobs
    invalid_action_pen: float   # ≤0: malformed actions, bad regions, etc.

    def to_dict(self) -> dict:
        return {
            "total":              round(self.total, 4),
            "carbon_score":       round(self.carbon_score, 4),
            "sla_score":          round(self.sla_score, 4),
            "deferral_quality":   round(self.deferral_quality, 4),
            "grid_stability":     round(self.grid_stability, 4),
            "urgency_penalty":    round(self.urgency_penalty, 4),
            "invalid_action_pen": round(self.invalid_action_pen, 4),
        }


# ── Weights ──────────────────────────────────────────────────────────────────
W_CARBON    = 0.40
W_SLA       = 0.30
W_DEFERRAL  = 0.15
W_STABILITY = 0.15


def compute_step_reward(
    assignments:    List[Assignment],
    jobs_map:       Dict[str, Job],
    datacenters:    Dict[str, DataCenter],
    current_hour:   int,
    noise_seed:     int = 0,
    valid_regions:  List[str] = None,
) -> RewardBreakdown:
    """
    Compute reward for a single scheduling step.
    Called once per step after the agent's action is parsed and applied.
    """
    if valid_regions is None:
        valid_regions = list(datacenters.keys())

    # ── 1. Carbon score ───────────────────────────────────────────────────
    total_naive_carbon   = 0.0
    total_actual_carbon  = 0.0
    carbon_assigned_count = 0

    for asgn in assignments:
        if asgn.deferred or asgn.region not in datacenters:
            continue
        job = jobs_map.get(asgn.job_id)
        if job is None:
            continue

        # actual carbon: energy × carbon intensity at assigned hour in assigned region
        dc = datacenters[asgn.region]
        forecast = get_carbon_forecast(asgn.region, current_hour, noise_seed)
        hour_offset = max(0, asgn.start_hour - current_hour)
        ci_at_start = forecast[min(hour_offset, len(forecast) - 1)]
        actual_carbon = job.energy_kwh * ci_at_start

        # naive: run right now in worst region
        naive_carbon = naive_carbon_for_job(job.energy_kwh, current_hour, noise_seed)

        total_actual_carbon += actual_carbon
        total_naive_carbon  += naive_carbon
        carbon_assigned_count += 1

    if total_naive_carbon > 0:
        carbon_score = max(0.0, (total_naive_carbon - total_actual_carbon) / total_naive_carbon)
    else:
        carbon_score = 0.0   # no assignments = no carbon saved = zero score

    # ── 2. SLA score ──────────────────────────────────────────────────────
    scheduled_count = sum(1 for a in assignments if not a.deferred and a.region in valid_regions)
    total_jobs      = len(assignments)

    sla_violations = 0
    for asgn in assignments:
        if asgn.deferred:
            continue
        job = jobs_map.get(asgn.job_id)
        if job is None:
            continue
        # Check: assigned start_hour does not exceed deadline
        if asgn.start_hour > job.deadline_hour:
            sla_violations += 1

    sla_score = 1.0 - (sla_violations / max(1, scheduled_count)) if scheduled_count > 0 else 0.0

    # ── 3. Deferral quality ───────────────────────────────────────────────
    # Good deferral: deferring LOW-priority jobs when carbon is high now
    # Bad deferral:  deferring URGENT jobs (they may miss deadline next step)
    urgency_penalty = 0.0
    smart_deferrals = 0
    dumb_deferrals  = 0

    for asgn in assignments:
        if not asgn.deferred:
            continue
        job = jobs_map.get(asgn.job_id)
        if job is None:
            continue

        if job.priority == Priority.URGENT:
            urgency_penalty -= 0.3   # heavy penalty: urgent job deferred
        elif job.priority == Priority.LOW and job.deferrable:
            smart_deferrals += 1
        else:
            dumb_deferrals  += 1

    total_deferrals = smart_deferrals + dumb_deferrals
    if total_deferrals > 0:
        deferral_quality = smart_deferrals / total_deferrals
    elif carbon_assigned_count == 0:
        deferral_quality = 0.0   # agent did nothing — punish
    else:
        deferral_quality = 0.8   # assigned everything, no deferrals — mostly fine

    # ── 4. Grid stability ─────────────────────────────────────────────────
    # Check that no region is overloaded by the batch of assignments
    region_load: Dict[str, float] = {r: 0.0 for r in valid_regions}
    for asgn in assignments:
        if asgn.deferred or asgn.region not in valid_regions:
            continue
        job = jobs_map.get(asgn.job_id)
        if job:
            region_load[asgn.region] += job.gpu_hours

    overload_score = 1.0
    for region, load in region_load.items():
        dc = datacenters[region]
        if load > dc.capacity_available:
            overload_fraction = (load - dc.capacity_available) / dc.capacity_available
            overload_score -= min(0.5, overload_fraction * 0.3)

    grid_stability = max(0.0, overload_score)

    # ── 5. Invalid action penalty ─────────────────────────────────────────
    invalid_action_pen = 0.0
    for asgn in assignments:
        if asgn.deferred:
            continue
        if asgn.region not in valid_regions:
            invalid_action_pen -= 0.1   # referenced a non-existent region
        if asgn.start_hour < current_hour:
            invalid_action_pen -= 0.1   # tried to schedule in the past
        if asgn.start_hour > 23:
            invalid_action_pen -= 0.1   # out of bounds hour

    # ── 6. Total ──────────────────────────────────────────────────────────
    total = (
        W_CARBON    * carbon_score +
        W_SLA       * sla_score +
        W_DEFERRAL  * deferral_quality +
        W_STABILITY * grid_stability +
        urgency_penalty +
        invalid_action_pen
    )
    total = max(-1.0, min(1.0, total))

    return RewardBreakdown(
        total              = total,
        carbon_score       = carbon_score,
        sla_score          = sla_score,
        deferral_quality   = deferral_quality,
        grid_stability     = grid_stability,
        urgency_penalty    = urgency_penalty,
        invalid_action_pen = invalid_action_pen,
    )


def compute_episode_reward(
    all_jobs:        List[Job],
    total_carbon:    float,
    naive_carbon:    float,
    episode_hours:   int = 24,
) -> Tuple[float, dict]:
    """
    End-of-episode summary reward — used for final scoring and logging.
    """
    completed = [j for j in all_jobs if j.completed]
    failed    = [j for j in all_jobs if j.failed]
    total     = len(all_jobs)

    completion_rate = len(completed) / max(1, total)
    failure_rate    = len(failed)    / max(1, total)

    # Carbon efficiency vs naive baseline
    if naive_carbon > 0:
        carbon_efficiency = max(0.0, 1.0 - total_carbon / naive_carbon)
    else:
        carbon_efficiency = 0.0

    # Urgency penalty: any urgent job that failed = large deduction
    urgent_failures = sum(1 for j in failed if j.priority == Priority.URGENT)
    urgency_deduction = urgent_failures * 0.2

    episode_reward = (
        carbon_efficiency * 0.45 +
        completion_rate   * 0.35 +
        (1 - failure_rate)* 0.20 -
        urgency_deduction
    )
    episode_reward = max(-1.0, min(1.0, episode_reward))

    return episode_reward, {
        "episode_reward":    round(episode_reward, 4),
        "carbon_efficiency": round(carbon_efficiency, 4),
        "completion_rate":   round(completion_rate, 4),
        "failure_rate":      round(failure_rate, 4),
        "urgent_failures":   urgent_failures,
        "total_carbon_gco2": round(total_carbon, 1),
        "naive_carbon_gco2": round(naive_carbon, 1),
        "carbon_saved_gco2": round(naive_carbon - total_carbon, 1),
    }
