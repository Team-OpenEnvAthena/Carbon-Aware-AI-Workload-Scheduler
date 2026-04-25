"""
Reward functions for CarbonSchedulerEnv.

Fixes vs original:
  - Imports now work correctly (server-internal only)
  - SLA penalty is SHAPED (continuous) not binary — gradient flows in early training
  - Deferral quality accounts for carbon context (smart defer = low priority + high carbon now)
  - Episode reward uses shaped urgency penalty not hard cutoff

Design: multiple independent components, each bounded, each interpretable.
Hard to game: gaming carbon_score tanks sla_score; gaming deferral tanks urgency.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

from .data_models import Job, DataCenter, InternalAssignment, Priority
from .carbon_data import naive_carbon_for_job, get_carbon_forecast, get_carbon_now


@dataclass
class RewardBreakdown:
    total: float
    carbon_score: float
    sla_score: float
    deferral_quality: float
    grid_stability: float
    urgency_penalty: float
    invalid_action_pen: float
    parse_error_pen: float = 0.0

    def to_dict(self) -> dict:
        return {k: round(v, 4) for k, v in self.__dict__.items()}


# ── Weights ────────────────────────────────────────────────────────────────────
W_CARBON    = 0.40
W_SLA       = 0.30
W_DEFERRAL  = 0.15
W_STABILITY = 0.15


def compute_step_reward(
    assignments: List[InternalAssignment],
    jobs_map: Dict[str, Job],
    datacenters: Dict[str, DataCenter],
    current_hour: int,
    noise_seed: int = 0,
    valid_regions: List[str] = None,
    parse_error: bool = False,
) -> RewardBreakdown:
    """
    Compute reward for one scheduling step.

    FIX: SLA is now a shaped continuous penalty, not a hard -1.0 cutoff.
    This allows gradients to flow during early training when the model
    frequently misses deadlines.
    """
    if valid_regions is None:
        valid_regions = list(datacenters.keys())

    # ── 1. Carbon score ────────────────────────────────────────────────────
    total_naive   = 0.0
    total_actual  = 0.0
    n_assigned    = 0

    for asgn in assignments:
        if asgn.deferred or asgn.region not in datacenters:
            continue
        job = jobs_map.get(asgn.job_id)
        if job is None:
            continue

        forecast   = get_carbon_forecast(asgn.region, current_hour, noise_seed)
        offset     = max(0, asgn.start_hour - current_hour)
        ci         = forecast[min(offset, len(forecast) - 1)]
        actual_co2 = job.energy_kwh * ci
        naive_co2  = naive_carbon_for_job(job.energy_kwh, current_hour, noise_seed)

        total_actual += actual_co2
        total_naive  += naive_co2
        n_assigned   += 1

    if total_naive > 0:
        carbon_score = max(0.0, (total_naive - total_actual) / total_naive)
    else:
        carbon_score = 0.0

    # ── 2. SLA score — SHAPED (not binary) ────────────────────────────────
    # FIX: instead of returning -1.0 for any violation, we compute a
    # proportional penalty based on how late each job is.
    # This gives the model a gradient signal even when it misses deadlines.
    sla_score = 1.0
    n_scheduled = sum(1 for a in assignments if not a.deferred and a.region in valid_regions)

    if n_scheduled > 0:
        total_lateness = 0.0
        for asgn in assignments:
            if asgn.deferred or asgn.region not in valid_regions:
                continue
            job = jobs_map.get(asgn.job_id)
            if job is None:
                continue
            if asgn.start_hour > job.deadline_hour:
                # Proportional penalty — 1 hour late = small penalty, 6 hours late = large
                lateness_ratio = (asgn.start_hour - job.deadline_hour) / max(1, 24 - job.created_hour)
                total_lateness += min(1.0, lateness_ratio)

        sla_score = max(0.0, 1.0 - (total_lateness / n_scheduled))

    # ── 3. Deferral quality ────────────────────────────────────────────────
    # Smart defer: LOW priority job + carbon_now is HIGH (wait for cleaner window)
    # Dumb defer:  URGENT job deferred at all
    urgency_penalty = 0.0
    smart_deferrals = 0
    dumb_deferrals  = 0
    carbon_threshold = 300.0  # gCO2/kWh — above this, deferring low-priority is smart

    for asgn in assignments:
        if not asgn.deferred:
            continue
        job = jobs_map.get(asgn.job_id)
        if job is None:
            continue

        if job.priority == Priority.URGENT:
            urgency_penalty -= 0.4   # heavy — urgent job must never be deferred
        elif job.priority == Priority.LOW and job.deferrable:
            # Check if current carbon is actually high (smart defer)
            dc_carbons = [dc.carbon_now for dc in datacenters.values()]
            min_carbon_now = min(dc_carbons) if dc_carbons else 999
            if min_carbon_now > carbon_threshold:
                smart_deferrals += 1
            else:
                dumb_deferrals += 1  # deferring when carbon is already low
        else:
            dumb_deferrals += 1

    total_deferrals = smart_deferrals + dumb_deferrals
    if total_deferrals > 0:
        deferral_quality = smart_deferrals / total_deferrals
    elif n_assigned == 0 and len(assignments) > 0:
        deferral_quality = 0.0   # did nothing
    else:
        deferral_quality = 0.8   # assigned everything cleanly

    # ── 4. Grid stability ──────────────────────────────────────────────────
    region_load: Dict[str, float] = {r: 0.0 for r in valid_regions}
    for asgn in assignments:
        if asgn.deferred or asgn.region not in valid_regions:
            continue
        job = jobs_map.get(asgn.job_id)
        if job:
            region_load[asgn.region] = region_load.get(asgn.region, 0.0) + job.gpu_hours

    grid_stability = 1.0
    for region, load in region_load.items():
        dc = datacenters.get(region)
        if dc and load > dc.capacity_available:
            overload_frac = (load - dc.capacity_available) / max(1.0, dc.capacity_available)
            grid_stability -= min(0.5, overload_frac * 0.3)
    grid_stability = max(0.0, grid_stability)

    # ── 5. Invalid action penalties ────────────────────────────────────────
    invalid_pen = 0.0
    for asgn in assignments:
        if asgn.deferred:
            continue
        if asgn.region not in valid_regions:
            invalid_pen -= 0.1
        if asgn.start_hour < current_hour:
            invalid_pen -= 0.1
        if asgn.start_hour > 23 or asgn.start_hour < 0:
            invalid_pen -= 0.1

    # ── 6. Parse error penalty ─────────────────────────────────────────────
    parse_pen = -0.3 if parse_error else 0.0

    # ── 7. Total ───────────────────────────────────────────────────────────
    total = (
        W_CARBON    * carbon_score +
        W_SLA       * sla_score +
        W_DEFERRAL  * deferral_quality +
        W_STABILITY * grid_stability +
        urgency_penalty +
        invalid_pen +
        parse_pen
    )
    total = max(-1.0, min(1.0, total))

    return RewardBreakdown(
        total              = total,
        carbon_score       = carbon_score,
        sla_score          = sla_score,
        deferral_quality   = deferral_quality,
        grid_stability     = grid_stability,
        urgency_penalty    = urgency_penalty,
        invalid_action_pen = invalid_pen,
        parse_error_pen    = parse_pen,
    )


def compute_episode_reward(
    all_jobs: List[Job],
    total_carbon: float,
    naive_carbon: float,
) -> Tuple[float, dict]:
    """End-of-episode summary reward — shaped, not binary."""
    completed = [j for j in all_jobs if j.completed]
    failed    = [j for j in all_jobs if j.failed]
    total     = len(all_jobs)

    completion_rate  = len(completed) / max(1, total)
    failure_rate     = len(failed)    / max(1, total)
    carbon_efficiency = max(0.0, 1.0 - total_carbon / naive_carbon) if naive_carbon > 0 else 0.0

    # Shaped urgency — proportional, not binary cutoff
    urgent_failures   = [j for j in failed if j.priority == Priority.URGENT]
    urgency_deduction = min(0.5, len(urgent_failures) * 0.15)

    # High-priority failures also matter
    high_failures     = [j for j in failed if j.priority == Priority.HIGH]
    high_deduction    = min(0.2, len(high_failures) * 0.05)

    episode_reward = (
        carbon_efficiency * 0.45 +
        completion_rate   * 0.35 +
        (1 - failure_rate)* 0.20 -
        urgency_deduction -
        high_deduction
    )
    episode_reward = max(-1.0, min(1.0, episode_reward))

    return episode_reward, {
        "episode_reward":    round(episode_reward, 4),
        "carbon_efficiency": round(carbon_efficiency, 4),
        "completion_rate":   round(completion_rate, 4),
        "failure_rate":      round(failure_rate, 4),
        "urgent_failures":   len(urgent_failures),
        "total_carbon_gco2": round(total_carbon, 1),
        "naive_carbon_gco2": round(naive_carbon, 1),
        "carbon_saved_gco2": round(naive_carbon - total_carbon, 1),
    }
