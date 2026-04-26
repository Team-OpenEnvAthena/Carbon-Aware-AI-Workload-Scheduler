"""
Reward functions for CarbonSchedulerEnv.

FIXES vs previous version:
  [1] Carbon score: baseline is now best-available-region-right-now, not Mumbai.
      Mumbai (680 gCO2/kWh) was too easy to beat — even Virginia scored 47%
      without learning anything. The new baseline forces real temporal reasoning.

  [2] Deferral quality: carbon_threshold=300 was broken — Oregon (~50 gCO2/kWh)
      always keeps min_carbon_now below 300, so every deferral was classified
      "dumb". Fixed to a dynamic check: is the best forecast carbon in the next
      8 hours meaningfully lower (>15%) than the best carbon right now?

  [3] SLA score: was 1.0 when nothing scheduled — a free reward hacking vector.
      Fixed: deferred jobs with imminent deadlines (<=2 hours away) now
      contribute a proportional lateness penalty to SLA.

  [4] Grid stability: overload penalty was too soft (0.3 multiplier, 0.5 cap).
      Fixed: 0.6 multiplier, 1.0 cap. 100% overload now costs 0.60 on this
      component, making capacity violations impossible to ignore.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

from server.data_models import Job, DataCenter, InternalAssignment, Priority
from server.carbon_data import (
    naive_carbon_for_job,
    get_carbon_forecast,
)

@dataclass
class RewardBreakdown:
    total:              float
    carbon_score:       float
    sla_score:          float
    deferral_quality:   float
    grid_stability:     float
    urgency_penalty:    float
    invalid_action_pen: float
    parse_error_pen:    float = 0.0

    def to_dict(self) -> dict:
        return {k: round(v, 4) for k, v in self.__dict__.items()}


W_CARBON    = 0.40
W_SLA       = 0.30
W_DEFERRAL  = 0.15
W_STABILITY = 0.15

DEFER_SAVING_THRESHOLD  = 0.15   # future must be >15% cleaner to justify deferral
IMMINENT_DEADLINE_HOURS = 2      # jobs expiring within N hours are at SLA risk


def compute_step_reward(
    assignments:   List[InternalAssignment],
    jobs_map:      Dict[str, Job],
    datacenters:   Dict[str, DataCenter],
    current_hour:  int,
    noise_seed:    int = 0,
    valid_regions: List[str] = None,
    parse_error:   bool = False,
) -> RewardBreakdown:

    if valid_regions is None:
        valid_regions = list(datacenters.keys())

    # Pre-compute best current and near-future carbon across active regions
    best_ci_now = min((dc.carbon_now for dc in datacenters.values()), default=999.0)
    best_ci_future = min(
        (min(get_carbon_forecast(r, current_hour, noise_seed)[:8]) for r in valid_regions),
        default=best_ci_now,
    )
    future_is_cleaner = best_ci_future < best_ci_now * (1 - DEFER_SAVING_THRESHOLD)

    # ── 1. Carbon score (fix [1]) ─────────────────────────────────────────────
    total_naive  = 0.0
    total_actual = 0.0
    n_assigned   = 0

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
        # FIX [1]: naive = best region right now, not Mumbai
        naive_co2  = naive_carbon_for_job(
            job.energy_kwh, current_hour, noise_seed, list(datacenters.keys())
        )
        total_actual += actual_co2
        total_naive  += naive_co2
        n_assigned   += 1

    carbon_score = max(0.0, (total_naive - total_actual) / total_naive) if total_naive > 0 else 0.0

    # ── 2. SLA score — shaped + imminent-deadline deferral (fix [3]) ──────────
    sla_penalty_total = 0.0
    sla_job_count     = 0

    for asgn in assignments:
        job = jobs_map.get(asgn.job_id)
        if job is None:
            continue

        if not asgn.deferred:
            sla_job_count += 1
            if asgn.region in valid_regions and asgn.start_hour > job.deadline_hour:
                lateness = asgn.start_hour - job.deadline_hour
                window   = max(1, job.deadline_hour - job.created_hour)
                sla_penalty_total += min(1.0, lateness / window)
        else:
            # FIX [3]: penalise deferring a job that's about to expire
            hours_left = job.deadline_hour - current_hour
            if hours_left <= IMMINENT_DEADLINE_HOURS:
                sla_job_count += 1
                risk_penalty = max(0.4, 1.0 - hours_left / (IMMINENT_DEADLINE_HOURS + 1))
                sla_penalty_total += risk_penalty

    sla_score = max(0.0, 1.0 - sla_penalty_total / sla_job_count) if sla_job_count > 0 else 1.0

    # ── 3. Deferral quality (fix [2]) ─────────────────────────────────────────
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
            urgency_penalty -= 0.4
        elif job.priority == Priority.LOW and job.deferrable:
            # FIX [2]: dynamic check — is a cleaner window actually coming?
            if future_is_cleaner:
                smart_deferrals += 1
            else:
                dumb_deferrals  += 1
        else:
            dumb_deferrals += 1

    total_deferrals = smart_deferrals + dumb_deferrals
    if total_deferrals > 0:
        deferral_quality = smart_deferrals / total_deferrals
    elif n_assigned == 0 and len(assignments) > 0:
        deferral_quality = 0.0   # produced output but scheduled nothing
    else:
        deferral_quality = 0.8   # cleanly assigned, no deferral needed

    # ── 4. Grid stability (fix [4]) ───────────────────────────────────────────
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
            overload_frac   = (load - dc.capacity_available) / max(1.0, dc.capacity_available)
            grid_stability -= min(1.0, overload_frac * 0.6)   # FIX [4]: was * 0.3, cap was 0.5
    grid_stability = max(0.0, grid_stability)

    # ── 5. Invalid action penalties ───────────────────────────────────────────
    invalid_pen = 0.0
    for asgn in assignments:
        if asgn.deferred:
            continue
        if asgn.region not in valid_regions:
            invalid_pen -= 0.1
        if asgn.start_hour < current_hour:
            invalid_pen -= 0.1
        if not (0 <= asgn.start_hour <= 23):
            invalid_pen -= 0.1

    # ── 6. Parse error ────────────────────────────────────────────────────────
    # When the agent produces invalid JSON and there are pending jobs, override
    # deferral_quality to 0 so the total is reliably negative even when SLA/grid
    # scores are high (empty action list keeps them at default highs).
    if parse_error:
        parse_pen        = -0.3
        deferral_quality = 0.0   # agent did nothing useful
        if jobs_map:             # there were jobs to handle — zero out passive scores
            sla_score      = 0.0   # failed to address any pending job
            grid_stability = 0.0   # cannot claim grid health without acting
    else:
        parse_pen = 0.0

    # ── 7. Total ──────────────────────────────────────────────────────────────
    total = (
        W_CARBON    * carbon_score     +
        W_SLA       * sla_score        +
        W_DEFERRAL  * deferral_quality +
        W_STABILITY * grid_stability   +
        urgency_penalty                +
        invalid_pen                    +
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
    all_jobs:     List[Job],
    total_carbon: float,
    naive_carbon: float,
) -> Tuple[float, dict]:
    """End-of-episode summary. naive_carbon = best-region-now baseline."""
    completed = [j for j in all_jobs if j.completed]
    failed    = [j for j in all_jobs if j.failed]
    total     = len(all_jobs)

    completion_rate   = len(completed) / max(1, total)
    failure_rate      = len(failed)    / max(1, total)
    carbon_efficiency = (
        max(0.0, 1.0 - total_carbon / naive_carbon) if naive_carbon > 0 else 0.0
    )

    urgent_failures   = [j for j in failed if j.priority == Priority.URGENT]
    high_failures     = [j for j in failed if j.priority == Priority.HIGH]
    urgency_deduction = min(0.5, len(urgent_failures) * 0.15)
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
        "high_failures":     len(high_failures),
        "total_carbon_gco2": round(total_carbon, 1),
        "naive_carbon_gco2": round(naive_carbon, 1),
        "carbon_saved_gco2": round(naive_carbon - total_carbon, 1),
    }