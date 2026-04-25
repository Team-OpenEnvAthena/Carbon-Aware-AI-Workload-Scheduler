"""
Carbon intensity profiles for 6 real data centre regions.
Values in gCO2eq/kWh, calibrated from Electricity Map 2024 data.

Profiles capture:
  - Solar regions: low midday, higher at night
  - Wind regions:  variable, often lowest at night / early morning
  - Hydro regions: consistently low
  - Gas/coal:      consistently high with minor variation
"""

import math
import random
from typing import List


# ── Regional base profiles (24-hour cycle, index = hour 0–23) ───────────────

def _solar_curve(base: float, peak_reduction: float) -> List[float]:
    """Dips around hours 10–15 when solar is generating."""
    result = []
    for h in range(24):
        solar = max(0.0, math.sin(math.pi * (h - 6) / 12))  # peaks at noon
        result.append(base - peak_reduction * solar)
    return result


def _wind_curve(base: float, amplitude: float, phase: float) -> List[float]:
    """Wind is variable — modelled as a slow sinusoid with noise seed."""
    return [
        base + amplitude * math.sin(2 * math.pi * h / 24 + phase)
        for h in range(24)
    ]


# Profiles keyed by region id
BASE_PROFILES = {
    "us-west-2": {
        "name":         "Oregon (Hydro + Wind)",
        "profile":      _wind_curve(base=45, amplitude=15, phase=0.5),
        "renewable":    0.82,
        "capacity":     200.0,
    },
    "us-west-1": {
        "name":         "California (Solar + Grid)",
        "profile":      _solar_curve(base=220, peak_reduction=140),
        "renewable":    0.52,
        "capacity":     150.0,
    },
    "us-east-1": {
        "name":         "Virginia (Gas + Nuclear)",
        "profile":      _wind_curve(base=360, amplitude=20, phase=1.0),
        "renewable":    0.24,
        "capacity":     180.0,
    },
    "eu-west-1": {
        "name":         "Ireland (Wind + Gas)",
        "profile":      _wind_curve(base=240, amplitude=80, phase=2.0),
        "renewable":    0.48,
        "capacity":     120.0,
    },
    "ap-southeast-1": {
        "name":         "Singapore (Natural Gas)",
        "profile":      _wind_curve(base=455, amplitude=10, phase=0.0),
        "renewable":    0.08,
        "capacity":     100.0,
    },
    "ap-south-1": {
        "name":         "Mumbai (Coal + Solar)",
        "profile":      _solar_curve(base=680, peak_reduction=200),
        "renewable":    0.18,
        "capacity":     90.0,
    },
}


def get_carbon_forecast(
    region: str,
    current_hour: int,
    noise_seed: int = 0,
    noise_level: float = 0.08,
) -> List[float]:
    """
    Return 24-hour forecast starting from current_hour.
    Adds realistic noise — the forecast is imperfect.
    """
    rng = random.Random(noise_seed + hash(region))
    profile = BASE_PROFILES[region]["profile"]

    forecast = []
    for offset in range(24):
        h = (current_hour + offset) % 24
        base_val = profile[h]
        noise = base_val * noise_level * (rng.random() * 2 - 1)
        forecast.append(max(10.0, base_val + noise))

    return forecast


def get_carbon_now(region: str, current_hour: int, noise_seed: int = 0) -> float:
    """Current carbon intensity — less noisy than forecast."""
    forecast = get_carbon_forecast(region, current_hour, noise_seed, noise_level=0.03)
    return round(forecast[0], 1)


def get_renewable_pct(region: str, current_hour: int) -> float:
    """Renewable percentage varies slightly with solar/wind availability."""
    base = BASE_PROFILES[region]["renewable"]
    profile_val  = BASE_PROFILES[region]["profile"][current_hour]
    profile_min  = min(BASE_PROFILES[region]["profile"])
    profile_max  = max(BASE_PROFILES[region]["profile"])
    spread       = max(1.0, profile_max - profile_min)
    # when carbon is low → renewables are high
    renewable_boost = 0.15 * (1 - (profile_val - profile_min) / spread)
    return min(1.0, max(0.0, base + renewable_boost))


def naive_carbon_for_job(
    energy_kwh: float,
    current_hour: int,
    noise_seed: int = 0,
) -> float:
    """
    Baseline: run job right now in the highest-traffic (worst) region.
    Used to compute carbon savings delta.
    """
    worst_region = "ap-south-1"
    ci = get_carbon_now(worst_region, current_hour, noise_seed)
    return energy_kwh * ci


def optimal_carbon_for_job(
    energy_kwh: float,
    current_hour: int,
    noise_seed: int = 0,
) -> float:
    """
    Oracle: run job at the cleanest future hour across all regions.
    Used to compute theoretical maximum savings (lower bound on reward).
    """
    best = float("inf")
    for region in BASE_PROFILES:
        forecast = get_carbon_forecast(region, current_hour, noise_seed)
        best = min(best, min(forecast))
    return energy_kwh * best
