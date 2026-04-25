from dataclasses import dataclass, field
from typing import List, Optional, Dict
from enum import Enum


class Priority(str, Enum):
    URGENT = "urgent"   # hard deadline, cannot defer, runs immediately or fails
    HIGH   = "high"     # soft deadline, can defer max 4 hours
    NORMAL = "normal"   # can defer up to 8 hours
    LOW    = "low"      # fully deferrable within the 24h episode


@dataclass
class Job:
    id: str
    name: str
    gpu_hours: float          # compute units required
    energy_kwh: float         # energy this job consumes
    priority: Priority
    deadline_hour: int        # must START by this hour or it fails
    created_hour: int         # hour it entered the queue
    deferrable: bool
    # filled in after scheduling
    assigned_region: Optional[str] = None
    assigned_hour:   Optional[int] = None
    completed:       bool = False
    failed:          bool = False

    def to_dict(self) -> dict:
        return {
            "id":             self.id,
            "name":           self.name,
            "gpu_hours":      self.gpu_hours,
            "energy_kwh":     self.energy_kwh,
            "priority":       self.priority.value,
            "deadline_hour":  self.deadline_hour,
            "created_hour":   self.created_hour,
            "deferrable":     self.deferrable,
            "assigned_region": self.assigned_region,
            "assigned_hour":   self.assigned_hour,
            "completed":      self.completed,
            "failed":         self.failed,
        }


@dataclass
class DataCenter:
    region: str
    name: str
    capacity_total: float        # max GPU-hours schedulable per hour
    capacity_used:  float        # GPU-hours already scheduled this hour
    carbon_now:     float        # gCO2/kWh right now
    carbon_forecast: List[float] # next 24 hours of carbon intensity
    renewable_pct:  float        # 0–1, fraction from renewables

    @property
    def capacity_available(self) -> float:
        return max(0.0, self.capacity_total - self.capacity_used)

    def to_dict(self) -> dict:
        return {
            "region":            self.region,
            "name":              self.name,
            "capacity_total":    round(self.capacity_total, 1),
            "capacity_available": round(self.capacity_available, 1),
            "carbon_now":        round(self.carbon_now, 1),
            "carbon_forecast":   [round(c, 1) for c in self.carbon_forecast[:12]],
            "renewable_pct":     round(self.renewable_pct * 100, 1),
        }


@dataclass
class Assignment:
    job_id:     str
    region:     str
    start_hour: int
    deferred:   bool = False


@dataclass
class Action:
    """What the agent outputs each step — a list of job assignments."""
    assignments: List[Assignment]

    @classmethod
    def from_dict(cls, d: dict) -> "Action":
        assignments = []
        for a in d.get("assignments", []):
            assignments.append(Assignment(
                job_id     = a["job_id"],
                region     = a.get("region", ""),
                start_hour = a.get("start_hour", -1),
                deferred   = a.get("defer", False),
            ))
        return cls(assignments=assignments)


@dataclass
class Observation:
    current_hour:       int
    step_number:        int
    jobs_pending:       List[dict]
    jobs_completed:     int
    jobs_failed:        int
    datacenters:        List[dict]
    carbon_saved_so_far: float   # gCO2 saved vs naive (run-immediately) baseline
    budget_remaining:   float    # normalised budget units

    def to_prompt(self) -> str:
        """Convert observation to the text prompt the LLM sees."""
        lines = [
            f"=== Carbon-Aware Scheduler | Hour {self.current_hour:02d}:00 | Step {self.step_number} ===",
            "",
            "PENDING JOBS:",
        ]
        for j in self.jobs_pending:
            defer_tag = "(deferrable)" if j["deferrable"] else "(NOT deferrable)"
            lines.append(
                f"  {j['id']}: {j['name']} | {j['gpu_hours']} GPU-hrs | "
                f"{j['energy_kwh']} kWh | priority={j['priority']} | "
                f"deadline=hour {j['deadline_hour']} {defer_tag}"
            )

        lines += ["", "DATA CENTERS (carbon forecast = next 12 hours):"]
        for dc in self.datacenters:
            forecast_str = " → ".join(str(int(c)) for c in dc["carbon_forecast"][:6])
            lines.append(
                f"  {dc['region']}: {dc['name']} | carbon now={dc['carbon_now']} gCO2/kWh | "
                f"forecast: {forecast_str} | available={dc['capacity_available']} GPU-hrs | "
                f"renewables={dc['renewable_pct']}%"
            )

        lines += [
            "",
            f"Progress: {self.jobs_completed} completed, {self.jobs_failed} failed, "
            f"{self.carbon_saved_so_far:.0f} gCO2 saved so far.",
            "",
            "TASK: Schedule each pending job. For each job output one of:",
            '  {"job_id": "X", "region": "us-west-2", "start_hour": 14}  -- assign to region at hour',
            '  {"job_id": "X", "defer": true}                             -- defer to next step',
            "",
            "Respond with ONLY valid JSON:",
            '{"assignments": [ ... ]}',
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "current_hour":        self.current_hour,
            "step_number":         self.step_number,
            "jobs_pending":        self.jobs_pending,
            "jobs_completed":      self.jobs_completed,
            "jobs_failed":         self.jobs_failed,
            "datacenters":         self.datacenters,
            "carbon_saved_so_far": round(self.carbon_saved_so_far, 2),
            "budget_remaining":    round(self.budget_remaining, 2),
            "prompt":              self.to_prompt(),
        }
