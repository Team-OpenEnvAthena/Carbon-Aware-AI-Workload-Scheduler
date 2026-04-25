"""
Internal data models — server-side only.
These are NOT exported to the client. Only CarbonSchedulerAction
and CarbonSchedulerObservation cross the client/server boundary.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum


class Priority(str, Enum):
    URGENT = "urgent"   # hard deadline — cannot defer
    HIGH   = "high"     # soft deadline — can defer max 4 hours
    NORMAL = "normal"   # can defer up to 8 hours
    LOW    = "low"      # fully deferrable within 24h episode


@dataclass
class Job:
    id: str
    name: str
    gpu_hours: float
    energy_kwh: float
    priority: Priority
    deadline_hour: int
    created_hour: int
    deferrable: bool
    assigned_region: Optional[str] = None
    assigned_hour: Optional[int] = None
    completed: bool = False
    failed: bool = False

    def to_dict(self) -> dict:
        return {
            "id":              self.id,
            "name":            self.name,
            "gpu_hours":       round(self.gpu_hours, 1),
            "energy_kwh":      round(self.energy_kwh, 1),
            "priority":        self.priority.value,
            "deadline_hour":   self.deadline_hour,
            "created_hour":    self.created_hour,
            "deferrable":      self.deferrable,
            "assigned_region": self.assigned_region,
            "assigned_hour":   self.assigned_hour,
            "completed":       self.completed,
            "failed":          self.failed,
        }


@dataclass
class DataCenter:
    region: str
    name: str
    capacity_total: float
    capacity_used: float
    carbon_now: float
    carbon_forecast: List[float]
    renewable_pct: float

    @property
    def capacity_available(self) -> float:
        return max(0.0, self.capacity_total - self.capacity_used)

    def to_dict(self) -> dict:
        return {
            "region":             self.region,
            "name":               self.name,
            "capacity_total":     round(self.capacity_total, 1),
            "capacity_available": round(self.capacity_available, 1),
            "carbon_now":         round(self.carbon_now, 1),
            "carbon_forecast":    [round(c, 1) for c in self.carbon_forecast[:12]],
            "renewable_pct":      round(self.renewable_pct * 100, 1),
        }


@dataclass
class InternalAssignment:
    """Internal representation of a scheduling decision."""
    job_id: str
    region: str
    start_hour: int
    deferred: bool = False
