from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ContactInfo:
    """
    A single active contact between two bodies, captured at one physics step.

    Picklable so it can cross the multiprocessing boundary via MujocoServerProxies.
    """

    sim_time: float
    body1_name: str
    body2_name: str
    geom1_name: str
    geom2_name: str
    normal_force: float
    category: str  # e.g. "base-static", "gripper-object", "arm-arm"


@dataclass
class StatusStretchContacts:
    """
    Snapshot of all contacts active in the most recent physics step.

    Shared from the MuJoCo process to the main process through MujocoServerProxies.
    Only contains currently-active contacts — nothing from previous steps is retained.
    """

    contacts: list[ContactInfo] = field(default_factory=list)
    sim_time: float = 0.0

    @staticmethod
    def default() -> "StatusStretchContacts":
        return StatusStretchContacts(contacts=[], sim_time=0.0)
