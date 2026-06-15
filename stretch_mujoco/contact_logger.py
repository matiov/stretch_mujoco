"""
Contact monitoring for MuJoCo simulation.

Tracks currently-active contacts involving robot bodies.
Called once per physics step by MujocoServer; results are published via
MujocoServerProxies so the main process can read them with pull_contacts().
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Set, Tuple

import mujoco

# ---------------------------------------------------------------------------
# Robot body classification helpers
# ---------------------------------------------------------------------------
# Body names are taken directly from:
#   stretch_mujoco/models/stretch.xml            (robot links)
#   stretch_ros2/stretch_simulation/config/*.xml  (scene furniture / objects)

# Bodies that belong to the mobile base (drive system, laser, aruco tags on base)
_BASE_BODIES: frozenset[str] = frozenset(
    {
        "base_link",
        "base_imu",
        "link_left_wheel",
        "link_right_wheel",
        "laser",
        "link_aruco_right_base",
        "link_aruco_left_base",
    }
)

# Bodies that belong to the lift / arm chain
_ARM_BODIES: frozenset[str] = frozenset(
    {
        "link_mast",
        "link_lift",
        "link_arm_l4",
        "link_arm_l3",
        "link_arm_l2",
        "link_arm_l1",
        "link_arm_l0",
        "link_aruco_shoulder",
        "link_aruco_top_wrist",
        "link_aruco_inner_wrist",
    }
)

# Bodies that belong to the head / nav-camera assembly
_HEAD_BODIES: frozenset[str] = frozenset(
    {
        "link_head",
        "link_head_pan",
        "link_head_tilt",
        "realsense",
        "link_SE3_head_nav_cam",
        "head_nav_cam",
    }
)

# Bodies that belong to the wrist / gripper end-effector
_GRIPPER_BODIES: frozenset[str] = frozenset(
    {
        "link_wrist_yaw",
        "link_DW3_wrist_pitch",
        "link_SG3_gripper_body",
        "link_SG3_aruco_d405",
        "link_d405",
        "d405_cam",
        "link_grasp_center",
        "link_gripper_slider",
        "link_gripper_finger_left",
        "link_gripper_finger_right",
        "rubber_tip_left",
        "rubber_tip_right",
        "link_SG3_gripper_left_finger_aruco",
        "link_SG3_gripper_right_finger_aruco",
    }
)

# All robot bodies combined
_ALL_ROBOT_BODIES: frozenset[str] = _BASE_BODIES | _ARM_BODIES | _HEAD_BODIES | _GRIPPER_BODIES

# Prefixes that identify static environment / furniture bodies in the scene XMLs
# (bottle_scene.xml, coffee_scene.xml, last_generated_scene.xml from stretch_ros2)
_STATIC_PREFIXES: Tuple[str, ...] = (
    "floor_",
    "wall_",
    "ceiling_",
    "window_",
    "cab_",
    "counter_",
    "shelves_",
    "stool_",
    "stove_",
    "fridge_",
    "microwave_",
    "sink_",
    "dishwasher_",
    "knife_block_",
    "paper_towel_",
    "plant_",
    "utensil_",
    "toaster_",
    "light_switch_",
    "outlet_",
    "utensil_rack_",
    "utensil_holder_",
    # robocasa generic interior
    "world",
)


def _is_robot_body(name: str | None) -> bool:
    return name in _ALL_ROBOT_BODIES


def _is_static_body(name: str | None) -> bool:
    if name is None:
        return False
    return any(name.startswith(p) for p in _STATIC_PREFIXES)


def _classify(name: str | None) -> str:
    """Return a short category tag for a body name."""
    if name is None:
        return "unknown"
    if name in _BASE_BODIES:
        return "base"
    if name in _ARM_BODIES:
        return "arm"
    if name in _HEAD_BODIES:
        return "head"
    if name in _GRIPPER_BODIES:
        return "gripper"
    if _is_static_body(name):
        return "static"
    return "object"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class ContactEvent:
    sim_time: float
    body1_name: str
    body2_name: str
    geom1_name: str
    geom2_name: str
    normal_force: float  # Approximate resultant normal force (N)
    category: str  # e.g. "base-object", "arm-gripper", "gripper-object"


# ---------------------------------------------------------------------------
# ContactLogger
# ---------------------------------------------------------------------------


class ContactLogger:
    """
    Maintains the set of contacts active in the current physics step.

    Call `update()` once per step (done automatically by MujocoServer).
    Read `_active_contacts` (frozenset key -> ContactEvent) to get the current snapshot.

    Args:
        mjmodel: MuJoCo model.
        mjdata:  MuJoCo data.
        min_force: Minimum normal-force estimate (N) to include. Default 0 = all contacts.
        robot_only: Only include contacts where at least one body is a robot body.
        log_robot_self_contacts: Include contacts between two robot bodies (e.g. arm vs base).
    """

    def __init__(
        self,
        mjmodel: mujoco.MjModel,
        mjdata: mujoco.MjData,
        min_force: float = 0.0,
        robot_only: bool = True,
        log_robot_self_contacts: bool = True,
        verbose: bool = False,
    ) -> None:
        self.mjmodel = mjmodel
        self.mjdata = mjdata
        self.min_force = min_force
        self.robot_only = robot_only
        self.log_robot_self_contacts = log_robot_self_contacts
        self.verbose = verbose

        # Currently-active contacts: frozenset({geom1_id, geom2_id}) -> ContactEvent
        self._active_contacts: Dict[frozenset, ContactEvent] = {}

    def update(self) -> None:
        """
        Refresh _active_contacts to match the contacts present in this physics step.
        New pairs are added; pairs no longer in mjdata.contact are removed.
        """
        current_pairs: Set[frozenset] = set()

        for i in range(self.mjdata.ncon):
            contact = self.mjdata.contact[i]
            geom1_id = int(contact.geom1)
            geom2_id = int(contact.geom2)

            pair_key = frozenset((geom1_id, geom2_id))
            current_pairs.add(pair_key)

            if pair_key in self._active_contacts:
                continue

            body1_id = int(self.mjmodel.geom_bodyid[geom1_id])
            body2_id = int(self.mjmodel.geom_bodyid[geom2_id])
            body1_name = (
                mujoco.mj_id2name(self.mjmodel, mujoco.mjtObj.mjOBJ_BODY, body1_id)
                or f"body_{body1_id}"
            )
            body2_name = (
                mujoco.mj_id2name(self.mjmodel, mujoco.mjtObj.mjOBJ_BODY, body2_id)
                or f"body_{body2_id}"
            )

            is_robot1 = _is_robot_body(body1_name)
            is_robot2 = _is_robot_body(body2_name)

            if self.robot_only and not (is_robot1 or is_robot2):
                continue

            if not self.log_robot_self_contacts and is_robot1 and is_robot2:
                continue

            normal_force = self._normal_force(contact)
            if normal_force < self.min_force:
                continue

            geom1_name = (
                mujoco.mj_id2name(self.mjmodel, mujoco.mjtObj.mjOBJ_GEOM, geom1_id)
                or f"geom_{geom1_id}"
            )
            geom2_name = (
                mujoco.mj_id2name(self.mjmodel, mujoco.mjtObj.mjOBJ_GEOM, geom2_id)
                or f"geom_{geom2_id}"
            )

            event = ContactEvent(
                sim_time=self.mjdata.time,
                body1_name=body1_name,
                body2_name=body2_name,
                geom1_name=geom1_name,
                geom2_name=geom2_name,
                normal_force=normal_force,
                category=f"{_classify(body1_name)}-{_classify(body2_name)}",
            )
            self._active_contacts[pair_key] = event

            if self.verbose:
                print(
                    f"CONTACT [t={event.sim_time:.4f}s] {event.category}  "
                    f"{event.body1_name} <-> {event.body2_name}  "
                    f"force={event.normal_force:.4f} N"
                )

        # Drop pairs no longer present
        for key in set(self._active_contacts) - current_pairs:
            del self._active_contacts[key]

    @staticmethod
    def _normal_force(contact) -> float:
        try:
            dist = float(contact.dist)
            stiffness = float(contact.solref[0])
            if stiffness > 0 and dist < 0:
                return abs(dist) * stiffness
            return abs(float(contact.solref[0])) + abs(float(contact.solref[1]))
        except Exception:
            return 0.0
