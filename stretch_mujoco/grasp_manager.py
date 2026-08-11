"""
Grasp management for MuJoCo simulation.
Handles object grasping and attachment to the robot gripper.
"""

import numpy as np
from typing import Dict, List, Set, Tuple
import mujoco

import stretch_mujoco.config as config


class GraspManager:
    """
    Manages object grasping in MuJoCo simulation.

    This class detects when objects are in contact with the gripper and maintains
    grasp state. When an object is grasped (gripper closed + contact), it can be
    attached to the gripper's end effector.
    """

    def __init__(self, mjmodel: mujoco.MjModel, mjdata: mujoco.MjData):
        """
        Initialize the grasp manager.

        Args:
            mjmodel: MuJoCo model
            mjdata: MuJoCo data
        """
        self.mjmodel = mjmodel
        self.mjdata = mjdata

        # Track grasped objects: body_name -> grasp_info
        self.grasped_objects: Dict[str, Dict] = {}

        # Track equality constraint IDs for welds: body_name -> eq_id
        self._grasped_eq_ids: Dict[str, int] = {}

        # Cache geom contact settings so they can be restored on release.
        self._disabled_geom_contacts: Dict[str, List[Tuple[int, int, int]]] = {}

        # Contact filter state for ignoring gripper collisions while an object is held.
        self._contact_filter_model_id: int | None = None
        self._gripper_body_ids: Set[int] = set()
        self._excluded_body_pairs: Set[Tuple[int, int]] = set()
        self._recently_released_objects: Dict[str, float] = {}
        self._gripper_was_open: bool = True

        # Tracks mjdata.time so a sim reset (mj_resetData, respawn, etc.) can be detected: time
        # rewinding means every absolute-time timestamp we've stored (regrasp cooldowns, the
        # debug log throttle) is stale and must be cleared, or it can silently block re-grasping
        # for as long as the previous run had been going.
        self._last_seen_sim_time: float = mjdata.time

        # Temporary diagnostics: throttled reporting of near-gripper contact state, to help
        # pin down where a real grasp attempt is failing (contact never happening vs. happening
        # against an unexpected body vs. some other gate). Remove once grasping is reliable.
        self.debug_logging = False
        self._debug_log_interval_seconds = 0.5
        self._last_debug_log_time = float("-inf")

        # Gripper parameters
        self.gripper_finger_left = "link_gripper_finger_left"
        self.gripper_finger_right = "link_gripper_finger_right"
        self.gripper_attachment_point = "link_grasp_center"
        self.gripper_slider = "link_gripper_slider"
        self.gripper_tip_left = "rubber_tip_left"
        self.gripper_tip_right = "rubber_tip_right"

        # Grasp detection thresholds
        self.gripper_closed_threshold = (
            0.01  # Joint position threshold (meters) - gripper closed when < 0
        )
         # Minimum contact force to consider grasping.
        self.contact_force_threshold = 0.01
        # Timeout to ensure the object is released properly.
        self.regrasp_cooldown_seconds = 2.0

    def get_gripper_state(self) -> Dict:
        """
        Get current gripper state.

        Returns:
            Dict with gripper_closed (bool), joint_pos (float), and ctrl (float)
        """
        try:
            # Get gripper slide joint position (main gripper actuator)
            gripper_joint = mujoco.mj_name2id(
                self.mjmodel, mujoco.mjtObj.mjOBJ_JOINT, "joint_gripper_slide"
            )
            if gripper_joint < 0:
                raise ValueError("Gripper joint not found")

            gripper_pos = self.mjdata.qpos[self.mjmodel.jnt_qposadr[gripper_joint]]

            gripper_actuator = mujoco.mj_name2id(
                self.mjmodel, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper"
            )
            if gripper_actuator < 0:
                raise ValueError("Gripper actuator not found")
            gripper_ctrl = self.mjdata.ctrl[gripper_actuator]

            # Base "closed" on the commanded target (ctrl), not the measured joint position.
            # The slide actuator is stiff (kp=4000) and stalls against whatever it's squeezing,
            # so once an object is between the cups the measured qpos plateaus well above the
            # threshold even though the gripper was commanded fully closed. Gating on qpos meant
            # grasping silently failed for any object thick enough to block full closure.
            gripper_closed = gripper_ctrl < self.gripper_closed_threshold

            return {
                "closed": gripper_closed,
                "joint_pos": gripper_pos,
                "ctrl": gripper_ctrl,
            }
        except Exception as e:
            print(f"Error getting gripper state: {e}")
            return {"closed": False, "joint_pos": 0.0, "ctrl": 0.0}

    def get_body_contacts(self, body_name: str) -> list:
        """
        Get all contacts involving a specific body.

        Args:
            body_name: Name of the body

        Returns:
            List of contact information dictionaries
        """
        body_id = mujoco.mj_name2id(self.mjmodel, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            return []

        contacts = []
        for i in range(self.mjdata.ncon):
            contact = self.mjdata.contact[i]
            geom1_id = contact.geom1
            geom2_id = contact.geom2

            geom1_body = self.mjmodel.geom_bodyid[geom1_id]
            geom2_body = self.mjmodel.geom_bodyid[geom2_id]

            if geom1_body == body_id or geom2_body == body_id:
                other_body = geom2_body if geom1_body == body_id else geom1_body
                other_body_name = mujoco.mj_id2name(
                    self.mjmodel, mujoco.mjtObj.mjOBJ_BODY, other_body
                )
                contacts.append(
                    {
                        "other_body": other_body_name,
                        "contact": contact,
                    }
                )

        return contacts

    def is_object_in_contact_with_gripper(self, object_name: str) -> Tuple[bool, float]:
        """
        Check if the object is pinched between the two fingers, i.e. in contact with the left
        side of the gripper AND the right side simultaneously.

        A single-sided touch (e.g. only the right finger grazing the object, from whichever
        direction) is not a grasp: for a parallel-jaw gripper the only way both sides register
        contact on the same body at once is if it's actually nested between them, so requiring
        both is a robust stand-in for "the object is within the two fingers" without needing to
        reason about contact normals directly.

        Args:
            object_name: Name of the object body

        Returns:
            Tuple of (is_in_contact, contact_force)
        """
        contacts = self.get_body_contacts(object_name)

        left_bodies = {self.gripper_finger_left, self.gripper_tip_left}
        right_bodies = {self.gripper_finger_right, self.gripper_tip_right}

        max_force = 0.0
        left_touching = False
        right_touching = False

        for contact_info in contacts:
            other_body = contact_info["other_body"]
            if other_body in left_bodies:
                left_touching = True
            elif other_body in right_bodies:
                right_touching = True
            else:
                continue

            # Estimate contact force (simplified - using normal force)
            contact = contact_info["contact"]
            # Contact force estimation from contact data
            force_mag = abs(contact.solref[0]) + abs(contact.solref[1])
            max_force = max(max_force, force_mag)
            # print(f"Object {object_name} in contact with {other_body}")

        in_contact = left_touching and right_touching

        return in_contact, max_force

    def get_attachment_point_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the pose (position and orientation) of the grasp center.

        Returns:
            Tuple of (position, quaternion)
        """
        try:
            body_id = mujoco.mj_name2id(
                self.mjmodel, mujoco.mjtObj.mjOBJ_BODY, self.gripper_attachment_point
            )
            if body_id < 0:
                raise ValueError(f"Body {self.gripper_attachment_point} not found")

            pos = self.mjdata.body(self.gripper_attachment_point).xpos
            mat = self.mjdata.body(self.gripper_attachment_point).xmat.reshape(3, 3)

            # Convert rotation matrix to quaternion
            # Using MuJoCo's mju_mat2Quat
            quat = np.zeros(4)
            mujoco._functions.mju_mat2Quat(quat, mat.flatten())

            return np.array(pos), quat
        except Exception as e:
            print(f"Error getting attachment point pose: {e}")
            return np.zeros(3), np.array([1, 0, 0, 0])

    def _get_body_quat(self, body_name: str) -> np.ndarray:
        """
        Get a body's world-frame orientation as a quaternion.

        Args:
            body_name: Name of the body

        Returns:
            Quaternion in MuJoCo wxyz order
        """
        body_mat = self.mjdata.body(body_name).xmat.reshape(3, 3)
        body_quat = np.zeros(4)
        mujoco.mju_mat2Quat(body_quat, body_mat.reshape(-1))
        return body_quat

    def _quat_conjugate(self, quat: np.ndarray) -> np.ndarray:
        """Return the inverse of a unit quaternion."""
        return np.array([quat[0], -quat[1], -quat[2], -quat[3]])

    def _mul_quat(self, quat_a: np.ndarray, quat_b: np.ndarray) -> np.ndarray:
        """Multiply two quaternions in MuJoCo wxyz order."""
        result = np.zeros(4)
        mujoco.mju_mulQuat(result, quat_a, quat_b)
        return result

    def _set_object_collisions_enabled(self, object_name: str, enabled: bool) -> None:
        """
        Toggle collisions for all geoms directly attached to a grasped body.

        This is the stable fallback: while an object is grasped we disable all of its
        contact resolution, then restore the original masks on release.
        """
        body_id = mujoco.mj_name2id(self.mjmodel, mujoco.mjtObj.mjOBJ_BODY, object_name)
        if body_id < 0:
            return

        geom_ids = np.flatnonzero(self.mjmodel.geom_bodyid == body_id)
        if enabled:
            cached_settings = self._disabled_geom_contacts.pop(object_name, [])
            for geom_id, contype, conaffinity in cached_settings:
                self.mjmodel.geom_contype[geom_id] = contype
                self.mjmodel.geom_conaffinity[geom_id] = conaffinity
            return

        if object_name in self._disabled_geom_contacts:
            return

        cached_settings: List[Tuple[int, int, int]] = []
        for geom_id in geom_ids:
            cached_settings.append(
                (
                    int(geom_id),
                    int(self.mjmodel.geom_contype[geom_id]),
                    int(self.mjmodel.geom_conaffinity[geom_id]),
                )
            )
            self.mjmodel.geom_contype[geom_id] = 0
            self.mjmodel.geom_conaffinity[geom_id] = 0

        if cached_settings:
            self._disabled_geom_contacts[object_name] = cached_settings

    def _refresh_contact_filter_cache(self) -> None:
        """
        Refresh body-level data used by the custom contact filter.
        """
        model_id = id(self.mjmodel)
        if self._contact_filter_model_id == model_id:
            return

        self._contact_filter_model_id = model_id

        gripper_body_names = {
            self.gripper_finger_left,
            self.gripper_finger_right,
            self.gripper_slider,
            self.gripper_tip_left,
            self.gripper_tip_right,
        }
        self._gripper_body_ids = {
            body_id
            for body_name in gripper_body_names
            if (body_id := mujoco.mj_name2id(self.mjmodel, mujoco.mjtObj.mjOBJ_BODY, body_name))
            >= 0
        }

        self._excluded_body_pairs = set()
        exclude_body1 = getattr(self.mjmodel, "exclude_body1id", None)
        exclude_body2 = getattr(self.mjmodel, "exclude_body2id", None)
        if exclude_body1 is not None and exclude_body2 is not None:
            self._excluded_body_pairs = {
                tuple(sorted((int(body1), int(body2))))
                for body1, body2 in zip(exclude_body1, exclude_body2)
            }

    def _prune_regrasp_cooldowns(self) -> None:
        """Drop expired post-release cooldown entries."""
        if not self._recently_released_objects:
            return

        current_time = self.mjdata.time
        expired_names = [
            object_name
            for object_name, release_time in self._recently_released_objects.items()
            if current_time >= release_time
        ]
        for object_name in expired_names:
            del self._recently_released_objects[object_name]

    def _body_name(self, body_id: int) -> str | None:
        if body_id < 0:
            return None
        return mujoco.mj_id2name(self.mjmodel, mujoco.mjtObj.mjOBJ_BODY, body_id)

    def _default_contact_filter(self, geom1_id: int, geom2_id: int) -> bool:
        """
        Approximate MuJoCo's default contact filter before applying grasp-specific rules.
        """
        contype_1 = int(self.mjmodel.geom_contype[geom1_id])
        conaffinity_1 = int(self.mjmodel.geom_conaffinity[geom1_id])
        contype_2 = int(self.mjmodel.geom_contype[geom2_id])
        conaffinity_2 = int(self.mjmodel.geom_conaffinity[geom2_id])
        if not ((contype_1 & conaffinity_2) or (contype_2 & conaffinity_1)):
            return False

        body1_id = int(self.mjmodel.geom_bodyid[geom1_id])
        body2_id = int(self.mjmodel.geom_bodyid[geom2_id])
        if body1_id == body2_id:
            return False

        body_parentid = getattr(self.mjmodel, "body_parentid", None)
        if body_parentid is not None and (
            int(body_parentid[body1_id]) == body2_id or int(body_parentid[body2_id]) == body1_id
        ):
            return False

        if tuple(sorted((body1_id, body2_id))) in self._excluded_body_pairs:
            return False

        body_weldid = getattr(self.mjmodel, "body_weldid", None)
        if body_weldid is not None and int(body_weldid[body1_id]) == int(body_weldid[body2_id]):
            return False

        return True

    def should_allow_contact(self, geom1_id: int, geom2_id: int) -> bool:
        """
        Custom contact filter that only suppresses object-gripper collisions while held
        or during a short post-release cooldown.
        """
        self._refresh_contact_filter_cache()
        self._prune_regrasp_cooldowns()

        if not self._default_contact_filter(geom1_id, geom2_id):
            return False

        body1_id = int(self.mjmodel.geom_bodyid[geom1_id])
        body2_id = int(self.mjmodel.geom_bodyid[geom2_id])
        body1_name = self._body_name(body1_id)
        body2_name = self._body_name(body2_id)

        body1_blocked = (
            body1_name in self.grasped_objects or body1_name in self._recently_released_objects
        )
        body2_blocked = (
            body2_name in self.grasped_objects or body2_name in self._recently_released_objects
        )

        if body1_blocked and body2_id in self._gripper_body_ids:
            return False
        if body2_blocked and body1_id in self._gripper_body_ids:
            return False

        return True

    def _handle_possible_sim_reset(self) -> None:
        """
        Detect mjdata.time rewinding (mj_resetData / respawn / any other reset) and drop all
        absolute-time bookkeeping. Without this, a regrasp cooldown timestamped e.g. 45.35s before
        a reset stays in _recently_released_objects and silently blocks grasping that same body
        name until sim time climbs all the way back past 45.35s post-reset.
        """
        current_time = self.mjdata.time
        if current_time < self._last_seen_sim_time:
            self._recently_released_objects.clear()
            self._last_debug_log_time = float("-inf")
            self._gripper_was_open = not self.get_gripper_state()["closed"]
        self._last_seen_sim_time = current_time

    def update_grasps(self) -> None:
        """
        Update grasp state for all objects.
        Should be called once per simulation step.
        """
        self._handle_possible_sim_reset()
        gripper_state = self.get_gripper_state()
        self._prune_regrasp_cooldowns()

        # Release fires on the rising edge of "open" (a fresh open command) rather than on
        # every step where the gripper merely reads as open. Grasping (below) no longer requires
        # the gripper to read "closed", so a contact-triggered grasp can happen while ctrl is
        # still at whatever open-ish value it had before the close command arrived; using a level
        # trigger here would immediately release that same object on the very next step just
        # because ctrl hadn't caught up yet.
        gripper_open = not gripper_state["closed"]
        opened_this_step = gripper_open and not self._gripper_was_open
        self._gripper_was_open = gripper_open

        if opened_this_step:
            for grasped_name in list(self.grasped_objects.keys()):
                self.release_object(grasped_name)

        # Grasping is triggered by contact with the rubber cups (or gripper) alone, regardless of
        # gripper_state["closed"]: the object itself can be what's physically stopping the
        # gripper from ever reaching a "closed" reading, so contact is already sufficient
        # evidence the cups are on the object. Requiring "closed" too used to cause grasps to be
        # missed (and the object to get shoved out from between the cups) whenever the closing
        # motion hadn't yet reached the threshold by the time contact happened.
        should_log = self.debug_logging and (
            self.mjdata.time - self._last_debug_log_time >= self._debug_log_interval_seconds
        )
        if should_log:
            self._last_debug_log_time = self.mjdata.time

        any_graspable_found = False
        for i in range(self.mjmodel.nbody):
            body_name = mujoco.mj_id2name(self.mjmodel, mujoco.mjtObj.mjOBJ_BODY, i)
            if body_name is None:
                continue

            if not self._is_graspable_object(body_name):
                continue
            any_graspable_found = True

            is_in_contact, contact_force = self.is_object_in_contact_with_gripper(body_name)

            if should_log:
                all_contacts = [c["other_body"] for c in self.get_body_contacts(body_name)]
                print(
                    f"[grasp debug] object={body_name} in_graspable_contact={is_in_contact} "
                    f"all_contacts={all_contacts} gripper_closed={gripper_state['closed']} "
                    f"gripper_ctrl={gripper_state['ctrl']:.4f} already_grasped={body_name in self.grasped_objects}"
                )

            if is_in_contact:
                if (
                    body_name not in self.grasped_objects
                    and body_name not in self._recently_released_objects
                ):
                    obj_name = config.REPLACEMENTS[body_name]
                    print(f"Grasping object: {obj_name} with contact force {contact_force}")
                    self.grasp_object(body_name)

        if should_log and not any_graspable_found:
            print(
                "[grasp debug] no bodies currently pass _is_graspable_object() "
                "(check body naming / free-joint setup for your target object)"
            )

    def _get_grasp_eq_id(self, object_name: str) -> int:
        """
        Look up (and cache) the id of the pre-declared, normally-inactive weld equality
        constraint that rigidly attaches `object_name` to the gripper attachment point.

        Returns -1 if no such constraint was declared in the model for this object.
        """
        if object_name in self._grasped_eq_ids:
            return self._grasped_eq_ids[object_name]

        eq_id = mujoco.mj_name2id(
            self.mjmodel, mujoco.mjtObj.mjOBJ_EQUALITY, f"grasp_{object_name}"
        )
        self._grasped_eq_ids[object_name] = eq_id
        return eq_id

    def grasp_object(self, object_name: str) -> None:
        """
        Grasp an object by attaching it to the gripper.

        Args:
            object_name: Name of the object body to grasp
        """
        if object_name in self.grasped_objects:
            return

        try:
            # Get current relative pose
            pos, gripper_quat = self.get_attachment_point_pose()
            obj_pos = self.mjdata.body(object_name).xpos
            obj_quat = self._get_body_quat(object_name)

            # Calculate relative transform (object in gripper frame)
            gripper_mat = self.mjdata.body(self.gripper_attachment_point).xmat.reshape(3, 3)
            rel_pos = obj_pos - pos
            rel_pos_local = gripper_mat.T @ rel_pos
            rel_quat = self._mul_quat(self._quat_conjugate(gripper_quat), obj_quat)

            self.grasped_objects[object_name] = {
                "attachment_point": self.gripper_attachment_point,
                "relative_pos": rel_pos_local,
                "relative_quat": rel_quat,
                "grasp_time": self.mjdata.time,
            }
            self._recently_released_objects.pop(object_name, None)
            self._set_object_collisions_enabled(object_name, enabled=False)

            # Rigidly weld the object to the gripper via a real equality constraint (enforced
            # by the solver every substep) instead of only re-imposing the pose after the fact.
            # anchor=(0,0,0) + relpose=(rel_pos_local, rel_quat) welds the object's origin to
            # rel_pos_local/rel_quat away from the gripper attachment frame, exactly matching
            # the relative transform captured above.
            eq_id = self._get_grasp_eq_id(object_name)
            if eq_id >= 0:
                self.mjmodel.eq_data[eq_id, 0:3] = 0.0
                self.mjmodel.eq_data[eq_id, 3:6] = rel_pos_local
                self.mjmodel.eq_data[eq_id, 6:10] = rel_quat
                self.mjdata.eq_active[eq_id] = 1
            else:
                print(
                    f"No pre-declared grasp weld for {object_name}; "
                    "falling back to per-step pose correction."
                )
        except Exception as e:
            print(f"Error grasping object {object_name}: {e}")

    def release_object(self, object_name: str) -> None:
        """
        Release a grasped object.

        Args:
            object_name: Name of the object body to release
        """
        if object_name in self.grasped_objects:
            self._set_object_collisions_enabled(object_name, enabled=True)
            self._recently_released_objects[object_name] = (
                self.mjdata.time + self.regrasp_cooldown_seconds
            )

            eq_id = self._grasped_eq_ids.get(object_name, -1)
            if eq_id >= 0:
                self.mjdata.eq_active[eq_id] = 0

            del self.grasped_objects[object_name]

    def apply_grasp_constraints(self) -> None:
        """
        Fallback pose correction for grasped objects that have no pre-declared weld equality
        constraint in the model. Objects with an active weld are already held rigidly by the
        solver and don't need this. Should be called in the control callback.
        """
        if not self.grasped_objects:
            return

        try:
            for object_name, grasp_info in self.grasped_objects.items():
                eq_id = self._grasped_eq_ids.get(object_name, -1)
                if eq_id >= 0:
                    continue
                self._constrain_object_to_gripper(object_name, grasp_info)
        except Exception as e:
            print(f"Error applying grasp constraints: {e}")

    def _constrain_object_to_gripper(self, object_name: str, grasp_info: Dict) -> None:
        """
        Constrain an object to follow the gripper using position/orientation alignment.

        Args:
            object_name: Name of the object body
            grasp_info: Grasp information dictionary
        """
        try:
            # Get gripper and object IDs
            gripper_body_id = mujoco.mj_name2id(
                self.mjmodel, mujoco.mjtObj.mjOBJ_BODY, grasp_info["attachment_point"]
            )
            object_body_id = mujoco.mj_name2id(self.mjmodel, mujoco.mjtObj.mjOBJ_BODY, object_name)

            if gripper_body_id < 0 or object_body_id < 0:
                return

            # Get current gripper pose and desired object pose
            gripper_mat = self.mjdata.body(grasp_info["attachment_point"]).xmat.reshape(3, 3)
            gripper_pos = self.mjdata.body(grasp_info["attachment_point"]).xpos
            gripper_quat = self._get_body_quat(grasp_info["attachment_point"])

            # Calculate desired object pose in world frame
            rel_pos = grasp_info["relative_pos"]
            desired_pos = gripper_pos + gripper_mat @ rel_pos
            desired_quat = self._mul_quat(gripper_quat, grasp_info["relative_quat"])

            # Get object state
            obj_qpos_adr = self.mjmodel.body_jntadr[object_body_id]
            obj_jnt_type = self.mjmodel.jnt_type[obj_qpos_adr]

            # Check if it's a free joint (assumed for grasped objects)
            if obj_jnt_type == mujoco.mjtJoint.mjJNT_FREE:
                # Update position (qpos[0:3])
                qpos_adr = self.mjmodel.jnt_qposadr[obj_qpos_adr]
                self.mjdata.qpos[qpos_adr : qpos_adr + 3] = desired_pos

                # Preserve the grasp-time relative orientation in the gripper frame.
                self.mjdata.qpos[qpos_adr + 3 : qpos_adr + 7] = desired_quat

                # Remove accumulated linear and angular velocity while the object is held.
                qvel_adr = self.mjmodel.jnt_dofadr[obj_qpos_adr]
                self.mjdata.qvel[qvel_adr : qvel_adr + 6] = 0.0

        except Exception as e:
            print(f"Error constraining object: {e}")

    def _is_graspable_object(self, body_name: str) -> bool:
        """
        Check if a body is a graspable object.

        Args:
            body_name: Name of the body

        Returns:
            True if the body is graspable, False otherwise
        """
        # Objects typically have "object" in their name or are free-floating bodies
        # This is a heuristic and may need adjustment based on your scene setup
        graspable_keywords = ["object", "coffee_cup", "item", "target"]

        # Skip robot and environment bodies
        non_graspable_keywords = [
            "base",
            "link_",
            "floor",
            "table",
            "world",
            "gripper",
            "finger",
            "wrist",
            "arm",
            "lift",
            "head",
            "mast",
        ]

        # Check if body matches non-graspable keywords
        for keyword in non_graspable_keywords:
            if keyword in body_name.lower():
                return False

        # Check if body has a free joint (objects typically do)
        try:
            body_id = mujoco.mj_name2id(self.mjmodel, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id >= 0:
                jnt_adr = self.mjmodel.body_jntadr[body_id]
                if jnt_adr >= 0:
                    jnt_type = self.mjmodel.jnt_type[jnt_adr]
                    return jnt_type == mujoco.mjtJoint.mjJNT_FREE
        except:
            pass

        return False
