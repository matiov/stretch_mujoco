"""
Grasp management for MuJoCo simulation.
Handles object grasping and attachment to the robot gripper.
"""

import numpy as np
from typing import Dict, Optional, Tuple
import mujoco


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

        # Gripper parameters
        self.gripper_finger_left = "link_gripper_finger_left"
        self.gripper_finger_right = "link_gripper_finger_right"
        self.gripper_attachment_point = "link_grasp_center"
        self.gripper_slider = "link_gripper_slider"
        self.gripper_tip_left = "rubber_tip_left"
        self.gripper_tip_right = "rubber_tip_right"

        # Grasp detection thresholds
        self.gripper_closed_threshold = (
            0.01
        )  # Joint position threshold (meters) - gripper closed when < 0
        self.contact_force_threshold = 0.01  # Minimum contact force to consider grasping

    def get_gripper_state(self) -> Dict:
        """
        Get current gripper state.

        Returns:
            Dict with gripper_closed (bool), joint_pos (float), and finger_positions
        """
        try:
            # Get gripper slide joint position (main gripper actuator)
            gripper_joint = mujoco.mj_name2id(
                self.mjmodel, mujoco.mjtObj.mjOBJ_JOINT, "joint_gripper_slide"
            )
            if gripper_joint < 0:
                raise ValueError("Gripper joint not found")

            gripper_pos = self.mjdata.qpos[self.mjmodel.jnt_qposadr[gripper_joint]]

            # Gripper is considered closed when joint position is negative (fingers moved inward)
            # Adjust threshold based on your gripper's range
            gripper_closed = gripper_pos < self.gripper_closed_threshold

            return {
                "closed": gripper_closed,
                "joint_pos": gripper_pos,
            }
        except Exception as e:
            print(f"Error getting gripper state: {e}")
            return {"closed": False, "joint_pos": 0.0}

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
        Check if object is in contact with gripper.

        Args:
            object_name: Name of the object body

        Returns:
            Tuple of (is_in_contact, contact_force)
        """
        contacts = self.get_body_contacts(object_name)

        graspable_bodies = {
            self.gripper_finger_left,
            self.gripper_finger_right,
            self.gripper_slider,
            self.gripper_tip_left,
            self.gripper_tip_right,
        }

        max_force = 0.0
        in_contact = False

        for contact_info in contacts:
            other_body = contact_info["other_body"]
            if other_body in graspable_bodies:
                in_contact = True
                # Estimate contact force (simplified - using normal force)
                contact = contact_info["contact"]
                # Contact force estimation from contact data
                force_mag = abs(contact.solref[0]) + abs(contact.solref[1])
                max_force = max(max_force, force_mag)
                # print(f"Object {object_name} in contact with {other_body}")

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

    def update_grasps(self) -> None:
        """
        Update grasp state for all objects.
        Should be called once per simulation step.
        """
        gripper_state = self.get_gripper_state()

        # Only release objects when the gripper is open (not closed)
        if not gripper_state["closed"]:
            # print("Gripper is open, releasing all grasped objects...")
            # Release all currently grasped objects
            for grasped_name in list(self.grasped_objects.keys()):
                self.release_object(grasped_name)

        else:
            # Find all objects in the scene
            for i in range(self.mjmodel.nbody):
                body_name = mujoco.mj_id2name(self.mjmodel, mujoco.mjtObj.mjOBJ_BODY, i)
                if body_name is None:
                    continue

                # Skip non-object bodies
                if not self._is_graspable_object(body_name):
                    continue
                
                is_in_contact, contact_force = self.is_object_in_contact_with_gripper(body_name)

                # Only grasp if gripper is closed and in contact, and not already grasped
                if gripper_state["closed"] and is_in_contact:
                    if body_name not in self.grasped_objects:
                        print(f"Grasping object: {body_name} with contact force {contact_force}")
                        self.grasp_object(body_name)

        

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
            pos, quat = self.get_attachment_point_pose()
            obj_pos = self.mjdata.body(object_name).xpos
            obj_mat = self.mjdata.body(object_name).xmat.reshape(3, 3)
            obj_quat = np.zeros(4)
            mujoco._functions.mju_mat2Quat(obj_quat, obj_mat.flatten())

            # Calculate relative transform (object in gripper frame)
            gripper_mat = self.mjdata.body(self.gripper_attachment_point).xmat.reshape(3, 3)
            rel_pos = obj_pos - pos
            rel_pos_local = gripper_mat.T @ rel_pos

            self.grasped_objects[object_name] = {
                "attachment_point": self.gripper_attachment_point,
                "relative_pos": rel_pos_local,
                "relative_quat": obj_quat,
                "grasp_time": self.mjdata.time,
            }

            # Weld constraint code removed for compatibility with MuJoCo Python API
        except Exception as e:
            print(f"Error grasping object {object_name}: {e}")

    def release_object(self, object_name: str) -> None:
        """
        Release a grasped object.

        Args:
            object_name: Name of the object body to release
        """
        if object_name in self.grasped_objects:
            # Weld constraint code removed for compatibility with MuJoCo Python API
            del self.grasped_objects[object_name]

    def apply_grasp_constraints(self) -> None:
        """
        Forcefully update the pose of all grasped objects to follow the gripper every step.
        This ensures the object stays attached, regardless of MuJoCo weld constraint behavior.
        Should be called in the control callback.
        """
        if not self.grasped_objects:
            return

        try:
            for object_name, grasp_info in self.grasped_objects.items():
                # Always update pose for free joint objects
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

            # Calculate desired object pose in world frame
            rel_pos = grasp_info["relative_pos"]
            desired_pos = gripper_pos + gripper_mat @ rel_pos

            # Get object state
            obj_qpos_adr = self.mjmodel.body_jntadr[object_body_id]
            obj_jnt_type = self.mjmodel.jnt_type[obj_qpos_adr]

            # Check if it's a free joint (assumed for grasped objects)
            if obj_jnt_type == mujoco.mjtJoint.mjJNT_FREE:
                # Update position (qpos[0:3])
                qpos_adr = self.mjmodel.jnt_qposadr[obj_qpos_adr]
                self.mjdata.qpos[qpos_adr : qpos_adr + 3] = desired_pos

                # Update orientation to match gripper (simplified - could improve)
                obj_quat = grasp_info["relative_quat"]
                self.mjdata.qpos[qpos_adr + 3 : qpos_adr + 7] = obj_quat

                # Dampen velocities to prevent jerky motion
                qvel_adr = self.mjmodel.jnt_dofadr[obj_qpos_adr]
                self.mjdata.qvel[qvel_adr : qvel_adr + 6] *= 0.95

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
