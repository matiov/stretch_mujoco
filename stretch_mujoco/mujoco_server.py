import contextlib
from dataclasses import dataclass
import math
import multiprocessing
from multiprocessing.managers import AcquirerProxy, DictProxy, SyncManager
import os
import signal
import threading
import time
from typing import Callable

import click
import mujoco
import mujoco._functions
import mujoco._enums
import numpy as np
from mujoco._structs import MjData, MjModel
import mujoco._enums

from stretch_mujoco.datamodels.status_stretch_camera import StatusStretchCameras
from stretch_mujoco.datamodels.status_stretch_contacts import ContactInfo, StatusStretchContacts
from stretch_mujoco.datamodels.status_stretch_joints import StatusStretchJoints
from stretch_mujoco.datamodels.status_stretch_sensors import StatusStretchSensors
from stretch_mujoco.enums.actuators import Actuators
from stretch_mujoco.enums.stretch_cameras import StretchCameras
import stretch_mujoco.config as config
from stretch_mujoco.enums.stretch_sensors import StretchSensors
from stretch_mujoco.mujoco_server_camera_manager import (
    MujocoServerCameraManagerThreaded,
    MujocoServerCameraManagerSync,
)
from stretch_mujoco.datamodels.status_command import (
    CommandBaseVelocity,
    CommandGraspObject,
    CommandMove,
    CommandTeleport,
    CommandTeleportObject,
    StatusCommand,
)
from stretch_mujoco.mujoco_server_sensor_manager import MujocoServerSensorManagerThreaded
import stretch_mujoco.utils as utils
from stretch_mujoco.utils import FpsCounter
from stretch_mujoco.grasp_manager import GraspManager
from stretch_mujoco.contact_logger import ContactLogger


@dataclass
class MujocoServerProxies:
    _command: "DictProxy[str, StatusCommand]"
    _status: "DictProxy[str, StatusStretchJoints]"
    _cameras: "DictProxy[str, StatusStretchCameras]"
    _sensors: "DictProxy[str, StatusStretchSensors]"
    _joint_limits: "DictProxy[str, dict[Actuators, tuple[float, float]]]"
    _contacts: "DictProxy[str, StatusStretchContacts]"

    def __setattr__(self, name: str, value) -> None:
        try:
            super().__setattr__(name, value)
        except BrokenPipeError:
            ...

    def get_status(self) -> StatusStretchJoints:
        return self._status["val"]

    def set_status(self, value: StatusStretchJoints):
        self._status["val"] = value

    def get_command(self) -> StatusCommand:
        return self._command["val"]

    def set_command(self, value: StatusCommand):
        self._command["val"] = value

    def get_cameras(self) -> StatusStretchCameras:
        return self._cameras["val"]

    def set_cameras(self, value: StatusStretchCameras):
        self._cameras["val"] = value

    def get_sensors(self) -> StatusStretchSensors:
        return self._sensors["val"]

    def set_sensors(self, value: StatusStretchSensors):
        self._sensors["val"] = value

    def get_joint_limits(self) -> dict[Actuators, tuple[float, float]]:
        return self._joint_limits["val"]

    def set_joint_limit(self, actuator: Actuators, min_max: tuple[float, float]):
        limits = self._joint_limits["val"]
        limits[actuator] = min_max

        self._joint_limits["val"] = limits

    def get_contacts(self) -> StatusStretchContacts:
        return self._contacts["val"]

    def set_contacts(self, value: StatusStretchContacts) -> None:
        self._contacts["val"] = value

    @staticmethod
    def default(manager: SyncManager) -> "MujocoServerProxies":
        return MujocoServerProxies(
            _command=manager.dict({"val": StatusCommand.default()}),
            _status=manager.dict({"val": StatusStretchJoints.default()}),
            _cameras=manager.dict({"val": StatusStretchCameras.default()}),
            _sensors=manager.dict({"val": StatusStretchSensors.default()}),
            _joint_limits=manager.dict({"val": {}}),
            _contacts=manager.dict({"val": StatusStretchContacts.default()}),
        )


class BaseController:
    """
    Note on `push_command`/`start_pose`: a multi-joint FollowJointTrajectory goal that
    includes `translate_mobile_base` alongside other joints (lift/arm/wrist - the
    common case for an IK-driven reach that also has to shift the base) dispatches one
    command per joint from the driver process in a tight loop. Each dispatch does its
    own read-modify-write of the shared command proxy. Before `_ctrl_callback` held
    `_command_lock` around that cycle, a dispatch landing between this process's read
    and write-back could revive an already-processed `translate_mobile_base` trigger,
    re-running `push_command` and resetting `start_pose` to wherever the base had
    already gotten to. Each reset restarts the "distance traveled so far" measurement
    `_base_translate_by` uses to know when to stop, so the base could be made to chase
    a moving target well past the originally commanded distance - not settling within
    the trajectory server's wait timeout and getting reported as obstructed even
    though nothing was actually in its way.
    """

    # Below this, the ramped velocity is treated as "stopped" and braking ends -
    # comfortably tighter than wait_while_is_moving's own settling atol (0.0005 in
    # stretch_mujoco_simulator.py), so it never masks that check.
    _BRAKE_DONE_LINEAR = 1e-4  # m/s
    _BRAKE_DONE_ANGULAR = 1e-4  # rad/s

    # Remaining distance/angle below which a move is considered "arrived" and the
    # command is cleared outright, rather than fed through the decelerate-to-stop
    # profile in _speed_for_remaining (which asymptotes towards, but never exactly
    # reaches, zero remaining distance).
    _ARRIVED_LINEAR = 0.002  # m
    _ARRIVED_ANGULAR = 0.005  # rad

    def __init__(self, mujoco_server: "MujocoServer") -> None:
        self.mujoco_server = mujoco_server
        self.last_command: CommandMove | CommandBaseVelocity | None = None
        self.start_pose = np.array([0, 0, 0])
        # Ramped (not commanded) velocity - see _set_base_velocity.
        self._current_v_linear = 0.0
        self._current_omega = 0.0
        self._braking = False

    def push_command(self, command: CommandMove | CommandBaseVelocity):
        """Push a command to the base. Call `update()` to set the next trajectory."""
        self.last_command = command
        self.start_pose = self.get_base_pose()
        self._braking = False

    def _clear_command(self, is_stop_motion: bool):
        self.last_command = None

        # Braking is also ramped (see _set_base_velocity), so it takes more than
        # one update() to bring the wheels to rest - keep ticking it from update()
        # below until the ramp actually reaches zero, instead of snapping the
        # velocity target to zero in a single step here.
        if is_stop_motion:
            self._braking = True

    def update(self):
        """
        The update method to set mujoco ctrl's for the base while in motion.
        """
        if self._braking:
            self._set_base_velocity(0.0, 0.0)
            if (
                abs(self._current_v_linear) < self._BRAKE_DONE_LINEAR
                and abs(self._current_omega) < self._BRAKE_DONE_ANGULAR
            ):
                self._braking = False
                # Snap the last, negligible-but-nonzero ramp residual to an exact
                # stop rather than leaving the wheels creeping at ~0.1 mm/s forever.
                self._current_v_linear = 0.0
                self._current_omega = 0.0
                self._set_base_velocity(0.0, 0.0)
            return

        if self.last_command is None:
            return

        if isinstance(self.last_command, CommandMove):
            return self.handle_move_by(self.last_command)

        if isinstance(self.last_command, CommandBaseVelocity):
            return self._set_base_velocity(self.last_command.v_linear, self.last_command.omega)

    def get_base_pose(self) -> np.ndarray:
        """Get the se(2) base pose: x, y, and theta"""
        xyz = self.mujoco_server.mjdata.body("base_link").xpos
        rotation = self.mujoco_server.mjdata.body("base_link").xmat.reshape(3, 3)
        theta = np.arctan2(rotation[1, 0], rotation[0, 0])
        return np.array([xyz[0], xyz[1], theta])

    def handle_move_by(self, command: CommandMove):
        if command.actuator_name == Actuators.base_translate.name:
            return self._base_translate_by(
                command.pos,
            )

        if command.actuator_name == Actuators.base_rotate.name:
            return self._base_rotate_by(
                command.pos,
            )

        raise NotImplementedError(f"Actuator {command.actuator_name} is not supported.")

    @staticmethod
    def _speed_for_remaining(remaining: float, cruise_speed: float, max_accel: float) -> float:
        """
        Target speed (magnitude) for a point mass decelerating at `max_accel` to land
        at exactly zero once `remaining` reaches zero: v = sqrt(2 * a * d), capped at
        cruise speed. Used to start slowing down *before* the target is reached
        (instead of driving at cruise speed until the target is passed and only then
        braking), so the base arrives instead of overshooting and having to walk back.
        """
        return min(cruise_speed, math.sqrt(2.0 * max_accel * max(remaining, 0.0)))

    def _base_translate_by(self, x_inc: float) -> None:
        """
        Translate the base by a certain w.r.t base global pose
        """
        start_pose = self.start_pose[:2]

        sign = 1 if x_inc > 0 else -1
        remaining = abs(x_inc) - np.linalg.norm(self.get_base_pose()[:2] - start_pose)
        if remaining <= self._ARRIVED_LINEAR:
            return self._clear_command(is_stop_motion=True)

        speed = self._speed_for_remaining(
            remaining, config.base_motion["default_x_vel"], config.base_motion["max_linear_accel"]
        )
        self._set_base_velocity(speed * sign, 0)

    def _base_rotate_by(self, theta_inc: float) -> None:
        """
        Rotate the base by a certain w.r.t base global pose
        """
        start_pose = self.start_pose[-1]
        sign = 1 if theta_inc > 0 else -1
        remaining = abs(theta_inc) - abs(start_pose - self.get_base_pose()[-1])
        if remaining <= self._ARRIVED_ANGULAR:
            return self._clear_command(is_stop_motion=True)

        speed = self._speed_for_remaining(
            remaining, config.base_motion["default_r_vel"], config.base_motion["max_angular_accel"]
        )
        self._set_base_velocity(0, speed * sign)

    def _set_base_velocity(self, v_linear: float, omega: float) -> None:
        """
        Ramp the base towards (v_linear, omega) and drive the wheels at the ramped
        velocity, rather than commanding the target directly.

        Args:
            v_linear: float, target linear velocity
            omega: float, target angular velocity
        """
        # Ramping the commanded velocity.
        dt = self.mujoco_server.mjmodel.opt.timestep
        max_dv = config.base_motion["max_linear_accel"] * dt
        max_domega = config.base_motion["max_angular_accel"] * dt
        self._current_v_linear += np.clip(v_linear - self._current_v_linear, -max_dv, max_dv)
        self._current_omega += np.clip(omega - self._current_omega, -max_domega, max_domega)

        w_left, w_right = utils.diff_drive_inv_kinematics(
            self._current_v_linear, self._current_omega
        )

        left_actuator = self.mujoco_server.mjdata.actuator(Actuators.left_wheel_vel.name)
        right_actuator = self.mujoco_server.mjdata.actuator(Actuators.right_wheel_vel.name)
        left_gear = self.mujoco_server.mjmodel.actuator(Actuators.left_wheel_vel.name).gear[0]
        right_gear = self.mujoco_server.mjmodel.actuator(Actuators.right_wheel_vel.name).gear[0]
        left_actuator.ctrl = w_left * left_gear
        right_actuator.ctrl = w_right * right_gear


class MujocoServer:
    """
    Use `MucocoServer.launch_server()` to start the headless simulator.

    This uses the mujoco simulator in headless mode.
    """

    @classmethod
    def launch_server(
        cls,
        scene_xml_path: str | None,
        model: MjModel | None,
        camera_hz: float,
        show_viewer_ui: bool,
        stop_mujoco_process_event: threading.Event,
        data_proxies: MujocoServerProxies,
        cameras_to_use: list[StretchCameras],
        start_translation: list | None,
        start_rotation_quat: list | None,
        command_lock: "AcquirerProxy | None" = None,
    ):
        server = cls(
            scene_xml_path,
            model,
            stop_mujoco_process_event,
            data_proxies,
            start_translation,
            start_rotation_quat,
            command_lock,
        )
        server.run(
            show_viewer_ui=show_viewer_ui,
            camera_hz=camera_hz,
            cameras_to_use=cameras_to_use,
        )

    def change_start_pose(
        self, model: MjModel, translation: list | None, rotation_quat: list | None
    ):
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")

        if body_id == -1:
            raise ValueError("Body 'base_link' not found in the MjModel.")

        joint_id = -1
        for j in range(model.njnt):
            if model.jnt_bodyid[j] == body_id:
                joint_id = j
                break

        # Since the model has a Free Joint, we must change the default QPOS (qpos0).
        qadr = model.jnt_qposadr[joint_id]

        if translation is not None:
            model.qpos0[qadr : qadr + 3] = translation

        if rotation_quat is not None:
            model.qpos0[qadr + 3 : qadr + 7] = rotation_quat

        print(f"Start pose: {model.qpos0[qadr:qadr+3]}, {model.qpos0[qadr+3:qadr+7]}")

        return model

    def __init__(
        self,
        scene_xml_path: str | None,
        model: MjModel | None,
        stop_mujoco_process_event: threading.Event,
        data_proxies: MujocoServerProxies,
        start_translation: list | None,
        start_rotation_quat: list | None,
        command_lock: "AcquirerProxy | None" = None,
    ):
        """
        Initialize the Simulator handle with a scene
        Args:
            scene_xml_path: str, path to the scene xml file
            model: MjModel, Mujoco model object
            command_lock: cross-process lock shared with the `StretchMujocoSimulator`
                client. Must be held around every read-modify-write of the shared
                command proxy (see `_ctrl_callback`) so a client dispatch (`move_to`/
                `move_by`/...) can't land between this process's read and write-back
                and get silently dropped or re-triggered. Falls back to a fresh,
                unshared lock for single-process use (e.g. the `tests/` scripts).
        """
        if scene_xml_path is None:
            scene_xml_path = utils.default_scene_xml_path

        if model is None:
            model = MjModel.from_xml_path(scene_xml_path)

        model = self.change_start_pose(model, start_translation, start_rotation_quat)

        self.mjmodel = model
        # utils.print_wheel_velocity_ctrlranges(self.mjmodel)

        self.mjdata = MjData(self.mjmodel)

        self._base_in_pos_motion = False

        self._stop_mujoco_process_event = stop_mujoco_process_event

        self._command_lock = command_lock if command_lock is not None else multiprocessing.Lock()

        self.data_proxies = data_proxies

        self.base_controller = BaseController(self)

        self.physics_fps_counter = FpsCounter()

        self.sensor_manager = MujocoServerSensorManagerThreaded(
            sensor_hz=15,
            sensors_to_use=StretchSensors.from_mjmodel(self.mjmodel),
            mujoco_server=self,
        )

        self.grasp_manager = GraspManager(self.mjmodel, self.mjdata)

        # ContactLogger runs inside this process and has direct access to mjmodel/mjdata.
        # verbose=False: we publish via the proxy instead of printing per-step.
        self.contact_logger = ContactLogger(
            mjmodel=self.mjmodel,
            mjdata=self.mjdata,
            verbose=False,
        )

        self.update_joint_limits()

        # Cache the free joint addresses for base_link (used by teleport)
        base_body_id = mujoco.mj_name2id(self.mjmodel, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        self._base_link_qadr: int = -1
        self._base_link_qdadr: int = -1
        for j in range(self.mjmodel.njnt):
            if self.mjmodel.jnt_bodyid[j] == base_body_id:
                self._base_link_qadr = int(self.mjmodel.jnt_qposadr[j])
                self._base_link_qdadr = int(self.mjmodel.jnt_dofadr[j])
                break

        signal.signal(signal.SIGTERM, lambda num, h: self.request_to_stop())
        signal.signal(signal.SIGINT, lambda num, h: self.request_to_stop())

    def update_joint_limits(self):
        for i in range(self.mjmodel.njnt):
            name = mujoco._functions.mj_id2name(self.mjmodel, mujoco._enums.mjtObj.mjOBJ_JOINT, i)
            joint_range = self.mjmodel.jnt_range[i]  # This gives [lower_limit, upper_limit]
            try:
                actuator = Actuators.get_actuator_by_joint_names_in_mjcf(name)
                self.data_proxies.set_joint_limit(
                    actuator=actuator, min_max=(joint_range[0], joint_range[1])
                )
            except:
                ...

    def set_camera_manager(
        self,
        camera_hz: float,
        cameras_to_use: list[StretchCameras],
        *,
        use_camera_thread: bool,
        use_threadpool_executor: bool,
    ):
        """
        This should be called before trying to render offscreen cameras.

        If `use_camera_thread` is false, `self.camera_manager.pull_camera_data_at_camera_rate()` should be called on a UI thread.
        This is the recommended usage.

        If `use_camera_thread` is true, a thread will be spawned to call Renderer.render().
        This may not work on all platforms since rendering should happen on the main thread.
        This mode is mainly used with the Mujoco Managed Viewer, to avoid rendering on the physics thread.
        """
        if use_camera_thread or use_threadpool_executor:
            self.camera_manager = MujocoServerCameraManagerThreaded(
                use_camera_thread=use_camera_thread,
                use_threadpool_executor=use_threadpool_executor,
                camera_hz=camera_hz,
                cameras_to_use=cameras_to_use,
                mujoco_server=self,
            )
        else:
            self.camera_manager = MujocoServerCameraManagerSync(
                camera_hz=camera_hz, cameras_to_use=cameras_to_use, mujoco_server=self
            )

    def run(
        self,
        show_viewer_ui: bool,
        camera_hz: float,
        cameras_to_use: list[StretchCameras],
    ):
        # self.__run_headless_simulation(camera_hz=camera_hz, cameras_to_use=cameras_to_use)
        self.__run_headless_simulation_with_physics_thread(
            camera_hz=camera_hz, cameras_to_use=cameras_to_use
        )

    def _is_requested_to_stop(self):
        try:
            return self._stop_mujoco_process_event.is_set()
        except (EOFError, BrokenPipeError):
            # We likely lost connection to the main process if we've hit this.
            return True

    def request_to_stop(self):
        try:
            self._stop_mujoco_process_event.set()
        except (EOFError, BrokenPipeError):
            # We likely lost connection to the main process if we've hit this.
            ...

    def close(self):
        """
        Clean up C++ resources
        """
        self.request_to_stop()

        if isinstance(self.camera_manager, MujocoServerCameraManagerThreaded):
            self.camera_manager.cameras_thread.join()

        if isinstance(self.sensor_manager, MujocoServerSensorManagerThreaded):
            self.sensor_manager.sensors_thread.join()

        self.camera_manager.close()

    def _run_ui_simulation(self, show_viewer_ui: bool) -> None:
        """
        Run the simulation with the viewer
        """
        raise NotImplementedError(
            "This is headless mode. Use MujocoServerPassive or MujocoServerManaged to run the UI simulator."
        )

    def _physics_step(self, lock: contextlib.AbstractContextManager):
        """
        Calls mj_step and _ctrl_callback, and sleeps until the next timestep.
        """
        start_time = time.perf_counter()

        with lock:
            mujoco._functions.mj_step(self.mjmodel, self.mjdata)
            self._ctrl_callback(self.mjmodel, self.mjdata)

        time_until_next_step = self.mjmodel.opt.timestep - (time.perf_counter() - start_time)
        if time_until_next_step > 0:
            # Sleep to match the timestep.
            time.sleep(time_until_next_step)

    def _physics_loop(
        self, lock: contextlib.AbstractContextManager, termination_check: Callable[[], bool]
    ):
        """
        A loop to use when starting physics in a thread.
        """
        while termination_check():
            self._physics_step(lock=lock)

        click.secho("Physics Loop has terminated.", fg="red")

    def __run_headless_simulation(
        self, camera_hz: float, cameras_to_use: list[StretchCameras]
    ) -> None:
        """
        Run the simulation without the viewer headless.

        Headless mode manages its own `set_camera_manager()` call.
        """
        print("Running headless simulation...")

        self.set_camera_manager(
            use_camera_thread=False,
            use_threadpool_executor=False,
            camera_hz=camera_hz,
            cameras_to_use=cameras_to_use,
        )

        while not self._is_requested_to_stop():
            self._physics_step(contextlib.nullcontext())
            self.camera_manager.pull_camera_data_at_camera_rate(is_sleep_until_ready=False)

        self.close()

    def __run_headless_simulation_with_physics_thread(
        self, camera_hz: float, cameras_to_use: list[StretchCameras]
    ) -> None:
        """
        Run the simulation without the viewer headless.

        Headless mode manages its own `set_camera_manager()` call.
        """
        print("Running headless simulation...")

        self.set_camera_manager(
            use_camera_thread=False,
            use_threadpool_executor=False,
            camera_hz=camera_hz,
            cameras_to_use=cameras_to_use,
        )

        physics_thread = threading.Thread(
            target=self._physics_loop,
            args=(self.camera_manager.camera_lock, lambda: not self._is_requested_to_stop()),
            daemon=True,
        )
        physics_thread.start()

        while not self._is_requested_to_stop():
            self.camera_manager.pull_camera_data_at_camera_rate(is_sleep_until_ready=True)

        physics_thread.join()
        self.close()

    def _ctrl_callback(self, model: MjModel, data: MjData) -> None:
        """
        Callback function that gets executed with mj_step
        """
        self.mjdata = data
        self.mjmodel = model

        # Update grasp manager references
        self.grasp_manager.mjdata = data
        self.grasp_manager.mjmodel = model

        # Keep contact logger in sync with the same step's data
        self.contact_logger.mjdata = data
        self.contact_logger.mjmodel = model

        if not self.mjdata or not self.mjdata.time:
            print("WARNING: no mujoco data to report")
            return

        self.physics_fps_counter.tick(sim_time=data.time)

        # Update grasps and apply grasp constraints
        self.grasp_manager.update_grasps()
        self.grasp_manager.apply_grasp_constraints()

        # Publish current active contacts to the shared proxy so pull_contacts() can read them.
        self.contact_logger.update()
        self._publish_contacts()

        self.pull_status()
        # Hold the lock across the whole read-modify-write cycle (get_command(),
        # push_command()'s in-place mutation of trigger flags, and its final
        # set_command()). StretchMujocoSimulator's move_to()/move_by()/etc. acquire
        # the same cross-process lock around their own get/set pair; without this,
        # a client dispatch landing between this process's read and write-back gets
        # silently overwritten (its trigger lost) or reasserted (its trigger revived
        # after this process already believed it was cleared) - see the note above
        # BaseController for the base-translation symptom this caused.
        with self._command_lock:
            self.push_command(self.data_proxies.get_command())

    def _publish_contacts(self) -> None:
        """
        Snapshot the currently-active contacts from the ContactLogger and push them
        to the shared proxy so the main process can read them via pull_contacts().
        Only contacts that are active *right now* are included; history is discarded.
        """
        active = list(self.contact_logger._active_contacts.values())
        status = StatusStretchContacts(
            contacts=[
                ContactInfo(
                    sim_time=ev.sim_time,
                    body1_name=ev.body1_name,
                    body2_name=ev.body2_name,
                    geom1_name=ev.geom1_name,
                    geom2_name=ev.geom2_name,
                    normal_force=ev.normal_force,
                    category=ev.category,
                )
                for ev in active
            ],
            sim_time=self.mjdata.time,
        )
        self.data_proxies.set_contacts(status)

    def pull_status(self):
        """
        Pull joints status of the robot from the simulator
        """

        new_status = StatusStretchJoints.default()
        new_status.fps = self.physics_fps_counter.fps

        new_status.time = self.mjdata.time
        new_status.sim_to_real_time_ratio_msg = self.physics_fps_counter.sim_to_real_time_ratio_msg
        new_status.lift.pos = self.mjdata.actuator("lift").length[0]
        new_status.lift.vel = self.mjdata.actuator("lift").velocity[0]

        new_status.arm.pos = self.mjdata.actuator("arm").length[0]
        new_status.arm.vel = self.mjdata.actuator("arm").velocity[0]

        new_status.head_pan.pos = self.mjdata.actuator("head_pan").length[0]
        new_status.head_pan.vel = self.mjdata.actuator("head_pan").velocity[0]

        new_status.head_tilt.pos = self.mjdata.actuator("head_tilt").length[0]
        new_status.head_tilt.vel = self.mjdata.actuator("head_tilt").velocity[0]

        new_status.wrist_yaw.pos = self.mjdata.actuator("wrist_yaw").length[0]
        new_status.wrist_yaw.vel = self.mjdata.actuator("wrist_yaw").velocity[0]

        new_status.wrist_pitch.pos = self.mjdata.actuator("wrist_pitch").length[0]
        new_status.wrist_pitch.vel = self.mjdata.actuator("wrist_pitch").velocity[0]

        new_status.wrist_roll.pos = self.mjdata.actuator("wrist_roll").length[0]
        new_status.wrist_roll.vel = self.mjdata.actuator("wrist_roll").velocity[0]

        new_status.gripper.pos = self._to_real_gripper_range(
            self.mjdata.actuator("gripper").length[0]
        )
        new_status.gripper.vel = self.mjdata.actuator("gripper").velocity[
            0
        ]  # This is still in sim gripper range

        left_wheel_vel = self.mjdata.actuator("left_wheel_vel").velocity[0]
        right_wheel_vel = self.mjdata.actuator("right_wheel_vel").velocity[0]
        (
            new_status.base.x_vel,
            new_status.base.theta_vel,
        ) = utils.diff_drive_fwd_kinematics(left_wheel_vel, right_wheel_vel)
        (
            new_status.base.x,
            new_status.base.y,
            new_status.base.theta,
        ) = self.base_controller.get_base_pose()

        # Get Object positions
        body_names = list(config.REPLACEMENTS.keys())
        object_names = list(config.REPLACEMENTS.values())
        for i, body in enumerate(body_names):
            try:
                xyz = np.round(self.mjdata.body(body).xpos, 3)
                rotation = self.mjdata.body(body).xmat.reshape(3, 3)
                rpy = np.round(utils.rotation_matrix_to_euler(rotation), 3)
                new_status.object_poses[object_names[i]] = {
                    "position": xyz,
                    "rotation": rpy,
                }
            except Exception as e:
                # print(f"Error getting position for object {body}: {e}")
                continue

        self.data_proxies.set_status(new_status)

    def _to_real_gripper_range(self, pos: float) -> float:
        """
        Map the gripper position to real gripper range
        """
        return utils.map_between_ranges(
            pos,
            config.robot_settings["sim_gripper_min_max"],
            config.robot_settings["gripper_min_max"],
        )

    def push_command(self, command_status: StatusCommand):
        """
        Handles setting mujoco ctrl properties to move joints.
        """
        # move_by
        for _, command in command_status.move_by.items():
            if command.trigger:
                command.trigger = False
                actuator_name = command.actuator_name
                pos = command.pos
                if actuator_name in (Actuators.base_translate.name, Actuators.base_rotate.name):
                    self.base_controller.push_command(command)
                else:
                    if actuator_name == Actuators.gripper.name:
                        current_value = self._to_real_gripper_range(
                            self.mjdata.actuator("gripper").length[0]
                        )
                        self.mjdata.actuator(actuator_name).ctrl = self._to_sim_gripper_range(
                            current_value + pos
                        )
                    else:
                        current_value = self.mjdata.actuator(actuator_name).length[0]
                        self.mjdata.actuator(actuator_name).ctrl = current_value + pos

        # move_to
        for _, command in command_status.move_to.items():
            if command.trigger:
                command.trigger = False
                actuator_name = command.actuator_name
                pos = command.pos
                if actuator_name == Actuators.gripper.name:
                    self.mjdata.actuator(actuator_name).ctrl = self._to_sim_gripper_range(pos)
                elif actuator_name in (Actuators.base_translate.name, Actuators.base_rotate.name):
                    raise NotImplementedError(
                        f"Cannot set move_to for {actuator_name}, which is a relative joint."
                    )
                else:
                    self.mjdata.actuator(actuator_name).ctrl = pos

        # set_base_velocity
        if command_status.base_velocity is not None and command_status.base_velocity.trigger:
            command_status.base_velocity.trigger = False
            self.base_controller.push_command(command_status.base_velocity)

        # respawn
        if command_status.respawn is not None and command_status.respawn.trigger:
            command_status.respawn.trigger = False
            for grasped_name in list(self.grasp_manager.grasped_objects.keys()):
                self.grasp_manager.release_object(grasped_name)
            mujoco._functions.mj_resetData(self.mjmodel, self.mjdata)
            self.base_controller.last_command = None

        # teleport
        if command_status.teleport is not None and command_status.teleport.trigger:
            command_status.teleport.trigger = False
            if self._base_link_qadr != -1:
                qadr = self._base_link_qadr
                qdadr = self._base_link_qdadr
                self.mjdata.qpos[qadr : qadr + 3] = command_status.teleport.position
                self.mjdata.qpos[qadr + 3 : qadr + 7] = command_status.teleport.rotation_quat
                self.mjdata.qvel[qdadr : qdadr + 6] = 0
                mujoco._functions.mj_forward(self.mjmodel, self.mjdata)
                self.base_controller.last_command = None

        # teleport_object
        if command_status.teleport_object is not None and command_status.teleport_object.trigger:
            command_status.teleport_object.trigger = False
            raw_name = command_status.teleport_object.object_name
            resolved_name = next(
                (k for k, v in config.REPLACEMENTS.items() if v == raw_name), None
            )
            body_id = mujoco.mj_name2id(self.mjmodel, mujoco.mjtObj.mjOBJ_BODY, resolved_name)
            if body_id < 0:
                click.secho(
                    f"teleport_object: body '{resolved_name}' not found in model.",
                    fg="red",
                )
            else:
                jnt_adr = self.mjmodel.body_jntadr[body_id]
                if jnt_adr < 0 or self.mjmodel.jnt_type[jnt_adr] != mujoco.mjtJoint.mjJNT_FREE:
                    click.secho(
                        f"teleport_object: body '{resolved_name}' has no free joint; cannot teleport.",
                        fg="red",
                    )
                else:
                    qadr = int(self.mjmodel.jnt_qposadr[jnt_adr])
                    qdadr = int(self.mjmodel.jnt_dofadr[jnt_adr])
                    self.mjdata.qpos[qadr : qadr + 3] = command_status.teleport_object.position
                    self.mjdata.qpos[qadr + 3 : qadr + 7] = (
                        command_status.teleport_object.rotation_quat
                    )
                    self.mjdata.qvel[qdadr : qdadr + 6] = 0
                    mujoco._functions.mj_forward(self.mjmodel, self.mjdata)

        # grasp_object: teleport object to gripper and force-attach it
        if command_status.grasp_object is not None and command_status.grasp_object.trigger:
            command_status.grasp_object.trigger = False
            raw_name = command_status.grasp_object.object_name
            resolved_name = next(
                (k for k, v in config.REPLACEMENTS.items() if v == raw_name), raw_name
            )
            body_id = mujoco.mj_name2id(self.mjmodel, mujoco.mjtObj.mjOBJ_BODY, resolved_name)
            if body_id < 0:
                click.secho(
                    f"grasp_object: body '{resolved_name}' not found in model.",
                    fg="red",
                )
            else:
                jnt_adr = self.mjmodel.body_jntadr[body_id]
                if jnt_adr >= 0 and self.mjmodel.jnt_type[jnt_adr] == mujoco.mjtJoint.mjJNT_FREE:
                    qadr = int(self.mjmodel.jnt_qposadr[jnt_adr])
                    qdadr = int(self.mjmodel.jnt_dofadr[jnt_adr])
                    grasp_pos = self.mjdata.body("link_grasp_center").xpos.copy()
                    self.mjdata.qpos[qadr : qadr + 3] = grasp_pos
                    self.mjdata.qvel[qdadr : qdadr + 6] = 0
                    mujoco._functions.mj_forward(self.mjmodel, self.mjdata)
                self.grasp_manager.grasp_object(resolved_name)

        # keyframe
        if command_status.keyframe is not None and command_status.keyframe.trigger:
            command_status.keyframe.trigger = False
            self.mjdata.ctrl = self.mjmodel.keyframe(command_status.keyframe.name).ctrl

        self.base_controller.update()

        self.data_proxies.set_command(command_status)

    def _to_sim_gripper_range(self, pos: float) -> float:
        """
        Map the gripper position to sim gripper range
        """
        return utils.map_between_ranges(
            pos,
            config.robot_settings["gripper_min_max"],
            config.robot_settings["sim_gripper_min_max"],
        )
