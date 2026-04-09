import threading
import time
import os
import re
import shutil
import subprocess
from stretch_mujoco.datamodels.status_command import StatusCommand
from stretch_mujoco.utils import Rx, Ry, Rz, override
import numpy as np

import click
import mujoco
import mujoco._functions
import mujoco.viewer
from mujoco._enums import mjtGeom
import stretch_mujoco.config as config
from stretch_mujoco.enums.stretch_cameras import StretchCameras
from stretch_mujoco.mujoco_server import MujocoServer
from stretch_mujoco.utils import FpsCounter

import mujoco._enums


class MujocoServerPassive(MujocoServer):
    """
    A MujocoServer flavor that uses the mujoco passive viewer.

    Use `MujocoServerPassive.launch_server()` to start the simulator.

    To render offscreen cameras, please call `set_camera_manager(False,,)` and then `camera_manager.pull_camera_data_at_camera_rate()` in the UI thread.

    On MacOS, this needs to be started with `mjpython`. If you're using StretchMujocoSimulator.start(), this is automatically handled.

    https://mujoco.readthedocs.io/en/stable/python.html#passive-viewer
    """

    @override
    def run(
        self,
        show_viewer_ui: bool,
        camera_hz: float,
        cameras_to_use: list[StretchCameras],
    ):
        # We're using the passive viewer, and have access to the UI thread. We can manage camera rendering on the UI thread:
        self.set_camera_manager(
            use_camera_thread=False,
            use_threadpool_executor=False,
            camera_hz=camera_hz,
            cameras_to_use=cameras_to_use,
        )

        self._run_ui_simulation(show_viewer_ui)

    @override
    def _run_ui_simulation(self, show_viewer_ui: bool):
        """
        Starts the mujoco viewer in Passive Mode. Also starts a physics thread for stepping the simulation.

        On MacOS, this needs to be started with `mjpython`.

        https://mujoco.readthedocs.io/en/stable/python.html#passive-viewer
        """
        self.viewer = mujoco.viewer.launch_passive(
            model=self.mjmodel,
            data=self.mjdata,
            show_left_ui=show_viewer_ui,
            show_right_ui=show_viewer_ui,
        )

        self._window_layout_applied = False
        self._window_layout_attempts = 0
        self._window_layout_unavailable_logged = False

        # Set a predictable startup camera so the initial view is not inside nearby geometry.
        self._set_startup_viewer_camera()

        self._last_printed_camera_pose: np.ndarray | None = None

        self.viewer._opt.flags[mujoco._enums.mjtVisFlag.mjVIS_RANGEFINDER] = (
            False  # Disables the lidar yellow lines.
        )

        with self.viewer as viewer:
            physics_thread = threading.Thread(
                target=self._physics_loop,
                name="PhysicsThread",
                args=(
                    viewer.lock(),
                    lambda: viewer.is_running() and not self._is_requested_to_stop(),
                ),
                daemon=True,
            )
            physics_thread.start()

            fps = FpsCounter()

            UI_FPS_CAP_RATE = (
                self.camera_manager.camera_rate
            )  # 1/Hz.Put the UI thread to sleep so that the physics thread can do work, to mitigate `viewer.lock()` locking physics thread.

            click.secho(
                f"Using the Mujoco Passive Viewer. Note: UI thread and camera rendering is capped to {1/UI_FPS_CAP_RATE}Hz to increase performance. You can set this rate using the `camera_rate` arugment.",
                fg="green",
            )

            # Replace the camera_lock with the viewer lock so that we're not accessing mjdata at the same time as the physics thread.
            self.camera_manager.camera_lock = viewer.lock()  # type: ignore

            while viewer.is_running() and not self._is_requested_to_stop():
                fps.tick()
                start_time = time.perf_counter()
                # print(f"UI thread: {fps.fps=}, {self.physics_fps_counter.fps=}, {self.camera_manager.camera_fps_counter.fps=}")

                self.camera_manager.pull_camera_data_at_camera_rate(is_sleep_until_ready=False)

                self._snap_window_if_configured()

                self._print_camera_pose_if_changed()

                viewer.sync()

                time_until_next_ui_update = UI_FPS_CAP_RATE - (time.perf_counter() - start_time)
                if time_until_next_ui_update > 0:
                    # Put the UI thread to sleep so that the physics thread can do work, to mitigate `viewer.lock()` taking up ticks.
                    time.sleep(time_until_next_ui_update)
                else:
                    click.secho(
                        f"WARNING: Passive viewer and camera rendering is below the requested {1/self.camera_manager.camera_rate}FPS on the last render.",
                        fg="yellow",
                    )

            self.close()

            # Wait for any active threads to close, otherwise the mujoco window gets stuck:
            active_threads = threading.enumerate()
            for index, thread in enumerate(active_threads):
                if thread != threading.main_thread() and not isinstance(
                    thread, threading._DummyThread
                ):
                    click.secho(
                        f"Stopping thread {index}/{len(active_threads)-1} on the Mujoco Process.",
                        fg="blue",
                    )
                    thread.join(timeout=5.0)

            click.secho("Mujoco viewer has terminated.", fg="blue")

    def _set_startup_viewer_camera(self) -> None:
        """Set the passive viewer free-camera pose from config."""
        camera_cfg = config.viewer_camera
        cam = self.viewer.cam

        cam.lookat[:] = camera_cfg["lookat"]
        cam.distance = camera_cfg["distance"]
        cam.azimuth = camera_cfg["azimuth"]
        cam.elevation = camera_cfg["elevation"]

    def _print_camera_pose_if_changed(self) -> None:
        """Print free-camera values when the user moves the camera."""
        debug_cfg = config.viewer_camera_debug
        if not debug_cfg["enabled"]:
            return

        cam = self.viewer.cam
        pose = np.array(
            [
                cam.lookat[0],
                cam.lookat[1],
                cam.lookat[2],
                cam.distance,
                cam.azimuth,
                cam.elevation,
            ],
            dtype=float,
        )

        if self._last_printed_camera_pose is None:
            self._last_printed_camera_pose = pose
            self._print_camera_pose(pose)
            return

        position_delta = np.max(np.abs(pose[:4] - self._last_printed_camera_pose[:4]))
        angle_delta = np.max(np.abs(pose[4:] - self._last_printed_camera_pose[4:]))
        if position_delta < float(debug_cfg["position_epsilon"]) and angle_delta < float(
            debug_cfg["angle_epsilon"]
        ):
            return

        self._last_printed_camera_pose = pose
        self._print_camera_pose(pose)

    def _print_camera_pose(self, pose: np.ndarray) -> None:
        click.secho(
            "camera="
            + "{"
            + f'"lookat": [{pose[0]:.3f}, {pose[1]:.3f}, {pose[2]:.3f}], '
            + f'"distance": {pose[3]:.3f}, '
            + f'"azimuth": {pose[4]:.3f}, '
            + f'"elevation": {pose[5]:.3f}'
            + "}",
            fg="cyan",
        )

    def _snap_window_if_configured(self) -> None:
        """Best-effort X11 placement to mimic split-screen layouts."""
        window_cfg = config.viewer_window
        if not window_cfg["enabled"]:
            return

        # In containers/headless sessions there is often no desktop window server.
        if not os.environ.get("DISPLAY"):
            if not self._window_layout_unavailable_logged:
                click.secho(
                    "No DISPLAY detected; skipping viewer window placement.",
                    fg="yellow",
                )
                self._window_layout_unavailable_logged = True
            self._window_layout_applied = True
            return

        if self._window_layout_applied:
            return

        max_attempts = int(window_cfg.get("max_attempts", 60))
        if self._window_layout_attempts >= max_attempts:
            self._window_layout_applied = True
            click.secho(
                "Could not place viewer window. Install xdotool or wmctrl for X11 sessions.",
                fg="yellow",
            )
            return

        self._window_layout_attempts += 1
        layout = window_cfg["layout"]

        if layout not in ("left-half", "right-half"):
            click.secho(f"Unknown viewer_window layout '{layout}', skipping.", fg="yellow")
            self._window_layout_applied = True
            return

        if self._snap_window_with_external_tools(layout=layout):
            self._window_layout_applied = True
            click.secho(
                f"Viewer window layout applied ({layout}).",
                fg="green",
            )

    def _snap_window_with_external_tools(self, layout: str) -> bool:
        """Try X11 window tools to place the viewer in a half-screen layout."""
        if layout not in ("left-half", "right-half"):
            return False

        if shutil.which("xdotool") is not None:
            try:
                search = subprocess.run(
                    ["xdotool", "search", "--onlyvisible", "--pid", str(os.getpid())],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                window_ids = [line.strip() for line in search.stdout.splitlines() if line.strip()]
                if window_ids:
                    window_id = window_ids[-1]
                    geometry = subprocess.run(
                        ["xdotool", "getdisplaygeometry"],
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                    cols = geometry.stdout.strip().split()
                    if len(cols) != 2:
                        return False

                    screen_width = int(cols[0])
                    screen_height = int(cols[1])
                    half_width = max(1, screen_width // 2)
                    target_x = 0 if layout == "left-half" else half_width

                    subprocess.run(
                        [
                            "xdotool",
                            "windowstate",
                            "--remove",
                            "MAXIMIZED_VERT",
                            "--remove",
                            "MAXIMIZED_HORZ",
                            window_id,
                        ],
                        check=False,
                        capture_output=True,
                        text=True,
                    )
                    subprocess.run(
                        ["xdotool", "windowmove", "--sync", window_id, str(target_x), "0"],
                        check=False,
                        capture_output=True,
                        text=True,
                    )
                    subprocess.run(
                        [
                            "xdotool",
                            "windowsize",
                            "--sync",
                            window_id,
                            str(half_width),
                            str(screen_height),
                        ],
                        check=False,
                        capture_output=True,
                        text=True,
                    )
                    return True
            except (subprocess.CalledProcessError, ValueError):
                ...

        if shutil.which("wmctrl") is not None:
            try:
                listed = subprocess.run(
                    ["wmctrl", "-lp"],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                window_id = None
                pid = str(os.getpid())
                for line in listed.stdout.splitlines():
                    cols = line.split(None, 4)
                    if len(cols) >= 3 and cols[2] == pid:
                        window_id = cols[0]

                if window_id is not None:
                    desktop = subprocess.run(
                        ["wmctrl", "-d"],
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                    screen_width = None
                    screen_height = None
                    for line in desktop.stdout.splitlines():
                        match = re.search(r"DG:\s*(\d+)x(\d+)", line)
                        if match is not None:
                            screen_width = int(match.group(1))
                            screen_height = int(match.group(2))
                            break

                    if screen_width is None or screen_height is None:
                        return False

                    half_width = max(1, screen_width // 2)
                    target_x = 0 if layout == "left-half" else half_width

                    subprocess.run(
                        ["wmctrl", "-ir", window_id, "-b", "remove,maximized_vert,maximized_horz"],
                        check=False,
                        capture_output=True,
                        text=True,
                    )
                    subprocess.run(
                        [
                            "wmctrl",
                            "-ir",
                            window_id,
                            "-e",
                            f"0,{target_x},0,{half_width},{screen_height}",
                        ],
                        check=False,
                        capture_output=True,
                        text=True,
                    )
                    return True
            except (subprocess.CalledProcessError, ValueError):
                ...

        return False

    def push_command(self, command_status: StatusCommand):

        command_arrows = command_status.coordinate_frame_arrows_viz.copy()

        for arrows in command_arrows:
            if arrows.trigger:
                self._add_axes_to_user_scn(
                    self.viewer.user_scn, np.array(arrows.position), arrows.rotation
                )

                command_status.coordinate_frame_arrows_viz.remove(arrows)

        super().push_command(command_status)

    @override
    def _add_axes_to_user_scn(
        self,
        user_scn,
        origin: np.ndarray,
        rotation: tuple[float, float, float],
        length: float = 0.2,
        radius: float = 0.006,
    ):
        """
        Draw a right-handed RGB frame in `user_scn` using mjv_initGeom.

        * +X red, +Y green, +Z blue
        * `origin` 3-vector in world frame
        * `R`      3×3 rotation matrix, columns are local x,y,z in world frame
        """
        colors = np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]])  # +X  # +Y  # +Z

        rot_matrix = Rx(rotation[0]) @ Ry(rotation[1]) @ Rz(rotation[2])
        for axis in range(3):
            if axis == 0:
                # Rotate +Z to +X: -90° about Y-axis
                R = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
            elif axis == 1:
                # Rotate +Z to +Y: -90° about X-axis
                R = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]])
            elif axis == 2:
                # No rotation needed
                R = np.eye(3)

            R = rot_matrix @ R

            size = [radius, radius, length]

            geom = user_scn.geoms[user_scn.ngeom]
            mujoco._functions.mjv_initGeom(
                geom,
                type=mjtGeom.mjGEOM_ARROW,
                size=size,
                pos=origin,
                mat=np.array(R).flatten(),
                rgba=colors[axis],
            )
            user_scn.ngeom += 1
