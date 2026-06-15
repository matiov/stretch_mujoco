"""
Monitor and log all contacts between the robot and other objects in the scene.

Useful for checking whether the robot is colliding with furniture, its own arm,
or grasped objects during navigation or manipulation.

Run:
    python examples/contact_monitor.py
    python examples/contact_monitor.py --headless
    python examples/contact_monitor.py --log contacts.csv
    python examples/contact_monitor.py --min-force 1.0  # only significant impacts

Controls (interactive mode):
    s  — print a contact summary to the terminal
    q  — quit
"""

import importlib.resources
import signal
import sys
import threading
import time

import click

from stretch_mujoco.stretch_mujoco_simulator import StretchMujocoSimulator
from stretch_mujoco.contact_logger import ContactLogger


@click.command()
@click.option("--headless", is_flag=True, default=False, help="Run without viewer window.")
@click.option(
    "--log",
    "log_file",
    default=None,
    metavar="FILE",
    help="CSV file to append contact events to (e.g. contacts.csv).",
)
@click.option(
    "--min-force",
    "min_force",
    default=0.0,
    type=float,
    show_default=True,
    help="Minimum estimated normal force (N) to log.",
)
@click.option(
    "--no-self-contacts",
    "no_self",
    is_flag=True,
    default=False,
    help="Suppress robot-vs-robot contacts (e.g. arm touching base).",
)
@click.option(
    "--all-contacts",
    "all_contacts",
    is_flag=True,
    default=False,
    help="Log every contact in the scene, not just robot contacts.",
)
@click.option(
    "--poll-hz",
    "poll_hz",
    default=50,
    type=float,
    show_default=True,
    help="How many times per second to poll sim contacts.",
)
def main(
    headless: bool,
    log_file: str | None,
    min_force: float,
    no_self: bool,
    all_contacts: bool,
    poll_hz: float,
):
    models_path = str(importlib.resources.files("stretch_mujoco") / "models")
    scene_xml_path = models_path + "/scene_clean.xml"

    sim = StretchMujocoSimulator(scene_xml_path=scene_xml_path)
    sim.start(headless=headless)

    # Wait until the simulation is running
    while sim.is_running() and sim.pull_status().time == 0:
        time.sleep(0.05)

    click.echo(
        click.style(
            f"\nContact monitor started "
            f"(min_force={min_force:.2f} N, poll_hz={poll_hz}, "
            f"log={'stdout' if log_file is None else log_file}).\n"
            f"Press  s  to print summary,  q  to quit.\n",
            fg="cyan",
        )
    )

    # The simulator runs the physics in a separate process; we cannot access
    # mjmodel/mjdata directly from here.  Instead we use the low-level server
    # references that StretchMujocoSimulator exposes when running in the same
    # process (passive / managed viewer).  If the server is in a separate
    # process, we fall back to a polling approach that reads contact counts
    # from the shared status object.
    #
    # For full contact details we need direct mjmodel/mjdata access.  The
    # simplest way to guarantee that is to reach into the server object that
    # lives in the same process when `sim._server` is available.
    logger: ContactLogger | None = None
    server = getattr(sim, "_server", None)

    if server is not None and hasattr(server, "mjmodel") and hasattr(server, "mjdata"):
        logger = ContactLogger(
            mjmodel=server.mjmodel,
            mjdata=server.mjdata,
            log_file=log_file,
            min_force=min_force,
            robot_only=not all_contacts,
            log_robot_self_contacts=not no_self,
            verbose=True,
        )
        click.echo(
            click.style(
                "Direct mjmodel/mjdata access: full contact details available.", fg="green"
            )
        )
    else:
        click.echo(
            click.style(
                "NOTE: simulator is running in a separate process.  "
                "Full contact details require single-process mode (see test_one_process_passive.py).\n"
                "Falling back to contact-count monitoring via pull_status().",
                fg="yellow",
            )
        )

    stop_event = threading.Event()

    def _keyboard_listener():
        while not stop_event.is_set():
            try:
                ch = click.getchar()
            except Exception:
                break
            if ch in ("q", "Q"):
                stop_event.set()
            elif ch in ("s", "S") and logger is not None:
                logger.print_summary()

    keyboard_thread = threading.Thread(target=_keyboard_listener, daemon=True)
    keyboard_thread.start()

    interval = 1.0 / max(poll_hz, 1.0)
    last_status_time = 0.0

    try:
        while sim.is_running() and not stop_event.is_set():
            if logger is not None:
                # Update mjdata reference in case the server swapped it
                if server is not None and hasattr(server, "mjdata"):
                    logger.mjdata = server.mjdata
                    logger.mjmodel = server.mjmodel
                logger.update()
            else:
                # Fallback: just report changes in the number of contacts via status
                status = sim.pull_status()
                if status.time != last_status_time:
                    last_status_time = status.time

            time.sleep(interval)

    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        if logger is not None:
            logger.print_summary()
            logger.close()
        sim.stop()


if __name__ == "__main__":
    main()
