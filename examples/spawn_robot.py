"""Simple script to visualize the Stretch Robot."""

import importlib.resources
from stretch_mujoco.stretch_mujoco_simulator import StretchMujocoSimulator

models_path = str(importlib.resources.files("stretch_mujoco") / "models")
scene_xml_path = models_path + "/scene_clean.xml"


def main():
    # Start simulator with viewer only (no cameras)
    sim = StretchMujocoSimulator(scene_xml_path=scene_xml_path)
    sim.start(headless=False)

    input("Press Enter to close the simulation...")
    sim.stop()


if __name__ == "__main__":
    main()
