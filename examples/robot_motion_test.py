
from stretch_mujoco.stretch_mujoco_simulator import StretchMujocoSimulator
from stretch_mujoco.enums.actuators import Actuators

# --- CONFIGURATION ---
# Adjust these values as needed for your scene/object
GRIPPER_OPEN_POS = 0.3  # open gripper (positive value)
GRIPPER_CLOSE_POS = -0.15  # close gripper (negative value)


def main():
    # Start simulator with viewer only (no cameras)
    sim = StretchMujocoSimulator()
    sim.start(headless=False)

    input("Press Enter to attempt the grasp...")

    for _ in range(3):
        # 1. Open gripper
        print("[STEP 1] Opening gripper...")
        sim.move_to(Actuators.gripper, GRIPPER_OPEN_POS)
        sim.wait_until_at_setpoint(Actuators.gripper)
        input("Press Enter to continue to the next step...")

        # 2. Close gripper
        print("[STEP 2] Closing gripper...")
        sim.move_to(Actuators.gripper, GRIPPER_CLOSE_POS)
        sim.wait_until_at_setpoint(Actuators.gripper)
        input("Press Enter to continue to the next step...")

    for _ in range(3):
        # 3. Stowing the arm
        print("[STEP 3] Stowing the arm")
        sim.stow()
        input("Press Enter to continue to the next step...")

        # 4. Homing the arm
        print("[STEP 4] Homing the arm")
        sim.home()
        input("Press Enter to continue to the next step...")


    print("[END] We're done here! Bye Bye!")
    sim.stop()


if __name__ == "__main__":
    main()
