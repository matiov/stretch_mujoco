import time
from stretch_mujoco.stretch_mujoco_simulator import StretchMujocoSimulator
from stretch_mujoco.enums.actuators import Actuators

# --- CONFIGURATION ---
# Adjust these values as needed for your scene/object
ARM_ALIGN_POS = 0.20  # meters, arm extension to align with object
LIFT_ALIGN_POS = 0.42  # meters, lift height to align with object
GRIPPER_OPEN_POS = 0.25  # open gripper (positive value)
GRIPPER_CLOSE_POS = 0.01  # close gripper (negative value)
LIFT_LIFTED_POS = 0.60  # lift after grasp


def main():
    # Start simulator with viewer only (no cameras)
    sim = StretchMujocoSimulator()
    sim.start(headless=False)

    input("Press Enter to attempt the grasp...")


    # 1. Open gripper
    print("[STEP 1] Opening gripper...")
    sim.move_to(Actuators.gripper, GRIPPER_OPEN_POS)
    sim.wait_until_at_setpoint(Actuators.gripper)
    input("Press Enter to continue to the next step...")

    # 2. Move arm/lift to align gripper with object
    print("[STEP 2] Aligning end effector with object...")
    sim.move_to(Actuators.arm, ARM_ALIGN_POS)
    sim.move_to(Actuators.lift, LIFT_ALIGN_POS)
    sim.wait_until_at_setpoint(Actuators.arm)
    sim.wait_until_at_setpoint(Actuators.lift)
    input("Press Enter to continue to the next step...")

    # 3. Close gripper
    print("[STEP 3] Closing gripper...")
    sim.move_to(Actuators.gripper, GRIPPER_CLOSE_POS)
    sim.wait_until_at_setpoint(Actuators.gripper)
    input("Press Enter to continue to the next step...")

    # 4. Lift arm
    print("[STEP 4] Lifting arm to test grasp...")
    sim.move_to(Actuators.lift, LIFT_LIFTED_POS)
    sim.wait_until_at_setpoint(Actuators.lift)
    print("[STEP 5] Done. Check if object is held.")
    input("Press Enter to continue to the next step...")

    # 6. Stowing arm
    print("[STEP 6] Stowing the arm")
    sim.stow()
    input("Press Enter to continue to the next step...")

    # 7. Releasing object
    print("[STEP 7] Releasing object. The object should fall now...")
    sim.move_to(Actuators.gripper, GRIPPER_OPEN_POS)
    sim.wait_until_at_setpoint(Actuators.gripper)

    # Shutting Down
    input("Press Enter to continue to the next step...")
    print("[END] We're done here! Bye Bye!")
    sim.stop()


if __name__ == "__main__":
    main()
