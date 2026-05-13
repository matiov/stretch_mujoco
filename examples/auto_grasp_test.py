import time
import random
from stretch_mujoco.stretch_mujoco_simulator import StretchMujocoSimulator
from stretch_mujoco.enums.actuators import Actuators

# --- CONFIGURATION ---
# Adjust these values as needed for your scene/object
ARM_ALIGN_POS = 0.20  # meters, arm extension to align with object
LIFT_ALIGN_POS = 0.42  # meters, lift height to align with object
GRIPPER_OPEN_POS = 0.25  # open gripper (positive value)
GRIPPER_CLOSE_POS = 0.01  # close gripper (negative value)
LIFT_LIFTED_POS = 0.60  # lift after grasp
WRIST_PITCH_NEUTRAL_POS = 0.0
WRIST_ROLL_NEUTRAL_POS = 0.0
WRIST_YAW_NEUTRAL_POS = 0.0

SLIP_TEST_REPETITIONS = 5
SLIP_TEST_WRIST_PITCH_DELTA = 0.25
SLIP_TEST_WRIST_ROLL_DELTA = 0.5
SLIP_TEST_WRIST_YAW_DELTA = 0.35


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

    # 6. Randomized EE slip test
    print(f"[STEP 6] Running {SLIP_TEST_REPETITIONS} random wrist motions to test for slipping...")
    for repetition in range(1, SLIP_TEST_REPETITIONS + 1):
        target_wrist_pitch = WRIST_PITCH_NEUTRAL_POS + random.uniform(
            -SLIP_TEST_WRIST_PITCH_DELTA, SLIP_TEST_WRIST_PITCH_DELTA
        )
        target_wrist_roll = WRIST_ROLL_NEUTRAL_POS + random.uniform(
            -SLIP_TEST_WRIST_ROLL_DELTA, SLIP_TEST_WRIST_ROLL_DELTA
        )
        target_wrist_yaw = WRIST_YAW_NEUTRAL_POS + random.uniform(
            -SLIP_TEST_WRIST_YAW_DELTA, SLIP_TEST_WRIST_YAW_DELTA
        )

        print(
            f"  - Motion {repetition}/{SLIP_TEST_REPETITIONS}: "
            f"wrist_pitch={target_wrist_pitch:.3f}, "
            f"wrist_roll={target_wrist_roll:.3f}, wrist_yaw={target_wrist_yaw:.3f}"
        )
        sim.move_to(Actuators.wrist_pitch, target_wrist_pitch)
        sim.move_to(Actuators.wrist_roll, target_wrist_roll)
        sim.move_to(Actuators.wrist_yaw, target_wrist_yaw)
        sim.wait_until_at_setpoint(Actuators.wrist_pitch)
        sim.wait_until_at_setpoint(Actuators.wrist_roll)
        sim.wait_until_at_setpoint(Actuators.wrist_yaw)
        time.sleep(0.5)

    print("[STEP 7] Random motion test complete. Check if object is still held.")
    input("Press Enter to continue to the next step...")

    # 8. Stowing arm
    print("[STEP 8] Stowing the arm")
    sim.stow()
    input("Press Enter to continue to the next step...")

    # 9. Releasing object
    print("[STEP 9] Releasing object. The object should fall now...")
    sim.move_to(Actuators.gripper, GRIPPER_OPEN_POS)
    sim.wait_until_at_setpoint(Actuators.gripper)

    # Shutting Down
    input("Press Enter to continue to the next step...")
    print("[END] We're done here! Bye Bye!")
    sim.stop()


if __name__ == "__main__":
    main()
