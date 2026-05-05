# TestGrasp README

This guide walks you from installation to validating automatic grasp stabilization using Docker.

Goal of the test:
- Close the gripper on an object.
- Confirm the object stays attached while moving.
- Open the gripper and confirm the object releases.
- No explicit grasp trigger from your pick policy should be required.

## 1. Prerequisites

You need:
- Linux with Docker and Docker Compose installed.
- Git.
- A working X11 display (for the MuJoCo viewer window).

Notes:
- The Docker image uses Python 3.10. This satisfies all pinned dependencies (`numpy==1.23.3`, `numba==0.56.4`) without any manual version management.
- The image is intentionally lightweight: RoboCasa kitchen assets are not downloaded during build.

## 2. Clone the repository

```bash
git clone https://github.com/hello-robot/stretch_mujoco --recurse-submodules
cd stretch_mujoco
```

If you already cloned without submodules:

```bash
git submodule update --init --recursive
```

## 3. Allow Docker to open windows on your display

Run this once per host login session:

```bash
xhost +local:docker
```

## 4. Build the Docker image

```bash
docker compose build
```

This will:
- Use Python 3.10 slim as the base.
- Install all pinned dependencies (numpy, numba, mujoco, opencv, robocasa, robosuite).
- Create required `macros_private.py` files non-interactively.

The build is optimized for quick testing and avoids large asset downloads.

## 5. Verify installation

```bash
docker compose run --rm stretch-mujoco python -c "import stretch_mujoco; print('stretch_mujoco import OK')"
```

Optional: verify automatic grasp hooks exist in server loop:

```bash
docker compose run --rm stretch-mujoco grep -n "update_grasps\|apply_grasp_constraints" stretch_mujoco/mujoco_server.py
```

Optional: visualize the robot:
```bash
docker compose run --rm stretch-mujoco python examples/spanw_robot.py
```


Expected: both calls appear in the control callback.

## 6. Run the quick grasp test (recommended)

### Run the automated grasp test script

To run the automated grasp test (no keyboard required), use:

```bash
docker compose run --rm stretch-mujoco python examples/auto_grasp_test.py
```

This will:
- Automatically align the gripper with an object
- Open and close the gripper to grasp
- Lift the arm to check grasp stability
- Keep the MuJoCo viewer open for observation

You can adjust the script parameters in `examples/auto_grasp_test.py` if needed.

### Start the simulation

```bash
docker compose up stretch-mujoco
```

This runs `python examples/keyboard_teleop.py --imagery-nav` and opens the MuJoCo viewer on your host display.



**Keyboard Controls:**

```text
W / A / S / D      Move BASE
T / F / G / H      Move HEAD
I / J / K / L      Move LIFT & ARM
O / P              Move WRIST YAW
C / V              Move WRIST PITCH
E / R              Move WRIST ROLL
N / M              Open & Close GRIPPER
ctrl+shift+)       Enable keyboard input
ctrl+shift+(       Disable keyboard input
Z                  Print status
Q                  Stop
```


### Test procedure

1. Open the gripper using N.
2. Move the robot and arm so the gripper encloses a target object.
3. Close gripper with M (press repeatedly as needed).
4. Lift and move the arm/base while keeping gripper closed.
5. Open gripper with N.

### Pass criteria

- While closed and in contact: object follows the end effector.
- While moving around: object stays attached and does not immediately fall.
- After opening: object detaches and falls/releases naturally.
- No policy-side grasp trigger is needed.

## 7. Optional RoboCasa test (requires assets download)

If you need coffee_cup / RoboCasa testing, download assets once:

```bash
docker compose run --rm stretch-mujoco-assets
```

Then run RoboCasa teleop:

```bash
docker compose run --rm stretch-mujoco python examples/keyboard_teleop.py --robocasa-env --imagery-nav
```

If you only need a minimal sanity test, run:

```bash
docker compose run --rm stretch-mujoco python examples/keyboard_teleop.py --imagery-nav
```

Default scene contains basic objects. The same pass criteria apply.

## 8. Troubleshooting

### No viewer window

Make sure you ran:

```bash
xhost +local:docker
```

Also ensure `DISPLAY` is set on your host:

```bash
echo $DISPLAY   # should print something like :0 or :1
```

### Keyboard does not control robot

- Click the viewer window first to give it focus.

### RoboCasa assets missing

Assets are not part of the default image build. Download them on demand:

```bash
docker compose run --rm stretch-mujoco-assets
```

### Object still slips or falls
- Ensure the gripper is fully closed and object is centered between fingers.
- Try slower lift and arm motions to avoid large transients.
- Confirm the tested object is considered graspable by current heuristics.

## 9. What to record when reporting results

Please capture:
- Exact command used to run the test.
- Whether pass criteria succeeded.
- Any error logs from terminal.
- If possible, a short screen recording.

That is enough to quickly confirm whether automatic grasp stabilization is behaving correctly in your environment.
