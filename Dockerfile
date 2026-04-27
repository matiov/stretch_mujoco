# Python 3.10 is required: robocasa pins numpy==1.23.3 and numba==0.56.4,
# which only have prebuilt wheels for Python 3.10.
FROM python:3.10-slim

# System dependencies for MuJoCo renderer, OpenCV, and X11 display forwarding.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    libgl1 \
    libgl1-mesa-dri \
    libegl1 \
    libgles2 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libx11-6 \
    libxrandr2 \
    libxinerama1 \
    libxcursor1 \
    libxi6 \
    libxfixes3 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project files.
COPY . .

# Upgrade pip toolchain first.
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install base package.
RUN pip install --no-cache-dir -e .

# Install robocasa extras (numpy/numba/opencv pinned versions).
RUN pip install --no-cache-dir -e ".[robocasa]"

# Install third-party submodules.
RUN pip install --no-cache-dir -e third_party/robosuite
RUN pip install --no-cache-dir -e third_party/robocasa

# Create private macro files non-interactively for Docker builds.
RUN cp -f third_party/robosuite/robosuite/macros.py third_party/robosuite/robosuite/macros_private.py
RUN cp -f third_party/robocasa/robocasa/macros.py third_party/robocasa/robocasa/macros_private.py

# RoboCasa assets are intentionally not downloaded at build time to keep image
# size/build time small for quick testing.

# Default command: print how to start the quick non-RoboCasa test.
CMD ["python", "-c", "import stretch_mujoco; print('stretch_mujoco ready. Quick test: python examples/keyboard_teleop.py --imagery-nav. RoboCasa assets: docker compose run --rm stretch-mujoco-assets')"]
