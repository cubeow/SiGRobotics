import time
from lerobot.teleoperators.omx_leader import OmxLeader, OmxLeaderConfig
from lerobot.robots.omx_follower import OmxFollower, OmxFollowerConfig
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.cameras import ColorMode, Cv2Rotation
from lerobot.utils.visualization_utils import (
    init_visualization,
    log_visualization_data,
    shutdown_visualization,
)

# 1. Configure the Follower Arm and Cameras
# Note: Check outputs/captured_images to assign the right index to your arm camera.
robot_config = OmxFollowerConfig(
    port="/dev/tty.usbmodem101",  # Replace with follower U2D2 port
    id="omx_follower",
    cameras={
        # U20CAM-720P on the arm (usually index 1 if 0 is your Mac's built-in webcam)
        # "wrist": OpenCVCameraConfig(
        #     index_or_path=0,
        #     width=1280,
        #     height=720,
        #     fps=30,
        #     color_mode=ColorMode.RGB,
        #     rotation=Cv2Rotation.NO_ROTATION,
        # ),
        # Optional: Mac built-in webcam or secondary overhead camera
        "top": OpenCVCameraConfig(
            index_or_path=0,
            width=1280,
            height=720,
            fps=30,
            color_mode=ColorMode.RGB,
            rotation=Cv2Rotation.NO_ROTATION,
        ),
    },
)

# 2. Configure the Leader Arm
teleop_config = OmxLeaderConfig(
    port="/dev/tty.usbmodem1101",  # Replace with leader U2D2 port
    id="omx_leader",
)

# 3. Initialize Visualization
init_visualization("rerun", session_name="omx_teleoperation")

# 4. Instantiate and Connect Devices
robot = OmxFollower(robot_config)
teleop_device = OmxLeader(teleop_config)

try:
    print("Connecting robot and teleoperator...")
    robot.connect()
    teleop_device.connect()
    print("Connected successfully! Starting teleoperation loop...")

    TARGET_HZ = 30
    TIME_PER_FRAME = 1.0 / TARGET_HZ

    while True:
        start_time = time.perf_counter()

        # Current state of the follower - joint positions and camera frames
        observation = robot.get_observation()

        # Read leader joint positions and send to follower
        action = teleop_device.get_action()
        robot.send_action(action)

        # Log observation (including camera feeds) and actions to Rerun visualizer
        log_visualization_data("rerun", observation=observation, action=action)

        # Enforce loop frequency
        elapsed_time = time.perf_counter() - start_time
        sleep_time = TIME_PER_FRAME - elapsed_time
        if sleep_time > 0:
            time.sleep(sleep_time)

except KeyboardInterrupt:
    print("\nStopping teleoperation...")
finally:
    robot.disconnect()
    teleop_device.disconnect()
    shutdown_visualization("rerun")
    print("Disconnected safely.")