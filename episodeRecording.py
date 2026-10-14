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
from lerobot.utils.feature_utils import hw_to_dataset_features
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.processor import make_default_processors
from lerobot.utils.utils import log_say
from lerobot.scripts.lerobot_record import record_loop
from pynput import keyboard
import shutil
from pathlib import Path

# Add this right before `dataset = LeRobotDataset.create(...)`
dataset_path = Path.home() / ".cache" / "huggingface" / "lerobot" / "cubeow" / "lerobot-data"
if dataset_path.exists():
    shutil.rmtree(dataset_path)

def init_keyboard_listener():
    events = {
        "exit_early": False,
        "rerecord_episode": False,
        "stop_recording": False,
    }

    def on_press(key):
        try:
            if key == keyboard.Key.esc:
                events["stop_recording"] = True
            elif hasattr(key, "char"):
                if key.char == "r":
                    events["rerecord_episode"] = True
                elif key.char == "e":
                    events["exit_early"] = True
                elif key.char == "q":
                    events["stop_recording"] = True
        except Exception:
            pass

    listener = keyboard.Listener(on_press=on_press)
    listener.daemon = True
    listener.start()
    return listener, events

NUM_EPISODES = 5
FPS = 15
EPISODE_TIME_SEC = 60
RESET_TIME_SEC = 10
TASK_DESCRIPTION = "My task description"

def main():
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

    teleop_config = OmxLeaderConfig(
    port="/dev/tty.usbmodem1101",  # Replace with leader U2D2 port
    id="omx_leader",
    )

    robot = OmxFollower(robot_config)
    teleop = OmxLeader(teleop_config)

    action_features = hw_to_dataset_features(robot.action_features, "action")
    obs_features = hw_to_dataset_features(robot.observation_features, "observation")
    dataset_features = {**action_features, **obs_features}

    dataset = LeRobotDataset.create(
        repo_id="cubeow/lerobot-data",
        fps=FPS,
        features=dataset_features,
        robot_type=robot.name,
        use_videos=True,
        image_writer_threads=8,
    )

    _, events = init_keyboard_listener()
    init_visualization("rerun", session_name="recording")

    robot.connect()
    teleop.connect()

    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    episode_idx = 0

    while episode_idx < NUM_EPISODES and not events["stop_recording"]:
        log_say(f"Recording episode {episode_idx + 1} of {NUM_EPISODES}")
        
        record_loop(
            robot=robot,
            events=events,
            fps=FPS,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
            teleop=teleop,
            dataset=dataset,
            control_time_s=EPISODE_TIME_SEC,
            single_task=TASK_DESCRIPTION,
            display_data=True,
        )

        # Reset the environment if not stopping or re-recording
        if not events["stop_recording"] and (episode_idx < NUM_EPISODES - 1 or events["rerecord_episode"]):
            log_say("Reset the environment")
            record_loop(
                robot=robot,
                events=events,
                fps=FPS,
                teleop_action_processor=teleop_action_processor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
                teleop=teleop,
                control_time_s=RESET_TIME_SEC,
                single_task=TASK_DESCRIPTION,
                display_data=True,
            )

        if events["rerecord_episode"]:
            log_say("Re-recording episode")
            events["rerecord_episode"] = False
            events["exit_early"] = False
            dataset.clear_episode_buffer()
            continue

        dataset.save_episode()
        episode_idx += 1
    # finalize dataset
    log_say("Finalizing dataset...")
    dataset.finalize()
    # Clean up
    log_say("Stop recording")
    robot.disconnect()
    teleop.disconnect()
    dataset.push_to_hub()
if __name__ == "__main__":
    main()