"""
Exports BridgeData V2 (the real WidowX 250 robot, https://rail-berkeley.github.io/bridgedata/) to the Hugging Face Hub
in the LeRobotDataset v3.0 format, with the features of our LIBERO datasets (see ../libero/regenerate_libero_dataset.py)
so the training code (mini-grp's load_lerobots) can mix them. ManiSkill's Bridge digital twins
(PutCarrotOnPlateInScene-v1, ...) evaluate policies trained on it.

Reads the official TFDS copy (bridge_orig: 53,192 train episodes), not the out-of-date one in Open X-Embodiment.

Notes:
    - The actions are relabelled as OpenVLA does (prismatic/vla/datasets/rlds/oxe/transforms.py,
      bridge_orig_dataset_transform): the first step is dropped (its action is all zeros), the motion is the one the
      robot reached (the next state minus this one) rather than the commanded one, and the last step is dropped (it
      has no next state). The gripper action is binarized.
    - The rotation is the axis-angle of R(next) R(this)^-1, the rotation between consecutive states in the robot
      base frame (the state's roll-pitch-yaw is extrinsic xyz; checked against the commanded actions).
    - Actions are in meters and radians per step (about 1 cm), at 5 Hz. LIBERO's and ManiSkill's are in [-1, 1]
      (0.05 m and 0.1 m per unit), at 20 Hz.
    - Gripper: -1 open, 1 close (the LIBERO convention).
    - Only the main camera (image_0, over the shoulder) is exported, as observation.images.image, so training runs that
      include Bridge use dataset.robot_image_key=observation.images.image. Bridge has a wrist camera (image_3), but in
      only 13% of the episodes (image_1 and image_2, other fixed views, are in 43%; first 2000 train episodes).
    - Bridge has no joint positions: observation.joint_state is all zeros, so the pose input (policy.use_pose_data)
      can't be used with it.
    - A third of the episodes have no language instruction; they are dropped (the policy is goal-conditioned).

Setup (own uv env, with TensorFlow to read TFDS and lerobot to write the export; uv-managed Python, no conda):
    uv venv --python 3.10 --python-preference only-managed .venv_bridge
    uv pip install --python .venv_bridge/bin/python tensorflow-cpu tensorflow-datasets "tensorflow-metadata>=1.16" lerobot==0.4.4 opencv-python-headless scipy tqdm

Usage:
    # 1) Download the TFDS files (132 GB, to the project folder, not $HOME)
    cd /network/projects/real-g-grp/bridge && wget -r -nH --cut-dirs=4 --reject="index.html*" \
        https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/bridge_dataset/

    # 2) Export to LeRobot v3 and push (--lerobot_root must not exist yet; --max_episodes N for a quick test)
    .venv_bridge/bin/python experiments/robot/bridge/regenerate_bridge_dataset.py \
        --lerobot_root /network/projects/real-g-grp/bridge/lerobot/bridge_orig --hf_repo_id gberseth/bridge_orig_lerobot

Besides the LeRobot files, the export writes meta/bridge_episodes.parquet: each episode's source file and episode id.
"""

import argparse
import os

import cv2
import numpy as np
import tqdm
from scipy.spatial.transform import Rotation

FPS = 5  # Bridge's control frequency


def binarize_gripper(g):
    """ The gripper actions (about 1 open, 0 closed) as 1 or 0. Values in between, when the gripper is moving, take
    the value of the next open or closed one (OpenVLA's binarize_gripper_actions). """
    out, carry = np.empty_like(g), float(g[-1] > 0.5)
    for i in reversed(range(len(g))):
        if g[i] > 0.95 or g[i] < 0.05:
            carry = float(g[i] > 0.95)
        out[i] = carry
    return out


def convert_episode(steps, res):
    """ One Bridge episode as (images, states, actions), with the actions relabelled (see the notes above). """
    steps = steps[1:]  # the first action is all zeros
    state = np.stack([s["observation"]["state"] for s in steps])  # x, y, z, roll, pitch, yaw, gripper opening
    open_gripper = binarize_gripper(np.stack([s["action"][6] for s in steps]))
    rot = Rotation.from_euler("xyz", state[:, 3:6])
    actions = np.concatenate([state[1:, :3] - state[:-1, :3],  # the motion reached, in the base frame
                              (rot[1:] * rot[:-1].inv()).as_rotvec(),
                              1 - 2 * open_gripper[:-1, None]], axis=1)  # -1 open, 1 close
    states = np.concatenate([state[:, :3], rot.as_rotvec(), state[:, 6:7], np.zeros((len(state), 1))], axis=1)
    images = [cv2.resize(s["observation"]["image_0"], (res, res), interpolation=cv2.INTER_AREA) for s in steps]
    return np.stack(images[:-1]), states[:-1].astype(np.float32), actions.astype(np.float32)  # the last step has no action


def bridge_features(res):
    ## The LIBERO features, without the wrist camera (Bridge's, image_3, is in only 13% of the episodes)
    return {
        "observation.images.image": {"dtype": "video", "shape": (res, res, 3), "names": ["height", "width", "channels"]},
        "observation.state": {  # End effector position + axis-angle orientation + gripper opening + 0
            "dtype": "float32", "shape": (8,),
            "names": ["x", "y", "z", "axis_angle1", "axis_angle2", "axis_angle3", "gripper", "pad"]},
        "observation.joint_state": {  # Not in Bridge: zeros
            "dtype": "float32", "shape": (7,), "names": [f"joint_{i}" for i in range(7)]},
        "action": {  # Delta end effector pose (m, rad) + gripper (-1 open, 1 close)
            "dtype": "float32", "shape": (7,),
            "names": ["x", "y", "z", "axis_angle1", "axis_angle2", "axis_angle3", "gripper"]},
    }


def export(args):
    import pandas as pd
    import tensorflow as tf
    import tensorflow_datasets as tfds
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    tf.config.set_visible_devices([], "GPU")  # TensorFlow only reads the files
    split = args.split if args.max_episodes is None else f"{args.split}[:{args.max_episodes}]"
    source = tfds.builder_from_directory(args.tfds_dir).as_dataset(split=split)
    dataset = LeRobotDataset.create(repo_id=args.hf_repo_id, root=args.lerobot_root, fps=FPS,
                                    features=bridge_features(args.image_resolution), robot_type="widowx")
    ## The layout of the official LeRobot datasets (e.g. lerobot/droid_1.0.1): many episodes per MP4, a new file past 100 MB (--video_files_size_in_mb): few Hub files
    ## to download (the Hub rate-limits requests); loaders seek to each episode's videos/<camera>/from_timestamp in its file.
    dataset.meta.update_chunk_settings(video_files_size_in_mb=args.video_files_size_in_mb)
    episodes, skipped = [], 0
    for ep in tqdm.tqdm(tfds.as_numpy(source), desc="Exporting episodes"):
        steps = list(ep["steps"])
        goal = steps[0]["language_instruction"].decode().strip().lower().rstrip(".")
        if not goal or len(steps) < 3:  # no language, or too short to have an action after relabelling
            skipped += 1
            continue
        images, states, actions = convert_episode(steps, args.image_resolution)
        for j in range(len(actions)):
            dataset.add_frame({
                "task": goal,
                "observation.images.image": images[j],
                "observation.state": states[j],
                "observation.joint_state": np.zeros(7, dtype=np.float32),
                "action": actions[j],
            })
        dataset.save_episode()
        meta = ep["episode_metadata"]
        episodes.append({"episode_index": len(episodes), "file_path": meta["file_path"].decode(),
                         "episode_id": int(meta["episode_id"])})
    dataset.finalize()
    pd.DataFrame(episodes).to_parquet(os.path.join(dataset.root, "meta", "bridge_episodes.parquet"))  # pushed with meta/
    print(f"Created dataset with {dataset.meta.total_episodes} episodes and {dataset.meta.total_frames} frames "
          f"({skipped} episodes skipped: no language or too short)")

    if not args.no_push:
        dataset.push_to_hub(tags=["BridgeData V2", "widowx"], private=False)
        print(f"Successfully pushed dataset to https://huggingface.co/datasets/{args.hf_repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tfds_dir", type=str, default="/network/projects/real-g-grp/bridge/bridge_dataset/1.0.0")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--image_resolution", type=int, default=64, help="Image (and video) height and width")
    parser.add_argument("--video_files_size_in_mb", type=float, default=100, help="Start a new MP4 file past this size; lerobot rewrites the whole MP4 to append each episode, so smaller files export faster (lerobot/droid_1.0.1 uses 500)")
    parser.add_argument("--max_episodes", type=int, default=None, help="Only use the first N episodes (for testing)")
    parser.add_argument("--hf_repo_id", type=str, required=True, help="Hub repo, e.g. gberseth/bridge_orig_lerobot")
    parser.add_argument("--lerobot_root", type=str, required=True, help="Local directory for the LeRobot dataset (must not exist)")
    parser.add_argument("--no_push", action="store_true", help="Build the LeRobot dataset locally without pushing it")
    export(parser.parse_args())
