"""
Regenerates a LIBERO dataset (HDF5 files) by replaying demonstrations in the environments,
and exports it to the Hugging Face Hub in the LeRobotDataset v3.0 format.

Notes:
    - We save image observations at --image_resolution x --image_resolution px (default 64).
    - We filter out transitions with "no-op" (zero) actions that do not change the robot's state.
    - We filter out unsuccessful demonstrations.
    - We rotate the images by 180 degrees because the environments return them upside down.
    - Regeneration (needs libero + robosuite) and export (needs lerobot >= 0.4) are separate steps,
      so they can run in different environments. Regeneration skips tasks that are already done,
      so it can be resumed or split over several jobs with --task_id.

Setup (own uv env in openvla/.venv, separate from the training code; uv-managed Python, no conda):
    uv venv --python 3.10 --python-preference only-managed .venv
    uv pip install --python .venv/bin/python "lerobot>=0.4" h5py tqdm mujoco==3.4.0 robosuite==1.4.0 \
        gym==0.25.2 bddl==1.0.1 easydict==1.9 future cloudpickle matplotlib
    uv pip install --python .venv/bin/python --no-deps -e ../LIBERO

Usage (with .venv/bin/python):
    # 1) Replay demos (optionally one task per job with --task_id)
    MUJOCO_GL=egl python experiments/robot/libero/regenerate_libero_dataset.py regenerate \
        --libero_task_suite libero_object \
        --libero_raw_data_dir /network/projects/real-g-grp/libero/libero_object/ \
        --libero_target_dir /network/projects/real-g-grp/libero/libero_object_no_noops/

    # 2) Export to LeRobot v3 and push
    python experiments/robot/libero/regenerate_libero_dataset.py push \
        --libero_task_suite libero_object \
        --libero_target_dir /network/projects/real-g-grp/libero/libero_object_no_noops/ \
        --lerobot_root /network/projects/real-g-grp/libero/lerobot/libero_object_no_noops \
        --hf_repo_id gberseth/libero_object_no_noops_lerobot

    # 3) Check the pushed dataset: re-download it and replay episodes in LIBERO (see test_libero_dataset_replay.py)
    MUJOCO_GL=egl python experiments/robot/libero/test_libero_dataset_replay.py \
        --hf_repo_id gberseth/libero_object_no_noops_lerobot --libero_task_suite libero_object

Besides the LeRobot files, the push writes meta/libero_init_states.parquet: each episode's initial MuJoCo state, which
replays need. LIBERO's own init states for a task are not the ones the demos started from.
"""

import argparse
import json
import os

import h5py
import numpy as np
import tqdm

FPS = 20  # LIBERO control frequency


def is_noop(action, prev_action=None, threshold=1e-4):
    """
    Returns whether an action is a no-op action.

    A no-op action satisfies two criteria:
        (1) All action dimensions, except for the last one (gripper action), are near zero.
        (2) The gripper action is equal to the previous timestep's gripper action.

    Criterion (2) keeps actions where the robot stays still but opens/closes its gripper.
    """
    if prev_action is None:
        return np.linalg.norm(action[:-1]) < threshold
    return np.linalg.norm(action[:-1]) < threshold and action[-1] == prev_action[-1]


def get_libero_env(task, resolution):
    """Initializes and returns the LIBERO environment, along with the task description."""
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=task_bddl_file, camera_heights=resolution, camera_widths=resolution)
    env.seed(0)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task.language


def replay_demo(env, demo_data):
    """Replays one demo without its no-op actions. Returns (success, dict of recorded arrays)."""
    import robosuite.utils.transform_utils as T

    orig_actions = demo_data["actions"][()]
    orig_states = demo_data["states"][()]

    # Reset environment, set initial state, and wait a few steps for environment to settle
    env.reset()
    env.set_init_state(orig_states[0])
    for _ in range(10):
        obs, reward, done, info = env.step([0, 0, 0, 0, 0, 0, -1])

    rec = {k: [] for k in ["states", "actions", "ee_states", "gripper_states", "joint_states",
                           "robot_states", "agentview_rgb", "eye_in_hand_rgb"]}
    for action in orig_actions:
        if is_noop(action, rec["actions"][-1] if rec["actions"] else None):
            continue
        if not rec["states"]:
            # In the first timestep copy the initial state over from the original demo
            rec["states"].append(orig_states[0])
            rec["robot_states"].append(demo_data["robot_states"][0])
        else:
            rec["states"].append(env.sim.get_state().flatten())
            rec["robot_states"].append(
                np.concatenate([obs["robot0_gripper_qpos"], obs["robot0_eef_pos"], obs["robot0_eef_quat"]]))
        rec["actions"].append(action)
        rec["gripper_states"].append(obs["robot0_gripper_qpos"])
        rec["joint_states"].append(obs["robot0_joint_pos"])
        rec["ee_states"].append(np.hstack((obs["robot0_eef_pos"], T.quat2axisangle(obs["robot0_eef_quat"]))))
        ## Rotate images by 180 degrees to account for upside-down rendering
        rec["agentview_rgb"].append(np.rot90(obs["agentview_image"], 2))
        rec["eye_in_hand_rgb"].append(np.rot90(obs["robot0_eye_in_hand_image"], 2))
        obs, reward, done, info = env.step(action.tolist())

    rec = {k: np.stack(v) for k, v in rec.items()} if rec["actions"] else {}
    return bool(done), rec


def is_complete(path):
    """Whether a regenerated task file exists and was fully written (a killed run can leave a broken file)."""
    try:
        with h5py.File(path, "r") as f:
            return bool(f["data"].attrs.get("complete", False))
    except (OSError, KeyError):
        return False


def regenerate(args):
    from libero.libero import benchmark

    print(f"Regenerating {args.libero_task_suite} dataset!")
    os.makedirs(args.libero_target_dir, exist_ok=True)
    task_suite = benchmark.get_benchmark_dict()[args.libero_task_suite]()
    task_ids = range(task_suite.n_tasks) if args.task_id is None else [args.task_id]

    for task_id in tqdm.tqdm(task_ids):
        task = task_suite.get_task(task_id)
        new_data_path = os.path.join(args.libero_target_dir, f"{task.name}_demo.hdf5")
        if is_complete(new_data_path):
            print(f"Skipping finished task: {task.name}")
            continue

        env, task_description = get_libero_env(task, args.image_resolution)
        orig_data_path = os.path.join(args.libero_raw_data_dir, f"{task.name}_demo.hdf5")
        assert os.path.exists(orig_data_path), f"Cannot find raw data file {orig_data_path}."
        with h5py.File(orig_data_path, "r") as orig_file, h5py.File(new_data_path + ".tmp", "w") as new_file:
            orig_data = orig_file["data"]
            grp = new_file.create_group("data")
            grp.attrs["language_instruction"] = task_description
            success_per_demo = {}
            for i in range(len(orig_data.keys()))[:args.max_demos_per_task]:
                done, rec = replay_demo(env, orig_data[f"demo_{i}"])
                success_per_demo[f"demo_{i}"] = done
                if not done:  # Only keep successful replays
                    continue
                n = len(rec["actions"])
                ep = grp.create_group(f"demo_{i}")
                obs_grp = ep.create_group("obs")
                for k in ["gripper_states", "joint_states", "ee_states", "agentview_rgb", "eye_in_hand_rgb"]:
                    obs_grp.create_dataset(k, data=rec[k])
                obs_grp.create_dataset("ee_pos", data=rec["ee_states"][:, :3])
                obs_grp.create_dataset("ee_ori", data=rec["ee_states"][:, 3:])
                ep.create_dataset("actions", data=rec["actions"])
                ep.create_dataset("states", data=rec["states"])
                ep.create_dataset("robot_states", data=rec["robot_states"])
                ep.create_dataset("rewards", data=(np.arange(n) == n - 1).astype(np.uint8))
                ep.create_dataset("dones", data=(np.arange(n) == n - 1).astype(np.uint8))
                ep.create_dataset("init_states", data=np.array(orig_data[f"demo_{i}"]["states"][0]))
            grp.attrs["success_per_demo"] = json.dumps(success_per_demo)
            grp.attrs["complete"] = args.max_demos_per_task is None
        env.close()
        os.replace(new_data_path + ".tmp", new_data_path)  # Only finished tasks appear under the final name
        print(f"Task '{task_description}': {sum(success_per_demo.values())}/{len(success_per_demo)} "
              f"successful replays -> {new_data_path}")


def libero_features(res):
    return {
        "observation.images.image": {"dtype": "video", "shape": (res, res, 3), "names": ["height", "width", "channels"]},
        "observation.images.wrist_image": {"dtype": "video", "shape": (res, res, 3), "names": ["height", "width", "channels"]},
        "observation.state": {  # End effector position + axis-angle orientation + gripper joint positions
            "dtype": "float32", "shape": (8,),
            "names": ["x", "y", "z", "axis_angle1", "axis_angle2", "axis_angle3", "gripper_left", "gripper_right"]},
        "observation.joint_state": {
            "dtype": "float32", "shape": (7,), "names": [f"joint_{i}" for i in range(7)]},
        "action": {  # Delta end effector pose + gripper (-1 open, 1 close)
            "dtype": "float32", "shape": (7,),
            "names": ["x", "y", "z", "axis_angle1", "axis_angle2", "axis_angle3", "gripper"]},
    }


def iter_demos(target_dir, max_demos_per_task=None):
    """Yields (task, demo name, demo group) for every regenerated demo in export order: files sorted by name, demos
    by number. The n-th demo yielded is episode n of the LeRobot dataset."""
    for hdf5_file in sorted(f for f in os.listdir(target_dir) if f.endswith(".hdf5")):
        with h5py.File(os.path.join(target_dir, hdf5_file), "r") as f:
            data_grp = f["data"]
            # Older regenerated files have no language attribute; their file name is the instruction
            task = data_grp.attrs.get("language_instruction", hdf5_file.replace("_demo.hdf5", "").replace("_", " "))
            for demo_key in sorted(data_grp.keys(), key=lambda k: int(k.split("_")[1]))[:max_demos_per_task]:
                yield task, f"{hdf5_file}/{demo_key}", data_grp[demo_key]


def write_init_states(target_dir, lerobot_root, max_demos_per_task=None):
    """Writes meta/libero_init_states.parquet: the MuJoCo state each episode starts from, which is needed to replay it
    in LIBERO. Its size differs between tasks, so it can't be a fixed-shape LeRobot feature."""
    import pandas as pd

    rows = [{"episode_index": i, "task": task, "demo": name, "init_state": demo["init_states"][()].tolist()}
            for i, (task, name, demo) in enumerate(iter_demos(target_dir, max_demos_per_task))]
    path = os.path.join(lerobot_root, "meta", "libero_init_states.parquet")
    pd.DataFrame(rows).to_parquet(path)
    return path


def push_to_huggingface_lerobot(args):
    """Converts the regenerated HDF5 files into a LeRobotDataset v3.0 and pushes it to the Hub."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    hdf5_files = sorted(f for f in os.listdir(args.libero_target_dir) if f.endswith(".hdf5"))
    print(f"Found {len(hdf5_files)} HDF5 files, writing LeRobot dataset to {args.lerobot_root}")
    # The video size is taken from the regenerated images, so it always matches the regenerate step
    res = None
    for hdf5_file in hdf5_files:
        with h5py.File(os.path.join(args.libero_target_dir, hdf5_file), "r") as f:
            res = res or next((d["obs/agentview_rgb"].shape[1] for d in f["data"].values()), None)
    dataset = LeRobotDataset.create(repo_id=args.hf_repo_id, root=args.lerobot_root, fps=FPS,
                                    features=libero_features(res), robot_type="panda")
    # The layout of the official LeRobot datasets (e.g. lerobot/droid_1.0.1): many episodes per MP4, a new file past 100 MB (--video_files_size_in_mb).
    # One MP4 per episode (as before 2026-10-12) made ~8k Hub files for libero_90, which the Hub's rate limit takes an hour
    # to download; loaders seek to each episode's videos/<camera>/from_timestamp in its file.
    dataset.meta.update_chunk_settings(video_files_size_in_mb=args.video_files_size_in_mb)
    for task, name, demo in tqdm.tqdm(iter_demos(args.libero_target_dir, args.max_demos_per_task), desc="Exporting demos"):
        obs = {k: v[()] for k, v in demo["obs"].items()}
        actions = demo["actions"][()].astype(np.float32)
        state = np.concatenate([obs["ee_states"], obs["gripper_states"]], axis=1).astype(np.float32)
        for j in range(len(actions)):
            dataset.add_frame({
                "task": task,
                "observation.images.image": obs["agentview_rgb"][j],
                "observation.images.wrist_image": obs["eye_in_hand_rgb"][j],
                "observation.state": state[j],
                "observation.joint_state": obs["joint_states"][j].astype(np.float32),
                "action": actions[j],
            })
        dataset.save_episode()
    dataset.finalize()  # Closes parquet writers, writes metadata footers
    write_init_states(args.libero_target_dir, dataset.root, args.max_demos_per_task)  # pushed with meta/
    print(f"Created dataset with {dataset.meta.total_episodes} episodes and {dataset.meta.total_frames} frames")

    if not args.no_push:
        print(f"Pushing dataset to Hugging Face Hub: {args.hf_repo_id}")
        dataset.push_to_hub(tags=["LIBERO", args.libero_task_suite, "panda"], private=False)
        print(f"Successfully pushed dataset to https://huggingface.co/datasets/{args.hf_repo_id}")
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["regenerate", "push"])
    parser.add_argument("--libero_task_suite", type=str, required=True,
                        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"])
    parser.add_argument("--libero_raw_data_dir", type=str, help="Directory with the raw LIBERO HDF5 files")
    parser.add_argument("--libero_target_dir", type=str, required=True, help="Directory for the regenerated HDF5 files")
    parser.add_argument("--image_resolution", type=int, default=64, help="Rendered image (and video) height and width")
    parser.add_argument("--task_id", type=int, default=None, help="Only regenerate this task (e.g. one SLURM array job per task)")
    parser.add_argument("--hf_repo_id", type=str, required=False, help="Hub repo, e.g. gberseth/libero_object_no_noops_lerobot")
    parser.add_argument("--lerobot_root", type=str, default=None, help="Local directory for the LeRobot dataset (must not exist); keep it in /network/projects/real-g-grp/libero/lerobot/, not $HF_HOME")
    parser.add_argument("--max_demos_per_task", type=int, default=None, help="Only use the first N demos per task (for testing)")
    parser.add_argument("--video_files_size_in_mb", type=float, default=100, help="Start a new MP4 file past this size; lerobot rewrites the whole MP4 to append each episode, so smaller files export faster (lerobot/droid_1.0.1 uses 500)")
    parser.add_argument("--no_push", action="store_true", help="Build the LeRobot dataset locally without pushing it")
    args = parser.parse_args()

    if args.step == "regenerate":
        regenerate(args)
    else:
        push_to_huggingface_lerobot(args)
