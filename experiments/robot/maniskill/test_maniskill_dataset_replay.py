"""
Round-trip test for a ManiSkill LeRobot dataset on the Hub (made with regenerate_maniskill_dataset.py): downloads it
fresh, replays the first episodes of each task open loop on the sim its demos were recorded on (the GPU sim for the PPO
demos, the CPU sim for the motion-planning ones; the CPU-recorded demos diverge on the GPU sim), and checks that the
replays succeed. Each episode starts from its stored reset
seed and first env state (meta/maniskill_episodes.parquet), and its stored actions are converted to the controller's as
mini-grp's sim/sim_eval.py eval_maniskill does (the gripper back to ManiSkill's sign; none for the stick). It also compares each episode's first
stored frames (both cameras) with what the env renders from the stored state, to check the cameras match.

A failure means the conversion lost something (the actions, stored as float32 and with the gripper flipped; the init
states; the order of episodes and tasks), or that the demo doesn't replay open loop (PushT's don't: pushing is chaotic
and its demos were recorded in 1024 parallel GPU envs; about 20-60% succeed).

Everything goes to --out_dir (default outputs/replay_test/<dataset>_<time>/): the download (dataset/), and per task a
video of the first replay (agent view) and an image of the first frames, stored (top) and rendered (bottom).

Usage (with openvla/.venv_maniskill/bin/python, on a GPU):
    python experiments/robot/maniskill/test_maniskill_dataset_replay.py --hf_repo_id gberseth/maniskill_panda_lerobot
"""

import argparse
import glob
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

CAMERAS = {"observation.images.image": "base_camera", "observation.images.wrist_image": "hand_camera"}


def download(repo_id, local_dir, episodes_per_task):
    """ The dataset's tables, then the videos of the first episodes_per_task episodes of each task. Returns (root,
    the episodes table with each selected episode's env id, seed and init state). """
    from huggingface_hub import snapshot_download
    root = snapshot_download(repo_id, repo_type="dataset", local_dir=local_dir, allow_patterns=["data/*", "meta/*"])
    episodes = pd.read_parquet(f"{root}/meta/maniskill_episodes.parquet")
    selected = episodes.groupby("env_id", sort=False).head(episodes_per_task).set_index("episode_index")
    files = video_files(root, selected.index)
    snapshot_download(repo_id, repo_type="dataset", local_dir=local_dir, allow_patterns=[f for f in files.values()])
    return root, selected


def video_files(root, episode_indices):
    """ {(episode, dataset camera): video path relative to root, with the episode's start and end times}. """
    meta = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{root}/meta/episodes/*/*.parquet"))]).set_index("episode_index")
    files = {}
    for e in episode_indices:
        for key in CAMERAS:
            v = f"videos/{key}/"
            files[e, key] = f"{v}chunk-{int(meta.loc[e, v + 'chunk_index']):03d}/file-{int(meta.loc[e, v + 'file_index']):03d}.mp4"
    return files


def first_frame(path):
    import av
    with av.open(path) as container:
        return next(container.decode(video=0)).to_ndarray(format="rgb24")


def replay_task(env_id, episodes, actions, root, files, out_dir):
    """ Replays the episodes (rows of the episodes table) of one task, one at a time, on the sim its demos were
    recorded on (open-loop replays of the CPU-recorded motion-planning demos diverge on the GPU sim), in
    regenerate_maniskill_dataset.make_env's env, which has the eval env's cameras. Returns (successes, mean absolute
    pixel difference of the first frames per camera). """
    import imageio
    from mani_skill.utils import common
    from regenerate_maniskill_dataset import TASKS, make_env

    env = make_env(env_id, 64, TASKS[env_id][1].split(".")[-1])  # the demo file's suffix: physx_cuda or physx_cpu
    stored = {k: np.stack([first_frame(os.path.join(root, files[e, k])) for e in episodes.index]) for k in CAMERAS}
    rendered, success = {k: [] for k in CAMERAS}, []
    for i, (e, episode) in enumerate(episodes.iterrows()):
        env.reset(seed=int(episode["episode_seed"]))
        env.set_state(torch.tensor(np.asarray(episode["init_state"])[None], dtype=torch.float32))
        obs = env.get_obs()
        for k, c in CAMERAS.items():
            rendered[k].append(common.to_numpy(obs["sensor_data"][c]["rgb"][0]))
        frames, ok = [], False
        for a in actions[e]:
            ## the sim's action units, as eval_maniskill converts the policy's: no gripper on the stick, ManiSkill's gripper sign
            a = a[:6] if env.single_action_space.shape[0] == 6 else np.append(a[:6], -a[6])
            obs, _, _, _, info = env.step(torch.tensor(a[None]))
            ok |= bool(info["success"][0])
            if i == 0:
                frames.append(common.to_numpy(obs["sensor_data"]["base_camera"]["rgb"][0]))
        success.append(ok)
        if i == 0:
            imageio.mimsave(os.path.join(out_dir, f"{env_id}_replay_0.mp4"), frames, fps=env.control_freq)
    env.close()
    rendered = {k: np.stack(v) for k, v in rendered.items()}
    pixel_error = {k: float(np.abs(stored[k].astype(float) - rendered[k]).mean()) for k in CAMERAS}
    imageio.imwrite(os.path.join(out_dir, f"{env_id}_first_frames.png"),
                    np.concatenate([np.concatenate([*stored[k], *rendered[k]], axis=0) for k in CAMERAS], axis=1))
    return np.array(success), pixel_error


def main(args):
    out_dir = args.out_dir or os.path.join("outputs", "replay_test", f"{args.hf_repo_id.split('/')[-1]}_{time.strftime('%Y-%m-%d_%H-%M-%S')}")
    root, selected = download(args.hf_repo_id, os.path.join(out_dir, "dataset"), args.episodes_per_task)
    frames = pd.concat([pd.read_parquet(f, columns=["episode_index", "frame_index", "action"])
                        for f in sorted(glob.glob(f"{root}/data/*/*.parquet"))])
    frames = frames[frames["episode_index"].isin(selected.index)]
    actions = {int(e): np.stack(g.sort_values("frame_index")["action"].to_numpy()) for e, g in frames.groupby("episode_index")}
    files = video_files(root, selected.index)

    results = {}
    for env_id, episodes in selected.groupby("env_id", sort=False):
        success, pixel_error = replay_task(env_id, episodes, actions, root, files, out_dir)
        results[env_id] = success.mean()
        print(f"{env_id}: {success.sum()}/{len(success)} replays succeeded; first frames differ from the stored ones by "
              + ", ".join(f"{CAMERAS[k]} {v:.1f}" for k, v in pixel_error.items()) + " (mean |pixel|, 0-255)", flush=True)
    print(f"Overall success rate: {np.mean(list(results.values())):.3f} (videos and first frames in {out_dir})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_repo_id", type=str, required=True, help="e.g. gberseth/maniskill_panda_lerobot")
    parser.add_argument("--episodes_per_task", type=int, default=10, help="Replay the first N episodes of each task")
    parser.add_argument("--out_dir", type=str, default=None, help="Run folder (default: outputs/replay_test/<dataset>_<time>)")
    main(parser.parse_args())
