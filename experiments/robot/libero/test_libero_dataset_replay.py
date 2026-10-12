"""
Round-trip test for a LIBERO LeRobot dataset on the Hub: downloads it fresh, replays its episodes open loop in LIBERO
with mini-grp's sim/sim_eval.py (eval_libero), and checks that every replay still succeeds. regenerate_libero_dataset.py
only keeps demos that succeeded on replay, so a failure here means the conversion lost something: the actions (stored
as float32), the initial states, or the order of episodes and tasks.

Everything goes to --out_dir (default outputs/replay_test/<suite>_<time>/): the fresh download (dataset/), eval_libero's
replay videos (replay/), and one side-by-side video per episode (compare/) with the agent view stored in the dataset on
the left and the LIBERO replay on the right, for checking by eye that they match.

Usage (with openvla/.venv/bin/python):
    MUJOCO_GL=egl python experiments/robot/libero/test_libero_dataset_replay.py \
        --hf_repo_id gberseth/libero_spatial_no_noops_lerobot --libero_task_suite libero_spatial --episodes_per_task 2
"""

import argparse
import glob
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

SETTLE_STEPS = 10  # no-op steps before the replay, as in regenerate_libero_dataset.replay_demo


class LeRobotTrajectories:
    """The episodes of a LeRobot v3 LIBERO dataset: their recorded actions and init states."""

    def __init__(self, root):
        frames = pd.concat([pd.read_parquet(f, columns=["episode_index", "frame_index", "action"])
                            for f in sorted(glob.glob(f"{root}/data/*/*.parquet"))])
        self.actions = {int(e): np.stack(g.sort_values("frame_index")["action"].to_numpy())
                        for e, g in frames.groupby("episode_index")}
        self.init = pd.read_parquet(f"{root}/meta/libero_init_states.parquet").set_index("episode_index")

    def episodes_of(self, task_name):
        """ Episodes of a LIBERO task by its name (scene + instruction): in libero_90 several scenes share an
        instruction, so match the demo file each episode came from, e.g. KITCHEN_SCENE2_..._demo.hdf5/demo_3. """
        return self.init.index[self.init["demo"].str.startswith(task_name + "_demo.hdf5/")].tolist()



class LeRobotReplayModel(nn.Module):
    """ What eval_libero replays open loop (a model with recorded_episode, like mini-grp's ReplayModel): the dataset's
    episodes of the task being tested (dataset.task_name; in libero_90 several scenes share an instruction). """

    def __init__(self, dataset, cfg):
        super().__init__()
        self.dataset, self._cfg = dataset, cfg

    def recorded_episode(self, goal_text, index):
        episode = self.dataset.episodes_of(self.dataset.task_name)[index]
        assert self.dataset.init.loc[episode, "task"] == goal_text, (episode, goal_text)
        return np.array(self.dataset.init.loc[episode, "init_state"]), self.dataset.actions[episode]

    def preprocess_goal_image(self, image):
        return image


def check_init_states(task_suite, task_ids, dataset, episodes_per_task):
    """Loads each selected episode's stored init state into a LIBERO env for its task and reads the sim state back.
    LIBERO can build a scene with different objects or goals than the dataset used (other bddl files or assets), and
    then the stored state doesn't fit: this fails here instead of as an unexplained replay failure."""
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    for task_id in task_ids:
        task = task_suite.get_task(task_id)
        env = OffScreenRenderEnv(bddl_file_name=os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file),
                                 camera_heights=64, camera_widths=64)
        env.seed(0)
        env.reset()
        for episode in dataset.episodes_of(task.name)[:episodes_per_task]:
            init_state = np.array(dataset.init.loc[episode, "init_state"])
            size = env.sim.get_state().flatten().size
            assert init_state.size == size, f"episode {episode}: init state has {init_state.size} values, the '{task.name}' scene has {size}"
            env.set_init_state(init_state)
            error = np.abs(env.sim.get_state().flatten() - init_state).max()
            assert error < 1e-8, f"episode {episode}: sim state differs from the stored init state by {error}"
        env.close()
    print(f"Init states of {len(task_ids)} tasks load into LIBERO unchanged")


def comparison_videos(root, dataset, task_suite, task_ids, episodes_per_task, replay_dir, out_dir):
    """Writes out_dir/task_<id>_episode_<n>.mp4 for each replayed episode: the agent-view frames stored in the dataset
    (left, upscaled from 64 px) next to the frames eval_libero rendered during the replay (right)."""
    import av
    import cv2
    import imageio

    os.makedirs(out_dir, exist_ok=True)
    key = "videos/observation.images.image/"
    episodes = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{root}/meta/episodes/*/*.parquet"))]).set_index("episode_index")
    for task_id in task_ids:
        for k, episode in enumerate(dataset.episodes_of(task_suite.get_task(task_id).name)[:episodes_per_task]):
            ep = episodes.loc[episode]
            path = f"{root}/{key}chunk-{int(ep[key + 'chunk_index']):03d}/file-{int(ep[key + 'file_index']):03d}.mp4"
            with av.open(path) as container:  # the episode's frames, also when a file holds several episodes
                stored = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)
                          if ep[key + "from_timestamp"] - 1e-3 <= f.time < ep[key + "to_timestamp"] - 1e-3]
            with av.open(f"{replay_dir}/libero-0-task-id-{task_id}-init-id-{k}.mp4") as container:
                ## eval_libero flips the render vertically only; the dataset rotated it 180 degrees, so flip horizontally too
                replay = [f.to_ndarray(format="rgb24")[:, ::-1] for f in container.decode(video=0)]
            size = replay[0].shape[0]
            frames = []
            for j, right in enumerate(replay):  # replay frame j is after action j, which is stored frame j + 1
                left = cv2.resize(stored[min(j + 1, len(stored) - 1)], (size, size), interpolation=cv2.INTER_NEAREST)
                frame = np.concatenate([left, right], axis=1).copy()
                cv2.putText(frame, "dataset", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f"LIBERO replay  t={j}", (size + 5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                frames.append(frame)
            imageio.mimsave(f"{out_dir}/task_{task_id}_episode_{episode}.mp4", frames, fps=20)
    print(f"Side-by-side videos (dataset left, replay right) in {out_dir}")


def download(repo_id, local_dir):
    """ A fresh download of the dataset into local_dir (a new folder per test, not the Hub cache); the wrist videos
    aren't needed for the check. The Hub allows 1000 requests per 5 minutes and a per-episode dataset has one file per
    episode, so when rate limited (a 429, or a snapshot that quietly comes back incomplete) wait and resume. """
    import json
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import HfHubHTTPError
    for attempt in range(60):
        try:
            root = snapshot_download(repo_id, repo_type="dataset", local_dir=local_dir,
                                     allow_patterns=["data/*", "meta/*", "videos/observation.images.image/*"])
            info = json.load(open(os.path.join(root, "meta", "info.json")))
            videos = glob.glob(os.path.join(root, "videos", "observation.images.image", "*", "*.mp4"))
            if len(videos) >= info["total_episodes"]: ## one per episode
                return root
            print(f"{repo_id}: {len(videos)} of {info['total_episodes']} videos downloaded (Hub rate limit?), retrying in 60 s")
        except HfHubHTTPError as e:
            if e.response is None or e.response.status_code != 429:
                raise
            print(f"{repo_id}: Hub rate limit, retrying in 60 s")
        except FileNotFoundError:
            print(f"{repo_id}: not all downloaded yet, retrying in 60 s")
        time.sleep(60)
    raise RuntimeError(f"{repo_id}: could not download the dataset")


def main(args):
    from huggingface_hub import snapshot_download
    from libero.libero import benchmark
    from omegaconf import OmegaConf

    sys.path.insert(0, args.sim_eval_dir)
    from sim_eval import eval_libero

    out_dir = args.out_dir or os.path.join("outputs", "replay_test", f"{args.libero_task_suite}_{time.strftime('%Y-%m-%d_%H-%M-%S')}")
    root = download(args.hf_repo_id, os.path.join(out_dir, "dataset"))
    dataset = LeRobotTrajectories(root)
    task_suite = benchmark.get_benchmark_dict()[args.libero_task_suite]()
    task_ids = range(task_suite.n_tasks) if args.task_ids is None else args.task_ids
    for task_id in task_ids:  # Every benchmark task must match a task in the dataset by its name (scene + instruction)
        name = task_suite.get_task(task_id).name
        assert len(dataset.episodes_of(name)) >= args.episodes_per_task, f"too few episodes for task '{name}'"
    check_init_states(task_suite, task_ids, dataset, args.episodes_per_task)
    selected = [e for t in task_ids for e in dataset.episodes_of(task_suite.get_task(t).name)[:args.episodes_per_task]]

    log_dir = os.path.join(out_dir, "replay")
    os.makedirs(log_dir, exist_ok=True)
    cfg = OmegaConf.create({
        "sim": {"task_set": args.libero_task_suite, "eval_tasks": [],
                "num_init_states": list(range(args.episodes_per_task)),  # trajectory index within each task
                "wait_steps": SETTLE_STEPS},  # eval_libero replays each episode to its end
        "policy": {"action_stacking": 1, "use_pose_data": False}, "device": "cpu",
    })
    per_task = {}
    for task_id in task_ids:  # one task at a time, so the replay model knows which scene's episodes to replay
        dataset.task_name = task_suite.get_task(task_id).name
        cfg.sim.eval_tasks = [int(task_id)]
        per_task[int(task_id)] = eval_libero(LeRobotReplayModel(dataset, cfg), cfg, log_dir=log_dir)
    comparison_videos(root, dataset, task_suite, task_ids, args.episodes_per_task, log_dir, os.path.join(out_dir, "compare"))

    print(f"\nReplayed {len(selected)} episodes of {args.hf_repo_id} (videos in {log_dir}):")
    for task_id, stats in per_task.items():
        n = stats["num_episodes"]
        print(f"  task {task_id}: {round(stats['success_rate'] * n)}/{n} succeeded  {task_suite.get_task(task_id).name}")
    success_rate = float(np.mean([s["success_rate"] for s in per_task.values()]))
    print(f"Overall success rate: {success_rate:.3f}")
    return success_rate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_repo_id", type=str, required=True, help="e.g. gberseth/libero_spatial_no_noops_lerobot")
    parser.add_argument("--libero_task_suite", type=str, required=True,
                        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"])
    parser.add_argument("--episodes_per_task", type=int, default=2, help="Replay the first N episodes of each task")
    parser.add_argument("--task_ids", type=int, nargs="*", default=None, help="Benchmark task ids to test (default: all)")
    parser.add_argument("--sim_eval_dir", type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../../mini-grp/sim"),
                        help="Directory with mini-grp's sim_eval.py")
    parser.add_argument("--out_dir", type=str, default=None, help="Run folder (default: outputs/replay_test/<suite>_<time>)")
    args = parser.parse_args()
    sys.exit(0 if main(args) == 1.0 else 1)
