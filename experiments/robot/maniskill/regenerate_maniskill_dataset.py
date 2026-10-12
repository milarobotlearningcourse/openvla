"""
Regenerates ManiSkill demo datasets with camera images, checks them in the simulator, and exports them to the Hugging
Face Hub in the LeRobotDataset v3.0 format, with the same features as our LIBERO datasets (see
../libero/regenerate_libero_dataset.py), so the training code (mini-grp's load_lerobots) can mix them.

The demos are ManiSkill's (https://maniskill.readthedocs.io/en/latest/user_guide/datasets/demos.html, the Hub repo
haosulab/ManiSkill_Demonstrations), recorded without observations. TASKS lists the 13 tabletop tasks of one arm that
have demos: 8 with demos of a PPO policy in pd_ee_delta_pose on the GPU sim (physx_cuda), and 5 with only
motion-planning demos in pd_joint_pos on the CPU sim, which ManiSkill's replay tool converts to pd_ee_delta_pose first
(step 2 below). They are exported as one dataset per robot:
    - gberseth/maniskill_panda_lerobot: the 11 tasks of the Panda with its gripper (panda, panda_wristcam);
    - gberseth/maniskill_panda_stick_lerobot: PushT-v1 and DrawTriangle-v1, the Panda with a stick (no gripper).
(PushT-v1 alone is also gberseth/maniskill_pusht_lerobot, made before the others.)

Notes:
    - Open-loop replay of the actions doesn't always reproduce a demo (e.g. PushT: 57% success on the GPU sim, 0% on
      the CPU sim: pushing is chaotic and the PPO demos were recorded in 1024 parallel envs). So each step restores the
      recorded env state and renders it. A demo is kept if (1) ManiSkill's evaluate() reports success on the restored
      states, and (2) the recorded action, applied from each restored state, moves every object to near its next
      recorded position (the actions are consistent with the states; --max_step_error).
    - Each demo is cut at its first success (the PPO rollouts keep going to the step limit), as the LIBERO demos end there.
    - Two cameras, as in LIBERO: the task's own third-person base camera, and a wrist camera. The Panda with a wrist
      camera (panda_wristcam: PegInsertionSide, PlugCharger, StackCube, StackPyramid) has its own; the plain Panda gets
      one at the same place on its hand; the Panda with a stick (PushT, DrawTriangle) one on the stick.
    - PickCube's goal (a green sphere) is shown to the cameras, which ManiSkill hides (its policies get the goal position).
    - The robot's pose at each step: its 7 arm joint positions (observation.joint_state) and its end effector pose and
      gripper fingers (observation.state).
    - Actions are ManiSkill's normalized pd_ee_delta_pose actions (0.1 m and 0.1 rad per unit), in the robot base
      frame, as recorded: the PPO demos' go well past [-1, 1] (up to about 5), and the env doesn't clip them, so clipping
      them breaks the demos (PokeCube: 10/10 replays succeed as recorded, 0/10 clipped; datasets before 2026-10-06,
      including gberseth/maniskill_pusht_lerobot, were clipped). The motion-planning demos stay within [-1, 1].
      LIBERO's are in [-1, 1], with other scales (0.05 m and 0.5 rad per unit). The gripper is stored in LIBERO's convention (-1 open,
      1 close), the opposite of ManiSkill's; the stick has no gripper, so it is always -1 (open).

Setup (own uv env, separate from the LIBERO one; uv-managed Python, no conda):
    uv venv --python 3.10 --python-preference only-managed .venv_maniskill
    uv pip install --python .venv_maniskill/bin/python mani_skill h5py tqdm

Usage:
    # 1) Download the demos (to the project folder, not $HOME)
    MS_ASSET_DIR=/network/projects/real-g-grp/maniskill .venv_maniskill/bin/python -m mani_skill.utils.download_demo \
        PickCube-v1 -o /network/projects/real-g-grp/maniskill/demos

    # 2) For the motion-planning demos only: convert their actions to pd_ee_delta_pose (CPU). PlugCharger's
    #    trajectory.json asks for a dense reward, which the env doesn't have: set its "reward_mode" to "sparse" first
    #    (the downloaded file is kept as trajectory.json.orig).
    .venv_maniskill/bin/python -m mani_skill.trajectory.replay_trajectory -c pd_ee_delta_pose -b physx_cpu --save-traj \
        -n 6 --traj-path /network/projects/real-g-grp/maniskill/demos/PegInsertionSide-v1/motionplanning/trajectory.h5

    # 3) Render and check them (a GPU for the PPO demos; experiments/robot/maniskill/regenerate.sbatch runs a task)
    .venv_maniskill/bin/python experiments/robot/maniskill/regenerate_maniskill_dataset.py regenerate \
        --env_ids PickCube-v1 --target_dir /network/projects/real-g-grp/maniskill

    # 4) Export to LeRobot v3 and push (with the LIBERO env, which has lerobot); several tasks make one dataset
    .venv/bin/python experiments/robot/maniskill/regenerate_maniskill_dataset.py push \
        --env_ids PushT-v1 DrawTriangle-v1 --target_dir /network/projects/real-g-grp/maniskill \
        --lerobot_root /network/projects/real-g-grp/maniskill/lerobot/panda_stick --hf_repo_id gberseth/maniskill_panda_stick_lerobot

Besides the LeRobot files, the push writes meta/maniskill_episodes.parquet: each episode's env id, reset seed and first
env state (env.get_state()'s vector), which a ManiSkill evaluation or replay needs: env.reset(seed=...), then
env.set_state(init_state).
"""

import argparse
import json
import os

import h5py
import numpy as np
import tqdm

FPS = 20  # ManiSkill tabletop control frequency, the same as LIBERO's

## Per task: the goal text, and the demo file (under /network/projects/real-g-grp/maniskill/demos/<env_id>/), in
## pd_ee_delta_pose. The goal texts must match mini-grp's sim/sim_eval.py MANISKILL_TASKS.
PPO = "rl/trajectory.none.pd_ee_delta_pose.physx_cuda"
MOTION_PLANNING = "motionplanning/trajectory.none.pd_ee_delta_pose.physx_cpu"  # converted in step 2
TASKS = {
    "PushT-v1": ("push the t-shaped block onto the t-shaped goal", PPO),
    "PickCube-v1": ("pick up the red cube and move it to the green goal", PPO),
    "PushCube-v1": ("push the cube onto the red and white target", PPO),
    "PullCube-v1": ("pull the cube onto the red and white target", PPO),
    "PokeCube-v1": ("use the peg to push the red cube to the goal", PPO),
    "StackCube-v1": ("stack the red cube on top of the green cube", PPO),
    "LiftPegUpright-v1": ("lift the peg upright", PPO),
    "RollBall-v1": ("roll the ball to the goal region", PPO),
    "PegInsertionSide-v1": ("insert the peg into the hole in the box", MOTION_PLANNING),
    "PlugCharger-v1": ("plug the charger into the receptacle", MOTION_PLANNING),
    "PullCubeTool-v1": ("use the tool to pull the cube closer", MOTION_PLANNING),
    "StackPyramid-v1": ("stack the red, green and blue cubes into a pyramid", MOTION_PLANNING),
    "DrawTriangle-v1": ("draw the triangle", MOTION_PLANNING),
}
DEMO_DIR = "/network/projects/real-g-grp/maniskill/demos"


def make_env(env_id, resolution, sim_backend):
    """ The task on the sim its demos were recorded in, with its base camera and a wrist camera at
    resolution x resolution px. """
    import sapien
    import mani_skill.envs  # noqa: F401 (registers the envs)
    from mani_skill.sensors.camera import CameraConfig
    from mani_skill.utils import sapien_utils
    from mani_skill.utils.registration import REGISTERED_ENVS

    class WithWristCamera(REGISTERED_ENVS[env_id].cls):
        def _load_scene(self, options):
            super()._load_scene(options)
            self._hidden_objects.clear()  # PickCube hides its goal from the cameras, the only way the policy sees it

        @property
        def _default_sensor_configs(self):
            ## A robot's own cameras replace an env camera of the same name, so the panda_wristcam keeps its own
            links = self.agent.robot.links_map
            if self.agent.uid != "panda_stick":  # where panda_wristcam's camera_link is on the hand, looking along the gripper
                mount, pose = links["panda_hand"], sapien.Pose(p=[0.0465, -0.02, 0.036], q=[0, 0.7071, 0, 0.7071])
            else:  # the stick: 10 cm above its tip, looking down along it at the table
                mount, pose = links["panda_hand_tcp"], sapien_utils.look_at(eye=[-0.05, 0, -0.1], target=[0, 0, 0.2])
            return super()._default_sensor_configs + [CameraConfig(
                "hand_camera", pose=pose, width=128, height=128, fov=np.pi / 2, near=0.01, far=100, mount=mount)]

    env = WithWristCamera(obs_mode="rgb", control_mode="pd_ee_delta_pose", sim_backend=sim_backend,
                          sensor_configs=dict(width=resolution, height=resolution))
    assert env.control_freq == FPS
    return env


def regenerate_demo(env, traj, seed, max_step_error):
    """ Renders one demo from its recorded env states. Returns (kept, dict of recorded arrays, stats); stats has the
    first state as env.get_state() gives it, which env.set_state() restores. """
    import torch
    from mani_skill.trajectory.dataset import load_h5_data
    from mani_skill.trajectory.utils import dict_to_list_of_dicts
    from mani_skill.utils import common

    states = dict_to_list_of_dicts(load_h5_data(traj["env_states"]))
    actions = traj["actions"][()]
    env.reset(seed=seed)
    ## every object's position (the first 3 of each actor's 13-D state)
    positions = lambda s: np.concatenate([common.to_numpy(a).reshape(-1, 13)[:, :3].reshape(-1) for a in s["actors"].values()])
    rec = {k: [] for k in ["agentview_rgb", "wrist_rgb", "joint_states", "ee_states", "actions"]}
    step_error, success = 0.0, False
    for t, action in enumerate(actions):
        env.set_state_dict(common.batch(states[t]))
        if t == 0:
            init_state = common.to_numpy(env.get_state()[0])
        obs = env.get_obs()
        qpos = common.to_numpy(obs["agent"]["qpos"][0])
        rec["agentview_rgb"].append(common.to_numpy(obs["sensor_data"]["base_camera"]["rgb"][0]))
        rec["wrist_rgb"].append(common.to_numpy(obs["sensor_data"]["hand_camera"]["rgb"][0]))
        rec["joint_states"].append(qpos[:7])  # the arm's joints (the gripper's fingers are in ee_states)
        rec["ee_states"].append(tcp_state(env, qpos[7:9]))
        ## LIBERO's gripper convention (-1 open, 1 close); the stick has none, so open
        rec["actions"].append(np.append(action[:6], -action[6] if len(action) == 7 else -1.0))
        success = bool(env.evaluate()["success"][0])
        if success:  # cut at the first success, which is where the LIBERO demos end
            break
        env.step(torch.as_tensor(action[None]))
        step_error = max(step_error, np.abs(positions(env.get_state_dict()) - positions(states[t + 1])).max())
    kept = success and step_error < max_step_error
    return kept, {k: np.stack(v) for k, v in rec.items()}, {"step_error": step_error, "success": success, "init_state": init_state}


def tcp_state(env, fingers):
    """ End effector position + axis-angle orientation, and the gripper's two finger joints (zeros for the stick,
    which has none): LIBERO's observation.state layout. """
    from scipy.spatial.transform import Rotation
    from mani_skill.utils import common
    pose = env.agent.tcp.pose.raw_pose[0]
    p, q = common.to_numpy(pose[:3]), common.to_numpy(pose[3:])  # q is w, x, y, z
    fingers = fingers if len(fingers) == 2 else np.zeros(2)
    return np.concatenate([p, Rotation.from_quat(q[[1, 2, 3, 0]]).as_rotvec(), fingers]).astype(np.float32)


def regenerate(args, env_id):
    goal, demo_file = TASKS[env_id]
    path = os.path.join(DEMO_DIR, env_id, demo_file)
    meta = json.load(open(path + ".json"))
    env = make_env(env_id, args.image_resolution, meta["env_info"]["env_kwargs"]["sim_backend"])
    target_file = os.path.join(args.target_dir, f"{env_id}_regenerated.h5")
    kept, errors = 0, []
    with h5py.File(path + ".h5", "r") as src, h5py.File(target_file + ".tmp", "w") as dst:
        grp = dst.create_group("data")
        grp.attrs["language_instruction"] = goal
        grp.attrs["robot"] = env.agent.uid
        for ep in tqdm.tqdm(meta["episodes"][:args.max_demos], desc=env_id, mininterval=30):
            traj = src[f"traj_{ep['episode_id']}"]
            ok, rec, stats = regenerate_demo(env, traj, ep["episode_seed"], args.max_step_error)
            errors.append(stats["step_error"])
            if not ok:
                continue
            demo = grp.create_group(f"demo_{ep['episode_id']}")
            for k, v in rec.items():
                demo.create_dataset(k, data=v)
            demo.attrs["episode_seed"] = ep["episode_seed"]
            demo.create_dataset("init_state", data=stats["init_state"])
            kept += 1
        grp.attrs["complete"] = args.max_demos is None
    env.close()
    os.replace(target_file + ".tmp", target_file)
    print(f"{env_id}: kept {kept}/{len(errors)} demos -> {target_file}; one-step object position error "
          f"median {np.median(errors) * 100:.2f} cm, max {np.max(errors) * 100:.2f} cm")


def maniskill_features(res):
    ## The same features as the LIBERO datasets
    return {
        "observation.images.image": {"dtype": "video", "shape": (res, res, 3), "names": ["height", "width", "channels"]},
        "observation.images.wrist_image": {"dtype": "video", "shape": (res, res, 3), "names": ["height", "width", "channels"]},
        "observation.state": {  # End effector position + axis-angle orientation + gripper joint positions (0: none)
            "dtype": "float32", "shape": (8,),
            "names": ["x", "y", "z", "axis_angle1", "axis_angle2", "axis_angle3", "gripper_left", "gripper_right"]},
        "observation.joint_state": {
            "dtype": "float32", "shape": (7,), "names": [f"joint_{i}" for i in range(7)]},
        "action": {  # Delta end effector pose + gripper (-1 open, 1 close)
            "dtype": "float32", "shape": (7,),
            "names": ["x", "y", "z", "axis_angle1", "axis_angle2", "axis_angle3", "gripper"]},
    }


def push_to_huggingface_lerobot(args):
    """ Converts the regenerated HDF5 files of args.env_ids into one LeRobotDataset v3.0 and pushes it to the Hub. """
    import pandas as pd
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    files = [h5py.File(os.path.join(args.target_dir, f"{env_id}_regenerated.h5"), "r") for env_id in args.env_ids]
    res = next(iter(files[0]["data"].values()))["agentview_rgb"].shape[1]
    robots = sorted({f["data"].attrs.get("robot", "panda_stick") for f in files})  # PushT's file predates the attribute
    dataset = LeRobotDataset.create(repo_id=args.hf_repo_id, root=args.lerobot_root, fps=FPS,
                                    features=maniskill_features(res), robot_type="+".join(robots))
    dataset.meta.update_chunk_settings(video_files_size_in_mb=1e-6)  # one MP4 per episode, as for LIBERO
    episodes = []
    for env_id, f in zip(args.env_ids, files):
        grp = f["data"]
        goal = grp.attrs["language_instruction"]
        for name in tqdm.tqdm(sorted(grp.keys(), key=lambda k: int(k.split("_")[1])), desc=f"Exporting {env_id}"):
            demo = {k: v[()] for k, v in grp[name].items()}
            for j in range(len(demo["actions"])):
                dataset.add_frame({
                    "task": goal,
                    "observation.images.image": demo["agentview_rgb"][j],
                    "observation.images.wrist_image": demo["wrist_rgb"][j],
                    "observation.state": demo["ee_states"][j],
                    "observation.joint_state": demo["joint_states"][j].astype(np.float32),
                    "action": demo["actions"][j].astype(np.float32),
                })
            dataset.save_episode()
            episodes.append({"episode_index": len(episodes), "env_id": env_id, "demo": name,
                             "episode_seed": int(grp[name].attrs["episode_seed"]), "init_state": demo["init_state"].tolist()})
    dataset.finalize()
    pd.DataFrame(episodes).to_parquet(os.path.join(dataset.root, "meta", "maniskill_episodes.parquet"))  # pushed with meta/
    print(f"Created dataset with {dataset.meta.total_episodes} episodes and {dataset.meta.total_frames} frames")

    if not args.no_push:
        from huggingface_hub import CommitOperationDelete, HfApi
        api = HfApi()
        if api.repo_exists(args.hf_repo_id, repo_type="dataset"):  # a rebuild: drop the old version's files (kept in the repo's history)
            api.create_commit(args.hf_repo_id, repo_type="dataset", commit_message="Remove the previous version",
                              operations=[CommitOperationDelete(f"{d}/", is_folder=True) for d in ["data", "meta", "videos"]])
        dataset.push_to_hub(tags=["ManiSkill", *args.env_ids, *robots], private=False)
        print(f"Successfully pushed dataset to https://huggingface.co/datasets/{args.hf_repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["regenerate", "push"])
    parser.add_argument("--env_ids", type=str, nargs="+", required=True, choices=list(TASKS))
    parser.add_argument("--target_dir", type=str, required=True,
                        help="Folder of the regenerated demos, one HDF5 file per task (<env_id>_regenerated.h5)")
    parser.add_argument("--image_resolution", type=int, default=64, help="Rendered image (and video) height and width")
    parser.add_argument("--max_step_error", type=float, default=0.01,
                        help="Drop demos whose recorded action, applied from a recorded state, moves an object "
                             "further than this (m) from its next recorded position")
    parser.add_argument("--max_demos", type=int, default=None, help="Only use the first N demos (for testing)")
    parser.add_argument("--hf_repo_id", type=str, help="Hub repo, e.g. gberseth/maniskill_panda_lerobot")
    parser.add_argument("--lerobot_root", type=str, help="Local directory for the LeRobot dataset (must not exist)")
    parser.add_argument("--no_push", action="store_true", help="Build the LeRobot dataset locally without pushing it")
    args = parser.parse_args()
    if args.step == "regenerate":
        assert len(args.env_ids) == 1, "one task per process: PhysX's GPU mode can only be set up once in a process"
        regenerate(args, args.env_ids[0])
    else:
        push_to_huggingface_lerobot(args)
