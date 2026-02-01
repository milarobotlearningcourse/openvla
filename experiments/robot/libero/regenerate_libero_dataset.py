"""
Regenerates a LIBERO dataset (HDF5 files) by replaying demonstrations in the environments.

Notes:
    - We save image observations at 256x256px resolution (instead of 128x128).
    - We filter out transitions with "no-op" (zero) actions that do not change the robot's state.
    - We filter out unsuccessful demonstrations.
    - In the LIBERO HDF5 data -> RLDS data conversion (not shown here), we rotate the images by
    180 degrees because we observe that the environments return images that are upside down
    on our platform.

Usage:
    python experiments/robot/libero/regenerate_libero_dataset.py \
        --libero_task_suite [ libero_spatial | libero_object | libero_goal | libero_10 ] \
        --libero_raw_data_dir <PATH TO RAW HDF5 DATASET DIR> \
        --libero_target_dir <PATH TO TARGET DIR>

    Example (LIBERO-Spatial):
        python experiments/robot/libero/regenerate_libero_dataset.py \
            --libero_task_suite libero_spatial \
            --libero_raw_data_dir ./LIBERO/libero/datasets/libero_spatial \
            --libero_target_dir ./LIBERO/libero/datasets/libero_spatial_no_noops

"""

import argparse
import json
import os
import time

import h5py
import numpy as np
import robosuite.utils.transform_utils as T
import tqdm
from libero.libero import benchmark
from datasets import Dataset, DatasetDict, Image, Features, Sequence, Value, Array2D, Array3D

from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
)


IMAGE_RESOLUTION = 64


def is_noop(action, prev_action=None, threshold=1e-4):
    """
    Returns whether an action is a no-op action.

    A no-op action satisfies two criteria:
        (1) All action dimensions, except for the last one (gripper action), are near zero.
        (2) The gripper action is equal to the previous timestep's gripper action.

    Explanation of (2):
        Naively filtering out actions with just criterion (1) is not good because you will
        remove actions where the robot is staying still but opening/closing its gripper.
        So you also need to consider the current state (by checking the previous timestep's
        gripper action as a proxy) to determine whether the action really is a no-op.
    """
    # Special case: Previous action is None if this is the first action in the episode
    # Then we only care about criterion (1)
    if prev_action is None:
        return np.linalg.norm(action[:-1]) < threshold

    # Normal case: Check both criteria (1) and (2)
    gripper_action = action[-1]
    prev_gripper_action = prev_action[-1]
    return np.linalg.norm(action[:-1]) < threshold and gripper_action == prev_gripper_action


def main(args):
    print(f"Regenerating {args.libero_task_suite} dataset!")

    # Create target directory
    if os.path.isdir(args.libero_target_dir):
        user_input = input(f"Target directory already exists at path: {args.libero_target_dir}\nEnter 'y' to overwrite the directory, or anything else to exit: ")
        if user_input != 'y':
            exit()
    os.makedirs(args.libero_target_dir, exist_ok=True)

    # Prepare JSON file to record success/false and initial states per episode
    metainfo_json_dict = {}
    metainfo_json_out_path = f"./openvla/experiments/robot/libero/{args.libero_task_suite}_metainfo.json"
    with open(metainfo_json_out_path, "w") as f:
        # Just test that we can write to this file (we overwrite it later)
        json.dump(metainfo_json_dict, f)
    
    # Prepare HDF5 file to record initial states and goal images indexed by demo number
    init_states_hdf5_path = f"./openvla/experiments/robot/libero/{args.libero_task_suite}_init_states_and_goals.hdf5"
    init_states_hdf5_file = h5py.File(init_states_hdf5_path, "w")

    # Get task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.libero_task_suite]()
    num_tasks_in_suite = task_suite.n_tasks

    # Setup
    num_replays = 0
    num_success = 0
    num_noops = 0

    for task_id in tqdm.tqdm(range(1)):
    # for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task in suite
        task = task_suite.get_task(task_id)
        env, task_description = get_libero_env(task, "llava", resolution=IMAGE_RESOLUTION)

        # Get dataset for task
        orig_data_path = os.path.join(args.libero_raw_data_dir, f"{task.name}_demo.hdf5")
        assert os.path.exists(orig_data_path), f"Cannot find raw data file {orig_data_path}."
        orig_data_file = h5py.File(orig_data_path, "r")
        orig_data = orig_data_file["data"]

        # Create new HDF5 file for regenerated demos
        new_data_path = os.path.join(args.libero_target_dir, f"{task.name}_demo.hdf5")
        new_data_file = h5py.File(new_data_path, "w")
        grp = new_data_file.create_group("data")

        # for i in range(len(orig_data.keys())):
        for i in range(1):
            # Get demo data
            demo_data = orig_data[f"demo_{i}"]
            orig_actions = demo_data["actions"][()]
            orig_states = demo_data["states"][()]

            # Reset environment, set initial state, and wait a few steps for environment to settle
            env.reset()
            env.set_init_state(orig_states[0])
            for _ in range(10):
                obs, reward, done, info = env.step(get_libero_dummy_action("llava"))

            # Set up new data lists
            states = []
            actions = []
            ee_states = []
            gripper_states = []
            joint_states = []
            robot_states = []
            # robot_pose = []
            agentview_images = []
            eye_in_hand_images = []

            # Replay original demo actions in environment and record observations
            for _, action in enumerate(orig_actions):
                # Skip transitions with no-op actions
                prev_action = actions[-1] if len(actions) > 0 else None
                if is_noop(action, prev_action):
                    print(f"\tSkipping no-op action: {action}")
                    num_noops += 1
                    continue

                if states == []:
                    # In the first timestep, since we're using the original initial state to initialize the environment,
                    # copy the initial state (first state in episode) over from the original HDF5 to the new one
                    states.append(orig_states[0])
                    robot_states.append(demo_data["robot_states"][0])
                else:
                    # For all other timesteps, get state from environment and record it
                    states.append(env.sim.get_state().flatten())
                    robot_states.append(
                        np.concatenate([obs["robot0_gripper_qpos"], obs["robot0_eef_pos"], obs["robot0_eef_quat"]])
                    )
                # robot_pose.append(np.concatenate([obs["robot0_eef_pos"], obs["robot0_eef_quat"], obs["robot0_gripper_qpos"]]))

                # Record original action (from demo)
                actions.append(action)

                # Record data returned by environment
                if "robot0_gripper_qpos" in obs:
                    gripper_states.append(obs["robot0_gripper_qpos"])
                joint_states.append(obs["robot0_joint_pos"])
                ee_states.append(
                    np.hstack(
                        (
                            obs["robot0_eef_pos"],
                            T.quat2axisangle(obs["robot0_eef_quat"]),
                        )
                    )
                )
                ## Rotate images by 180 degrees to account for upside-down rendering
                obs["agentview_image"] = np.rot90(obs["agentview_image"], 2)
                obs["robot0_eye_in_hand_image"] = np.rot90(obs["robot0_eye_in_hand_image"], 2)
                agentview_images.append(obs["agentview_image"])
                eye_in_hand_images.append(obs["robot0_eye_in_hand_image"])

                # Execute demo action in environment
                obs, reward, done, info = env.step(action.tolist())

            # At end of episode, save replayed trajectories to new HDF5 files (only keep successes)
            if done:
                dones = np.zeros(len(actions)).astype(np.uint8)
                dones[-1] = 1
                rewards = np.zeros(len(actions)).astype(np.uint8)
                rewards[-1] = 1
                assert len(actions) == len(agentview_images)

                ep_data_grp = grp.create_group(f"demo_{i}")
                obs_grp = ep_data_grp.create_group("obs")
                obs_grp.create_dataset("gripper_states", data=np.stack(gripper_states, axis=0))
                obs_grp.create_dataset("joint_states", data=np.stack(joint_states, axis=0))
                obs_grp.create_dataset("ee_states", data=np.stack(ee_states, axis=0))
                obs_grp.create_dataset("ee_pos", data=np.stack(ee_states, axis=0)[:, :3])
                obs_grp.create_dataset("ee_ori", data=np.stack(ee_states, axis=0)[:, 3:])
                obs_grp.create_dataset("agentview_rgb", data=np.stack(agentview_images, axis=0))
                obs_grp.create_dataset("eye_in_hand_rgb", data=np.stack(eye_in_hand_images, axis=0))
                ep_data_grp.create_dataset("actions", data=actions)
                ep_data_grp.create_dataset("states", data=np.stack(states))
                ep_data_grp.create_dataset("robot_states", data=np.stack(robot_states, axis=0))
                ep_data_grp.create_dataset("rewards", data=rewards)
                ep_data_grp.create_dataset("dones", data=dones)
                ep_data_grp.create_dataset("init_states", data=np.array(orig_states[0]))
                # import os
                path_ = os.path.join("./", f"libero-{0}-task-id-{task_id}-init-id-{i}.mp4")
                import imageio
                imageio.mimsave(path_, agentview_images, fps=20)

                num_success += 1

            num_replays += 1

            # Record success/false and initial environment state in metainfo dict
            task_key = task_description.replace(" ", "_")
            episode_key = f"demo_{i}"
            if task_key not in metainfo_json_dict:
                metainfo_json_dict[task_key] = {}
            if episode_key not in metainfo_json_dict[task_key]:
                metainfo_json_dict[task_key][episode_key] = {}
            metainfo_json_dict[task_key][episode_key]["success"] = bool(done)
            metainfo_json_dict[task_key][episode_key]["initial_state"] = orig_states[0].tolist()
            
            # Record initial state and goal image in HDF5 file
            if task_key not in init_states_hdf5_file:
                task_grp = init_states_hdf5_file.create_group(task_key)
            else:
                task_grp = init_states_hdf5_file[task_key]
            
            demo_grp = task_grp.create_group(episode_key)
            demo_grp.create_dataset("init_state", data=orig_states[0])
            if len(agentview_images) > 0:
                demo_grp.create_dataset("goal_img", data=agentview_images[-1])

            # Write metainfo dict to JSON file
            # (We repeatedly overwrite, rather than doing this once at the end, just in case the script crashes midway)
            with open(metainfo_json_out_path, "w") as f:
                json.dump(metainfo_json_dict, f, indent=2)

            # Count total number of successful replays so far
            print(
                f"Total # episodes replayed: {num_replays}, Total # successes: {num_success} ({num_success / num_replays * 100:.1f} %)"
            )

            # Report total number of no-op actions filtered out so far
            print(f"  Total # no-op actions filtered out: {num_noops}")

        # Close HDF5 files
        orig_data_file.close()
        new_data_file.close()
        print(f"Saved regenerated demos for task '{task_description}' at: {new_data_path}")

    # Close the init_states HDF5 file
    init_states_hdf5_file.close()
    
    print(f"Dataset regeneration complete! Saved new dataset at: {args.libero_target_dir}")
    print(f"Saved metainfo JSON at: {metainfo_json_out_path}")
    print(f"Saved init_states HDF5 at: {init_states_hdf5_path}")


def push_to_huggingface(dataset_dir, repo_id, task_suite_name):
    """
    Push regenerated LIBERO dataset to Hugging Face Hub.
    
    Args:
        dataset_dir: Path to the directory containing regenerated HDF5 files
        repo_id: Hugging Face repository ID (e.g., 'gberseth/libero_spatial')
        task_suite_name: Name of the task suite (e.g., 'libero_spatial')
    """
    from PIL import Image
    print(f"Preparing to push dataset to Hugging Face: {repo_id}")
    
    # Collect all HDF5 files in the dataset directory
    hdf5_files = [f for f in os.listdir(dataset_dir) if f.endswith('.hdf5')]
    print(f"Found {len(hdf5_files)} HDF5 files")
    
    all_episodes = []
    
    for hdf5_file in tqdm.tqdm(hdf5_files, desc="Processing HDF5 files"):
        task_name = hdf5_file.replace('_demo.hdf5', '')
        file_path = os.path.join(dataset_dir, hdf5_file)
        
        with h5py.File(file_path, 'r') as f:
            data_grp = f['data']
            num_demos = len(data_grp.keys())
            
            for demo_idx in range(num_demos):
                demo_key = f"demo_{demo_idx}"
                if demo_key not in data_grp:
                    continue
                    
                demo = data_grp[demo_key]
                obs_grp = demo['obs']
                
                # Convert images from numpy arrays to list of PIL Images
                agentview_images = obs_grp['agentview_rgb'][()]
                eye_in_hand_images = obs_grp['eye_in_hand_rgb'][()]

                for j in range(agentview_images.shape[0]):
                    action = demo['actions'][j]
                    # action[6] = ((action[6] + 1.0) / 2.0) * -1.0  # Convert gripper action to [0, 1] range and invert
                    _data = {
                        'goal_text_full': task_name,
                        'task_suite': task_suite_name,
                        'demo_id': demo_idx,
                        'episode_length': len(demo['actions']),
                        # Observations
                        'gripper_states': obs_grp['gripper_states'][j],
                        'joint_states': obs_grp['joint_states'][j],
                        'ee_states': obs_grp['ee_states'][j],
                        'ee_pos': obs_grp['ee_pos'][j],
                        'ee_ori': obs_grp['ee_ori'][j],
                        'img':  Image.fromarray(agentview_images[j].astype(np.uint8), mode='RGB'),
                        'eye_in_hand_rgb': Image.fromarray(eye_in_hand_images[j].astype(np.uint8), mode='RGB'),
                        'goal_img': Image.fromarray(agentview_images[-1].astype(np.uint8), mode='RGB'),
                        # Actions and states
                        'action': action,
                        'states': demo['states'][j],
                        'pose': demo['robot_states'][j],
                        'rewards': demo['rewards'][j],
                        'terminated': demo['dones'][j],
                        'init_state': np.array(demo['init_states']),
                    }
                    # episode_data.append(_data)
                
                    all_episodes.append(_data)
    
    print(f"Collected {len(all_episodes)} episodes total")
    
    # Define features for the dataset
    # features = Features({
    #     'task_name': Value('string'),
    #     'task_suite': Value('string'),
    #     'demo_id': Value('int32'),
    #     'episode_length': Value('int32'),
    #     'gripper_states': Sequence(Value('float32')),
    #     'joint_states': Sequence(Value('float32')),
    #     'ee_states': Sequence(Value('float32')),
    #     'ee_pos': Sequence(Value('float32')),
    #     'ee_ori': Sequence(Value('float32')),
    #     'img': Image(),
    #     'eye_in_hand_rgb': Image(),
    #     'goal_img': Image(),
    #     'actions': Sequence(Value('float32')),
    #     'states': Sequence(Value('float32')),
    #     'robot_states': Sequence(Value('float32')),
    #     'rewards': Value('uint8'),
    #     'dones': Value('uint8'),
    #     'init_states': Value('float32'),
    # })
    
    # Create dataset
    dataset = Dataset.from_list(all_episodes)
    
    print(f"Created dataset with {len(dataset)} episodes")
    print(f"Dataset features: {dataset.features}")
    
    # Push to hub
    print(f"Pushing dataset to Hugging Face Hub: {repo_id}")
    dataset.push_to_hub(repo_id, private=False)
    
    print(f"Successfully pushed dataset to {repo_id}")
    return dataset


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--libero_task_suite", type=str, choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"],
                        help="LIBERO task suite. Example: libero_spatial", required=True)
    parser.add_argument("--libero_raw_data_dir", type=str,
                        help="Path to directory containing raw HDF5 dataset. Example: ./LIBERO/libero/datasets/libero_spatial", required=True)
    parser.add_argument("--libero_target_dir", type=str,
                        help="Path to regenerated dataset directory. Example: ./LIBERO/libero/datasets/libero_spatial_no_noops", required=True)
    parser.add_argument("--push_to_hub", action="store_true",
                        help="Push the regenerated dataset to Hugging Face Hub")
    parser.add_argument("--hf_repo_id", type=str, default="gberseth/libero_spatial",
                        help="Hugging Face repository ID. Example: gberseth/libero_spatial")
    args = parser.parse_args()

    # Start data regeneration
    main(args)
    
    # Optionally push to Hugging Face
    if args.push_to_hub:
        push_to_huggingface(args.libero_target_dir, args.hf_repo_id, args.libero_task_suite)
