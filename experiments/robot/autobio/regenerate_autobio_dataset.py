"""
Generates AutoBio demo datasets (AutoBio: a MuJoCo biology-lab benchmark, arXiv 2505.14030, github.com/autobio-bench/AutoBio,
cloned next to this repo as ../AutoBio) with its scripted experts, checks them in the simulator by replaying their
actions, and exports them to the Hugging Face Hub in the LeRobotDataset v3.0 format with the same features as our LIBERO
and ManiSkill datasets (../libero/regenerate_libero_dataset.py, ../maniskill/regenerate_maniskill_dataset.py), so
mini-grp's training code can mix them.

Why regenerate instead of converting AutoBio's own datasets (autobio-bench/<task>-mujoco on the Hub, 100 demos each):
    - They were recorded with AutoBio's closed binary MuJoCo plugins (libmjlab.so, built for MuJoCo 3.3.0 only), which
      we don't use (see PLUGIN-FREE PHYSICS below): our demos come from the same physics as our sim evals.
    - Their actions are absolute joint-position targets at 50 fps; ours are delta end-effector poses at 20 Hz (the
      experts' own control rate), with no resampling.
    - They have no object states, so their episodes can't be restarted in the sim; ours keep each episode's seed.

PLUGIN-FREE PHYSICS. load_scene reads a copy of AutoBio's MJCF models without its plugins (write_plugin_free_models):
    - mjlab.sdf.thread (the screw caps' threads): the SDF bolt and nut geoms are dropped, and each cap is welded stiffly
      to its tube, so caps can't be unscrewed (only the bimanual screw tasks turn one; they aren't in TASKS).
    - mjlab.passive.detent (the thermal cycler's lid lever, the vortex mixer's switch and knob): dropped with the joint's
      spring, which the detent held off at its stops, and the joint gets friction instead, so it stays where it's pushed.
    AutoBio's experts in this physics (mini-grp outputs/autobio/run_experts.py, 20 episodes each, 2026-11-02): pickup
    20/20, thermal_cycler_close 20/20, thermal_mixer 20/20, insert_centrifuge_5430 16/20, insert 13/20.

The demos:
    - TASKS: AutoBio's 5 single-arm tasks (a UR5e with a Robotiq 2F-85, or one Aloha arm). The bimanual ones (screw
      caps, pipette) need 14-D actions and come later.
    - Recorded every 0.05 s (20 Hz, the experts' control rate) while the expert runs, until the task first succeeds
      (as the LIBERO demos end), and kept only if it does.
    - Two cameras, as in LIBERO: the task's third-person camera (AutoBio's "image") and the wrist camera, rendered by
      AutoBio's Evaluator (with the thermal mixer's display and the liquids drawn).
    - observation.state: the end effector (the gripper's pinch site) pose in the robot base frame, position + axis-angle,
      and the gripper's actuator position twice (LIBERO's layout has two finger joints).
      observation.joint_state: the 6 arm joints and the gripper's actuator position.
    - action: our gold standard, the end effector motion the robot reached in the step (next pose - this pose, in the
      base frame: delta position in m, delta rotation as axis-angle in rad) and the gripper command (-1 open, 1 close).
      A sim eval turns it back into joint targets with ee_delta_to_ctrl (differential IK, chaining each step's target
      from the last), as check_replay does. Replays (check, 2026-11-02): pickup, insert, thermal_cycler_close and
      insert_centrifuge_5430 10/10 each; thermal_mixer 0/10: the replay touches its buttons but never holds one long
      enough to press it (open). Labelling the expert's commanded motion instead (the forward kinematics of its joint
      targets) didn't fix it (0/5) and did no better on the others.

Setup (own uv env; AutoBio's dependencies; MuJoCo as in mini-grp):
    uv venv --python 3.12 --python-preference only-managed .venv_autobio
    uv pip install --python .venv_autobio/bin/python mujoco==3.4.0 numpy scipy "jax[cpu]" toppra trimesh shapely \
        triangle manifold3d sympy zstandard tqdm networkx matplotlib scikit-image h5py "imageio[ffmpeg]"

Usage:
    # 1) Generate the demos (HDF5, one file per task)
    MUJOCO_GL=egl .venv_autobio/bin/python experiments/robot/autobio/regenerate_autobio_dataset.py regenerate \
        --tasks pickup insert --num_demos 100 --target_dir /network/projects/real-g-grp/autobio
    # 2) Replay their actions with the eval's controller and report the success rate
    MUJOCO_GL=egl .venv_autobio/bin/python experiments/robot/autobio/regenerate_autobio_dataset.py check \
        --tasks pickup --target_dir /network/projects/real-g-grp/autobio --max_demos 20
    # 3) Export to LeRobot v3 and push (with the LIBERO env, which has lerobot); several tasks make one dataset
    .venv/bin/python experiments/robot/autobio/regenerate_autobio_dataset.py push --tasks pickup insert ... \
        --target_dir /network/projects/real-g-grp/autobio --lerobot_root /network/projects/real-g-grp/autobio/lerobot \
        --hf_repo_id gberseth/autobio_lerobot
"""

import argparse
import contextlib
import os
import pathlib
import sys
import xml.etree.ElementTree as ET

import numpy as np

FPS = 20  # the experts' control rate, and LIBERO's and ManiSkill's
AUTOBIO = pathlib.Path(os.environ.get("AUTOBIO_DIR", pathlib.Path(__file__).resolve().parents[4] / "AutoBio" / "autobio"))

## AutoBio's single-arm tasks: {name: (module, task class)} in AutoBio's code
TASKS = {"pickup": ("pickup_centrifuge_tube", "Pickup"), "insert": ("transfer_centrifuge_tube", "Insert"),
         "thermal_cycler_close": ("mani_thermal_cycler", "ThermalCyclerManipulate"),
         "thermal_mixer": ("mani_thermal_mixer", "ThermalMixerManipulate"),
         "insert_centrifuge_5430": ("load_centrifuge_5430", "InsertCentrifuge5430")}

## The gripper actuator's open and closed commands, by arm class: the experts' values (the 2F-85's 0-255 closing
## command; the Aloha's finger position in m)
GRIPPER_OPEN_CLOSED = {"UR5eArm": (0.0, 240.0), "AlohaArm": (0.030, 0.013)}

DETENT_FRICTION = 0.05  # N m, enough to hold the thermal cycler's lid lever where it's pushed
## The caps' weld: the default 0.02 s lets the tube swing under its gripped cap and jam in the rack (insert: 6/10 expert
## successes, 13/20 with this); the stiffest that stays stable at the scenes' 2 ms step
WELD_TIMECONST = 0.004


## ---- Plugin-free physics ----

def strip_plugins(src, dst):
    """ Writes MJCF file src to dst without AutoBio's plugins (see PLUGIN-FREE PHYSICS above). """
    text = src.read_text()
    root = ET.fromstring(text[text.index("?>") + 2:] if text.startswith("<?xml") else text)  # cryovial_1-8ml.xml's declaration has a typo
    for parent in list(root.iter()):
        for child in list(parent):
            if child.tag == "extension" and child.find("plugin") is not None:
                parent.remove(child)
            elif child.tag in ("mesh", "geom") and child.find("plugin") is not None:  # the threads' SDF meshes and geoms
                parent.remove(child)
            elif child.tag == "plugin" and child.get("instance") and parent.tag == "body":  # a detent
                parent.remove(child)
                for joint in parent.findall("joint"):
                    joint.attrib.pop("springdamper", None)
                    joint.set("frictionloss", str(DETENT_FRICTION))
    for contact in root.iter("contact"):  # the contact pairs of the dropped thread geoms (e.g. "1/nut")
        for pair in list(contact.findall("pair")):
            if {pair.get("geom1", "").split("/")[-1], pair.get("geom2", "").split("/")[-1]} & {"nut", "bolt"}:
                contact.remove(pair)
    dst.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(dst)


def write_plugin_free_models():
    for src in (AUTOBIO / "model").rglob("*.xml"):
        strip_plugins(src, AUTOBIO / "model_noplugin" / src.relative_to(AUTOBIO / "model"))


def load_scene(name):
    """ The MjSpec of AutoBio scene name, from the plugin-free copy, with each screw cap welded to its tube. """
    import mujoco
    spec = mujoco.MjSpec.from_file(str(AUTOBIO / "model_noplugin" / "scene" / f"{name}.xml"))
    caps = [b for b in spec.bodies if b.name.endswith("screw_cap") or b.name.endswith("cryovial_cap")]
    tubes = [b for b in spec.bodies if b.name.endswith("screw_body") or b.name.endswith("cryovial_body")]
    for cap in caps:  # the scenes attach each object's cap and tube in order (cap 1/, tube 2/; cap 3/, tube 4/, ...)
        kind = cap.name.split("/")[-1].removesuffix("_cap")
        tube = next((t for t in tubes if t.name.split("/")[-1].removesuffix("_body") == kind), None)
        if tube is not None:
            tubes.remove(tube)
            ## data all zeros but the torque scale: the weld keeps the pose qpos0 has (each tube assembled)
            spec.add_equality(type=mujoco.mjtEq.mjEQ_WELD, objtype=mujoco.mjtObj.mjOBJ_BODY, name1=cap.name,
                              name2=tube.name, data=[0] * 10 + [1], solref=[WELD_TIMECONST, 1])
    return spec


def make_task(name, expert=True):
    """ AutoBio's task (or its scripted expert) in the plugin-free scene. AutoBio's task files load its plugins when
    imported and read files relative to its folder, so this works from there with the plugin loader turned off. """
    import mujoco
    mujoco.mj_loadPluginLibrary = lambda path: None
    os.chdir(AUTOBIO)
    if str(AUTOBIO) not in sys.path:
        sys.path.insert(0, str(AUTOBIO))
    if not (AUTOBIO / "model_noplugin").exists():
        write_plugin_free_models()
    module, cls_name = TASKS[name]
    cls = getattr(__import__(module), cls_name)
    task = (cls.Expert if expert else cls)(cls.prepare(load_scene(cls.default_scene.stem)))
    task.serializer = NoSerializer()
    return task


class NoSerializer:
    """ Stands in for AutoBio's trajectory logger, which some experts (the thermal mixer's) call without checking. """
    def record(self, info): pass
    def finish(self): pass
    def within_save_dir(self): return contextlib.nullcontext()


def succeeded(task):
    return float(task.check()) >= 1.0  # the thermal mixer's check is a score in [0, 1]


## ---- The robot's state and the action space ----

def base_frame(task):
    data, body = task.data, task.model.body(task.arm.base_name).id
    return data.xpos[body], data.xmat[body].reshape(3, 3)


def ee_pose(task):
    """ The end effector's position and rotation matrix in the robot base frame. """
    data, arm = task.data, task.arm
    base_pos, base_rot = base_frame(task)
    return base_rot.T @ (data.site_xpos[arm.site_id] - base_pos), base_rot.T @ data.site_xmat[arm.site_id].reshape(3, 3)


def gripper_command(task):
    """ The gripper's current command: -1 open, 1 close (the nearer of the expert's two values). """
    open_, closed = GRIPPER_OPEN_CLOSED[type(task.arm).__name__]
    ctrl = task.data.ctrl[task.arm.gripper_id]
    return 1.0 if abs(ctrl - closed) < abs(ctrl - open_) else -1.0


def ee_delta_to_ctrl(task, action, target=None, iterations=5, damping=1e-3):
    """ Sets the arm's joint targets (and the gripper's) so the end effector moves by action (delta position, delta
    axis-angle rotation, both in the base frame; gripper -1 open / 1 close): damped least squares differential IK on
    the arm's joints, from their current positions. The motion is added to the previous step's target pose (target,
    as this returns it; None at an episode's start: the end effector's pose), not to the pose reached: the arm's
    position servos lag their targets, and adding each step's motion to the lagging pose compounds the lag (pickup's
    replays drift 85 mm by the grasp and all fail). Returns the new target pose (position, rotation matrix). """
    import mujoco
    from scipy.spatial.transform import Rotation
    model, arm = task.model, task.arm
    pos, rot = ee_pose(task) if target is None else target
    target_pos = pos + action[:3]
    target_rot = Rotation.from_rotvec(action[3:6]).as_matrix() @ rot
    base_pos, base_rot = base_frame(task)
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = task.data.qpos
    joints = list(arm.jnt_span)  # the arm's hinge joints' qpos addresses, and their dofs
    dofs = [model.jnt_dofadr[j] for j in range(model.njnt) if model.jnt_qposadr[j] in joints]
    jac_pos, jac_rot = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    for _ in range(iterations):
        mujoco.mj_kinematics(model, scratch)
        mujoco.mj_comPos(model, scratch)
        cur_pos = base_rot.T @ (scratch.site_xpos[arm.site_id] - base_pos)
        cur_rot = base_rot.T @ scratch.site_xmat[arm.site_id].reshape(3, 3)
        error = np.concatenate([base_rot @ (target_pos - cur_pos),  # in the world frame, as the Jacobian
                                base_rot @ Rotation.from_matrix(target_rot @ cur_rot.T).as_rotvec()])
        mujoco.mj_jacSite(model, scratch, jac_pos, jac_rot, arm.site_id)
        jac = np.concatenate([jac_pos, jac_rot])[:, dofs]
        scratch.qpos[joints] += jac.T @ np.linalg.solve(jac @ jac.T + damping * np.eye(6), error)
    task.data.ctrl[list(arm.act_span)] = scratch.qpos[joints]
    open_, closed = GRIPPER_OPEN_CLOSED[type(arm).__name__]
    task.data.ctrl[arm.gripper_id] = closed if action[6] > 0 else open_
    return target_pos, target_rot


## ---- 1) Generate ----

def record_demo(task, renderer, seed):
    """ Runs the expert from reset(seed), recording every 1 / FPS s until the task first succeeds. Returns the demo's
    arrays, or None if it never succeeds. """
    from scipy.spatial.transform import Rotation
    task.reset(seed)
    task.serializer = NoSerializer()
    renderer.reset()
    steps_per_frame = int(round(1 / FPS / task.dt))
    rec = {k: [] for k in ["agentview_rgb", "wrist_rgb", "ee_pos", "ee_rot", "gripper", "joint_states"]}

    class Done(Exception): pass

    def record():
        images = renderer.get_images()
        pos, rot = ee_pose(task)
        gripper_pos = task.data.actuator_length[task.arm.gripper_id]
        rec["agentview_rgb"].append(images["observation/image"])
        rec["wrist_rgb"].append(images["observation/wrist_image"])
        rec["ee_pos"].append(pos)
        rec["ee_rot"].append(Rotation.from_matrix(rot).as_rotvec())
        rec["gripper"].append(gripper_command(task))
        rec["joint_states"].append(np.append(task.data.qpos[list(task.arm.jnt_span)], gripper_pos))
        if succeeded(task):
            raise Done

    step = type(task).step_and_log
    def step_and_record(info):
        if round(task.data.time / task.dt) % steps_per_frame == 0:
            record()
        step(task, info)
    task.step_and_log = step_and_record
    try:
        task.execute()
        return None  # the expert finished without success
    except Done:
        pass
    except Exception as e:
        print(f"seed {seed}: {type(e).__name__}: {str(e)[:150]}")
        return None
    finally:
        del task.step_and_log
        renderer.render_finish()  # the thermal mixer's display figures
    demo = {k: np.stack(v) for k, v in rec.items()}
    ## the action of each frame: the end effector motion it reached by the next frame (the last frame repeats its
    ## predecessor's motion), and the gripper command in effect from it
    rots = Rotation.from_rotvec(demo["ee_rot"])
    motion = np.concatenate([np.diff(demo["ee_pos"], axis=0), (rots[1:] * rots[:-1].inv()).as_rotvec()], axis=1)
    motion = np.concatenate([motion, motion[-1:]])
    demo["actions"] = np.concatenate([motion, demo["gripper"][:, None]], axis=1).astype(np.float32)
    demo["ee_states"] = np.concatenate([demo["ee_pos"], demo["ee_rot"], demo["joint_states"][:, -1:].repeat(2, axis=1)],
                                       axis=1).astype(np.float32)
    return demo


def regenerate(args, name):
    import h5py, tqdm
    task = make_task(name)
    from evaluator import Evaluator  # AutoBio's (on the path from make_task): the cameras, with the displays and liquids
    renderer = Evaluator(task, image_height=args.image_resolution, image_width=args.image_resolution)
    target_file = os.path.join(args.target_dir, f"{name}_regenerated.h5")
    os.makedirs(args.target_dir, exist_ok=True)
    kept, seed = 0, 0
    with h5py.File(target_file + ".tmp", "w") as dst, tqdm.tqdm(total=args.num_demos, desc=name, mininterval=30) as bar:
        grp = dst.create_group("data")
        grp.attrs["robot"] = type(task.arm).__name__
        grp.attrs["control_hz"] = FPS
        while kept < args.num_demos and seed < 3 * args.num_demos:
            demo = record_demo(task, renderer, seed)
            if demo is not None:
                group = grp.create_group(f"demo_{kept}")
                for k, v in demo.items():
                    group.create_dataset(k, data=v, compression="gzip" if v.dtype == np.uint8 else None)
                group.attrs["seed"] = seed  # task.reset(seed) restarts the episode
                group.attrs["language_instruction"] = task.task_info["prefix"]  # insert's names its target slot
                kept += 1
                bar.update(1)
            seed += 1
    os.replace(target_file + ".tmp", target_file)
    print(f"{name}: kept {kept}/{seed} expert demos -> {target_file}")


## ---- 2) Check: replay the actions with the eval's controller ----

def check_replay(args, name):
    """ Replays each demo's actions open loop through ee_delta_to_ctrl, one per 1 / FPS s, from the demo's seed, and
    reports how many succeed within the demo's length plus 2 s. """
    import h5py
    task = make_task(name, expert=False)
    steps_per_frame = int(round(1 / FPS / task.dt))
    successes, demos = 0, 0
    with h5py.File(os.path.join(args.target_dir, f"{name}_regenerated.h5"), "r") as f:
        for key in sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))[:args.max_demos]:
            actions = f["data"][key]["actions"][()]
            task.reset(int(f["data"][key].attrs["seed"]))
            ok, target = False, None
            for action in list(actions) + [np.append(np.zeros(6), actions[-1, 6])] * 2 * FPS:
                target = ee_delta_to_ctrl(task, action, target)
                for _ in range(steps_per_frame):
                    task.step()
                if succeeded(task):
                    ok = True
                    break
            successes += ok
            demos += 1
    print(f"{name}: {successes}/{demos} replays succeed with ee_delta_to_ctrl")


## ---- 3) Push ----

def autobio_features(res):
    ## The same features as the LIBERO and ManiSkill datasets
    return {
        "observation.images.image": {"dtype": "video", "shape": (res, res, 3), "names": ["height", "width", "channels"]},
        "observation.images.wrist_image": {"dtype": "video", "shape": (res, res, 3), "names": ["height", "width", "channels"]},
        "observation.state": {  # End effector position + axis-angle orientation (base frame) + the gripper's position twice
            "dtype": "float32", "shape": (8,),
            "names": ["x", "y", "z", "axis_angle1", "axis_angle2", "axis_angle3", "gripper_left", "gripper_right"]},
        "observation.joint_state": {  # The 6 arm joints and the gripper's actuator position
            "dtype": "float32", "shape": (7,), "names": [f"joint_{i}" for i in range(7)]},
        "action": {  # Delta end effector pose (base frame, m and axis-angle rad) + gripper (-1 open, 1 close)
            "dtype": "float32", "shape": (7,),
            "names": ["x", "y", "z", "axis_angle1", "axis_angle2", "axis_angle3", "gripper"]},
    }


def push_to_huggingface_lerobot(args):
    """ Converts the generated HDF5 files of args.tasks into one LeRobotDataset v3.0 and pushes it to the Hub. """
    import h5py, pandas as pd, tqdm
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    files = [h5py.File(os.path.join(args.target_dir, f"{name}_regenerated.h5"), "r") for name in args.tasks]
    res = next(iter(files[0]["data"].values()))["agentview_rgb"].shape[1]
    robots = sorted({f["data"].attrs["robot"] for f in files})
    dataset = LeRobotDataset.create(repo_id=args.hf_repo_id, root=args.lerobot_root, fps=FPS,
                                    features=autobio_features(res), robot_type="+".join(robots))
    dataset.meta.update_chunk_settings(video_files_size_in_mb=args.video_files_size_in_mb)
    episodes = []
    for name, f in zip(args.tasks, files):
        grp = f["data"]
        for key in tqdm.tqdm(sorted(grp.keys(), key=lambda k: int(k.split("_")[1])), desc=f"Exporting {name}"):
            demo = {k: v[()] for k, v in grp[key].items()}
            for j in range(len(demo["actions"])):
                dataset.add_frame({
                    "task": grp[key].attrs["language_instruction"],
                    "observation.images.image": demo["agentview_rgb"][j],
                    "observation.images.wrist_image": demo["wrist_rgb"][j],
                    "observation.state": demo["ee_states"][j],
                    "observation.joint_state": demo["joint_states"][j].astype(np.float32),
                    "action": demo["actions"][j],
                })
            dataset.save_episode()
            episodes.append({"episode_index": len(episodes), "task_name": name, "demo": key,
                             "seed": int(grp[key].attrs["seed"])})
    dataset.finalize()
    ## each episode's AutoBio task and reset seed, which an evaluation or replay needs: make_task(name).reset(seed)
    pd.DataFrame(episodes).to_parquet(os.path.join(dataset.root, "meta", "autobio_episodes.parquet"))
    print(f"Created dataset with {dataset.meta.total_episodes} episodes and {dataset.meta.total_frames} frames")
    if not args.no_push:
        dataset.push_to_hub(tags=["AutoBio", *args.tasks, *robots], private=False)
        print(f"Successfully pushed dataset to https://huggingface.co/datasets/{args.hf_repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["regenerate", "check", "push"])
    parser.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    parser.add_argument("--target_dir", default="/network/projects/real-g-grp/autobio")
    parser.add_argument("--num_demos", type=int, default=100, help="successful demos per task")
    parser.add_argument("--max_demos", type=int, default=None, help="check: replay at most this many demos per task")
    parser.add_argument("--image_resolution", type=int, default=128)
    parser.add_argument("--lerobot_root", default="/network/projects/real-g-grp/autobio/lerobot")
    parser.add_argument("--hf_repo_id", default="gberseth/autobio_lerobot")
    parser.add_argument("--video_files_size_in_mb", type=int, default=500)
    parser.add_argument("--no_push", action="store_true")
    args = parser.parse_args()
    ## make_task changes the working directory to AutoBio's
    args.target_dir, args.lerobot_root = os.path.abspath(args.target_dir), os.path.abspath(args.lerobot_root)
    if args.command == "push":
        push_to_huggingface_lerobot(args)
    else:
        for name in args.tasks:
            (regenerate if args.command == "regenerate" else check_replay)(args, name)
