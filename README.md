# OpenVLA: An Open-Source Vision-Language-Action Model

[![arXiv](https://img.shields.io/badge/arXiv-2406.09246-df2a2a.svg?style=for-the-badge)](https://arxiv.org/abs/2406.09246)
[![HF Models](https://img.shields.io/badge/%F0%9F%A4%97-Models-yellow?style=for-the-badge)](https://huggingface.co/openvla/openvla-7b)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.0-EE4C2C.svg?style=for-the-badge&logo=pytorch)](https://pytorch.org/get-started/locally/)
[![Python](https://img.shields.io/badge/python-3.10-blue?style=for-the-badge)](https://www.python.org)
[![License](https://img.shields.io/github/license/TRI-ML/prismatic-vlms?style=for-the-badge)](LICENSE)
 
[**Regenerating the robot datasets**](#regenerating-the-robot-datasets-lerobot-exports-for-mini-grp) | [**Getting Started**](#getting-started) | [**Pretrained VLAs**](#pretrained-vlas) | [**Installation**](#installation) | [**Fine-Tuning OpenVLA via LoRA**](#fine-tuning-openvla-via-lora) | [**Fully Fine-Tuning OpenVLA**](#fully-fine-tuning-openvla) |
[**Training VLAs from Scratch**](#training-vlas-from-scratch) | [**Evaluating OpenVLA**](#evaluating-openvla) | [**Project Website**](https://openvla.github.io/)


<hr style="border: 2px solid gray;"></hr>

## Latest Updates
- [2025-03-03] OFT (Optimized Fine-Tuning recipe for VLAs) was recently released! Compared to vanilla OpenVLA fine-tuning, OFT enables 25-50x faster inference speed, higher task success rates, multiple input images, and high-frequency bimanual robot control. Unlike FAST, OFT uses continuous actions for greater model quality. See project website [here](https://openvla-oft.github.io/).
- [2025-01-16] The FAST action tokenizer was recently released! Compared to vanilla OpenVLA-style 256-bin action discretization, FAST allows action chunks to be compressed into fewer tokens, speeding up inference by up to 15x when using discrete robot actions. See project website [here](https://www.physicalintelligence.company/research/fast).
- [2024-10-15] Added a [VLA Performance Troubleshooting](#vla-performance-troubleshooting) section to the README with best practices for debugging poor VLA performance after fine-tuning.
- [2024-09-04] Added LIBERO simulation benchmark fine-tuning experiments to paper (see v2 on [arXiv](https://arxiv.org/abs/2406.09246));
  added instructions for reproducing OpenVLA results in [LIBERO Simulation Benchmark Evaluations](#libero-simulation-benchmark-evaluations) section
- [2024-08-14] Added new section, [Evaluating OpenVLA](#evaluating-openvla), with instructions for running BridgeData V2 WidowX robot evals
- [2024-07-08] Added new sections: [Fine-Tuning OpenVLA via LoRA](#fine-tuning-openvla-via-lora), [Fully Fine-Tuning OpenVLA](#fully-fine-tuning-openvla)
- [2024-06-13] Initial release

<hr style="border: 2px solid gray;"></hr>

## Regenerating the robot datasets (LeRobot exports for mini-grp)

This fork also builds the robot datasets that [mini-grp](../mini-grp) trains on (`micro-vlm/mini_fm.py`, which loads
them with `load_lerobots` in `mini-grp/data_loading.py`). Each one is a [LeRobotDataset v3.0](https://huggingface.co/docs/lerobot)
on the Hugging Face Hub, with the same features, so they can be mixed in one training run:

| Feature | Shape | Contents |
|---|---|---|
| `observation.images.image` | 64×64×3 video | agent (third-person) view |
| `observation.images.wrist_image` | 64×64×3 video | wrist camera |
| `observation.state` | 8 | end-effector position + axis-angle orientation + 2 gripper finger positions |
| `observation.joint_state` | 7 | arm joint positions (mini-grp's pose input) |
| `action` | 7 | end-effector delta position + delta axis-angle + gripper (−1 open, 1 close), at 20 Hz |

Only demos that still succeed when checked in the simulator are kept, and each episode ends at its first success
(LIBERO and ManiSkill; Bridge is real robot data, so it can't be checked in a simulator).
Each dataset also stores what a replay or an eval needs to restart an episode in its simulator: `meta/libero_init_states.parquet`
(LIBERO) or `meta/maniskill_episodes.parquet` (ManiSkill).

| Hub dataset | Source | Episodes | Frames |
|---|---|---|---|
| `gberseth/libero_spatial_no_noops_lerobot` | LIBERO-Spatial | 443 | 54,442 |
| `gberseth/libero_object_no_noops_lerobot` | LIBERO-Object | 464 | 68,388 |
| `gberseth/libero_goal_no_noops_lerobot` | LIBERO-Goal | 431 | 52,869 |
| `gberseth/libero_10_no_noops_lerobot` | LIBERO-10 | 382 | 101,850 |
| `gberseth/libero_90_no_noops_lerobot` | LIBERO-90 | 3,933 | 571,207 |
| `gberseth/maniskill_pusht_lerobot` | ManiSkill `PushT-v1` | 706 | 22,355 |
| `gberseth/bridge_orig_lerobot` | BridgeData V2 (real WidowX) | 38,660 | 1,305,714 |

Bridge (real WidowX data) differs from this layout in three ways: the export has no wrist camera (Bridge's is in only
13% of its episodes), its `observation.joint_state` is zeros, and it runs at 5 Hz.

**Dataset details.** For the details of every Open X-Embodiment dataset (robot, episodes, action space, cameras,
language, collection method, control frequency), see the project's [dataset sheet](https://docs.google.com/spreadsheets/d/1rPBD77tk60AEIGZrGSODwyyzs5FgCU9Uz3h-3_t2A9g), the reference to check
before adding one. mini-grp's `log/dataset_notes.md` has a copy of that table (with which datasets are on the cluster)
and notes on the issues of the datasets we use.

**Where files go.** All raw and intermediate data is in the project folder `/network/projects/real-g-grp/`, not `$HOME`,
which is nearly full:
- `libero/<suite>/`: the raw LIBERO HDF5 files;
- `libero/<suite>_no_noops/`: the regenerated HDF5 files;
- `libero/lerobot/`: the LIBERO LeRobot exports;
- `maniskill/`: the ManiSkill demos, regenerated HDF5 files and LeRobot exports, and the ManiSkill assets
  (`MS_ASSET_DIR`, e.g. the Bridge digital twins');
- `bridge/`: the BridgeData V2 TFDS files (`bridge_dataset/1.0.0/`) and the LeRobot export (`lerobot/bridge_orig/`).

**Environments.** Each simulator has its own uv env in this repo, with a uv-managed Python (no conda). Both LeRobot
exports run in `.venv`, which has `lerobot`:
```bash
# LIBERO (robosuite + MuJoCo) and lerobot
uv venv --python 3.10 --python-preference only-managed .venv
uv pip install --python .venv/bin/python "lerobot>=0.4" h5py tqdm mujoco==3.4.0 robosuite==1.4.0 \
    gym==0.25.2 bddl==1.0.1 easydict==1.9 future cloudpickle matplotlib
uv pip install --python .venv/bin/python --no-deps -e ../LIBERO

# ManiSkill (SAPIEN; its GPU sim and renderer need an NVIDIA GPU with Vulkan)
uv venv --python 3.10 --python-preference only-managed .venv_maniskill
uv pip install --python .venv_maniskill/bin/python mani_skill h5py tqdm
```
```bash
# Bridge (reads TFDS with TensorFlow and writes the LeRobot export, in one env)
uv venv --python 3.10 --python-preference only-managed .venv_bridge
uv pip install --python .venv_bridge/bin/python tensorflow-cpu tensorflow-datasets lerobot==0.4.4 opencv-python-headless scipy tqdm
```
Pushing needs a Hugging Face login with write access (`huggingface-cli login`).

### LIBERO (`experiments/robot/libero/regenerate_libero_dataset.py`)

The raw LIBERO demos are human teleoperation, recorded at 256×256. Replaying them in the sim does three things:
- re-renders the images at 64×64 (`--image_resolution`);
- rotates them 180°, since the sim renders them upside down;
- drops the no-op actions (no arm motion and no change of the gripper).

Demos that fail on replay are dropped.

```bash
# 0) Download the raw demos (libero_100 is libero_10 + libero_90)
.venv/bin/python ../LIBERO/benchmark_scripts/download_libero_datasets.py --use-huggingface \
    --datasets libero_spatial --download-dir /network/projects/real-g-grp/libero/

# 1) Replay them. Finished tasks are skipped, so a killed run can be resumed; --task_id N does one task
#    (e.g. one Slurm array job per task)
MUJOCO_GL=egl .venv/bin/python experiments/robot/libero/regenerate_libero_dataset.py regenerate \
    --libero_task_suite libero_spatial \
    --libero_raw_data_dir /network/projects/real-g-grp/libero/libero_spatial/ \
    --libero_target_dir /network/projects/real-g-grp/libero/libero_spatial_no_noops/

# 2) Export to LeRobot v3 and push (--lerobot_root must not exist yet; --no_push builds it locally only)
.venv/bin/python experiments/robot/libero/regenerate_libero_dataset.py push \
    --libero_task_suite libero_spatial \
    --libero_target_dir /network/projects/real-g-grp/libero/libero_spatial_no_noops/ \
    --lerobot_root /network/projects/real-g-grp/libero/lerobot/libero_spatial_no_noops \
    --hf_repo_id gberseth/libero_spatial_no_noops_lerobot

# 3) Check the pushed dataset: download it fresh and replay its episodes open loop in LIBERO with mini-grp's
#    sim/sim_eval.py. Every replay should succeed; side-by-side videos go to outputs/replay_test/
MUJOCO_GL=egl .venv/bin/python experiments/robot/libero/test_libero_dataset_replay.py \
    --hf_repo_id gberseth/libero_spatial_no_noops_lerobot --libero_task_suite libero_spatial --episodes_per_task 2
```
The suites are `libero_spatial`, `libero_object`, `libero_goal`, `libero_10` and `libero_90`. LIBERO's own init states
for a task are not the ones its demos started from, so the export saves each episode's first MuJoCo state in
`meta/libero_init_states.parquet`.

### ManiSkill (`experiments/robot/maniskill/regenerate_maniskill_dataset.py`)

ManiSkill's [demos](https://maniskill.readthedocs.io/en/latest/user_guide/datasets/demos.html) (Hub `haosulab/ManiSkill_Demonstrations`)
have no observations. Only `PushT-v1` is supported so far: 719 rollouts of a PPO policy, recorded with `pd_ee_delta_pose`
on the GPU sim. Its robot (`panda_stick`) has a stick instead of a gripper and no wrist camera.

These demos don't reproduce when their actions are replayed open loop: 57% succeed on the GPU sim and 0% on the CPU sim.
Pushing is chaotic, and the demos were recorded with 1024 envs in parallel. So the converter restores the recorded env
state at every step and renders it. The base camera is the agent view, and a camera 10 cm up the stick is the wrist view.

A demo is kept if two checks pass:
- ManiSkill's `evaluate()` reports success;
- every recorded action, applied from its recorded state, lands the block within `--max_step_error` (1 cm) of the next
  recorded state.

For `PushT-v1` that keeps 706 of 719 demos (median one-step error 0.12 cm).

To fit the layout above, the actions are clipped to [−1, 1], as the controller does, and padded with gripper −1 (open).
Their scales differ from LIBERO's: ManiSkill moves 0.1 m and 0.1 rad per unit, LIBERO 0.05 m and 0.5 rad.

```bash
# 0) Download the demos
MS_ASSET_DIR=/network/projects/real-g-grp/maniskill .venv_maniskill/bin/python -m mani_skill.utils.download_demo \
    PushT-v1 -o /network/projects/real-g-grp/maniskill/demos

# 1) Render and check them (on a GPU; about 18 min for PushT-v1 on an A100). --max_demos N for a quick test
.venv_maniskill/bin/python experiments/robot/maniskill/regenerate_maniskill_dataset.py regenerate \
    --env_id PushT-v1 --target_file /network/projects/real-g-grp/maniskill/PushT-v1_regenerated.h5

# 2) Export to LeRobot v3 and push, with the LIBERO env, which has lerobot (--lerobot_root must not exist yet)
.venv/bin/python experiments/robot/maniskill/regenerate_maniskill_dataset.py push \
    --env_id PushT-v1 --target_file /network/projects/real-g-grp/maniskill/PushT-v1_regenerated.h5 \
    --lerobot_root /network/projects/real-g-grp/maniskill/lerobot/PushT-v1 --hf_repo_id gberseth/maniskill_pusht_lerobot
```
`meta/maniskill_episodes.parquet` holds each episode's reset seed and first state. Calling `env.reset(seed=episode_seed)`
and then `env.set_state(init_state)` restarts an episode exactly where its demo started.

To add another ManiSkill task:
1. Add its goal text and demo file to `TASKS` in the converter.
2. In mini-grp's `sim/sim_eval.py`, which runs the ManiSkill sim evals during training, add a task set
   `{env id: the same goal text}` to `MANISKILL_TASKS`, and map the Hub name to it in `MANISKILL_DATASETS`.
3. If the task has a gripper, keep its gripper action instead of padding it.

### Bridge (`experiments/robot/bridge/regenerate_bridge_dataset.py`)

[BridgeData V2](https://rail-berkeley.github.io/bridgedata/) is teleoperated real data from a WidowX 250 arm. The
export reads the official TFDS copy (`bridge_orig`: 53,192 train episodes). It does not use the copy in Open X-Embodiment,
which is out of date and has about 25k episodes; that one is on the Hub as `jxu124/OpenX-Embodiment` and on the cluster in
`/network/datasets/openxembodiment/datasets/bridge/0.1.0/`. `IPEC-COMMUNITY/bridge_orig_lerobot` has the same data in
LeRobot v2.0, but at 256×256 with Euler angles and one file per episode (about 106k Hub files).

The actions are relabelled as OpenVLA does in `prismatic/vla/datasets/rlds/oxe/transforms.py` (`bridge_orig_dataset_transform`):
- the first step is dropped, since its action is all zeros;
- the motion is the one the robot reached (the next state minus this one), not the commanded one, and the last step is
  dropped;
- the gripper is binarized.

The rotation is stored as the axis-angle of R(next)·R(this)⁻¹ in the robot base frame. The state's roll-pitch-yaw is extrinsic
`xyz`, which matched the commanded actions best of the conventions tried. Actions are in meters and radians per step,
about 1 cm per step at 5 Hz, unlike the [−1, 1] actions of LIBERO and ManiSkill. Only the main camera (`image_0`) is exported. The wrist camera (`image_3`) is in only 13% of the episodes, and
`image_1` and `image_2` (other fixed views) in 43%. It is exported as
`observation.images.image`. Episodes without a language instruction (a third of them) are dropped, and the
instructions are lowercased without a final period.

```bash
# 0) Download the TFDS files (132 GB with the val split; about 1.5 h)
cd /network/projects/real-g-grp/bridge && wget -r -nH --cut-dirs=4 --reject="index.html*" \
    https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/bridge_dataset/

# 1) Export to LeRobot v3 and push (about 5 h on CPU; --max_episodes N --no_push for a quick test)
.venv_bridge/bin/python experiments/robot/bridge/regenerate_bridge_dataset.py \
    --lerobot_root /network/projects/real-g-grp/bridge/lerobot/bridge_orig --hf_repo_id gberseth/bridge_orig_lerobot
# or the same as a Slurm CPU job, so it doesn't depend on an interactive session (log in outputs/slurm/)
mkdir -p outputs/slurm && sbatch --job-name=export_bridge_orig_lerobot experiments/robot/bridge/export_bridge.sbatch
```
`--tfds_dir` reads another TFDS copy with the same features, e.g. the 743-episode subtask subset in
`/network/projects/real-g-grp/tensorflow_datasets/bridge_subtasks/1.0.0`. The videos are 1 MB MP4 files of about 40
episodes each (`--video_files_size_in_mb`). One MP4 per episode would be 35k Hub files, while opening a 200 MB file
for every sample is slow. Sampling a frame from one of these takes 1.1 ms, against 0.45 ms with one file per episode.
`meta/bridge_episodes.parquet` holds each episode's source file and episode id.

mini-grp evaluates models trained on Bridge in ManiSkill's four
[Bridge digital twins](https://maniskill.readthedocs.io/en/latest/tasks/digital_twins/index.html): carrot on plate, spoon
on towel, stack the green cube on the yellow one, and eggplant in basket (`sim.task_sets=[bridge]` in `sim/sim_eval.py`).
Their assets go in the project folder:
```bash
MS_ASSET_DIR=/network/projects/real-g-grp/maniskill .venv_maniskill/bin/python -m mani_skill.utils.download_asset bridge_v2_real2sim -y
MS_ASSET_DIR=/network/projects/real-g-grp/maniskill .venv_maniskill/bin/python -m mani_skill.utils.download_asset widowx250s -y
```

<hr style="border: 2px solid gray;"></hr>

A simple and scalable codebase for training and fine-tuning vision-language-action models (VLAs) for generalist robotic 
manipulation:

- **Different Dataset Mixtures**: We natively support arbitrary datasets in RLDS format, including arbitrary mixtures of
  data from the [Open X-Embodiment Dataset](https://robotics-transformer-x.github.io/).
- **Easy Scaling**: Powered by PyTorch FSDP and Flash-Attention, we can quickly and efficiently train models from 1B - 
  34B parameters, with easily adaptable model architectures.
- **Native Fine-Tuning Support**: Built-in support (with examples) for various forms of fine-tuning (full, 
  partial, LoRA).

Built on top of [Prismatic VLMs](https://github.com/TRI-ML/prismatic-vlms).

## Getting Started

To get started with loading and running OpenVLA models for inference, we provide a lightweight interface that leverages
HuggingFace `transformers` AutoClasses, with minimal dependencies.

For example, to load `openvla-7b` for zero-shot instruction following in the
[BridgeData V2 environments](https://rail-berkeley.github.io/bridgedata/) with a WidowX robot:

```python
# Install minimal dependencies (`torch`, `transformers`, `timm`, `tokenizers`, ...)
# > pip install -r https://raw.githubusercontent.com/openvla/openvla/main/requirements-min.txt
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image

import torch

# Load Processor & VLA
processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b", 
    attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
    torch_dtype=torch.bfloat16, 
    low_cpu_mem_usage=True, 
    trust_remote_code=True
).to("cuda:0")

# Grab image input & format prompt
image: Image.Image = get_from_camera(...)
prompt = "In: What action should the robot take to {<INSTRUCTION>}?\nOut:"

# Predict Action (7-DoF; un-normalize for BridgeData V2)
inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)
action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)

# Execute...
robot.act(action, ...)
```

We also provide an [example script for fine-tuning OpenVLA models for new tasks and 
embodiments](./vla-scripts/finetune.py); this script supports different fine-tuning modes -- including (quantized) 
low-rank adaptation (LoRA) supported by [HuggingFace's PEFT library](https://huggingface.co/docs/peft/en/index). 

For deployment, we provide a lightweight script for [serving OpenVLA models over a REST API](./vla-scripts/deploy.py), 
providing an easy way to integrate OpenVLA models into existing robot control stacks, 
removing any requirement for powerful on-device compute.

## Pretrained VLAs

We release two OpenVLA models trained as part of our work, with checkpoints, configs, and model cards available [on our
HuggingFace page](https://huggingface.co/openvla):
- [`openvla-7b`](https://huggingface.co/openvla/openvla-7b): The flagship model from our paper, trained from 
  the Prismatic `prism-dinosiglip-224px` VLM (based on a fused DINOv2 and SigLIP vision backbone, and Llama-2 LLM). 
  Trained on a large mixture of datasets from Open X-Embodiment spanning 970K trajectories 
  ([mixture details - see "Open-X Magic Soup++"](./prismatic/vla/datasets/rlds/oxe/mixtures.py)).
- [`openvla-v01-7b`](https://huggingface.co/openvla/openvla-7b-v01): An early model used during development, trained from
  the Prismatic `siglip-224px` VLM (singular SigLIP vision backbone, and a Vicuña v1.5 LLM). Trained on the same mixture
  of datasets as [Octo](https://github.com/octo-models/octo), but for significantly fewer GPU hours than our final model 
  ([mixture details - see "Open-X Magic Soup"](./prismatic/vla/datasets/rlds/oxe/mixtures.py)).

**Explicit Notes on Model Licensing & Commercial Use**: While all code in this repository is released under an MIT 
License, our pretrained models may inherit restrictions from the underlying base models we use. Specifically, both the
above models are derived from Llama-2, and as such are subject to the 
[Llama Community License](https://ai.meta.com/llama/license/).

---

## Installation

> **Note**: These installation instructions are for full-scale pretraining (and distributed fine-tuning); if looking to
  just run inference with OpenVLA models (or perform lightweight fine-tuning), see instructions above!

This repository was built using Python 3.10, but should be backwards compatible with any Python >= 3.8. We require
PyTorch 2.2.* -- installation instructions [can be found here](https://pytorch.org/get-started/locally/). The latest 
version of this repository was developed and thoroughly tested with:
  - PyTorch 2.2.0, torchvision 0.17.0, transformers 4.40.1, tokenizers 0.19.1, timm 0.9.10, and flash-attn 2.5.5

**[5/21/24] Note**: Following reported regressions and breaking changes in later versions of `transformers`, `timm`, and
`tokenizers` we explicitly pin the above versions of the dependencies. We are working on implementing thorough tests, 
and plan on relaxing these constraints as soon as we can.

Use the setup commands below to get started:

```bash
# Create and activate conda environment
conda create -n openvla python=3.10 -y
conda activate openvla

# Install PyTorch. Below is a sample command to do this, but you should check the following link
# to find installation instructions that are specific to your compute platform:
# https://pytorch.org/get-started/locally/
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y  # UPDATE ME!

# Clone and install the openvla repo
git clone https://github.com/openvla/openvla.git
cd openvla
pip install -e .

# Install Flash Attention 2 for training (https://github.com/Dao-AILab/flash-attention)
#   =>> If you run into difficulty, try `pip cache remove flash_attn` first
pip install packaging ninja
ninja --version; echo $?  # Verify Ninja --> should return exit code "0"
pip install "flash-attn==2.5.5" --no-build-isolation
```

If you run into any problems during the installation process, please file a GitHub Issue.

**Note:** See `vla-scripts/` for full training and verification scripts for OpenVLA models. Note that `scripts/` is
mostly a holdover from the original (base) `prismatic-vlms` repository, with support for training and evaluating
visually-conditioned language models; while you can use this repo to train VLMs AND VLAs, note that trying to generate
language (via `scripts/generate.py`) with existing OpenVLA models will not work (as we only train current OpenVLA models
to generate actions, and actions alone).

## Fine-Tuning OpenVLA via LoRA

**(2025-03-03 Update: We recommend trying the new OFT recipe for fine-tuning OpenVLA to produce faster and more successful policies. See project website [here](https://openvla-oft.github.io/).)**

In this section, we discuss fine-tuning OpenVLA using Low-Rank Adaptation (LoRA) via the Hugging Face `transformers` library,
which is recommended if you do not have sufficient compute to fully fine-tune a 7B-parameter model. The main script for LoRA
fine-tuning is `vla-scripts/finetune.py`. (If you instead wish to do full fine-tuning, please see the
[Fully Fine-Tuning OpenVLA](#fully-fine-tuning-openvla) section.)

Below we show an example of how you can fine-tune the main OpenVLA checkpoint ([`openvla-7b`](https://huggingface.co/openvla/openvla-7b))
via LoRA. Here we fine-tune on [BridgeData V2](https://rail-berkeley.github.io/bridgedata/) using a single A100
GPU with 80 GB VRAM. (You can also fine-tune with a smaller GPU, as long as it has at least ~27 GB of memory,
by modifying the batch size.)

First, download the BridgeData V2 dataset:

```bash
# Change directory to your base datasets folder
cd <PATH TO BASE DATASETS DIR>

# Download the full dataset (124 GB)
wget -r -nH --cut-dirs=4 --reject="index.html*" https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/bridge_dataset/

# Rename the dataset to `bridge_orig` (NOTE: Omitting this step may lead to runtime errors later)
mv bridge_dataset bridge_orig
```

Now, launch the LoRA fine-tuning script, as shown below. Note that `--batch_size==16` with `--grad_accumulation_steps==1`
requires ~72 GB GPU memory. If you have a smaller GPU, you should reduce `--batch_size` and increase `--grad_accumulation_steps`
to maintain an effective batch size that is large enough for stable training. If you have multiple GPUs and wish to train via
PyTorch Distributed Data Parallel (DDP), simply set `--nproc-per-node` in the `torchrun` command below to the number of available GPUs.

```bash
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path "openvla/openvla-7b" \
  --data_root_dir <PATH TO BASE DATASETS DIR> \
  --dataset_name bridge_orig \
  --run_root_dir <PATH TO LOG/CHECKPOINT DIR> \
  --adapter_tmp_dir <PATH TO TEMPORARY DIR TO SAVE ADAPTER WEIGHTS> \
  --lora_rank 32 \
  --batch_size 16 \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --image_aug <True or False> \
  --wandb_project <PROJECT> \
  --wandb_entity <ENTITY> \
  --save_steps <NUMBER OF GRADIENT STEPS PER CHECKPOINT SAVE>
```

Note: If you set `--image_aug==False` in the command above, you will observe nearly 100% `action_accuracy` in the training logs,
since the [`openvla-7b`](https://huggingface.co/openvla/openvla-7b) model is already pretrained (without augmentations) on a
superset of datasets that includes BridgeData V2.

To LoRA fine-tune on a different dataset, you can download the dataset from the [Open X-Embodiment (OXE)](https://robotics-transformer-x.github.io/)
mixture (see [this custom script](https://github.com/moojink/rlds_dataset_mod/blob/main/prepare_open_x.sh) for an example of how to download datasets
from OXE). Alternatively, if you have a custom dataset that is not part of OXE, you can either (a) convert the dataset to the RLDS format which is
compatible with our fine-tuning script (see [this repo](https://github.com/kpertsch/rlds_dataset_builder) for instructions on this), or (b) use your own
custom PyTorch Dataset wrapper (see comments in `vla-scripts/finetune.py` for instructions). We recommend option (a) for most users; the RLDS dataset and
dataloader are tested more extensively since we used these for all of our pretraining and fine-tuning experiments.

For option (a), after you converted your dataset to RLDS, you need to register it with our data loader, by registering a dataset
config [here](prismatic/vla/datasets/rlds/oxe/configs.py#L54) and a dataset transform function [here](prismatic/vla/datasets/rlds/oxe/transforms.py#L828).

Once you have integrated your new dataset, you can launch LoRA fine-tuning with the same `vla-scripts/finetune.py` script above. If you run into any issues,
please visit the [VLA Troubleshooting](#vla-troubleshooting) section or search for a similar issue in the [OpenVLA GitHub Issues page](https://github.com/openvla/openvla/issues?q=)
(including "Closed" issues). If you cannot find a similar issue there, feel free to create a new issue.

## Fully Fine-Tuning OpenVLA

**(2025-03-03 Update: We recommend trying the new OFT recipe for fine-tuning OpenVLA to produce faster and more successful policies. See project website [here](https://openvla-oft.github.io/).)**

In this section, we discuss <ins>fully fine-tuning</ins> OpenVLA (all 7.5 billion parameters) via native PyTorch Fully Sharded Data Parallel (FSDP)
using the [Prismatic VLMs](https://github.com/TRI-ML/prismatic-vlms) training script. Full fine-tuning is more advanced/involved and is only recommended
if you have sufficient compute (e.g., a full node of 8 A100 GPUs) and if LoRA fine-tuning is insufficient for your use case (e.g., if the fine-tuning distribution
varies drastically from the pretraining distribution). Otherwise, we recommend that you try parameter-efficient fine-tuning via LoRA, which is described in the 
[Fine-Tuning OpenVLA via LoRA](#fine-tuning-openvla-via-lora) section.

For full fine-tuning, you will need to download [a different version of the OpenVLA model checkpoint](https://huggingface.co/openvla/openvla-7b-prismatic) that is compatible
with the Prismatic VLMs codebase, which we built on top of to develop the OpenVLA model. You can download this Prismatic-compatible OpenVLA checkpoint using the git commands below
(alternatively, you can download via the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/main/en/guides/cli)):

```bash
# Change directory to your base model checkpoints folder
cd <PATH TO BASE MODEL CHECKPOINTS DIR>

# Download checkpoint (30 GB) -- may take a few minutes
git clone git@hf.co:openvla/openvla-7b-prismatic

# If the command above did not download the full checkpoint,
# manually fetch it via git Large File Storage (LFS)
# Note: You may have to configure an SSH key for this to work
cd openvla-7b-prismatic
git lfs fetch --all
```

We show how you can fully fine-tune OpenVLA on [BridgeData V2](https://rail-berkeley.github.io/bridgedata/) using a single node with 8 GPUs. If you wish to
use a different number of GPUs (or nodes), you can modify the VLA training configuration in [`prismatic/conf/vla.py`](prismatic/conf/vla.py).

Download the BridgeData V2 dataset:

```bash
# Change directory to your base datasets folder
cd <PATH TO BASE DATASETS DIR>

# Download the full dataset (124 GB)
wget -r -nH --cut-dirs=4 --reject="index.html*" https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/bridge_dataset/

# Rename the dataset to `bridge_orig` (NOTE: Omitting this step may lead to runtime errors later)
mv bridge_dataset bridge_orig
```

Next, create a [Hugging Face user access token](https://huggingface.co/docs/hub/en/security-tokens) and copy the token value (a string that starts with
`hf_...`) into a file named `.hf_token` at the root directory of this repo (`openvla/.hf_token`).

```bash
# Go to openvla root directory
cd openvla

# Copy HF token value into token file. Replace "hf_..." with your own token value!
# See: https://huggingface.co/docs/hub/en/security-tokens
echo hf_... >>> .hf_token
```

Now, launch the training script. If you wish to use a different number of nodes or GPUs, modify the VLA training configuration in
[`prismatic/conf/vla.py`](prismatic/conf/vla.py) and then change the `--nnodes` and `--nproc-per-node` arguments below accordingly.

```bash
torchrun --standalone --nnodes 1 --nproc-per-node 8 vla-scripts/train.py \
  --pretrained_checkpoint <PATH TO openvla/openvla-7b-prismatic CHECKPOINT FILE: step-295000-epoch-40-loss=0.2200.pt> \
  --vla.type prism-dinosiglip-224px+mx-bridge \
  --data_root_dir <PATH TO BASE DATASETS DIR> \
  --run_root_dir <PATH TO LOG/CHECKPOINT DIR> \
  --run_id <OPTIONAL RUN ID FOR WANDB LOGGING> \
  --image_aug <True or False> \
  --wandb_project <PROJECT> \
  --wandb_entity <ENTITY> \
  --save_interval <NUMBER OF GRADIENT STEPS PER CHECKPOINT SAVE> \
  --is_resume False
```

Note that the `--is_resume` argument is set to `False` above since we are fine-tuning a pretrained checkpoint rather than resuming a paused training run.

If your training run gets paused and you wish to resume from the latest checkpoint, change `--pretrained_checkpoint` to the latest checkpoint path,
and then set `--is_resume==True` and specify `--resume_step` and `--resume_epoch` as the step and epoch number, respectively. For example, if you wish to
resume training from a checkpoint named `step-010000-epoch-20-loss=0.0160.pt`, you would set `is_resume==True`, `resume_step==10000`, and `resume_epoch==20`.

Note: If you run the BridgeData V2 fine-tuning command above, you should observe nearly 100% Action Token Accuracy in the training logs, since the
[`openvla-7b`](https://huggingface.co/openvla/openvla-7b) model is already pretrained on a superset of datasets that includes BridgeData V2.

To fully fine-tune OpenVLA on a different dataset, you can download the dataset from the [Open X-Embodiment (OXE)](https://robotics-transformer-x.github.io/)
mixture (see [this custom script](https://github.com/moojink/rlds_dataset_mod/blob/main/prepare_open_x.sh) for an example of how to download datasets from OXE).
Alternatively, if you have a custom dataset that is not part of OXE, you can convert the dataset to the RLDS format, which is compatible with our fine-tuning script
(see [this repo](https://github.com/kpertsch/rlds_dataset_builder) for instructions on this). After downloading/converting the dataset, you will need to modify the following files:

* [`prismatic/conf/vla.py`](prismatic/conf/vla.py): Add a new training configuration by creating an experiment class, and then register it in the `VLARegistry` at the bottom of the file.
  * Make sure to create a new unique `vla_id` for your fine-tuning run, and adjust some configuration variables as needed – e.g., `expected_world_size` (number of GPUs),
  `per_device_batch_size` (batch size per GPU), `global_batch_size` (total batch size), `shuffle_buffer_size` (number of samples in shuffle buffer per GPU), etc. See comments
  under the `VLAConfig` class at the top of the file to understand the purpose of each variable.
* [`prismatic/vla/datasets/rlds/oxe/mixtures.py`](prismatic/vla/datasets/rlds/oxe/mixtures.py): Define a new mixture for your fine-tuning mixture in the `OXE_NAMED_MIXTURES` dictionary.
* [`prismatic/vla/datasets/rlds/oxe/transforms.py`](prismatic/vla/datasets/rlds/oxe/transforms.py): Define a new dataset transform function for your fine-tuning dataset, and add it to the
`OXE_STANDARDIZATION_TRANSFORMS` registry at the bottom of the file.
* [`prismatic/vla/datasets/rlds/oxe/configs.py`](prismatic/vla/datasets/rlds/oxe/configs.py): Add a new configuration specifying your fine-tuning dataset's observation and action spaces
to the `OXE_DATASET_CONFIGS` dictionary.

After completing the steps above, you can start full fine-tuning using the `vla-scripts/train.py` script. Make sure to set the `--vla.type` argument to the new `vla_id` that you added in `prismatic/conf/vla.py`.

When you are finished with fine-tuning, you will need to convert the final model checkpoint to a version that is
compatible with the Hugging Face `transformers` library. See the [Converting Prismatic Models to Hugging Face](#converting-prismatic-models-to-hugging-face) section for instructions.

If you run into any issues, please visit the [VLA Troubleshooting](#vla-troubleshooting) section or search for a similar issue in the
[OpenVLA GitHub Issues page](https://github.com/openvla/openvla/issues?q=) (including "Closed" issues). If you cannot find a similar issue there, feel free to create a new issue.

### Converting Prismatic Models to Hugging Face

If you have used the Prismatic VLMs codebase to train your model (e.g., if you did full fine-tuning of OpenVLA on a
new dataset), you will need to convert the final checkpoint to a version that is compatible with Hugging Face
`transformers` AutoClasses. We discuss how to do so in this section.

Let's say your training run directory is `PRISMATIC_RUN_DIR` (e.g., `prism-dinosiglip-224px+mx-oxe-magic-soup-plus+n8+b32+x7`).
Inside this directory, there should be a directory called `checkpoints` which contains saved model checkpoints (e.g.,
`step-295000-epoch-40-loss=0.2200.pt`). The Prismatic-to-Hugging-Face conversion script
([convert_openvla_weights_to_hf.py](vla-scripts/extern/convert_openvla_weights_to_hf.py)) expects a checkpoint file
named `latest-checkpoint.pt`. Therefore, you should first create a symbolic link called `latest-checkpoint.pt` that
points to the checkpoint file that you wish to convert:

```bash
# Go to your Prismatic training run's `checkpoints` directory
cd PRISMATIC_RUN_DIR/checkpoints

# Create symbolic link pointing to your checkpoint file
ln -s <YOUR CHECKPOINT FILENAME> latest-checkpoint.pt
```

Then, launch the conversion script to convert the checkpoint from the Prismatic VLMs format to the Hugging Face format:

```bash
python vla-scripts/extern/convert_openvla_weights_to_hf.py \
    --openvla_model_path_or_id <PRISMATIC_RUN_DIR> \
    --output_hf_model_local_path <OUTPUT DIR FOR CONVERTED CHECKPOINT>
```

The command above will save the HF-compatible checkpoint in `output_hf_model_local_path`. Now you can load the checkpoint
with HF AutoClasses as normal, as shown below. Note that there is an additional necessary step to register the OpenVLA model
to HF AutoClasses before loading it because you are loading a locally saved checkpoint rather than one that is pushed to the
HF Hub (see [here](https://huggingface.co/docs/transformers/en/custom_models#registering-a-model-with-custom-code-to-the-auto-classes)
for details).

```python
import torch
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

# Register OpenVLA model to HF AutoClasses (not needed if you pushed model to HF Hub)
AutoConfig.register("openvla", OpenVLAConfig)
AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

# Load Processor & VLA
processor = AutoProcessor.from_pretrained("<PATH TO CONVERTED CHECKPOINT DIR>", trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    "<PATH TO CONVERTED CHECKPOINT DIR>",
    attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
).to("cuda:0")

...
```

## Training VLAs from Scratch

We provide full instructions and configurations for training VLA models on (arbitrary subsets of) the
[Open X-Embodiment (OXE) Dataset](https://robotics-transformer-x.github.io/). If you run in to any issues with 
the following, see [VLA Troubleshooting](#vla-troubleshooting) below (or file a GitHub Issue).

### VLA Pretraining Datasets

We download and preprocess individual datasets from Open X-Embodiment in [RLDS format](https://github.com/google-research/rlds) following 
[this custom script](https://github.com/moojink/rlds_dataset_mod/blob/main/prepare_open_x.sh). See 
[mixtures.py](./prismatic/vla/datasets/rlds/oxe/mixtures.py) for the full list of component datasets (and mixture 
weights) we use to train `openvla-7b`. 
- **Important**: For the BridgeData V2 component, the version in OXE is out of date (as of 12/20/2023). Instead,
  you should download the dataset from the [official website](https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/bridge_dataset/) and place it under the subdirectory `bridge_orig/`. 
  Replace any reference to `bridge` in the OXE code with `bridge_orig`.

### VLA Configuration & Training Script

The entry point for VLA training is [`vla-scripts/train.py`](vla-scripts/train.py). We use 
[`draccus`](https://pypi.org/project/draccus) to provide a modular, dataclass-based interface for specifying VLA 
training configurations; existing VLA configurations are in [`prismatic/conf/vla.py`](prismatic/conf/vla.py). You can 
add your own training configuration and refer to it using the `--vla.type` command line argument.

We use PyTorch Fully Sharded Data Parallel (FSDP) to distribute training across GPUs. Launch training via `torchrun`:

```bash
# Train VLA on BridgeData V2 with the Prismatic DINO-SigLIP 224px Backbone on a Single Node (w/ 8 GPUs)
torchrun --standalone --nnodes 1 --nproc-per-node 8 vla-scripts/train.py \
  --vla.type "prism-dinosiglip-224px+mx-bridge" \
  --data_root_dir <PATH TO OXE DATA ROOT> \
  --run_root_dir <PATH TO LOG/CHECKPOINT ROOT> \
  --wandb_project "<PROJECT>" \
  --wandb_entity "<ENTITY>"
```

### VLA Troubleshooting

The following are a list of known problems and corresponding fixes:

```bash
FileNotFoundError: Failed to construct dataset "fractal20220817_data", builder_kwargs "{'data_dir': '/path/to/processed/datasets/'}": Could not load dataset info from fractal20220817_data/0.1.0/dataset_info.json
```
- **Fix**: Downgrade `tensorflow-datasets` via `pip install tensorflow-datasets==4.9.3`.


```bash
AttributeError: 'DLataset' object has no attribute 'traj_map'. Did you mean: 'flat_map'?
```
- **Fix**: Upgrade `dlimp` to the newest version. You may have to `--force-reinstall` like so:
`pip install --no-deps --force-reinstall git+https://github.com/moojink/dlimp_openvla`

---

## Evaluating OpenVLA

### BridgeData V2 WidowX Evaluations

#### Setup

Clone the [BridgeData V2 WidowX controller repo](https://github.com/rail-berkeley/bridge_data_robot) and install the `widowx_envs` package:

```bash
git clone https://github.com/rail-berkeley/bridge_data_robot.git
cd bridge_data_robot
pip install -e widowx_envs
```

Additionally, install the [`edgeml`](https://github.com/youliangtan/edgeml) library:
```bash
git clone https://github.com/youliangtan/edgeml.git
cd edgeml
pip install -e .
```

Follow the instructions in the `bridge_data_robot` README to create the Bridge WidowX Docker container.

#### Launching BridgeData V2 Evaluations

There are multiple ways to run BridgeData V2 evaluations. We describe the server-client method below.

In one Terminal window (e.g., in tmux), start the WidowX Docker container:

```bash
cd bridge_data_robot
./generate_usb_config.sh
USB_CONNECTOR_CHART=$(pwd)/usb_connector_chart.yml docker compose up --build robonet
```

In a second Terminal window, run the WidowX robot server:

```bash
cd bridge_data_robot
docker compose exec robonet bash -lic "widowx_env_service --server"
```

In a third Terminal window, run the OpenVLA policy evaluation script:

```bash
cd openvla
python experiments/robot/bridge/run_bridgev2_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b
```

If you run into an error such as `ModuleNotFoundError: No module named 'moviepy.editor'`, you can work around it by fixing the
moviepy version to an older version, v1.0.3, in the bridge_data_robot repo's requirements.txt file
[here](https://github.com/rail-berkeley/bridge_data_robot/blob/main/widowx_envs/requirements.txt). I.e., simply replace `moviepy`
with `moviepy==1.0.3` in the requirements.txt file. Then, go back to the first step above and restart the WidowX Docker container;
it should be rebuilt with the older moviepy version.


### LIBERO Simulation Benchmark Evaluations

In the [updated OpenVLA paper (v2)](https://arxiv.org/abs/2406.09246), we discuss fine-tuning OpenVLA
on a simulated benchmark, [LIBERO](https://libero-project.github.io/main.html), in Appendix E.
Please see the paper for details, such as how we modify the provided demonstration datasets to
improve the overall performance of all methods.

We copy the results to the section below and then discuss how to reproduce the results for OpenVLA.

#### OpenVLA Fine-Tuning Results

| Method | LIBERO-Spatial | LIBERO-Object | LIBERO-Goal | LIBERO-Long | Average |
|--------|----------------|---------------|-------------|-------------|---------|
| Diffusion Policy from scratch | 78.3 ± 1.1% | **92.5 ± 0.7%** | 68.3 ± 1.2% | 50.5 ± 1.3% | 72.4 ± 0.7% |
| Octo fine-tuned | 78.9 ± 1.0% | 85.7 ± 0.9% | **84.6 ± 0.9%** | 51.1 ± 1.3% | 75.1 ± 0.6% |
| OpenVLA fine-tuned (ours) | **84.7 ± 0.9%** | 88.4 ± 0.8% | 79.2 ± 1.0% | **53.7 ± 1.3%** | **76.5 ± 0.6%** |

Each success rate is the average over 3 random seeds x 500 rollouts each (10 tasks x 50 rollouts per task).

#### LIBERO Setup

Clone and install the [LIBERO repo](https://github.com/Lifelong-Robot-Learning/LIBERO):

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -e .
```

Additionally, install other required packages:
```bash
cd openvla
pip install -r experiments/robot/libero/libero_requirements.txt
```

(Optional) To download the modified versions of the LIBERO datasets that we used in our fine-tuning
experiments, run the command below. This will download the LIBERO-Spatial, LIBERO-Object, LIBERO-Goal,
and LIBERO-10 datasets in RLDS data format (~10 GB total). You can use these to fine-tune OpenVLA or
train other methods. This step is optional since we provide pretrained OpenVLA checkpoints below.
(Also, you can find the script we used to generate the modified datasets in raw HDF5 format
[here](experiments/robot/libero/regenerate_libero_dataset.py) and the code we used to convert these
datasets to the RLDS format [here](https://github.com/moojink/rlds_dataset_builder).)
```bash
git clone git@hf.co:datasets/openvla/modified_libero_rlds
```

#### Launching LIBERO Evaluations

We fine-tuned OpenVLA via LoRA (r=32) on four LIBERO task suites independently: LIBERO-Spatial, LIBERO-Object, LIBERO-Goal, and LIBERO-10 (also called LIBERO-Long).
The four checkpoints are available on Hugging Face:
* [openvla/openvla-7b-finetuned-libero-spatial](https://huggingface.co/openvla/openvla-7b-finetuned-libero-spatial)
* [openvla/openvla-7b-finetuned-libero-object](https://huggingface.co/openvla/openvla-7b-finetuned-libero-object)
* [openvla/openvla-7b-finetuned-libero-goal](https://huggingface.co/openvla/openvla-7b-finetuned-libero-goal)
* [openvla/openvla-7b-finetuned-libero-10](https://huggingface.co/openvla/openvla-7b-finetuned-libero-10)

To start evaluation with one of these checkpoints, run one of the commands below. Each will automatically download the appropriate checkpoint listed above.

```bash
# Launch LIBERO-Spatial evals
python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --center_crop True

# Launch LIBERO-Object evals
python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object \
  --center_crop True

# Launch LIBERO-Goal evals
python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --center_crop True

# Launch LIBERO-10 (LIBERO-Long) evals
python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --center_crop True
```

Notes:
* The evaluation script will run 500 trials by default (10 tasks x 50 episodes each). You can modify the number of
  trials per task by setting `--num_trials_per_task`. You can also change the random seed via `--seed`.
* **NOTE: Setting `--center_crop True` is important** because we fine-tuned OpenVLA with random crop augmentations
  (we took a random crop with 90% area in every training sample, so at test time we simply take the center 90% crop).
* The evaluation script logs results locally. You can also log results in Weights & Biases
  by setting `--use_wandb True` and specifying `--wandb_project <PROJECT>` and `--wandb_entity <ENTITY>`.
* The results reported in our paper were obtained using **Python 3.10.13, PyTorch 2.2.0, transformers 4.40.1, and
  flash-attn 2.5.5** on an **NVIDIA A100 GPU**, averaged over three random seeds. Please stick to these package versions.
  Note that results may vary slightly if you use a different GPU for evaluation due to GPU nondeterminism in large models
  (though we have tested that results were consistent across different machines with A100 GPUs).

Please file a GitHub Issue if you run into any problems.

---

## Repository Structure

High-level overview of repository/project file-tree:

+ `prismatic` - Package source; provides core utilities for model loading, training, data preprocessing, etc.
+ `vla-scripts/` - Core scripts for training, fine-tuning, and deploying VLAs.
+ `experiments/` - Code for evaluating OpenVLA policies in robot environments.
+ `LICENSE` - All code is made available under the MIT License; happy hacking!
+ `Makefile` - Top-level Makefile (by default, supports linting - checking & auto-fix); extend as needed.
+ `pyproject.toml` - Full project configuration details (including dependencies), as well as tool configurations.
+ `README.md` - You are here!

---


# VLA Performance Troubleshooting

In this section we cover best practices for debugging poor VLA performance after fine-tuning on your target domain robot dataset.

**Note**: OpenVLA typically requires fine-tuning on a small demonstration dataset (~100 demos) from your target domain robot. Out-of-the-box, it only works well on domains from the training dataset.

**Sanity checks**:
- replay the actions from a demonstration from your fine-tuning dataset and make sure that the robot can execute the task successfully (this ensures that your data collection pipeline is correct)
- once you fine-tuned a model, load the model in your inference pipeline (as if you would run it to control the robot), but feed images from the fine-tuning dataset into the model (pretending they come from the robot) and verify that you can reproduce the token accuracies / L1 errors from training (this ensures that your inference pipeline is correct)

**Best practices for fine-tuning data collection**:
If your setup passed the above two sanity checks, the issue may not be in model training, but in the data you fine-tuned the model with. Some best practices for data collection:
- *Collect at a control frequency around 5-10Hz.* OpenVLA is not trained with action chunking, empirically the model struggles with high-frequency data. If your robot setup uses a high-frequency controller (eg 50 Hz), consider downsampling your actions to 5Hz. Verify first that your robot can still solve the task when using 5Hz actions (ie repeat sanity check (1) above with 5Hz actions)
- *Avoid pauses / small actions during data collection.* Because OpenVLA is trained without action chunking, the model can be sensitive to idle actions in the fine-tuning data. If your data contains steps in which the robot barely moves, the model may "get stuck" in these steps at inference time. Try to collect fine-tuning demonstrations with continuous, slow movement.
- *Ensure sufficient data coverage.* If you plan to test the model with some variation, e.g. different initial positions of objects, make sure that your fine-tuning data contains sufficient diversity of such conditions as well, e.g. shows demonstrations with diverse initial conditions.
- *Use consistent task strategies during data collection.* This is not a hard constraint, but may make your life easier. Try to demonstrate tasks in consistent ways, e.g. approach objects from the same side, perform sub-steps in the same order even if they could be performed in arbitrary sequences. Being consistent gives you a less multi-modal fine-tuning dataset, which makes the modeling problem easier.


---

#### Citation

If you find our code or models useful in your work, please cite [our paper](https://arxiv.org/abs/2406.09246):

```bibtex
@article{kim24openvla,
    title={OpenVLA: An Open-Source Vision-Language-Action Model},
    author={{Moo Jin} Kim and Karl Pertsch and Siddharth Karamcheti and Ted Xiao and Ashwin Balakrishna and Suraj Nair and Rafael Rafailov and Ethan Foster and Grace Lam and Pannag Sanketi and Quan Vuong and Thomas Kollar and Benjamin Burchfiel and Russ Tedrake and Dorsa Sadigh and Sergey Levine and Percy Liang and Chelsea Finn},
    journal = {arXiv preprint arXiv:2406.09246},
    year={2024}
} 
```
