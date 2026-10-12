#!/bin/bash
## Regenerates one ManiSkill task's demos (regenerate_maniskill_dataset.py, steps 2 and 3): converts motion-planning
## demos to pd_ee_delta_pose, then renders and checks them. Rendering needs a GPU, even for the CPU sim. Submit from
## the openvla repo root, one job per task:
##   mkdir -p outputs/slurm && sbatch --job-name=regenerate_maniskill_PegInsertionSide-v1 experiments/robot/maniskill/regenerate.sbatch PegInsertionSide-v1
## Jobs on long can be preempted and are not requeued: check `sacct -j <id>` and resubmit.

#SBATCH --partition=long
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --constraint=ampere|lovelace                 # the uv env's torch has no kernels for the V100 (compute capability 7.0)
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH -o outputs/slurm/%x-%j.out                    # relative to the submit dir; outputs/slurm must exist
#SBATCH --no-requeue

cd "$SLURM_SUBMIT_DIR"
export MS_ASSET_DIR=/network/projects/real-g-grp/maniskill
ENV_ID=$1
DEMOS=/network/projects/real-g-grp/maniskill/demos/$ENV_ID
echo "Regenerating $ENV_ID on $SLURM_JOB_NODELIST"
case $ENV_ID in PegInsertionSide-v1|PlugCharger-v1|PullCubeTool-v1|StackPyramid-v1|DrawTriangle-v1) ## motion-planning demos
    .venv_maniskill/bin/python -m mani_skill.trajectory.replay_trajectory -c pd_ee_delta_pose -b physx_cpu --save-traj \
        -n $SLURM_CPUS_PER_TASK --traj-path $DEMOS/motionplanning/trajectory.h5 2>&1 | grep -v "it/s\|step/s";;
esac
.venv_maniskill/bin/python experiments/robot/maniskill/regenerate_maniskill_dataset.py regenerate \
    --env_ids $ENV_ID --target_dir /network/projects/real-g-grp/maniskill
