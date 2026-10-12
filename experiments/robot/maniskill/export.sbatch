#!/bin/bash
## Exports regenerated ManiSkill tasks to one LeRobot v3 dataset and pushes it to the Hub (regenerate_maniskill_dataset.py
## push, step 4), as a CPU job. Submit from the openvla repo root, with the push step's arguments, e.g.
##   mkdir -p outputs/slurm && sbatch --job-name=export_maniskill_panda_stick_lerobot experiments/robot/maniskill/export.sbatch \
##       --env_ids PushT-v1 DrawTriangle-v1 --lerobot_root /network/projects/real-g-grp/maniskill/lerobot/panda_stick \
##       --hf_repo_id gberseth/maniskill_panda_stick_lerobot
## It can't resume: a failed job leaves a partial --lerobot_root, which must be moved away before resubmitting.

#SBATCH --partition=main-cpu                          # not preempted (the export can't resume)
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH -o outputs/slurm/%x-%j.out                    # relative to the submit dir; outputs/slurm must exist

cd "$SLURM_SUBMIT_DIR"
set -o pipefail ## fail the job when the export fails, not only when the grep below does
echo "Exporting ManiSkill on $SLURM_JOB_NODELIST, args: $*"
.venv/bin/python experiments/robot/maniskill/regenerate_maniskill_dataset.py push \
    --target_dir /network/projects/real-g-grp/maniskill "$@" 2>&1 | grep -v "Svt\[info\]"
