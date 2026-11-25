#!/bin/bash
#SBATCH --job-name=regenerate_libero_state
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:59:00
#SBATCH --output=regenerate_libero_state-%j.out
# The whole LIBERO data pipeline for one suite, on a GPU node (rendering needs EGL): regenerate its demos from the raw
# HDF5s (replay without no-ops, keep the successes, record images, robot state and, since 2026-11-24, every object's
# pose, obs/object_states), then export them as a LeRobot dataset with observation.object_state (for state policies,
# e.g. mini-grp's MLP). Both go to new folders, so the existing no_noops data is untouched. It doesn't push: after
# checking the log, push the export as gberseth/<suite>_no_noops_state_lerobot. Run from the repo root on mila
# (default suite libero_spatial):
#   sbatch experiments/robot/libero/regenerate_libero_state.sh [suite]
set -e
SUITE=${1:-libero_spatial}
PY=.venv/bin/python
RAW=/network/projects/real-g-grp/libero/${SUITE}/
HDF5=/network/projects/real-g-grp/libero/${SUITE}_no_noops_state/
TMP=$SLURM_TMPDIR/${SUITE}_no_noops_state
OUT=/network/projects/real-g-grp/libero/lerobot/${SUITE}_no_noops_state
## LIBERO is ../LIBERO-Industry, installed once in .venv (regenerate_libero_dataset.py's setup) and the one
## ~/.libero/config.yaml points to, as mini-grp's image simulates
$PY -c "import libero.libero as l, os; assert 'LIBERO-Industry' in os.path.realpath(l.__file__), l.__file__"
export MUJOCO_GL=egl
$PY experiments/robot/libero/regenerate_libero_dataset.py regenerate \
    --libero_task_suite $SUITE --libero_raw_data_dir $RAW --libero_target_dir $HDF5
$PY experiments/robot/libero/regenerate_libero_dataset.py push --object_state --no_push \
    --libero_task_suite $SUITE --libero_target_dir $HDF5 --lerobot_root $TMP \
    --hf_repo_id gberseth/${SUITE}_no_noops_state_lerobot
ls -la $TMP/meta $TMP/data/*/ $TMP/videos/*/chunk-000
rm -rf $OUT && cp -r $TMP $OUT
echo "$SUITE done: $HDF5 and $OUT"
