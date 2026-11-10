#!/bin/bash
#SBATCH --job-name=relayout_libero_100mb
#SBATCH --partition=long-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=relayout_libero_100mb-%j.out
# Rewrites the LIBERO LeRobot datasets on the Hub (one MP4 per episode) into 100 MB MP4s, as libero_90 since
# 2026-10-13, so downloads stay under the Hub's request limit. For each suite: download it (from the HF cache if it's
# there), relayout on the node's disk (relayout_lerobot_videos.py: each episode's packets copied once and checked),
# copy to the project disk and check it again there. It doesn't push; after checking the log, push each suite with
#   python experiments/robot/libero/push_relayout.py /network/projects/real-g-grp/libero/lerobot/<suite>_no_noops_relayout gberseth/<suite>_no_noops_lerobot
# Run from the repo root on mila (the suites default to the four still one MP4 per episode):
#   sbatch experiments/robot/libero/relayout_libero_suites.sh [suite ...]
set -e
SUITES=${@:-libero_spatial libero_object libero_goal libero_10}
PY=.venv/bin/python
export HF_HOME=${HF_HOME:-$SCRATCH/.cache/huggingface}
for suite in $SUITES; do
    echo "=== $suite"
    SRC=$($PY -c "from huggingface_hub import snapshot_download; print(snapshot_download('gberseth/${suite}_no_noops_lerobot', repo_type='dataset'))")
    TMP=$SLURM_TMPDIR/${suite}_no_noops_relayout
    OUT=/network/projects/real-g-grp/libero/lerobot/${suite}_no_noops_relayout
    $PY experiments/robot/libero/relayout_lerobot_videos.py $SRC $TMP --video_files_size_in_mb 100
    ls -la $TMP/videos/*/chunk-000
    rm -rf $OUT && cp -r $TMP $OUT
    $PY experiments/robot/libero/relayout_lerobot_videos.py $SRC $OUT --check_only
    echo "$suite done: $OUT"
done
