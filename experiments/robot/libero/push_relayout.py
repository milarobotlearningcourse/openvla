"""Replaces a LeRobot dataset on the Hub with a local re-export in one commit: uploads meta/, data/ and videos/ and deletes
the old files under them (e.g. one MP4 per episode), then moves the codebase-version tag to the new commit.
Usage: python push_relayout.py <lerobot_root> <repo_id>"""
import json
import os
import sys

from huggingface_hub import HfApi

root, repo_id = sys.argv[1:3]
info = json.load(open(os.path.join(root, "meta", "info.json")))
print(repo_id, info["codebase_version"], info["total_episodes"], "episodes,", info["total_frames"], "frames,",
      "video", info["video_files_size_in_mb"], "MB")
api = HfApi()
commit = api.upload_folder(repo_id=repo_id, repo_type="dataset", folder_path=root,
                           allow_patterns=["meta/**", "data/**", "videos/**"],
                           delete_patterns=["meta/**", "data/**", "videos/**"],
                           commit_message=f"Many episodes per MP4 ({info['video_files_size_in_mb']:g} MB files, as lerobot/droid_1.0.1), each episode's packets copied once from its own MP4")
tag = info["codebase_version"]
api.delete_tag(repo_id, tag=tag, repo_type="dataset")
api.create_tag(repo_id, tag=tag, revision=commit.oid, repo_type="dataset")
files = api.list_repo_files(repo_id, repo_type="dataset")
print(commit.commit_url, f"{len(files)} files now:", [f for f in files if not f.startswith("videos/")][:20],
      sum(f.startswith("videos/") for f in files), "videos")
