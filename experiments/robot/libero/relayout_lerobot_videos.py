"""Rewrites a LeRobot v3 dataset's videos into the layout of the official datasets (lerobot/droid_1.0.1): many episodes
per MP4, a new file past --video_files_size_in_mb. Each episode's packets are copied from its own MP4 unchanged (no
re-encoding), once. lerobot's export instead appends each episode by rewriting the whole shared MP4 (about 1,500 times
per 100 MB file), and that left damaged packets in libero_90 (mini-grp's log/2026-10-12.md), while the same episodes
exported one per MP4 are clean. Afterwards every output MP4 is read back: each episode's packets must be byte for byte
its source's, and every packet must decode. data/ and the rest of meta/ are copied as they are.
Usage: python relayout_lerobot_videos.py <source root> <output root> [--video_files_size_in_mb 100] [--check_only]
(--check_only: only check an output made before, e.g. after copying it to another disk)"""
import argparse
import glob
import hashlib
import json
import os
import shutil

import av
import pandas as pd


def read_episodes(root):
    files = sorted(glob.glob(os.path.join(root, "meta", "episodes", "*", "*.parquet")))
    return pd.concat([pd.read_parquet(f) for f in files]).sort_values("episode_index").reset_index(drop=True)


def video_path(root, info, key, chunk, file):
    return os.path.join(root, info["video_path"].format(video_key=key, chunk_index=chunk, file_index=file))


def packets(container, start, end):
    """ The packets of [start, end) s of an MP4's video stream, in order. """
    stream = container.streams.video[0]
    container.seek(int(start / stream.time_base), stream=stream)  # to the keyframe at or before start
    for packet in container.demux(stream):
        if packet.pts is None:
            continue
        t = float(packet.pts * stream.time_base)
        if t >= end - 1e-6:
            break
        if t >= start - 1e-6:
            yield packet


def relayout(src, out, key, info, episodes, limit_bytes):
    """ Copies key's episodes into new MP4s, returns their new [chunk, file, from, to] columns and each episode's
    packet hash. """
    col = f"videos/{key}/"
    columns, hashes = {c: [] for c in ("chunk_index", "file_index", "from_timestamp", "to_timestamp")}, []
    container, file_index, size, offset = None, -1, 0, 0.0
    for _, ep in episodes.iterrows():
        with av.open(video_path(src, info, key, int(ep[col + "chunk_index"]), int(ep[col + "file_index"]))) as source:
            stream = source.streams.video[0]
            if container is None or size >= limit_bytes:  # a new file
                if container is not None:
                    container.close()
                file_index, size, offset = file_index + 1, 0, 0.0
                path = video_path(out, info, key, file_index // info["chunks_size"], file_index)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                container = av.open(path, "w", options={"movflags": "faststart"})
                out_stream = container.add_stream_from_template(stream, opaque=True)  # a stream copy
                time_base = stream.time_base
            assert stream.time_base == time_base, "the episodes' MP4s must share a time base"
            start, length = ep[col + "from_timestamp"], ep["length"] / info["fps"]
            shift = int(round((offset - start) / time_base))
            digest = hashlib.sha256()
            for packet in packets(source, start, start + length):
                digest.update(bytes(packet))
                size += packet.size
                packet.pts, packet.dts = packet.pts + shift, packet.dts + shift
                packet.stream = out_stream
                container.mux(packet)
        hashes.append(digest.hexdigest())
        for c, v in zip(columns, (file_index // info["chunks_size"], file_index, offset, offset + length)):
            columns[c].append(v)
        offset += length
    container.close()
    return columns, hashes


def check(root, key, info, episodes, hashes=None):
    """ Every packet of key's MP4s decodes, and (with hashes) each episode's packets are the source's. """
    col = f"videos/{key}/"
    bad, mismatched = 0, []
    for (chunk, file), in_file in episodes.groupby([col + "chunk_index", col + "file_index"]):
        with av.open(video_path(root, info, key, int(chunk), int(file))) as container:
            for i, ep in in_file.iterrows():
                digest = hashlib.sha256()
                for packet in packets(container, ep[col + "from_timestamp"], ep[col + "to_timestamp"]):
                    digest.update(bytes(packet))
                    try:
                        packet.decode()
                    except av.error.InvalidDataError:
                        bad += 1
                if hashes is not None and digest.hexdigest() != hashes[i]:
                    mismatched.append(int(ep["episode_index"]))
    n_files = episodes.groupby([col + "chunk_index", col + "file_index"]).ngroups
    print(f"{key}: {n_files} MP4s, {len(episodes)} episodes, {bad} packets that fail to decode"
          + ("" if hashes is None else f", {len(mismatched)} episodes whose packets differ from the source {mismatched[:10]}"), flush=True)
    return bad == 0 and not mismatched


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("src")
    parser.add_argument("out")
    parser.add_argument("--video_files_size_in_mb", type=float, default=100)
    parser.add_argument("--check_only", action="store_true")
    args = parser.parse_args()
    info = json.load(open(os.path.join(args.src, "meta", "info.json")))
    keys = [k for k, f in info["features"].items() if f["dtype"] == "video"]
    if args.check_only:
        out_info = json.load(open(os.path.join(args.out, "meta", "info.json")))
        assert all([check(args.out, key, out_info, read_episodes(args.out)) for key in keys]), "damaged MP4s"
        return

    episodes = read_episodes(args.src)
    shutil.copytree(os.path.join(args.src, "data"), os.path.join(args.out, "data"))
    shutil.copytree(os.path.join(args.src, "meta"), os.path.join(args.out, "meta"), ignore=shutil.ignore_patterns("episodes"))
    new, hashes = episodes.copy(), {}
    for key in keys:
        columns, hashes[key] = relayout(args.src, args.out, key, info, episodes, args.video_files_size_in_mb * 1024 ** 2)
        for c, values in columns.items():
            new[f"videos/{key}/{c}"] = values
    os.makedirs(os.path.join(args.out, "meta", "episodes", "chunk-000"))
    new["meta/episodes/chunk_index"], new["meta/episodes/file_index"] = 0, 0  # all the episodes' rows are in that one file
    new.to_parquet(os.path.join(args.out, "meta", "episodes", "chunk-000", "file-000.parquet"), index=False)
    info["video_files_size_in_mb"] = args.video_files_size_in_mb
    json.dump(info, open(os.path.join(args.out, "meta", "info.json"), "w"), indent=4)
    print(f"{len(episodes)} episodes into {args.out}", flush=True)
    assert all([check(args.out, key, info, new, hashes[key]) for key in keys]), "damaged MP4s"


if __name__ == "__main__":
    main()
