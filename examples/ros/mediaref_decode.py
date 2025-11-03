#!/usr/bin/env python3
"""Read MediaRef-converted bag files (ROS1/ROS2).

MediaRef bags contain JSON references to video files instead of raw image data:
- Original: CompressedImage messages (large binary data)
- MediaRef: String messages with JSON like {"uri": "output.media/camera.mp4", "pts_ns": 123456789}
"""

import argparse
import sys
from pathlib import Path

import cv2
from rosbags.rosbag1 import Reader as Rosbag1Reader
from rosbags.rosbag2 import Reader as Rosbag2Reader
from rosbags.typesys import Stores, get_typestore

from mediaref import MediaRef, batch_decode


def detect_format(path: Path) -> str:
    """Detect bag format."""
    if path.is_file() and path.suffix == ".bag":
        return "rosbag1"
    elif path.is_dir() and (path / "metadata.yaml").exists():
        return "rosbag2"
    raise ValueError(f"Unknown format: {path}")


def read_mediaref_messages(bag_path: Path, max_count: int, fmt: str) -> list[tuple[MediaRef, str]]:
    """Read MediaRef messages from bag and return list of (MediaRef, topic) tuples."""
    if fmt == "rosbag1":
        Reader = Rosbag1Reader
        typestore = get_typestore(Stores.ROS1_NOETIC)
        deserialize = typestore.deserialize_ros1
        is_string_msg = lambda msgtype: "String" in msgtype  # noqa: E731
    else:  # rosbag2
        Reader = Rosbag2Reader
        typestore = get_typestore(Stores.ROS2_HUMBLE)
        deserialize = typestore.deserialize_cdr
        is_string_msg = lambda msgtype: msgtype == "std_msgs/msg/String"  # noqa: E731

    refs = []
    with Reader(bag_path) as reader:
        for connection, _, rawdata in reader.messages():
            if not is_string_msg(connection.msgtype):
                continue

            msg = deserialize(rawdata, connection.msgtype)
            ref = MediaRef.model_validate_json(msg.data)
            refs.append((ref, connection.topic))

            if len(refs) >= max_count:
                break

    return refs


def display_messages(refs: list[tuple[MediaRef, str]]):
    """Display MediaRef messages."""
    print(f"\nMediaRef messages ({len(refs)}):\n")
    for i, (ref, topic) in enumerate(refs):
        print(f"[{i}] {topic}: {ref.uri} @ {ref.pts_ns / 1e9:.3f}s")


def decode_and_save(refs: list[tuple[MediaRef, str]], bag_path: Path, output_dir: Path):
    """Decode frames from MediaRef and save to files."""
    # Resolve relative paths
    bag_dir = str(bag_path.parent)
    resolved_refs = [ref.resolve_relative_path(bag_dir) for ref, _ in refs]

    # Batch decode
    print(f"\nDecoding {len(resolved_refs)} frames...")
    frames = batch_decode(resolved_refs, decoder="pyav")
    print(f"Decoded {len(frames)} frames (shape: {frames[0].shape}, dtype: {frames[0].dtype})")

    # Save frames organized by topic
    output_dir.mkdir(exist_ok=True, parents=True)
    topic_counters = {}

    for frame, (_, topic) in zip(frames, refs):
        topic_name = topic.replace("/", "_").strip("_")
        topic_dir = output_dir / topic_name
        topic_dir.mkdir(exist_ok=True)

        if topic not in topic_counters:
            topic_counters[topic] = 0

        frame_idx = topic_counters[topic]
        topic_counters[topic] += 1

        filename = topic_dir / f"frame_{frame_idx:04d}.jpg"
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(filename), frame_bgr)

    # Summary
    print(f"\nSaved to {output_dir}/:")
    for topic, count in topic_counters.items():
        topic_name = topic.replace("/", "_").strip("_")
        print(f"  {topic_name}/: {count} frames")


def main():
    parser = argparse.ArgumentParser(
        description="Read MediaRef-converted bag files (auto-detects ROS1/ROS2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("bag_path", type=Path, help="Path to bag file or directory")
    parser.add_argument("-n", "--max-messages", type=int, default=30, help="Max messages (default: 30)")
    parser.add_argument("--decode", action="store_true", help="Decode frames and save to files")
    parser.add_argument("-o", "--output", type=Path, default=Path("decoded_frames"), help="Output directory")
    args = parser.parse_args()

    if not args.bag_path.exists():
        print(f"Error: Path not found: {args.bag_path}", file=sys.stderr)
        raise SystemExit(1)

    try:
        fmt = detect_format(args.bag_path)
        refs = read_mediaref_messages(args.bag_path, args.max_messages, fmt)

        if args.decode:
            decode_and_save(refs, args.bag_path, args.output)
        else:
            display_messages(refs)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
