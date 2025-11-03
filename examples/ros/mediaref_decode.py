#!/usr/bin/env python3
"""Read and decode MediaRef-converted bag files (ROS1/ROS2)."""

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


def read_bag(bag_path: Path, max_messages: int, fmt: str, decode_frames: bool = False, output_dir: Path | None = None):
    """Read bag with MediaRef messages and optionally decode frames."""
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

    with Reader(bag_path) as reader:
        mediaref_topics = [conn for conn in reader.connections if is_string_msg(conn.msgtype)]
        print(f"Found {len(mediaref_topics)} MediaRef topics: {[c.topic for c in mediaref_topics]}")

        # Collect MediaRef objects
        refs = []
        topics = []
        count = 0

        for connection, _, rawdata in reader.messages():
            if not is_string_msg(connection.msgtype):
                continue

            msg = deserialize(rawdata, connection.msgtype)
            ref = MediaRef.model_validate_json(msg.data)

            if decode_frames:
                refs.append(ref)
                topics.append(connection.topic)
                if len(refs) >= max_messages:
                    break
            else:
                print(f"[{count}] {connection.topic}: {ref.uri} @ {ref.pts_ns / 1e9:.3f}s")
                count += 1
                if count >= max_messages:
                    break

    # Batch decode if requested
    if decode_frames and refs and output_dir:
        print(f"\nCollected {len(refs)} MediaRef objects from topics: {set(topics)}")

        # Resolve relative paths against bag file directory
        bag_dir = str(bag_path.parent)
        refs = [ref.resolve_relative_path(bag_dir) for ref in refs]

        # Batch decode frames
        print("Batch decoding frames...")
        frames = batch_decode(refs, decoder="pyav")
        print(f"Decoded {len(frames)} frames (shape: {frames[0].shape}, dtype: {frames[0].dtype})\n")

        # Save frames to files organized by topic
        output_dir.mkdir(exist_ok=True, parents=True)
        print(f"Saving frames to {output_dir}/")

        topic_counters = {}
        for frame, topic in zip(frames, topics):
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

        # Print summary
        print("")
        for topic, count in topic_counters.items():
            topic_name = topic.replace("/", "_").strip("_")
            print(f"  {topic_name}/: {count} frames")

        print(f"\n✓ Batch decode complete! Saved {len(frames)} frames to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Read and decode MediaRef-converted bag files (auto-detects ROS1/ROS2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("bag_path", type=Path, help="Path to bag file or directory")
    parser.add_argument(
        "-n", "--max-messages", type=int, default=10, help="Max messages to display/decode (default: 10)"
    )
    parser.add_argument("--batch-decode", action="store_true", help="Decode frames and save to files")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("decoded_frames"),
        help="Output directory for batch decode (default: decoded_frames)",
    )
    args = parser.parse_args()

    if not args.bag_path.exists():
        print(f"Error: Path not found: {args.bag_path}", file=sys.stderr)
        raise SystemExit(1)

    try:
        fmt = detect_format(args.bag_path)
        print(f"Format: {fmt}\n")

        read_bag(
            args.bag_path,
            args.max_messages,
            fmt,
            decode_frames=args.batch_decode,
            output_dir=args.output if args.batch_decode else None,
        )
    except Exception as e:
        print(f"Error: Failed to read bag: {e}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
