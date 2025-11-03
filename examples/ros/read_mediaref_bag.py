#!/usr/bin/env python3
"""
Minimal demo to read MediaRef-converted bag files.
Supports ROS1 bag, ROS2 bag, and MCAP formats.
Demonstrates batch decoding and saving frames.
"""

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
    elif path.is_file() and path.suffix == ".mcap":
        return "mcap"
    elif path.is_dir() and (path / "metadata.yaml").exists():
        return "rosbag2"
    raise ValueError(f"Unknown format: {path}")


def read_rosbag1(bag_path: Path, max_messages: int = 10):
    """Read ROS1 bag with MediaRef messages."""
    typestore = get_typestore(Stores.ROS1_NOETIC)

    with Rosbag1Reader(bag_path) as reader:
        # Find MediaRef topics (String messages - can be ROS1 or ROS2 format)
        mediaref_topics = [conn for conn in reader.connections if "String" in conn.msgtype]

        print(f"Found {len(mediaref_topics)} MediaRef topics:")
        for conn in mediaref_topics:
            print(f"  {conn.topic}")

        print(f"\nReading first {max_messages} MediaRef messages:\n")

        count = 0
        for connection, timestamp, rawdata in reader.messages():
            if "String" not in connection.msgtype:
                continue

            # Deserialize String message
            msg = typestore.deserialize_ros1(rawdata, connection.msgtype)

            # Parse MediaRef from JSON
            ref = MediaRef.model_validate_json(msg.data)

            print(f"[{count}] Topic: {connection.topic}")
            print(f"    URI: {ref.uri}")
            print(f"    PTS: {ref.pts_ns / 1e9:.3f}s")
            print()

            count += 1
            if count >= max_messages:
                break


def read_rosbag2(bag_path: Path, max_messages: int = 10):
    """Read ROS2 bag with MediaRef messages."""
    typestore = get_typestore(Stores.ROS2_FOXY)

    with Rosbag2Reader(bag_path) as reader:
        mediaref_topics = [conn for conn in reader.connections if conn.msgtype == "std_msgs/msg/String"]

        print(f"Found {len(mediaref_topics)} MediaRef topics:")
        for conn in mediaref_topics:
            print(f"  {conn.topic}")

        print(f"\nReading first {max_messages} MediaRef messages:\n")

        count = 0
        for connection, timestamp, rawdata in reader.messages():
            if connection.msgtype != "std_msgs/msg/String":
                continue

            msg = typestore.deserialize_ros1(rawdata, connection.msgtype)
            ref = MediaRef.model_validate_json(msg.data)

            print(f"[{count}] Topic: {connection.topic}")
            print(f"    URI: {ref.uri}")
            print(f"    PTS: {ref.pts_ns / 1e9:.3f}s")
            print()

            count += 1
            if count >= max_messages:
                break


def batch_decode_demo(bag_path: Path, output_dir: Path = Path("decoded_frames")):
    """Demo: Batch decode frames and save to files."""
    typestore = get_typestore(Stores.ROS1_NOETIC)

    print(f"\n{'=' * 60}")
    print("BATCH DECODE DEMO")
    print(f"{'=' * 60}\n")

    # Collect MediaRef objects from bag
    refs = []
    topics = []

    with Rosbag1Reader(bag_path) as reader:
        count = 0
        for connection, timestamp, rawdata in reader.messages():
            if "String" not in connection.msgtype:
                continue

            msg = typestore.deserialize_ros1(rawdata, connection.msgtype)
            ref = MediaRef.model_validate_json(msg.data)
            refs.append(ref)
            topics.append(connection.topic)

            count += 1
            if count >= 20:  # Batch decode 20 frames
                break

    print(f"Collected {len(refs)} MediaRef objects")
    print(f"Topics: {set(topics)}\n")

    # Batch decode frames
    print("Batch decoding frames...")
    frames = batch_decode(refs, decoder="pyav")

    print(f"Decoded {len(frames)} frames")
    if frames:
        print(f"Frame shape: {frames[0].shape}, dtype: {frames[0].dtype}\n")

    # Save frames to files organized by topic
    output_dir.mkdir(exist_ok=True)
    print(f"Saving frames to {output_dir}/")

    # Group frames by topic
    topic_counters = {}

    for i, (frame, topic) in enumerate(zip(frames, topics)):
        # Clean topic name for directory
        topic_name = topic.replace("/", "_").strip("_")

        # Create topic subdirectory
        topic_dir = output_dir / topic_name
        topic_dir.mkdir(exist_ok=True)

        # Track frame count per topic
        if topic not in topic_counters:
            topic_counters[topic] = 0

        frame_idx = topic_counters[topic]
        topic_counters[topic] += 1

        # Save with topic-specific numbering
        filename = topic_dir / f"frame_{frame_idx:04d}.jpg"

        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(filename), frame_bgr)

    # Print summary
    print()
    for topic, count in topic_counters.items():
        topic_name = topic.replace("/", "_").strip("_")
        print(f"  {topic_name}/: {count} frames")

    print(f"\n✓ Batch decode complete! Saved {len(frames)} frames to {output_dir}/")


def main():
    if len(sys.argv) < 2:
        print("Usage: python read_mediaref_bag.py <bag_file> [max_messages] [--batch-decode]")
        print("\nExamples:")
        print("  python read_mediaref_bag.py data_mediaref.bag")
        print("  python read_mediaref_bag.py data_mediaref.bag 5")
        print("  python read_mediaref_bag.py data_mediaref.bag 10 --batch-decode")
        sys.exit(1)

    bag_path = Path(sys.argv[1])

    # Check for --batch-decode flag
    batch_mode = "--batch-decode" in sys.argv

    # Get max_messages (skip --batch-decode if present)
    max_messages = 10
    for arg in sys.argv[2:]:
        if arg != "--batch-decode" and arg.isdigit():
            max_messages = int(arg)
            break

    if not bag_path.exists():
        print(f"Error: {bag_path} not found")
        sys.exit(1)

    fmt = detect_format(bag_path)

    if batch_mode:
        # Run batch decode demo
        batch_decode_demo(bag_path)
    else:
        # Run normal read demo
        print(f"Format: {fmt}\n")

        if fmt == "rosbag1":
            read_rosbag1(bag_path, max_messages)
        elif fmt == "rosbag2":
            read_rosbag2(bag_path, max_messages)
        else:
            print("MCAP format not yet implemented")


if __name__ == "__main__":
    main()
