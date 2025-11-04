#!/usr/bin/env python3
"""Show contents of ROS bag files (ROS1/ROS2)."""

import argparse
from datetime import datetime
from pathlib import Path

from rosbags.rosbag1 import Reader as Rosbag1Reader
from rosbags.rosbag2 import Reader as Rosbag2Reader
from rosbags.typesys import Stores, get_typestore


def detect_format(path: Path) -> str:
    """Detect bag format from path."""
    if path.is_file() and path.suffix == ".bag":
        return "rosbag1"
    elif path.is_dir() and (path / "metadata.yaml").exists():
        return "rosbag2"
    raise ValueError(f"Unknown bag format: {path}")


def format_timestamp(timestamp_ns: int) -> str:
    """Convert nanosecond timestamp to readable format."""
    return datetime.fromtimestamp(timestamp_ns / 1e9).strftime("%Y-%m-%d %H:%M:%S")


def format_value(value, indent: int = 0) -> str:
    """Format message values for display."""
    indent_str = "  " * indent

    if hasattr(value, "__msgtype__"):
        lines = []
        for field in dir(value):
            if not field.startswith("_"):
                field_value = getattr(value, field)
                if hasattr(field_value, "__msgtype__"):
                    lines.append(f"{indent_str}{field}:")
                    lines.append(format_value(field_value, indent + 1))
                elif isinstance(field_value, __import__('numpy').ndarray):
                    lines.append(f"{indent_str}{field}: ndarray")
                else:
                    lines.append(f"{indent_str}{field}: {field_value}")
        return "\n".join(lines)
    return f"{indent_str}{value}"


def show_rosbag(bag_path: Path, max_messages: int, fmt: str):
    """Show contents of ROS bag file."""
    if fmt == "rosbag1":
        print(f"ROS1 BAG FILE: {bag_path.name}")
        Reader = Rosbag1Reader
        typestore = get_typestore(Stores.ROS1_NOETIC)
        deserialize = typestore.deserialize_ros1
    else:  # rosbag2
        print(f"ROS2 BAG DIRECTORY: {bag_path.name}")
        Reader = Rosbag2Reader
        typestore = get_typestore(Stores.ROS2_JAZZY)
        deserialize = typestore.deserialize_cdr

    with Reader(bag_path) as reader:
        duration = reader.duration / 1e9
        print(f"Duration: {duration:.2f}s | Messages: {reader.message_count:,} | Topics: {len(reader.topics)}")
        print(f"Time: {format_timestamp(reader.start_time)} → {format_timestamp(reader.end_time)}")

        # Collect topic info
        topic_info = {}
        for connection in reader.connections:
            topic = connection.topic
            if topic not in topic_info:
                topic_info[topic] = {"msgtype": connection.msgtype, "count": 0}
            topic_info[topic]["count"] += connection.msgcount

        print(f"\nTOPICS ({len(topic_info)}):")
        for topic, info in sorted(topic_info.items()):
            freq = info["count"] / duration if duration > 0 else 0
            print(f"  {topic}: {info['msgtype']} ({info['count']:,} msgs, {freq:.2f} Hz)")

        if max_messages > 0:
            print(f"\nSAMPLE MESSAGES (max {max_messages} per topic):")
            topic_message_count = {}

            for connection, timestamp, rawdata in reader.messages():
                topic = connection.topic
                if topic not in topic_message_count:
                    topic_message_count[topic] = 0
                if topic_message_count[topic] >= max_messages:
                    continue

                topic_message_count[topic] += 1
                msg = deserialize(rawdata, connection.msgtype)
                print(f"\n[{topic}] {connection.msgtype} @ {format_timestamp(timestamp)}")
                print(format_value(msg, 1))


def main():
    parser = argparse.ArgumentParser(
        description="Show contents of ROS bag files (auto-detects ROS1/ROS2 format)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("bag_path", type=Path, help="Path to bag file or directory")
    parser.add_argument(
        "-n", "--max-messages", type=int, default=1, help="Max messages to show per topic (default: 1, 0=none)"
    )
    args = parser.parse_args()

    if not args.bag_path.exists():
        print(f"Error: Path not found: {args.bag_path}", file=__import__("sys").stderr)
        raise SystemExit(1)

    try:
        fmt = detect_format(args.bag_path)
        show_rosbag(args.bag_path, args.max_messages, fmt)
    except Exception as e:
        print(f"Error: Failed to read bag: {e}", file=__import__("sys").stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
