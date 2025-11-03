#!/usr/bin/env python3
"""
Show contents of ROS1 .bag files using rosbags library.
"""

from pathlib import Path
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore
from datetime import datetime
import argparse


def format_timestamp(timestamp_ns):
    """Convert nanosecond timestamp to readable format."""
    timestamp_s = timestamp_ns / 1e9
    return datetime.fromtimestamp(timestamp_s).strftime("%Y-%m-%d %H:%M:%S")


def format_value(value, indent=0):
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
                elif str(type(field_value).__name__) == "ndarray":
                    lines.append(f"{indent_str}{field}: ndarray")
                else:
                    lines.append(f"{indent_str}{field}: {field_value}")
        return "\n".join(lines)
    else:
        return f"{indent_str}{value}"


def show_bag_contents(bag_path, max_messages=1):
    """Show contents of ROS1 bag file."""
    bag_path = Path(bag_path)

    print(f"ROS1 BAG FILE: {bag_path.name}")

    typestore = get_typestore(Stores.ROS1_NOETIC)

    with Reader(bag_path) as reader:
        duration = reader.duration / 1e9
        print(f"Duration: {duration:.2f} seconds")
        print(f"Start time: {format_timestamp(reader.start_time)}")
        print(f"End time: {format_timestamp(reader.end_time)}")
        print(f"Message count: {reader.message_count:,}")
        print(f"Topics: {len(reader.topics)}")

        print("\nTOPICS:")

        topic_info = {}
        for connection in reader.connections:
            topic = connection.topic
            if topic not in topic_info:
                topic_info[topic] = {"msgtype": connection.msgtype, "count": 0}
            topic_info[topic]["count"] += connection.msgcount

        for topic, info in sorted(topic_info.items()):
            freq = info["count"] / duration if duration > 0 else 0
            print(f"  {topic}: {info['msgtype']} ({info['count']:,} msgs, {freq:.2f} Hz)")

        if max_messages > 0:
            print("\nSAMPLE MESSAGES:")

            topic_message_count = {}

            for connection, timestamp, rawdata in reader.messages():
                topic = connection.topic

                if topic not in topic_message_count:
                    topic_message_count[topic] = 0

                if topic_message_count[topic] >= max_messages:
                    continue

                topic_message_count[topic] += 1

                msg = typestore.deserialize_ros1(rawdata, connection.msgtype)

                print(f"\n[{topic}] {connection.msgtype} @ {format_timestamp(timestamp)}")
                print(format_value(msg, 1))


def main():
    parser = argparse.ArgumentParser(description="Show contents of ROS1 .bag files")
    parser.add_argument("bag_file", type=str, help="Path to ROS1 .bag file")
    parser.add_argument("--max-messages", type=int, default=1, help="Maximum messages to show per topic (default: 1)")
    args = parser.parse_args()
    show_bag_contents(args.bag_file, args.max_messages)


if __name__ == "__main__":
    main()
