#!/usr/bin/env python3
"""
Show contents of MCAP files using mcap library.
"""

from pathlib import Path
from mcap.reader import make_reader
from datetime import datetime
import argparse


def format_timestamp(timestamp_ns):
    """Convert nanosecond timestamp to readable format."""
    timestamp_s = timestamp_ns / 1e9
    return datetime.fromtimestamp(timestamp_s).strftime("%Y-%m-%d %H:%M:%S")


def show_mcap_contents(mcap_path, max_messages=1):
    """Show contents of MCAP file."""
    mcap_path = Path(mcap_path)

    print(f"MCAP FILE: {mcap_path.name}")

    with open(mcap_path, "rb") as f:
        reader = make_reader(f)
        summary = reader.get_summary()

        duration = 0
        if summary and summary.statistics:
            stats = summary.statistics
            duration = (stats.message_end_time - stats.message_start_time) / 1e9
            print(f"Duration: {duration:.2f} seconds")
            print(f"Start time: {format_timestamp(stats.message_start_time)}")
            print(f"End time: {format_timestamp(stats.message_end_time)}")
            print(f"Message count: {stats.message_count:,}")
            print(f"Channels: {stats.channel_count}")

        print("\nCHANNELS:")

        if summary and summary.channels:
            for channel_id, channel in summary.channels.items():
                schema = summary.schemas.get(channel.schema_id)
                schema_name = schema.name if schema else "unknown"

                msg_count = sum(1 for _ in reader.iter_messages(topics=[channel.topic]))
                freq = msg_count / duration if duration > 0 else 0

                print(
                    f"  {channel.topic}: {schema_name} [{channel.message_encoding}] ({msg_count:,} msgs, {freq:.2f} Hz)"
                )

        if max_messages > 0:
            print("\nSAMPLE MESSAGES:")

            topic_message_count = {}

            for schema, channel, message in reader.iter_messages():
                topic = channel.topic

                if topic not in topic_message_count:
                    topic_message_count[topic] = 0

                if topic_message_count[topic] >= max_messages:
                    continue

                topic_message_count[topic] += 1

                schema_name = schema.name if schema else "unknown"
                print(f"\n[{topic}] {schema_name} [{channel.message_encoding}] @ {format_timestamp(message.log_time)}")
                print(f"  Data size: {len(message.data)} bytes, Sequence: {message.sequence}")
                if len(message.data) > 0:
                    preview = message.data[:100]
                    print(f"  Data preview: {preview.hex()[:200]}...")


def main():
    parser = argparse.ArgumentParser(description="Show contents of MCAP files")
    parser.add_argument("mcap_file", type=str, help="Path to MCAP file")
    parser.add_argument("--max-messages", type=int, default=1, help="Maximum messages to show per topic (default: 1)")
    args = parser.parse_args()
    show_mcap_contents(args.mcap_file, args.max_messages)


if __name__ == "__main__":
    main()
