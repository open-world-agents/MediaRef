#!/usr/bin/env python3
"""
Convert ROS bag files with embedded CompressedImage messages to MediaRef format.

This script extracts embedded images to external video files and replaces
CompressedImage messages with lightweight MediaRef references.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List

from rosbags.rosbag1 import Reader as Rosbag1Reader, Writer as Rosbag1Writer
from rosbags.typesys import Stores, get_typestore
import cv2
import numpy as np
import av
from mediaref import MediaRef
from tqdm import tqdm


class VideoWriter:
    """Manages video encoding for a single topic using PyAV with fixed keyframe interval."""

    def __init__(self, output_path: Path, fps: float = 30.0, keyframe_interval_sec: float = 1.0):
        self.output_path = output_path
        self.fps = fps
        self.keyframe_interval_sec = keyframe_interval_sec
        self.container = None
        self.stream = None
        self.frame_count = 0
        self.width = None
        self.height = None

    def add_frame(self, image_data: bytes, timestamp_ns: int):
        """Add a frame to the video."""
        try:
            # Decode compressed image (JPEG/PNG) using OpenCV
            img_array = np.frombuffer(image_data, dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            if frame is None:
                print("Warning: Failed to decode frame")
                return

            # Initialize writer on first frame
            if self.container is None:
                self._init_writer(frame.shape[1], frame.shape[0])

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Create PyAV VideoFrame
            video_frame = av.VideoFrame.from_ndarray(frame_rgb, format="rgb24")
            video_frame.pts = self.frame_count

            # Encode frame
            for packet in self.stream.encode(video_frame):
                self.container.mux(packet)

            self.frame_count += 1

        except Exception as e:
            print(f"Warning: Failed to process frame: {e}")

    def _init_writer(self, width: int, height: int):
        """Initialize PyAV container with fixed keyframe interval."""
        try:
            self.width = width
            self.height = height
            self.output_path.parent.mkdir(parents=True, exist_ok=True)

            # Calculate keyframe interval in frames
            keyframe_interval = max(1, int(self.fps * self.keyframe_interval_sec))

            # Open container
            self.container = av.open(str(self.output_path), mode="w")

            # Add video stream with H.264 codec
            self.stream = self.container.add_stream("h264", rate=int(self.fps))
            self.stream.width = width
            self.stream.height = height
            self.stream.pix_fmt = "yuv420p"

            # Set keyframe interval (GOP size)
            self.stream.gop_size = keyframe_interval

            # Set codec options for fixed keyframe interval
            self.stream.options = {
                "crf": "23",  # Quality
                "g": str(keyframe_interval),  # GOP size
                "keyint_min": str(keyframe_interval),  # Min keyframe interval
                "sc_threshold": "0",  # Disable scene change detection
            }
        except Exception as e:
            print(f"Error initializing video writer: {e}")
            raise

    def close(self):
        """Finalize and close the video file."""
        if self.stream is not None:
            # Flush encoder
            for packet in self.stream.encode():
                self.container.mux(packet)

        if self.container is not None:
            self.container.close()


def detect_format(input_path: Path) -> str:
    """Detect input format: rosbag1, rosbag2, or mcap."""
    if input_path.is_file():
        if input_path.suffix == ".bag":
            return "rosbag1"
        elif input_path.suffix == ".mcap":
            return "mcap"
    elif input_path.is_dir():
        # Check for ROS2 bag metadata
        if (input_path / "metadata.yaml").exists():
            return "rosbag2"

    raise ValueError(f"Cannot detect format for: {input_path}")


def sanitize_topic_name(topic: str) -> str:
    """Convert topic name to valid filename."""
    return topic.strip("/").replace("/", "_")


def convert_rosbag1(input_path: Path, output_path: Path, media_dir: Path, fps: float):
    """Convert ROS1 bag file to MediaRef format."""
    typestore = get_typestore(Stores.ROS1_NOETIC)
    video_writers: Dict[str, VideoWriter] = {}
    image_topics: List[str] = []
    first_timestamps: Dict[str, int] = {}  # Track first timestamp per topic

    print(f"Reading ROS1 bag: {input_path}")

    # First pass: identify image topics and create videos
    with Rosbag1Reader(input_path) as reader:
        # Find CompressedImage topics
        for conn in reader.connections:
            if "CompressedImage" in conn.msgtype:
                image_topics.append(conn.topic)
                video_name = sanitize_topic_name(conn.topic)
                video_path = media_dir / f"{video_name}.mp4"
                video_writers[conn.topic] = VideoWriter(video_path, fps=fps)
                print(f"  Found image topic: {conn.topic}")

        if not image_topics:
            print("Warning: No CompressedImage topics found!")
            return

        # Extract images to videos
        print("\nExtracting images to video files...")
        with tqdm(desc="Encoding videos", unit="msg") as pbar:
            for connection, timestamp, rawdata in reader.messages():
                if connection.topic in image_topics:
                    msg = typestore.deserialize_ros1(rawdata, connection.msgtype)

                    # Calculate nanosecond timestamp
                    pts_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec

                    # Track first timestamp for relative timing
                    if connection.topic not in first_timestamps:
                        first_timestamps[connection.topic] = pts_ns

                    # Add frame to video
                    video_writers[connection.topic].add_frame(bytes(msg.data), pts_ns)
                    pbar.update(1)

    # Close all video writers
    print("\nFinalizing video files...")
    for topic, writer in tqdm(video_writers.items(), desc="Closing videos", unit="video"):
        writer.close()
        tqdm.write(f"  {topic}: {writer.frame_count} frames → {writer.output_path}")

    # Second pass: create new bag with MediaRef messages
    print(f"\nCreating MediaRef bag: {output_path}")
    with Rosbag1Reader(input_path) as reader, Rosbag1Writer(output_path) as writer:
        # Create connections
        conn_map = {}
        for connection in reader.connections:
            if connection.topic in image_topics:
                # Replace with String message for MediaRef
                conn = writer.add_connection(connection.topic, "std_msgs/msg/String", typestore=typestore)
                conn_map[connection.id] = conn
            else:
                # Copy other connections as-is
                conn = writer.add_connection(connection.topic, connection.msgtype, typestore=typestore)
                conn_map[connection.id] = conn

        # Write messages
        with tqdm(desc="Writing bag", unit="msg") as pbar:
            for connection, timestamp, rawdata in reader.messages():
                if connection.topic in image_topics:
                    # Deserialize original message
                    msg = typestore.deserialize_ros1(rawdata, connection.msgtype)

                    # Create MediaRef with relative timestamp
                    video_name = sanitize_topic_name(connection.topic)
                    abs_pts_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
                    relative_pts_ns = abs_pts_ns - first_timestamps[connection.topic]

                    ref = MediaRef(uri=f"media/{video_name}.mp4", pts_ns=relative_pts_ns)

                    # Create String message with JSON
                    ref_msg = typestore.types["std_msgs/msg/String"](data=ref.model_dump_json())

                    # Serialize and write
                    ref_data = typestore.serialize_cdr(ref_msg, "std_msgs/msg/String")
                    writer.write(conn_map[connection.id], timestamp, ref_data)
                else:
                    # Copy other messages as-is
                    writer.write(conn_map[connection.id], timestamp, rawdata)

                pbar.update(1)

    print("\nConversion complete!")
    print_statistics(input_path, output_path, media_dir)


def print_statistics(input_path: Path, output_path: Path, media_dir: Path):
    """Print file size statistics."""
    input_size = (
        input_path.stat().st_size
        if input_path.is_file()
        else sum(f.stat().st_size for f in input_path.rglob("*") if f.is_file())
    )
    output_size = (
        output_path.stat().st_size
        if output_path.is_file()
        else sum(f.stat().st_size for f in output_path.rglob("*") if f.is_file())
    )
    media_size = sum(f.stat().st_size for f in media_dir.glob("*.mp4"))
    total_size = output_size + media_size

    print(f"\n{'=' * 60}")
    print("Storage Statistics:")
    print(f"{'=' * 60}")
    print(f"Original bag:     {input_size / 1024 / 1024:>8.2f} MB")
    print(f"MediaRef bag:     {output_size / 1024 / 1024:>8.2f} MB ({output_size / input_size * 100:>5.1f}%)")
    print(f"Video files:      {media_size / 1024 / 1024:>8.2f} MB")
    print(f"Total:            {total_size / 1024 / 1024:>8.2f} MB ({total_size / input_size * 100:>5.1f}%)")
    print(
        f"Savings:          {(input_size - total_size) / 1024 / 1024:>8.2f} MB ({(1 - total_size / input_size) * 100:>5.1f}%)"
    )
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert ROS bag files with embedded images to MediaRef format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s input.bag
  %(prog)s input.mcap --fps 30
  %(prog)s ros2_bag_directory --output converted.bag
        """,
    )

    parser.add_argument("input", type=Path, help="Input bag file or directory")
    parser.add_argument("--output", type=Path, help="Output bag file (default: auto-generate)")
    parser.add_argument("--fps", type=float, default=30.0, help="Video frame rate (default: 30.0)")

    args = parser.parse_args()

    # Validate input
    if not args.input.exists():
        print(f"Error: Input path does not exist: {args.input}")
        sys.exit(1)

    # Detect format
    try:
        format_type = detect_format(args.input)
        print(f"Detected format: {format_type}")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        if args.input.is_file():
            output_path = args.input.parent / f"{args.input.stem}_mediaref{args.input.suffix}"
        else:
            output_path = args.input.parent / f"{args.input.name}_mediaref"

    # Create media directory
    media_dir = output_path.parent / "media"
    media_dir.mkdir(parents=True, exist_ok=True)

    # Convert based on format
    if format_type == "rosbag1":
        convert_rosbag1(args.input, output_path, media_dir, args.fps)
    elif format_type == "rosbag2":
        print("Error: ROS2 bag conversion not yet implemented")
        sys.exit(1)
    elif format_type == "mcap":
        print("Error: MCAP conversion not yet implemented")
        sys.exit(1)


if __name__ == "__main__":
    main()
