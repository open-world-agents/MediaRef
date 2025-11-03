#!/usr/bin/env python3
"""Convert ROS bag files with embedded CompressedImage messages to MediaRef format."""

import argparse
import sys
from pathlib import Path

import av
import cv2
import numpy as np
from rosbags.rosbag1 import Reader as Reader1
from rosbags.rosbag1 import Writer as Writer1
from rosbags.rosbag2 import Reader as Reader2
from rosbags.rosbag2 import Writer as Writer2
from rosbags.typesys import Stores, get_typestore
from rosbags.typesys.base import TypesysError
from tqdm import tqdm

from mediaref import MediaRef


# NOTE: We must deal with bagfile with short duration / with different duration than embedded images. e.g. 0d83527337fb0277195e6c3264d84804.bag
class VideoWriter:
    """Video encoder using PyAV with CFR and precise timestamps."""

    def __init__(self, output_path: Path, fps: float = 30.0):
        self.output_path = output_path
        self.fps = fps
        self.container = None
        self.stream = None
        self._closed = False
        self.frame_count = 0

    def add_frame(self, image_data: bytes):
        """Add compressed image frame to video.

        Args:
            image_data: JPEG compressed image data
        """
        img_array = np.frombuffer(image_data, dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if frame is None:
            return

        if self.container is None:
            self._init_writer(frame.shape[1], frame.shape[0])

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_frame = av.VideoFrame.from_ndarray(frame_rgb, format="rgb24")

        # Use frame count as pts for CFR
        video_frame.pts = self.frame_count
        self.frame_count += 1

        for packet in self.stream.encode(video_frame):
            self.container.mux(packet)

    def _init_writer(self, width: int, height: int):
        """Initialize video encoder with CFR."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        gop_size = 30  # Ensure keyframe every 30 frames

        self.container = av.open(str(self.output_path), mode="w")
        self.stream = self.container.add_stream("h264", rate=int(self.fps))
        self.stream.width = width
        self.stream.height = height
        self.stream.pix_fmt = "yuv420p"
        self.stream.codec_context.gop_size = gop_size
        # Set options for fixed keyframe interval
        self.stream.codec_context.options = {
            "g": str(gop_size),
            "sc_threshold": "0",  # Disable scenecut
            "bf": "0",  # Disable B-frames (only I and P frames)
        }

    def close(self):
        """Finalize video file."""
        if self._closed:
            return
        if self.stream:
            for packet in self.stream.encode():
                self.container.mux(packet)
        if self.container:
            self.container.close()
        self._closed = True


def convert_bag(
    input_path: Path, output_path: Path, media_dir: Path, fps: float, keyframe_interval_sec: float, fmt: str
):
    """Convert ROS bag to MediaRef format."""
    # Select Reader/Writer and serialization based on format
    if fmt == "rosbag1":
        Reader = Reader1
        typestore = get_typestore(Stores.ROS1_NOETIC)
        deserialize = typestore.deserialize_ros1
        serialize = typestore.serialize_ros1
        writer_factory = lambda path: Writer1(path)  # noqa: E731
    else:  # rosbag2
        Reader = Reader2
        typestore = get_typestore(Stores.ROS2_JAZZY)
        deserialize = typestore.deserialize_cdr
        serialize = typestore.serialize_cdr
        writer_factory = lambda path: Writer2(path, version=9)  # noqa: E731

    video_writers = {}
    frame_indices = {}  # Track frame index per topic for CFR

    # Media directory name for URI (e.g., "output.media")
    media_dir_name = media_dir.name

    # Get bag start time and duration
    with Reader(input_path) as reader:
        bag_start_time_ns = reader.start_time
        duration_ns = reader.duration

    # Pass 1: Extract images to videos
    with Reader(input_path) as reader:
        for conn in reader.connections:
            if "CompressedImage" in conn.msgtype:
                topic_name = conn.topic.strip("/").replace("/", "_")
                video_path = media_dir / f"{topic_name}.mp4"
                video_writers[conn.topic] = VideoWriter(video_path, fps, keyframe_interval_sec)
                frame_indices[conn.topic] = 0

        if not video_writers:
            print("Error: No CompressedImage topics found", file=sys.stderr)
            raise SystemExit(1)

        with tqdm(
            total=duration_ns / 1e9,
            desc="Encoding videos",
            unit="s",
            bar_format="{l_bar}{bar}| {n:.2f}/{total:.2f}s [{elapsed}<{remaining}]",
        ) as pbar:
            last_time = bag_start_time_ns
            for connection, timestamp, rawdata in reader.messages():
                if connection.topic not in video_writers:
                    continue

                msg = deserialize(rawdata, connection.msgtype)
                video_writers[connection.topic].add_frame(bytes(msg.data))
                frame_indices[connection.topic] += 1

                pbar.update((timestamp - last_time) / 1e9)
                last_time = timestamp

    for writer in video_writers.values():
        writer.close()

    # Pass 2: Create MediaRef bag
    # Reset frame indices for Pass 2
    current_frame_indices = {topic: 0 for topic in video_writers}

    with Reader(input_path) as reader, writer_factory(output_path) as writer:
        conn_map = {}
        for connection in reader.connections:
            msgtype = "std_msgs/msg/String" if connection.topic in video_writers else connection.msgtype

            # Try to add connection, skip if type is unknown. TODO: check whether this is safe for `tf2_msgs/msg/TFMessage`
            try:
                conn = writer.add_connection(connection.topic, msgtype, typestore=typestore)
                conn_map[connection.id] = conn
            except TypesysError as e:
                print(f"Warning: Skipping topic {connection.topic} ({msgtype}): {e}", file=sys.stderr)
                continue

        with tqdm(
            total=duration_ns / 1e9,
            desc="Writing bag",
            unit="s",
            bar_format="{l_bar}{bar}| {n:.2f}/{total:.2f}s [{elapsed}<{remaining}]",
        ) as pbar:
            last_time = bag_start_time_ns
            for connection, timestamp, rawdata in reader.messages():
                # Skip if connection was not added (unknown type)
                if connection.id not in conn_map:
                    continue

                if connection.topic in video_writers:
                    # Calculate CFR-based pts_ns from frame index
                    frame_idx = current_frame_indices[connection.topic]
                    cfr_pts_ns = int(frame_idx * 1_000_000_000 / fps)
                    current_frame_indices[connection.topic] += 1

                    topic_name = connection.topic.strip("/").replace("/", "_")
                    ref = MediaRef(uri=f"{media_dir_name}/{topic_name}.mp4", pts_ns=cfr_pts_ns)
                    ref_msg = typestore.types["std_msgs/msg/String"](data=ref.model_dump_json())
                    ref_data = serialize(ref_msg, "std_msgs/msg/String")
                    writer.write(conn_map[connection.id], timestamp, ref_data)
                else:
                    writer.write(conn_map[connection.id], timestamp, rawdata)

                pbar.update((timestamp - last_time) / 1e9)
                last_time = timestamp

    print_stats(input_path, output_path, media_dir)


def detect_format(path: Path) -> str:
    """Detect bag format."""
    if path.is_file() and path.suffix == ".bag":
        return "rosbag1"
    elif path.is_dir() and (path / "metadata.yaml").exists():
        return "rosbag2"
    raise ValueError(f"Unknown format: {path}")


def print_stats(input_path: Path, output_path: Path, media_dir: Path):
    """Print file size statistics."""
    if input_path.is_file():
        input_size = input_path.stat().st_size
    else:
        # ROS2 bag is a directory
        input_size = sum(f.stat().st_size for f in input_path.rglob("*") if f.is_file())

    if output_path.is_file():
        output_size = output_path.stat().st_size
    else:
        output_size = sum(f.stat().st_size for f in output_path.rglob("*") if f.is_file())

    media_size = sum(f.stat().st_size for f in media_dir.glob("*.mp4"))
    total_size = output_size + media_size

    mb = 1024 * 1024
    print(f"\nOriginal: {input_size / mb:.1f} MB")
    print(f"MediaRef bag: {output_size / mb:.1f} MB")
    print(f"Videos: {media_size / mb:.1f} MB")
    print(f"Total: {total_size / mb:.1f} MB ({total_size / input_size * 100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Convert ROS bag files to MediaRef format (auto-detects ROS1/ROS2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", type=Path, help="Input bag file (.bag) or directory (ROS2)")
    parser.add_argument("-o", "--output", type=Path, help="Output bag path (auto-generated if not specified)")
    parser.add_argument("--fps", type=float, default=30.0, help="Video FPS (default: 30.0)")
    parser.add_argument(
        "--keyframe-interval", type=float, default=1.0, help="Keyframe interval in seconds (default: 1.0)"
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress progress bars")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: Input not found: {args.input}", file=sys.stderr)
        raise SystemExit(1)

    # Detect format
    try:
        fmt = detect_format(args.input)
        print(f"Detected format: {fmt}")
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        raise SystemExit(1)

    # Determine output path and media directory
    if fmt == "rosbag1":
        output_path = args.output or args.input.parent / f"{args.input.stem}_mediaref.bag"
        media_dir = output_path.parent / f"{output_path.stem}.media"
    else:  # rosbag2
        output_path = args.output or args.input.parent / f"{args.input.name}_mediaref"
        media_dir = output_path.parent / f"{output_path.name}.media"

    if output_path.exists():
        print(f"Error: Output already exists: {output_path}", file=sys.stderr)
        raise SystemExit(1)

    if media_dir.exists():
        print(f"Error: Media directory already exists: {media_dir}", file=sys.stderr)
        raise SystemExit(1)

    media_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_path}")
    print(f"Media: {media_dir}")

    convert_bag(args.input, output_path, media_dir, args.fps, args.keyframe_interval, fmt)


if __name__ == "__main__":
    main()
