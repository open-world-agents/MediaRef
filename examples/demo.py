#!/usr/bin/env python3
"""MediaRef usage demonstration - shows all features from README."""

import json
import tempfile
from pathlib import Path

import av
import cv2
import numpy as np
from mediaref import MediaRef, load_batch

# ============================================================
# Setup: Create test files for demonstration
# ============================================================
tmp = Path(tempfile.mkdtemp())
image_path = tmp / "test.png"
video_path = tmp / "test.mp4"

# Create a simple test image (100x150 pixels, BGR color)
cv2.imwrite(str(image_path), np.full((100, 150, 3), [50, 100, 150], dtype=np.uint8))

# Create a test video with 30 frames (1 second at 30fps)
# Each frame gets progressively brighter
container = av.open(str(video_path), "w")
stream = container.add_stream("h264", rate=30)
stream.width, stream.height, stream.pix_fmt = 150, 100, "yuv420p"
for i in range(30):
    arr = np.full((100, 150, 3), i * 8, dtype=np.uint8)
    for packet in stream.encode(av.VideoFrame.from_ndarray(arr, format="rgb24")):
        container.mux(packet)
for packet in stream.encode():
    container.mux(packet)
container.close()

print("MediaRef Demo\n" + "=" * 60)

# ============================================================
# 1. Reference Creation - Lightweight, no loading yet
# ============================================================
print("\n1. Create references (no loading yet)")

# Reference to a local image file
ref_image = MediaRef(uri=str(image_path))

# Reference to a specific video frame (at 0.5 seconds)
ref_video = MediaRef(uri=str(video_path), pts_ns=500_000_000)

# Reference to a remote URL (not fetched until loaded)
ref_url = MediaRef(uri="https://example.com/image.jpg")

print(f"   Image:  {ref_image.is_local=}, {ref_image.is_video=}")
print(f"   Video:  {ref_video.is_video=}, pts_ns={ref_video.pts_ns}")
print(f"   URL:    {ref_url.is_remote=}")

# ============================================================
# 2. Loading - Convert references to actual image data
# ============================================================
print("\n2. Load media")

# Load as NumPy array (RGB format, uint8)
rgb = ref_image.to_rgb_array()

# Load as PIL Image object
pil = ref_image.to_pil_image()

print(f"   to_rgb_array():   {rgb.shape}, {rgb.dtype}")
print(f"   to_pil_image():   {pil.size}, {pil.mode}")

# ============================================================
# 3. Batch Loading - Efficient loading with automatic caching
# ============================================================
print("\n3. Batch load video frames (with caching)")

# Create 10 references to different frames in the same video
refs = [MediaRef(uri=str(video_path), pts_ns=int(i * 0.1e9)) for i in range(10)]

# load_batch() opens the video container once and reuses it
frames = load_batch(refs)

print(f"   Loaded {len(frames)} frames: {frames[0].shape}")

# ============================================================
# 4. Embedding - Encode media as base64 data URIs
# ============================================================
print("\n4. Embed as data URI")

# Convert image to self-contained data URI (base64 encoded)
data_uri = ref_image.embed_as_data_uri(format="png")

# Create a new reference from the data URI
embedded = MediaRef(uri=data_uri)

print(f"   Data URI length: {len(data_uri)}")
print(f"   {embedded.is_embedded=}")
print(f"   Can load: {embedded.to_rgb_array().shape}")

# ============================================================
# 5. Path Resolution - Handle relative paths in datasets
# ============================================================
print("\n5. Path resolution (for MCAP/rosbag)")

# Simulate MCAP file structure with relative video paths
mcap_path = tmp / "recording.mcap"
mcap_path.touch()
(tmp / "videos").mkdir()

# Create reference with relative path
rel_ref = MediaRef(uri="videos/clip.mp4", pts_ns=123)

# Resolve relative path against MCAP file location
resolved = rel_ref.resolve_relative_path(str(mcap_path))

print(f"   Original:  '{rel_ref.uri}' ({rel_ref.is_relative_path=})")
print(f"   Resolved:  '{resolved.uri}' ({resolved.is_relative_path=})")

# ============================================================
# 6. Serialization - Save/load references as JSON
# ============================================================
print("\n6. Serialization (Pydantic)")

# Serialize to Python dict
data = ref_video.model_dump()

# Serialize to JSON string
json_str = ref_video.model_dump_json()

print(f"   model_dump():      {data}")
print(f"   model_dump_json(): {json_str}")

# Deserialize from JSON string (round-trip)
restored = MediaRef.model_validate_json(json_str)
print(f"   Restored: {restored.uri}, pts_ns={restored.pts_ns}")

# Practical example: Save dataset metadata to JSON file
dataset = {
    "frames": [
        MediaRef(uri=str(video_path), pts_ns=0).model_dump(),
        MediaRef(uri=str(image_path)).model_dump(),
    ]
}
json_path = tmp / "dataset.json"
json.dump(dataset, open(json_path, "w"))

# Load dataset metadata from JSON file
loaded = [MediaRef.model_validate(f) for f in json.load(open(json_path))["frames"]]
print(f"   Saved & loaded {len(loaded)} refs from JSON")

print("\n" + "=" * 60)
print("All features demonstrated successfully")
