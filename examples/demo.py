#!/usr/bin/env python3
"""MediaRef usage demonstration - follows README structure exactly.

This demo shows the actual code usage, not output formatting.
Run this file to see MediaRef in action.
"""

import shutil
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from mediaref import DataURI, MediaRef, batch_decode

# ============================================================================
# Setup: Create test files
# ============================================================================

tmp = Path(tempfile.mkdtemp())
image_path = tmp / "image.png"
video_path = tmp / "video.mp4"

# Create test image
test_image = np.zeros((48, 64, 3), dtype=np.uint8)
test_image[:, :, 0] = 255
cv2.imwrite(str(image_path), test_image)

# Create test video (if video support available)
video_available = False
try:
    import av

    video_available = True
    container = av.open(str(video_path), "w")
    stream = container.add_stream("h264", rate=30)
    stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
    for i in range(30):
        arr = np.full((48, 64, 3), min(i * 8, 255), dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
except ImportError:
    print("⚠️  Video support not available (install: pip install mediaref[video])\n")

# ============================================================================
# Basic Usage
# ============================================================================

print("\n1. Create references (lightweight, no loading yet)")
ref = MediaRef(uri=str(image_path))  # Local file
ref = MediaRef(uri="https://example.com/image.jpg")  # Remote URL
ref = MediaRef(uri=str(video_path), pts_ns=1_000_000_000)  # Video frame at 1.0s
print(f"   ✓ Created refs: local, remote (is_remote={ref.is_remote}), video (is_video={ref.is_video})")

print("\n2. Load media")
ref = MediaRef(uri=str(image_path))
rgb = ref.to_rgb_array()  # Returns (H, W, 3) numpy array
pil = ref.to_pil_image()  # Returns PIL.Image
print(f"   ✓ Loaded: rgb={rgb.shape}, pil={pil.size}")

print("\n3. Embed as data URI")
data_uri = DataURI.from_image(rgb, format="png")  # e.g., "data:image/png;base64,iVBORw0KG..."
ref = MediaRef(uri=data_uri)  # Self-contained reference
print(f"   ✓ Embedded: is_embedded={ref.is_embedded}, can load back: {ref.to_rgb_array().shape}")

# 4. Batch decode video frames (opens video once, reuses handle)
if video_available:
    print("\n4. Batch decode video frames - Performance comparison")
    refs = [MediaRef(uri=str(video_path), pts_ns=int(i * 0.1e9)) for i in range(10)]

    # Individual loading (naive approach)
    start = time.perf_counter()
    frames_individual = [ref.to_rgb_array() for ref in refs]
    time_individual = time.perf_counter() - start

    # Batch loading (optimized)
    start = time.perf_counter()
    frames_batch = batch_decode(refs)
    time_batch = time.perf_counter() - start

    speedup = time_individual / time_batch
    print(f"   ✓ Individual: {time_individual * 1000:.1f}ms, Batch: {time_batch * 1000:.1f}ms → {speedup:.1f}x faster")

# ============================================================================
# Batch Decoding - Optimized Video Frame Loading
# ============================================================================

if video_available:
    print("\n5. Batch decoding strategies - Performance comparison")
    from mediaref.video_decoder import BatchDecodingStrategy

    refs = [MediaRef(uri=str(video_path), pts_ns=int(i * 0.1e9)) for i in range(10)]

    # Strategy 1: SEQUENTIAL (simple, always works)
    start = time.perf_counter()
    frames = batch_decode(refs, decoder="pyav", strategy=BatchDecodingStrategy.SEQUENTIAL)
    time_sequential = time.perf_counter() - start

    # Strategy 2: SEQUENTIAL_PER_KEYFRAME_BLOCK (adaptive, recommended)
    start = time.perf_counter()
    frames = batch_decode(refs, decoder="pyav", strategy=BatchDecodingStrategy.SEQUENTIAL_PER_KEYFRAME_BLOCK)
    time_adaptive = time.perf_counter() - start

    print(f"   ✓ SEQUENTIAL: {time_sequential * 1000:.1f}ms")
    print(f"   ✓ SEQUENTIAL_PER_KEYFRAME_BLOCK: {time_adaptive * 1000:.1f}ms (adaptive, recommended)")

    # Or use TorchCodec for GPU-accelerated decoding
    # frames = batch_decode(refs, decoder="torchcodec")  # Requires: pip install torchcodec>=0.4.0

# ============================================================================
# Embedding Media Directly in MediaRef
# ============================================================================

print("\n6. Embedding media")
# Create embedded MediaRef from numpy array
rgb = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
embedded_ref = MediaRef(uri=DataURI.from_image(rgb, format="png"))

# Or from file
embedded_ref = MediaRef(uri=DataURI.from_file(str(image_path)))

# Or from PIL Image
pil_img = Image.open(str(image_path))
embedded_ref = MediaRef(uri=DataURI.from_image(pil_img, format="jpeg", quality=90))
print("   ✓ Created embedded refs from: numpy, file, PIL Image")

# Use just like any other MediaRef
rgb = embedded_ref.to_rgb_array()  # (H, W, 3) numpy array
pil = embedded_ref.to_pil_image()  # PIL Image

# Serialize with embedded data
serialized = embedded_ref.model_dump_json()  # Contains image data
restored = MediaRef.model_validate_json(serialized)  # No external file needed!
print(f"   ✓ Serialized: {len(serialized)} bytes, restored: {restored.to_rgb_array().shape}")

# Properties
data_uri = DataURI.from_image(rgb, format="png")
print(f"   ✓ DataURI: mimetype={data_uri.mimetype}, length={len(data_uri)}, is_image={data_uri.is_image}")

# ============================================================================
# Path Resolution & Serialization
# ============================================================================

print("\n7. Path resolution")
# Resolve relative paths
ref = MediaRef(uri="relative/video.mkv", pts_ns=123456)
resolved = ref.resolve_relative_path(str(tmp / "recordings"))
print(
    f"   ✓ Resolved: is_relative {ref.is_relative_path} → {resolved.is_relative_path}, pts_ns preserved={resolved.pts_ns == 123456}"
)

# Handle unresolvable URIs (embedded/remote)
remote = MediaRef(uri="https://example.com/image.jpg")
resolved = remote.resolve_relative_path(str(tmp), on_unresolvable="ignore")  # No warning
print(f"   ✓ Unresolvable URI unchanged: {resolved.uri == remote.uri}")

print("\n8. Serialization (Pydantic-based)")
# Serialization (Pydantic-based)
ref = MediaRef(uri=str(image_path))
data = ref.model_dump()  # {'uri': '...', 'pts_ns': ...}
json_str = ref.model_dump_json()  # JSON string
ref = MediaRef.model_validate(data)  # From dict
ref = MediaRef.model_validate_json(json_str)  # From JSON
print("   ✓ Serialized and restored from dict and JSON")

# ============================================================================
# Cleanup
# ============================================================================

try:
    shutil.rmtree(tmp)
except Exception:
    pass

print("✅ Demo completed successfully!")
