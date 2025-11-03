# MediaRef for ROS

MediaRef replaces embedded CompressedImage data in ROS bags with references to external video files, reducing storage requirements by 70-90% through inter-frame compression.

## Overview

Traditional ROS bags store each image frame as an independent JPEG/PNG in a CompressedImage message (~50KB per frame). MediaRef instead stores a JSON reference (~60 bytes) pointing to a frame in an H.264/H.265 encoded video file.

```python
# Traditional: CompressedImage message
{
  "format": "jpeg",
  "data": b"\xff\xd8\xff\xe0..."  # 50KB JPEG
}

# MediaRef: String message
{
  "data": '{"uri": "bag.media/camera.mp4", "pts_ns": 123456789}'  # 60 bytes
}
```

## Benchmark

Sample bag: `ba234b52c88d7f1f0da04baab375f574.bag`
- Duration: 45 seconds
- Topics: 4× CompressedImage (front/left/right cameras + local map)
- Frames: 2,445 total (653-672 per topic @ 14-15 Hz)

| Format | Size | Compression Ratio |
|--------|------|-------------------|
| Original (CompressedImage) | 64.0 MB | 1.0× |
| MediaRef bag | 3.0 MB | 21.3× |
| MediaRef videos | 8.4 MB | - |
| **Total** | **11.4 MB** | **5.6×** |

Storage reduction: 82% (52.6 MB saved)

## Installation

```bash
uv pip install -r requirements.txt
```

## Usage

### Convert bags to MediaRef format

```bash
# ROS1
uv run bag_to_mediaref.py input.bag

# ROS2
uv run bag_to_mediaref.py input/

# Output structure:
# input_mediaref.bag         # Bag with String messages containing MediaRef JSON
# input_mediaref.media/      # H.264 encoded videos (one per CompressedImage topic)
#   ├── camera_front.mp4
#   └── camera_back.mp4
```

### Read MediaRef bags

```bash
# Display MediaRef messages
uv run mediaref_decode.py input_mediaref.bag -n 30

# Decode frames and save as images
uv run mediaref_decode.py input_mediaref.bag --decode -n 100 -o frames/
```

### Inspect bag contents

```bash
uv run bag_info.py input.bag -n 5
```

## Implementation

### Conversion (`bag_to_mediaref.py`)

1. Read CompressedImage messages from input bag
2. Decode JPEG/PNG to raw frames
3. Encode frames to H.264 video (one MP4 per topic)
4. Write String messages with MediaRef JSON to output bag
5. Preserve exact timestamps in `pts_ns` field

```python
# Input: sensor_msgs/CompressedImage
# Output: std_msgs/String with JSON payload
{
  "uri": "bag_mediaref.media/camera_front.mp4",
  "pts_ns": 1729561964520000000  # Nanosecond timestamp
}
```

### Decoding (`mediaref_decode.py`)

```python
from mediaref import MediaRef, batch_decode

# Read String messages from bag
msg = deserialize(rawdata, "std_msgs/msg/String")
ref = MediaRef.model_validate_json(msg.data)

# Batch decode frames from video
refs = [ref1, ref2, ref3, ...]
frames = batch_decode(refs, decoder="pyav")  # Returns numpy arrays [H, W, 3] uint8
```

### File structure

```
ROS1:                               ROS2:
input.bag                           input/
  ↓                                   ↓
input_mediaref.bag                  input_mediaref/
input_mediaref.media/               input_mediaref.media/
  ├── topic1.mp4                      ├── topic1.mp4
  └── topic2.mp4                      └── topic2.mp4
```

## Command Reference

### `bag_to_mediaref.py`

Convert CompressedImage topics to MediaRef format.

```bash
uv run bag_to_mediaref.py input.bag [-o OUTPUT] [--fps FPS] [--keyframe-interval SEC]
```

Options:
- `-o, --output PATH` - Output bag path (default: `{input}_mediaref.bag`)
- `--fps FLOAT` - Video frame rate (default: 30.0)
- `--keyframe-interval FLOAT` - Keyframe interval in seconds (default: 1.0)

### `mediaref_decode.py`

Read and decode MediaRef bags.

```bash
uv run mediaref_decode.py input.bag [-n COUNT] [--decode] [-o DIR]
```

Options:
- `-n, --max-messages INT` - Max messages to process (default: 30)
- `--decode` - Decode frames and save as images
- `-o, --output PATH` - Output directory for decoded frames (default: `decoded_frames`)

### `bag_info.py`

Display bag contents (auto-detects ROS1/ROS2).

```bash
uv run bag_info.py input.bag [-n COUNT]
```

Options:
- `-n, --max-messages INT` - Max sample messages per topic (default: 1)

## Performance

### Storage

| Metric | CompressedImage | MediaRef | Notes |
|--------|----------------|----------|-------|
| Per-frame overhead | ~50KB (JPEG) | ~60 bytes (JSON) | 833× reduction in message size |
| Compression | Intra-frame only | Inter-frame (H.264/H.265) | P-frames, B-frames |
| Typical reduction | - | 70-90% | Depends on scene complexity |
| Benchmark (45s, 4 cameras) | 64 MB | 11.4 MB | 82% reduction |

### I/O Performance

| Operation | CompressedImage | MediaRef |
|-----------|----------------|----------|
| Write | Direct JPEG copy | Video encoding (slower) |
| Sequential read | JPEG decompress | Batch decode from video |
| Random access | O(1) per frame | O(1) to keyframe, O(n) within GOP |
| Seek granularity | Per-frame | Keyframe interval (configurable) |

### Decoding Throughput

- Sequential: 100-500 FPS (hardware accelerated H.264 decode)
- Random access: 10-50 FPS (depends on keyframe interval)
- Batch decode: Optimal for processing multiple consecutive frames

## Technical Details

### Video Encoding Parameters

Default settings:
- Codec: H.264 (libx264)
- Frame rate: 30 FPS
- Keyframe interval: 1.0 second (30 frames)
- Pixel format: YUV420P
- Container: MP4

Keyframe interval affects:
- Seek performance: Shorter interval = faster random access
- File size: Longer interval = better compression
- Recommended: 0.5-2.0 seconds depending on use case

### Message Format

MediaRef uses `std_msgs/String` messages with JSON payload:

```json
{
  "uri": "relative/path/to/video.mp4",
  "pts_ns": 1729561964520000000
}
```

Fields:
- `uri`: Relative path from bag directory to video file
- `pts_ns`: Presentation timestamp in nanoseconds (matches ROS timestamp)

### Batch Decoding

For efficient processing of multiple frames:

```python
from mediaref import MediaRef, batch_decode

# Collect MediaRef objects
refs = [(MediaRef.model_validate_json(msg.data), topic) for msg, topic in messages]

# Resolve relative paths
bag_dir = str(bag_path.parent)
resolved = [ref.resolve_relative_path(bag_dir) for ref, _ in refs]

# Batch decode (efficient video seeking)
frames = batch_decode(resolved, decoder="pyav")  # numpy arrays [H, W, 3] uint8
```

## Limitations

- **Lossy compression**: Both JPEG and H.264/H.265 are lossy. Quality depends on bitrate.
- **Encoding overhead**: Initial conversion is slower than direct bag copy.
- **Random access**: Seeking to arbitrary frames requires decoding from last keyframe.
- **Message type support**: Currently only `sensor_msgs/CompressedImage` is supported.

## Use Cases

**Recommended:**
- Long-duration recordings (hours)
- Multi-camera systems (4+ topics)
- Archival storage
- Network transfer
- Offline processing

**Not recommended:**
- Real-time processing with strict latency requirements
- Applications requiring lossless image data
- Frequent random access to arbitrary frames
