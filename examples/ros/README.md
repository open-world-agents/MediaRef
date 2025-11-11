# MediaRef for ROS

MediaRef replaces embedded CompressedImage data in ROS bags with references to external video files, reducing storage requirements by 70-90% through inter-frame compression.

## Overview

Traditional ROS bags store each image frame as an independent JPEG in a CompressedImage message. MediaRef stores a JSON reference pointing to a frame in an H.264 encoded video file.

```python
# CompressedImage: ~50KB per message
{"format": "jpeg", "data": b"\xff\xd8\xff\xe0..."}

# MediaRef: ~60 bytes per message
{"data": '{"uri": "bag.media/camera.mp4", "pts_ns": 123456789}'}
```

## Benchmark

Sample bag: `ba234b52c88d7f1f0da04baab375f574.bag` (bagfile is not included, try your own!)
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

```bash
# Convert bag to MediaRef format
uv run bag_to_mediaref.py input.bag

# Display MediaRef messages
uv run mediaref_decode.py input_mediaref.bag -n 30

# Decode frames and save as images
uv run mediaref_decode.py input_mediaref.bag --decode -n 100 -o frames/

# Inspect bag contents
uv run bag_info.py input.bag -n 5
```

## Implementation

Conversion process:
1. Read CompressedImage messages from bag
2. Decode JPEG to raw frames
3. Encode frames to H.264 video (one MP4 per topic)
4. Write String messages with MediaRef JSON to output bag

```python
# MediaRef message format (std_msgs/String)
{"uri": "bag_mediaref.media/camera.mp4", "pts_ns": 1729561964520000000}

# Decoding
from mediaref import MediaRef, batch_decode
refs = [MediaRef.model_validate_json(msg.data) for msg in messages]
frames = batch_decode(refs, decoder="pyav")  # numpy arrays [H, W, 3] uint8
```

## Options

```bash
# bag_to_mediaref.py
-o, --output PATH              Output bag path (default: {input}_mediaref.bag)
--fps FLOAT                    Video frame rate (default: 30.0)
--keyframe-interval FLOAT      Keyframe interval in seconds (default: 1.0)

# mediaref_decode.py
-n, --max-messages INT         Max messages to process (default: 30)
--decode                       Decode frames and save as images
-o, --output PATH              Output directory (default: decoded_frames)

# bag_info.py
-n, --max-messages INT         Max sample messages per topic (default: 1)
```

## Performance

| Metric | CompressedImage | MediaRef |
|--------|----------------|----------|
| Message size | ~50KB (JPEG) | ~60 bytes (JSON) |
| Compression | Intra-frame | Inter-frame (H.264) |
| Storage (benchmark) | 64 MB | 11.4 MB (82% reduction) |
| Write speed | Fast | Slower (video encoding) |
| Sequential read | JPEG decompress | Batch decode |
| Random access | O(1) per frame | O(1) to keyframe, O(n) within GOP |

## Technical Details

Video encoding:
- Codec: H.264 (libx264)
- Frame rate: 30 FPS (configurable)
- Keyframe interval: 1.0 second (configurable)
- Pixel format: YUV420P
- Container: MP4

Message format (std_msgs/String):
```json
{"uri": "relative/path/to/video.mp4", "pts_ns": 1729561964520000000}
```

## Limitations

- Lossy compression (both JPEG and H.264 are lossy)
- Slower conversion (video encoding overhead)
- Random access requires decoding from last keyframe
- Only supports sensor_msgs/CompressedImage
