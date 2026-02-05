# Playback Semantics

MediaRef's video decoders (`PyAVVideoDecoder` and `TorchCodecVideoDecoder`) follow unified **playback semantics** for frame retrieval. This ensures consistent behavior regardless of which decoder backend you use.

## The Display Model

When you query a frame at a specific timestamp, MediaRef returns the frame that would be **displayed on screen** at that moment—not the closest frame, not the next frame, but the frame currently being shown.

```
frame[i].pts <= query_time < frame[i+1].pts  →  Returns frame[i]
```

This model is based on how video players work: a frame is displayed starting at its presentation timestamp (PTS) until the next frame's PTS.

### Example

For a 25 FPS video where frames have PTS values at 0.00s, 0.04s, 0.08s, ...:

| Query Time | Returned Frame | Explanation |
|------------|----------------|-------------|
| `0.00s` | Frame 0 | Exactly at frame 0's PTS |
| `0.02s` | Frame 0 | Between frame 0 and frame 1 |
| `0.04s` | Frame 1 | Exactly at frame 1's PTS |
| `0.05s` | Frame 1 | Between frame 1 and frame 2 |

## Boundary Conditions

| Condition | Behavior |
|-----------|----------|
| `timestamp < begin_stream_seconds` | `ValueError` |
| `timestamp >= end_stream_seconds` | `ValueError` |
| `timestamp == frame[i].pts` | Returns `frame[i]` |
| `frame[i].pts < timestamp < frame[i+1].pts` | Returns `frame[i]` |

### Important Notes

- **`begin_stream_seconds`**: The first frame's PTS. This is often `0.0`, but not always—some videos start at non-zero timestamps.
- **`end_stream_seconds`**: Calculated as `begin_stream_seconds + duration_seconds`. Queries at or beyond this point raise `ValueError`.

## Last Frame Handling

The last frame's valid query range is `[last_frame.pts, end_stream_seconds)`.

For example, if the last frame has PTS = 9.96s and `end_stream_seconds` = 10.0s:
- Query `9.96s` → Returns last frame ✓
- Query `9.99s` → Returns last frame ✓
- Query `10.0s` → `ValueError` ✗

## Design Rationale

This semantics is adopted from [TorchCodec](https://github.com/pytorch/torchcodec), which explains the reasoning:

> We look at nextPts for a frame, and not its pts or duration. Our abstract player displays frames starting at the pts for that frame until the pts for the next frame. There are two consequences:
>
> 1. We ignore the duration for a frame. A frame is played until the next frame replaces it. This model is robust to durations being 0 or incorrect; our source of truth is the pts for frames.
>
> 2. In order to establish if the start of an interval maps to a particular frame, we need to figure out if it is ordered after the frame's pts, but before the next frame's pts.

By adopting this model, MediaRef ensures that:
1. Both `PyAVVideoDecoder` and `TorchCodecVideoDecoder` return identical frames for the same query
2. Frame selection is deterministic and matches real video player behavior
3. Edge cases (sparse keyframes, variable frame rates) are handled consistently