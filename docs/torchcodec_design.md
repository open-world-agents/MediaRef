Here is the design of TorchCodec's playback semantics and boundary conditions.

## Playback Semantics of TorchCodec

```
    // Note that we look at nextPts for a frame, and not its pts or duration.
    // Our abstract player displays frames starting at the pts for that frame
    // until the pts for the next frame. There are two consequences:
    //
    //   1. We ignore the duration for a frame. A frame is played until the
    //   next frame replaces it. This model is robust to durations being 0 or
    //   incorrect; our source of truth is the pts for frames. If duration is
    //   accurate, the nextPts for a frame would be equivalent to pts +
    //   duration.
    //   2. In order to establish if the start of an interval maps to a
    //   particular frame, we need to figure out if it is ordered after the
    //   frame's pts, but before the next frames's pts.
```

## Boundary Condition Handling

| Condition | Behavior |
|-----------|----------|
| `timestamp < begin_stream_seconds` | ValueError |
| `timestamp >= end_stream_seconds` | ValueError |
| `timestamp == frame[i].pts` | Returns frame[i] |
| `frame[i].pts < timestamp < frame[i+1].pts` | Returns frame[i] |

---

## Last Frame Handling

TorchCodec sets the `nextPts` of the last frame to `INT64_MAX`.
This means the last frame's valid range is `[pts, end_stream_seconds)`.

Note: In `get_frames_played_at`, if `max(seconds) >= end_stream_seconds`,
a ValueError is raised.