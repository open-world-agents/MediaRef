# How MediaRef compares to other media-reference systems

MediaRef occupies a niche between two existing patterns:

- **`datasets.Video`** — HuggingFace's Arrow feature type for video. One row = one whole video file, embedded as bytes. Decoded into a `torchcodec.VideoDecoder` at access.
- **lerobot `VideoFrame`** — the LeRobotDataset feature for robotic data. One row = one frame, but tied to LeRobotDataset's directory layout (single global fps, episode-based shards, path templates).

Both work well within their assumptions. Neither covers the case where per-frame addressing is needed, the data isn't shaped like a LeRobotDataset, and rows shouldn't carry the bytes of the entire video.

## At-a-glance

|                       | `datasets.Video`                                   | lerobot `VideoFrame`                                                | **MediaRef**                                                              |
| --------------------- | -------------------------------------------------- | ------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| Granularity           | one row = one **video file**                       | one row = one frame, **inside LeRobotDataset's layout**             | one row = **one frame, anywhere**                                         |
| Storage assumption    | Embedded `bytes` + `path`                          | Path templates + per-episode metadata; fixed global fps             | None — just `(uri, pts_ns)`                                               |
| Container             | Parquet                                            | LeRobotDataset directory layout                                     | Any (Parquet, mcap, rosbag, HDF5, JSON, …)                                |
| Standards basis       | Arrow / Parquet schema                             | Arrow / Parquet schema                                              | RFC 3986 / RFC 2397                                                       |
| Cross-call batching for many frames per video | One `VideoDecoder` per row; multi-frame access from that decoder | LRU-cached decoders + episode-shard prefetch inside lerobot's loaders | `batch_decode()` groups refs by URI and opens each container once         |
| Hub-native            | yes                                                | yes (lerobot-style datasets)                                        | yes via `mediaref.hf` (`register_feature("MediaRef", …)`)                 |

## When to use which

- **`datasets.Video`** if your row *is* a whole clip you'll decode end-to-end.
- **lerobot `VideoFrame`** if your data fits LeRobotDataset's layout (single global fps, episode-based directory).
- **MediaRef** if neither describes your data — per-frame references with no assumptions about container, fps, or directory layout.

See [`SPEC.md`](SPEC.md) for the wire format.
