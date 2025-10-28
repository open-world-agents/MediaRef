# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-10-28

### Added
- Initial release of mediaref
- `MediaRef` Pydantic model for media references
- Support for multiple URI types: file paths, URLs, data URIs, video timestamps
- Lazy loading with `to_rgb_array()` and `to_pil_image()` methods
- Batch loading with `load_batch()` function
- Video container caching for efficient frame extraction
- Path resolution for relative paths (MCAP/rosbag support)
- Data URI embedding with `embed_as_data_uri()`
- PyAVVideoDecoder with TorchCodec-compatible interface
- TorchCodecVideoDecoder (optional, requires torchcodec>=0.4.0)
- VideoReader and VideoWriter classes
- Resource cache with reference counting and LRU eviction
- Optional `[loader]` extra for loading dependencies
- Full Pydantic serialization support
- Type hints with py.typed marker
- Comprehensive test suite (26 tests)
- Demo script showing all features

### Features
- **Core package**: Lightweight MediaRef definition (only pydantic dependency)
- **Loader extra**: Full loading capabilities with numpy, opencv, pillow, av, requests
- **Video decoders**: PyAV and TorchCodec-based decoders for batch frame extraction
- **Caching**: Automatic video container caching with cleanup on exit
- **Validation**: URI validation and file existence checking
- **Serialization**: JSON serialization via Pydantic

[0.1.0]: https://github.com/open-world-agents/mediaref/releases/tag/v0.1.0

