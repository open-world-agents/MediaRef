"""Interop helpers for converting between MediaRef and other ecosystems.

These submodules provide adapters between MediaRef and foreign data
models. Most of them are pure-Python and have no required runtime
dependencies; the ones that wrap a third-party library import that
library lazily (and the corresponding optional extra documents the
install command).

Available adapters:
    - :mod:`mediaref.compat.lerobot` — convert to/from lerobot's
      ``VideoFrame`` (path, timestamp seconds) representation. Pure
      Python; no lerobot install needed.
    - :mod:`mediaref.compat.mcap` — read and write MediaRef messages
      from/to mcap streams (robotics, autonomous driving, OWA-style
      desktop recordings). Requires ``pip install 'mediaref[mcap]'``.
"""

__all__: list[str] = []
