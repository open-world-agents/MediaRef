"""Interop helpers for converting between MediaRef and other ecosystems.

These submodules provide pure-Python adapters and have no required runtime
dependencies on the foreign libraries — you can import the helpers and
construct/consume MediaRef objects without installing them.

Available adapters:
    - :mod:`mediaref.compat.lerobot` — convert to/from lerobot's
      ``VideoFrame`` (path, timestamp seconds) representation.
"""

__all__: list[str] = []
