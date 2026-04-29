import atexit
import gc
import os
import sys
import threading
import time
from dataclasses import dataclass
from typing import Callable, ContextManager, Dict, Generic, Optional, TypeVar

from loguru import logger

T = TypeVar("T")


@dataclass
class _Entry(Generic[T]):
    """Internal cache entry record. Not part of the public API."""

    obj: T
    cleanup_callback: Callable
    last_used: float
    refs: int


class ResourceCache(Generic[T]):
    """Thread-safe reference-counted resource cache with LRU eviction.

    Public surface:

    Mutations (atomic, thread-safe):
        - :meth:`try_add` — insert if absent (returns ``bool``).
        - :meth:`try_acquire` — increment refs if present (returns ``Optional[T]``).
        - :meth:`release` — decrement refs; raises ``KeyError`` if absent.
        - :meth:`evict` — forcefully remove regardless of refs (returns ``bool``).
        - :meth:`clear` — remove all entries.

    Inspection (read-only):
        - ``key in cache`` (``__contains__``).
        - ``len(cache)``.
        - :meth:`refs` — current ref count for a key (0 if absent).

    Configuration:
        - ``max_size`` — soft cap; LRU eviction happens after :meth:`release`
          drops a refs-0 entry below the cap. ``0`` (default) disables.

    Eviction (LRU sweep, :meth:`evict`, :meth:`clear`) invokes the registered
    ``cleanup_callback`` for each removed entry. Pass ``cleanup_callback=None``
    to :meth:`try_add` to fall back to ``obj.__exit__(None, None, None)``;
    the object must then implement the context-manager protocol.

    Strict variants (e.g. "raise on miss") are intentionally not provided —
    compose them at the call site with ``if not cache.try_add(...): raise ...``
    or ``if (x := cache.try_acquire(...)) is None: raise ...``. This keeps
    the primitive surface minimal.
    """

    def __init__(self, max_size: int = 0):
        self.max_size = max_size
        self._entries: Dict[str, _Entry[T]] = {}
        self._lock = threading.RLock()
        self._register_cleanup_handlers()

    def _register_cleanup_handlers(self):
        if sys.platform != "win32":
            os.register_at_fork(before=lambda: (self.clear(), gc.collect()))
        atexit.register(self.clear)

    @staticmethod
    def _default_cleanup(obj: T) -> Callable:
        if not isinstance(obj, ContextManager):
            raise ValueError(f"Object {obj} does not implement context manager protocol")
        return lambda: obj.__exit__(None, None, None)

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def try_add(self, key: str, obj: T, cleanup_callback: Optional[Callable] = None) -> bool:
        """Atomically insert a new entry with refs=1. Returns True if added, False if the key already exists."""
        with self._lock:
            if key in self._entries:
                return False
            if cleanup_callback is None:
                cleanup_callback = self._default_cleanup(obj)
            self._entries[key] = _Entry(obj=obj, cleanup_callback=cleanup_callback, refs=1, last_used=time.time())
            logger.debug(f"cache add: {key=}, total={len(self._entries)}")
            return True

    def try_acquire(self, key: str) -> Optional[T]:
        """Atomically increment refs and return the object, or ``None`` if the key is absent."""
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            entry.refs += 1
            entry.last_used = time.time()
            return entry.obj

    def release(self, key: str) -> None:
        """Decrement refs of an entry. Raises ``KeyError`` if the key is absent.

        After decrement, if the entry is unreferenced and the cache exceeds
        ``max_size``, an LRU sweep runs.
        """
        with self._lock:
            entry = self._entries[key]
            entry.refs -= 1
            logger.debug(f"cache release: {key=}, refs={entry.refs}")
            self._cleanup_if_needed()

    def evict(self, key: str) -> bool:
        """Forcefully remove an entry regardless of refs, invoking its cleanup callback.

        Returns True if the entry was present, False otherwise.
        """
        with self._lock:
            entry = self._entries.pop(key, None)
            if entry is None:
                return False
            entry.cleanup_callback()
            logger.debug(f"cache evict: {key=}, total={len(self._entries)}")
            return True

    def clear(self) -> None:
        """Remove all entries, invoking each cleanup callback."""
        with self._lock:
            for entry in self._entries.values():
                entry.cleanup_callback()
            logger.debug(f"cache clear: total={len(self._entries)}")
            self._entries.clear()

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def __contains__(self, key: object) -> bool:
        return key in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def refs(self, key: str) -> int:
        """Return the current ref count for ``key``, or ``0`` if absent."""
        with self._lock:
            entry = self._entries.get(key)
            return 0 if entry is None else entry.refs

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _cleanup_if_needed(self) -> None:
        """LRU sweep: evict the oldest unreferenced entries until at or below ``max_size``.

        Caller must hold ``self._lock``.
        """
        if self.max_size == 0 or len(self._entries) <= self.max_size:
            return
        unreferenced = [(k, e.last_used) for k, e in self._entries.items() if e.refs == 0]
        unreferenced.sort(key=lambda kv: kv[1])
        excess = len(self._entries) - self.max_size
        for key, _ in unreferenced[:excess]:
            entry = self._entries.pop(key)
            entry.cleanup_callback()
            logger.debug(f"cache LRU evict: {key=}, total={len(self._entries)}")
