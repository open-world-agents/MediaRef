import atexit
import gc
import os
import sys
import threading
import time
from dataclasses import dataclass
from typing import Callable, ContextManager, Dict, Generic, Optional, Tuple, TypeVar

from loguru import logger

T = TypeVar("T")


@dataclass
class CacheEntry(Generic[T]):
    """Container for cached resources with metadata."""

    obj: T
    cleanup_callback: Callable
    last_used: float = 0.0
    refs: int = 0


class ResourceCache(Generic[T], Dict[str, CacheEntry[T]]):
    """Thread-safe reference-counted resource cache with LRU eviction.

    All mutating operations and compound check-then-act primitives are serialized through
    an internal :class:`threading.RLock`. Read-only Dict accesses (``key in cache``,
    ``cache[key]``) inherit Python's GIL-level atomicity but are *not* lock-protected —
    call sites that need check-then-act semantics must use :meth:`try_acquire`,
    :meth:`try_add`, or :meth:`get_or_add` rather than composing ``__contains__`` and
    :meth:`acquire_entry` themselves.
    """

    def __init__(self, *args, max_size: int = 0, **kwargs):
        self.max_size = max_size
        self._lock = threading.RLock()
        super().__init__(*args, **kwargs)
        self._register_cleanup_handlers()

    def _register_cleanup_handlers(self):
        """Register cleanup callbacks for process lifecycle events."""
        if sys.platform != "win32":
            os.register_at_fork(before=lambda: (self.clear(), gc.collect()))
        atexit.register(self.clear)

    @staticmethod
    def _default_cleanup(obj: T) -> Callable:
        if not isinstance(obj, ContextManager):
            raise ValueError(f"Object {obj} does not implement context manager protocol")
        return lambda: obj.__exit__(None, None, None)

    # ------------------------------------------------------------------
    # Internal lock-held helpers
    # ------------------------------------------------------------------

    def _insert_locked(self, key: str, obj: T, cleanup_callback: Optional[Callable]) -> None:
        """Insert a new entry with refs=1. Caller must hold ``self._lock``."""
        if cleanup_callback is None:
            cleanup_callback = self._default_cleanup(obj)
        self[key] = CacheEntry(obj=obj, cleanup_callback=cleanup_callback, refs=1, last_used=time.time())
        logger.debug(f"cache add: {key=}, total={len(self)}")

    def _acquire_locked(self, key: str) -> T:
        """Increment refs of an existing entry and return its object. Caller must hold ``self._lock``."""
        entry = self[key]
        entry.refs += 1
        entry.last_used = time.time()
        return entry.obj

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_entry(self, key: str, obj: T, cleanup_callback: Optional[Callable] = None):
        """Add new entry with refs=1. Raises ``ValueError`` if key already exists.

        Thread-safe. Prefer :meth:`try_add` or :meth:`get_or_add` from concurrent
        code paths to avoid the check-then-add race.
        """
        with self._lock:
            if key in self:
                raise ValueError(f"Entry {key} already exists. Use acquire_entry() to increment refs.")
            self._insert_locked(key, obj, cleanup_callback)

    def try_add(self, key: str, obj: T, cleanup_callback: Optional[Callable] = None) -> bool:
        """Atomically add entry only if key is absent. Returns True if added. Thread-safe."""
        with self._lock:
            if key in self:
                return False
            self._insert_locked(key, obj, cleanup_callback)
            return True

    def acquire_entry(self, key: str) -> T:
        """Increment refs and return cached object. Raises ``KeyError`` if not found. Thread-safe."""
        with self._lock:
            if key not in self:
                raise KeyError(f"Entry {key} not found in cache")
            return self._acquire_locked(key)

    def try_acquire(self, key: str) -> Optional[T]:
        """Atomic check-and-acquire. Returns the object with refs incremented, or ``None`` if absent."""
        with self._lock:
            if key not in self:
                return None
            return self._acquire_locked(key)

    def get_or_add(self, key: str, factory: Callable[[], Tuple[T, Optional[Callable]]]) -> T:
        """Atomically acquire if present, else create via ``factory()`` and add. Thread-safe.

        ``factory()`` must return ``(obj, cleanup_callback)``; the callback may be ``None``
        to fall back to the context-manager default. The factory is invoked while the cache
        lock is held — keep it cheap. For heavy construction that should not serialize
        across cache keys, use :meth:`try_acquire` followed by an unlocked construction
        and :meth:`try_add` to commit.
        """
        with self._lock:
            if key in self:
                return self._acquire_locked(key)
            obj, cleanup_callback = factory()
            self._insert_locked(key, obj, cleanup_callback)
            return obj

    def release_entry(self, key: str):
        """Decrement refs and trigger LRU cleanup if needed. Thread-safe."""
        with self._lock:
            entry = self[key]
            entry.refs -= 1
            logger.debug(f"cache release: {key=}, refs={entry.refs}")
            self._cleanup_if_needed()

    def pop(self, key: str, default: Optional[CacheEntry[T]] = None) -> Optional[CacheEntry[T]]:  # type: ignore[override]
        """Remove entry and execute cleanup callback. Thread-safe."""
        with self._lock:
            if key in self:
                self[key].cleanup_callback()
            logger.debug(f"cache pop: {key=}, total={len(self) - (1 if key in self else 0)}")
            return super().pop(key, default)

    def clear(self):
        """Clear all entries and execute cleanup callbacks. Thread-safe."""
        with self._lock:
            for entry in self.values():
                entry.cleanup_callback()
            logger.debug(f"cache clear: total={len(self)}")
            super().clear()

    def _cleanup_if_needed(self):
        """Evict unreferenced entries using LRU policy when cache exceeds ``max_size``.

        The RLock allows nested acquisition from :meth:`release_entry`.
        """
        with self._lock:
            if self.max_size == 0 or len(self) <= self.max_size:
                return
            unreferenced = [k for k, v in self.items() if v.refs == 0]
            oldest_first = sorted(unreferenced, key=lambda k: self[k].last_used)
            excess_count = len(self) - self.max_size
            for key in oldest_first[:excess_count]:
                self.pop(key)
