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
class _Entry(Generic[T]):
    obj: T
    cleanup_callback: Callable
    last_used: float
    refs: int


class ResourceCache(Generic[T]):
    """Thread-safe reference-counted resource cache with LRU eviction.

    Two atomic mutation primitives cover disjoint caller intents:
    :meth:`try_acquire` for callers without an obj, and
    :meth:`try_insert_or_acquire` for atomic lazy-init by callers that have
    built one. Plus :meth:`release`, :meth:`evict`, :meth:`clear`, and
    inspection (``__contains__``, ``__len__``, :meth:`refs`).

    ``max_size`` is a soft cap; LRU eviction runs after :meth:`release` drops
    a refs-0 entry past the cap. ``0`` (default) disables the cap.
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

    def try_acquire(self, key: str) -> Optional[T]:
        """Increment refs and return the object, or ``None`` if absent.

        Caller MUST call :meth:`release` for any non-``None`` return.
        """
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            entry.refs += 1
            entry.last_used = time.monotonic()
            return entry.obj

    def try_insert_or_acquire(
        self,
        key: str,
        obj: T,
        cleanup_callback: Optional[Callable] = None,
    ) -> Tuple[T, bool]:
        """If absent: insert ``obj`` (refs=1) and return ``(obj, True)``.
        If present: increment refs on the existing entry and return ``(existing, False)``.

        Caller MUST call :meth:`release` in either case. When ``was_added`` is
        ``False``, caller MUST also discard their own ``obj`` — the canonical
        is something else.
        """
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                entry.refs += 1
                entry.last_used = time.monotonic()
                return entry.obj, False
            if cleanup_callback is None:
                cleanup_callback = self._default_cleanup(obj)
            self._entries[key] = _Entry(obj=obj, cleanup_callback=cleanup_callback, refs=1, last_used=time.monotonic())
            logger.debug(f"cache add: {key=}, total={len(self._entries)}")
            return obj, True

    def release(self, key: str) -> None:
        """Decrement refs. Raises ``KeyError`` if absent. Triggers LRU sweep."""
        with self._lock:
            entry = self._entries[key]
            entry.refs -= 1
            logger.debug(f"cache release: {key=}, refs={entry.refs}")
            self._cleanup_if_needed()

    def evict(self, key: str) -> bool:
        """Forcefully remove an entry regardless of refs. Returns True if it was present."""
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
        if self.max_size == 0 or len(self._entries) <= self.max_size:
            return
        unreferenced = [(k, e.last_used) for k, e in self._entries.items() if e.refs == 0]
        unreferenced.sort(key=lambda kv: kv[1])
        excess = len(self._entries) - self.max_size
        for key, _ in unreferenced[:excess]:
            entry = self._entries.pop(key)
            entry.cleanup_callback()
            logger.debug(f"cache LRU evict: {key=}, total={len(self._entries)}")
