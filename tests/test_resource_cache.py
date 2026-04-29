"""Thread-safety and behavioral tests for :class:`mediaref.resource_cache.ResourceCache`.

The cache is a hot-path primitive used under multi-threaded dataloader
workloads. These tests exercise both the single-threaded contract
(``add_entry`` / ``acquire_entry`` / ``release_entry`` / LRU eviction)
and the atomic concurrency primitives (``try_add`` / ``try_acquire`` /
``get_or_add``).
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from typing import List

import pytest

from mediaref.resource_cache import ResourceCache


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _Resource:
    """Minimal context-manager resource used as a cached value."""

    def __init__(self, name: str = "r") -> None:
        self.name = name
        self.closed = False

    def __enter__(self) -> "_Resource":
        return self

    def __exit__(self, *_exc) -> None:
        self.closed = True


def _wait_then(barrier: threading.Barrier, fn):
    """Run ``fn`` after all threads reach the barrier — maximises contention."""
    barrier.wait()
    return fn()


# ---------------------------------------------------------------------------
# Single-threaded contract
# ---------------------------------------------------------------------------


class TestBasicContract:
    def test_add_then_acquire_increments_refs(self):
        cache: ResourceCache[_Resource] = ResourceCache()
        r = _Resource()
        cache.add_entry("k", r)
        assert cache["k"].refs == 1
        out = cache.acquire_entry("k")
        assert out is r
        assert cache["k"].refs == 2

    def test_release_decrements_refs(self):
        cache: ResourceCache[_Resource] = ResourceCache()
        cache.add_entry("k", _Resource())
        cache.acquire_entry("k")
        cache.release_entry("k")
        assert cache["k"].refs == 1
        cache.release_entry("k")
        assert cache["k"].refs == 0

    def test_add_existing_key_raises(self):
        cache: ResourceCache[_Resource] = ResourceCache()
        cache.add_entry("k", _Resource())
        with pytest.raises(ValueError):
            cache.add_entry("k", _Resource())

    def test_acquire_missing_key_raises(self):
        cache: ResourceCache[_Resource] = ResourceCache()
        with pytest.raises(KeyError):
            cache.acquire_entry("absent")

    def test_pop_invokes_cleanup(self):
        cache: ResourceCache[_Resource] = ResourceCache()
        r = _Resource()
        cache.add_entry("k", r)
        cache.pop("k")
        assert r.closed is True
        assert "k" not in cache

    def test_clear_invokes_all_cleanups(self):
        cache: ResourceCache[_Resource] = ResourceCache()
        resources = [_Resource(f"r{i}") for i in range(5)]
        for i, r in enumerate(resources):
            cache.add_entry(f"k{i}", r)
        cache.clear()
        assert all(r.closed for r in resources)
        assert len(cache) == 0

    def test_lru_eviction_skips_referenced(self):
        cache: ResourceCache[_Resource] = ResourceCache(max_size=2)
        r1, r2, r3 = _Resource("1"), _Resource("2"), _Resource("3")
        cache.add_entry("k1", r1)
        cache.acquire_entry("k1")  # k1 has refs=2; never evictable while held
        cache.add_entry("k2", r2)
        cache.release_entry("k2")  # k2 refs=0, evictable
        cache.add_entry("k3", r3)
        cache.release_entry("k3")  # triggers cleanup; size>max
        # k1 stays (still referenced); the older of {k2, k3} drops.
        assert "k1" in cache
        assert ("k2" in cache) ^ ("k3" in cache)


# ---------------------------------------------------------------------------
# Atomic primitives
# ---------------------------------------------------------------------------


class TestAtomicPrimitives:
    def test_try_acquire_present(self):
        cache: ResourceCache[_Resource] = ResourceCache()
        r = _Resource()
        cache.add_entry("k", r)
        out = cache.try_acquire("k")
        assert out is r
        assert cache["k"].refs == 2

    def test_try_acquire_absent(self):
        cache: ResourceCache[_Resource] = ResourceCache()
        assert cache.try_acquire("absent") is None

    def test_try_add_inserts_when_absent(self):
        cache: ResourceCache[_Resource] = ResourceCache()
        added = cache.try_add("k", _Resource())
        assert added is True
        assert cache["k"].refs == 1

    def test_try_add_returns_false_when_present(self):
        cache: ResourceCache[_Resource] = ResourceCache()
        cache.add_entry("k", _Resource())
        added = cache.try_add("k", _Resource())
        assert added is False
        assert cache["k"].refs == 1  # the existing entry's refs unchanged

    def test_get_or_add_factory_runs_only_on_miss(self):
        cache: ResourceCache[_Resource] = ResourceCache()
        calls = {"n": 0}

        def factory():
            calls["n"] += 1
            return _Resource(), None

        out1 = cache.get_or_add("k", factory)
        assert calls["n"] == 1
        out2 = cache.get_or_add("k", factory)
        assert calls["n"] == 1  # not called on hit
        assert out1 is out2
        assert cache["k"].refs == 2


# ---------------------------------------------------------------------------
# Thread-safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    @pytest.mark.parametrize("n_threads", [50])
    def test_concurrent_get_or_add_returns_one_object(self, n_threads):
        """100 threads racing for the same key must see exactly one factory call
        and all receive the same cached object."""
        cache: ResourceCache[_Resource] = ResourceCache()
        factory_invocations = 0
        factory_lock = threading.Lock()

        def factory():
            nonlocal factory_invocations
            with factory_lock:
                factory_invocations += 1
            return _Resource(), None

        barrier = threading.Barrier(n_threads)
        with ThreadPoolExecutor(max_workers=n_threads) as ex:
            results = list(
                ex.map(lambda _: _wait_then(barrier, lambda: cache.get_or_add("k", factory)), range(n_threads))
            )

        assert factory_invocations == 1
        assert all(r is results[0] for r in results)
        assert cache["k"].refs == n_threads

    @pytest.mark.parametrize("n_threads", [100])
    def test_concurrent_acquire_increments_correctly(self, n_threads):
        cache: ResourceCache[_Resource] = ResourceCache()
        cache.add_entry("k", _Resource())
        starting_refs = cache["k"].refs

        barrier = threading.Barrier(n_threads)
        with ThreadPoolExecutor(max_workers=n_threads) as ex:
            list(ex.map(lambda _: _wait_then(barrier, lambda: cache.acquire_entry("k")), range(n_threads)))

        assert cache["k"].refs == starting_refs + n_threads

    @pytest.mark.parametrize("n_threads", [100])
    def test_concurrent_release_decrements_correctly(self, n_threads):
        cache: ResourceCache[_Resource] = ResourceCache()
        cache.add_entry("k", _Resource())
        for _ in range(n_threads):
            cache.acquire_entry("k")
        # refs is now 1 + n_threads
        starting_refs = cache["k"].refs

        barrier = threading.Barrier(n_threads)
        with ThreadPoolExecutor(max_workers=n_threads) as ex:
            list(ex.map(lambda _: _wait_then(barrier, lambda: cache.release_entry("k")), range(n_threads)))

        assert cache["k"].refs == starting_refs - n_threads

    @pytest.mark.parametrize("n_threads", [50])
    def test_concurrent_try_add_exactly_one_succeeds(self, n_threads):
        cache: ResourceCache[_Resource] = ResourceCache()
        barrier = threading.Barrier(n_threads)
        results: List[bool] = []

        def attempt(_):
            return _wait_then(barrier, lambda: cache.try_add("k", _Resource()))

        with ThreadPoolExecutor(max_workers=n_threads) as ex:
            results = list(ex.map(attempt, range(n_threads)))

        assert sum(results) == 1
        assert cache["k"].refs == 1

    def test_concurrent_add_release_clear_does_not_crash(self):
        """Stress test mixing add / acquire / release / clear from many threads.

        Doesn't assert specific final state — just that no thread raises and
        no internal invariant blows up (ref counts staying integer-valued,
        no KeyError leaks past the contract).
        """
        cache: ResourceCache[_Resource] = ResourceCache(max_size=20)
        keys = [f"k{i}" for i in range(50)]

        def worker(seed: int):
            errors: List[str] = []
            try:
                for i in range(200):
                    k = keys[(seed + i) % len(keys)]
                    if i % 5 == 0:
                        cache.try_add(k, _Resource(k))
                    elif i % 5 == 1:
                        cache.try_acquire(k)
                    elif i % 5 == 2:
                        try:
                            cache.release_entry(k)
                        except KeyError:
                            pass  # Acceptable: someone evicted between threads
                    elif i % 5 == 3:
                        cache.get_or_add(k, lambda kk=k: (_Resource(kk), None))
                    else:
                        if i % 50 == 4:
                            cache.clear()
            except Exception as e:  # pragma: no cover - meaningful to surface in failure
                errors.append(str(e))
            return errors

        with ThreadPoolExecutor(max_workers=16) as ex:
            futures = [ex.submit(worker, i) for i in range(16)]
            all_errors = []
            for f in as_completed(futures):
                all_errors.extend(f.result())

        assert not all_errors, f"Concurrent stress run raised: {all_errors[:3]}"


# ---------------------------------------------------------------------------
# Idempotent close pattern
# ---------------------------------------------------------------------------


@contextmanager
def _double_close_safe_resource(cache: ResourceCache[_Resource], key: str):
    r = _Resource(key)
    cache.add_entry(key, r)
    try:
        yield r
    finally:
        cache.release_entry(key)


class TestRefCountInvariants:
    def test_release_after_pop_raises_keyerror(self):
        """If an entry has been evicted, a stale ``release_entry`` should raise
        rather than silently corrupt internal state."""
        cache: ResourceCache[_Resource] = ResourceCache()
        cache.add_entry("k", _Resource())
        cache.release_entry("k")  # refs=0
        cache.pop("k")
        with pytest.raises(KeyError):
            cache.release_entry("k")


# ---------------------------------------------------------------------------
# Cleanup callback semantics
# ---------------------------------------------------------------------------


class TestCleanupCallback:
    """Eviction (via :meth:`pop` / :meth:`clear` / LRU sweep) must invoke the
    registered ``cleanup_callback`` exactly once per evicted entry, with no
    re-entry back into the cache (which would deadlock or corrupt state).
    """

    def test_pop_invokes_custom_cleanup_callback(self):
        cache: ResourceCache[object] = ResourceCache()
        closed: List[str] = []
        cache.add_entry("k", object(), lambda: closed.append("k"))
        cache.pop("k")
        assert closed == ["k"]

    def test_clear_invokes_custom_cleanup_callback(self):
        cache: ResourceCache[object] = ResourceCache()
        closed: List[str] = []
        cache.add_entry("a", object(), lambda: closed.append("a"))
        cache.add_entry("b", object(), lambda: closed.append("b"))
        cache.clear()
        assert sorted(closed) == ["a", "b"]

    def test_lru_eviction_invokes_custom_cleanup_callback(self):
        cache: ResourceCache[object] = ResourceCache(max_size=2)
        closed: List[str] = []
        cache.add_entry("a", object(), lambda: closed.append("a"))
        cache.release_entry("a")  # refs=0, evictable
        cache.add_entry("b", object(), lambda: closed.append("b"))
        cache.release_entry("b")  # refs=0, evictable
        cache.add_entry("c", object(), lambda: closed.append("c"))
        cache.release_entry("c")  # triggers cleanup_if_needed; size>max
        # Oldest unreferenced ("a") should have been evicted.
        assert "a" in closed
        assert "a" not in cache

    def test_cleanup_callback_does_not_re_enter_cache(self):
        """The factory-supplied callback must perform terminal cleanup —
        not call back into ``release_entry``. Regression test for the
        ``__exit__`` → ``close`` → ``release_entry`` cycle that lived in
        the cached_av factory before the Gemini review fix.
        """
        cache: ResourceCache[object] = ResourceCache()
        re_entries: List[str] = []

        def bad_callback():
            # Simulate a reentrant call — should never run during normal
            # eviction. We assert that cleanup runs *once* and doesn't try
            # to release a non-existent entry.
            try:
                cache.release_entry("k")
                re_entries.append("re-entered")
            except KeyError:
                re_entries.append("missing")

        cache.add_entry("k", object(), bad_callback)
        cache.pop("k")
        # Even with a misbehaving callback, the entry was successfully popped
        # and the callback was invoked exactly once.
        assert "k" not in cache
        assert len(re_entries) == 1


# ---------------------------------------------------------------------------
# Shared-instance close semantics
# ---------------------------------------------------------------------------


class TestSharedInstanceCloseSemantics:
    """When multiple callers share a single cached instance, each caller
    is responsible for one matching release. Per-instance "already closed"
    flags are unsafe — they break refs counting for shared resources.
    """

    def test_n_concurrent_owners_each_release_once_drops_to_zero(self):
        """N owners each call release once → refs hits 0 exactly once.

        Regression for the ``_released`` per-instance flag bug, where the
        first close set the flag and subsequent closes (from other threads)
        became no-ops, leaking refs forever.
        """
        n_owners = 32
        cache: ResourceCache[_Resource] = ResourceCache()

        # Simulate N owners acquiring the same shared instance.
        cache.add_entry("k", _Resource(), lambda: None)
        for _ in range(n_owners - 1):
            cache.acquire_entry("k")
        assert cache["k"].refs == n_owners

        # Each owner calls release concurrently.
        barrier = threading.Barrier(n_owners)
        with ThreadPoolExecutor(max_workers=n_owners) as ex:
            list(ex.map(lambda _: _wait_then(barrier, lambda: cache.release_entry("k")), range(n_owners)))

        assert cache["k"].refs == 0
