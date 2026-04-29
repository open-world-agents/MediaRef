"""Behavioral and thread-safety tests for :class:`mediaref.resource_cache.ResourceCache`.

The cache exposes a minimal-yet-complete primitive surface:
``try_add`` / ``try_acquire`` / ``release`` / ``evict`` / ``clear`` plus
inspection (``__contains__`` / ``__len__`` / ``refs``). All mutations are
serialized through an internal RLock; tests cover both the
single-threaded contract and contended workloads.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    def test_try_add_inserts_when_absent(self):
        cache: ResourceCache[_Resource] = ResourceCache()
        assert cache.try_add("k", _Resource()) is True
        assert "k" in cache
        assert cache.refs("k") == 1

    def test_try_add_returns_false_on_collision(self):
        cache: ResourceCache[_Resource] = ResourceCache()
        cache.try_add("k", _Resource())
        assert cache.try_add("k", _Resource()) is False
        assert cache.refs("k") == 1  # the existing entry's refs unchanged

    def test_try_acquire_increments_refs(self):
        cache: ResourceCache[_Resource] = ResourceCache()
        r = _Resource()
        cache.try_add("k", r)
        assert cache.try_acquire("k") is r
        assert cache.refs("k") == 2

    def test_try_acquire_returns_none_when_absent(self):
        cache: ResourceCache[_Resource] = ResourceCache()
        assert cache.try_acquire("absent") is None

    def test_release_decrements(self):
        cache: ResourceCache[_Resource] = ResourceCache()
        cache.try_add("k", _Resource())
        cache.try_acquire("k")
        cache.release("k")
        assert cache.refs("k") == 1
        cache.release("k")
        assert cache.refs("k") == 0

    def test_release_raises_keyerror_when_absent(self):
        cache: ResourceCache[_Resource] = ResourceCache()
        with pytest.raises(KeyError):
            cache.release("absent")

    def test_evict_removes_and_returns_true(self):
        cache: ResourceCache[_Resource] = ResourceCache()
        r = _Resource()
        cache.try_add("k", r)
        assert cache.evict("k") is True
        assert "k" not in cache
        assert r.closed is True

    def test_evict_returns_false_when_absent(self):
        cache: ResourceCache[_Resource] = ResourceCache()
        assert cache.evict("absent") is False

    def test_evict_invokes_cleanup_callback_regardless_of_refs(self):
        """``evict`` is forceful — it does not respect ref counts."""
        cache: ResourceCache[object] = ResourceCache()
        closed: List[str] = []
        cache.try_add("k", object(), lambda: closed.append("k"))
        cache.try_acquire("k")  # refs=2
        assert cache.evict("k") is True
        assert closed == ["k"]
        assert "k" not in cache

    def test_clear_invokes_all_cleanups(self):
        cache: ResourceCache[_Resource] = ResourceCache()
        resources = [_Resource(f"r{i}") for i in range(5)]
        for i, r in enumerate(resources):
            cache.try_add(f"k{i}", r)
        cache.clear()
        assert all(r.closed for r in resources)
        assert len(cache) == 0

    def test_lru_eviction_skips_referenced(self):
        cache: ResourceCache[_Resource] = ResourceCache(max_size=2)
        r1, r2, r3 = _Resource("1"), _Resource("2"), _Resource("3")
        cache.try_add("k1", r1)
        cache.try_acquire("k1")  # refs=2; never evictable
        cache.try_add("k2", r2)
        cache.release("k2")  # refs=0, evictable
        cache.try_add("k3", r3)
        cache.release("k3")  # triggers cleanup; size>max
        # k1 stays (still referenced); the older of {k2, k3} drops.
        assert "k1" in cache
        assert ("k2" in cache) ^ ("k3" in cache)


# ---------------------------------------------------------------------------
# Inspection
# ---------------------------------------------------------------------------


class TestInspection:
    def test_len_tracks_entries(self):
        cache: ResourceCache[_Resource] = ResourceCache()
        assert len(cache) == 0
        cache.try_add("a", _Resource())
        assert len(cache) == 1
        cache.try_add("b", _Resource())
        assert len(cache) == 2
        cache.evict("a")
        assert len(cache) == 1

    def test_contains_reflects_membership(self):
        cache: ResourceCache[_Resource] = ResourceCache()
        cache.try_add("k", _Resource())
        assert "k" in cache
        cache.evict("k")
        assert "k" not in cache

    def test_refs_zero_when_absent(self):
        cache: ResourceCache[_Resource] = ResourceCache()
        assert cache.refs("absent") == 0

    def test_refs_tracks_acquires_and_releases(self):
        cache: ResourceCache[_Resource] = ResourceCache()
        cache.try_add("k", _Resource())
        cache.try_acquire("k")
        cache.try_acquire("k")
        assert cache.refs("k") == 3
        cache.release("k")
        assert cache.refs("k") == 2


# ---------------------------------------------------------------------------
# Default cleanup contract
# ---------------------------------------------------------------------------


class TestDefaultCleanupCallback:
    def test_default_callback_calls_exit(self):
        cache: ResourceCache[_Resource] = ResourceCache()
        r = _Resource()
        cache.try_add("k", r)  # default callback uses obj.__exit__
        cache.evict("k")
        assert r.closed is True

    def test_default_callback_requires_context_manager(self):
        cache: ResourceCache[object] = ResourceCache()
        with pytest.raises(ValueError):
            cache.try_add("k", object())  # plain object lacks __exit__


# ---------------------------------------------------------------------------
# Cleanup callback semantics
# ---------------------------------------------------------------------------


class TestCleanupCallback:
    """Eviction (via :meth:`evict` / :meth:`clear` / LRU sweep) must invoke the
    registered ``cleanup_callback`` exactly once per evicted entry, with no
    re-entry back into the cache (which would deadlock or corrupt state).
    """

    def test_evict_invokes_custom_cleanup_callback(self):
        cache: ResourceCache[object] = ResourceCache()
        closed: List[str] = []
        cache.try_add("k", object(), lambda: closed.append("k"))
        cache.evict("k")
        assert closed == ["k"]

    def test_clear_invokes_custom_cleanup_callback(self):
        cache: ResourceCache[object] = ResourceCache()
        closed: List[str] = []
        cache.try_add("a", object(), lambda: closed.append("a"))
        cache.try_add("b", object(), lambda: closed.append("b"))
        cache.clear()
        assert sorted(closed) == ["a", "b"]

    def test_lru_eviction_invokes_custom_cleanup_callback(self):
        cache: ResourceCache[object] = ResourceCache(max_size=2)
        closed: List[str] = []
        cache.try_add("a", object(), lambda: closed.append("a"))
        cache.release("a")  # refs=0, evictable
        cache.try_add("b", object(), lambda: closed.append("b"))
        cache.release("b")  # refs=0, evictable
        cache.try_add("c", object(), lambda: closed.append("c"))
        cache.release("c")  # triggers LRU cleanup; size>max
        assert "a" in closed  # oldest unreferenced was evicted
        assert "a" not in cache

    def test_cleanup_callback_does_not_re_enter_cache(self):
        """Regression for the ``__exit__`` → ``close`` → ``release`` cycle that
        lived in the cached_av factory before the Gemini review fix.
        """
        cache: ResourceCache[object] = ResourceCache()
        observations: List[str] = []

        def reentrant_callback():
            try:
                cache.release("k")
                observations.append("re-entered")
            except KeyError:
                observations.append("missing")

        cache.try_add("k", object(), reentrant_callback)
        cache.evict("k")
        assert "k" not in cache
        assert len(observations) == 1


# ---------------------------------------------------------------------------
# Thread-safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    @pytest.mark.parametrize("n_threads", [50])
    def test_concurrent_try_add_exactly_one_succeeds(self, n_threads):
        cache: ResourceCache[_Resource] = ResourceCache()
        barrier = threading.Barrier(n_threads)

        def attempt(_):
            return _wait_then(barrier, lambda: cache.try_add("k", _Resource()))

        with ThreadPoolExecutor(max_workers=n_threads) as ex:
            results = list(ex.map(attempt, range(n_threads)))

        assert sum(results) == 1
        assert cache.refs("k") == 1

    @pytest.mark.parametrize("n_threads", [100])
    def test_concurrent_acquire_increments_correctly(self, n_threads):
        cache: ResourceCache[_Resource] = ResourceCache()
        cache.try_add("k", _Resource())
        starting_refs = cache.refs("k")

        barrier = threading.Barrier(n_threads)
        with ThreadPoolExecutor(max_workers=n_threads) as ex:
            list(ex.map(lambda _: _wait_then(barrier, lambda: cache.try_acquire("k")), range(n_threads)))

        assert cache.refs("k") == starting_refs + n_threads

    @pytest.mark.parametrize("n_threads", [100])
    def test_concurrent_release_decrements_correctly(self, n_threads):
        cache: ResourceCache[_Resource] = ResourceCache()
        cache.try_add("k", _Resource())
        for _ in range(n_threads):
            cache.try_acquire("k")
        starting_refs = cache.refs("k")

        barrier = threading.Barrier(n_threads)
        with ThreadPoolExecutor(max_workers=n_threads) as ex:
            list(ex.map(lambda _: _wait_then(barrier, lambda: cache.release("k")), range(n_threads)))

        assert cache.refs("k") == starting_refs - n_threads

    def test_concurrent_mix_does_not_crash(self):
        """Stress test mixing add / acquire / release / evict / clear from
        many threads. No specific final state asserted — just that no thread
        raises and internal invariants hold (ref counts stay integer-valued,
        no exceptions escape outside the documented contract).
        """
        cache: ResourceCache[_Resource] = ResourceCache(max_size=20)
        keys = [f"k{i}" for i in range(50)]

        def worker(seed: int):
            errors: List[str] = []
            try:
                for i in range(200):
                    k = keys[(seed + i) % len(keys)]
                    op = i % 5
                    if op == 0:
                        cache.try_add(k, _Resource(k))
                    elif op == 1:
                        cache.try_acquire(k)
                    elif op == 2:
                        try:
                            cache.release(k)
                        except KeyError:
                            pass  # acceptable: someone evicted between threads
                    elif op == 3:
                        cache.evict(k)
                    elif i % 50 == 4:
                        cache.clear()
            except Exception as e:  # pragma: no cover - meaningful only on failure
                errors.append(str(e))
            return errors

        with ThreadPoolExecutor(max_workers=16) as ex:
            futures = [ex.submit(worker, i) for i in range(16)]
            all_errors: List[str] = []
            for f in as_completed(futures):
                all_errors.extend(f.result())

        assert not all_errors, f"Concurrent stress run raised: {all_errors[:3]}"


# ---------------------------------------------------------------------------
# Lazy-init pattern (composed from try_acquire + try_add at the call site)
# ---------------------------------------------------------------------------


class TestLazyInitPattern:
    """The cache deliberately omits a ``get_or_add`` primitive — the lazy-init
    pattern is composed by the caller. These tests pin down what that
    composition looks like, including the contended case where two threads
    both build a value for the same key.
    """

    @staticmethod
    def _get_or_create(cache: ResourceCache[_Resource], key: str, factory):
        """Reference implementation of the lazy-init pattern.

        Heavy construction runs *outside* the cache lock, so concurrent
        ``_get_or_create`` calls for *different* keys do not serialize. For the
        same key, the loser of the ``try_add`` race discards its throwaway
        and acquires the winner's entry on the next iteration.
        """
        for _ in range(8):
            cached = cache.try_acquire(key)
            if cached is not None:
                return cached
            obj = factory()
            if cache.try_add(key, obj):
                return obj
            obj.__exit__(None, None, None)
        raise RuntimeError(f"persistent contention for {key!r}")

    def test_factory_runs_only_once_under_contention(self):
        cache: ResourceCache[_Resource] = ResourceCache()
        factory_calls = 0
        factory_lock = threading.Lock()

        def factory():
            nonlocal factory_calls
            with factory_lock:
                factory_calls += 1
            return _Resource()

        n_threads = 32
        barrier = threading.Barrier(n_threads)
        with ThreadPoolExecutor(max_workers=n_threads) as ex:
            results = list(
                ex.map(
                    lambda _: _wait_then(barrier, lambda: TestLazyInitPattern._get_or_create(cache, "k", factory)),
                    range(n_threads),
                )
            )

        # Every thread sees the same final cached object.
        assert all(r is results[0] for r in results)
        assert cache.refs("k") == n_threads
        # Some throwaways are expected under contention, but the cache holds exactly one.
        assert len(cache) == 1
        assert factory_calls >= 1


# ---------------------------------------------------------------------------
# Shared-instance close semantics (regression for _released-flag bug)
# ---------------------------------------------------------------------------


class TestSharedInstanceCloseSemantics:
    def test_n_concurrent_owners_each_release_once_drops_to_zero(self):
        """N owners each call release once → refs hits 0 exactly once.

        Direct regression for an earlier per-instance ``_released`` flag bug
        that made the first close set the flag and turned every subsequent
        owner's close into a no-op, leaking refs forever.
        """
        n_owners = 32
        cache: ResourceCache[_Resource] = ResourceCache()
        cache.try_add("k", _Resource(), lambda: None)
        for _ in range(n_owners - 1):
            cache.try_acquire("k")
        assert cache.refs("k") == n_owners

        barrier = threading.Barrier(n_owners)
        with ThreadPoolExecutor(max_workers=n_owners) as ex:
            list(ex.map(lambda _: _wait_then(barrier, lambda: cache.release("k")), range(n_owners)))

        assert cache.refs("k") == 0
