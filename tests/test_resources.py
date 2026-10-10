from __future__ import annotations

import threading

from msst_api.resources import SharedModelCache


class _Closable:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_global_lru_evicts_across_namespaces():
    cache = SharedModelCache(max_loaded_models=2, max_concurrency=1)
    a, b, c = _Closable(), _Closable(), _Closable()
    cache.put(("a",), a, namespace="msst", label="a", closer=lambda o: o.close())
    cache.put(("b",), b, namespace="rvc", label="b", closer=lambda o: o.close())
    # Adding a third model evicts the globally least-recently-used one.
    cache.put(("c",), c, namespace="msst", label="c", closer=lambda o: o.close())

    assert a.closed is True
    assert b.closed is False and c.closed is False
    assert cache.loaded() == ["b", "c"]
    assert cache.loaded("msst") == ["c"]
    assert cache.loaded("rvc") == ["b"]


def test_get_touches_lru_order():
    cache = SharedModelCache(max_loaded_models=2, max_concurrency=1)
    a, b, c = _Closable(), _Closable(), _Closable()
    cache.put(("a",), a, namespace="msst", label="a", closer=lambda o: o.close())
    cache.put(("b",), b, namespace="msst", label="b", closer=lambda o: o.close())
    assert cache.get(("a",)) is a  # refresh "a"
    cache.put(("c",), c, namespace="msst", label="c", closer=lambda o: o.close())

    assert a.closed is False  # refreshed, kept
    assert b.closed is True  # least recently used
    assert cache.loaded() == ["a", "c"]


def test_clear_only_clears_namespace():
    cache = SharedModelCache(max_loaded_models=4, max_concurrency=1)
    a, b = _Closable(), _Closable()
    cache.put(("a",), a, namespace="msst", label="a", closer=lambda o: o.close())
    cache.put(("b",), b, namespace="rvc", label="b", closer=lambda o: o.close())

    cache.clear("msst")
    assert a.closed is True and b.closed is False
    assert cache.loaded() == ["b"]


def test_inference_slot_bounds_concurrency():
    cache = SharedModelCache(max_loaded_models=1, max_concurrency=1)
    with cache.inference_slot():
        # The single slot is taken; a non-blocking acquire must fail.
        assert cache._semaphore.acquire(blocking=False) is False
    # Released: acquiring again succeeds.
    assert cache._semaphore.acquire(blocking=False) is True
    cache._semaphore.release()


def test_inference_slot_is_reentrant_across_threads():
    cache = SharedModelCache(max_loaded_models=1, max_concurrency=1)
    order: list[str] = []
    started = threading.Event()

    def worker(name: str) -> None:
        with cache.inference_slot():
            started.set()
            order.append(f"{name}-enter")
            order.append(f"{name}-exit")

    with cache.inference_slot():
        thread = threading.Thread(target=worker, args=("w",))
        thread.start()
        # The worker must be blocked while we hold the only slot.
        assert not started.wait(0.2)
    thread.join(timeout=5)
    assert order == ["w-enter", "w-exit"]
