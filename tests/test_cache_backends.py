"""T6.11: Redis backend with an in-memory fallback, cleared on a KB change, safe under threads
(SPEC section 4 cache AC, FR-F6, F7, H6). No Redis server is needed: a fake client stands in, and an unreachable URL
checks the real client's failure path. The exact and semantic cache suites also run on both backends (`new_backend`)."""
import sys
import threading
from types import SimpleNamespace

import pytest
from redis.exceptions import ConnectionError as RedisConnectionError

from smartdesk_ai.agent.state import ChunkRef, PartResult
from smartdesk_ai.config.env import EnvSettings
from smartdesk_ai.ops.cache import (
    AnswerCache,
    FallbackBackend,
    MemoryBackend,
    RedisBackend,
    clear_answers,
    open_backend,
)
from smartdesk_ai.ops.cache_log import CacheLog, load_cache_records
from smartdesk_ai.ops.health import Component, HealthRegistry, Status
from smartdesk_ai.rag.embeddings import HashingEmbeddings
from smartdesk_ai.rag.ingest import ingest
from tests.fake_redis import FakeRedis

QUESTION = "How do I reset my forgotten password?"


def answered(text="Use the self-service page.") -> PartResult:
    return PartResult(answer=f"{text}\n\nSources: IT-1 (Password)", citations=["IT-1"],
                      chunks=[ChunkRef(entry_id="IT-1", title="Password", score=0.8)], answerable=True)


# --- both backends keep the same contract -----------------------------------------------------------


def test_a_backend_stores_reads_overwrites_lists_and_clears(new_backend):
    backend = new_backend(50)
    assert backend.get("a") is None and len(backend) == 0 and backend.entries() == []
    backend.put("a", "1")
    backend.put("b", "2")
    backend.put("a", "3")
    assert backend.get("a") == "3" and len(backend) == 2 and dict(backend.entries()) == {"a": "3", "b": "2"}
    backend.clear()
    assert len(backend) == 0 and backend.get("a") is None


def test_a_full_backend_drops_its_oldest_entries_first(new_backend):
    backend = new_backend(3)
    for index in range(5):
        backend.put(f"k{index}", str(index))
    assert len(backend) == 3 and [k for k, _ in sorted(backend.entries())] == ["k2", "k3", "k4"]


def test_namespaces_do_not_share_entries():
    client = FakeRedis()
    one, two = RedisBackend(client, namespace="a", max_entries=5), RedisBackend(client, namespace="b", max_entries=5)
    one.put("k", "1")
    assert two.get("k") is None and len(two) == 0
    two.clear()
    assert one.get("k") == "1"


# --- Redis goes away and comes back --------------------------------------------------------------------


class Switch:
    """Records the health callbacks and gives the fallback a clock it can move."""

    def __init__(self):
        self.now, self.downs, self.ups = 1000.0, [], 0

    def clock(self):
        return self.now

    def down(self, error):
        self.downs.append(type(error).__name__)

    def up(self):
        self.ups += 1


def fallback(client, switch, retry_after=30.0):
    return FallbackBackend(RedisBackend(client, namespace="t", max_entries=10), MemoryBackend(10),
                           retry_after=retry_after, clock=switch.clock, on_down=switch.down, on_up=switch.up)


def test_the_cache_keeps_working_in_memory_while_redis_is_down():
    client, switch = FakeRedis(), Switch()
    client.down = True
    cache = AnswerCache(fallback(client, switch), kb_version=lambda: "v1")
    cache.store(cache.lookup(QUESTION, None), answered(), compute_ms=50.0)
    assert cache.lookup(QUESTION, None).kind == "exact"  # served from the in-memory substitute
    assert switch.downs == ["ConnectionError"]  # reported once, not on every call
    cache.clear()  # must not raise either
    assert cache.lookup(QUESTION, None).kind == "miss"


def test_redis_is_tried_again_after_the_wait_and_used_once_it_answers():
    client, switch = FakeRedis(), Switch()
    backend = fallback(client, switch, retry_after=30.0)
    client.down = True
    backend.put("k", "in memory")
    calls = client.calls
    backend.get("k")
    assert client.calls == calls  # still waiting: Redis is not asked on every call
    client.down = False
    switch.now += 31
    backend.put("k2", "in redis")
    assert switch.ups == 1 and RedisBackend(client, namespace="t", max_entries=10).get("k2") == "in redis"


def test_a_redis_that_stays_down_is_asked_once_per_wait():
    client, switch = FakeRedis(), Switch()
    client.down = True
    backend = fallback(client, switch)
    for _ in range(5):
        backend.get("k")
    first = client.calls
    switch.now += 31
    backend.get("k")
    assert first == 1 and client.calls == 2 and switch.downs == ["ConnectionError", "ConnectionError"]


# --- choosing the backend ---------------------------------------------------------------------------------


def env(url):
    return EnvSettings(redis_url=url)


def test_without_a_redis_url_the_cache_is_in_memory(settings):
    registry = HealthRegistry()
    backend = open_backend(settings, env(None), registry, mode="mock")
    assert isinstance(backend, MemoryBackend) and registry.get(Component.REDIS).status is Status.UNKNOWN


def test_a_reachable_redis_is_the_backend_and_is_marked_ok(settings):
    registry, client = HealthRegistry(), FakeRedis()
    backend = open_backend(settings, env("redis://cache:6379/0"), registry, mode="mock", client_factory=lambda url, t: client)
    assert isinstance(backend, FallbackBackend) and registry.get(Component.REDIS).status is Status.OK
    backend.put("k", "v")
    assert RedisBackend(client, namespace="smartdesk:answers:mock", max_entries=10).get("k") == "v"


def test_a_redis_that_fails_to_connect_is_a_degraded_mode_not_a_crash(settings):
    registry = HealthRegistry()

    def refuse(url, timeout):
        raise RedisConnectionError("connection refused")

    backend = open_backend(settings, env("redis://cache:6379/0"), registry, mode="mock", client_factory=refuse)
    assert isinstance(backend, MemoryBackend)
    health = registry.get(Component.REDIS)
    assert health.status is Status.DEGRADED and health.mode == "in-memory cache"


def test_the_real_client_against_a_port_nobody_listens_on_degrades_quickly(settings):
    quick = settings.model_copy(update={"cache": settings.cache.model_copy(update={"redis_timeout_seconds": 0.3})})
    registry = HealthRegistry()
    backend = open_backend(quick, env("redis://127.0.0.1:1/0"), registry, mode="mock")
    assert isinstance(backend, MemoryBackend) and registry.get(Component.REDIS).status is Status.DEGRADED


def test_a_redis_that_drops_later_marks_the_component_degraded_and_back(settings):
    registry, client = HealthRegistry(), FakeRedis()
    backend = open_backend(settings, env("redis://cache:6379/0"), registry, mode="mock", client_factory=lambda url, t: client)
    client.down = True
    backend.put("k", "v")
    assert registry.get(Component.REDIS).status is Status.DEGRADED
    assert registry.get(Component.REDIS).reason.startswith("ConnectionError")  # the type only: no URL, no message


# --- threads --------------------------------------------------------------------------------------------


def run_threads(count, work):
    """Start `count` threads together and fail on the first error. Threads switch every microsecond, so a missing lock
    shows up as an error instead of hiding behind the interpreter's usual 5 ms slices."""
    errors, barrier = [], threading.Barrier(count)

    def target(index):
        try:
            barrier.wait()
            work(index)
        except Exception as error:  # noqa: BLE001 - collected and asserted on below
            errors.append(error)

    threads = [threading.Thread(target=target, args=(i,)) for i in range(count)]
    previous = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        [t.start() for t in threads]
        [t.join(30) for t in threads]
    finally:
        sys.setswitchinterval(previous)
    assert not any(t.is_alive() for t in threads), "a thread hung"
    assert errors == []


def test_many_threads_storing_and_reading_never_corrupt_the_cache(new_backend):
    cache = AnswerCache(new_backend(1000), kb_version=lambda: "v1")
    threads, per_thread = 16, 40

    def work(index):
        for n in range(per_thread):
            question = f"question {index} number {n}?"
            cache.store(cache.lookup(question, None), answered(f"answer {index}-{n}"), compute_ms=5.0)
            hit = cache.lookup(question, None)  # always my own answer, whatever the others are doing
            assert hit.kind == "exact"
            assert hit.part.answer.split("\n")[0] == f"answer {index}-{n}"
            cache.lookup(f"question {(index + 1) % threads} number {n}?", None)  # read what others are writing

    run_threads(threads, work)
    assert len(cache) == threads * per_thread


def test_threads_reading_while_others_write_and_clear_see_no_errors(new_backend):
    embedder = HashingEmbeddings()
    cache = AnswerCache(new_backend(30), kb_version=lambda: "v1", embed=embedder.embed_query, cutoff=0.9)

    def work(index):
        for n in range(60):
            question = f"how do i handle topic {n % 12} on device {index}?"
            probe = cache.lookup(question, None)  # scans every stored vector while the others change the set
            cache.store(probe, answered(f"a{n}"), compute_ms=1.0)
            if index == 0 and n % 20 == 19:
                cache.clear()

    run_threads(8, work)
    assert len(cache) <= 30


def test_every_access_to_the_memory_backends_data_happens_under_its_lock():
    """Deterministic, unlike a stress run: the data refuses to be touched unless the backend's lock is held."""
    from collections import OrderedDict

    backend = MemoryBackend(max_entries=2)

    class Guarded(OrderedDict):
        def _held(self):
            assert backend._lock.locked(), "the shared data was touched without the lock"

    for name in ("get", "pop", "popitem", "items", "clear", "__setitem__", "__len__"):
        def guard(self, *args, _name=name, **kwargs):
            self._held()
            return getattr(OrderedDict, _name)(self, *args, **kwargs)

        setattr(Guarded, name, guard)
    backend._data = Guarded()
    for key in ("a", "b", "c"):
        backend.put(key, "x")
    backend.get("c")
    backend.entries()
    len(backend)
    backend.clear()


def test_the_memory_backend_survives_a_tight_race_on_a_tiny_limit():
    backend = MemoryBackend(max_entries=5)

    def work(index):
        for n in range(1000):
            backend.put(f"{index}-{n % 7}", "x")
            assert len(backend) <= 5  # never seen mid-put, while the new entry is in and the oldest not yet out
            backend.get(f"{index}-{(n + 1) % 7}")
            assert len(backend.entries()) <= 5
            if n % 250 == 249:
                backend.clear()

    run_threads(12, work)
    assert len(backend) <= 5


# --- a KB change clears the cache -----------------------------------------------------------------------


def edit_first_policy(kb_dir):
    """A small content edit to an indexed KB file (its hash changes; the entries stay valid)."""
    target = kb_dir / "it_support.md"
    target.write_text(target.read_text().replace("password", "passphrase", 1))
    return target


def test_re_ingest_after_a_kb_edit_clears_the_cache_and_a_no_op_ingest_does_not(settings, kb_copy, tmp_path, new_backend):
    index_dir = tmp_path / "index"
    log = CacheLog(tmp_path / "cache.jsonl")
    cache = AnswerCache(new_backend(50), kb_version=lambda: "v1", log=log)
    handler = lambda report: clear_answers(cache.backend, log)  # noqa: E731 - what the app and the ingest script pass
    ingest(kb_copy, index_dir, HashingEmbeddings(), settings.chunking, on_kb_changed=handler)  # first build: cleared too
    cache.store(cache.lookup(QUESTION, None), answered(), compute_ms=10.0)
    assert ingest(kb_copy, index_dir, HashingEmbeddings(), settings.chunking, on_kb_changed=handler).kb_changed is False
    assert len(cache) == 1  # nothing changed, nothing cleared
    edit_first_policy(kb_copy)
    assert ingest(kb_copy, index_dir, HashingEmbeddings(), settings.chunking, on_kb_changed=handler).kb_changed is True
    assert len(cache) == 0 and cache.lookup(QUESTION, None).kind == "miss"
    assert [r.event for r in load_cache_records(tmp_path / "cache.jsonl")].count("clear") == 2


def mock_stack(settings, kb_copy, tmp_path, redis_client, monkeypatch):
    from smartdesk_ai.agent.factory import build_stack
    from smartdesk_ai.agent.graph import make_kb_responder

    monkeypatch.setattr("smartdesk_ai.ops.cache.redis_from_url", lambda url, timeout: redis_client)
    stack = build_stack(settings, environ={"REDIS_URL": "redis://cache:6379/0"}, dotenv_path=None, data_dir=tmp_path,
                        kb_dir=kb_copy, registry=HealthRegistry())
    return stack, make_kb_responder(stack.graph)


def test_the_app_keeps_its_answers_in_redis_and_a_kb_edit_plus_ingest_script_clears_them(settings, kb_copy, tmp_path, monkeypatch):
    from smartdesk_ai.rag import ingest as ingest_script

    client = FakeRedis()
    stack, responder = mock_stack(settings, kb_copy, tmp_path, client, monkeypatch)
    assert stack.registry.get(Component.REDIS).status is Status.OK
    first = responder("s1", QUESTION)
    assert first.debug["cache"]["status"] == "miss" and client.hlen("smartdesk:answers:mock:entries") == 1
    assert responder("s2", QUESTION).debug["cache"]["status"] == "hit"  # served from Redis
    edit_first_policy(kb_copy)
    code = ingest_script.main(["--kb-dir", str(kb_copy), "--index-dir", str(tmp_path / "index_mock"), "--mock"],
                              environ={"REDIS_URL": "redis://cache:6379/0"})
    assert code == 0 and client.hlen("smartdesk:answers:mock:entries") == 0  # cleared by the script
    assert responder("s3", QUESTION).debug["cache"]["status"] == "miss"


def test_a_re_ingest_in_another_process_makes_a_running_apps_in_memory_answers_unreachable(settings, kb_copy, tmp_path):
    from smartdesk_ai.agent.factory import build_stack
    from smartdesk_ai.agent.graph import make_kb_responder

    stack = build_stack(settings, environ={}, dotenv_path=None, data_dir=tmp_path, kb_dir=kb_copy, registry=HealthRegistry())
    responder = make_kb_responder(stack.graph)
    responder("s1", QUESTION)
    assert responder("s2", QUESTION).debug["cache"]["status"] == "hit"
    edit_first_policy(kb_copy)
    ingest(kb_copy, tmp_path / "index_mock", HashingEmbeddings(), settings.chunking)  # the script, with no way to reach this process
    assert responder("s3", QUESTION).debug["cache"]["status"] == "miss"


def test_starting_the_app_on_an_edited_kb_clears_the_shared_redis_cache(settings, kb_copy, tmp_path, monkeypatch):
    client = FakeRedis()
    _, responder = mock_stack(settings, kb_copy, tmp_path, client, monkeypatch)
    responder("s1", QUESTION)
    assert client.hlen("smartdesk:answers:mock:entries") == 1
    edit_first_policy(kb_copy)
    mock_stack(settings, kb_copy, tmp_path, client, monkeypatch)  # a restart: its ingest sees the edit
    assert client.hlen("smartdesk:answers:mock:entries") == 0


# --- the size bound holds when the order set and the hash disagree (deterministic, no threads) ---------------------

class ClearBetweenTheWrites(FakeRedis):
    """A `clear()` from another thread lands between one put's `hset` and its `zadd`, once: the put then leaves a name in
    the order set that is in no hash. This is the interleaving behind the rare size-bound failure under threads."""

    def __init__(self):
        super().__init__()
        self.armed = False

    def zadd(self, name, mapping):
        if self.armed:
            self.armed = False
            self.delete("ns:entries", "ns:order")
        super().zadd(name, mapping)


def test_a_clear_between_a_puts_two_writes_does_not_break_the_size_bound():
    client = ClearBetweenTheWrites()
    backend = RedisBackend(client, namespace="ns", max_entries=3)
    client.armed = True
    backend.put("lost", "x")  # its hash entry is wiped by the clear, its name stays in the order set
    assert len(backend) == 0
    for key in "abcdef":
        backend.put(key, key)
        assert len(backend) <= 3, f"over the bound after putting {key}"
    assert sorted(k for k, _ in backend.entries()) == ["d", "e", "f"], "the newest three survive"


def test_names_in_the_order_set_that_are_in_no_hash_are_skipped_not_counted():
    client = FakeRedis()
    backend = RedisBackend(client, namespace="ns", max_entries=2)
    client.zadd("ns:order", {"ghost1": 0.1, "ghost2": 0.2, "ghost3": 0.3})
    for key in "abcd":
        backend.put(key, key)
    assert sorted(k for k, _ in backend.entries()) == ["c", "d"]


def test_entries_missing_from_the_order_set_are_still_evicted_to_keep_the_bound():
    client = FakeRedis()
    backend = RedisBackend(client, namespace="ns", max_entries=3)
    for key in "vwxyz":  # written without an order entry, as a half-finished put or an older writer would leave them
        client.hset("ns:entries", key, key)
    backend.put("new", "n")
    assert len(backend) <= 3 and backend.get("new") == "n", "the bound holds and the new entry is the one kept"


def test_the_bound_holds_through_overwrites_and_evictions(new_backend):
    backend = new_backend(3)
    for round_ in range(3):
        for key in "abcde":
            backend.put(key, f"{key}{round_}")
            assert len(backend) <= 3
    assert len(backend) == 3
