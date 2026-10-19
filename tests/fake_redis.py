"""A small in-process stand-in for the redis client: just the calls `RedisBackend` makes, thread-safe, with a switch to
make every call fail like a lost connection. No server, no network."""
import threading

from redis.exceptions import ConnectionError as RedisConnectionError


class FakeRedis:
    def __init__(self) -> None:
        self.down = False  # True: every call raises, as when the server is unreachable
        self.calls = 0
        self._lock = threading.Lock()
        self._hashes: dict[str, dict[str, str]] = {}
        self._sorted: dict[str, dict[str, float]] = {}

    def _enter(self) -> None:
        self.calls += 1
        if self.down:
            raise RedisConnectionError("fake redis is down")

    def ping(self) -> bool:
        with self._lock:
            self._enter()
            return True

    def hget(self, name, key):
        with self._lock:
            self._enter()
            return self._hashes.get(name, {}).get(key)

    def hset(self, name, key, value):
        with self._lock:
            self._enter()
            self._hashes.setdefault(name, {})[key] = value

    def hgetall(self, name):
        with self._lock:
            self._enter()
            return dict(self._hashes.get(name, {}))

    def hlen(self, name):
        with self._lock:
            self._enter()
            return len(self._hashes.get(name, {}))

    def hdel(self, name, *keys):
        with self._lock:
            self._enter()
            for key in keys:
                self._hashes.get(name, {}).pop(key, None)

    def zadd(self, name, mapping):
        with self._lock:
            self._enter()
            self._sorted.setdefault(name, {}).update(mapping)

    def zpopmin(self, name, count):
        with self._lock:
            self._enter()
            members = sorted(self._sorted.get(name, {}).items(), key=lambda item: item[1])[:count]
            for member, _ in members:
                del self._sorted[name][member]
            return members

    def delete(self, *names):
        with self._lock:
            self._enter()
            for name in names:
                self._hashes.pop(name, None)
                self._sorted.pop(name, None)
