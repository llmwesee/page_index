from __future__ import annotations

from collections import OrderedDict
import threading
import time
from typing import Dict, Generic, Hashable, Optional, TypeVar


KeyT = TypeVar("KeyT", bound=Hashable)
ValueT = TypeVar("ValueT")


class LruTtlCache(Generic[KeyT, ValueT]):
    def __init__(
        self,
        *,
        name: str,
        max_entries: int = 128,
        ttl_seconds: Optional[int] = None,
    ) -> None:
        self.name = name
        self.max_entries = max(1, int(max_entries))
        self.ttl_seconds = max(1, int(ttl_seconds)) if ttl_seconds else None
        self._items: OrderedDict[KeyT, tuple[float, ValueT]] = OrderedDict()
        self._lock = threading.RLock()
        self._stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "evictions": 0,
            "expirations": 0,
        }

    def configure(self, *, max_entries: Optional[int] = None, ttl_seconds: Optional[int] = None) -> None:
        with self._lock:
            if max_entries is not None:
                self.max_entries = max(1, int(max_entries))
            self.ttl_seconds = max(1, int(ttl_seconds)) if ttl_seconds else None
            self._expire_locked()
            self._trim_locked()

    def get(self, key: KeyT) -> Optional[ValueT]:
        with self._lock:
            self._expire_locked()
            if key not in self._items:
                self._stats["misses"] += 1
                return None
            created_at, value = self._items.pop(key)
            self._items[key] = (created_at, value)
            self._stats["hits"] += 1
            return value

    def set(self, key: KeyT, value: ValueT) -> None:
        with self._lock:
            self._expire_locked()
            if key in self._items:
                self._items.pop(key)
            self._items[key] = (time.time(), value)
            self._stats["sets"] += 1
            self._trim_locked()

    def pop(self, key: KeyT, default: Optional[ValueT] = None) -> Optional[ValueT]:
        with self._lock:
            entry = self._items.pop(key, None)
            if entry is None:
                return default
            _, value = entry
            return value

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def snapshot(self) -> Dict[str, int]:
        with self._lock:
            self._expire_locked()
            return {
                "max_entries": self.max_entries,
                "ttl_seconds": self.ttl_seconds or 0,
                "size": len(self._items),
                "hits": self._stats["hits"],
                "misses": self._stats["misses"],
                "sets": self._stats["sets"],
                "evictions": self._stats["evictions"],
                "expirations": self._stats["expirations"],
            }

    def _expire_locked(self) -> None:
        if not self.ttl_seconds:
            return
        cutoff = time.time() - self.ttl_seconds
        expired_keys = [key for key, (created_at, _) in self._items.items() if created_at < cutoff]
        for key in expired_keys:
            self._items.pop(key, None)
            self._stats["expirations"] += 1

    def _trim_locked(self) -> None:
        while len(self._items) > self.max_entries:
            self._items.popitem(last=False)
            self._stats["evictions"] += 1
