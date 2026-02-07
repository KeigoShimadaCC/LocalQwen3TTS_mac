from __future__ import annotations

from collections import OrderedDict
from typing import Hashable, Tuple


class AudioCache:
    def __init__(self, max_size: int = 32):
        self.max_size = max_size
        self._store: OrderedDict[Hashable, Tuple[bytes, str, int, float]] = (
            OrderedDict()
        )

    def _make_key(self, *parts: Hashable) -> Hashable:
        return parts

    def get(self, key: Hashable):
        if key not in self._store:
            return None
        value = self._store.pop(key)
        self._store[key] = value
        return value

    def put(self, key: Hashable, value):
        if key in self._store:
            self._store.pop(key)
        self._store[key] = value
        if len(self._store) > self.max_size:
            self._store.popitem(last=False)
