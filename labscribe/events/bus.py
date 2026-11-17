from __future__ import annotations

import queue
import threading
from typing import Any


class EventBus:
    """Fan-out of runtime events to CLI/SSE-style subscribers."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subscribers: list[queue.Queue] = []

    def subscribe(self, maxsize: int = 1000) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=maxsize)
        with self._lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    def publish(self, kind: str, payload: Any) -> None:
        with self._lock:
            subscribers = list(self._subscribers)
        for q in subscribers:
            try:
                q.put_nowait({"kind": kind, "payload": payload})
            except queue.Full:
                continue
