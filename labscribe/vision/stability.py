from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class StableReading:
    value: float | None
    confidence: float
    stable: bool
    changed: bool = False
    since_ts: float | None = None


class ReadingStabilizer:
    """Turns throttled OCR samples into a debounced, stable display value.

    A value is considered stable when the last ``stable_samples`` samples agree
    within ``tolerance_abs + tolerance_rel * |value|``.  ``changed`` is emitted
    exactly once per transition to a new stable value, which keeps frame-level
    OCR jitter from producing duplicate "reading changed" events.
    """

    def __init__(
        self,
        stable_samples: int = 3,
        tolerance_abs: float = 0.25,
        tolerance_rel: float = 0.01,
        max_gap_s: float = 2.0,
    ):
        self.stable_samples = max(1, int(stable_samples))
        self.tolerance_abs = float(tolerance_abs)
        self.tolerance_rel = float(tolerance_rel)
        self.max_gap_s = float(max_gap_s)
        self._samples: list[tuple[float, float, float]] = []
        self._last_ts: float | None = None
        self._last_stable: float | None = None
        self._last_stable_conf: float = 0.0
        self._stable_since: float | None = None

    @property
    def current(self) -> float | None:
        return self._last_stable

    def reset(self) -> None:
        self._samples.clear()
        self._last_ts = None
        self._last_stable = None
        self._last_stable_conf = 0.0
        self._stable_since = None

    def update(self, ts: float, value: float | None, confidence: float = 1.0) -> StableReading:
        if self._last_ts is not None and ts - self._last_ts > self.max_gap_s:
            self._samples.clear()
            self._last_stable = None
            self._last_stable_conf = 0.0
            self._stable_since = None
        self._last_ts = ts

        if value is None or confidence <= 0:
            return StableReading(
                value=self._last_stable,
                confidence=self._last_stable_conf,
                stable=False,
                changed=False,
                since_ts=self._stable_since,
            )

        self._samples.append((float(ts), float(value), float(confidence)))
        cutoff = ts - self.max_gap_s
        self._samples = [s for s in self._samples if s[0] >= cutoff]
        if len(self._samples) > self.stable_samples * 4:
            self._samples = self._samples[-self.stable_samples * 4 :]

        if len(self._samples) < self.stable_samples:
            return StableReading(
                value=self._last_stable,
                confidence=self._last_stable_conf,
                stable=False,
                changed=False,
                since_ts=self._stable_since,
            )

        window = self._samples[-self.stable_samples :]
        values = [s[1] for s in window]
        mean = sum(values) / len(values)
        spread = max(values) - min(values)
        tolerance = self.tolerance_abs + self.tolerance_rel * abs(mean)
        if spread > tolerance:
            return StableReading(
                value=self._last_stable,
                confidence=self._last_stable_conf,
                stable=False,
                changed=False,
                since_ts=self._stable_since,
            )

        stable_value = values[-1]
        confidence = min(s[2] for s in window)
        changed = False
        if self._last_stable is None or abs(stable_value - self._last_stable) > tolerance:
            changed = self._last_stable is not None
            self._last_stable = stable_value
            self._last_stable_conf = confidence
            self._stable_since = ts
        else:
            self._last_stable_conf = confidence
        return StableReading(
            value=self._last_stable,
            confidence=self._last_stable_conf,
            stable=True,
            changed=changed,
            since_ts=self._stable_since,
        )
