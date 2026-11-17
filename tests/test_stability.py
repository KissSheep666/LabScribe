from __future__ import annotations

from labscribe.vision.stability import ReadingStabilizer


def test_stable_after_configured_samples():
    stabilizer = ReadingStabilizer(stable_samples=3, tolerance_abs=0.25, max_gap_s=2.0)
    first = stabilizer.update(0.0, 1.0)
    second = stabilizer.update(0.3, 1.0)
    third = stabilizer.update(0.6, 1.0)
    assert first.stable is False
    assert second.stable is False
    assert third.stable is True
    assert third.value == 1.0
    # The first stable value establishes the baseline, it is not a change.
    assert third.changed is False
    assert stabilizer.current == 1.0


def test_change_emitted_exactly_once():
    stabilizer = ReadingStabilizer(stable_samples=3, tolerance_abs=0.25, max_gap_s=2.0)
    for i in range(3):
        stabilizer.update(i * 0.3, 1.0)
    changes = []
    for i in range(6):
        result = stabilizer.update(1.0 + i * 0.3, 2.0)
        if result.changed:
            changes.append(result.value)
    assert changes == [2.0]
    assert stabilizer.current == 2.0


def test_jitter_within_tolerance_is_ignored():
    stabilizer = ReadingStabilizer(stable_samples=3, tolerance_abs=0.3, max_gap_s=2.0)
    stabilizer.update(0.0, 1.00)
    stabilizer.update(0.3, 1.08)
    result = stabilizer.update(0.6, 0.96)
    assert result.stable is True
    assert result.value == 0.96
    assert result.changed is False


def test_large_gap_resets_stability():
    stabilizer = ReadingStabilizer(stable_samples=2, max_gap_s=2.0)
    stabilizer.update(0.0, 1.0)
    assert stabilizer.update(0.3, 1.0).stable is True
    result = stabilizer.update(10.0, None)
    assert result.stable is False
    assert result.value is None


def test_missing_samples_do_not_confirm_a_change():
    stabilizer = ReadingStabilizer(stable_samples=3, max_gap_s=5.0)
    for i in range(3):
        stabilizer.update(i * 0.3, 1.0)
    # Two new values, then the display is occluded before the third sample.
    stabilizer.update(2.0, 2.0)
    stabilizer.update(2.3, 2.0)
    result = stabilizer.update(2.6, None)
    assert result.stable is False
