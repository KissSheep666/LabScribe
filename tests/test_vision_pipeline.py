from __future__ import annotations

from labscribe.config import Config
from labscribe.vision.detector import MarkerDetector
from labscribe.vision.ocr import TemplateOCR
from labscribe.vision.pipeline import VisionPipeline
from labscribe.vision.synthetic import SyntheticSource, make_demo_scenario
from labscribe.vision.tracker import IoUTracker


def test_template_ocr_reads_digital_display(cfg):
    spec = make_demo_scenario()
    source = SyntheticSource(spec)
    ocr = TemplateOCR(cfg)
    cases = [(0.5, 0.0), (4.5, 3.5), (9.2, 7.2), (11.6, 15.2)]
    for ts, expected in cases:
        frame = source.render(int(ts * spec.fps))
        result = ocr.read(frame.image, spec.display_bbox)
        assert result.ok, (ts, result)
        assert result.value == expected, (ts, result)
        assert result.confidence >= 0.5


def test_occluded_display_yields_no_reliable_reading(cfg):
    spec = make_demo_scenario()
    source = SyntheticSource(spec)
    ocr = TemplateOCR(cfg)
    frame = source.render(int(8.4 * spec.fps))
    result = ocr.read(frame.image, spec.display_bbox)
    assert (not result.ok) or result.confidence < 0.6


def test_marker_detector_finds_instrument_and_actor(cfg):
    spec = make_demo_scenario()
    source = SyntheticSource(spec)
    detector = MarkerDetector(cfg)
    frame = source.render(int(2.0 * spec.fps))
    detections = detector.detect(frame)
    labels = {d.label for d in detections}
    assert labels == {"instrument", "actor"}
    instrument = next(d for d in detections if d.label == "instrument")
    actor = next(d for d in detections if d.label == "actor")
    assert instrument.confidence > 0.8
    assert actor.confidence > 0.8
    assert "display_roi" in instrument.meta


def test_pipeline_marks_display_occlusion(cfg):
    spec = make_demo_scenario()
    source = SyntheticSource(spec)
    pipeline = VisionPipeline(
        cfg, detector=MarkerDetector(cfg), tracker=IoUTracker(), ocr=TemplateOCR(cfg)
    )
    # Warm up through the first session so the tracker keeps stable ids.
    for index in range(0, int(8.4 * spec.fps)):
        pipeline.process(source.render(index))
    frame = source.render(int(8.4 * spec.fps))
    observations = pipeline.process(frame)
    assert len(observations) == 1
    observation = observations[0]
    assert observation.occluded is True
    assert observation.reading is None or observation.reading_stable is False


def test_pipeline_emits_stable_reading_after_change(cfg):
    spec = make_demo_scenario()
    source = SyntheticSource(spec)
    pipeline = VisionPipeline(
        cfg, detector=MarkerDetector(cfg), tracker=IoUTracker(), ocr=TemplateOCR(cfg)
    )
    stable_value = None
    for index in range(int(3.8 * spec.fps), int(6.0 * spec.fps)):
        observations = pipeline.process(source.render(index))
        if observations and observations[0].reading is not None:
            stable_value = observations[0].reading
    assert stable_value == 3.5


def test_tracker_keeps_instrument_id_stable(cfg):
    spec = make_demo_scenario()
    source = SyntheticSource(spec)
    pipeline = VisionPipeline(
        cfg, detector=MarkerDetector(cfg), tracker=IoUTracker(), ocr=TemplateOCR(cfg)
    )
    ids = set()
    for index in range(0, 40):
        for observation in pipeline.process(source.render(index)):
            ids.add(observation.object_id)
    assert ids == {"instrument-1"}
