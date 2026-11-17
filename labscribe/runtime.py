from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Any

from .agent import LabAgent
from .agent.tools import LabTools
from .config import Config
from .events import EventBus, EventProcessor, RuleStateMachine
from .schema import LabEvent
from .sop import SOPRegistry
from .store import Database, FrameStore
from .vision.frame_source import build_source
from .vision.pipeline import VisionPipeline
from .vision.synthetic import (
    ScenarioSpec,
    SyntheticSource,
    make_demo_scenario,
    make_scenario,
)

log = logging.getLogger(__name__)


class Runtime:
    """Owns the vision thread, the agent worker and all shared services."""

    def __init__(self, cfg: Config | None = None, agent_mode: str | None = None):
        self.cfg = cfg or Config.load()
        self.cfg.ensure_dirs()
        self.db = Database(self.cfg.db_path)
        self.frame_store = FrameStore(
            self.db,
            self.cfg.frames_dir,
            max_buffered=int(self.cfg.get("store.max_buffered_frames", 240)),
        )
        self.bus = EventBus()
        self.machine = RuleStateMachine(self.cfg)
        self.processor = EventProcessor(
            self.cfg, self.machine, self.db, self.frame_store, self.bus
        )
        self.sops = SOPRegistry(self.cfg.sop_dir)
        self.tools = LabTools(self.cfg, self.db, self.sops)
        self.agent = LabAgent(
            self.cfg, self.db, self.tools, mode=agent_mode or str(self.cfg.get("agent.mode", "llm"))
        )
        self.agent_queue: queue.Queue[LabEvent | None] = queue.Queue()
        self._lock = threading.RLock()
        self._vision_thread: threading.Thread | None = None
        self._agent_thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._vision_done = threading.Event()
        self.scenario: ScenarioSpec | None = None
        self.source_kind = str(self.cfg.get("vision.source", "synthetic"))
        self.frames_processed = 0
        self.events_emitted = 0
        self.last_error: str | None = None
        self.started_at: float | None = None
        self.finished_at: float | None = None
        self.realtime = bool(self.cfg.get("vision.realtime", False))

    # -- lifecycle --------------------------------------------------------
    @property
    def running(self) -> bool:
        return bool(self._vision_thread and self._vision_thread.is_alive())

    def start(
        self,
        scenario: ScenarioSpec | None = None,
        agent_mode: str | None = None,
        realtime: bool | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            if self.running:
                self.stop()
            if agent_mode and agent_mode != self.agent.mode:
                self.agent = LabAgent(self.cfg, self.db, self.tools, mode=agent_mode)
            if realtime is not None:
                self.realtime = bool(realtime)
            # Fresh rule state per run so sessions never bleed across scenarios.
            self.machine = RuleStateMachine(self.cfg)
            self.processor = EventProcessor(
                self.cfg, self.machine, self.db, self.frame_store, self.bus
            )
            self.scenario = scenario
            source = build_source(self.cfg, scenario)
            self.source_kind = (
                "synthetic"
                if isinstance(source, SyntheticSource)
                else str(self.cfg.get("vision.source", "synthetic"))
            )
            pipeline = VisionPipeline(self.cfg)
            self._stop.clear()
            self._vision_done.clear()
            self.agent_queue = queue.Queue()
            self.frames_processed = 0
            self.events_emitted = 0
            self.last_error = None
            self.started_at = time.time()
            self.finished_at = None
            self._vision_thread = threading.Thread(
                target=self._vision_loop,
                args=(source, pipeline),
                name="labscribe-vision",
                daemon=True,
            )
            self._agent_thread = threading.Thread(
                target=self._agent_loop, name="labscribe-agent", daemon=True
            )
            self._vision_thread.start()
            self._agent_thread.start()
            return self.status()

    def stop(self, timeout: float = 5.0) -> dict[str, Any]:
        self._stop.set()
        try:
            self.agent_queue.put_nowait(None)
        except queue.Full:
            pass
        for thread in (self._vision_thread, self._agent_thread):
            if thread and thread.is_alive():
                thread.join(timeout=timeout)
        self._vision_thread = None
        return self.status()

    def wait_idle(self, timeout: float = 180.0) -> bool:
        deadline = time.time() + timeout
        for thread in (self._vision_thread, self._agent_thread):
            if thread is None:
                continue
            remaining = max(0.1, deadline - time.time())
            thread.join(timeout=remaining)
            if thread.is_alive():
                return False
        return True

    def dispatch_event(self, event: LabEvent) -> None:
        """Send a manually confirmed/ingested event to the agent."""
        if self.running:
            self.agent_queue.put(event)
            return
        threading.Thread(
            target=self._handle_single_event, args=(event,), daemon=True
        ).start()

    def _handle_single_event(self, event: LabEvent) -> None:
        try:
            result = self.agent.handle_event(event)
            self.bus.publish("agent", result.to_dict())
        except Exception as exc:  # noqa: BLE001
            log.exception("agent failed on dispatched event")
            self.last_error = f"agent: {type(exc).__name__}: {exc}"

    def close(self) -> None:
        self.stop()
        self.db.close()

    # -- loops ------------------------------------------------------------
    def _vision_loop(self, source, pipeline: VisionPipeline) -> None:
        fps = float(self.cfg.get("vision.fps", 10) or 10)
        speed = max(0.1, float(self.cfg.get("synthetic.speed", 1.0) or 1.0))
        max_frames = int(self.cfg.get("vision.max_frames", 0) or 0)
        interval = 1.0 / (fps * speed) if self.realtime else 0.0
        next_at = time.perf_counter()
        try:
            for frame in source.iter_frames():
                if self._stop.is_set():
                    break
                self.frame_store.keep(frame)
                observations = pipeline.process(frame)
                for event in self.processor.handle_observations(observations):
                    self.agent_queue.put(event)
                    self.events_emitted += 1
                    self.bus.publish("observation_event", event.to_dict())
                self.frames_processed += 1
                if max_frames and self.frames_processed >= max_frames:
                    break
                if interval > 0:
                    next_at += interval
                    delay = next_at - time.perf_counter()
                    if delay > 0:
                        time.sleep(delay)
        except Exception as exc:  # noqa: BLE001
            log.exception("vision loop failed")
            self.last_error = f"{type(exc).__name__}: {exc}"
            self.bus.publish("error", {"stage": "vision", "error": self.last_error})
        finally:
            try:
                pipeline.close()
            except Exception:  # noqa: BLE001
                pass
            self._vision_done.set()
            self.finished_at = time.time()
            try:
                self.agent_queue.put_nowait(None)
            except queue.Full:
                pass
            self.bus.publish("vision_done", self.status())

    def _agent_loop(self) -> None:
        while True:
            try:
                item = self.agent_queue.get(timeout=0.25)
            except queue.Empty:
                if self._vision_done.is_set() or self._stop.is_set():
                    break
                continue
            if item is None:
                if self._vision_done.is_set() or self._stop.is_set():
                    break
                continue
            try:
                result = self.agent.handle_event(item)
                self.bus.publish("agent", result.to_dict())
            except Exception as exc:  # noqa: BLE001
                log.exception("agent worker failed")
                self.last_error = f"agent: {type(exc).__name__}: {exc}"
                self.bus.publish("error", {"stage": "agent", "error": self.last_error})
                if self._stop.is_set():
                    break

    # -- status -----------------------------------------------------------
    def status(self) -> dict[str, Any]:
        stats = self.db.stats()
        return {
            "running": self.running,
            "agent_running": bool(self._agent_thread and self._agent_thread.is_alive()),
            "vision_done": self._vision_done.is_set(),
            "source": self.source_kind,
            "scenario_id": self.scenario.scenario_id if self.scenario else None,
            "scenario_seed": self.scenario.seed if self.scenario else None,
            "agent_mode": self.agent.mode,
            "realtime": self.realtime,
            "frames_processed": self.frames_processed,
            "events_emitted": self.events_emitted,
            "agent_queue": self.agent_queue.qsize(),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "elapsed_s": round((self.finished_at or time.time()) - self.started_at, 2)
            if self.started_at
            else 0.0,
            "last_error": self.last_error,
            "stats": stats,
        }

    def run_demo(self, agent_mode: str | None = None) -> dict[str, Any]:
        seed = int(self.cfg.get("synthetic.seed", 7))
        scenario = make_demo_scenario(fps=int(self.cfg.get("vision.fps", 10)))
        scenario.seed = seed
        return self.start(scenario=scenario, agent_mode=agent_mode)

    def run_scenario(
        self,
        seed: int,
        occluded: bool = False,
        agent_mode: str | None = None,
        scenario_id: str | None = None,
    ) -> dict[str, Any]:
        scenario = make_scenario(
            seed=seed,
            scenario_id=scenario_id or f"api-{seed}",
            occluded=occluded,
            fps=int(self.cfg.get("vision.fps", 10)),
        )
        return self.start(scenario=scenario, agent_mode=agent_mode)
