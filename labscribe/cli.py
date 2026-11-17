from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

from .config import Config
from .paths import resolve_path
from .runtime import Runtime
from .vision.synthetic import make_demo_scenario, make_scenario


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )


def _fmt(value, digits: int = 1) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _drain_bus(rt: Runtime, timeout: float = 1.0) -> list[dict]:
    q = getattr(rt, "_cli_queue", None)
    messages: list[dict] = []
    if q is None:
        return messages
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            messages.append(q.get(timeout=0.2))
        except Exception:  # noqa: BLE001 - queue.Empty
            break
    return messages


def _print_run_summary(rt: Runtime, started_wall: float, agent_mode: str) -> None:
    # Simulation timestamps start at 0, so select this run by wall-clock agent
    # traces and by the events emitted during the run.
    messages = _drain_bus(rt)
    emitted = [m["payload"] for m in messages if m["kind"] == "observation_event"]
    agent_results = [m["payload"] for m in messages if m["kind"] == "agent"]
    pending = [m["payload"] for m in messages if m["kind"] == "pending"]
    errors = [m["payload"] for m in messages if m["kind"] == "error"]

    print("\n=== Events ===")
    if emitted:
        for e in sorted(emitted, key=lambda x: x["ts"]):
            print(
                f"  t={_fmt(e['ts'], 2):>6}s  {e['type']:<14} reading={str(e.get('reading')):<6} "
                f"conf={_fmt(e['confidence'], 2)}  session={e.get('session_id') or '-'}  frame={e['source_frame']}"
            )
    else:
        print("  (none)")

    print("\n=== Agent runs ===")
    if agent_results:
        for r in agent_results:
            print(
                f"  [{r['status']:<13}] {r['mode']:<5} event={r['event_id']} "
                f"tools={r['tool_calls']} {r['duration_ms']:.0f}ms  {r['summary'][:100]}"
            )
    else:
        print("  (none)")

    print("\n=== Experiment records ===")
    records = [r for r in rt.db.list_records(limit=500) if r.get("created_at", 0) >= started_wall - 1.0]
    if records:
        for r in sorted(records, key=lambda x: x["created_at"]):
            note = (r.get("note") or "").splitlines()[0]
            print(f"  [{r.get('step_id') or '--':<3}] {note}")
            print(f"        frame={r.get('source_frame')} event={r.get('event_id')} record={r.get('record_id')}")
    else:
        print("  (none)")

    alerts = [a for a in rt.db.list_alerts(limit=500) if a.get("created_at", 0) >= started_wall - 1.0]
    if alerts:
        print("\n=== Alerts ===")
        for a in alerts:
            print(f"  {a['severity'].upper():<8} {a['message']}  (event={a.get('event_id')})")

    experiment_id = str(rt.cfg.get("agent.experiment_id", "exp-titration"))
    completed = rt.db.completed_steps(experiment_id, f"progress:{experiment_id}")
    sop = rt.sops.get(experiment_id)
    next_step = None
    if sop:
        next_step = next((s.id for s in sop.steps if s.id not in completed), None)
    print(f"\n=== SOP {experiment_id} ===")
    print(f"  completed={completed}  next={next_step}")
    if pending:
        print(f"  pending low-confidence observations: {len(pending)}")
    if errors:
        print(f"  errors: {errors}")
    stats = rt.db.stats()
    print(
        f"\n=== Run ===\n  frames={rt.frames_processed} events={rt.events_emitted} "
        f"agent={agent_mode} scenario={rt.scenario.scenario_id if rt.scenario else rt.source_kind} "
        f"db={stats['events']} events / {stats['records']} records"
    )


def cmd_simulate(args: argparse.Namespace) -> int:
    cfg = Config.load(args.config)
    agent_mode = args.agent or str(cfg.get("agent.mode", "llm"))
    rt = Runtime(cfg, agent_mode=agent_mode)
    rt._cli_queue = rt.bus.subscribe()  # type: ignore[attr-defined]
    if args.seed is not None:
        spec = make_scenario(
            seed=args.seed,
            scenario_id=args.scenario_id or f"sim-{args.seed}",
            occluded=args.occluded,
            fps=int(cfg.get("vision.fps", 10)),
        )
    else:
        spec = make_demo_scenario(fps=int(cfg.get("vision.fps", 10)))
    print(
        f"LabScribe simulate · scenario={spec.scenario_id} seed={spec.seed} "
        f"duration={spec.duration}s fps={spec.fps} agent={agent_mode}"
    )
    started_wall = time.time()
    rt.start(scenario=spec, agent_mode=agent_mode, realtime=args.realtime)
    idle = rt.wait_idle(args.timeout)
    if not idle:
        print(f"warning: run did not finish within {args.timeout}s", file=sys.stderr)
        rt.stop()
    _print_run_summary(rt, started_wall, agent_mode)
    if rt.last_error:
        print(f"error: {rt.last_error}", file=sys.stderr)
    rt.close()
    return 0 if idle and not rt.last_error else 1


def cmd_run(args: argparse.Namespace) -> int:
    cfg = Config.load(args.config)
    if args.source:
        cfg.set("vision.source", args.source)
    if args.camera is not None:
        cfg.set("vision.camera_index", args.camera)
    if args.video:
        cfg.set("vision.video_path", args.video)
    if args.images:
        cfg.set("vision.image_dir", args.images)
    if args.max_frames:
        cfg.set("vision.max_frames", args.max_frames)
    agent_mode = args.agent or str(cfg.get("agent.mode", "llm"))
    rt = Runtime(cfg, agent_mode=agent_mode)
    rt._cli_queue = rt.bus.subscribe()  # type: ignore[attr-defined]
    started_wall = time.time()
    print(f"LabScribe run · source={cfg.get('vision.source')} agent={agent_mode}")
    # CameraCapture paces itself; video/image sources are paced by the runtime.
    realtime = str(cfg.get("vision.source")) != "camera"
    rt.start(agent_mode=agent_mode, realtime=realtime)
    try:
        idle = rt.wait_idle(args.timeout)
    except KeyboardInterrupt:
        print("\ninterrupted; stopping…")
        rt.stop()
        idle = True
    _print_run_summary(rt, started_wall, agent_mode)
    rt.close()
    return 0 if idle and not rt.last_error else 1


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .api.app import create_app

    cfg = Config.load(args.config)
    if args.agent:
        cfg.set("agent.mode", args.agent)
    host = args.host or str(cfg.get("api.host", "127.0.0.1"))
    port = args.port or int(cfg.get("api.port", 8000))
    app = create_app(cfg)
    print(f"LabScribe dashboard: http://{host}:{port}/  (agent={cfg.get('agent.mode')})")
    uvicorn.run(app, host=host, port=port, log_level="info")
    return 0


def cmd_dataset(args: argparse.Namespace) -> int:
    from .eval import build_dataset

    cfg = Config.load(args.config)
    result = build_dataset(
        count=args.count or int(cfg.get("eval.count", 100)),
        out_path=args.out or cfg.dataset_path,
        seed=args.seed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from .eval import run_evaluation

    cfg = Config.load(args.config)
    report = run_evaluation(
        cfg,
        dataset_path=args.dataset,
        limit=args.limit,
        write_reports=not args.no_report,
        verbose=True,
    )
    from .eval.harness import render_markdown

    print("\n" + render_markdown(report))
    if "report_paths" in report:
        print(f"\nreports: {report['report_paths']}")
    return 0


def cmd_trace(args: argparse.Namespace) -> int:
    from .store import Database

    cfg = Config.load(args.config)
    db = Database(cfg.db_path)
    record = db.get_record(args.record_id)
    if record is None:
        matching = [r for r in db.list_records(limit=1000) if args.record_id in r["record_id"]]
        if not matching:
            print(f"record not found: {args.record_id}", file=sys.stderr)
            db.close()
            return 1
        record = matching[0]
    event = db.get_event(record.get("event_id") or "")
    frame = db.get_frame(record.get("source_frame") or "")
    runs = [db.get_run(r["run_id"]) for r in db.runs_for_event(record.get("event_id") or "")]
    payload = {
        "trace_chain": ["record", "event", "frame"],
        "record": record,
        "event": event,
        "frame": frame,
        "agent_runs": runs,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if frame:
        path = resolve_path(frame.get("path"))
        print(f"\nframe file: {path} (exists={path.exists() if path else False})", file=sys.stderr)
    db.close()
    return 0


def cmd_pending(args: argparse.Namespace) -> int:
    from .store import Database

    cfg = Config.load(args.config)
    db = Database(cfg.db_path)
    items = db.list_pending(status=args.status, limit=args.limit)
    if not items:
        print(f"no pending items with status={args.status}")
    for item in items:
        draft = item.get("draft_event") or {}
        print(
            f"{item['pending_id']}  {item['kind']:<14} conf={item['confidence']:.2f} "
            f"t={item['ts']:.1f}s  reading={draft.get('reading')}  frame={item['source_frame']}\n"
            f"    reason: {item['reason']}"
        )
    db.close()
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    from .store import Database

    cfg = Config.load(args.config)
    db = Database(cfg.db_path)
    print(json.dumps(db.stats(), ensure_ascii=False, indent=2))
    db.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="labscribe",
        description="LabScribe: lab operation recording agent (vision -> events -> tool calls)",
    )
    parser.add_argument("--config", help="path to config yaml (default: configs/default.yaml)")
    parser.add_argument("--verbose", action="store_true", help="enable debug/info logs")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("simulate", help="run a synthetic scenario end to end")
    p.add_argument("--agent", choices=["llm", "rules", "off"])
    p.add_argument("--seed", type=int)
    p.add_argument("--scenario-id")
    p.add_argument("--occluded", action="store_true")
    p.add_argument("--realtime", action="store_true")
    p.add_argument("--timeout", type=float, default=300.0)
    p.set_defaults(func=cmd_simulate)

    p = sub.add_parser("run", help="run from camera / video / image directory")
    p.add_argument("--agent", choices=["llm", "rules", "off"])
    p.add_argument("--source", choices=["camera", "video", "images", "synthetic"])
    p.add_argument("--camera", type=int)
    p.add_argument("--video")
    p.add_argument("--images")
    p.add_argument("--max-frames", type=int)
    p.add_argument("--timeout", type=float, default=3600.0)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("serve", help="start the FastAPI dashboard")
    p.add_argument("--host")
    p.add_argument("--port", type=int)
    p.add_argument("--agent", choices=["llm", "rules", "off"])
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("eval", help="run the annotated event evaluation")
    p.add_argument("--dataset")
    p.add_argument("--limit", type=int)
    p.add_argument("--no-report", action="store_true")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("dataset", help="(re)build the annotated event dataset")
    p.add_argument("--count", type=int)
    p.add_argument("--seed", type=int, default=20260101)
    p.add_argument("--out")
    p.set_defaults(func=cmd_dataset)

    p = sub.add_parser("trace", help="print record -> event -> frame trace")
    p.add_argument("record_id")
    p.set_defaults(func=cmd_trace)

    p = sub.add_parser("pending", help="list low-confidence pending observations")
    p.add_argument("--status", default="open")
    p.add_argument("--limit", type=int, default=50)
    p.set_defaults(func=cmd_pending)

    sub.add_parser("stats", help="print storage statistics").set_defaults(func=cmd_stats)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
