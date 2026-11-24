"""Queued per-session timing events and an offline JSON summary.

Run python timing_logger.py artifacts/timing/<session>.jsonl to summarize a run.
"""
import argparse
from datetime import datetime, timezone
import json
from importlib.metadata import PackageNotFoundError, version
import logging
import math
import os
import platform
from pathlib import Path
import queue
import statistics
import threading
import time
import uuid

log = logging.getLogger('sir2')


def runtime_environment():
    packages = {}
    for name in ['torch', 'torchvision', 'timm', 'open_clip_torch', 'ultralytics', 'pillow', 'numpy', 'fastapi', 'uvicorn']:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return dict(python=platform.python_version(), platform=platform.platform(), packages=packages,
                timing_clock='perf_counter', clock_resolution_ms=time.get_clock_info('perf_counter').resolution*1000)


class TimingLogger:
    def __init__(self, directory):
        self.session_id = uuid.uuid4().hex
        self.started = time.perf_counter()
        self.path = None
        self.error = None
        self.dropped_events = 0
        self.closed = False
        self.queue = queue.Queue(maxsize=4096)
        self.thread = None
        if os.environ.get('SIR2_TIMING_LOG', '1') == '0':
            return
        try:
            directory = Path(directory)
            directory.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
            self.path = directory / f'{stamp}_{self.session_id}.jsonl'
            self.stream = self.path.open('x', encoding='utf-8')
        except OSError as exc:
            self.error = str(exc)
            log.exception('Timing log could not be opened')
            return
        self.thread = threading.Thread(target=self._write, name='sir2-timing', daemon=True)
        self.thread.start()
        log.warning('Timing log: %s', self.path)

    def info(self):
        return dict(enabled=self.thread is not None and self.error is None and not self.closed,
                    session_id=self.session_id, path=str(self.path) if self.path else None,
                    error=self.error, dropped_events=self.dropped_events)

    def record(self, event, **fields):
        if not self.info()['enabled']:
            return
        row = dict(event=event, session_id=self.session_id,
                   timestamp_utc=datetime.now(timezone.utc).isoformat(),
                   elapsed_ms=(time.perf_counter() - self.started) * 1000, **fields)
        try:
            self.queue.put_nowait(row)
        except queue.Full:
            self.dropped_events += 1
            if self.dropped_events == 1:
                log.warning('Timing queue full; lost events are counted in timing_log.dropped_events')

    def _write(self):
        try:
            with self.stream:
                while True:
                    row = self.queue.get()
                    if row is None:
                        break
                    self.stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                    self.stream.flush()
        except (OSError, TypeError, ValueError) as exc:
            self.error = str(exc)
            log.exception('Timing logging stopped; application processing continues')

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.thread is not None:
            end = dict(event='session_end', session_id=self.session_id,
                       timestamp_utc=datetime.now(timezone.utc).isoformat(),
                       elapsed_ms=(time.perf_counter()-self.started)*1000,
                       dropped_events=self.dropped_events)
            while self.thread.is_alive():
                try:
                    self.queue.put(end, timeout=.1)
                    break
                except queue.Full:
                    continue
            # A failed writer must never leave shutdown waiting on a full queue.
            while self.thread.is_alive():
                try:
                    self.queue.put(None, timeout=.1)
                    break
                except queue.Full:
                    continue
            self.thread.join()


def distribution(values):
    ordered = sorted(values)
    position = (len(ordered) - 1) * .95
    lower, upper = math.floor(position), math.ceil(position)
    return dict(count=len(ordered), mean_ms=statistics.mean(ordered),
                median_ms=statistics.median(ordered),
                p95_ms=ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower),
                min_ms=ordered[0], max_ms=ordered[-1])


def summarize(path):
    groups = {}
    counts = {}
    completions = []
    dropped_events = 0
    incomplete_lines = 0
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                incomplete_lines += 1
                continue
            event = row['event']
            counts[event] = counts.get(event, 0) + 1
            dropped_events = max(dropped_events, row.get('dropped_events', 0))
            if event == 'frame_processed' and not row.get('reprocessed', False):
                completions.append(row['elapsed_ms'])
            if event not in ('frame_processed', 'client_timing', 'frame_received', 'preview_requested'):
                continue
            # Client upload and viewer measurements stay distinct from server stages.
            key = '/'.join([event, row.get('kind', ''), row.get('source', 'unknown'), row.get('status', '')])
            metrics = groups.setdefault(key, {})
            values = dict(row.get('timings', {}))
            values.update({key: value for key, value in row.items() if key.endswith('_ms') and key != 'elapsed_ms'})
            for name, value in values.items():
                if isinstance(value, (float, int)) and math.isfinite(value):
                    metrics.setdefault(name, []).append(value)
    span = max(completions) - min(completions) if len(completions) > 1 else 0
    return dict(events=counts, dropped_log_events=dropped_events, incomplete_lines=incomplete_lines,
                processed_session_fps=(len(completions) - 1) * 1000 / span if span > 0 else None,
                fps_scope='Completed live frames between first and last completion, including pauses; benchmarks excluded.',
                timings={key: {name: distribution(values) for name, values in metrics.items()}
                         for key, metrics in groups.items()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    text = json.dumps(summarize(args.log), ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(text + '\n', encoding='utf-8')
    else:
        print(text)
