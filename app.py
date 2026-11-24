import asyncio
import contextlib
import logging
import os
import secrets
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from species_pipeline import decode_frame
from timing_logger import TimingLogger, runtime_environment

ROOT = Path(__file__).resolve().parent
log = logging.getLogger("sir2")


class Settings(BaseModel):
    segmentation: Literal["clipseg", "sam3", "grabcut", "yolo"] = "yolo"
    clipseg_threshold: float = Field(default=0.5, ge=0.1, le=0.95)
    sam3_threshold: float = Field(default=0.5, ge=0.1, le=0.95)
    yolo_threshold: float = Field(default=0.5, ge=0.1, le=0.95)
    roi: float = Field(default=0.78, ge=0.45, le=0.9)
    masked: bool = False


Duration = Annotated[float, Field(ge=0, le=3_600_000, allow_inf_nan=False)]


class ClientTiming(BaseModel):
    kind: Literal['upload', 'display']
    sequence: int = Field(ge=1)
    source: Literal['phone', 'desktop', 'photo', 'viewer']
    result_version: int | None = Field(default=None, ge=1)
    capture_ms: Duration | None = None
    jpeg_encode_ms: Duration | None = None
    upload_round_trip_ms: Duration | None = None
    capture_to_ack_ms: Duration | None = None
    state_poll_ms: Duration | None = None
    overlay_download_ms: Duration | None = None
    display_update_ms: Duration | None = None


def create_app(pipeline_factory=None, timing_dir=None):
    inference_lock = asyncio.Lock()
    completions = deque(maxlen=90)
    state = {
        "latest": None,
        "sequence": 0,
        "processed_sequence": 0,
        "received_at": None,
        "result_at": None,
        "result_frame_at": None,
        "result": None,
        "result_version": 0,
        "error": None,
        "settings": Settings().model_dump(),
        "settings_version": 0,
        "processed_settings_version": -1,
        "token": secrets.token_urlsafe(24),
        "sender": None,
        "sender_at": 0,
        "busy": False,
        "benchmarking": False,
        "frame_context": None,
        "active_sequence": None,
    }

    async def worker():
        while True:
            if state["latest"] is None or (
                state["sequence"] == state["processed_sequence"]
                and state["settings_version"] == state["processed_settings_version"]
            ):
                await asyncio.sleep(0.03)
                continue
            # A single in-flight inference; capture replaces latest instead of queuing.
            data, sequence = state["latest"], state["sequence"]
            received_at = state["received_at"]
            context = state['frame_context'].copy()
            reprocessed = state['processed_sequence'] == sequence
            settings, version = state["settings"].copy(), state["settings_version"]
            state['active_sequence'] = sequence
            queued_at = time.perf_counter()
            try:
                async with inference_lock:
                    state['busy'] = True
                    processing_started = time.perf_counter()
                    result = await asyncio.to_thread(app.state.pipeline.process, data, settings)
                    finished = time.perf_counter()
                state["result"] = result
                state["result_at"] = time.perf_counter()
                state["result_frame_at"] = received_at
                state["error"] = None
                completions.append(time.perf_counter())
                app.state.timing.record('frame_processed', **context,
                    reprocessed=reprocessed,
                    settings_version=version, result_version=state['result_version'] + 1,
                    status=result.get('status'), accepted=result.get('accepted'),
                    species=result.get('species'), condition_class=result.get('condition_class'),
                    input_size=result.get('input_size'), roi_size=result.get('roi_size'),
                    leaf_count=result.get('leaf_count'), timings=result.get('timings', {}),
                    processing_ms=result.get('processing_ms'),
                    worker_wait_ms=(queued_at - received_at) * 1000,
                    lock_wait_ms=(processing_started - queued_at) * 1000,
                    inference_wall_ms=(finished - processing_started) * 1000,
                    server_frame_latency_ms=(finished - context['request_started']) * 1000,
                    processed_fps=((len(completions)-1)/(completions[-1]-completions[0])
                                   if len(completions)>1 and completions[-1]>completions[0] else None))
            except asyncio.CancelledError:
                app.state.timing.record('frame_cancelled', **context,
                    settings_version=version, reason='server_shutdown')
                raise
            except Exception:
                log.exception("Frame processing failed")
                app.state.timing.record('frame_error', **context,
                    settings_version=version, result_version=state['result_version'] + 1,
                    server_frame_latency_ms=(time.perf_counter()-context['request_started'])*1000)
                state["result"] = None
                state['result_at'] = time.perf_counter()
                state['result_frame_at'] = received_at
                state["error"] = "Obrada nije uspela. Detalji su u terminalu servera."
            finally:
                state['busy'] = False
                state['active_sequence'] = None
            state["processed_sequence"] = sequence
            state["processed_settings_version"] = version
            state["result_version"] += 1

    @asynccontextmanager
    async def lifespan(app):
        app.state.timing = TimingLogger(timing_dir or os.environ.get('SIR2_TIMING_DIR', ROOT/'artifacts/timing'))
        factory = pipeline_factory
        if factory is None:
            from species_pipeline import SpeciesPipeline
            factory = SpeciesPipeline
        task = None
        started = time.perf_counter()
        try:
            app.state.pipeline = await asyncio.to_thread(factory)
            initialized = time.perf_counter()
            app.state.timing.record('session_start', model=app.state.pipeline.manifest, runtime=runtime_environment(),
                                   initialization_ms=(initialized-started)*1000)
            task = asyncio.create_task(worker())
            yield
        except Exception:
            app.state.timing.record('session_error', initialization_ms=(time.perf_counter()-started)*1000)
            raise
        finally:
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            await asyncio.to_thread(app.state.timing.close)

    app = FastAPI(title="SIR2 — Leaf Lab", lifespan=lifespan)
    app.add_middleware(
        TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "testserver"]
    )
    app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")

    def authorize(request):
        if not secrets.compare_digest(
            request.headers.get("x-sir2-token", ""), state["token"]
        ):
            raise HTTPException(403, "Osvezite stranicu: sesija je istekla.")
        origin = request.headers.get("origin")
        if origin and origin != str(request.base_url).rstrip("/"):
            raise HTTPException(403, "Nedozvoljeno poreklo zahteva.")

    @app.get("/")
    async def index():
        return FileResponse(ROOT / "static/species.html")

    @app.get("/phone")
    async def phone():
        return FileResponse(ROOT / "static/phone.html")

    @app.get("/api/state")
    async def status():
        now = time.perf_counter()
        age = None if state["received_at"] is None else now - state["received_at"]
        result = state["result"] or {}
        recent = [t for t in completions if now-t < 3]
        span = recent[-1]-recent[0] if len(recent)>1 else 0
        return {
            "connected": age is not None and age < 3,
            "busy": state['busy'],
            "benchmarking": state['benchmarking'],
            "processed_fps": ((len(recent)-1)/span if span>0 else None),
            "frame_age_ms": None if age is None else age * 1000,
            "sequence": state["sequence"],
            "processed_sequence": state["processed_sequence"],
            "result_version": state["result_version"],
            "result_age_ms": (
                None
                if state["result_frame_at"] is None
                else (now - state["result_frame_at"]) * 1000
            ),
            "server_latency_ms": (
                None
                if state["result_at"] is None
                else (state["result_at"] - state["result_frame_at"]) * 1000
            ),
            "result": {
                k: v for k, v in result.items() if k not in ("overlay", "mask", "input")
            },
            "error": state["error"],
            "settings": state["settings"],
            "token": state["token"],
            "timing_log": app.state.timing.info(),
            "sam3": (
                app.state.pipeline.sam3.status()
                if hasattr(app.state.pipeline, "sam3")
                else None
            ),
            "clipseg": (
                app.state.pipeline.clipseg.status()
                if hasattr(app.state.pipeline, "clipseg")
                else None
            ),
            "yolo": (
                dict(loaded=True, **app.state.pipeline.segmenter.metadata)
                if hasattr(app.state.pipeline, "segmenter")
                else None
            ),
            "model": {
                k: v for k, v in app.state.pipeline.manifest.items() if k != "classes"
            },
        }

    @app.post("/api/settings")
    async def settings(value: Settings, request: Request):
        authorize(request)
        state["settings"] = value.model_dump()
        state["settings_version"] += 1
        return state["settings"]

    class BenchmarkRequest(BaseModel):
        repeats: int = Field(default=10, ge=3, le=30)

    @app.post('/api/timing/client')
    async def client_timing(value: ClientTiming, request: Request):
        authorize(request)
        if value.sequence > state['sequence']:
            raise HTTPException(400, 'Nepoznat kadar.')
        if value.result_version is not None and value.result_version > state['result_version']:
            raise HTTPException(400, 'Nepoznat rezultat.')
        app.state.timing.record('client_timing', **value.model_dump(exclude_none=True))
        return {'logged': app.state.timing.info()['enabled']}

    @app.post('/api/benchmark')
    async def benchmark(value: BenchmarkRequest, request: Request):
        authorize(request)
        if state['latest'] is None:
            raise HTTPException(400, 'Prvo učitaj fotografiju ili pošalji kadar kamere.')
        if state['busy'] or state['benchmarking']:
            raise HTTPException(409, 'Obrada je u toku. Sačekaj rezultat pa ponovi merenje.')
        state['benchmarking'] = True
        sequence, data = state['sequence'], state['latest']
        started = time.perf_counter()
        try:
            async with inference_lock:
                result = await asyncio.to_thread(app.state.pipeline.benchmark, data, value.repeats)
            app.state.timing.record('benchmark', sequence=sequence, **result,
                                   benchmark_request_ms=(time.perf_counter()-started)*1000)
            return dict(**result, sequence=sequence)
        except ValueError as exc:
            app.state.timing.record('benchmark_error', sequence=sequence,
                                   benchmark_request_ms=(time.perf_counter()-started)*1000)
            raise HTTPException(400, str(exc)) from exc
        except Exception:
            app.state.timing.record('benchmark_error', sequence=sequence,
                                   benchmark_request_ms=(time.perf_counter()-started)*1000)
            raise
        finally:
            state['benchmarking'] = False

    @app.post("/api/frame")
    async def frame(request: Request):
        request_started = time.perf_counter()
        authorize(request)
        sender = request.headers.get("x-sir2-sender", "")
        if not sender or len(sender) > 100:
            raise HTTPException(400, "Nedostaje identifikator kamere.")
        now = time.perf_counter()
        if state["sender"] not in (None, sender) and now - state["sender_at"] < 3:
            raise HTTPException(
                409, "Druga kamera vec salje video. Zaustavite je pre povezivanja nove."
            )
        chunks = bytearray()
        async for chunk in request.stream():
            chunks.extend(chunk)
            if len(chunks) > 1_500_000:
                raise HTTPException(413, "Kadar je prevelik.")
        data = bytes(chunks)
        body_received = time.perf_counter()
        try:
            await asyncio.to_thread(decode_frame, data)
        except ValueError as exc:
            app.state.timing.record('frame_invalid', bytes=len(data),
                body_read_ms=(body_received-request_started)*1000,
                validation_ms=(time.perf_counter()-body_received)*1000)
            raise HTTPException(400, str(exc)) from exc
        # Recheck ownership after decoding (another sender may have claimed it).
        now = time.perf_counter()
        if state["sender"] not in (None, sender) and now - state["sender_at"] < 3:
            raise HTTPException(409, "Druga kamera vec salje video.")
        state["sender"], state["sender_at"] = sender, now
        previous = state['frame_context']
        if previous and state['sequence'] not in (state['processed_sequence'], state['active_sequence']):
            app.state.timing.record('frame_dropped', **previous, reason='replaced_by_newer_frame')
        state["latest"], state["received_at"] = data, now
        state["sequence"] += 1
        source = request.headers.get('x-sir2-source', 'unknown')
        if source not in ('phone', 'desktop', 'photo'):
            source = 'unknown'
        state['frame_context'] = dict(sequence=state['sequence'], source=source, bytes=len(data),
            request_started=request_started, body_read_ms=(body_received-request_started)*1000,
            validation_ms=(now-body_received)*1000)
        app.state.timing.record('frame_received', **state['frame_context'],
                               receive_request_ms=(now-request_started)*1000)
        return {"sequence": state["sequence"], "settings": state["settings"]}

    @app.get("/api/image/{kind}")
    async def image(kind: str, after: int = -1, result_version: int = -1):
        headers = {"Cache-Control": "no-store"}
        if kind == "live":
            if after == state["sequence"]:
                return Response(status_code=204, headers=headers)
            data, media = state["latest"], "image/jpeg"
            headers["X-Frame-Sequence"] = str(state["sequence"])
        elif kind in ("overlay", "mask", "input"):
            if result_version >= 0 and result_version != state["result_version"]:
                return Response(status_code=204, headers=headers)
            headers["X-Result-Version"] = str(state["result_version"])
            headers["X-Frame-Sequence"] = str(state["processed_sequence"])
            data = (state["result"] or {}).get(kind)
            media = "image/png" if kind in ("mask", "input") else "image/jpeg"
        else:
            raise HTTPException(404)
        if data is None:
            return Response(status_code=204)
        if kind == 'overlay':
            app.state.timing.record('preview_requested', sequence=state['processed_sequence'],
                result_version=state['result_version'], bytes=len(data),
                result_ready_to_preview_request_ms=(time.perf_counter()-state['result_at'])*1000)
        return Response(data, media_type=media, headers=headers)

    return app


app = create_app()
