"""Background analysis jobs: one inference at a time, with progress and cancel.

The model saturates the CPU and takes 1.5-4 GB per run, so a single worker
thread runs jobs in order and a short queue refuses the rest (QueueFull ->
HTTP 503). Results, uploads included, stay in memory only and expire after
JOB_TTL_S; at most MAX_JOBS are kept.
"""
import logging
import os
import threading
import time
import traceback
import uuid
from collections import OrderedDict, deque

from . import aoi, imagery, infer, osm_data, uploads

log = logging.getLogger(__name__)

MAX_QUEUE = int(os.environ.get("URBANSEG_MAX_QUEUE", "4"))     # waiting, not running
MAX_JOBS = int(os.environ.get("URBANSEG_MAX_JOBS", "12"))      # kept, incl. finished
JOB_TTL_S = float(os.environ.get("URBANSEG_JOB_TTL_S", "3600"))
STATES = ("queued", "running", "done", "failed", "cancelled")


class QueueFull(RuntimeError):
    """Too many jobs are waiting; try again later."""


class Job:
    """One analysis. fn(job) -> (summary, files) runs in the worker thread and
    reports through job.progress(); job.cancelled() is its cancel poll."""

    def __init__(self, kind, title, fn):
        self.id = uuid.uuid4().hex
        self.kind, self.title, self.fn = kind, title, fn
        self.state, self.stage, self.label, self.fraction = "queued", None, None, 0.0
        self.created, self.started, self.finished = time.time(), None, None
        self.error = None       # {"code", "message", "status"}
        self.summary = None     # JSON-ready result
        self.files = {}         # name -> pipeline.File
        self._cancel = threading.Event()
        self.done = threading.Event()

    def progress(self, stage=None, label=None, fraction=None, **_):
        self.stage, self.label = stage, label
        if fraction is not None:
            self.fraction = max(self.fraction, float(fraction))

    def cancelled(self):
        return self._cancel.is_set()

    def public(self, with_result=True):
        out = {"id": self.id, "kind": self.kind, "title": self.title, "state": self.state,
               "stage": self.stage, "label": self.label, "progress": round(self.fraction, 3),
               "created": self.created, "started": self.started, "finished": self.finished,
               "error": self.error}
        if with_result and self.state == "done":
            out["result"] = self.summary
        return out


def describe_error(ex):
    """(status, code, message) for the API; never echoes URLs, keys or internals."""
    if isinstance(ex, infer.Cancelled) or (isinstance(ex, imagery.TileFetchError)
                                           and ex.reason == "cancelled"):
        return 499, "cancelled", "cancelled"
    if isinstance(ex, aoi.AOIError):
        return 422, ex.code, str(ex)
    if isinstance(ex, uploads.UploadError):
        status = {"too_large": 413, "too_small": 422, "unreadable": 400}.get(ex.code, 415)
        return (422 if ex.param else status), ex.code, str(ex)
    if isinstance(ex, imagery.TileFetchError):
        msg = {"auth": "Google rejected the API key: check that the Maps Static API is "
                       "enabled for it and that billing is set up",
               "quota": "the Google Maps quota for this key is used up",
               "network": "Google Maps could not be reached: check the internet connection",
               "server": "Google Maps had a server error: try again in a minute",
               "bad_image": "Google Maps returned an unexpected image: try again",
               "bad_request": "Google Maps refused the request"}.get(ex.reason, "tile fetch failed")
        return 502, f"imagery_{ex.reason}", msg
    if isinstance(ex, osm_data.OverpassError):
        return 502, "osm_unavailable", "OpenStreetMap could not be reached"
    if isinstance(ex, MemoryError):
        return 507, "out_of_memory", ("the server ran out of memory: try a smaller area or "
                                      "give Docker more memory")
    return 500, "internal", "the analysis failed unexpectedly (see the server log)"


class JobManager:
    def __init__(self):
        self._lock = threading.Condition()
        self._jobs = OrderedDict()
        self._queue = deque()
        self._running = None
        self._thread = None

    def submit(self, kind, title, fn):
        with self._lock:
            self._prune()
            if len(self._queue) >= MAX_QUEUE:
                raise QueueFull(f"{len(self._queue)} analyses are already waiting")
            job = Job(kind, title, fn)
            self._jobs[job.id] = job
            self._queue.append(job)
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._work, name="urbanseg-jobs",
                                                daemon=True)
                self._thread.start()
            self._lock.notify()
            return job

    def get(self, job_id):
        with self._lock:
            self._prune()
            return self._jobs.get(job_id)

    def cancel(self, job_id):
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            job._cancel.set()
            if job.state == "queued":
                self._queue.remove(job)
                self._finish(job, "cancelled")
            return job

    def status(self):
        with self._lock:
            return {"queued": len(self._queue), "running": self._running is not None,
                    "max_queue": MAX_QUEUE}

    def _prune(self):
        now = time.time()
        finished = [j for j in self._jobs.values()
                    if j.state in ("done", "failed", "cancelled")]
        for j in finished:
            if now - (j.finished or now) > JOB_TTL_S:
                del self._jobs[j.id]
        finished = [j for j in self._jobs.values()
                    if j.state in ("done", "failed", "cancelled")]
        while len(self._jobs) > MAX_JOBS and finished:
            del self._jobs[finished.pop(0).id]

    def _finish(self, job, state):
        job.state, job.finished = state, time.time()
        job.done.set()

    def _work(self):
        while True:
            with self._lock:
                while not self._queue:
                    self._lock.wait()
                job = self._queue.popleft()
                job.state, job.started, self._running = "running", time.time(), job
            try:
                job.summary, job.files = job.fn(job)
                state = "done"
            except Exception as ex:  # noqa: BLE001 - a job must never kill the worker
                status, code, message = describe_error(ex)
                state = "cancelled" if code == "cancelled" else "failed"
                job.error = {"status": status, "code": code, "message": message}
                if status == 500 or status == 507:
                    # the traceback may hold a URL with the key: redact it
                    log.error("job %s failed:\n%s", job.id,
                              imagery.redact("".join(traceback.format_exception(ex))))
            with self._lock:
                self._running = None
                self._finish(job, state)
