"""UI-free background predictions for the GUI.

A ChargE3Net prediction takes seconds on a GPU and minutes on a CPU, longer than a
Streamlit script run should block: any click would restart the run and lose the work.
So each prediction runs in a thread of its own, registered under a key (file digest and
device), and the page only polls its progress.
"""

from __future__ import annotations

import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Optional

from pydemi import ml
from pydemi.io.base import Structure, VolumetricData


@dataclass
class Job:
    structure: Structure
    device: Optional[str] = None
    shape: Optional[tuple[int, int, int]] = None      # None: the grid VASP would use
    done: int = 0
    total: int = 0
    started: float = field(default_factory=time.time)
    finished: Optional[float] = None
    first_chunk: Optional[tuple[float, int]] = None   # (time, points) after the first chunk
    result: Optional[VolumetricData] = None
    error: Optional[str] = None
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: Optional[threading.Thread] = None

    @property
    def state(self) -> str:
        if self.result is not None:
            return "done"
        if self.error is not None:
            return "error"
        if self._stop.is_set() and self.finished is not None:
            return "cancelled"
        return "running"

    def start(self) -> "Job":
        self._thread = threading.Thread(target=self._work, name="pydemi-predict", daemon=True)
        self._thread.start()
        return self

    def cancel(self) -> None:
        self._stop.set()

    def wait(self, timeout: Optional[float] = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def _progress(self, done: int, total: int) -> None:
        self.done, self.total = done, total
        if self.first_chunk is None:
            self.first_chunk = (time.time(), done)

    def _work(self) -> None:
        try:
            if self.shape:
                self.result = ml.predict(self.structure, shape=self.shape, device=self.device,
                                         progress=self._progress, should_stop=self._stop.is_set)
            else:
                self.result = ml.predict(self.structure, device=self.device, progress=self._progress,
                                         should_stop=self._stop.is_set)
        except ml.Cancelled:
            pass
        except Exception as exc:  # shown to the user; the traceback goes to the terminal
            traceback.print_exc()
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            self.finished = time.time()

    def elapsed(self) -> float:
        return (self.finished or time.time()) - self.started

    def seconds_left(self) -> Optional[float]:
        """From the rate since the first chunk (which includes model loading and atoms)."""
        if self.first_chunk is None or self.done <= self.first_chunk[1]:
            return None
        t0, n0 = self.first_chunk
        rate = (self.done - n0) / max(time.time() - t0, 1e-9)
        return (self.total - self.done) / rate


class Jobs:
    """Predictions by key; at most one runs at a time (the network is shared)."""

    KEEP = 4  # finished predictions kept in memory

    def __init__(self) -> None:
        self._jobs: dict[Any, Job] = {}
        self._lock = threading.Lock()

    def get(self, key: Any) -> Optional[Job]:
        return self._jobs.get(key)

    def running(self) -> Optional[Any]:
        return next((k for k, j in self._jobs.items() if j.state == "running"), None)

    def submit(self, key: Any, structure: Structure, device: Optional[str] = None,
               shape: Optional[tuple[int, int, int]] = None) -> Job:
        """Start a prediction for ``key``, or return the one already running or done. The
        grid is ``shape``, or the one VASP would use."""
        with self._lock:
            job = self._jobs.get(key)
            if job is not None and job.state in ("running", "done"):
                return job
            busy = self.running()
            if busy is not None:
                raise RuntimeError("another prediction is running; wait for it or cancel it")
            finished = [k for k, j in self._jobs.items() if j.state != "running"]
            for k in finished[:max(0, len(finished) - self.KEEP + 1)]:
                del self._jobs[k]
            n1, n2, n3 = shape or ml.vasp_grid_shape(structure.lattice)
            job = self._jobs[key] = Job(structure, device, shape, total=n1 * n2 * n3).start()
            return job

    def forget(self, key: Any) -> None:
        with self._lock:
            job = self._jobs.pop(key, None)
        if job is not None:
            job.cancel()
