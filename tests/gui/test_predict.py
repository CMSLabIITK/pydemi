"""Background predictions of the GUI (pydemi.gui.predict), with a stand-in for the model."""

import threading
import time

import numpy as np
import pytest

from pydemi import ml
from pydemi.gui import predict
from pydemi.io.base import Grid, Lattice, Structure, VolumetricData

S = Structure(Lattice(np.eye(3) * 3.0), ["Si"], [[0, 0, 0]])


@pytest.fixture
def fake(monkeypatch):
    """ml.predict that reports progress in 4 chunks, each waiting for ``gate``."""
    gate = threading.Event()

    def run(structure, device=None, progress=None, should_stop=None):
        for i in range(1, 5):
            gate.wait(5)
            if should_stop():
                raise ml.Cancelled()
            progress(i * 25, 100)
        if structure.species == ["Og"]:
            raise ValueError("boom")
        return VolumetricData(structure, Grid(np.ones((2, 2, 2)), structure.lattice), sources={})

    monkeypatch.setattr(ml, "predict", run)
    return gate


def test_job_runs_to_completion(fake):
    jobs = predict.Jobs()
    job = jobs.submit("a", S)
    assert job.state == "running" and job.total == np.prod(ml.vasp_grid_shape(S.lattice))
    assert jobs.submit("a", S) is job                     # same key: the running job
    with pytest.raises(RuntimeError, match="another prediction"):
        jobs.submit("b", S)
    fake.set()
    job.wait(5)
    assert job.state == "done" and job.result is not None and job.done == 100
    assert job.seconds_left() is not None or job.done == job.total
    assert jobs.submit("a", S) is job                     # done: kept, not redone


def test_cancel_and_failure(fake):
    jobs = predict.Jobs()
    job = jobs.submit("a", S)
    job.cancel()
    fake.set()
    job.wait(5)
    assert job.state == "cancelled" and job.result is None
    again = jobs.submit("a", S)                           # a cancelled job can be restarted
    assert again is not job
    again.wait(5)
    bad = jobs.submit("og", Structure(S.lattice, ["Og"], [[0, 0, 0]]))
    bad.wait(5)
    assert bad.state == "error" and "ValueError: boom" in bad.error


def test_only_a_few_finished_jobs_are_kept(fake):
    fake.set()
    jobs = predict.Jobs()
    for k in range(8):
        jobs.submit(k, S).wait(5)
    assert len(jobs._jobs) == predict.Jobs.KEEP
    assert jobs.get(7) is not None and jobs.get(0) is None
    jobs.forget(7)
    assert jobs.get(7) is None
