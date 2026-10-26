"""``pydemi gui``: the command it runs, and a clear message when the extra is missing."""

import io
import os
import signal
import subprocess
import sys
import time

import pytest

from pydemi import cli
from pydemi import gui


def test_command_runs_the_packaged_app_locally():
    cmd = gui.command()
    assert cmd[:4] == [sys.executable, "-m", "streamlit", "run"]
    assert cmd[4] == str(gui.APP) and gui.APP.exists()
    assert "--server.address=localhost" in cmd
    assert not any(c.startswith("--server.port") for c in cmd)     # Streamlit picks a free port
    assert "--server.port=9000" in gui.command(port=9000)
    assert "--server.headless=true" in cmd and "--browser.gatherUsageStats=false" in cmd


def test_reference_pdf_ships_with_the_app():
    assert gui.APP.with_name("descriptors.pdf").stat().st_size > 10_000


def test_missing_extra_is_reported(monkeypatch, capsys):
    monkeypatch.setattr(gui, "missing", lambda: ["streamlit"])
    assert cli.main(["gui", "--no-browser"]) == 1
    err = capsys.readouterr().err
    assert "streamlit" in err and 'pip install "pydemi[gui]"' in err


def test_relay_opens_the_first_url(monkeypatch, capsys):
    opened = []
    monkeypatch.setattr(gui.webbrowser, "open", opened.append)
    gui._relay(io.StringIO("starting\n  Local URL: http://localhost:8502\n"
                           "  Network URL: http://10.0.0.1:8502\n"), browser=True)
    assert opened == ["http://localhost:8502"]
    assert "pydemi GUI: http://localhost:8502" in capsys.readouterr().out
    gui._relay(io.StringIO("  URL: http://0.0.0.0:8501\n"), browser=False)
    assert opened == ["http://localhost:8502"]                   # not opened again
    assert "pydemi GUI: http://localhost:8501" in capsys.readouterr().out


FAKE_SERVER = """
import os, signal, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)     # a server that ignores polite requests
signal.signal(signal.SIGINT, signal.SIG_IGN)
print("pid", os.getpid(), flush=True)
print("  URL: http://localhost:1", flush=True)
time.sleep(120)
"""

LAUNCHER = """
import sys
from pydemi import gui
gui.missing = lambda: []
gui.command = lambda host, port: [sys.executable, "-c", {fake!r}]
gui._stop.__defaults__ = (0.5,)                   # short grace periods for the test
raise SystemExit(gui.launch(browser=False))
"""


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
@pytest.mark.parametrize("sig", ["SIGTERM", "SIGHUP", "SIGINT"])
def test_server_never_outlives_the_launcher(sig):
    proc = subprocess.Popen([sys.executable, "-c", LAUNCHER.format(fake=FAKE_SERVER)],
                            stdout=subprocess.PIPE, text=True)
    server = int(proc.stdout.readline().split()[1])
    proc.stdout.readline()                                     # the URL: launcher is waiting
    os.kill(proc.pid, getattr(signal, sig))
    assert proc.wait(15) == 0
    deadline = time.time() + 5
    while time.time() < deadline:
        try:
            os.kill(server, 0)
        except ProcessLookupError:
            return
        time.sleep(0.1)
    os.kill(server, signal.SIGKILL)
    pytest.fail(f"server {server} outlived the launcher after {sig}")
