"""
pydemi.gui
==========
A graphical interface that runs on the user's own machine and opens in the browser:

    pip install "pydemi[gui]"
    pydemi gui

Upload a CHGCAR, then view its structure and density slices, compute the
descriptors, or browse a plot for each. Nothing leaves the machine: the server
listens on ``localhost`` unless ``--host`` says otherwise.

``core`` (upload handling and tables) needs only the core dependencies;
``plots``, ``viewer`` and ``app`` need the ``gui`` extra (Streamlit, plotly).
"""

from __future__ import annotations

import importlib.util
import re
import signal
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path
from typing import IO, Any, Optional

APP = Path(__file__).with_name("app.py")
REQUIRES = ("streamlit", "plotly")
SETTINGS = {
    "server.headless": "true",              # no first-run prompt; the browser is opened below
    "server.maxUploadSize": "1024",         # MB; the largest dataset CHGCAR is ~350 MB
    "browser.gatherUsageStats": "false",
    "client.toolbarMode": "viewer",         # hide Streamlit's developer menu
    "theme.primaryColor": "#2F6DB5",
}
INSTALL_HINT = 'pip install "pydemi[gui]"'
_URL = re.compile(r"(https?://\S+)")


def app_views() -> tuple[str, ...]:
    """Labels of the view tabs of the app (for tests and scripts that drive it)."""
    return (":material/view_in_ar: Structure", ":material/table_chart: Descriptors",
            ":material/insights: Plots", ":material/science: Properties",
            ":material/compare_arrows: Compare")


def missing() -> list[str]:
    """The GUI's packages that are not installed."""
    return [m for m in REQUIRES if importlib.util.find_spec(m) is None]


def command(host: str = "localhost", port: Optional[int] = None) -> list[str]:
    """The ``streamlit run`` command line for the app."""
    cmd = [sys.executable, "-m", "streamlit", "run", str(APP), f"--server.address={host}"]
    if port is not None:   # without a port Streamlit takes 8501, or the next free one
        cmd.append(f"--server.port={port}")
    return cmd + [f"--{k}={v}" for k, v in SETTINGS.items()]


def _relay(stream: IO[str], browser: bool) -> None:
    """Echo the server's output; at the first URL it prints, say where the GUI is and
    (if ``browser``) open it."""
    found = False
    for line in stream:
        sys.stdout.write(line)
        sys.stdout.flush()
        m = _URL.search(line)
        if m and not found:
            found = True
            url = m.group(1).replace("0.0.0.0", "localhost")
            print(f"\npydemi GUI: {url}  (Ctrl+C to stop)\n", flush=True)
            if browser:
                webbrowser.open(url)


def _stop(proc: "subprocess.Popen[str]", grace: float = 5.0) -> None:
    """Stop the server: interrupt (Ctrl+C), then terminate, then kill, ``grace`` s apart."""
    steps = [proc.terminate, proc.kill]
    if sys.platform != "win32":       # on Windows terminate() already ends the process
        steps.insert(0, lambda: proc.send_signal(signal.SIGINT))
    for step in steps:
        if proc.poll() is not None:
            return
        step()
        try:
            proc.wait(grace)
        except subprocess.TimeoutExpired:
            continue


def launch(host: str = "localhost", port: Optional[int] = None, browser: bool = True) -> int:
    """Run the GUI until Ctrl+C (or SIGTERM / SIGHUP); returns the server's exit code.

    The server always stops with the launcher, so no orphaned process keeps the port.
    """
    if missing():
        print(f"pydemi gui needs {', '.join(missing())}. Install them with:\n    {INSTALL_HINT}",
              file=sys.stderr)
        return 1
    proc = subprocess.Popen(command(host, port), stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert proc.stdout is not None
    threading.Thread(target=_relay, args=(proc.stdout, browser), daemon=True).start()

    def interrupt(signum: int, frame: Any) -> None:
        raise KeyboardInterrupt

    previous = {}
    if threading.current_thread() is threading.main_thread():
        for name in ("SIGTERM", "SIGHUP"):
            sig = getattr(signal, name, None)
            if sig is not None:
                previous[sig] = signal.signal(sig, interrupt)
    try:
        return proc.wait()
    except KeyboardInterrupt:
        return 0
    finally:
        _stop(proc)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
