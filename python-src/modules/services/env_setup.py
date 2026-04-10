from __future__ import annotations

import os
import sys
from contextlib import suppress


def setup_environment() -> None:
    """Prepare Tkinter environment variables for macOS builds."""
    if sys.platform != "darwin":  # Not macOS, return directly
        return

    # Check if in packaged environment
    if not (getattr(sys, "frozen", False) or "MTGA_GUI" in sys.executable):
        return  # Development environment doesn't need special handling

    # Nuitka packaged environment
    executable_dir = os.path.dirname(sys.executable)

    # Switch working directory - this is critical
    # When launched from Finder on macOS, working directory is "/", must switch
    if os.getcwd() == "/":
        # Prefer switching to user home directory (safer)
        home_dir = os.path.expanduser("~")
        try:
            os.chdir(home_dir)
        except OSError:
            with suppress(OSError):
                os.chdir(executable_dir)

    # Set TCL/TK library paths (if they exist)
    tcl_library = os.path.join(executable_dir, "tcl-files")
    tk_library = os.path.join(executable_dir, "tk-files")

    if os.path.exists(tcl_library):
        os.environ["TCL_LIBRARY"] = tcl_library

    if os.path.exists(tk_library):
        os.environ["TK_LIBRARY"] = tk_library
