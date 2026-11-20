#!/usr/bin/env python3
"""Create isolated runtime and installation-tool environments for THz-TDS."""

from pathlib import Path
import os
import subprocess
import sys
import venv


def main():
    if sys.version_info[:2] != (3, 12):
        raise SystemExit(
            "Python 3.12 is required. On Windows run: py -3.12 install.py. "
            "On macOS or Linux run: python3.12 install.py."
        )
    try:
        import tkinter
    except ImportError as error:
        raise SystemExit(
            "Tkinter is required. Install Python 3.12 with Tcl/Tk support "
            "or your Linux distribution's python3.12-tk package, then retry."
        ) from error

    root = Path(__file__).resolve().parent
    executable = Path("Scripts/python.exe") if os.name == "nt" else Path("bin/python")
    runtime = root / ".venv" / executable
    tools = root / ".venv-tools" / executable
    for directory in (root / ".venv", root / ".venv-tools"):
        environment_python = directory / executable
        if not environment_python.exists():
            venv.EnvBuilder(with_pip=True).create(directory)
        version = subprocess.check_output(
            [str(environment_python), "-c", "import sys; print('%d.%d' % sys.version_info[:2])"],
            text=True,
        ).strip()
        if version != "3.12":
            raise SystemExit(
                f"{directory.name} uses Python {version}. Rename that environment "
                "and rerun the installer with Python 3.12."
            )

    environment = os.environ.copy()
    environment.pop("VIRTUAL_ENV", None)
    environment["POETRY_KEYRING_ENABLED"] = "false"
    commands = (
        [str(tools), "-m", "pip", "install", "poetry==2.2.1"],
        [str(tools), "-m", "poetry", "env", "use", str(runtime)],
        [str(tools), "-m", "poetry", "install", "--only", "main", "--no-root"],
        [str(runtime), "-m", "pip", "check"],
        [str(runtime), "-c", "import tkinter, numpy, scipy, pandas, openpyxl, typer; print('Runtime imports passed.')"],
    )
    for command in commands:
        subprocess.run(command, cwd=root, env=environment, check=True)
    print("Installation complete. No environment activation is required.")
    print("Run the commands in README.md to select a dataset and start processing.")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        raise SystemExit(
            f"Installation stopped with exit code {error.returncode}. "
            "Check the error above, then rerun the same installation command."
        ) from error
