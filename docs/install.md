# Installing GhostGUI

Linux/Ubuntu is the primary tested platform. The macOS and Windows installers
are provided for convenience but remain experimental until verified on those
platforms.

## Requirements

- Python 3.10 or newer
- A working OpenGL environment
- Git
- Platform packages installed by the scripts below

GhostGUI uses PySide6 Essentials for Qt Core, GUI, Widgets, and OpenGL Widgets;
it does not require the much larger PySide6 Addons distribution. Each platform
installer creates or reuses `.venv`, verifies that it does not contain the
legacy full-Qt packages, upgrades the Python packaging tools, and installs the
checkout in editable mode without retaining downloaded wheels:

```bash
python -m pip install --no-cache-dir -e .
```

Editable installation keeps the registered model and mesh paths anchored to the
repository checkout.

## Linux / Ubuntu

```bash
git clone https://github.com/yx-lim/ghostgui.git
cd ghostgui
bash scripts/install_linux.sh
bash scripts/run_linux.sh
```

The installer uses `apt` to install Python, Qt/XCB runtime libraries, and
OpenGL/EGL libraries before creating the virtual environment.

## macOS

```bash
git clone https://github.com/yx-lim/ghostgui.git
cd ghostgui
bash scripts/install_macos.sh
bash scripts/run_macos.sh
```

The installer checks that Python, `.venv`, and `mjpython` use the Mac's native
architecture. It supports Apple Silicon (`arm64`) and Intel (`x86_64`); on Apple
Silicon it prefers Homebrew Python at `/opt/homebrew/bin/python3`.

MuJoCo passive-viewer subprocesses must run through `mjpython` on macOS. The
installer verifies that command after installing the Python dependencies.

## Windows PowerShell

```powershell
git clone https://github.com/yx-lim/ghostgui.git
cd ghostgui
powershell -ExecutionPolicy Bypass -File scripts/install_windows.ps1
powershell -ExecutionPolicy Bypass -File scripts/run_windows.ps1
```

The installer uses the Python launcher (`py -3`) when available and falls back
to `python`.

## Migrating An Existing Environment

Changing the project dependency does not make pip remove packages that were
installed previously. If `.venv` contains `PySide6` or `PySide6-Addons`, the
installer stops before changing it. Targeted uninstallation is not supported
because the PySide6 wheels contain overlapping files.

Check an activated environment with:

```bash
python scripts/check_qt_install.py
```

To reclaim the space, preserve any separately installed packages you need,
remove the dedicated `.venv`, and rerun the platform installer. The installer
never removes the environment automatically. Projects, robot models, and CSV
files are not stored inside `.venv`.

After a fresh installation, the check must report `Essentials only`. The exact
size varies by operating system, architecture, Python, and Qt version.

## Launching An Installed Checkout

Activate the environment and run the packaged command.

Linux or macOS:

```bash
source .venv/bin/activate
ghostgui
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
ghostgui
```

Select a bundled model at startup:

```bash
ghostgui --model g1
ghostgui --model go2
ghostgui --model h2
ghostgui --model z1
```

For development checkouts, the compatibility launcher remains available:

```bash
python3 scripts/run_gui.py --model g1
```

## Next Steps

- Follow the [first motion workflow](user_guide.md#first-motion-workflow).
- Read [Troubleshooting](troubleshooting.md) if Qt, OpenGL, MuJoCo, or a model
  fails to initialize.
- Use the [contributor setup](../CONTRIBUTING.md) when changing the codebase.
