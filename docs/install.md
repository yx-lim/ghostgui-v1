# GhostGUI Installation Notes

Linux/Ubuntu is the primary tested platform for GhostGUI. macOS and Windows
support is currently experimental unless verified on the target machine.

## Install Flow

GhostGUI uses `pyproject.toml` for Python package metadata and dependencies.
The platform scripts create or reuse `.venv`, install or check the platform
setup, upgrade `pip`, `setuptools`, and `wheel`, then run:

```bash
python -m pip install -e .
```

Editable installation is intentional for version `0.1.0`: the registered robot
models and bundled mesh assets are loaded from the cloned repository.

## Commands

Linux:

```bash
bash scripts/install_linux.sh
bash scripts/run_linux.sh
```

macOS:

```bash
bash scripts/install_macos.sh
bash scripts/run_macos.sh
```

The macOS installer supports both native Mac architectures: `arm64` for Apple
Silicon and `x86_64` for Intel Macs. A virtual environment does not change
Python architecture; it inherits the interpreter used to create it. The script
therefore checks that Python, `.venv`, and `mjpython` match the native Mac
architecture. On Apple Silicon, it prefers `/opt/homebrew/bin/python3` and stops
with instructions if it detects an Intel/Rosetta `x86_64` Python or venv.

The standalone MuJoCo viewer uses `mujoco.viewer.launch_passive()`. On macOS,
MuJoCo requires that passive viewer scripts run through `mjpython`, which is
installed by the `mujoco` package. GhostGUI uses `mjpython` automatically for
that subprocess on macOS.

Windows PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/install_windows.ps1
powershell -ExecutionPolicy Bypass -File scripts/run_windows.ps1
```

## Troubleshooting

If Qt fails to initialize on Linux, confirm the system packages from
`scripts/install_linux.sh` were installed. Missing `libxcb-cursor0`,
`libxcb-xinerama0`, `libxcb-xinput0`, `libxkbcommon-x11-0`, `libgl1`, or
`libegl1` commonly appears as a Qt platform plugin or OpenGL initialization
failure.

If MuJoCo fails to open an OpenGL context, confirm the machine has working GPU
drivers or a valid software OpenGL stack. Remote desktops, containers, and WSL
may need additional display/OpenGL configuration.

If model loading fails, run GhostGUI from an editable install created from the
cloned repository. The current model registry expects the `models/` directory to
live beside the source tree.

If the `ghostgui` command is not found, activate the virtual environment first:

Linux/macOS:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```
