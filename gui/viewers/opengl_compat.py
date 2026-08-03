"""Desktop OpenGL compatibility requests and realized-context checks."""

from __future__ import annotations

import os

from PySide6.QtGui import QSurfaceFormat


MINIMUM_OPENGL_VERSION = (2, 1)
DEFAULT_DEPTH_BUFFER_SIZE = 24
OPENGL_MODE_ENV = "GHOSTGUI_OPENGL_MODE"
OPENGL_MODE_COMPATIBILITY = "compatibility"
OPENGL_MODE_DEFAULT = "default"
OPENGL_MODES = (OPENGL_MODE_COMPATIBILITY, OPENGL_MODE_DEFAULT)


def normalize_opengl_mode(mode: str | None) -> str:
    """Normalize an A/B render mode, falling back to the supported runtime."""
    value = str(mode or "").strip().lower()
    return value if value in OPENGL_MODES else OPENGL_MODE_COMPATIBILITY


def current_opengl_mode() -> str:
    return normalize_opengl_mode(os.environ.get(OPENGL_MODE_ENV))


def desktop_compatibility_format(
    base_format: QSurfaceFormat | None = None,
) -> QSurfaceFormat:
    """Return the fixed-function surface format used by GhostGUI's viewer."""
    surface_format = (
        QSurfaceFormat(base_format)
        if base_format is not None
        else QSurfaceFormat()
    )
    surface_format.setRenderableType(QSurfaceFormat.RenderableType.OpenGL)
    surface_format.setVersion(*MINIMUM_OPENGL_VERSION)
    # Profiles do not exist for OpenGL 2.1. Request deprecated functions so a
    # driver that promotes the version must retain this renderer's fixed API.
    surface_format.setProfile(QSurfaceFormat.OpenGLContextProfile.NoProfile)
    surface_format.setOption(
        QSurfaceFormat.FormatOption.DeprecatedFunctions,
        on=True,
    )
    surface_format.setDepthBufferSize(DEFAULT_DEPTH_BUFFER_SIZE)
    return surface_format


def surface_format_for_mode(
    base_format: QSurfaceFormat | None = None,
    mode: str | None = None,
) -> QSurfaceFormat:
    """Return either the hardened format or Qt's unmodified default format."""
    if normalize_opengl_mode(mode or current_opengl_mode()) == (
        OPENGL_MODE_COMPATIBILITY
    ):
        return desktop_compatibility_format(base_format)
    return (
        QSurfaceFormat(base_format)
        if base_format is not None
        else QSurfaceFormat(QSurfaceFormat.defaultFormat())
    )


def configure_default_surface_format(mode: str | None = None) -> QSurfaceFormat:
    """Configure the selected process format before constructing QApplication."""
    selected_mode = normalize_opengl_mode(mode or current_opengl_mode())
    surface_format = surface_format_for_mode(mode=selected_mode)
    if selected_mode == OPENGL_MODE_COMPATIBILITY:
        QSurfaceFormat.setDefaultFormat(surface_format)
    return surface_format


def compatibility_context_failure(context) -> str | None:
    """Describe why a realized Qt context cannot run the legacy renderer."""
    if context is None:
        return "Qt did not create an OpenGL context"
    if not context.isValid():
        return "Qt created an invalid OpenGL context"
    if context.isOpenGLES():
        return "Qt selected OpenGL ES instead of desktop OpenGL"

    actual_format = context.format()
    version = (
        int(actual_format.majorVersion()),
        int(actual_format.minorVersion()),
    )
    if version < MINIMUM_OPENGL_VERSION:
        return (
            f"the available desktop OpenGL version is {version[0]}.{version[1]} "
            "(2.1 or newer is required)"
        )
    if version >= (3, 2) and actual_format.profile() != (
        QSurfaceFormat.OpenGLContextProfile.CompatibilityProfile
    ):
        return (
            f"Qt created an OpenGL {version[0]}.{version[1]} core-only "
            "context without fixed-function compatibility"
        )
    if version in ((3, 0), (3, 1)) and not actual_format.testOption(
        QSurfaceFormat.FormatOption.DeprecatedFunctions
    ):
        return (
            f"Qt created an OpenGL {version[0]}.{version[1]} context without "
            "the deprecated fixed-function API"
        )
    if actual_format.depthBufferSize() <= 0:
        return "the OpenGL context has no depth buffer"
    return None
