"""Desktop OpenGL compatibility requests and realized-context checks."""

from __future__ import annotations

from PySide6.QtGui import QSurfaceFormat


MINIMUM_OPENGL_VERSION = (2, 1)
DEFAULT_DEPTH_BUFFER_SIZE = 24


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


def configure_default_surface_format() -> QSurfaceFormat:
    """Set the process-wide format; call before constructing QApplication."""
    surface_format = desktop_compatibility_format()
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
