"""OpenGL motion-frame capture adapter for the AI application boundary."""

from __future__ import annotations

from PySide6.QtCore import QBuffer, QByteArray, QIODevice, QRectF, Qt, QThread
from PySide6.QtGui import QColor, QFont, QPainter

from application.ai.frame_capture import EncodedFrame, FrameCaptureError


class RobotViewerFrameRenderer:
    """Capture a clean robot pose without changing editor or camera state."""

    def __init__(self, viewer, *, maximum_dimension: int = 768):
        if viewer.robot_model is None:
            raise FrameCaptureError("robot frame capture requires a loaded model")
        if maximum_dimension <= 0:
            raise ValueError("maximum capture dimension must be positive")
        self._viewer = viewer
        self._canvas = viewer.canvas
        self._capture_state = viewer.robot_model.create_state()
        self._maximum_dimension = int(maximum_dimension)

    def render_frame(self, qpos, *, time_seconds, variant):
        canvas = self._canvas
        if QThread.currentThread() is not canvas.thread():
            raise FrameCaptureError("OpenGL frame capture must run on the GUI thread")
        if not canvas.isValid():
            raise FrameCaptureError("OpenGL canvas is not ready for frame capture")

        saved = _CanvasPresentation.capture(canvas)
        try:
            self._capture_state.set_qpos(qpos)
            canvas.set_robot_states(self._capture_state, None, None)
            canvas.preview_visible = False
            canvas.show_ghosts = False
            canvas.trajectory = None
            canvas.show_trajectory_lines = False
            canvas.show_keyframes = False
            canvas.transform_gizmo_visible = False
            canvas.selected_target_kind = None
            canvas.selected_target_name = None
            canvas.selected_body_id = None
            image = canvas.grabFramebuffer()
            if image.isNull():
                raise FrameCaptureError("OpenGL canvas returned an empty frame")
            image = _bounded_image(image, self._maximum_dimension)
            _paint_timestamp(image, float(time_seconds))
            return EncodedFrame(_encode_png(image), "image/png")
        finally:
            saved.restore(canvas)
            canvas.update()


class _CanvasPresentation:
    _FIELDS = (
        "robot_state",
        "preview_state",
        "ghost_renderer",
        "preview_visible",
        "show_ghosts",
        "trajectory",
        "show_trajectory_lines",
        "show_keyframes",
        "transform_gizmo_visible",
        "selected_target_kind",
        "selected_target_name",
        "selected_body_id",
    )

    def __init__(self, values):
        self._values = values

    @classmethod
    def capture(cls, canvas):
        return cls({name: getattr(canvas, name) for name in cls._FIELDS})

    def restore(self, canvas):
        for name, value in self._values.items():
            setattr(canvas, name, value)


def _bounded_image(image, maximum_dimension):
    if max(image.width(), image.height()) <= maximum_dimension:
        return image
    return image.scaled(
        maximum_dimension,
        maximum_dimension,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )


def _paint_timestamp(image, time_seconds):
    text = f"t={time_seconds:.2f} s"
    painter = QPainter(image)
    try:
        font = QFont(painter.font())
        font.setBold(True)
        font.setPixelSize(max(14, min(image.width(), image.height()) // 28))
        painter.setFont(font)
        metrics = painter.fontMetrics()
        padding = max(6, font.pixelSize() // 3)
        box = QRectF(
            12,
            12,
            metrics.horizontalAdvance(text) + 2 * padding,
            metrics.height() + 2 * padding,
        )
        painter.fillRect(box, QColor(0, 0, 0, 190))
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(box, Qt.AlignmentFlag.AlignCenter, text)
    finally:
        painter.end()


def _encode_png(image):
    data = QByteArray()
    buffer = QBuffer(data)
    if not buffer.open(QIODevice.OpenModeFlag.WriteOnly):
        raise FrameCaptureError("could not open the PNG output buffer")
    try:
        if not image.save(buffer, "PNG"):
            raise FrameCaptureError("could not encode the captured frame as PNG")
        return bytes(data)
    finally:
        buffer.close()
