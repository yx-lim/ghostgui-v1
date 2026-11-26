"""Shared value ranges. The GUI binder applies the same bounds to its widgets."""

from typing import Annotated

from .schema import Doc, Len, Range, Unit

TimeSeconds = Annotated[float, Range(0, 120), Unit("s")]
PlaybackSpeed = Annotated[float, Range(0.10, 4.00), Unit("x")]
Opacity = Annotated[float, Range(0.0, 1.0)]
CameraYaw = Annotated[float, Range(-360, 360), Unit("deg")]
CameraPitch = Annotated[float, Range(-89, 89), Unit("deg")]
CameraDistance = Annotated[float, Range(0.2, 50), Unit("m")]
Name = Annotated[str, Len(1, 80)]
ShortText = Annotated[str, Len(1, 200)]
Question = Annotated[str, Len(1, 500), Doc("A short question shown to the user.")]
# GUI ranges (robot_viewer_3d.py). Legacy MCP tool ranges are intentionally left unchanged.
GUI_EXPORT_INTERVAL = (0.01, 10.0)
GUI_PLAYBACK_SPEED = (0.10, 4.00)
