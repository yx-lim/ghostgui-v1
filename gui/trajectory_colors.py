from PySide6.QtCore import Qt


FRAME_QT_COLORS = {
    "pelvis": Qt.GlobalColor.darkGreen,
    "torso": Qt.GlobalColor.darkCyan,
    "left_foot": Qt.GlobalColor.blue,
    "right_foot": Qt.GlobalColor.magenta,
    "left_hand": Qt.GlobalColor.darkYellow,
    "right_hand": Qt.GlobalColor.darkRed,
}


FRAME_GL_COLORS = {
    "pelvis": (0.05, 0.85, 0.25),
    "torso": (0.00, 0.72, 0.85),
    "left_foot": (0.18, 0.42, 1.00),
    "right_foot": (0.85, 0.20, 0.95),
    "left_hand": (0.95, 0.70, 0.10),
    "right_hand": (0.95, 0.15, 0.12),
}


DEFAULT_QT_COLOR = Qt.GlobalColor.green
DEFAULT_GL_COLOR = (0.10, 0.95, 0.35)


def qt_color_for_frame(frame_name):
    return FRAME_QT_COLORS.get(frame_name, DEFAULT_QT_COLOR)


def gl_color_for_frame(frame_name):
    return FRAME_GL_COLORS.get(frame_name, DEFAULT_GL_COLOR)
