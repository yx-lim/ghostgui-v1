"""Application UI theme tokens and QSS generation."""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication


THEME_SYSTEM = "system"
THEME_LIGHT = "light"
THEME_DARK = "dark"
THEME_MODES = (
    ("System", THEME_SYSTEM),
    ("Light", THEME_LIGHT),
    ("Dark", THEME_DARK),
)


@dataclass(frozen=True)
class Theme:
    name: str
    window: str
    panel: str
    panel_raised: str
    input_bg: str
    text: str
    text_muted: str
    border: str
    divider: str
    button: str
    button_hover: str
    button_pressed: str
    accent: str
    accent_hover: str
    accent_soft: str
    selection_text: str
    disabled_bg: str
    disabled_text: str
    danger: str
    warning: str
    success: str
    overlay_bg: str
    overlay_border: str
    scrim: str


LIGHT_THEME = Theme(
    name=THEME_LIGHT,
    window="#eef2f6",
    panel="#f8fafc",
    panel_raised="#ffffff",
    input_bg="#ffffff",
    text="#1f2933",
    text_muted="#52606d",
    border="#b8c2cc",
    divider="#d7dee7",
    button="#f3f6fa",
    button_hover="#e6edf5",
    button_pressed="#d9e8fb",
    accent="#2f80ed",
    accent_hover="#1f6fd0",
    accent_soft="#d8eaff",
    selection_text="#ffffff",
    disabled_bg="#e5e9ef",
    disabled_text="#8a97a6",
    danger="#c92a2a",
    warning="#a05a00",
    success="#247a3d",
    overlay_bg="rgba(248, 250, 252, 224)",
    overlay_border="rgba(88, 104, 124, 128)",
    scrim="rgba(17, 24, 39, 132)",
)

DARK_THEME = Theme(
    name=THEME_DARK,
    window="#151a20",
    panel="#1d232b",
    panel_raised="#252c35",
    input_bg="#11161c",
    text="#e8edf3",
    text_muted="#a9b4c0",
    border="#3c4652",
    divider="#303943",
    button="#27313b",
    button_hover="#313d49",
    button_pressed="#203c5f",
    accent="#68a7ff",
    accent_hover="#8bbcff",
    accent_soft="#18395c",
    selection_text="#07111e",
    disabled_bg="#20262e",
    disabled_text="#687483",
    danger="#ff7b72",
    warning="#d9a441",
    success="#63d083",
    overlay_bg="rgba(29, 35, 43, 224)",
    overlay_border="rgba(150, 164, 181, 132)",
    scrim="rgba(0, 0, 0, 150)",
)


def normalized_theme_mode(mode):
    mode = str(mode or THEME_SYSTEM).lower()
    if mode in {value for _label, value in THEME_MODES}:
        return mode
    return THEME_SYSTEM


class ThemeManager:
    SETTINGS_KEY = "ui/themeMode"

    def __init__(self, settings=None):
        self.settings = settings or QSettings("GhostGUI", "GhostGUI")
        self._mode = normalized_theme_mode(
            self.settings.value(self.SETTINGS_KEY, THEME_SYSTEM)
        )

    def mode(self):
        return self._mode

    def set_mode(self, mode):
        self._mode = normalized_theme_mode(mode)
        self.settings.setValue(self.SETTINGS_KEY, self._mode)
        self.apply()
        return self._mode

    def active_theme(self):
        mode = self._mode
        if mode == THEME_SYSTEM:
            mode = self.system_mode()
        return DARK_THEME if mode == THEME_DARK else LIGHT_THEME

    def system_mode(self):
        app = QApplication.instance()
        if app is None:
            return THEME_LIGHT
        style_hints = app.styleHints()
        color_scheme = getattr(style_hints, "colorScheme", None)
        if callable(color_scheme):
            scheme = color_scheme()
            if scheme == Qt.ColorScheme.Dark:
                return THEME_DARK
            if scheme == Qt.ColorScheme.Light:
                return THEME_LIGHT
        window_color = app.palette().color(QPalette.ColorRole.Window)
        return THEME_DARK if window_color.lightness() < 128 else THEME_LIGHT

    def apply(self):
        app = QApplication.instance()
        if app is None:
            return self.active_theme()
        theme = self.active_theme()
        theme_key = f"{self._mode}:{theme.name}"
        if app.property("ghostguiThemeKey") == theme_key:
            return theme
        app.setPalette(self.palette_for(theme))
        app.setStyleSheet(stylesheet_for(theme))
        app.setProperty("ghostguiThemeKey", theme_key)
        return theme

    @staticmethod
    def palette_for(theme):
        palette = QPalette()
        colors = {
            QPalette.ColorRole.Window: theme.window,
            QPalette.ColorRole.WindowText: theme.text,
            QPalette.ColorRole.Base: theme.input_bg,
            QPalette.ColorRole.AlternateBase: theme.panel,
            QPalette.ColorRole.ToolTipBase: theme.panel_raised,
            QPalette.ColorRole.ToolTipText: theme.text,
            QPalette.ColorRole.Text: theme.text,
            QPalette.ColorRole.Button: theme.button,
            QPalette.ColorRole.ButtonText: theme.text,
            QPalette.ColorRole.BrightText: theme.danger,
            QPalette.ColorRole.Highlight: theme.accent,
            QPalette.ColorRole.HighlightedText: theme.selection_text,
            QPalette.ColorRole.Link: theme.accent,
        }
        for role, value in colors.items():
            palette.setColor(role, QColor(value))
        palette.setColor(
            QPalette.ColorGroup.Disabled,
            QPalette.ColorRole.Text,
            QColor(theme.disabled_text),
        )
        palette.setColor(
            QPalette.ColorGroup.Disabled,
            QPalette.ColorRole.ButtonText,
            QColor(theme.disabled_text),
        )
        palette.setColor(
            QPalette.ColorGroup.Disabled,
            QPalette.ColorRole.Button,
            QColor(theme.disabled_bg),
        )
        return palette


def stylesheet_for(theme):
    return f"""
    QMainWindow, QDialog {{
        background: {theme.window};
        color: {theme.text};
    }}
    QScrollArea, QAbstractScrollArea {{
        background: {theme.window};
        border: none;
    }}
    QSplitter::handle {{
        background: {theme.divider};
    }}
    QWidget#CollapsibleSection {{
        border: 1px solid {theme.border};
        border-radius: 6px;
        background: {theme.panel};
    }}
    QWidget#sectionContent {{
        background: {theme.panel};
    }}
    QToolButton#sectionHeader {{
        border: none;
        border-bottom: 1px solid {theme.border};
        padding: 7px 9px;
        font-weight: 600;
        text-align: left;
        color: {theme.text};
        background: {theme.panel_raised};
    }}
    QToolButton#sectionHeader:hover {{
        background: {theme.button_hover};
    }}
    QLabel, QCheckBox, QRadioButton, QGroupBox {{
        color: {theme.text};
    }}
    QGroupBox {{
        border: 1px solid {theme.border};
        border-radius: 5px;
        margin-top: 8px;
        padding-top: 8px;
        background: {theme.panel};
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 6px;
        padding: 0 3px;
        color: {theme.text_muted};
        background: {theme.panel};
    }}
    QPushButton, QToolButton {{
        color: {theme.text};
        background: {theme.button};
        border: 1px solid {theme.border};
        border-radius: 4px;
        padding: 4px 7px;
    }}
    QPushButton:hover, QToolButton:hover {{
        background: {theme.button_hover};
        border-color: {theme.accent};
    }}
    QPushButton:pressed, QPushButton:checked, QToolButton:pressed, QToolButton:checked {{
        background: {theme.button_pressed};
        border-color: {theme.accent};
    }}
    QPushButton:disabled, QToolButton:disabled {{
        color: {theme.disabled_text};
        background: {theme.disabled_bg};
        border-color: {theme.divider};
    }}
    QComboBox, QAbstractSpinBox, QLineEdit, QTextEdit, QPlainTextEdit {{
        color: {theme.text};
        background: {theme.input_bg};
        border: 1px solid {theme.border};
        border-radius: 4px;
        padding: 2px 4px;
        selection-background-color: {theme.accent};
        selection-color: {theme.selection_text};
    }}
    QComboBox:hover, QAbstractSpinBox:hover, QLineEdit:hover, QTextEdit:hover {{
        border-color: {theme.accent};
    }}
    QComboBox::drop-down {{
        border-left: 1px solid {theme.border};
        width: 18px;
    }}
    QComboBox QAbstractItemView {{
        color: {theme.text};
        background: {theme.input_bg};
        border: 1px solid {theme.border};
        selection-background-color: {theme.accent};
        selection-color: {theme.selection_text};
    }}
    QTableWidget, QTableView {{
        color: {theme.text};
        background: {theme.input_bg};
        alternate-background-color: {theme.panel};
        border: 1px solid {theme.border};
        gridline-color: {theme.divider};
        selection-background-color: {theme.accent};
        selection-color: {theme.selection_text};
    }}
    QHeaderView::section {{
        color: {theme.text};
        background: {theme.panel_raised};
        border: none;
        border-right: 1px solid {theme.divider};
        border-bottom: 1px solid {theme.border};
        padding: 3px 4px;
    }}
    QTabWidget::pane {{
        border: 1px solid {theme.border};
        background: {theme.panel};
    }}
    QTabBar::tab {{
        color: {theme.text_muted};
        background: {theme.button};
        border: 1px solid {theme.border};
        padding: 5px 7px;
    }}
    QTabBar::tab:selected {{
        color: {theme.text};
        background: {theme.panel_raised};
        border-bottom-color: {theme.panel_raised};
    }}
    QSlider::groove:horizontal {{
        height: 5px;
        border-radius: 2px;
        background: {theme.divider};
    }}
    QSlider::sub-page:horizontal {{
        border-radius: 2px;
        background: {theme.accent};
    }}
    QSlider::handle:horizontal {{
        width: 13px;
        margin: -5px 0;
        border-radius: 6px;
        background: {theme.panel_raised};
        border: 1px solid {theme.accent};
    }}
    QSlider::handle:horizontal:hover {{
        background: {theme.accent_soft};
    }}
    QProgressBar {{
        color: {theme.text};
        border: 1px solid {theme.border};
        border-radius: 4px;
        min-height: 16px;
        text-align: center;
        background: {theme.input_bg};
    }}
    QProgressBar::chunk {{
        background: {theme.accent};
        border-radius: 3px;
    }}
    QWidget#viewerQuickActions {{
        background: {theme.overlay_bg};
        border: 1px solid {theme.overlay_border};
        border-radius: 6px;
    }}
    QWidget#renderProgressOverlay {{
        background: {theme.scrim};
    }}
    QWidget#renderProgressCard {{
        background: {theme.panel_raised};
        border: 1px solid {theme.border};
        border-radius: 6px;
    }}
    QLabel#renderTitle {{
        color: {theme.text};
        font-size: 18px;
        font-weight: 700;
    }}
    QLabel#renderDetail {{
        color: {theme.text_muted};
        font-size: 12px;
    }}
    QToolTip {{
        color: {theme.text};
        background: {theme.panel_raised};
        border: 1px solid {theme.border};
    }}
    """
