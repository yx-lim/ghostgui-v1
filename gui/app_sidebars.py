"""Application-level sidebar sections for the GhostGUI main window."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


class CollapsibleSection(QWidget):
    """A compact section that hides content without deleting its widgets."""

    def __init__(self, title, content, expanded=True, parent=None):
        super().__init__(parent)
        self.title = title
        self.content = content
        self.content.setObjectName(self.content.objectName() or "sectionContent")
        self.content.setMinimumWidth(0)

        self.header = QToolButton()
        self.header.setObjectName("sectionHeader")
        self.header.setText(title)
        self.header.setCheckable(True)
        self.header.setChecked(bool(expanded))
        self.header.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.header.toggled.connect(self.set_expanded)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.header)
        layout.addWidget(self.content)

        self.setObjectName("CollapsibleSection")
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )
        self.setStyleSheet(
            """
            QWidget#CollapsibleSection {
                border: 1px solid #454545;
                border-radius: 6px;
                background: #252525;
            }
            QToolButton#sectionHeader {
                border: none;
                border-bottom: 1px solid #454545;
                padding: 7px 9px;
                font-weight: 600;
                text-align: left;
                background: #e0e0e0;
            }
            QToolButton#sectionHeader:hover {
                background: #383838;
            }
            """
        )
        self.set_expanded(bool(expanded))

    def set_expanded(self, expanded):
        expanded = bool(expanded)
        self.content.setVisible(expanded)
        self.header.setChecked(expanded)
        self.header.setArrowType(
            Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
        )


class AppSidebar(QWidget):
    """One scroll area containing all app-level sidebar sections."""

    SECTION_MAX_WIDTH = 240

    def __init__(self, parent=None):
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.body = QWidget()
        self.body_layout = QVBoxLayout(self.body)
        self.body_layout.setContentsMargins(4, 4, 4, 4)
        self.body_layout.setSpacing(4)
        self.scroll.setWidget(self.body)
        self.sections = []

        root.addWidget(self.scroll)

    def add_section(self, title, widget, expanded=True):
        widget.setMaximumWidth(self.SECTION_MAX_WIDTH - 16)
        section = CollapsibleSection(title, widget, expanded=False)
        section.setMaximumWidth(self.SECTION_MAX_WIDTH)
        self.body_layout.addWidget(section)
        self.sections.append(section)
        return section

    def add_sections(self, sections):
        added = []
        for title, widget, expanded in sections:
            added.append(self.add_section(title, widget, expanded=expanded))
        return added

    def add_stretch(self):
        self.body_layout.addStretch(1)


class AppLeftSidebar(AppSidebar):
    def __init__(self, trajectory_controls, editor_tabs=None, parent=None):
        super().__init__(parent)
        self.add_sections(trajectory_controls.workflow_sections())
        self.view_panel = self._build_view_panel(
            trajectory_controls.view_panel, editor_tabs
        )
        self.add_section("View", self.view_panel, expanded=False)
        self.add_stretch()

    def _build_view_panel(self, display_panel, editor_tabs):
        layout = display_panel.layout()
        if editor_tabs is None:
            return display_panel
        buttons = []
        for index in range(editor_tabs.count()):
            button = QPushButton(editor_tabs.tabText(index))
            button.setCheckable(True)
            button.clicked.connect(
                lambda checked=False, tab_index=index: editor_tabs.setCurrentIndex(
                    tab_index
                )
            )
            buttons.append(button)
            layout.addWidget(button)

        def sync_active(active_index):
            for index, button in enumerate(buttons):
                button.setChecked(index == active_index)

        editor_tabs.currentChanged.connect(sync_active)
        sync_active(editor_tabs.currentIndex())
        return display_panel


class AppRightSidebar(AppSidebar):
    SECTION_MAX_WIDTH = 260

    def __init__(self, status_panel, base_sections=None, parent=None):
        super().__init__(parent)
        if base_sections:
            self.add_sections(base_sections)
        self.add_section("Status", status_panel, expanded=True)
        self.add_stretch()

    def set_context_widget(self, widget):
        return

    def current_context_widget(self):
        return None
