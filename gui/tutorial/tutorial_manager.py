"""Tutorial manager that coordinates steps, highlighting, and navigation."""

from PySide6.QtCore import QEvent, QObject, QPoint, QRect, QTimer
from PySide6.QtWidgets import QMenu, QWidget

from .tutorial_card import TutorialCard
from .tutorial_overlay import TutorialOverlay
from .tutorial_steps import FIRST_MOTION_TUTORIAL


class TutorialManager(QObject):
    """Owns the first-motion guided tutorial for a main window."""

    def __init__(self, main_window):
        super().__init__(main_window)
        self.main_window = main_window
        self.steps = []
        self.current_index = 0
        self.overlay = None
        self.menu_overlay = None
        self.toolbar_overlay = None
        self.card = None
        self.active = False

    def start_first_motion(self):
        self.start(FIRST_MOTION_TUTORIAL)

    def start(self, steps):
        if not steps:
            return
        self.steps = list(steps)
        self.current_index = 0
        self.active = True
        self._ensure_widgets()
        self.main_window.installEventFilter(self)
        self.overlay.setGeometry(self.main_window.rect())
        self.overlay.show()
        self.menu_overlay.show()
        self.toolbar_overlay.show()
        self.card.show()
        self.show_current_step()

    def stop(self):
        if not self.active:
            return
        self.active = False
        self.main_window.removeEventFilter(self)
        if self.overlay is not None:
            self.overlay.hide()
            self.overlay.set_target_rect(QRect())
        if self.menu_overlay is not None:
            self.menu_overlay.hide()
            self.menu_overlay.set_target_rect(QRect())
        if self.toolbar_overlay is not None:
            self.toolbar_overlay.hide()
            self.toolbar_overlay.set_target_rect(QRect())
        if self.card is not None:
            self.card.hide()

    def next_step(self):
        if not self.active:
            return
        if self.current_index >= len(self.steps) - 1:
            self.stop()
            return
        self.current_index += 1
        self.show_current_step()

    def previous_step(self):
        if not self.active or self.current_index <= 0:
            return
        self.current_index -= 1
        self.show_current_step()

    def show_current_step(self):
        if not self.active:
            return
        step = self.steps[self.current_index]
        self._run_before_show(step.before_show)
        self.card.set_step(step, self.current_index, len(self.steps))
        QTimer.singleShot(0, self.reposition)

    def reposition(self):
        if not self.active:
            return
        self.overlay.setGeometry(self.main_window.rect())
        step = self.steps[self.current_index]
        target_rect = self._target_rect(step.target)
        self.overlay.set_target_rect(target_rect)
        menu_bar = self.main_window.menuBar()
        toolbar = self.main_window.app_toolbar
        self.menu_overlay.setGeometry(menu_bar.rect())
        self.menu_overlay.set_target_rect(
            self._target_rect_on_surface(step.target, menu_bar)
        )
        self.toolbar_overlay.setGeometry(toolbar.rect())
        self.toolbar_overlay.set_target_rect(
            self._target_rect_on_surface(step.target, toolbar)
        )
        self._position_card(
            target_rect if step.position_card_near_target else QRect(),
            placement=step.card_placement,
        )
        self.overlay.raise_()
        menu_bar.raise_()
        toolbar.raise_()
        self.menu_overlay.show()
        self.menu_overlay.raise_()
        self.toolbar_overlay.show()
        self.toolbar_overlay.raise_()
        self.overlay.stackUnder(self.card)
        self.card.raise_()

    def eventFilter(self, watched, event):
        if watched is self.main_window and event.type() in (
            QEvent.Type.Resize,
            QEvent.Type.Show,
        ):
            QTimer.singleShot(0, self.reposition)
        return super().eventFilter(watched, event)

    def _ensure_widgets(self):
        if self.overlay is None:
            self.overlay = TutorialOverlay(self.main_window)
        if self.menu_overlay is None:
            self.menu_overlay = TutorialOverlay(self.main_window.menuBar())
            self.menu_overlay.setObjectName("tutorialMenuOverlay")
        if self.toolbar_overlay is None:
            self.toolbar_overlay = TutorialOverlay(self.main_window.app_toolbar)
            self.toolbar_overlay.setObjectName("tutorialToolbarOverlay")
        if self.card is None:
            self.card = TutorialCard(self.main_window)
            self.card.back_requested.connect(self.previous_step)
            self.card.next_requested.connect(self.next_step)
            self.card.skip_requested.connect(self.stop)

    def _target_rect(self, target_spec):
        if not target_spec:
            return QRect()
        object_names = (
            (target_spec,)
            if isinstance(target_spec, str)
            else tuple(target_spec)
        )
        target_rect = QRect()
        for object_name in object_names:
            target = self.main_window.findChild(QWidget, object_name)
            if isinstance(target, QMenu):
                menu_bar = self.main_window.menuBar()
                action_rect = menu_bar.actionGeometry(target.menuAction())
                if action_rect.isNull():
                    continue
                top_left = menu_bar.mapTo(
                    self.main_window,
                    action_rect.topLeft(),
                )
                widget_rect = QRect(top_left, action_rect.size())
            elif target is None or not target.isVisible():
                continue
            else:
                top_left = target.mapTo(self.main_window, QPoint(0, 0))
                widget_rect = QRect(top_left, target.size())
            target_rect = (
                widget_rect
                if target_rect.isNull()
                else target_rect.united(widget_rect)
            )
        return target_rect

    def _target_rect_on_surface(self, target_spec, surface):
        if not target_spec:
            return QRect()
        object_names = (
            (target_spec,)
            if isinstance(target_spec, str)
            else tuple(target_spec)
        )
        target_rect = QRect()
        menu_bar = self.main_window.menuBar()
        toolbar = self.main_window.app_toolbar
        for object_name in object_names:
            target = self.main_window.findChild(QWidget, object_name)
            if isinstance(target, QMenu):
                if surface is not menu_bar:
                    continue
                widget_rect = menu_bar.actionGeometry(target.menuAction())
            else:
                if target is None or not target.isVisible():
                    continue
                belongs_to_toolbar = target is toolbar or toolbar.isAncestorOf(target)
                if surface is toolbar and not belongs_to_toolbar:
                    continue
                if surface is menu_bar:
                    continue
                if surface is self.main_window and belongs_to_toolbar:
                    continue
                if surface is not toolbar and surface is not self.main_window:
                    continue
                top_left = target.mapTo(surface, QPoint(0, 0))
                widget_rect = QRect(top_left, target.size())
            if widget_rect.isNull():
                continue
            target_rect = (
                widget_rect
                if target_rect.isNull()
                else target_rect.united(widget_rect)
            )
        return target_rect

    def _position_card(self, target_rect, *, placement="auto"):
        self.card.adjustSize()
        margin = 14
        window_rect = self.main_window.rect()
        card_size = self.card.size()

        if target_rect.isNull():
            x = (window_rect.width() - card_size.width()) // 2
            y = (window_rect.height() - card_size.height()) // 2
        elif placement == "below":
            x = target_rect.left()
            toolbar = self.main_window.app_toolbar
            toolbar_bottom = toolbar.mapTo(
                self.main_window,
                QPoint(0, toolbar.height()),
            ).y()
            y = toolbar_bottom + margin
        else:
            right_x = target_rect.right() + margin
            left_x = target_rect.left() - card_size.width() - margin
            below_y = target_rect.bottom() + margin
            above_y = target_rect.top() - card_size.height() - margin

            if right_x + card_size.width() <= window_rect.right() - margin:
                x = right_x
                y = target_rect.top()
            elif left_x >= margin:
                x = left_x
                y = target_rect.top()
            else:
                x = target_rect.left()
                y = below_y if below_y + card_size.height() <= window_rect.bottom() else above_y

            toolbar = self.main_window.app_toolbar
            toolbar_bottom = toolbar.mapTo(
                self.main_window,
                QPoint(0, toolbar.height()),
            ).y()
            if target_rect.top() < toolbar_bottom and y < toolbar_bottom + margin:
                y = toolbar_bottom + margin

        x = max(margin, min(x, window_rect.width() - card_size.width() - margin))
        y = max(margin, min(y, window_rect.height() - card_size.height() - margin))
        self.card.move(x, y)

    def _run_before_show(self, hook):
        if hook == "show_3d_view":
            self.main_window.viewer_tabs.setCurrentWidget(
                self.main_window.viewer_3d_stack
            )
            self.main_window.update_editor_context()
        elif hook == "expand_setup":
            self._run_before_show("show_3d_view")
            toolbar = getattr(self.main_window, "app_toolbar", None)
            if toolbar is not None:
                toolbar.show()
        elif hook == "expand_target_pose":
            self._run_before_show("show_3d_view")
            self._set_section_expanded("Target", True)
        elif hook == "expand_end_effector_editor":
            self._run_before_show("expand_target_pose")
            self._set_section_expanded("Editing Mode", True)
            self.main_window.controls.set_editing_mode("end_effector")
        elif hook == "expand_time_slices":
            self._run_before_show("show_3d_view")
            self._set_section_expanded("Planning", True)

    def _set_section_expanded(self, title, expanded):
        for sidebar in (self.main_window.left_sidebar_content, self.main_window.right_sidebar_content):
            for section in sidebar.sections:
                if section.title == title:
                    section.set_expanded(expanded)
                    return
