"""
viewer.py

Compatibility import for older code that still imports RobotCanvas from
gui.viewer. New code should import from gui.viewer_2d or gui.viewer_3d.
"""

from .viewer_2d import RobotCanvas
