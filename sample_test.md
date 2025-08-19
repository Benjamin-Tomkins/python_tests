import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseEvent, KeyEvent
import importlib
import types
import builtins
import pathlib

# ---- utilities --------------------------------------------------------------

def _make_temp_image(tmp_path):
    # simple 50x50 RGB gradient
    img = np.linspace(0, 1, 50*50).reshape(50, 50)
    p = tmp_path / "test.png"
    plt.imsave(p.as_posix(), img, cmap="gray")
    return p

class SubplotsCapture:
    """Capture fig/ax produced inside the function under test."""
    def __init__(self, real_subplots):
        self.real_subplots = real_subplots
        self.fig = None
        self.ax = None

    def __call__(self, *a, **k):
        self.fig, self.ax = self.real_subplots(*a, **k)
        return self.fig, self.ax

def _data_to_pixels(ax, x, y):
    xpix, ypix = ax.transData.transform((x, y))
    return float(xpix), float(ypix)

def _post_mouse(fig, ax, name, xdata, ydata, button):
    xpix, ypix = _data_to_pixels(ax, xdata, ydata)
    ev = MouseEvent(name, fig.canvas, xpix, ypix, button=button)
    fig.canvas.callbacks.process(name, ev)

def _post_key(fig, name, key):
    ev = KeyEvent(name, fig.canvas, key=key)
    fig.canvas.callbacks.process(name, ev)

# ---- tests -----------------------------------------------------------------

def test_single_rectangle_from_two_left_clicks(monkeypatch, tmp_path):
    img_path = _make_temp_image(tmp_path)

    # Import the module fresh each time, so monkeypatches apply cleanly
    import importlib
    mod = importlib.import_module("new_annotation_helper")

    # capture fig/ax created inside annotation_helper
    capturer = SubplotsCapture(plt.subplots)
    monkeypatch.setattr(plt, "subplots", capturer)

    # Replace plt.show with a driver that posts two left-clicks
    def fake_show():
        # first corner at (10, 20), second at (30, 40)
        _post_mouse(capturer.fig, capturer.ax, "button_press_event", 10, 20, button=1)
        _post_mouse(capturer.fig, capturer.ax, "button_press_event", 30, 40, button=1)
    monkeypatch.setattr(plt, "show", fake_show)

    rects = mod.annotation_helper(img_path.as_posix())
    assert rects == [[(10, 20), (30, 40)]]

def test_right_click_is_ignored(monkeypatch, tmp_path):
    img_path = _make_temp_image(tmp_path)
    mod = importlib.import_module("new_annotation_helper")

    capturer = SubplotsCapture(plt.subplots)
    monkeypatch.setattr(plt, "subplots", capturer)

    def fake_show():
        # Right click (button=3) should be ignored
        _post_mouse(capturer.fig, capturer.ax, "button_press_event", 5, 5, button=3)
        # Then two left-clicks start and finish a rectangle
        _post_mouse(capturer.fig, capturer.ax, "button_press_event", 1, 1, button=1)
        _post_mouse(capturer.fig, capturer.ax, "button_press_event", 2, 2, button=1)
    monkeypatch.setattr(plt, "show", fake_show)

    rects = mod.annotation_helper(img_path.as_posix())
    assert rects == [[(1, 1), (2, 2)]]

def test_multiple_rectangles(monkeypatch, tmp_path):
    img_path = _make_temp_image(tmp_path)
    mod = importlib.import_module("new_annotation_helper")

    capturer = SubplotsCapture(plt.subplots)
    monkeypatch.setattr(plt, "subplots", capturer)

    def fake_show():
        # First rectangle
        _post_mouse(capturer.fig, capturer.ax, "button_press_event", 0, 0, button=1)
        _post_mouse(capturer.fig, capturer.ax, "button_press_event", 10, 10, button=1)
        # Second rectangle
        _post_mouse(capturer.fig, capturer.ax, "button_press_event", 20, 5, button=1)
        _post_mouse(capturer.fig, capturer.ax, "button_press_event", 25, 15, button=1)
    monkeypatch.setattr(plt, "show", fake_show)

    rects = mod.annotation_helper(img_path.as_posix())
    assert rects == [
        [(0, 0), (10, 10)],
        [(20, 5), (25, 15)],
    ]

def test_formatter_lists_each_rectangle_on_its_own_line():
    # Import without executing __main__
    mod = importlib.import_module("new_annotation_helper")
    rects = [[(10, 20), (30, 40)], [(1, 2), (3, 4)]]
    formatted = mod.format_rectangles_from_matplotlib_to_annotate(rects)

    # Current implementation (per your screenshot) returns a 3-element list
    # of: leading newline, joined lines, trailing newline.
    assert formatted[0] == "\n"
    assert formatted[2] == "\n"
    assert formatted[1].splitlines() == ["10, 20, 30, 40.", "1, 2, 3, 4."]

