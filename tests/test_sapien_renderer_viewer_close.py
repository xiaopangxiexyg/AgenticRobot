from __future__ import annotations

import contextlib
import io
import types
import unittest

from sim.rm65.sapien_renderer import SapienArmHandRenderer


class _FakeScene:
    def __init__(self):
        self.update_render_count = 0

    def update_render(self):
        self.update_render_count += 1


class _FakeViewer:
    def __init__(self, *, should_close: bool | None):
        self.window = None if should_close is None else types.SimpleNamespace(should_close=should_close)
        self.close_called = False
        self.render_called = False

    @property
    def closed(self):
        return self.window is None

    def close(self):
        self.close_called = True
        self.window = None

    def render(self):
        self.render_called = True


def _renderer_with_fake_viewer(viewer: _FakeViewer) -> SapienArmHandRenderer:
    renderer = SapienArmHandRenderer.__new__(SapienArmHandRenderer)
    renderer.left_robot = None
    renderer.right_robot = None
    renderer.scene = _FakeScene()
    renderer.viewer = viewer
    renderer._viewer_close_reported = False
    return renderer


class SapienRendererViewerCloseTest(unittest.TestCase):
    def test_render_data_continues_when_viewer_window_is_none(self) -> None:
        viewer = _FakeViewer(should_close=None)
        renderer = _renderer_with_fake_viewer(viewer)

        with contextlib.redirect_stdout(io.StringIO()):
            result = renderer.render_data(None, None, None, None, capture_image=False)

        self.assertIsNone(result)
        self.assertIsNone(renderer.viewer)
        self.assertFalse(viewer.render_called)
        self.assertEqual(renderer.scene.update_render_count, 1)

    def test_render_data_disables_viewer_when_window_requests_close(self) -> None:
        viewer = _FakeViewer(should_close=True)
        renderer = _renderer_with_fake_viewer(viewer)

        with contextlib.redirect_stdout(io.StringIO()):
            result = renderer.render_data(None, None, None, None, capture_image=False)

        self.assertIsNone(result)
        self.assertTrue(viewer.close_called)
        self.assertFalse(viewer.render_called)
        self.assertIsNone(renderer.viewer)
        self.assertEqual(renderer.scene.update_render_count, 1)


if __name__ == "__main__":
    unittest.main()
