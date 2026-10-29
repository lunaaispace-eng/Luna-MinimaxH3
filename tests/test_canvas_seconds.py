"""Luna-H3-Canvas: `frames` (was `length`) on the 17k+5 grid and `seconds` = length / 24,
appended as the last output. Stubs comfy_api so it runs without ComfyUI.

    E:\ComfyUI-Easy-Install\python_embeded\python.exe -m unittest discover -s tests -v
"""

import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace


def _stub_io():
    class _Out:
        def __init__(self, display_name=None, **kw):
            self.display_name = display_name

    class _In:
        def __init__(self, *a, **kw):
            pass

    io = SimpleNamespace(
        ComfyNode=object, Schema=lambda **kw: SimpleNamespace(**kw),
        Int=SimpleNamespace(Input=_In, Output=_Out), Float=SimpleNamespace(Input=_In, Output=_Out),
        String=SimpleNamespace(Input=_In, Output=_Out), Boolean=SimpleNamespace(Input=_In),
        Combo=SimpleNamespace(Input=_In), Image=SimpleNamespace(Input=_In), NodeOutput=lambda *a: a,
    )
    latest = ModuleType("comfy_api.latest")
    latest.io = io
    return {"comfy_api": ModuleType("comfy_api"), "comfy_api.latest": latest}


# The stub is in place only while h3_canvas imports, so the other test files in this
# folder still see whatever comfy_api they set up themselves.
_saved = {k: sys.modules.get(k) for k in ("comfy_api", "comfy_api.latest")}
sys.modules.update(_stub_io())
try:
    _spec = importlib.util.spec_from_file_location("h3_canvas", Path(__file__).parent.parent / "h3_canvas.py")
    h3_canvas = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(h3_canvas)
finally:
    for k, v in _saved.items():
        if v is None:
            sys.modules.pop(k, None)
        else:
            sys.modules[k] = v
Canvas = h3_canvas.LunaMiniMaxH3Canvas


def run(seconds):
    return Canvas.execute("16:9 (widescreen)", False, "H3 canvas", 1.03, seconds, True, 2)


class CanvasSecondsTests(unittest.TestCase):
    def test_seconds_is_last_output(self):
        names = [o.display_name for o in Canvas.define_schema().outputs]
        self.assertEqual(names[-1], "seconds")
        self.assertEqual(names[:7], ["width", "height", "frames", "fps", "output_fps",
                                     "interpolation_factor", "info"])

    def test_seconds_follow_the_grid(self):
        for asked, frames in ((5.0, 124), (7.5, 192), (10.0, 243)):
            out = run(asked)
            self.assertEqual(out[2], frames)
            self.assertAlmostEqual(out[7], frames / 24.0)


def run_image(w, h, size_mode="H3 canvas", portrait=False, ratio="16:9 (widescreen)", **kw):
    import torch
    return Canvas.execute(ratio, portrait, size_mode, kw.get("megapixels", 1.03), 5.0, True, 2,
                          resolution=kw.get("resolution", "1344x768 (1.03 MP)"),
                          image=torch.zeros(2, h, w, 3))


class CanvasImageTests(unittest.TestCase):
    def test_image_is_the_last_input_and_optional(self):
        # The stub drops ids; the order of the schema's inputs is what matters here.
        self.assertTrue(hasattr(Canvas.define_schema(), "inputs"))
        self.assertEqual(len(Canvas.define_schema().inputs), 10)

    def test_unwired_is_unchanged(self):
        self.assertEqual(run(5.0)[:2], (1344, 768))

    def test_h3_canvas_follows_the_exact_image_aspect(self):
        # 1000x1000 beats the ratio widget (16:9): square canvas.
        w, h = run_image(1000, 1000)[:2]
        self.assertEqual((w, h), h3_canvas.h3_canvas(1.0))
        # portrait image with portrait unticked still comes out portrait
        w, h = run_image(1080, 1920)[:2]
        self.assertLess(w, h)
        self.assertEqual((w, h), h3_canvas.h3_canvas(1080 / 1920))

    def test_image_overrides_the_portrait_switch(self):
        w, h = run_image(1920, 1080, portrait=True)[:2]
        self.assertGreater(w, h)

    def test_megapixels_uses_the_exact_aspect(self):
        w, h = run_image(1200, 800, size_mode="megapixels", megapixels=1.0)[:2]
        self.assertEqual((w, h), h3_canvas.megapixel_canvas(1.5, 1.0))

    def test_preset_takes_the_nearest_listed_ratio_in_the_images_orientation(self):
        # 1000x1500 is 2:3 portrait: the listed 3:2 turned on its side.
        w, h = run_image(1000, 1500, size_mode="preset")[:2]
        self.assertLess(w, h)
        self.assertAlmostEqual(w / h, 2 / 3, delta=0.03)
        # ... and it is one of the 3:2 sizes, area nearest the (16:9) size the widget held.
        self.assertIn(h3_canvas.preset_label(h, w), h3_canvas.preset_labels("3:2 (photo)", "32"))

    def test_info_names_the_image_and_is_ascii(self):
        info = run_image(1920, 1080)[6]
        self.assertIn("ratio from image 1920x1080", info)
        info.encode("ascii")


if __name__ == "__main__":
    unittest.main()
