"""Luna-H3-AssetLoader picture handling (asset_images.py), fixed 2026-10-25.

    E:\\ComfyUI-Easy-Install\\python_embeded\\python.exe -m unittest discover -s tests -v
"""

import os
import sys
import unittest

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from asset_images import apply_state, conform, snap_size, to_rgb  # noqa: E402


def stripes(w, h):
    """Left half red, right half blue, a green band on the last 4 rows: shows crops."""
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:, : w // 2] = (255, 0, 0)
    arr[:, w // 2:] = (0, 0, 255)
    arr[-4:, :] = (0, 255, 0)
    return Image.fromarray(arr)


class SnapTests(unittest.TestCase):
    def test_snap_crops_never_stretches(self):
        out = apply_state(stripes(1000, 563), {"mode": "off", "snap": 32})
        self.assertEqual(out.size, (992, 544))  # both edges DOWN to the grid
        # 1000x563 -> cover 992x544 needs scale 0.992 -> 992x559, then 15 rows cropped:
        # the content is scaled evenly, so the red/blue split stays at the middle.
        row = np.asarray(out)[272].astype(int)
        self.assertTrue((abs(row[:490] - (255, 0, 0)) <= 2).all())
        self.assertTrue((abs(row[502:] - (0, 0, 255)) <= 2).all())

    def test_snap_never_upscales(self):
        self.assertEqual(apply_state(stripes(1010, 1010), {"snap": 32}).size, (992, 992))
        self.assertEqual(snap_size(1010, 1010, 32), (992, 992))

    def test_snap_on_grid_is_untouched(self):
        img = stripes(1024, 576)
        self.assertIs(apply_state(img, {"snap": 32}), img)

    def test_snap_after_scale(self):
        out = apply_state(stripes(4000, 2250), {"mode": "max_mp", "max_mp": 1.0, "snap": 32})
        self.assertEqual(out.size, (1312, 736))  # 1333x750 down to the grid
        self.assertLessEqual(out.size[0] * out.size[1], 1_000_000)

    def test_float_slack(self):
        # 2048 * 0.5 must be 1024 on a /32 grid, not 992 from a hair under it.
        self.assertEqual(snap_size(2048 * 0.49999999999, 1024, 32), (1024, 1024))

    def test_crop_anchor_bottom_keeps_last_row(self):
        out = apply_state(stripes(1000, 600), {"snap": 32, "crop_anchor": "bottom"})
        self.assertEqual(out.size, (992, 576))
        last = np.asarray(out)[-1, 10:-10].astype(int)
        self.assertTrue((abs(last - (0, 255, 0)) <= 2).all())

    def test_no_snap_scales_whole(self):
        self.assertEqual(apply_state(stripes(1000, 563), {"mode": "scale_factor", "scale_factor": 0.5}).size,
                         (500, 282))


class LoadModeTests(unittest.TestCase):
    def test_transparent_filled_with_background(self):
        img = Image.new("RGBA", (4, 4), (255, 0, 0, 0))  # fully transparent red
        rgb, note = to_rgb(img, "#ffffff")
        self.assertEqual(rgb.getpixel((0, 0)), (255, 255, 255))
        self.assertEqual(note, "transparent areas filled with #ffffff")

    def test_half_transparent_blends(self):
        rgb, _ = to_rgb(Image.new("RGBA", (2, 2), (255, 0, 0, 128)), "#000000")
        self.assertEqual(rgb.getpixel((0, 0)), (128, 0, 0))

    def test_opaque_alpha_has_no_note(self):
        rgb, note = to_rgb(Image.new("RGBA", (2, 2), (10, 20, 30, 255)))
        self.assertEqual((rgb.getpixel((0, 0)), note), ((10, 20, 30), None))

    def test_palette_transparency(self):
        img = Image.new("P", (2, 2), 0)
        img.putpalette([255, 0, 0] + [0, 0, 0] * 255)
        img.info["transparency"] = 0
        rgb, note = to_rgb(img, "#00ff00")
        self.assertEqual(rgb.getpixel((0, 0)), (0, 255, 0))
        self.assertIsNotNone(note)

    def test_16_bit_grey_is_grey(self):
        img = Image.fromarray(np.full((2, 2), 32768, dtype=np.uint16))
        self.assertTrue(img.mode.startswith("I;16"))
        rgb, note = to_rgb(img)
        self.assertEqual((rgb.getpixel((0, 0)), note), ((128, 128, 128), None))

    def test_16_bit_png_file(self):
        path = os.path.join(os.environ.get("TEMP", "."), "luna_asset_g16_test.png")
        Image.fromarray(np.array([[0, 65535]], dtype=np.uint16)).save(path)
        try:
            with Image.open(path) as img:
                rgb, _ = to_rgb(img)
                self.assertEqual([rgb.getpixel((0, 0)), rgb.getpixel((1, 0))], [(0, 0, 0), (255, 255, 255)])
        finally:
            os.remove(path)

    def test_plain_rgb_untouched(self):
        img = stripes(8, 4)
        rgb, note = to_rgb(img)
        self.assertEqual((np.asarray(rgb) == np.asarray(img)).all(), True)
        self.assertIsNone(note)


class ConformTests(unittest.TestCase):
    def test_portrait_is_letterboxed_not_squeezed(self):
        land = torch.ones(1, 768, 1344, 3)
        port = torch.ones(1, 1344, 768, 3)
        batch = conform([land, port])
        self.assertEqual(tuple(batch.shape), (2, 768, 1344, 3))
        second = batch[1]
        # Fitted whole: 768 high -> 439 wide, centred; bars either side are black.
        self.assertEqual(float(second[:, :400].max()), 0.0)
        self.assertEqual(float(second[:, -400:].max()), 0.0)
        self.assertGreater(float(second[384, 672].min()), 0.99)

    def test_same_size_passes_through(self):
        a = torch.rand(1, 64, 96, 3)
        b = torch.rand(1, 64, 96, 3)
        self.assertTrue(torch.equal(conform([a, b])[1], b[0]))


if __name__ == "__main__":
    unittest.main()
