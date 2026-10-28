"""Luna-H3-Canvas: `preset` mode and the `multiple` grid.

    E:\\ComfyUI-Easy-Install\\python_embeded\\python.exe -m unittest discover -s tests -v

The JS mirror is checked against these lists by tests/check_canvas_js.mjs.
"""

import unittest

from test_canvas_seconds import h3_canvas

Canvas = h3_canvas.LunaMiniMaxH3Canvas
WIDE = "16:9 (widescreen)"


def run(ratio=WIDE, portrait=False, mode="preset", multiple="32",
        resolution="1344x768 (1.03 MP)", megapixels=1.03):
    return Canvas.execute(ratio, portrait, mode, megapixels, 5.0, True, 2,
                          multiple=multiple, resolution=resolution)


class PresetListTests(unittest.TestCase):
    def test_every_size_is_on_the_grid_and_in_range(self):
        for ratio, (rw, rh) in h3_canvas.ASPECT_RATIOS.items():
            for multiple in h3_canvas.MULTIPLES:
                m = int(multiple)
                sizes = h3_canvas.preset_sizes(rw, rh, m)
                self.assertTrue(sizes, (ratio, multiple))
                for w, h in sizes:
                    self.assertEqual((w % m, h % m), (0, 0), (ratio, w, h))
                    self.assertTrue(200_000 <= w * h <= 2_000_000, (ratio, w, h))
                    self.assertGreaterEqual(w, h)
                    # Nearest step on one edge: never more than one /32 step off the
                    # ratio (finer lists also hold the coarser grids' sizes).
                    self.assertLess(abs(w * rh - h * rw), 32 * max(rw, rh), (ratio, w, h))

    def test_model_canvas_is_on_every_32_list(self):
        for ratio, (rw, rh) in h3_canvas.ASPECT_RATIOS.items():
            self.assertIn(h3_canvas.h3_canvas(rw / rh), h3_canvas.preset_sizes(rw, rh, 32), ratio)

    def test_finer_grids_hold_more_sizes(self):
        for ratio in h3_canvas.ASPECT_RATIOS:
            l32, l16, l8 = (set(h3_canvas.preset_labels(ratio, m)) for m in ("32", "16", "8"))
            self.assertTrue(l32 < l16 < l8, ratio)  # changing the grid keeps your size
        # 1920x1080 is 2.07 MP, past the 2 MP top; 1824x1024 on /8 is in.
        self.assertNotIn("1920x1080 (2.07 MP)", h3_canvas.ALL_PRESETS)
        self.assertIn("1824x1024 (1.87 MP)", h3_canvas.preset_labels(WIDE, "8"))

    def test_default_and_combo_union(self):
        self.assertIn(h3_canvas.DEFAULT_PRESET, h3_canvas.preset_labels(WIDE, "32"))
        for ratio in h3_canvas.ASPECT_RATIOS:
            for m in h3_canvas.MULTIPLES:
                self.assertTrue(set(h3_canvas.preset_labels(ratio, m)) <= set(h3_canvas.ALL_PRESETS))

    def test_label_reads_its_size(self):
        self.assertEqual(h3_canvas.preset_label(1344, 768), "1344x768 (1.03 MP)")
        self.assertEqual(h3_canvas.preset_label(640, 352), "640x352 (0.23 MP)")


class PresetRunTests(unittest.TestCase):
    def test_preset_landscape_and_portrait(self):
        out = run()
        self.assertEqual(out[:2], (1344, 768))
        self.assertNotIn("H3 video makes", out[6])
        self.assertEqual(run(portrait=True)[:2], (768, 1344))

    def test_eight_grid_says_what_h3_makes(self):
        odd = next(l for l in h3_canvas.preset_labels(WIDE, "8")
                   if any(int(v) % 16 for v in l.split(" ")[0].split("x")))
        w, h = (int(v) for v in odd.split(" ")[0].split("x"))
        out = run(multiple="8", resolution=odd)
        self.assertEqual(out[:2], (w, h))
        self.assertIn(f"H3 video makes {w // 16 * 16}x{h // 16 * 16}", out[6])

    def test_label_off_the_list_refuses(self):
        with self.assertRaises(ValueError) as caught:
            run(ratio="1:1 (square)", resolution="1344x768 (1.03 MP)")
        self.assertIn("not a 1:1 (square) size on the /32 grid", str(caught.exception))

    def test_megapixels_follow_the_grid(self):
        for multiple in ("32", "16", "8"):
            w, h = run(mode="megapixels", multiple=multiple, megapixels=1.5)[:2]
            self.assertEqual((w % int(multiple), h % int(multiple)), (0, 0))
        self.assertEqual(run(mode="megapixels", megapixels=1.03)[:2], (1344, 768))

    def test_old_call_without_new_inputs(self):
        out = Canvas.execute(WIDE, False, "H3 canvas", 1.03, 5.0, True, 2)
        self.assertEqual(out[:3], (1344, 768, 124))


if __name__ == "__main__":
    unittest.main()
