"""Prints every preset list of h3_canvas.py as JSON, for tests/check_canvas_js.mjs."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from test_canvas_seconds import h3_canvas  # noqa: E402  (imports with comfy_api stubbed)

print(json.dumps({f"{r}|{m}": h3_canvas.preset_labels(r, m)
                  for r in h3_canvas.ASPECT_RATIOS for m in h3_canvas.MULTIPLES}))
