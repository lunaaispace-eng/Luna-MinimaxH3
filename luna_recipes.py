"""Per-LoRA recipes for Luna-H3-Sampler.

The JSON file beside this module is keyed by the LoRA file name ComfyUI lists.
A null field means that source did not say. `check_values` refuses a schedule or a
strength that cannot run, and warns when an arm disagrees with its LoRA's recipe.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

RECIPE_FIELDS = (
    "strength",
    "total_steps",
    "transition",
    "shift",
    "audio_shift",
    "sampler",
    "scheduler",
    "cfg",
    "source",
    "label",
)
# Compared in this order. The first five numbers and the sampler/scheduler/cfg are the run.
COMPARE_FIELDS = (
    "strength",
    "total_steps",
    "transition",
    "shift",
    "audio_shift",
    "sampler",
    "scheduler",
    "cfg",
)
ARM_FIELDS = {"strength", "shift", "audio_shift", "sampler", "cfg"}
_FLOAT_FIELDS = {"strength", "shift", "audio_shift", "cfg"}

_ROUTES_READY = False


def show_value(field, value):
    """The spelling used in recipe warnings. Whole float fields keep one decimal."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return str(value)
    number = float(value)
    if field not in _FLOAT_FIELDS and number.is_integer():
        return str(int(number))
    if number.is_integer():
        return f"{number:.1f}"
    return f"{number:.4f}".rstrip("0").rstrip(".")


def load_recipes(path=None) -> dict:
    """The recipe object. `path` defaults to `luna_recipes.json` beside this module."""
    if path is None:
        path = Path(__file__).with_name("luna_recipes.json")
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("Luna-H3-Sampler: recipes file must be an object keyed by LoRA file name")
    return data


def _as_int(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, float) and not value.is_integer():
        return None
    return int(value)


def _strength_ok(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return 0.0 <= float(value) <= 2.0


def _equal(left, right):
    if isinstance(left, bool) or isinstance(right, bool):
        return left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return float(left) == float(right)
    return left == right


def _actual(values, arm, field):
    if field in ARM_FIELDS:
        return arm.get(field)
    return values.get(field)


def check_values(values: dict, recipes: dict | None = None) -> tuple[list[str], list[str]]:
    """Return `(errors, warnings)` for one run.

    `values` has `total_steps`, `transition`, `scheduler` and `passes` (pass 1, then
    pass 2). Each pass has `lora`, `strength`, `shift`, `audio_shift`, `cfg` and
    `sampler` (the sampler name, or None when the object does not say). `none` is
    not a LoRA. An unknown file is a warning, not an error.
    """
    if recipes is None:
        recipes = load_recipes()
    errors = []
    warnings = []
    total = _as_int(values.get("total_steps"))
    transition = _as_int(values.get("transition"))
    if total is None or transition is None or not 1 <= transition <= total - 1:
        limit = total - 1 if total is not None else "?"
        shown = transition if transition is not None else values.get("transition")
        errors.append(f"transition {shown} is not in [1, {limit}]")

    for index, arm in enumerate(values.get("passes") or [], start=1):
        strength = arm.get("strength")
        if not _strength_ok(strength):
            errors.append(f"pass {index} strength {show_value('strength', strength)} is outside [0, 2]")
        lora = arm.get("lora") or "none"
        if lora == "none":
            continue
        recipe = recipes.get(lora)
        if recipe is None:
            recipe = recipes.get(str(lora).replace("\\", "/"))
        if not isinstance(recipe, dict):
            warnings.append(f"pass {index} {lora}: no recipe")
            continue
        for field in COMPARE_FIELDS:
            wanted = recipe.get(field)
            if wanted is None:
                continue
            actual = _actual(values, arm, field)
            if actual is None or _equal(actual, wanted):
                continue
            warnings.append(
                f"pass {index} {lora} recipe: {field} {show_value(field, wanted)}, set {show_value(field, actual)}"
            )
    return errors, warnings


def register_routes():
    """GET /luna_sampler/recipes returns the recipe JSON. Safe to call once."""
    global _ROUTES_READY
    if _ROUTES_READY:
        return
    try:
        from aiohttp import web
        from server import PromptServer
    except Exception as exc:
        logging.warning("[Luna-H3-Sampler] recipes route not registered: %s", exc)
        return
    server = PromptServer.instance
    if server is None:
        return

    @server.routes.get("/luna_sampler/recipes")
    async def luna_sampler_recipes(_request):
        return web.json_response(load_recipes())

    _ROUTES_READY = True
