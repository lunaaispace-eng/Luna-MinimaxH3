"""Per-LoRA recipes and the recipe menu (⚙) for Luna-H3-Sampler.

`luna_recipes.json` beside this module is keyed by the LoRA path ComfyUI lists; a
LoRA is matched by its file name, so the same file in another folder finds its
recipe. A null field means that source did not say. `check_values` refuses a
schedule or a strength that cannot run, and warns when an arm disagrees with its
LoRA's recipe.

`luna_pair_recipes.json` holds the tested recipes for a LoRA pair the menu ships.
The user's own recipes live in ComfyUI's user folder (`luna_sampler_recipes.json`),
so a pack update never overwrites them. `recommend` computes the Custom tab's values
from each LoRA's own sigma points.
"""

from __future__ import annotations

import json
import logging
import math
import os
from datetime import date
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
    "sigma_points",
)
PAIR_ARM_FIELDS = ("lora", "strength", "sampler", "scheduler", "shift", "audio_shift", "cfg")
USER_FILE = "luna_sampler_recipes.json"
SIGMA_TOLERANCE = 1e-3
TARGET_MP = 1.0  # pass 1 at about 1 MP (Peter, 2026-11-10, SPEC Decisions log)
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
ARM_FIELDS = {"strength", "shift", "audio_shift", "sampler", "scheduler", "cfg"}
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
        # A run built before each pass had its own scheduler gives one shared `scheduler`.
        if field == "scheduler" and arm.get(field) is None:
            return values.get(field)
        return arm.get(field)
    return values.get(field)


def check_values(values: dict, recipes: dict | None = None) -> tuple[list[str], list[str]]:
    """Return `(errors, warnings)` for one run.

    `values` has `total_steps`, `transition` and `passes` (pass 1, then pass 2). Each
    pass has `lora`, `strength`, `shift`, `audio_shift`, `cfg`, `scheduler` and
    `sampler` (the sampler name, or None when the object does not say). A shared
    `scheduler` in `values` stands in for a pass that has none. `none` is not a LoRA.
    An unknown file is a warning, not an error.
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
        recipe = recipe_for(lora, recipes)
        if recipe is None:
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


# ---------------------------------------------------------------- LoRA by file name


def file_name(path) -> str:
    """The file name without its folder: `MiniMaxH3/x.safetensors` -> `x.safetensors`."""
    return str(path or "").replace("\\", "/").rsplit("/", 1)[-1]


def recipe_for(lora, recipes: dict):
    """The per-LoRA recipe for `lora`, by exact key first, then by file name (Peter, 2026-11-10: "Yes, by file name")."""
    if not lora or lora == "none":
        return None
    for key in (lora, str(lora).replace("\\", "/")):
        item = recipes.get(key)
        if isinstance(item, dict):
            return item
    wanted = file_name(lora)
    for key, item in recipes.items():
        if isinstance(item, dict) and file_name(key) == wanted:
            return item
    return None


def resolve_lora(name, available) -> tuple[str | None, list[str]]:
    """`(path, matches)`: the first of ComfyUI's LoRA paths with this file name, and every one that has it."""
    wanted = file_name(name)
    matches = [path for path in available if file_name(path) == wanted]
    return (matches[0] if matches else None), matches


def available_loras() -> list[str]:
    """ComfyUI's LoRA list (every folder it knows, `extra_model_paths.yaml` included), with `/` separators."""
    try:
        import folder_paths

        return [name.replace("\\", "/") for name in folder_paths.get_filename_list("loras")]
    except Exception:
        return []


# ---------------------------------------------------------------- sigma points


def _computed_points(scheduler: str, steps: int, shift: float) -> list[float]:
    """ComfyUI's own `calculate_sigmas` for this scheduler and step count at `shift`."""
    import comfy.model_sampling
    import comfy.samplers

    class _Config:
        sampling_settings = {"shift": float(shift), "multiplier": 1.0}

    sampling = comfy.model_sampling.ModelSamplingDiscreteFlow(_Config())
    return [float(value) for value in comfy.samplers.calculate_sigmas(sampling, scheduler, int(steps))]


def sigma_points(recipe) -> list[float] | None:
    """A LoRA's own sigma points, 1 down to 0, or None when its recipe does not know them.

    `sigma_points` is either a list, or `{"scheduler", "steps"}` computed at the recipe's `shift`.
    """
    if not isinstance(recipe, dict):
        return None
    spec = recipe.get("sigma_points")
    if isinstance(spec, list):
        points = [float(value) for value in spec]
    elif isinstance(spec, dict) and spec.get("scheduler") and _as_int(spec.get("steps")):
        points = _computed_points(spec["scheduler"], _as_int(spec["steps"]), recipe.get("shift") or 1.0)
    else:
        return None
    if len(points) < 3 or any(not math.isfinite(p) for p in points):
        return None
    if any(later >= earlier for earlier, later in zip(points, points[1:])) or points[-1] != 0.0:
        return None
    return points


def grid_points(recipe) -> list[float] | None:
    """A LoRA's sigma points when a scheduler makes them (`{"scheduler", "steps"}`), else None."""
    if not isinstance(recipe, dict) or not isinstance(recipe.get("sigma_points"), dict):
        return None
    return sigma_points(recipe)


def points_scheduler(recipe):
    """The scheduler a LoRA's points come from, else its recipe's scheduler."""
    spec = recipe.get("sigma_points") if isinstance(recipe, dict) else None
    if isinstance(spec, dict) and spec.get("scheduler"):
        return spec["scheduler"]
    return recipe.get("scheduler") if isinstance(recipe, dict) else None


def lowres_for(width, height, target_mp: float = TARGET_MP) -> float | None:
    """`lowres_scale` that puts pass 1 at about `target_mp`, in steps of 0.05, inside the widget's [0.25, 1]."""
    try:
        area = float(width) * float(height)
    except (TypeError, ValueError):
        return None
    if not area > 0:
        return None
    scale = math.sqrt(target_mp * 1_000_000 / area)
    return min(1.0, max(0.25, round(round(scale / 0.05) * 0.05, 2)))


def _side(index):
    return f"pass {index}"


def recommend(lora_low, lora_high, width=None, height=None, recipes: dict | None = None) -> dict:
    """The Custom tab's recommendation for a LoRA pair and a final size.

    Hand-over: the lowest sigma point both LoRAs share that leaves pass 2 at least two of its own steps. Steps: pass 1
    runs its own points above the hand-over, pass 2 its own points from it to 0. Each pass takes the scheduler its
    points come from and its recipe's shifts and CFG. With no shared point, or a LoRA with no known points, the
    hand-over is 0 (the step split) and the steps come from pass 1's recipe. Returns `{"values", "why", "notes",
    "shared"}`; `values` holds only what is known.
    """
    if recipes is None:
        recipes = load_recipes()
    values, why, notes = {}, {}, []
    arms = (("low", 1, lora_low), ("high", 2, lora_high))
    found = {}
    for side, index, lora in arms:
        recipe = recipe_for(lora, recipes)
        found[side] = recipe
        if recipe is None:
            if lora and lora != "none":
                notes.append(f"{_side(index)} {file_name(lora)}: no recipe, its values stay as set")
            continue
        for field in ("strength", "shift", "audio_shift", "cfg"):
            if recipe.get(field) is not None:
                values[f"{field}_{side}"] = recipe[field]
        scheduler = points_scheduler(recipe)
        if scheduler:
            values[f"scheduler_{side}"] = scheduler
        if recipe.get("sampler"):
            values[f"sampler_{side}"] = recipe["sampler"]

    # Only points a scheduler makes can place the hand-over: the node builds each pass with a scheduler, so a custom
    # list (Pika's knots) would be missed (pika.md: "6 steps" in an ordinary scheduler is a different schedule).
    low_points = grid_points(found["low"])
    high_points = grid_points(found["high"])
    shared = []
    choice = None
    if low_points and high_points:
        shared = [p for p in low_points if 0.0 < p < 1.0 and any(abs(p - q) <= SIGMA_TOLERANCE for q in high_points)]
        for point in sorted(shared):
            steps_low = sum(1 for p in low_points if p > point + SIGMA_TOLERANCE)
            steps_high = sum(1 for q in high_points if q < point - SIGMA_TOLERANCE)
            if steps_high >= 2 and steps_low >= 1:
                choice = (point, steps_low, steps_high)
                break
    if choice:
        point, steps_low, steps_high = choice
        values["handover_sigma"] = round(point, 6)
        values["transition"] = steps_low
        values["total_steps"] = steps_low + steps_high
        shown = " ".join(f"{p:.3f}".lstrip("0") for p in shared)
        why["handover_sigma"] = f"Both LoRAs know {shown}; the lowest that leaves pass 2 two of its own steps."
        why["steps"] = (f"Pass 1: its own points 1 -> {point:.3f} ({steps_low}). Pass 2: its own points {point:.3f} -> 0 "
                        f"({steps_high}). total_steps {steps_low + steps_high}, transition {steps_low}.")
    else:
        values["handover_sigma"] = 0.0
        listed = [file_name(lora) for side, _index, lora in arms
                  if lora and lora != "none" and grid_points(found[side]) is None and sigma_points(found[side])]
        missing = [file_name(lora) for side, _index, lora in arms
                   if lora and lora != "none" and sigma_points(found[side]) is None]
        if listed:
            notes.append(f"{', '.join(listed)}: its sigma points are a custom list no scheduler makes, "
                         "so the step split (hand-over 0)")
        if missing:
            notes.append(f"{', '.join(missing)}: no known sigma points, so the step split (hand-over 0)")
        elif low_points and high_points:
            notes.append("No sigma point both LoRAs share leaves pass 2 two steps, so the step split (hand-over 0)")
        base = found["low"] or found["high"]
        if base and _as_int(base.get("total_steps")) and _as_int(base.get("transition")):
            values["total_steps"] = _as_int(base["total_steps"])
            values["transition"] = _as_int(base["transition"])
            why["steps"] = f"From the {'pass 1' if found['low'] else 'pass 2'} LoRA's recipe."
        why["handover_sigma"] = "Step split: pass 1 runs the first transition steps of one schedule."
    scale = lowres_for(width, height)
    if scale is not None:
        values["lowres_scale"] = scale
        w, h = round(float(width) * scale), round(float(height) * scale)
        why["lowres_scale"] = f"Pass 1 at {w}x{h}, about {w * h / 1_000_000:.2f} MP of {int(width)}x{int(height)}."
    return {"values": values, "why": why, "notes": notes, "shared": shared}


# ---------------------------------------------------------------- pair recipes


def load_pair_recipes(path=None) -> list:
    """The shipped pair recipes, `luna_pair_recipes.json` beside this module."""
    if path is None:
        path = Path(__file__).with_name("luna_pair_recipes.json")
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("Luna-H3-Sampler: pair recipes file must be a list")
    return data


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def check_pair(recipe) -> list[str]:
    """Errors that stop a pair recipe from being applied or saved. Empty means it can run."""
    if not isinstance(recipe, dict):
        return ["a recipe must be an object"]
    errors = []
    name = recipe.get("name")
    if not isinstance(name, str) or not name.strip():
        errors.append("the recipe has no name")
    for side, index in (("low", 1), ("high", 2)):
        arm = recipe.get(side)
        if not isinstance(arm, dict):
            errors.append(f"pass {index} is missing")
            continue
        if not isinstance(arm.get("lora"), str) or not arm["lora"]:
            errors.append(f"pass {index} has no LoRA")
        if not _strength_ok(arm.get("strength")):
            errors.append(f"pass {index} strength {show_value('strength', arm.get('strength'))} is outside [0, 2]")
        for field in ("sampler", "scheduler"):
            if not isinstance(arm.get(field), str) or not arm[field]:
                errors.append(f"pass {index} has no {field}")
        for field in ("shift", "audio_shift"):
            if not _number(arm.get(field)) or not float(arm[field]) > 0:
                errors.append(f"pass {index} {field} must be above 0")
        if not _number(arm.get("cfg")) or float(arm["cfg"]) < 0:
            errors.append(f"pass {index} cfg must be 0 or more")
    total = _as_int(recipe.get("total_steps"))
    transition = _as_int(recipe.get("transition"))
    if total is None or transition is None or not 1 <= transition <= total - 1:
        errors.append(f"transition {recipe.get('transition')} is not in [1, {total - 1 if total else '?'}]")
    handover = recipe.get("handover_sigma")
    if not _number(handover) or not 0.0 <= float(handover) <= 0.95:
        errors.append("handover_sigma must be in [0, 0.95]")
    scale = recipe.get("lowres_scale")
    if not _number(scale) or not 0.25 <= float(scale) <= 1.0:
        errors.append("lowres_scale must be in [0.25, 1]")
    return errors


def pair_to_values(recipe) -> dict:
    """The `check_values` shape for a pair recipe."""
    passes = []
    for side in ("low", "high"):
        arm = recipe.get(side) or {}
        passes.append({field: arm.get(field) for field in PAIR_ARM_FIELDS})
    return {"total_steps": recipe.get("total_steps"), "transition": recipe.get("transition"), "passes": passes}


def locate(recipe, available) -> dict:
    """Where each of the recipe's LoRAs is in ComfyUI's list: `{"low": path|None, "high": ..., "missing", "duplicates"}`."""
    out = {"missing": [], "duplicates": []}
    for side in ("low", "high"):
        lora = (recipe.get(side) or {}).get("lora")
        if not lora or lora == "none":
            out[side] = "none"
            continue
        path, matches = resolve_lora(lora, available)
        out[side] = path
        if path is None and file_name(lora) not in out["missing"]:
            out["missing"].append(file_name(lora))
        if len(matches) > 1:
            out["duplicates"].append(f"{file_name(lora)} is in {len(matches)} folders; using {path}")
    return out


def validate_pair(recipe, recipes: dict | None = None, available=None) -> tuple[list[str], list[str]]:
    """`(errors, warnings)` for the panel's Validate: the recipe's shape, its LoRAs on disk, and `check_values`."""
    errors = check_pair(recipe)
    if errors:
        return errors, []
    found = locate(recipe, available_loras() if available is None else available)
    errors.extend(f"missing: {name}" for name in found["missing"])
    more_errors, warnings = check_values(pair_to_values(recipe), recipes)
    errors.extend(more_errors)
    warnings = list(found["duplicates"]) + warnings
    return errors, warnings


# ---------------------------------------------------------------- the user's recipes


def user_recipes_path() -> Path:
    """`<ComfyUI>/user/default/luna_sampler_recipes.json`, outside the pack."""
    import folder_paths

    return Path(folder_paths.get_user_directory()) / "default" / USER_FILE


def load_user_recipes(path=None) -> list:
    """The user's saved recipes. No file is an empty list; a file that cannot be read raises, it is never half-read."""
    path = Path(path) if path is not None else user_recipes_path()
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list) or any(check_pair(item) for item in data):
        raise ValueError(f"Luna-H3-Sampler: {path} is not a list of recipes")
    return data


def _write_user_recipes(items: list, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with open(temp, "w", encoding="utf-8") as handle:
        json.dump(items, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def change_user_recipes(action: str, payload: dict, path=None, shipped_names=None) -> list:
    """Save, rename or delete one of the user's recipes; returns the new list. Raises ValueError with the reason."""
    path = Path(path) if path is not None else user_recipes_path()
    items = load_user_recipes(path)
    shipped = set(shipped_names if shipped_names is not None else (r.get("name") for r in load_pair_recipes()))
    names = {item["name"] for item in items}
    if action == "save":
        recipe = dict(payload.get("recipe") or {})
        recipe["name"] = str(recipe.get("name") or "").strip()
        errors = check_pair(recipe)
        if errors:
            raise ValueError("; ".join(errors))
        if recipe["name"] in shipped or recipe["name"] in names:
            raise ValueError(f"a recipe called {recipe['name']} already exists")
        recipe["label"] = "MINE"
        recipe["date"] = recipe.get("date") or date.today().isoformat()
        for key in ("tested_on", "verdict"):
            recipe[key] = str(recipe.get(key) or "").strip()
        items.append(recipe)
    elif action == "rename":
        old, new = payload.get("name"), str(payload.get("new_name") or "").strip()
        if old not in names:
            raise ValueError(f"no saved recipe called {old}")
        if not new:
            raise ValueError("the new name is empty")
        if new != old and (new in shipped or new in names):
            raise ValueError(f"a recipe called {new} already exists")
        for item in items:
            if item["name"] == old:
                item["name"] = new
    elif action == "delete":
        name = payload.get("name")
        if name not in names:
            raise ValueError(f"no saved recipe called {name}")
        items = [item for item in items if item["name"] != name]
    else:
        raise ValueError(f"unknown action {action}")
    _write_user_recipes(items, path)
    return items


# ---------------------------------------------------------------- routes


def register_routes():
    """The recipe routes. Safe to call once.

    GET /luna_sampler/recipes: the per-LoRA recipes. GET /luna_sampler/pair_recipes: shipped and saved recipes with
    where their LoRAs are. POST /luna_sampler/recommend, /luna_sampler/validate, /luna_sampler/user_recipes.
    """
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

    def _bad(message, status=400):
        return web.json_response({"error": str(message)}, status=status)

    async def _body(request):
        try:
            data = await request.json()
        except Exception:
            return None
        return data if isinstance(data, dict) else None

    def _menu():
        available = available_loras()
        shipped = load_pair_recipes()
        try:
            user, user_error = load_user_recipes(), None
        except Exception as exc:  # a broken file is reported, never overwritten
            user, user_error = [], str(exc)
        return {
            "shipped": [dict(item, where=locate(item, available)) for item in shipped],
            "user": [dict(item, where=locate(item, available)) for item in user],
            "user_error": user_error,
        }

    @server.routes.get("/luna_sampler/recipes")
    async def luna_sampler_recipes(_request):
        return web.json_response(load_recipes())

    @server.routes.get("/luna_sampler/pair_recipes")
    async def luna_sampler_pair_recipes(_request):
        try:
            return web.json_response(_menu())
        except Exception as exc:
            return _bad(exc, 500)

    @server.routes.post("/luna_sampler/recommend")
    async def luna_sampler_recommend(request):
        data = await _body(request)
        if data is None:
            return _bad("expected a JSON object")
        try:
            return web.json_response(
                recommend(data.get("lora_low"), data.get("lora_high"), data.get("width"), data.get("height"))
            )
        except Exception as exc:
            return _bad(exc, 500)

    @server.routes.post("/luna_sampler/validate")
    async def luna_sampler_validate(request):
        data = await _body(request)
        if data is None:
            return _bad("expected a JSON object")
        errors, warnings = validate_pair(data.get("recipe"))
        return web.json_response({"errors": errors, "warnings": warnings})

    @server.routes.post("/luna_sampler/user_recipes")
    async def luna_sampler_user_recipes(request):
        data = await _body(request)
        if data is None:
            return _bad("expected a JSON object")
        try:
            change_user_recipes(str(data.get("action")), data)
        except ValueError as exc:
            return _bad(exc)
        except Exception as exc:
            return _bad(exc, 500)
        return web.json_response(_menu())

    _ROUTES_READY = True
