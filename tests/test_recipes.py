import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from luna_recipes import (  # noqa: E402
    RECIPE_FIELDS,
    change_user_recipes,
    check_pair,
    check_values,
    file_name,
    load_pair_recipes,
    load_recipes,
    load_user_recipes,
    locate,
    lowres_for,
    pair_to_values,
    recipe_for,
    recommend,
    validate_pair,
)

DMAD = "MiniMaxH3/minimax_h3_DMAD_4step_full_lora_avg_rank_39_bf16.safetensors"


def values(low="none", high="none", total=12, transition=8, scheduler="simple", **arm):
    def one(lora):
        base = {
            "lora": lora,
            "strength": 0.8,
            "shift": 12.0,
            "audio_shift": 3.0,
            "sampler": "er_sde",
            "cfg": 1.0,
        }
        base.update(arm)
        return base

    return {
        "total_steps": total,
        "transition": transition,
        "scheduler": scheduler,
        "passes": [one(low), one(high)],
    }


class RecipeTests(unittest.TestCase):
    def test_load_seed_entry(self):
        recipes = load_recipes()
        self.assertIn(DMAD, recipes)
        self.assertEqual(len(recipes), 6)
        for name, item in recipes.items():
            self.assertEqual(set(item), set(RECIPE_FIELDS), name)
            self.assertLess(item["transition"], item["total_steps"], name)
        entry = recipes[DMAD]
        self.assertEqual(entry["strength"], 1.0)
        self.assertEqual(entry["total_steps"], 6)
        self.assertEqual(entry["transition"], 4)
        self.assertEqual(entry["shift"], 12.0)
        self.assertEqual(entry["audio_shift"], 2.0)
        self.assertEqual(entry["sampler"], "er_sde")
        self.assertEqual(entry["scheduler"], "simple")
        self.assertEqual(entry["cfg"], 1.0)
        self.assertIn("audio 2", entry["source"])
        self.assertEqual(entry["label"], "TESTED")
        self.assertEqual(load_recipes().keys(), load_recipes(Path(__file__).resolve().parents[1] / "luna_recipes.json").keys())

    def test_errors_for_transition_and_strength(self):
        errors, warnings = check_values(values(total=12, transition=12, strength=2.5))
        self.assertEqual(errors, [
            "transition 12 is not in [1, 11]",
            "pass 1 strength 2.5 is outside [0, 2]",
            "pass 2 strength 2.5 is outside [0, 2]",
        ])
        self.assertEqual(warnings, [])

        for transition in (0, 4):
            with self.subTest(transition=transition):
                bad, _ = check_values(values(total=4, transition=transition))
                self.assertTrue(any(item.startswith("transition") for item in bad))

        for strength in (0.0, 2.0):
            with self.subTest(strength=strength):
                ok, _ = check_values(values(strength=strength))
                self.assertEqual(ok, [])

    def test_warning_wording(self):
        recipes = {DMAD: {"audio_shift": 2.0, "strength": 0.8, "transition": None}}
        _errors, warnings = check_values(values(high=DMAD, audio_shift=3.0), recipes)
        self.assertEqual(warnings, [f"pass 2 {DMAD} recipe: audio_shift 2.0, set 3.0"])

    def test_unknown_lora_and_a_matching_run(self):
        _errors, warnings = check_values(values(low="missing.safetensors", high="none"))
        self.assertEqual(warnings, ["pass 1 missing.safetensors: no recipe"])

        clean, quiet = check_values(values(high=DMAD, audio_shift=2.0, strength=1.0, total=6, transition=4))
        self.assertEqual(clean, [])
        self.assertEqual(quiet, [])

        _errors, off = check_values(values(high=DMAD, audio_shift=2.0, strength=1.0, total=8, transition=4))
        self.assertEqual(off, [f"pass 2 {DMAD} recipe: total_steps 6, set 8"])

    def test_lora_in_another_folder_finds_its_recipe(self):
        recipes = load_recipes()
        for name in (file_name(DMAD), "other/" + file_name(DMAD), "a\\b\\" + file_name(DMAD)):
            with self.subTest(name=name):
                self.assertIs(recipe_for(name, recipes), recipes[DMAD])
                _errors, warnings = check_values(values(high=name, audio_shift=3.0, strength=1.0, total=6, transition=4))
                self.assertEqual(warnings, [f"pass 2 {name} recipe: audio_shift 2.0, set 3.0"])
        self.assertIsNone(recipe_for("none", recipes))
        self.assertIsNone(recipe_for("nothing.safetensors", recipes))


# MEASURED 2026-11-10 with ComfyUI's calculate_sigmas at shift 12 (plans/recipe-menu.md).
DMAD8_POINTS = [1.0, 0.9882, 0.973, 0.9524, 0.9231, 0.878, 0.8, 0.6316, 0.0]
DMAD4_POINTS = [1.0, 0.973, 0.9231, 0.8, 0.0]


# Scheduler-made points without ComfyUI: each fake scheduler name returns its list (the real one is in test_two_pass.py).
FAKE_GRIDS = {"g8": DMAD8_POINTS, "g4": DMAD4_POINTS, "ga": [1.0, 0.95, 0.5, 0.0], "gb": [1.0, 0.9, 0.3, 0.0]}


def listed(grid, **extra):
    spec = {"scheduler": grid, "steps": len(FAKE_GRIDS[grid]) - 1} if grid else None
    base = {"strength": 1.0, "total_steps": 6, "transition": 4, "shift": 12.0, "audio_shift": 2.0,
            "sampler": "lcm", "scheduler": "simple", "cfg": 1.0, "sigma_points": spec}
    base.update(extra)
    return base


@patch("luna_recipes._computed_points", lambda scheduler, steps, shift: list(FAKE_GRIDS[scheduler]))
class RecommendTests(unittest.TestCase):
    def test_dmad8_to_dmad4_at_2mp_is_todays_recipe(self):
        recipes = {"a/d8.safetensors": listed("g8"), "d4.safetensors": listed("g4", sampler="er_sde")}
        out = recommend("x/d8.safetensors", "d4.safetensors", 1920, 1088, recipes)
        v = out["values"]
        self.assertEqual(v["handover_sigma"], 0.9231)
        self.assertEqual((v["total_steps"], v["transition"]), (6, 4))
        self.assertEqual(v["lowres_scale"], 0.7)
        self.assertEqual((v["sampler_low"], v["sampler_high"]), ("lcm", "er_sde"))
        self.assertEqual((v["shift_low"], v["audio_shift_high"]), (12.0, 2.0))
        self.assertEqual((v["scheduler_low"], v["scheduler_high"]), ("g8", "g4"))
        self.assertEqual(out["shared"], [0.973, 0.9231, 0.8])
        self.assertEqual(out["notes"], [])
        self.assertIn("1344x762", out["why"]["lowres_scale"])

    def test_unknown_points_fall_back_to_the_step_split(self):
        recipes = {"d8.safetensors": listed("g8"), "lx.safetensors": listed(None, total_steps=12, transition=9)}
        out = recommend("lx.safetensors", "d8.safetensors", 1920, 1088, recipes)
        self.assertEqual(out["values"]["handover_sigma"], 0.0)
        self.assertEqual((out["values"]["total_steps"], out["values"]["transition"]), (12, 9))
        self.assertEqual(out["notes"], ["lx.safetensors: no known sigma points, so the step split (hand-over 0)"])

        none = recommend("new.safetensors", "none", None, None, recipes)
        self.assertEqual(none["notes"][0], "pass 1 new.safetensors: no recipe, its values stay as set")
        self.assertNotIn("lowres_scale", none["values"])

    def test_no_shared_point(self):
        recipes = {"a.safetensors": listed("ga"), "b.safetensors": listed("gb")}
        out = recommend("a.safetensors", "b.safetensors", 1280, 720, recipes)
        self.assertEqual(out["values"]["handover_sigma"], 0.0)
        self.assertEqual(out["shared"], [])
        self.assertTrue(out["notes"][0].startswith("No sigma point both LoRAs share"))

    def test_a_custom_list_never_places_the_hand_over(self):
        # Pika's knots are a list no scheduler makes: the node would miss them (review 2026-11-10).
        recipes = load_recipes()
        pika = "pika_h3_flex_ref2va_v1_avg_rank_55_bf16.safetensors"
        out = recommend(pika, "pika_h3_flex_fl2va_v1_bf16.safetensors", 1920, 1088, recipes)
        v = out["values"]
        self.assertEqual(v["handover_sigma"], 0.0)
        self.assertEqual((v["total_steps"], v["transition"]), (8, 5))
        self.assertEqual((v["scheduler_low"], v["sampler_low"]), ("linear_quadratic", "lcm"))
        self.assertTrue(out["notes"][0].endswith("a custom list no scheduler makes, so the step split (hand-over 0)"))

    def test_lowres_for(self):
        self.assertEqual(lowres_for(1920, 1088), 0.7)
        self.assertEqual(lowres_for(1344, 768), 1.0)
        self.assertEqual(lowres_for(3840, 2176), 0.35)
        self.assertEqual(lowres_for(8000, 8000), 0.25)
        self.assertIsNone(lowres_for(0, 1088))
        self.assertIsNone(lowres_for(None, 1088))


AVAILABLE = [
    "MiniMaxH3/minimax_h3_DMAD_8step_full_lora_avg_rank_58_bf16.safetensors",
    "minimax_h3_DMAD_4step_full_lora_avg_rank_39_bf16.safetensors",
    "MiniMaxH3/pika_h3_flex_ref2va_v1_avg_rank_55_bf16.safetensors",
    "MiniMaxH3/pika_h3_flex_fl2va_v1_bf16.safetensors",
    "old/pika_h3_flex_fl2va_v1_bf16.safetensors",
]


class PairRecipeTests(unittest.TestCase):
    def test_shipped_recipes_are_valid_and_match_their_lora_recipes(self):
        pairs = load_pair_recipes()
        self.assertEqual([p["name"] for p in pairs], ["Natural skin · 2 MP", "Smooth · 2 MP", "Fast · lightx 12 / 9"])
        for pair in pairs:
            with self.subTest(pair=pair["name"]):
                self.assertEqual(check_pair(pair), [])
                self.assertEqual(check_values(pair_to_values(pair)), ([], []))
                for side in ("low", "high"):
                    self.assertNotIn("/", pair[side]["lora"])
        natural = pairs[0]
        self.assertEqual((natural["total_steps"], natural["transition"]), (6, 4))
        self.assertEqual((natural["handover_sigma"], natural["lowres_scale"]), (0.923077, 0.7))
        self.assertEqual((natural["low"]["sampler"], natural["high"]["sampler"]), ("lcm", "er_sde"))

    def test_locate_by_file_name(self):
        natural, smooth, fast = load_pair_recipes()
        where = locate(natural, AVAILABLE)
        self.assertEqual(where["low"], AVAILABLE[0])
        self.assertEqual(where["high"], AVAILABLE[1])
        self.assertEqual((where["missing"], where["duplicates"]), ([], []))
        two = locate(smooth, AVAILABLE)
        self.assertEqual(two["high"], AVAILABLE[3])
        self.assertEqual(len(two["duplicates"]), 1)
        gone = locate(fast, AVAILABLE)
        self.assertIsNone(gone["low"])
        self.assertEqual(gone["missing"], [fast["low"]["lora"]])
        errors, _warnings = validate_pair(fast, available=AVAILABLE)
        self.assertEqual(errors, [f"missing: {fast['low']['lora']}"])
        self.assertEqual(validate_pair(natural, available=AVAILABLE), ([], []))

    def test_check_pair_errors(self):
        bad = json.loads(json.dumps(load_pair_recipes()[0]))
        bad["name"] = " "
        bad["low"]["strength"] = 3
        bad["transition"] = 6
        bad["handover_sigma"] = 0.99
        bad["lowres_scale"] = 0.1
        self.assertEqual(check_pair(bad), [
            "the recipe has no name",
            "pass 1 strength 3.0 is outside [0, 2]",
            "transition 6 is not in [1, 5]",
            "handover_sigma must be in [0, 0.95]",
            "lowres_scale must be in [0.25, 1]",
        ])


class UserRecipeTests(unittest.TestCase):
    def setUp(self):
        self.folder = Path(tempfile.mkdtemp())
        self.path = self.folder / "default" / "luna_sampler_recipes.json"
        self.shipped = ["Natural skin · 2 MP"]

    def tearDown(self):
        for item in sorted(self.folder.rglob("*"), reverse=True):
            item.rmdir() if item.is_dir() else item.unlink()
        self.folder.rmdir()

    def mine(self, name="Close-up · 2 MP"):
        recipe = json.loads(json.dumps(load_pair_recipes()[0]))
        recipe.update(name=name, tested_on="one singer, close-up, 5 s", verdict="truer face")
        return recipe

    def test_round_trip_rename_delete(self):
        self.assertEqual(load_user_recipes(self.path), [])
        items = change_user_recipes("save", {"recipe": self.mine()}, self.path, self.shipped)
        self.assertEqual(len(items), 1)
        again = load_user_recipes(self.path)
        self.assertEqual(again, items)
        saved = again[0]
        self.assertEqual(saved["label"], "MINE")
        self.assertEqual((saved["total_steps"], saved["handover_sigma"]), (6, 0.923077))
        self.assertEqual(saved["low"], load_pair_recipes()[0]["low"])
        self.assertTrue(saved["date"])
        renamed = change_user_recipes("rename", {"name": "Close-up · 2 MP", "new_name": "Close"}, self.path, self.shipped)
        self.assertEqual(renamed[0]["name"], "Close")
        self.assertEqual(change_user_recipes("delete", {"name": "Close"}, self.path, self.shipped), [])
        self.assertEqual(load_user_recipes(self.path), [])

    def test_refusals(self):
        with self.assertRaisesRegex(ValueError, "already exists"):
            change_user_recipes("save", {"recipe": self.mine("Natural skin · 2 MP")}, self.path, self.shipped)
        change_user_recipes("save", {"recipe": self.mine()}, self.path, self.shipped)
        with self.assertRaisesRegex(ValueError, "already exists"):
            change_user_recipes("save", {"recipe": self.mine()}, self.path, self.shipped)
        broken = self.mine("Broken")
        broken["transition"] = 0
        with self.assertRaisesRegex(ValueError, "transition"):
            change_user_recipes("save", {"recipe": broken}, self.path, self.shipped)
        with self.assertRaisesRegex(ValueError, "no saved recipe"):
            change_user_recipes("delete", {"name": "nope"}, self.path, self.shipped)

    def test_a_bad_file_is_refused_not_overwritten(self):
        self.path.parent.mkdir(parents=True)
        self.path.write_text('[{"name": "half"}]', encoding="utf-8")
        with self.assertRaises(ValueError):
            load_user_recipes(self.path)
        with self.assertRaises(ValueError):
            change_user_recipes("save", {"recipe": self.mine()}, self.path, self.shipped)
        self.assertEqual(self.path.read_text(encoding="utf-8"), '[{"name": "half"}]')


if __name__ == "__main__":
    unittest.main()
