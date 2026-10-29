import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from luna_recipes import check_values, load_recipes  # noqa: E402

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
        self.assertEqual(set(recipes), {DMAD})
        entry = recipes[DMAD]
        self.assertEqual(entry["strength"], 0.8)
        self.assertEqual(entry["total_steps"], 12)
        self.assertIsNone(entry["transition"])
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

        clean, quiet = check_values(values(high=DMAD, audio_shift=2.0))
        self.assertEqual(clean, [])
        self.assertEqual(quiet, [])

        _errors, off = check_values(values(high=DMAD, audio_shift=2.0, total=8, transition=5))
        self.assertEqual(off, [f"pass 2 {DMAD} recipe: total_steps 12, set 8"])


if __name__ == "__main__":
    unittest.main()
