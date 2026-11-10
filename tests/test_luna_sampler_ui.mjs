// Luna-H3-Sampler's frontend: the readout file and its place in the pack's help and collapse lists.
// The kit files' identity with Luna-Media-Tools is checked in test_registration.py.
//     node --test tests/test_luna_sampler_ui.mjs
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");

function code(name) {
    return readFileSync(join(root, "web", name), "utf8").split(/\r?\n/)
        .map((line) => line.replace(/\/\/.*$/, "")).join("\n");
}

test("pack theme lists Luna-H3-Sampler for help and collapse, tint stays off", () => {
    const src = code("luna_pack_theme.js");
    const calls = [...src.matchAll(/registerLuna(?:Help|Collapse)\(\s*app\s*,\s*\[([^\]]*)\]/g)];
    assert.equal(calls.length, 2);
    for (const call of calls) {
        assert.ok(call[1].includes('"Luna-H3-Sampler"'));
    }
    assert.ok(!src.includes("registerLunaTheme("));
});

test("readout file registers no second help, collapse or tint", () => {
    const src = code("luna_sampler.js");
    assert.ok(!/registerLuna(?:Help|Collapse|Theme)\(/.test(src));
    assert.match(src, /fetch\("\/luna_sampler\/recipes"\)/);
    assert.match(
        src,
        /pass 1: steps 1-\$\{transition\} \(low\) \/ pass 2: steps \$\{transition \+ 1\}-\$\{total\} \(high\)/,
    );
    assert.match(src, /recipe\.strength/);
    assert.match(src, /recipe\.total_steps/);
    assert.match(src, /recipe\.shift/);
    assert.match(src, /recipe\.audio_shift/);
});

// The readout file's functions, run outside ComfyUI: its two imports are replaced by stubs.
function samplerFunctions() {
    const src = readFileSync(join(root, "web", "luna_sampler.js"), "utf8")
        .replace(/^import .*$/gm, "");
    const make = new Function("app", "LUNA", `${src}\nreturn { migrateSingleScheduler, recipeNotes, scheduleLine, recipeFor, recipeToFlat, nodeRecipe, matchesRecipe, GEAR_INSET, num };`);
    return make({ registerExtension() {} }, {});
}

function fakeNode(names) {
    return { widgets: names.map((name) => ({ name, value: undefined })) };
}

const NEW_ORDER = [
    "lora_low", "strength_low", "shift_low", "audio_shift_low", "cfg_low", "scheduler_low",
    "lora_high", "strength_high", "shift_high", "audio_shift_high", "cfg_high", "scheduler_high",
    "total_steps", "transition", "highres_denoise", "noise_seed", "control_after_generate",
    "lowres_scale", "upscaler_model", "rho", "w_min", "w_max",
];

test("a workflow saved with one scheduler opens with it in both passes and every value in place", () => {
    const { migrateSingleScheduler } = samplerFunctions();
    // H3 Luna Sampler_Main.json, node 156, as saved on 2026-11-01.
    const saved = [
        "MiniMaxH3/pika_h3_flex_ref2va_v1_avg_rank_55_bf16.safetensors", 1, 12, 3, 1,
        "MiniMaxH3/pika_h3_flex_ref2va_v1_avg_rank_55_bf16.safetensors", 1, 12, 3, 1,
        "linear_quadratic", 8, 5, 1, 0, "randomize", 0.5, "minimax_h3_latent_upscaler_3d_fp16.safetensors", 0, 0.5, 1,
    ];
    const node = fakeNode(NEW_ORDER);
    assert.equal(migrateSingleScheduler(node, saved), true);
    const value = (name) => node.widgets.find((widget) => widget.name === name).value;
    assert.equal(value("scheduler_low"), "linear_quadratic");
    assert.equal(value("scheduler_high"), "linear_quadratic");
    assert.equal(value("cfg_low"), 1);
    assert.equal(value("lora_high"), saved[5]);
    assert.equal(value("total_steps"), 8);
    assert.equal(value("transition"), 5);
    assert.equal(value("noise_seed"), 0);
    assert.equal(value("control_after_generate"), "randomize");
    assert.equal(value("lowres_scale"), 0.5);
    assert.equal(value("upscaler_model"), "minimax_h3_latent_upscaler_3d_fp16.safetensors");
    assert.equal(value("w_max"), 1);
});

test("a workflow already saved with two schedulers, or an older layout, is left as it loaded", () => {
    const { migrateSingleScheduler } = samplerFunctions();
    const node = fakeNode(NEW_ORDER);
    const current = NEW_ORDER.map((name) => name);
    assert.equal(migrateSingleScheduler(node, current), false);
    // Luna-H3-Sampler test - Snow.json: 21 values, but saved before the upscaler widget existed.
    const older = ["lora", 0.8, 12, 3, 1, "lora", 0.8, 12, 3, 1, "simple", 12, 8, 1, 86, "fixed", 0.5, 0, 0.5, 1, 1];
    assert.equal(migrateSingleScheduler(node, older), false);
    assert.ok(node.widgets.every((widget) => widget.value === undefined));
});

test("each pass's recipe is checked against its own scheduler", () => {
    const { recipeNotes } = samplerFunctions();
    const values = {
        lora_low: "a.safetensors", lora_high: "b.safetensors", total_steps: 8, transition: 5,
        scheduler_low: "linear_quadratic", scheduler_high: "simple",
    };
    const node = { widgets: Object.entries(values).map(([name, value]) => ({ name, value })) };
    const notes = recipeNotes(node, {
        "a.safetensors": { scheduler: "linear_quadratic" },
        "b.safetensors": { scheduler: "linear_quadratic" },
    });
    assert.deepEqual(notes, ["pass 2 b recipe: scheduler linear_quadratic, set simple"]);
});

test("the LoRA lists keep their folders: only the closed widget is shortened", () => {
    const src = code("luna_sampler.js");
    assert.ok(!src.includes("getOptionLabel"));
    assert.match(src, /defineProperty\(widget, "_displayValue"/);
});

test("the drawn readout names the hand-over sigma, and 0 keeps the step split", () => {
    const { scheduleLine } = samplerFunctions();
    const node = (values) => ({ widgets: Object.entries(values).map(([name, value]) => ({ name, value })) });
    const base = { total_steps: 12, transition: 9, highres_denoise: 0.5 };
    assert.equal(scheduleLine(node({ ...base, handover_sigma: 0.6 })),
        "pass 1: steps 1-9 (low) / pass 2: steps 10-12 (high), hand-over at sigma 0.6");
    assert.equal(scheduleLine(node({ ...base, handover_sigma: 0 })),
        "pass 1: steps 1-9 (low) / pass 2: steps 10-12 (high), denoise 0.5");
});

// The recipe menu (plans/recipe-menu.md).
const NATURAL = {
    name: "Natural skin · 2 MP",
    low: { lora: "minimax_h3_DMAD_8step_full_lora_avg_rank_58_bf16.safetensors", strength: 1, sampler: "lcm", scheduler: "simple", shift: 12, audio_shift: 2, cfg: 1 },
    high: { lora: "minimax_h3_DMAD_4step_full_lora_avg_rank_39_bf16.safetensors", strength: 1, sampler: "er_sde", scheduler: "simple", shift: 12, audio_shift: 2, cfg: 1 },
    total_steps: 6, transition: 4, handover_sigma: 0.923077, lowres_scale: 0.7,
};

test("a LoRA in any folder finds its recipe by file name", () => {
    const { recipeFor } = samplerFunctions();
    const recipes = { "MiniMaxH3/d8.safetensors": { total_steps: 6 } };
    assert.equal(recipeFor("MiniMaxH3/d8.safetensors", recipes).total_steps, 6);
    assert.equal(recipeFor("other\\d8.safetensors", recipes).total_steps, 6);
    assert.equal(recipeFor("d8.safetensors", recipes).total_steps, 6);
    assert.equal(recipeFor("d9.safetensors", recipes), null);
    assert.equal(recipeFor("none", recipes), null);
});

test("Apply writes the found LoRA paths and every value; the node then matches the recipe", () => {
    const { recipeToFlat, nodeRecipe, matchesRecipe } = samplerFunctions();
    const where = { low: "MiniMaxH3/" + NATURAL.low.lora, high: "x/" + NATURAL.high.lora };
    const flat = recipeToFlat(NATURAL, where);
    assert.equal(flat.lora_low, where.low);
    assert.equal(flat.lora_high, where.high);
    assert.equal(flat.sampler_low, "lcm");
    assert.equal(flat.sampler_high, "er_sde");
    assert.equal(flat.handover_sigma, 0.923077);
    assert.equal(flat.audio_shift_high, 2);
    const names = Object.keys(flat).filter((name) => !name.startsWith("sampler_"));
    const node = fakeNode(names);
    for (const name of names) node.widgets.find((widget) => widget.name === name).value = flat[name];
    assert.equal(nodeRecipe(node).low.lora, NATURAL.low.lora);
    assert.equal(matchesRecipe(node, NATURAL), true);
    node.widgets.find((widget) => widget.name === "lowres_scale").value = 0.75;
    assert.equal(matchesRecipe(node, NATURAL), false);
});

test("the ⚙ sits one slot left of the chevron", () => {
    const { GEAR_INSET } = samplerFunctions();
    assert.equal(GEAR_INSET, 56);
    const src = code("luna_sampler.js");
    for (const route of ["/luna_sampler/pair_recipes", "/luna_sampler/recommend", "/luna_sampler/validate", "/luna_sampler/user_recipes"]) {
        assert.ok(src.includes(`"${route}"`), route);
    }
});

test("an empty or mistyped panel field is null, not 0", () => {
    const { num } = samplerFunctions();
    for (const bad of ["", "  ", null, undefined, "abc"]) assert.equal(num(bad), null, String(bad));
    assert.equal(num("0"), 0);
    assert.equal(num("0.923077"), 0.923077);
});
