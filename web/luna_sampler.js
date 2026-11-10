// Luna-H3-Sampler — the live schedule readout and the recipe menu (⚙).
//
// The ⓘ and the chevron are registered with the pack's other nodes in
// luna_pack_theme.js. Title-bar tint stays off: ComfyUI's header stays, and the
// readout below uses the Luna palette.
//
// The readout is drawn, not sampled. It says which steps are low and which are
// high, and it checks the widgets against GET /luna_sampler/recipes. Picking a
// LoRA fills that file's strength, total steps and shifts when the recipe has them.
// Recipes are matched by the LoRA's file name, whatever folder it sits in.
//
// The ⚙ beside the chevron opens the recipe menu (plans/recipe-menu.md). Its look
// and its placement (pinned to the node's right edge, panning and zooming with it,
// closed by ⚙, ✕, Done or Escape) follow Luna-LLMPrompt's settings panel
// (Luna-LLM-Tools web/llm_prompt_panel.mjs, itself after Pixaroma's node_panel.mjs,
// MIT, see THIRD_PARTY_NOTICES.md). The panel writes the node's own widgets and the
// sampler_name of the KSamplerSelect nodes wired into sampler_low / sampler_high.

import { app } from "../../scripts/app.js";
import { LUNA } from "./luna_theme.mjs";

const WATCH = new Set([
    "lora_low", "lora_high",
    "strength_low", "strength_high",
    "shift_low", "shift_high",
    "audio_shift_low", "audio_shift_high",
    "cfg_low", "cfg_high",
    "scheduler_low", "scheduler_high",
    "total_steps", "transition", "highres_denoise", "handover_sigma",
]);

// A workflow saved before each pass had its own scheduler stores 21 widget values in this order. The new
// node has 22, so ComfyUI's positional load shifts every value after cfg_low. Such a node is set again by
// name, and its one scheduler goes to both passes.
const SINGLE_SCHEDULER_ORDER = [
    "lora_low", "strength_low", "shift_low", "audio_shift_low", "cfg_low",
    "lora_high", "strength_high", "shift_high", "audio_shift_high", "cfg_high",
    "scheduler", "total_steps", "transition", "highres_denoise", "noise_seed", "control_after_generate",
    "lowres_scale", "upscaler_model", "rho", "w_min", "w_max",
];

function isSingleSchedulerLayout(values) {
    return Array.isArray(values)
        && values.length === SINGLE_SCHEDULER_ORDER.length
        && typeof values[10] === "string"
        && typeof values[17] === "string";
}

function migrateSingleScheduler(node, values) {
    if (!isSingleSchedulerLayout(values)) return false;
    SINGLE_SCHEDULER_ORDER.forEach((name, index) => {
        if (name === "scheduler") {
            setWidget(node, "scheduler_low", values[index]);
            setWidget(node, "scheduler_high", values[index]);
        } else {
            setWidget(node, name, values[index]);
        }
    });
    return true;
}
const FLOAT_FIELDS = new Set(["strength", "shift", "audio_shift", "cfg"]);
const COMPARE_FIELDS = ["strength", "total_steps", "transition", "shift", "audio_shift", "scheduler", "cfg"];

let recipesPromise = null;
let recipesCache = null;

function guard(label, fn, fallback) {
    try {
        return fn();
    } catch (error) {
        if (!guard.reported.has(label)) {
            guard.reported.add(label);
            console.error(`[Luna-H3-Sampler] ${label} failed.`, error);
        }
        return fallback;
    }
}
guard.reported = new Set();

// Long model file names hide the widget label; the closed widget shows the file's own name without
// folder or extension. Not through options.getOptionLabel: with it ComfyUI builds the list item by item
// and the list loses rgthree's folders. The list keeps the full paths.
function shortName(value) {
    if (!value || value === "none") return value ?? "";
    return String(value).split(/[\\/]/).pop().replace(/\.(safetensors|ckpt|pt|pth|bin)$/i, "");
}
const SHORT_NAMED = new Set(["lora_low", "lora_high", "upscaler_model"]);

function shortenCombos(node) {
    for (const widget of node.widgets || []) {
        if (!SHORT_NAMED.has(widget.name)) continue;
        Object.defineProperty(widget, "_displayValue", {
            configurable: true,
            get() {
                return this.computedDisabled ? "" : shortName(this.value);
            },
        });
    }
}

function widgetValue(node, name) {
    return node.widgets?.find((widget) => widget.name === name)?.value;
}

function showValue(field, value) {
    const number = Number(value);
    if (!Number.isFinite(number)) return String(value);
    if (!FLOAT_FIELDS.has(field) && Number.isInteger(number)) return String(number);
    if (Number.isInteger(number)) return number.toFixed(1);
    return String(Math.round(number * 10000) / 10000);
}

function fileName(value) {
    return String(value ?? "").split(/[\\/]/).pop();
}

// By exact key first, then by file name (Peter, 2026-11-10: "Yes, by file name").
function recipeFor(lora, recipes) {
    if (!lora || lora === "none" || !recipes) return null;
    const exact = recipes[lora] ?? recipes[String(lora).replace(/\\/g, "/")];
    if (exact) return exact;
    const wanted = fileName(lora);
    for (const [key, item] of Object.entries(recipes)) {
        if (fileName(key) === wanted) return item;
    }
    return null;
}

function differs(actual, wanted) {
    if (typeof wanted === "number" && Number.isFinite(Number(actual))) return Number(actual) !== wanted;
    return actual !== wanted;
}

// One fetch for the session. The route is the pack's luna_recipes.json.
function ensureRecipes() {
    if (!recipesPromise) {
        recipesPromise = fetch("/luna_sampler/recipes")
            .then((response) => (response.ok ? response.json() : {}))
            .then((data) => {
                recipesCache = data && typeof data === "object" ? data : {};
                return recipesCache;
            })
            .catch(() => {
                recipesCache = {};
                return recipesCache;
            });
    }
    return recipesPromise;
}

function recipeNotes(node, recipes) {
    if (!recipes) return [];
    const notes = [];
    const total = Number(widgetValue(node, "total_steps"));
    const transition = Number(widgetValue(node, "transition"));
    if (!(transition >= 1 && transition <= total - 1)) {
        notes.push(`transition ${transition} is not in [1, ${total - 1}]`);
    }
    const arms = [
        ["1", "low", widgetValue(node, "lora_low")],
        ["2", "high", widgetValue(node, "lora_high")],
    ];
    for (const [index, side, lora] of arms) {
        if (!lora || lora === "none") continue;
        const recipe = recipeFor(lora, recipes);
        if (!recipe) {
            notes.push(`pass ${index} ${shortName(lora)}: no recipe`);
            continue;
        }
        const actual = {
            strength: widgetValue(node, `strength_${side}`),
            total_steps: total,
            transition,
            shift: widgetValue(node, `shift_${side}`),
            audio_shift: widgetValue(node, `audio_shift_${side}`),
            scheduler: widgetValue(node, `scheduler_${side}`),
            cfg: widgetValue(node, `cfg_${side}`),
        };
        for (const field of COMPARE_FIELDS) {
            const wanted = recipe[field];
            if (wanted === null || wanted === undefined) continue;
            if (!differs(actual[field], wanted)) continue;
            notes.push(
                `pass ${index} ${shortName(lora)} recipe: ${field} ${showValue(field, wanted)}, set ${showValue(field, actual[field])}`,
            );
        }
    }
    return notes;
}

function scheduleLine(node) {
    const total = Number(widgetValue(node, "total_steps"));
    const transition = Number(widgetValue(node, "transition"));
    let line = `pass 1: steps 1-${transition} (low) / pass 2: steps ${transition + 1}-${total} (high)`;
    const handover = Number(widgetValue(node, "handover_sigma"));
    if (Number.isFinite(handover) && handover > 0) return `${line}, hand-over at sigma ${handover}`;
    const denoise = Number(widgetValue(node, "highres_denoise"));
    if (Number.isFinite(denoise) && denoise < 1) line += `, denoise ${denoise}`;
    return line;
}

function setWidget(node, name, value) {
    if (value === null || value === undefined) return;
    const widget = node.widgets?.find((item) => item.name === name);
    if (!widget) return;
    widget.value = value;
}

// Picking a LoRA fills the recipe's strength, total steps and shifts. Null fields stay.
function applyRecipe(node, widgetName, recipes) {
    const pass = widgetName === "lora_low" ? "low" : "high";
    const lora = widgetValue(node, widgetName);
    const recipe = lora && lora !== "none" ? recipeFor(lora, recipes) : null;
    if (!recipe) return;
    setWidget(node, `strength_${pass}`, recipe.strength);
    setWidget(node, `shift_${pass}`, recipe.shift);
    setWidget(node, `audio_shift_${pass}`, recipe.audio_shift);
    setWidget(node, "total_steps", recipe.total_steps);
}

const PAD = 10;
const MIN_WIDTH = 520;
const LINE = 14;

// Split text into lines that fit `width` px in the current font.
function wrap(ctx, text, width) {
    const lines = [];
    let current = "";
    for (const word of String(text).split(" ")) {
        const next = current ? `${current} ${word}` : word;
        if (!current || ctx.measureText(next).width <= width) {
            current = next;
            continue;
        }
        lines.push(current);
        current = word;
    }
    if (current) lines.push(current);
    return lines;
}

function readoutLines(ctx, node, width) {
    const out = [];
    const label = guard("recipeLabel", () => recipeLabel(node), null);
    if (label) {
        ctx.font = "12px sans-serif";
        for (const text of wrap(ctx, `recipe: ${label}`, width)) out.push({ text, font: "12px sans-serif", color: LUNA.accent });
    }
    ctx.font = "12px sans-serif";
    for (const text of wrap(ctx, scheduleLine(node), width)) out.push({ text, font: "12px sans-serif", color: LUNA.text });
    ctx.font = "11px sans-serif";
    for (const note of recipeNotes(node, recipesCache).slice(0, 8)) {
        const color = note.includes("no recipe") ? LUNA.muted : LUNA.danger;
        for (const text of wrap(ctx, note, width)) out.push({ text, font: "11px sans-serif", color });
    }
    return out;
}

function readoutBox(node) {
    const x = 14;
    const outer = Number(node?.size?.[0]) || 320;
    return { x, w: Math.max(160, outer - x * 2) };
}

function drawReadout(ctx, node, y) {
    const { x, w } = readoutBox(node);
    ctx.save();
    try {
        const lines = readoutLines(ctx, node, w - PAD * 2);
        const h = 14 + lines.length * LINE;
        ctx.beginPath();
        if (typeof ctx.roundRect === "function") ctx.roundRect(x, y, w, h, 6);
        else ctx.rect(x, y, w, h);
        ctx.fillStyle = LUNA.panel;
        ctx.fill();
        ctx.strokeStyle = LUNA.border;
        ctx.lineWidth = 1;
        ctx.stroke();
        ctx.textBaseline = "alphabetic";
        ctx.textAlign = "left";
        lines.forEach((line, index) => {
            ctx.font = line.font;
            ctx.fillStyle = line.color;
            ctx.fillText(line.text, x + PAD, y + 18 + index * LINE);
        });
    } finally {
        ctx.restore();
    }
}

function onWidget(node, name) {
    if (name === "lora_low" || name === "lora_high") {
        ensureRecipes().then((recipes) => {
            guard("recipe", () => applyRecipe(node, name, recipes), undefined);
            node.setDirtyCanvas?.(true, true);
        });
        return;
    }
    node.setDirtyCanvas?.(true, true);
}

function watchWidgets(node) {
    for (const widget of node.widgets || []) {
        if (!WATCH.has(widget.name) || widget._lunaSampler) continue;
        widget._lunaSampler = true;
        const previous = widget.callback;
        widget.callback = function (...args) {
            const result = previous?.apply(this, args);
            guard("sync", () => onWidget(node, widget.name), undefined);
            return result;
        };
    }
}

/* ------------------------------------------------------------ recipe menu (⚙) */

let menuCache = null; // { shipped, user, user_error } from GET /luna_sampler/pair_recipes
let menuPromise = null;

function loadMenu(force = false) {
    if (force || !menuPromise) {
        menuPromise = fetch("/luna_sampler/pair_recipes")
            .then((response) => response.json().then((data) => (response.ok ? data : { error: data?.error })))
            .then((data) => {
                menuCache = data && Array.isArray(data.shipped)
                    ? data
                    : { shipped: [], user: [], user_error: data?.error || "the recipes did not load" };
                return menuCache;
            })
            .catch((error) => {
                menuCache = { shipped: [], user: [], user_error: String(error) };
                return menuCache;
            });
    }
    return menuPromise;
}

async function post(url, body) {
    const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
    });
    let data = null;
    try {
        data = await response.json();
    } catch {
        data = null;
    }
    if (!response.ok) throw new Error(data?.error || `${url}: ${response.status}`);
    return data;
}

const SIDES = [["low", 1], ["high", 2]];
const ARM_NUMBERS = ["strength", "shift", "audio_shift", "cfg"];
const EPS = 1e-6;

function getLink(graph, id) {
    if (id == null || !graph) return null;
    return graph.getLink?.(id) ?? graph.links?.get?.(id) ?? graph.links?.[id] ?? null;
}

// The KSamplerSelect feeding sampler_low / sampler_high, or why there is none.
function samplerSource(node, side) {
    const input = node.inputs?.find((item) => item.name === `sampler_${side}`);
    const link = getLink(node.graph, input?.link);
    if (!link) return { widget: null, why: `sampler_${side} is not connected` };
    const origin = node.graph.getNodeById?.(link.origin_id);
    const widget = origin?.widgets?.find((item) => item.name === "sampler_name");
    if (!origin || !widget) {
        return { widget: null, why: `sampler_${side} is fed by ${origin?.title || origin?.type || "a node"}, not a KSamplerSelect` };
    }
    return { node: origin, widget };
}

// One of width / height on the node that makes latent_image: its own widget, or, when wired, the node feeding it
// (Luna-H3-Canvas publishes its resolved size as `lunaCanvasSize`; a primitive has a `value` widget).
function sideValue(graph, origin, name) {
    const input = origin.inputs?.find((item) => item.name === name);
    const link = getLink(graph, input?.link);
    if (!link) return { value: num(widgetValue(origin, name)), from: origin };
    const source = graph.getNodeById?.(link.origin_id);
    if (!source) return { value: null };
    const output = source.outputs?.[link.origin_slot]?.name;
    if (Array.isArray(source.lunaCanvasSize) && (output === "width" || output === "height")) {
        return { value: num(source.lunaCanvasSize[output === "width" ? 0 : 1]), from: source };
    }
    return { value: num(widgetValue(source, "value") ?? widgetValue(source, output)), from: source };
}

// A boolean switch's value: its own widget, or the primitive wired into it.
function switchValue(graph, node) {
    const input = node.inputs?.find((item) => item.name === "switch");
    const link = getLink(graph, input?.link);
    const source = link ? graph.getNodeById?.(link.origin_id) : null;
    const value = source ? widgetValue(source, "value") : widgetValue(node, "switch");
    return typeof value === "boolean" ? value : null;
}

// The final size from the graph, or null when it cannot be read there (the panel then asks for it). Walks back from
// latent_image through switches and other pass-through nodes (a switch's chosen branch first) to the first node that
// has width and height, e.g. MiniMaxH3ReferenceToVideo fed by Luna-H3-Canvas.
function upstreamSize(node) {
    const graph = node.graph;
    const seen = new Set();
    const walk = (target, inputName, depth) => {
        const input = target.inputs?.find((item) => item.name === inputName);
        const link = getLink(graph, input?.link);
        const origin = link ? graph.getNodeById?.(link.origin_id) : null;
        if (!origin || depth > 8 || seen.has(origin.id)) return null;
        seen.add(origin.id);
        const has = (name) => origin.inputs?.some((item) => item.name === name) || origin.widgets?.some((item) => item.name === name);
        if (has("width") && has("height")) {
            const w = sideValue(graph, origin, "width");
            const h = sideValue(graph, origin, "height");
            if (!(w.value > 0 && h.value > 0)) return null;
            const from = w.from || origin;
            return { w: w.value, h: h.value, from: from.title || from.type };
        }
        const chosen = { true: "on_true", false: "on_false" }[switchValue(graph, origin)];
        const names = (origin.inputs || [])
            .filter((item) => item.link != null && item.name !== "switch")
            .map((item) => item.name)
            .sort((a, b) => (b === chosen) - (a === chosen));
        for (const name of names) {
            const found = walk(origin, name, depth + 1);
            if (found) return found;
        }
        return null;
    };
    return walk(node, "latent_image", 0);
}

// An empty or mistyped field is null, never 0 (Number("") is 0): check_pair then reports it.
function num(value) {
    if (value === null || value === undefined || (typeof value === "string" && !value.trim())) return null;
    const n = Number(value);
    return Number.isFinite(n) ? n : null;
}

// What the node will run now, in the recipe shape (LoRAs by file name).
function nodeRecipe(node) {
    const recipe = {
        total_steps: num(widgetValue(node, "total_steps")),
        transition: num(widgetValue(node, "transition")),
        handover_sigma: num(widgetValue(node, "handover_sigma")) ?? 0,
        lowres_scale: num(widgetValue(node, "lowres_scale")),
    };
    for (const [side] of SIDES) {
        const arm = { lora: fileName(widgetValue(node, `lora_${side}`)), scheduler: widgetValue(node, `scheduler_${side}`) };
        for (const field of ARM_NUMBERS) arm[field] = num(widgetValue(node, `${field}_${side}`));
        arm.sampler = samplerSource(node, side).widget?.value ?? null;
        recipe[side] = arm;
    }
    return recipe;
}

function same(a, b) {
    if (typeof a === "number" || typeof b === "number") return num(a) !== null && num(b) !== null && Math.abs(num(a) - num(b)) <= EPS;
    return a === b;
}

function matchesRecipe(node, recipe) {
    const now = nodeRecipe(node);
    for (const key of ["total_steps", "transition", "handover_sigma", "lowres_scale"]) {
        if (!same(now[key], recipe[key])) return false;
    }
    for (const [side] of SIDES) {
        const a = now[side];
        const b = recipe[side] || {};
        if (fileName(a.lora) !== fileName(b.lora) || a.scheduler !== b.scheduler) return false;
        for (const field of ARM_NUMBERS) if (!same(a[field], b[field])) return false;
        if (a.sampler !== null && a.sampler !== b.sampler) return false;
    }
    return true;
}

function allRecipes() {
    return [...(menuCache?.shipped || []), ...(menuCache?.user || [])];
}

// The readout's and the badge's name: the recipe last applied while the widgets still match it.
function recipeLabel(node) {
    const name = node.properties?.lunaRecipe;
    if (!name || !menuCache) return null;
    const recipe = allRecipes().find((item) => item.name === name);
    if (!recipe) return null;
    return matchesRecipe(node, recipe) ? name : `custom (changed from ${name})`;
}

// Writes the widgets and the two sampler names. Returns notes for what could not be set.
function applyFlat(node, flat) {
    const notes = [];
    for (const [name, value] of Object.entries(flat)) {
        if (value === null || value === undefined) continue;
        if (name === "sampler_low" || name === "sampler_high") {
            const source = samplerSource(node, name.slice(8));
            if (!source.widget) {
                notes.push(`${source.why}: set ${value} there by hand`);
                continue;
            }
            const options = source.widget.options?.values;
            if (Array.isArray(options) && !options.includes(value)) {
                notes.push(`${value} is not a sampler this ComfyUI lists`);
                continue;
            }
            source.widget.value = value;
            source.node.setDirtyCanvas?.(true, true);
            continue;
        }
        setWidget(node, name, value);
    }
    node.setDirtyCanvas?.(true, true);
    node.graph?.change?.();
    return notes;
}

function recipeToFlat(recipe, where) {
    const flat = {
        total_steps: recipe.total_steps,
        transition: recipe.transition,
        handover_sigma: recipe.handover_sigma,
        lowres_scale: recipe.lowres_scale,
    };
    for (const [side] of SIDES) {
        const arm = recipe[side] || {};
        flat[`lora_${side}`] = where?.[side] ?? arm.lora;
        flat[`scheduler_${side}`] = arm.scheduler;
        flat[`sampler_${side}`] = arm.sampler;
        for (const field of ARM_NUMBERS) flat[`${field}_${side}`] = arm[field];
    }
    return flat;
}

const esc = (text) => String(text ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const fmtSigma = (value) => (num(value) ? String(Math.round(num(value) * 1e6) / 1e6) : "0");

const MENU_CSS = (C) => `
.lsr{position:fixed;z-index:1300;width:480px;max-height:780px;transform-origin:0 0;display:flex;flex-direction:column;background:${C.panel};border:1px solid ${C.accent}66;border-radius:8px;box-shadow:0 10px 30px #000a;font:13.5px Inter,system-ui,sans-serif;color:${C.text}}
.lsr-h{display:flex;align-items:center;gap:8px;padding:10px 12px;border-bottom:1px solid ${C.border};user-select:none}
.lsr-h .t{font-weight:600;font-size:15px;flex:1}
.lsr-lv{font-size:11.5px;padding:1px 7px;border-radius:4px;background:${C.accent}22;color:${C.accent};max-width:220px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.lsr-x{cursor:pointer;color:${C.muted};background:none;border:0;font-size:13.5px;padding:0 2px}
.lsr-x:hover{color:${C.accent}}
.lsr-tabs{display:flex;gap:4px;padding:10px 14px 0}
.lsr-tabs span{font-size:13px;padding:4px 14px;border:1px solid ${C.border};border-radius:4px;cursor:pointer}
.lsr-tabs span.on{border-color:${C.accent};color:${C.accent};background:${C.accent}14}
.lsr-b{overflow:auto;padding:2px 14px 12px}
.lsr-sh{display:flex;align-items:center;gap:6px;color:${C.accent};font-size:13px;font-weight:600;margin:12px 0 6px}
.lsr-sh .n{margin-left:auto;color:${C.muted};font-size:11.5px;font-weight:400}
.lsr-note{font-size:12px;color:${C.muted};margin:2px 0 6px}
.lsr-bad{font-size:12px;color:${C.danger};margin:2px 0}
.lsr-warn{font-size:12px;color:${C.accent};margin:2px 0}
.lsr-ok{font-size:12px;color:#7fb069;margin:2px 0}
.lsr-card{border:1px solid ${C.border};border-radius:6px;padding:8px 10px;margin:6px 0;cursor:pointer}
.lsr-card:hover{border-color:${C.muted}}
.lsr-card.on{border-color:${C.accent};background:${C.accent}10}
.lsr-card.gone{opacity:.55}
.lsr-card .top{display:flex;align-items:center;gap:8px}
.lsr-card .nm{font-weight:600;flex:1}
.lsr-card .sub{color:${C.muted};font-size:12px;margin-top:3px}
.lsr-tag{font-size:11px;padding:1px 6px;border-radius:4px;border:1px solid ${C.border};color:${C.muted}}
.lsr-tag.t{border-color:#7fb06988;color:#7fb069}
.lsr-tag.m{border-color:${C.accent}88;color:${C.accent}}
.lsr-act{font-size:11.5px;color:${C.muted};cursor:pointer}
.lsr-act:hover{color:${C.accent}}
.lsr-kv{display:grid;grid-template-columns:110px minmax(0,1fr) minmax(0,1fr);gap:3px 10px;font-size:12.5px;margin:4px 0}
.lsr-kv .k{color:${C.muted}}
.lsr-kv .hd{color:${C.muted};font-size:11.5px;border-bottom:1px solid ${C.border};padding-bottom:2px}
.lsr-kv span{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.lsr-r{display:grid;grid-template-columns:150px minmax(0,1fr);align-items:center;gap:10px;margin:5px 0;min-height:26px}
.lsr-r .l{color:${C.text}}
.lsr-r .pair{display:flex;gap:6px;align-items:center}
.lsr-why{font-size:11.5px;color:${C.muted};margin:-3px 0 4px 160px}
.lsr-mine{color:${C.accent};font-size:11px;margin-left:6px;cursor:pointer}
.lsr-in,.lsr-sel{width:100%;box-sizing:border-box;background:${C.bg};border:1px solid ${C.border};border-radius:4px;color:${C.text};font-size:12.5px;padding:3px 6px}
.lsr-in.num{text-align:center}
.lsr-in.over{border-color:${C.accent}}
.lsr-in:focus,.lsr-sel:focus{outline:none;border-color:${C.accent}}
.lsr-btns{display:flex;gap:6px;margin-top:10px}
.lsr-btn{flex:1;text-align:center;border:1px solid ${C.border};border-radius:5px;padding:6px 0;font-size:13px;cursor:pointer;background:none;color:${C.text}}
.lsr-btn:hover{border-color:${C.muted}}
.lsr-btn.pri{border-color:${C.accent};color:${C.accent};background:${C.accent}14}
.lsr-btn:disabled{opacity:.4;cursor:default}
.lsr-f{border-top:1px solid ${C.border};padding:8px 14px;display:flex;justify-content:flex-end;color:${C.muted};font-size:12.5px}
.lsr-f span{cursor:pointer}.lsr-f span:hover{color:${C.accent}}
`;

let _menu = null; // { el, node, cleanup, refresh }
let _lastClose = { node: null, t: 0 };

function isMenuOpenFor(node) {
    return !!_menu && _menu.node === node;
}

function closeMenu() {
    if (!_menu) return;
    const menu = _menu;
    _menu = null;
    _lastClose = { node: menu.node, t: Date.now() };
    // A focused field is committed first: Chrome fires no change event on a removed element.
    if (menu.el.contains(document.activeElement)) document.activeElement.blur();
    menu.cleanup();
    menu.el.remove();
    menu.node.setDirtyCanvas?.(true, true);
}

function nodeScreenRect(node) {
    const canvas = app.canvas;
    const rect = canvas.canvas.getBoundingClientRect();
    const { scale, offset } = canvas.ds;
    const title = window.LiteGraph?.NODE_TITLE_HEIGHT ?? 30;
    return {
        left: rect.left + (node.pos[0] + offset[0]) * scale,
        top: rect.top + (node.pos[1] - title + offset[1]) * scale,
        width: node.size[0] * scale,
    };
}

// Pinned to the node's right edge; follows pan and zoom every frame (LiteGraph emits nothing on either).
function follow(el, node) {
    let frame = 0;
    let last = "";
    const tick = () => {
        frame = requestAnimationFrame(tick);
        if (!node.graph) {
            closeMenu();
            return;
        }
        const rect = nodeScreenRect(node);
        const scale = app.canvas?.ds?.scale || 1;
        const left = rect.left + rect.width + 10 * scale;
        // The panel never runs past the window's bottom edge: its height is capped to the room below its top,
        // in panel units (it is scaled with the canvas), so Apply stays reachable at any zoom.
        const room = Math.max(240, Math.min(780, (window.innerHeight - Math.max(rect.top, 0) - 12) / scale));
        const key = `${left.toFixed(1)}|${rect.top.toFixed(1)}|${scale}|${room.toFixed(0)}`;
        if (key === last) return;
        last = key;
        el.style.left = `${left}px`;
        el.style.top = `${rect.top}px`;
        el.style.transform = `scale(${scale})`;
        el.style.maxHeight = `${room}px`;
    };
    tick();
    return () => cancelAnimationFrame(frame);
}

function cardHtml(recipe, selected, mine) {
    const low = recipe.low || {};
    const high = recipe.high || {};
    const missing = recipe.where?.missing || [];
    const split = num(recipe.handover_sigma) ? `hand-over ${fmtSigma(recipe.handover_sigma)}` : "step split";
    const samplers = low.sampler === high.sampler ? low.sampler : `${low.sampler} / ${high.sampler}`;
    const strengths = same(low.strength, 1) && same(high.strength, 1) ? "" : ` · ${low.strength} / ${high.strength}`;
    const tag = mine ? `<span class="lsr-tag m">MINE</span>` : `<span class="lsr-tag t">${esc(recipe.label || "")}</span>`;
    const actions = mine ? `<span class="lsr-act" data-act="rename">rename</span><span class="lsr-act" data-act="delete">delete</span>` : "";
    const told = mine
        ? [recipe.tested_on && `Tested on: ${recipe.tested_on}`, recipe.verdict && `Verdict: ${recipe.verdict}`, recipe.date].filter(Boolean).join(" · ")
        : recipe.note;
    return `<div class="lsr-card${selected ? " on" : ""}${missing.length ? " gone" : ""}" data-name="${esc(recipe.name)}">
<div class="top"><span class="nm">${esc(recipe.name)}</span>${actions}${tag}</div>
<div class="sub">${esc(shortName(low.lora))} → ${esc(shortName(high.lora))} · ${recipe.total_steps} / ${recipe.transition} · ${esc(split)} · ${esc(samplers)}${esc(strengths)}</div>
${told ? `<div class="sub">${esc(told)}</div>` : ""}
${missing.map((name) => `<div class="lsr-bad">missing: ${esc(name)}</div>`).join("")}</div>`;
}

function setsHtml(recipe) {
    const low = recipe.low || {};
    const high = recipe.high || {};
    const row = (label, a, b = "") => `<span class="k">${label}</span><span title="${esc(a)}">${esc(a)}</span><span title="${esc(b)}">${esc(b)}</span>`;
    return `<div class="lsr-sh">≡ What Apply sets</div><div class="lsr-kv">
<span class="hd"></span><span class="hd">pass 1 (low)</span><span class="hd">pass 2 (high)</span>
${row("LoRA", shortName(recipe.where?.low || low.lora), shortName(recipe.where?.high || high.lora))}
${row("strength", low.strength, high.strength)}
${row("sampler", low.sampler, high.sampler)}
${row("scheduler", low.scheduler, high.scheduler)}
${row("shift / audio", `${low.shift} / ${low.audio_shift}`, `${high.shift} / ${high.audio_shift}`)}
${row("cfg", low.cfg, high.cfg)}
${row("steps", `${recipe.total_steps} total, transition ${recipe.transition}`)}
${row("hand-over", num(recipe.handover_sigma) ? fmtSigma(recipe.handover_sigma) : "0 (step split)")}
${row("lowres_scale", recipe.lowres_scale)}
</div>${(recipe.where?.duplicates || []).map((text) => `<div class="lsr-warn">${esc(text)}</div>`).join("")}
<div class="lsr-note">The sampler names go to the KSamplerSelect nodes wired into sampler_low and sampler_high.</div>`;
}

function messageHtml(message) {
    if (!message) return "";
    const lines = [];
    if (message.ok) lines.push(`<div class="lsr-ok">✓ ${esc(message.ok)}</div>`);
    for (const text of message.errors || []) lines.push(`<div class="lsr-bad">${esc(text)}</div>`);
    for (const text of message.warnings || []) lines.push(`<div class="lsr-warn">${esc(text)}</div>`);
    return `<div style="margin-top:10px">${lines.join("")}</div>`;
}

// The Custom tab's recommended fields, in the widget's own names.
const RECOMMENDED = [
    ["handover_sigma", "Hand-over sigma"],
    ["total_steps", "Total steps"],
    ["transition", "Transition (pass 1 steps)"],
    ["scheduler_low", "Pass 1 scheduler"],
    ["scheduler_high", "Pass 2 scheduler"],
    ["shift_low", "Pass 1 shift / audio"],
    ["shift_high", "Pass 2 shift / audio"],
    ["lowres_scale", "lowres_scale"],
];
const WHY_AFTER = { handover_sigma: "handover_sigma", transition: "steps", lowres_scale: "lowres_scale" };

function openMenu(node) {
    if (isMenuOpenFor(node)) {
        closeMenu();
        return;
    }
    closeMenu();
    if (!document.getElementById("lsr-css")) {
        const style = document.createElement("style");
        style.id = "lsr-css";
        style.textContent = MENU_CSS(LUNA);
        document.head.appendChild(style);
    }
    node.properties = node.properties || {};
    const el = document.createElement("div");
    el.className = "lsr";
    el.innerHTML = `<div class="lsr-h"><span>⚙</span><span class="t">Recipes</span><span class="lsr-lv"></span>
<button class="lsr-x" title="Close">✕</button></div>
<div class="lsr-tabs"><span data-tab="recipes">Recipes</span><span data-tab="custom">Custom</span></div>
<div class="lsr-b"></div><div class="lsr-f"><span data-act="close">Done</span></div>`;
    const body = el.querySelector(".lsr-b");
    const badge = el.querySelector(".lsr-lv");
    const lastSize = node.properties.lunaFinalSize;
    const found = upstreamSize(node);
    const state = {
        tab: node.properties.lunaRecipeTab === "custom" ? "custom" : "recipes",
        selected: node.properties.lunaRecipe || null,
        message: null,
        saving: false,
        renaming: null,
        custom: null, // { lora_low, lora_high, strength_low, strength_high, sampler_low, sampler_high, width, height }
        rec: null, // the last /recommend answer
        over: {}, // fields the user changed after the recommendation
        sizeFrom: found?.from || null,
    };
    state.size = found ? [found.w, found.h] : Array.isArray(lastSize) ? lastSize : [1920, 1088];

    const refreshBadge = () => {
        badge.textContent = recipeLabel(node) || "custom";
        for (const tab of el.querySelectorAll("[data-tab]")) tab.classList.toggle("on", tab.dataset.tab === state.tab);
    };

    const selectedRecipe = () => allRecipes().find((item) => item.name === state.selected) || null;

    async function validate(recipe) {
        try {
            return await post("/luna_sampler/validate", { recipe });
        } catch (error) {
            return { errors: [String(error.message || error)], warnings: [] };
        }
    }

    async function applyRecipe(recipe, flat, name) {
        const checked = await validate(recipe);
        if (checked.errors.length) {
            state.message = { errors: checked.errors, warnings: checked.warnings };
            render();
            return;
        }
        const notes = applyFlat(node, flat);
        node.properties.lunaRecipe = name;
        state.message = { ok: name ? `applied ${name}` : "applied", warnings: [...checked.warnings, ...notes] };
        render();
    }

    function renderRecipes() {
        const data = menuCache || { shipped: [], user: [] };
        const selected = selectedRecipe();
        const mine = (data.user || []).map((recipe) => {
            if (state.renaming === recipe.name) {
                return `<div class="lsr-card on" data-name="${esc(recipe.name)}"><input class="lsr-in" data-rename="${esc(recipe.name)}" value="${esc(recipe.name)}"></div>`;
            }
            return cardHtml(recipe, recipe.name === state.selected, true);
        }).join("");
        body.innerHTML = `<div class="lsr-sh">◆ Tested <span class="n">ship with the node</span></div>
${(data.shipped || []).map((recipe) => cardHtml(recipe, recipe.name === state.selected, false)).join("")}
<div class="lsr-sh">★ Mine <span class="n">saved by you, in ComfyUI's user folder</span></div>
${data.user_error ? `<div class="lsr-bad">${esc(data.user_error)}</div>` : ""}
${mine || `<div class="lsr-note">None yet. Run, judge the clip, then Custom › Save as recipe.</div>`}
${selected ? setsHtml(selected) : `<div class="lsr-note" style="margin-top:12px">Pick a recipe to see what Apply sets.</div>`}
${messageHtml(state.message)}
<div class="lsr-btns"><button class="lsr-btn" data-act="validate"${selected ? "" : " disabled"}>Validate</button>
<button class="lsr-btn pri" data-act="apply"${selected && !(selected.where?.missing || []).length ? "" : " disabled"}>Apply</button></div>`;
        const input = body.querySelector("[data-rename]");
        if (input) {
            input.focus();
            input.select();
            const commit = async () => {
                const old = input.dataset.rename;
                const name = input.value.trim();
                state.renaming = null;
                if (name && name !== old) {
                    try {
                        menuCache = await post("/luna_sampler/user_recipes", { action: "rename", name: old, new_name: name });
                        if (state.selected === old) state.selected = name;
                        if (node.properties.lunaRecipe === old) node.properties.lunaRecipe = name;
                        state.message = { ok: `renamed to ${name}` };
                    } catch (error) {
                        state.message = { errors: [String(error.message || error)] };
                    }
                }
                render();
            };
            input.addEventListener("keydown", (event) => {
                if (event.key === "Enter") input.blur();
                if (event.key === "Escape") {
                    event.stopPropagation();
                    input.value = input.dataset.rename;
                    input.blur();
                }
            });
            input.addEventListener("blur", commit, { once: true });
        }
    }

    function customValue(name) {
        if (name in state.over) return state.over[name];
        if (state.rec?.values && name in state.rec.values) return state.rec.values[name];
        return widgetValue(node, name);
    }

    async function recommendNow() {
        const c = state.custom;
        try {
            state.rec = await post("/luna_sampler/recommend", {
                lora_low: c.lora_low, lora_high: c.lora_high, width: state.size[0], height: state.size[1],
            });
        } catch (error) {
            state.rec = { values: {}, why: {}, notes: [String(error.message || error)] };
        }
        state.over = {};
        // The user's own strength and sampler picks win over the LoRA recipe's.
        for (const [side] of SIDES) {
            if (c[`strength_${side}`] !== null) state.over[`strength_${side}`] = c[`strength_${side}`];
            if (c[`sampler_${side}`]) state.over[`sampler_${side}`] = c[`sampler_${side}`];
        }
        render();
    }

    function customRecipe(name = "custom") {
        const recipe = {
            name,
            total_steps: num(customValue("total_steps")),
            transition: num(customValue("transition")),
            handover_sigma: num(customValue("handover_sigma")),
            lowres_scale: num(customValue("lowres_scale")),
        };
        for (const [side] of SIDES) {
            recipe[side] = {
                lora: fileName(state.custom[`lora_${side}`]),
                strength: num(customValue(`strength_${side}`)),
                sampler: customValue(`sampler_${side}`) ?? samplerSource(node, side).widget?.value ?? null,
                scheduler: customValue(`scheduler_${side}`),
                shift: num(customValue(`shift_${side}`)),
                audio_shift: num(customValue(`audio_shift_${side}`)),
                cfg: num(customValue(`cfg_${side}`)),
            };
        }
        return recipe;
    }

    function renderCustom() {
        if (!state.custom) {
            state.custom = { width: state.size[0], height: state.size[1] };
            for (const [side] of SIDES) {
                state.custom[`lora_${side}`] = widgetValue(node, `lora_${side}`);
                state.custom[`strength_${side}`] = null;
                state.custom[`sampler_${side}`] = null;
            }
            recommendNow();
            body.innerHTML = `<div class="lsr-note" style="margin-top:12px">Computing…</div>`;
            return;
        }
        const loraOptions = node.widgets?.find((item) => item.name === "lora_low")?.options?.values || [];
        const schedulerOptions = node.widgets?.find((item) => item.name === "scheduler_low")?.options?.values || [];
        const options = (list, value) => list.map((item) => `<option value="${esc(item)}"${item === value ? " selected" : ""}>${esc(item)}</option>`).join("");
        const arms = SIDES.map(([side, index]) => {
            const source = samplerSource(node, side);
            const samplers = source.widget?.options?.values || [];
            const sampler = customValue(`sampler_${side}`) ?? source.widget?.value;
            return `<div class="lsr-r"><span class="l">Pass ${index} LoRA</span><select class="lsr-sel" data-c="lora_${side}">${options(loraOptions, state.custom[`lora_${side}`])}</select></div>
<div class="lsr-r"><span class="l">Pass ${index} strength</span><input class="lsr-in num" data-c="strength_${side}" value="${esc(customValue(`strength_${side}`))}"></div>
<div class="lsr-r"><span class="l">Pass ${index} sampler</span>${source.widget
                ? `<select class="lsr-sel" data-c="sampler_${side}">${options(samplers, sampler)}</select>`
                : `<span class="lsr-warn">${esc(source.why)}</span>`}</div>`;
        }).join("");
        const field = (name, label) => {
            const why = state.rec?.why?.[WHY_AFTER[name]];
            const recommended = state.rec?.values?.[name];
            const mine = name in state.over && recommended !== undefined && !same(state.over[name], recommended);
            let control;
            if (name.startsWith("scheduler_")) {
                control = `<select class="lsr-sel${mine ? " over" : ""}" data-o="${name}">${options(schedulerOptions, customValue(name))}</select>`;
            } else if (name.startsWith("shift_")) {
                const audio = name.replace("shift_", "audio_shift_");
                const audioMine = audio in state.over && state.rec?.values?.[audio] !== undefined && !same(state.over[audio], state.rec.values[audio]);
                control = `<span class="pair"><input class="lsr-in num${mine ? " over" : ""}" data-o="${name}" value="${esc(customValue(name))}">
<input class="lsr-in num${audioMine ? " over" : ""}" data-o="${audio}" value="${esc(customValue(audio))}"></span>`;
            } else {
                control = `<input class="lsr-in num${mine ? " over" : ""}" data-o="${name}" value="${esc(name === "handover_sigma" ? fmtSigma(customValue(name)) : customValue(name))}">`;
            }
            const back = mine ? `<span class="lsr-mine" data-reset="${name}" title="Back to ${esc(recommended)}">yours ↺</span>` : "";
            return `<div class="lsr-r" data-f="${name}"><span class="l">${label}${back}</span>${control}</div>${why ? `<div class="lsr-why">${esc(why)}</div>` : ""}`;
        };
        const notes = (state.rec?.notes || []).map((text) => `<div class="lsr-warn">${esc(text)}</div>`).join("");
        body.innerHTML = state.saving ? saveHtml() : `<div class="lsr-sh">◐ You choose</div>${arms}
<div class="lsr-r"><span class="l">Final size</span><span class="pair"><input class="lsr-in num" data-size="0" value="${state.size[0]}"> × <input class="lsr-in num" data-size="1" value="${state.size[1]}"></span></div>
${state.sizeFrom === "typed" ? `<div class="lsr-why">typed</div>`
        : state.sizeFrom ? `<div class="lsr-why">from ${esc(state.sizeFrom)}</div>`
        : `<div class="lsr-why" style="color:${LUNA.accent}">Not read from the graph: check it, lowres_scale is computed from it.</div>`}
<div class="lsr-sh">✦ The node recommends <span class="n">every value can be changed</span></div>
${state.rec ? RECOMMENDED.map(([name, label]) => field(name, label)).join("") : `<div class="lsr-note">Computing…</div>`}
${notes}${messageHtml(state.message)}
<div class="lsr-btns"><button class="lsr-btn" data-act="cvalidate">Validate</button><button class="lsr-btn pri" data-act="capply">Apply</button>
<button class="lsr-btn" data-act="saveopen">Save as recipe…</button></div>`;
    }

    function saveHtml() {
        const now = nodeRecipe(node);
        return `<div class="lsr-sh">★ Save as recipe <span class="n">after a clip you judged</span></div>
<div class="lsr-note">Saves what the node runs now: ${esc(shortName(now.low.lora))} → ${esc(shortName(now.high.lora))}, ${now.total_steps} / ${now.transition}, hand-over ${fmtSigma(now.handover_sigma)}, lowres_scale ${now.lowres_scale}.</div>
<div class="lsr-r"><span class="l">Name</span><input class="lsr-in" data-s="name"></div>
<div class="lsr-r"><span class="l">Tested on</span><input class="lsr-in" data-s="tested_on" placeholder="subject, shot, size, length"></div>
<div class="lsr-r"><span class="l">Verdict</span><input class="lsr-in" data-s="verdict"></div>
${messageHtml(state.message)}
<div class="lsr-btns"><button class="lsr-btn" data-act="savecancel">Cancel</button><button class="lsr-btn pri" data-act="save">Save</button></div>`;
    }

    function render() {
        refreshBadge();
        if (state.tab === "recipes") renderRecipes();
        else renderCustom();
    }

    body.addEventListener("click", async (event) => {
        const target = event.target;
        const act = target.closest?.("[data-act]")?.dataset.act;
        const reset = target.dataset?.reset;
        if (reset) {
            delete state.over[reset];
            render();
            return;
        }
        const card = target.closest?.(".lsr-card");
        if (act === "delete" && card) {
            const name = card.dataset.name;
            if (!window.confirm(`Delete your recipe "${name}"? This cannot be undone.`)) return;
            try {
                menuCache = await post("/luna_sampler/user_recipes", { action: "delete", name });
                if (state.selected === name) state.selected = null;
                state.message = { ok: `deleted ${name}` };
            } catch (error) {
                state.message = { errors: [String(error.message || error)] };
            }
            render();
            return;
        }
        if (act === "rename" && card) {
            state.renaming = card.dataset.name;
            render();
            return;
        }
        if (card && !act && !target.closest("[data-rename]")) {
            state.selected = card.dataset.name;
            state.message = null;
            render();
            return;
        }
        if (act === "validate" || act === "apply") {
            const recipe = selectedRecipe();
            if (!recipe) return;
            if (act === "apply") {
                await applyRecipe(recipe, recipeToFlat(recipe, recipe.where), recipe.name);
            } else {
                const checked = await validate(recipe);
                state.message = checked.errors.length ? checked : { ok: "valid", warnings: checked.warnings };
                render();
            }
            return;
        }
        if (act === "cvalidate" || act === "capply") {
            const recipe = customRecipe();
            if (act === "capply") {
                const flat = recipeToFlat(recipe, { low: state.custom.lora_low, high: state.custom.lora_high });
                await applyRecipe(recipe, flat, null);
            } else {
                const checked = await validate(recipe);
                state.message = checked.errors.length ? checked : { ok: "valid", warnings: checked.warnings };
                render();
            }
            return;
        }
        if (act === "saveopen") {
            state.saving = true;
            state.message = null;
            render();
            return;
        }
        if (act === "savecancel") {
            state.saving = false;
            state.message = null;
            render();
            return;
        }
        if (act === "save") {
            const read = (key) => body.querySelector(`[data-s="${key}"]`)?.value?.trim() || "";
            const recipe = { ...nodeRecipe(node), name: read("name"), tested_on: read("tested_on"), verdict: read("verdict") };
            try {
                menuCache = await post("/luna_sampler/user_recipes", { action: "save", recipe });
                await loadMenu(true).catch(() => null);
                state.saving = false;
                state.tab = "recipes";
                node.properties.lunaRecipeTab = "recipes";
                state.selected = recipe.name;
                node.properties.lunaRecipe = recipe.name;
                state.message = { ok: `saved ${recipe.name}` };
            } catch (error) {
                state.message = { errors: [String(error.message || error)] };
            }
            render();
        }
    });

    body.addEventListener("change", (event) => {
        const target = event.target;
        if (target.dataset.c) {
            const key = target.dataset.c;
            state.custom[key] = key.startsWith("strength_") ? num(target.value) : target.value;
            state.message = null;
            if (key.startsWith("lora_")) {
                state.custom[`strength_${key.slice(5)}`] = null;
                recommendNow();
            } else {
                // No render here: a field's change fires on the next button's mousedown, and a re-render would
                // detach that button so its click is lost.
                state.over[key] = state.custom[key];
            }
            return;
        }
        if (target.dataset.size) {
            const index = Number(target.dataset.size);
            const value = Math.round(Number(target.value));
            if (value > 0) {
                state.size[index] = value;
                state.sizeFrom = "typed";
                node.properties.lunaFinalSize = [...state.size];
                recommendNow();
            } else {
                target.value = state.size[index];
            }
            return;
        }
        if (target.dataset.o) {
            const name = target.dataset.o;
            state.over[name] = name.startsWith("scheduler_") ? target.value : num(target.value);
            state.message = null;
            markOver(target, name);
        }
    });

    // Marks a changed recommended field in place (see the note above on why there is no render).
    function markOver(target, name) {
        const recommended = state.rec?.values?.[name];
        const mine = recommended !== undefined && !same(state.over[name], recommended);
        target.classList.toggle("over", mine);
        const row = target.closest(".lsr-r");
        if (!row || row.dataset.f !== name) return;
        const label = row.querySelector(".l");
        const mark = label?.querySelector(".lsr-mine");
        if (mine && label && !mark) {
            label.insertAdjacentHTML("beforeend", `<span class="lsr-mine" data-reset="${name}" title="Back to ${esc(recommended)}">yours ↺</span>`);
        } else if (!mine && mark) {
            mark.remove();
        }
    }

    el.querySelector(".lsr-tabs").addEventListener("click", (event) => {
        const tab = event.target.dataset?.tab;
        if (!tab) return;
        state.tab = tab;
        state.message = null;
        state.saving = false;
        node.properties.lunaRecipeTab = tab;
        render();
    });
    el.querySelector(".lsr-x").addEventListener("click", (event) => {
        event.stopPropagation();
        closeMenu();
    });
    el.querySelector(".lsr-f").addEventListener("click", (event) => {
        if (event.target.dataset?.act === "close") closeMenu();
    });
    // ComfyUI binds single letters (b = bypass): keys stay in the panel. Escape closes it.
    el.addEventListener("keydown", (event) => {
        if (event.key !== "Escape") event.stopPropagation();
    });
    el.addEventListener("pointerdown", (event) => event.stopPropagation());
    el.addEventListener("wheel", (event) => event.stopPropagation(), { passive: true });

    document.body.appendChild(el);
    const stopFollow = follow(el, node);
    const onEsc = (event) => {
        if (event.key !== "Escape") return;
        const target = event.target;
        if (target && !el.contains(target) && (target.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName))) return;
        if (target?.dataset?.rename !== undefined) return; // the rename field's own Escape cancels the rename
        event.stopPropagation();
        closeMenu();
    };
    const timer = setTimeout(() => document.addEventListener("keydown", onEsc, true), 0);
    _menu = {
        el,
        node,
        refresh: render,
        cleanup() {
            clearTimeout(timer);
            stopFollow();
            document.removeEventListener("keydown", onEsc, true);
        },
    };
    body.innerHTML = `<div class="lsr-note" style="margin-top:12px">Loading…</div>`;
    refreshBadge();
    loadMenu(true).then(() => {
        if (isMenuOpenFor(node)) render();
    });
}

/* ------------------------------------------------------------ the ⚙ on the title bar */

// The ⓘ sits at size[0] - 16 and the chevron at size[0] - 36 (luna_help.mjs, luna_collapse.mjs): the ⚙ is one slot left.
const GEAR_INSET = 56;
const GEAR_R = 7;

function gearCentre(node) {
    const title = window.LiteGraph?.NODE_TITLE_HEIGHT ?? 30;
    return [node.size[0] - GEAR_INSET, -title / 2];
}

function isVue() {
    return !!window.LiteGraph?.vueNodesMode;
}

function onGear(node, pos) {
    if (isVue() || node.flags?.collapsed || !pos) return false;
    const [cx, cy] = gearCentre(node);
    return Math.hypot(pos[0] - cx, pos[1] - cy) <= GEAR_R + 3;
}

function drawGear(node, ctx) {
    if (isVue() || node.flags?.collapsed) return;
    const [cx, cy] = gearCentre(node);
    const on = node.__lunaGearHover || isMenuOpenFor(node);
    ctx.save();
    ctx.strokeStyle = ctx.fillStyle = on ? LUNA.accent : LUNA.muted;
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    ctx.arc(cx, cy, 4.2, 0, Math.PI * 2);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(cx, cy, 1.6, 0, Math.PI * 2);
    ctx.fill();
    for (let k = 0; k < 8; k++) {
        const a = (k * Math.PI) / 4;
        ctx.beginPath();
        ctx.moveTo(cx + Math.cos(a) * 4.2, cy + Math.sin(a) * 4.2);
        ctx.lineTo(cx + Math.cos(a) * 6.6, cy + Math.sin(a) * 6.6);
        ctx.stroke();
    }
    ctx.restore();
}

function wrapGear(nodeType) {
    const P = nodeType.prototype;
    const onDraw = P.onDrawForeground;
    P.onDrawForeground = function (ctx) {
        const result = onDraw?.apply(this, arguments);
        guard("gear draw", () => drawGear(this, ctx), undefined);
        return result;
    };
    const onDown = P.onMouseDown;
    P.onMouseDown = function (event, pos, canvas) {
        if (onGear(this, pos)) {
            // LiteGraph arms the title rename on a double click before it asks the node.
            if (canvas?.pointer) delete canvas.pointer.onDoubleClick;
            const recent = _lastClose.node === this && Date.now() - _lastClose.t < 400;
            if (!recent) guard("menu", () => openMenu(this), undefined);
            return true;
        }
        return onDown?.apply(this, arguments);
    };
    const onMove = P.onMouseMove;
    P.onMouseMove = function (event, pos) {
        const hover = onGear(this, pos);
        if (hover !== !!this.__lunaGearHover) {
            this.__lunaGearHover = hover;
            this.setDirtyCanvas?.(true, false);
        }
        return onMove?.apply(this, arguments);
    };
    const onLeave = P.onMouseLeave;
    P.onMouseLeave = function () {
        if (this.__lunaGearHover) {
            this.__lunaGearHover = false;
            this.setDirtyCanvas?.(true, false);
        }
        return onLeave?.apply(this, arguments);
    };
    const onDbl = P.onDblClick;
    P.onDblClick = function (event, pos) {
        if (onGear(this, pos)) return true;
        return onDbl?.apply(this, arguments);
    };
    const onRemoved = P.onRemoved;
    P.onRemoved = function () {
        if (isMenuOpenFor(this)) closeMenu();
        return onRemoved?.apply(this, arguments);
    };
}

app.registerExtension({
    name: "LunaH3Sampler.Readout",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== "Luna-H3-Sampler") return;
        wrapGear(nodeType);
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            const configured = onConfigure?.apply(this, arguments);
            guard("migrate", () => migrateSingleScheduler(this, info?.widgets_values), false);
            this.size[0] = Math.max(Number(this.size[0]) || 0, MIN_WIDTH);
            return configured;
        };
        const onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const created = onCreated?.apply(this, arguments);
            const node = this;
            this.addCustomWidget({
                type: "luna_sampler_readout",
                name: "readout",
                serialize: false,
                value: null,
                options: { serialize: false },
                computeSize() {
                    const count = guard("computeSize", () => {
                        const ctx = document.createElement("canvas").getContext("2d");
                        return readoutLines(ctx, node, readoutBox(node).w - PAD * 2).length;
                    }, 1);
                    return [0, 22 + count * LINE];
                },
                draw(ctx, _node, _width, widgetY) {
                    guard("draw", () => drawReadout(ctx, node, widgetY), undefined);
                },
            });
            // Wide enough that the LoRA widgets keep their labels next to the file name.
            node.size[0] = Math.max(Number(node.size[0]) || 0, MIN_WIDTH);
            guard("init", () => {
                shortenCombos(node);
                watchWidgets(node);
                ensureRecipes().then(() => node.setDirtyCanvas?.(true, true));
                loadMenu().then(() => node.setDirtyCanvas?.(true, true));
            }, undefined);
            return created;
        };
    },
});
