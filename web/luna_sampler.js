// Luna-H3-Sampler — the live schedule readout.
//
// The ⓘ and the chevron are registered with the pack's other nodes in
// luna_pack_theme.js. Title-bar tint stays off: ComfyUI's header stays, and the
// readout below uses the Luna palette.
//
// The readout is drawn, not sampled. It says which steps are low and which are
// high, and it checks the widgets against GET /luna_sampler/recipes. Picking a
// LoRA fills that file's strength, total steps and shifts when the recipe has them.

import { app } from "../../scripts/app.js";
import { LUNA } from "./luna_theme.mjs";

const WATCH = new Set([
    "lora_low", "lora_high",
    "strength_low", "strength_high",
    "shift_low", "shift_high",
    "audio_shift_low", "audio_shift_high",
    "cfg_low", "cfg_high",
    "scheduler", "total_steps", "transition", "highres_denoise",
]);
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

// Long model file names hide the widget label; show the file's own name without folder or extension.
function shortName(value) {
    if (!value || value === "none") return value ?? "";
    return String(value).split(/[\\/]/).pop().replace(/\.(safetensors|ckpt|pt|pth|bin)$/i, "");
}
const SHORT_NAMED = new Set(["lora_low", "lora_high", "upscaler_model"]);

function shortenCombos(node) {
    for (const widget of node.widgets || []) {
        if (!SHORT_NAMED.has(widget.name)) continue;
        widget.options = widget.options || {};
        widget.options.getOptionLabel = (value) => shortName(value);
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
        const recipe = recipes[lora];
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
            scheduler: widgetValue(node, "scheduler"),
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
    const recipe = lora && lora !== "none" ? recipes?.[lora] : null;
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

app.registerExtension({
    name: "LunaH3Sampler.Readout",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== "Luna-H3-Sampler") return;
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const configured = onConfigure?.apply(this, arguments);
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
            }, undefined);
            return created;
        };
    },
});
