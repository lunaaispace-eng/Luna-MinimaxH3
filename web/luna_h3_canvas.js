// Luna-H3-Canvas — live readout on the node.
//
// The point of this node is that the numbers are DERIVED, so showing them only
// after a run defeats it. This draws the resolved canvas, frame count and real
// duration as you turn the dials, and hides the widgets the chosen size_mode does
// not use. In `preset` mode it narrows the resolution list to the chosen ratio and
// grid; Python accepts the union of every list.
//
// The maths below mirrors h3_canvas.py. That file is the source of truth — if the
// model's constants ever change, change them there first, then here.

import { app } from "../../scripts/app.js";
import { LUNA } from "./luna_theme.mjs";

const CANVAS_MULTIPLE = 32;
const BASE_SHORT_EDGE = 768;
const MAX_PIXELS = 768 * 1344;
const FPS = 24;
const FRAME_GRID = 17;
const FRAME_PHASE = 5;
const TRAINED_MIN = 124;
const TRAINED_MAX = 362;
const LATENT_MULTIPLE = 16;
const PRESET_MIN_PIXELS = 200000;
const PRESET_MAX_PIXELS = 2000000;

// Widget order before 2026-10-25, when `multiple` and `resolution` went in under
// size_mode. Positional-only saves restore through these names.
const OLD_ORDER = [
    "ratio", "portrait", "size_mode", "megapixels", "duration_seconds",
    "interpolation_enabled", "interpolation_factor",
];

// Landscape form only; the `portrait` widget flips it. See h3_canvas.py.
const ASPECT_RATIOS = {
    "2.39:1 (anamorphic)": [239, 100],
    "21:9 (ultrawide)": [21, 9],
    "2:1": [2, 1],
    "1.91:1 (link preview)": [191, 100],
    "16:9 (widescreen)": [16, 9],
    "16:10": [16, 10],
    "3:2 (photo)": [3, 2],
    "4:3": [4, 3],
    "5:4": [5, 4],
    "1:1 (square)": [1, 1],
};

const roundTo = (v, multiple = CANVAS_MULTIPLE) =>
    Math.max(multiple, Math.round(v / multiple) * multiple);

// JS % keeps the sign of the dividend, Python's does not — hence the double mod.
// Without it every frame count below the phase lands wrong.
const alignFrames = (n) => {
    n = Math.max(FRAME_PHASE, Math.round(n));
    return n + ((((FRAME_PHASE - n) % FRAME_GRID) + FRAME_GRID) % FRAME_GRID);
};

function h3Canvas(ratio) {
    let short = BASE_SHORT_EDGE;
    let long = BASE_SHORT_EDGE * Math.max(ratio, 1 / ratio);
    if (short * long > MAX_PIXELS) {
        const shrink = Math.sqrt(MAX_PIXELS / (short * long));
        short *= shrink;
        long *= shrink;
    }
    const [w, h] = ratio >= 1 ? [long, short] : [short, long];
    return [roundTo(w), roundTo(h)];
}

function megapixelCanvas(ratio, megapixels, multiple) {
    const area = Math.max(0.01, megapixels) * 1e6;
    return [roundTo(Math.sqrt(area * ratio), multiple), roundTo(Math.sqrt(area / ratio), multiple)];
}

// num / den rounded half-up to a multiple, in integers, the same sum as Python's
// _nearest_step. The preset lists must match h3_canvas.py label for label.
const nearestStep = (num, den, multiple) =>
    Math.floor((2 * num + den * multiple) / (2 * den * multiple)) * multiple;

function presetSizes(ratioW, ratioH, multiple) {
    const sizes = new Map();
    const add = (w, h) => sizes.set(`${w}x${h}`, [w, h]);
    // A finer grid holds every size of the coarser ones, as in h3_canvas.py.
    if (multiple * 2 <= CANVAS_MULTIPLE) {
        for (const [w, h] of presetSizes(ratioW, ratioH, multiple * 2)) add(w, h);
    }
    for (let edge = multiple; edge * edge * ratioW <= 2 * PRESET_MAX_PIXELS * ratioH; edge += multiple) {
        add(Math.max(edge, nearestStep(edge * ratioW, ratioH, multiple)), edge);
    }
    for (let edge = multiple; edge * edge * ratioH <= 2 * PRESET_MAX_PIXELS * ratioW; edge += multiple) {
        add(edge, Math.min(edge, Math.max(multiple, nearestStep(edge * ratioH, ratioW, multiple))));
    }
    add(...h3Canvas(ratioW / ratioH));
    return [...sizes.values()]
        .filter(([w, h]) => w * h >= PRESET_MIN_PIXELS && w * h <= PRESET_MAX_PIXELS)
        .sort((a, b) => a[0] - b[0] || a[1] - b[1]);
}

function presetLabel(w, h) {
    const hundredths = Math.floor((w * h + 5000) / 10000);
    return `${w}x${h} (${Math.floor(hundredths / 100)}.${String(hundredths % 100).padStart(2, "0")} MP)`;
}

function presetLabels(ratioName, multiple) {
    const [rw, rh] = ASPECT_RATIOS[ratioName] || [16, 9];
    return presetSizes(rw, rh, Number(multiple) || CANVAS_MULTIPLE).map(([w, h]) => presetLabel(w, h));
}

const labelSize = (label) => String(label).split(" ")[0].split("x").map(Number);

const widgetValue = (node, name) => node.widgets?.find((w) => w.name === name)?.value;

function resolve(node) {
    const base = ASPECT_RATIOS[widgetValue(node, "ratio")] || [16, 9];
    const pair = widgetValue(node, "portrait") ? [base[1], base[0]] : base;
    const ratio = pair[0] / pair[1];
    const mode = widgetValue(node, "size_mode");
    const multiple = Number(widgetValue(node, "multiple")) || CANVAS_MULTIPLE;
    let width, height;
    if (mode === "megapixels") {
        [width, height] = megapixelCanvas(ratio, Number(widgetValue(node, "megapixels")) || 1.03, multiple);
    } else if (mode === "preset") {
        [width, height] = labelSize(widgetValue(node, "resolution"));
        if (widgetValue(node, "portrait")) [width, height] = [height, width];
    } else {
        [width, height] = h3Canvas(ratio);
    }

    const asked = Number(widgetValue(node, "duration_seconds")) || 0;
    const length = alignFrames(asked * FPS);
    const enabled = widgetValue(node, "interpolation_enabled") !== false;
    const factor = enabled
        ? Math.max(1, Number(widgetValue(node, "interpolation_factor")) || 1)
        : 1;
    const seconds = length / FPS;

    const notes = [];
    // Half a grid step — see h3_canvas.py. Below that the snap is unavoidable and
    // the main line already shows the real duration.
    if (Math.abs(seconds - asked) >= FRAME_GRID / FPS / 2) notes.push(`asked ${asked}s`);
    if (length < TRAINED_MIN || length > TRAINED_MAX) notes.push("outside trained range");
    if (width * height > MAX_PIXELS) {
        notes.push(`${(width * height / MAX_PIXELS).toFixed(2)}x default canvas`);
    }
    if (width % LATENT_MULTIPLE || height % LATENT_MULTIPLE) {
        const floor = (v) => Math.floor(v / LATENT_MULTIPLE) * LATENT_MULTIPLE;
        notes.push(`H3 video makes ${floor(width)}x${floor(height)}`);
    }

    return { width, height, length, seconds, outputFps: FPS * factor, factor, notes, pair };
}

// The resolution list holds only the chosen ratio's sizes on the chosen grid. A
// value that left the list (ratio or grid changed) moves to the nearest area, so the
// run never meets a size Python would refuse.
function syncPresets(node) {
    const widget = node.widgets?.find((w) => w.name === "resolution");
    if (!widget) return;
    const labels = presetLabels(widgetValue(node, "ratio"), widgetValue(node, "multiple"));
    if (!labels.length) return;
    // Set in place: the frontend keeps its own reference to the options object.
    if (widget.options) widget.options.values = labels;
    if (labels.includes(widget.value)) return;
    const [w0, h0] = labelSize(widget.value);
    const area = (w0 * h0) || 768 * 1344;
    let best = labels[0];
    for (const label of labels) {
        const [w, h] = labelSize(label);
        const [bw, bh] = labelSize(best);
        if (Math.abs(w * h - area) < Math.abs(bw * bh - area)) best = label;
    }
    widget.value = best;
}

// Hide the widgets that currently do nothing: `megapixels` outside megapixels mode,
// `resolution` outside preset mode, `multiple` in H3 canvas mode (the model's own
// canvas is always /32), and `interpolation_factor` while interpolation is gated off.
// widget.hidden is the flag that actually works — computeSize skips hidden widgets,
// so the node shrinks too.
function syncWidgets(node) {
    syncPresets(node);
    const mode = widgetValue(node, "size_mode");
    const wanted = {
        megapixels: mode !== "megapixels",
        resolution: mode !== "preset",
        multiple: mode !== "megapixels" && mode !== "preset",
        interpolation_factor: widgetValue(node, "interpolation_enabled") === false,
    };
    let changed = false;
    for (const [name, hidden] of Object.entries(wanted)) {
        const widget = node.widgets?.find((w) => w.name === name);
        if (!widget || widget.hidden === hidden) continue;
        widget.hidden = hidden;
        changed = true;
    }
    if (!changed) return;
    node.setSize(node.computeSize());
    node.setDirtyCanvas(true, true);
}

function drawReadout(ctx, node, width, y) {
    const r = resolve(node);
    const pad = 10;
    const x = 14;
    // node.size[0] is authoritative. The width LiteGraph hands a custom widget is
    // not always the node's, and trusting it drew the panel off the side of the node.
    const outer = Number(node?.size?.[0]) || Number(width) || 240;
    const w = Math.max(120, outer - x * 2);
    const h = r.notes.length ? 74 : 56;

    // save() must be paired with restore() even if something below throws, or every
    // node drawn after this one inherits our fill style, font and path.
    ctx.save();
    try {
    ctx.beginPath();
    // roundRect is not on every canvas implementation; a plain rect is a fine panel.
    if (typeof ctx.roundRect === "function") ctx.roundRect(x, y, w, h, 6);
    else ctx.rect(x, y, w, h);
    ctx.fillStyle = LUNA.panel;
    ctx.fill();
    ctx.strokeStyle = LUNA.border;
    ctx.lineWidth = 1;
    ctx.stroke();

    ctx.textBaseline = "alphabetic";
    ctx.textAlign = "left";
    ctx.fillStyle = LUNA.accent;
    ctx.font = "600 16px sans-serif";
    ctx.fillText(`${r.width} × ${r.height}`, x + pad, y + 24);

    ctx.textAlign = "right";
    ctx.fillStyle = LUNA.muted;
    ctx.font = "11px sans-serif";
    ctx.fillText(
        `${r.pair[0]}:${r.pair[1]}   ${(r.width * r.height / 1e6).toFixed(2)} MP`,
        x + w - pad, y + 24,
    );

    ctx.textAlign = "left";
    ctx.fillStyle = LUNA.text;
    ctx.font = "11px sans-serif";
    const rate = r.factor > 1 ? `${FPS} → ${r.outputFps} fps` : `${FPS} fps`;
    ctx.fillText(`${r.length} frames  ${r.seconds.toFixed(2)}s  ${rate}`, x + pad, y + 44);

    if (r.notes.length) {
        ctx.fillStyle = LUNA.danger;
        ctx.font = "11px sans-serif";
        ctx.fillText(r.notes.join("   "), x + pad, y + 63);
    }
    } finally {
        ctx.restore();
    }
}

// LiteGraph draws every node in one pass and does not catch. An exception thrown
// from a widget's draw aborts the rest of that frame, so unrelated nodes render
// without their widgets and the graph looks broken. Nothing in this file may throw
// into the host — hence the guards on every entry point.
function guard(label, fn, fallback) {
    try {
        return fn();
    } catch (error) {
        if (!guard.reported.has(label)) {
            guard.reported.add(label);
            console.error(`[Luna-H3-Canvas] ${label} failed; readout disabled.`, error);
        }
        return fallback;
    }
}
guard.reported = new Set();

app.registerExtension({
    name: "LunaMinimaxH3.H3Canvas",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== "Luna-H3-Canvas") return;
        nodeData.fallbackWidgetsValuesNames = OLD_ORDER;

        const onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onCreated?.apply(this, arguments);
            const node = this;

            this.addCustomWidget({
                type: "luna_h3_readout",
                name: "readout",
                // Not a real input: never serialise it into the workflow.
                serialize: false,
                value: null,
                options: { serialize: false },
                computeSize() {
                    return [0, guard("computeSize", () => (resolve(node).notes.length ? 82 : 64), 64)];
                },
                draw(ctx, _node, widgetWidth, widgetY) {
                    guard("draw", () => drawReadout(ctx, node, widgetWidth, widgetY), undefined);
                },
            });

            // Relayout whenever a widget feeding the readout changes. Closure over
            // `node` rather than touching `this` — LiteGraph calls widget callbacks
            // as (value, canvas, node, pos, event) with `this` set to the widget,
            // and the original callback needs that left alone.
            for (const widget of this.widgets || []) {
                if (widget.type === "luna_h3_readout") continue;
                const previous = widget.callback;
                widget.callback = function (...args) {
                    const result = previous?.apply(this, args);
                    guard("sync", () => {
                        syncWidgets(node);
                        node.setDirtyCanvas(true, true);
                    }, undefined);
                    return result;
                };
            }

            guard("init", () => {
                syncWidgets(this);
                this.setSize(this.computeSize());
            }, undefined);
        };

        // Loading a saved workflow restores widget values without firing callbacks.
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            onConfigure?.apply(this, arguments);
            guard("configure", () => syncWidgets(this), undefined);
        };
    },
});
