// Luna-H3-RefsOut — dims the sockets of a category that is switched off.
//
// The five switches (pictures, clips, clip_sound, sounds, song) decide what the
// node delivers. A switched-off category delivers nothing, so its output sockets
// are drawn dimmed (muted dot, muted "off" label) and come back when the switch
// goes on. Only the drawing changes; links and slots are never touched.
//
// Output slots (fixed by the node): 0-8 ref_image_N, 9-11 ref_video_N,
// 12-14 ref_video_audio_N, 15-17 ref_audio_N, 18 asset_brief, 19 audio.

import { app } from "../../scripts/app.js";
import { LUNA } from "./luna_theme.mjs";

const NODE = "Luna-H3-RefsOut";

// switch -> the output slots it governs
const CATEGORIES = {
    pictures: [0, 1, 2, 3, 4, 5, 6, 7, 8],
    clips: [9, 10, 11, 12, 13, 14],
    clip_sound: [12, 13, 14],
    sounds: [15, 16, 17],
    song: [19],
};

const OFF_MARK = " (off)";
const widgetValue = (node, name) => node.widgets?.find((w) => w.name === name)?.value;

// A switch that cannot be read counts as on: never dim by mistake.
const isOn = (node, name) => widgetValue(node, name) !== false;

function dim(slot, off) {
    if (!slot) return;
    // The slot is saved with the workflow, label and colours included, so a dimmed slot
    // can come back from a file without the remembered originals: recognise it by its mark.
    const marked = typeof slot.label === "string" && slot.label.endsWith(OFF_MARK);
    if (marked && !slot._lunaDim) {
        slot._lunaDim = { color_on: undefined, color_off: undefined, label: undefined };
    }
    if (off) {
        if (!slot._lunaDim) {
            slot._lunaDim = { color_on: slot.color_on, color_off: slot.color_off, label: slot.label };
        }
        slot.color_on = LUNA.border;
        slot.color_off = LUNA.border;
        slot.label = `${slot._lunaDim.label ?? slot.name}${OFF_MARK}`;
    } else if (slot._lunaDim) {
        const was = slot._lunaDim;
        for (const key of ["color_on", "color_off", "label"]) {
            if (was[key] === undefined) delete slot[key]; else slot[key] = was[key];
        }
        delete slot._lunaDim;
    }
}

function refresh(node) {
    try {
        const off = new Set();
        for (const [name, slots] of Object.entries(CATEGORIES)) {
            if (!isOn(node, name)) slots.forEach((i) => off.add(i));
        }
        (node.outputs || []).forEach((slot, i) => dim(slot, off.has(i)));
        node.setDirtyCanvas?.(true, true);
    } catch (error) {
        console.warn("[Luna-H3-RefsOut] dimming skipped.", error);
    }
}

function watch(node) {
    for (const name of Object.keys(CATEGORIES)) {
        const widget = node.widgets?.find((w) => w.name === name);
        if (!widget || widget._lunaRefsOut) continue;
        widget._lunaRefsOut = true;
        const previous = widget.callback;
        widget.callback = function (...args) {
            const result = previous?.apply(this, args);
            refresh(node);
            return result;
        };
    }
}

app.registerExtension({
    name: "LunaMinimaxH3.RefsOut",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== NODE) return;

        const onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onCreated?.apply(this, arguments);
            watch(this);
            refresh(this);
            return result;
        };

        // A loaded workflow sets the switches and the saved slots after creation, and the
        // frontend re-applies names by position, so refresh once now and once a frame later.
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = onConfigure?.apply(this, arguments);
            watch(this);
            refresh(this);
            requestAnimationFrame(() => refresh(this));
            return result;
        };
    },
});
