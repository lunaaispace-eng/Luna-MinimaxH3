// Applies the Luna house theme to this pack's nodes.
//
// Roles come from the section tints in luna_theme.mjs. One line per node — this
// file is the per-pack map; the three luna_*.mjs modules it imports are the shared
// Luna frontend kit, copied verbatim from Luna-Media-Tools (same author, Apache-2.0).
// There is no cross-pack import in ComfyUI's frontend, so a fix in one copy is
// copied across by hand.

import { app } from "../../scripts/app.js";
import { registerLunaTheme } from "./luna_theme.mjs";
import { registerLunaHelp } from "./luna_help.mjs";
import { registerLunaCollapse } from "./luna_collapse.mjs";

// Title-bar tinting is off for now — Peti prefers ComfyUI's default red header,
// with the widget's own palette inside. The role map stays here so turning it
// back on is uncommenting one call; the colours are also editable in
// Settings > Luna > Node colours.
//
// registerLunaTheme(app, {
//     "Luna-H3-Canvas": "input",
//     "Luna-H3-ReferenceLoader": "input",
//     "Luna-H3-AssetLoader": "input",
//     "Luna-H3-ShotList": "input",
//     "Luna-H3-LoopStart": "processing",
//     "Luna-H3-LoopScene": "processing",
//     "Luna-H3-LoopProfile": "processing",
//     "Luna-H3-GuideApply": "processing",
//     "Luna-H3-GuideClip": "processing",
//     "Luna-H3-ConcatAVLatent": "processing",
//     "Luna-H3-RefsOut": "processing",
//     "Luna-H3-Sampler": "processing",
//     "Luna-H3-LoopEnd": "output",
// }, "LunaMinimaxH3.Theme");

// The ⓘ on the title bar. Content is each node's own description plus its input
// tooltips (and, for Luna-H3-AssetLoader, its output tooltips), so there is no
// separate help file to keep up to date. All 13 nodes have it.
registerLunaHelp(app, [
    "Luna-H3-Canvas", "Luna-H3-ConcatAVLatent", "Luna-H3-ReferenceLoader", "Luna-H3-AssetLoader",
    "Luna-H3-RefsOut", "Luna-H3-ShotList", "Luna-H3-LoopStart", "Luna-H3-LoopScene",
    "Luna-H3-LoopProfile", "Luna-H3-GuideApply", "Luna-H3-GuideClip", "Luna-H3-LoopEnd",
    "Luna-H3-Sampler",
], "LunaMinimaxH3.Help");

// The chevron beside it folds the settings away, leaving the sockets and whatever the
// node draws for itself: Luna-H3-Canvas keeps its readout, Luna-H3-ShotList its scene table,
// Luna-H3-AssetLoader its cards, Luna-H3-Sampler its schedule readout. Luna-H3-ReferenceLoader is nothing but settings (27 stock
// widgets), so folding it leaves the five sockets a wired graph needs to show. The other
// nodes are a few widgets and sockets; folding hides the widgets.
registerLunaCollapse(app, [
    "Luna-H3-Canvas", "Luna-H3-ConcatAVLatent", "Luna-H3-ReferenceLoader", "Luna-H3-AssetLoader",
    "Luna-H3-RefsOut", "Luna-H3-ShotList", "Luna-H3-LoopStart", "Luna-H3-LoopScene",
    "Luna-H3-LoopProfile", "Luna-H3-GuideApply", "Luna-H3-GuideClip", "Luna-H3-LoopEnd",
    "Luna-H3-Sampler",
], "LunaMinimaxH3.Collapse");
