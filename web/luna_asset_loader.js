// Luna-H3-AssetLoader — ordered reference cards and every output, always.
//
// Two jobs in one file:
//   1. keep the node's outputs exactly Python's: the ten fixed outputs, image_1..image_9
//      whatever the number of picture cards, then fps and vocals (SPEC § 2), and bring
//      older saves to that order without moving a saved link (the output migration block)
//   2. a card UI that owns the image_paths field: upload, reorder, remove
//
// On (1): ComfyUI has no dynamic outputs. Autogrow is input-only
// (comfy_api/latest/_io.py — `class Autogrow(ComfyTypeI)`), and RETURN_TYPES is a
// fixed class attribute; the frontend sends each link's own slot number with the
// prompt and the backend reads that index of the result. So no slot may come or go:
// image_count only labels the image sockets past it as spare.
//
// The upload flow uses ComfyUI's stock /upload/image endpoint, the same one Load Image posts to.

import { app } from "../../scripts/app.js";
import { LUNA } from "./luna_theme.mjs";
import { keepNodeWidth } from "./luna_widget_width.mjs";

const NODE = "Luna-H3-AssetLoader";
const MAX = 9;
const PATHS_WIDGET = "image_paths";
const COUNT_WIDGET = "image_count";
const PREVIEW_H = 170;   // height of the selected-image preview, in pixels

/* ------------------------------------------------------------------ styles */

// Palette comes from luna_theme.mjs — the single place the Luna colours are
// defined for this pack (itself a copy of ComfyUI-LunaRunner/web/style.css, the
// house scheme). Declared under luna- prefixed CSS variables so they cannot
// collide with ComfyUI's own. Nothing here hardcodes a colour.
//
//   accent  #e0a458  warm amber          text  #e7e5df   muted   #8b8e99
//   panel   #1c1e25  border #2e313c      radius 8px
//   section tints: input #7bb47b · processing #7da4c8 · output #d98a6a
//
// One accent throughout: the amber carries the badges, the buttons, the active
// options and the selection. The section tints stay defined in luna_theme.mjs but
// are not used here — a second colour on the same node just read as noise.
const CSS = `
.luna-al { --luna-accent:${LUNA.accent}; --luna-accent-soft:${LUNA.accentSoft}; --luna-panel:${LUNA.panel};
           --luna-border:${LUNA.border}; --luna-text:${LUNA.text}; --luna-muted:${LUNA.muted};
           --luna-input:${LUNA.input}; --luna-danger:${LUNA.danger}; --luna-bg:${LUNA.bg};
           --luna-radius:${LUNA.radius}px;
           display:flex; flex-direction:column; gap:6px; padding:6px; box-sizing:border-box;
           font-family:"Segoe UI", system-ui, sans-serif; font-size:11px; color:var(--luna-text); }
.luna-al-bar { display:flex; gap:6px; flex-wrap:wrap; align-items:center; }
.luna-al-btn { flex:0 0 auto; padding:5px 10px; border-radius:6px; cursor:pointer;
               background:var(--luna-panel); border:1px solid var(--luna-border);
               color:var(--luna-text); transition:background .1s, border-color .1s, color .1s; }
.luna-al-btn:hover { background:var(--luna-accent); border-color:var(--luna-accent); color:var(--luna-bg); }
.luna-al-btn.danger:hover { background:var(--luna-danger); border-color:var(--luna-danger); color:#fff; }
.luna-al-count { margin-left:auto; color:var(--luna-muted); }
/* Transient one-liner for the one thing the node refuses to do silently: a slot
   that will not go away because it is wired. Everything else just happens. */
.luna-al-note { color:var(--luna-accent); }
.luna-al-cards { display:grid; grid-template-columns:repeat(auto-fill, minmax(112px, 1fr));
                 gap:8px; overflow-y:auto; flex:1 1 auto; align-content:start; padding:2px; }
.luna-al-card { position:relative; border-radius:8px; overflow:hidden; cursor:grab;
                background:var(--luna-panel); border:1px solid var(--luna-border); transition:border-color .1s; }
.luna-al-card:hover { border-color:var(--luna-accent); }
.luna-al-card.drag { opacity:.35; }
.luna-al-card.over { border-color:var(--luna-accent); box-shadow:0 0 0 1px var(--luna-accent) inset; }
/* The image is the card. Everything else floats on top of it, so the picture
   gets the space instead of a column of text beside a postage stamp. */
.luna-al-thumb { display:block; width:100%; aspect-ratio:1/1; object-fit:contain;
                 background:var(--luna-bg); }
.luna-al-pic { position:absolute; top:4px; left:4px; padding:1px 6px; border-radius:5px;
               background:var(--luna-accent); color:var(--luna-bg); font-weight:700; font-size:10px; }
.luna-al-x { position:absolute; top:4px; right:4px; width:18px; height:18px; line-height:17px;
             text-align:center; border-radius:5px; cursor:pointer; font-size:11px;
             background:rgba(20,21,26,.85); color:var(--luna-muted); opacity:0; transition:opacity .1s; }
.luna-al-card:hover .luna-al-x { opacity:1; }
.luna-al-x:hover { background:var(--luna-danger); color:#fff; }
.luna-al-cap { position:absolute; left:0; right:0; bottom:0; padding:3px 5px; font-size:10px;
               background:linear-gradient(to top, rgba(12,12,15,.95), rgba(12,12,15,0)); }
.luna-al-dim { color:var(--luna-text); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.luna-al-name { opacity:.45; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.luna-al-preview { position:relative; border:1px solid var(--luna-border); border-radius:8px;
                   background:var(--luna-bg); overflow:hidden; flex:0 0 auto; }
.luna-al-preview img { display:block; width:100%; height:100%; object-fit:contain; }
.luna-al-pvlbl { position:absolute; left:6px; bottom:5px; padding:1px 6px; border-radius:5px;
                 background:rgba(20,21,26,.82); font-size:10px; color:var(--luna-text); }
.luna-al-empty { opacity:.55; padding:10px 4px; }
.luna-al-warn { color:var(--luna-accent); padding:2px 4px; }
.luna-al-card.sel { border-color:var(--luna-accent); box-shadow:0 0 0 1px var(--luna-accent) inset; }

/* Resize panel. The state schema behind it — the key names, their defaults and
   the JSON-blob approach — is modelled on ComfyUI-Pixaroma (MIT, Copyright (c)
   2026 pixaroma), nodes/_resize_helpers.py; see the header of asset_loader.py.
   Theirs edits one image so the panel is always on; ours holds up to nine, so it
   edits whichever card is selected.
   NOTE: no backticks anywhere in this block. It lives inside the CSS template
   literal, so one would close the string and turn the rest of the file into
   code. That happened once and broke the whole extension; node --check still
   passed, because the wreckage is still valid syntax. Only loading it catches
   this, so check the browser, not the linter. */
.luna-al-panel { border-top:1px solid var(--luna-border); padding-top:6px; display:flex;
                 flex-direction:column; gap:5px; }
.luna-al-row { display:flex; gap:4px; align-items:center; flex-wrap:wrap; }
.luna-al-lbl { color:var(--luna-muted); min-width:44px; }
.luna-al-seg { flex:1 1 auto; display:flex; gap:3px; }
.luna-al-opt { flex:1 1 auto; text-align:center; padding:3px 5px; border-radius:5px; cursor:pointer;
               background:var(--luna-panel); border:1px solid var(--luna-border); color:var(--luna-text); font-size:10px;
               white-space:nowrap; }
.luna-al-opt:hover { border-color:var(--luna-accent); }
.luna-al-opt.on { background:var(--luna-accent); border-color:var(--luna-accent); color:var(--luna-bg); }
.luna-al-num { width:64px; padding:3px 5px; border-radius:5px; background:var(--luna-bg);
               border:1px solid var(--luna-border); color:var(--luna-text); font-size:10px; }
.luna-al-io { opacity:.75; font-size:10px; }
.luna-al-io b { color:var(--luna-accent); font-weight:600; }

/* Sound cards and the song editor (v3). Pink marks sound everywhere, the same
   colour ComfyUI gives AUDIO sockets, so a sound card cannot be mistaken for a
   picture. Still no backticks in here. */
.luna-al-card.snd { cursor:pointer; }
.luna-al-snd { display:block; width:100%; aspect-ratio:1/1; background:var(--luna-bg); }
.luna-al-pic.snd { background:#e07bb4; }
.luna-al-aed { display:flex; flex-direction:column; gap:6px; border:1px solid var(--luna-border);
               border-radius:8px; background:var(--luna-bg); padding:7px; }
.luna-al-ahead { display:flex; align-items:center; gap:8px; color:var(--luna-muted); font-size:10.5px; }
.luna-al-ahead b { color:#e07bb4; font-size:11px; white-space:nowrap; flex:0 0 auto; }
.luna-al-ahead .nm { min-width:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.luna-al-ahead .sp { flex:1 1 auto; }
.luna-al-play { width:26px; height:20px; border-radius:5px; border:1px solid var(--luna-border);
                background:var(--luna-panel); color:var(--luna-text); text-align:center; line-height:18px; cursor:pointer; }
.luna-al-play:hover { border-color:var(--luna-accent); color:var(--luna-accent); }
.luna-al-wave { position:relative; height:64px; border-radius:6px; background:#101116; overflow:hidden; cursor:crosshair; }
.luna-al-wave canvas, .luna-al-zoom canvas { position:absolute; inset:0; width:100%; height:100%; }
.luna-al-win { position:absolute; top:0; bottom:0; background:#e0a45826; border-left:2px solid var(--luna-accent);
               border-right:2px solid var(--luna-accent); cursor:grab; box-sizing:border-box; }
.luna-al-win .h { position:absolute; top:0; bottom:0; width:10px; cursor:ew-resize; }
.luna-al-win .h.l { left:-6px; } .luna-al-win .h.r { right:-6px; }
.luna-al-win.locked .h { cursor:not-allowed; }
.luna-al-win .tag { position:absolute; top:2px; left:4px; font-size:10px; color:var(--luna-accent);
                    white-space:nowrap; text-shadow:0 1px 2px #000; pointer-events:none; }
.luna-al-ph { position:absolute; top:0; bottom:0; width:2px; background:#fff; display:none; pointer-events:none; }
.luna-al-ruler { position:relative; height:12px; font-size:9.5px; color:var(--luna-muted); }
.luna-al-ruler span { position:absolute; transform:translateX(-50%); white-space:nowrap; }
.luna-al-zoom { position:relative; height:40px; border-radius:6px; background:#101116; overflow:hidden; cursor:ew-resize; }
.luna-al-zlbl { display:flex; justify-content:space-between; font-size:10px; color:var(--luna-muted); }
.luna-al-nums { display:grid; grid-template-columns:auto auto auto 1fr; gap:5px 7px; align-items:center; font-size:10.5px; }
.luna-al-nums .l { color:var(--luna-muted); }
.luna-al-nf { display:flex; align-items:center; gap:3px; }
.luna-al-nf input { width:74px; padding:3px 5px; border-radius:5px; background:#0d0e12; border:1px solid var(--luna-border);
                    color:var(--luna-text); font:11px ui-monospace, Consolas, monospace; text-align:center; }
.luna-al-nf input:focus { outline:none; border-color:var(--luna-accent); }
.luna-al-nf input.lockd { color:var(--luna-accent); border-color:#e0a45855; background:#e0a45810; }
.luna-al-nud { width:20px; height:20px; border-radius:4px; border:1px solid var(--luna-border); background:var(--luna-panel);
               color:var(--luna-muted); text-align:center; line-height:18px; cursor:pointer; user-select:none; }
.luna-al-nud:hover { color:var(--luna-accent); border-color:var(--luna-accent); }
.luna-al-lock { font-size:10px; color:var(--luna-accent); white-space:nowrap; }
.luna-al-astat { font-size:10px; color:var(--luna-muted); border-top:1px solid var(--luna-border); padding-top:5px; line-height:1.45; }
.luna-al-astat b { color:var(--luna-text); font-weight:600; }
.luna-al-astat .w { color:#f0a49a; }

/* Song zoom (v3): − + Fit above the waveform, an overview bar under it when zoomed. */
.luna-al-ztools { display:flex; align-items:center; gap:4px; font-size:10.5px; color:var(--luna-muted); }
.luna-al-ztools .l { margin-right:2px; }
.luna-al-ztools .zfit { width:auto; padding:0 6px; }
.luna-al-ztools .zx { margin-left:4px; color:var(--luna-text); }
.luna-al-ztools .sp { flex:1 1 auto; }
.luna-al-ovw { position:relative; height:10px; border-radius:4px; background:#101116; cursor:pointer; overflow:hidden; }
.luna-al-ovw .vw { position:absolute; top:0; bottom:0; background:#ffffff1f; border:1px solid #8b8e99; border-radius:3px; box-sizing:border-box; cursor:grab; }
.luna-al-ovw .wn { position:absolute; top:3px; bottom:3px; background:#e07bb4cc; pointer-events:none; }
/* Long readouts ("577 frames at 24 fps · from …") wrap instead of running past the edge. */
.luna-al-nums { grid-template-columns:auto minmax(0, max-content) auto minmax(0, 1fr) !important; }
.luna-al-nf { flex-wrap:wrap; min-width:0; }
.luna-al-lock { white-space:normal !important; min-width:0; }

/* Video cards and the clip editor (v3). Orange marks clips, pink sound. */
.luna-al-pic.vid { background:#d98a6a; }
.luna-al-vthumb { display:block; width:100%; aspect-ratio:1/1; object-fit:cover; background:var(--luna-bg); }
.luna-al-ved { display:flex; flex-direction:column; gap:6px; border:1px solid var(--luna-border);
               border-radius:8px; background:var(--luna-bg); padding:7px; }
.luna-al-ved .luna-al-ahead b { color:#d98a6a; }
.luna-al-video { display:block; width:100%; height:150px; background:#000; border-radius:6px; object-fit:contain; }
.luna-al-strip { position:relative; height:46px; border-radius:6px; background-color:#101116;
                 background-size:100% 100%; background-repeat:no-repeat; overflow:hidden; cursor:crosshair; }
.luna-al-sect { font-size:10.5px; color:var(--luna-accent); font-weight:600; margin-top:2px; }

/* The three rows (v3): Pictures, Clips, Sounds — each kind on its own. */
.luna-al-sec { display:flex; flex-direction:column; gap:6px; border-top:1px solid var(--luna-border); padding-top:6px; }
.luna-al-sech { display:flex; align-items:center; gap:8px; font-size:11.5px; font-weight:600; }
.luna-al-sec.k-p .luna-al-sech .tt { color:var(--luna-accent); }
.luna-al-sec.k-v .luna-al-sech .tt { color:#d98a6a; }
.luna-al-sec.k-a .luna-al-sech .tt { color:#e07bb4; }
.luna-al-sech .n { color:var(--luna-muted); font-weight:400; }
.luna-al-sech .sp { flex:1 1 auto; }
.luna-al-sech .add { padding:2px 9px; font-weight:400; }
.luna-al-sec .luna-al-empty { padding:4px 2px; font-size:10.5px; }

/* The shot timeline (v3): the video being generated, one lane per kind. */
.luna-al-tl { border:1px solid var(--luna-border); border-radius:8px; background:var(--luna-bg); padding:7px;
              display:flex; flex-direction:column; gap:4px; }
.luna-al-tlh { display:flex; align-items:center; gap:7px; font-size:10.5px; color:var(--luna-muted); }
.luna-al-tlh b { color:var(--luna-text); }
.luna-al-tlh .sp { flex:1 1 auto; }
.luna-al-lane { display:grid; grid-template-columns:76px 1fr; align-items:center; gap:6px; }
.luna-al-lane .ln { font-size:10px; color:var(--luna-muted); text-align:right; }
.luna-al-track { position:relative; height:24px; border-radius:5px; background:#101116; border:1px solid #1f2129; }
.luna-al-truler { position:relative; height:13px; font-size:9.5px; color:var(--luna-muted); }
.luna-al-truler span { position:absolute; transform:translateX(-50%); white-space:nowrap; }
.luna-al-blk { position:absolute; top:2px; bottom:2px; border-radius:4px; cursor:grab; font-size:9.5px; font-weight:600;
               color:var(--luna-bg); display:flex; align-items:center; padding:0 5px; white-space:nowrap; overflow:hidden;
               box-sizing:border-box; min-width:6px; }
.luna-al-blk.vid { background:#d98a6a; } .luna-al-blk.snd { background:#e07bb4; }
.luna-al-blk.key { width:20px; margin-left:-10px; justify-content:center; padding:0; background:var(--luna-accent); }
.luna-al-blk.sel { box-shadow:0 0 0 2px #fff; }
/* One row that scrolls sideways: the timeline's height is fixed arithmetic (TIMELINE), so the chips never wrap. */
.luna-al-refs { display:flex; gap:4px; flex-wrap:nowrap; align-items:center; min-height:22px; overflow-x:auto;
                overflow-y:hidden; scrollbar-width:thin; min-width:0; }
.luna-al-chip { font-size:9.5px; font-weight:700; padding:2px 6px; border-radius:5px; cursor:pointer; color:var(--luna-bg);
                white-space:nowrap; flex:0 0 auto; }
.luna-al-chip.p { background:var(--luna-accent); } .luna-al-chip.v { background:#d98a6a; } .luna-al-chip.a { background:#e07bb4; }
.luna-al-tlnote { font-size:10px; color:var(--luna-muted); line-height:1.4; }

/* The settings panel (v3), after LLM Prompt's. Fixed colours: it lives on
   document.body, outside the node's variables. */
.luna-al-sp { position:fixed; z-index:1300; width:330px; transform-origin:0 0; background:#1c1e25;
              border:1px solid #e0a45866; border-radius:8px; box-shadow:0 10px 30px #000a;
              font:12px "Segoe UI", system-ui, sans-serif; color:#e7e5df; }
.luna-al-sp .hd { display:flex; align-items:center; gap:8px; padding:9px 12px; border-bottom:1px solid #2e313c; }
.luna-al-sp .hd .t { flex:1; font-weight:600; font-size:13px; }
.luna-al-sp .x, .luna-al-sp .rs, .luna-al-sp .dn { cursor:pointer; color:#8b8e99; }
.luna-al-sp .x:hover, .luna-al-sp .rs:hover, .luna-al-sp .dn:hover { color:#e0a458; }
.luna-al-sp .bd { padding:2px 12px 10px; }
.luna-al-sp .sh { color:#e0a458; font-weight:600; font-size:12px; margin:10px 0 5px; }
.luna-al-sp .rw { display:grid; grid-template-columns:110px 1fr; align-items:center; gap:8px; margin:5px 0; }
.luna-al-sp .ch { display:flex; gap:4px; flex-wrap:wrap; align-items:center; color:#8b8e99; }
.luna-al-sp .ch span { padding:2px 8px; border:1px solid #2e313c; border-radius:4px; color:#e7e5df; cursor:pointer; }
.luna-al-sp .ch span:hover { border-color:#8b8e99; }
.luna-al-sp .ch span.on { border-color:#e0a458; color:#e0a458; background:#e0a45814; }
.luna-al-sp .nb { width:56px; background:#14151a; border:1px solid #2e313c; border-radius:4px; color:#e7e5df; text-align:center; padding:2px 0; }
.luna-al-sp .nt { font-size:11px; color:#8b8e99; margin:0 0 4px 118px; line-height:1.4; }
.luna-al-sp .ft { border-top:1px solid #2e313c; padding:7px 12px; display:flex; justify-content:space-between; }
`;

function injectCSS() {
    if (document.getElementById("luna-al-css")) return;
    const s = document.createElement("style");
    s.id = "luna-al-css";
    s.textContent = CSS;
    document.head.appendChild(s);
}

/* ------------------------------------------------------------------- utils */

const widget = (node, name) => node.widgets?.find((w) => w.name === name);

function readPaths(node) {
    const w = widget(node, PATHS_WIDGET);
    return String(w?.value ?? "")
        .replace(/\r/g, "\n").split("\n")
        .map((s) => s.trim().replace(/^"|"$/g, ""))
        .filter(Boolean);
}

// Set while writePaths fires the widget callback, so the hand-edit hook on
// image_paths can tell a user typing from the cards writing — otherwise every
// card action would sync and re-render twice.
let internalWrite = false;

function writePaths(node, list) {
    const w = widget(node, PATHS_WIDGET);
    if (!w) return;
    w.value = list.join("\n");
    internalWrite = true;
    try { w.callback?.(w.value); } finally { internalWrite = false; }
    node.graph?.setDirtyCanvas(true, true);
}

// --- derived accent -------------------------------------------------------
// The widget's accent follows node.color, so ComfyUI's right-click > Colors (or
// our role theme) recolours the buttons, chips and badges too. Node colours are
// dark title-bar tones, so they are lifted to a usable lightness first.
const DEFAULT_ACCENT = LUNA.accent;   // one source: luna_theme.mjs

// targetL 0.55 rather than 0.60, and a saturation WINDOW rather than a floor.
// ComfyUI's palette is deliberately muted; forcing saturation up to 0.38 turned
// its plum (#422342, hue 300) into hot pink. Keeping the source saturation
// preserves each colour's character — the floor only rescues near-greys, and the
// ceiling stops cyan (saturation 1.0) from arriving as neon.
function brighten(hex, targetL = 0.55, minS = 0.18, maxS = 0.72) {
    const m = /^#?([0-9a-f]{3}|[0-9a-f]{6})$/i.exec(String(hex || ""));
    if (!m) return null;
    let h = m[1];
    if (h.length === 3) h = h.split("").map((c) => c + c).join("");
    const r = parseInt(h.slice(0, 2), 16) / 255,
          g = parseInt(h.slice(2, 4), 16) / 255,
          b = parseInt(h.slice(4, 6), 16) / 255;
    const mx = Math.max(r, g, b), mn = Math.min(r, g, b), d = mx - mn;
    let H = 0;
    if (d) {
        if (mx === r) H = ((g - b) / d) % 6;
        else if (mx === g) H = (b - r) / d + 2;
        else H = (r - g) / d + 4;
    }
    H *= 60; if (H < 0) H += 360;
    const L0 = (mx + mn) / 2;
    let S = d === 0 ? 0 : d / (1 - Math.abs(2 * L0 - 1));
    // A near-grey node colour has no hue worth keeping — leave the default amber
    // rather than turning the whole panel grey.
    if (S < 0.06) return null;
    S = Math.min(maxS, Math.max(S, minS));
    const C = (1 - Math.abs(2 * targetL - 1)) * S, X = C * (1 - Math.abs(((H / 60) % 2) - 1)),
          mm = targetL - C / 2;
    const seg = [[C,X,0],[X,C,0],[0,C,X],[0,X,C],[X,0,C],[C,0,X]][Math.floor(H / 60) % 6];
    const to = (v) => Math.round((v + mm) * 255).toString(16).padStart(2, "0");
    return `#${to(seg[0])}${to(seg[1])}${to(seg[2])}`;
}

// Accent derivation is OFF by default: the amber buttons and green
// <Picture N> badges stay regardless of the title colour, with the title left at
// ComfyUI's default red like the other nodes. Set node._lunaDeriveAccent = true
// (or flip DERIVE_ACCENT) to make the widget follow node.color again.
const DERIVE_ACCENT = false;

function syncAccent(node, rootEl) {
    const derive = node._lunaDeriveAccent ?? DERIVE_ACCENT;
    const accent = (derive && brighten(node.color)) || DEFAULT_ACCENT;
    if (rootEl._lunaAccent === accent) return;
    rootEl._lunaAccent = accent;
    rootEl.style.setProperty("--luna-accent", accent);
    rootEl.style.setProperty("--luna-accent-soft", accent + "22");
}

// One cache-busting token per file per page load: a fresh one on EVERY call made
// each re-render download every thumbnail again (seen 2026-10-16 once the editors
// re-render on each change). Uploads never overwrite (overwrite=false gives a new
// name), so a stable URL cannot show a stale picture.
const _viewTok = new Map();
const viewURL = (name) => {
    if (!_viewTok.has(name)) _viewTok.set(name, Math.random().toString(36).slice(2));
    return `/view?filename=${encodeURIComponent(name)}&type=input&subfolder=&rand=${_viewTok.get(name)}`;
};
// Stable URL for sound files: the waveform decode and the player share the
// browser's cache instead of downloading a song twice.
const soundURL = (name) => `/view?filename=${encodeURIComponent(name)}&type=input&subfolder=`;

/* ------------------------------------------------------------- sound cards */

const MEDIA_WIDGET = "media_state";
const WIRED_INPUTS = ["seconds", "shot", "width", "height", "song_start"];   // sockets whose value the editors show
const MAX_VIDEOS = 3;
const VIDEO_EXT = /\.(mp4|mov|mkv|webm|avi|m4v|mpg|mpeg|wmv|flv)$/i;
const isVideoFile = (f) => (f.type || "").startsWith("video/") || VIDEO_EXT.test(f.name || "");
const MEMORY_CAP_GB = 6;          // asset_loader.py MEMORY_CAP; the run refuses above it
const NEW_VIDEO = { start: 0, frames: 124, fps: "target", count: "target", size: "canvas", fit: "crop",
                    nth: 1, sound: "keep", precision: "fp16", position: "free", shot: 1 };

// Clip facts (size, fps, length, sound) come from the server: the browser's
// <video> element does not expose a frame rate.
const _mediaInfo = new Map();
const _hasSound = new Map();   // file -> does it have a sound track (filled as info arrives)
const _mediaFps = new Map();   // file -> its frame rate (filled as info arrives), for the shot rate (shotRate)
function mediaInfo(file) {
    if (!_mediaInfo.has(file)) {
        const p = fetch(`/lunasave/media-info?file=${encodeURIComponent(file)}`)
            .then((r) => (r.ok ? r.json() : Promise.reject(new Error(r.status))));
        p.then((mi) => { _hasSound.set(file, !!mi.audio); if (Number(mi.fps) > 0) _mediaFps.set(file, Number(mi.fps)); })
            .catch(() => _mediaInfo.delete(file));
        _mediaInfo.set(file, p);
    }
    return _mediaInfo.get(file);
}
const stripURL = (file, n) => `/lunasave/filmstrip?file=${encodeURIComponent(file)}&n=${n}`;

// Roles: a reference is shown to H3 and named in the prompt; "on the shot" is
// anchored at a frame. Defaults: pictures and clips are references, sound 1 is
// the song locked to the shot, later sounds are references.
const picRole = (m, i) => (m.pictures[i]?.role === "key" ? "key" : "ref");
const clipRole = (c) => (c.role === "guide" ? "guide" : "ref");
// Mirror of asset_timeline.normalize_sound_role: only sound 1 (the song) can be
// locked — a later sound asked to lock is a guide.
const soundRole = (c, j) => (c.role === "lock" ? (j === 0 ? "lock" : "guide")
    : ["guide", "ref"].includes(c.role) ? c.role : j === 0 ? "lock" : "ref");

// --- frame rule: begin (pure; tests\test_asset_frames.mjs runs this block against luna_units.py) ---
// Mirror of luna_units.py (Luna-MinimaxH3 SPEC § 1, § 2): every count made for a shot rounds UP onto the target's
// grid (H3 17k+5, LTX 8k+1, General any whole count); above 362 on H3 a warning, never a cut. A length in seconds
// becomes whole frames first (half up). A cut to what a file holds rounds DOWN. One overlap list on every node.
const H3_FPS = 24;
const TRAINED_MAX = 362;
const GRIDS = { h3: [17, 5], ltx: [8, 1], free: [1, 0] };
const TARGET_FPS = { h3: 24, ltx: 25 };
const OVERLAPS = [0, 22, 39, 90, 141, 192];
const OVERLAP_ROW = [22, 39, 90, 141, 192];   // the node face's row; 0 is the first clip of a chain
const DEFAULT_OVERLAP = 22;
const gridSpec = (grid) => GRIDS[grid] || GRIDS.free;
const mod = (a, b) => ((a % b) + b) % b;   // Python's %, never negative
const firstFrames = (grid) => { const [, p] = gridSpec(grid); return p > 0 ? p : 1; };
function framesUp(n, grid = "h3") {
    const [step, phase] = gridSpec(grid);
    const f = Math.max(Math.ceil(Number(n) - 1e-9), firstFrames(grid));
    return f + mod(phase - f, step);
}
function framesDown(n, grid = "h3") {
    const [step, phase] = gridSpec(grid);
    const f = Math.floor(Number(n) + 1e-9);
    return f < firstFrames(grid) ? 0 : f - mod(f - phase, step);
}
const wholeFrames = (seconds, fps = H3_FPS) => Math.floor(Number(seconds) * Number(fps) + 0.5);
const framesFor = (seconds, fps = H3_FPS, grid = "h3") => framesUp(wholeFrames(seconds, fps), grid);
const framesWarning = (frames, grid = "h3") => (grid === "h3" && Math.trunc(frames) > TRAINED_MAX
    ? `${Math.trunc(frames)} frames is past the ${TRAINED_MAX} frames H3 was trained on; it is kept, not cut.` : "");
const gridFor = (target) => ({ h3: "h3", ltx: "ltx" })[String(target ?? "").trim().toLowerCase()] || "free";
function fpsFor(target, source) {
    const fixed = TARGET_FPS[String(target ?? "").trim().toLowerCase()];
    if (fixed) return fixed;
    const r = Number(source);
    return Number.isFinite(r) && r > 0 && source !== null && source !== "" ? r : H3_FPS;
}
// [value on OVERLAPS, warning or ""]; a tie takes the smaller value.
function snapOverlap(value) {
    const v = Number(value);
    if (!Number.isFinite(v)) return [DEFAULT_OVERLAP, `Overlap ${String(value)} is not a number of frames; ${DEFAULT_OVERLAP} is used.`];
    let best = OVERLAPS[0];
    for (const o of OVERLAPS) if (Math.abs(o - v) < Math.abs(best - v)) best = o;
    return [best, best === v ? "" : `Overlap ${v} is not on the list ${OVERLAPS.join(", ")} frames; ${best} is used.`];
}
// The node's overlap (asset_loader._overlap_setting): settings.overlap when set, else sound card 1's (saves made
// before the node-wide row), else 22; snapped onto the list. m: readMedia's shape.
function nodeOverlap(m) {
    const set = m?.settings?.overlap;
    const card = (m?.audio || []).find((c) => c && c.file);
    const raw = set ?? (card && card.overlap != null ? card.overlap : DEFAULT_OVERLAP);
    return snapOverlap(raw);
}
// media_state text -> the cards and settings the editor works on (readMedia).
function parseMedia(text) {
    try {
        const m = JSON.parse(String(text || "") || "{}");
        return {
            audio: Array.isArray(m.audio) ? m.audio.filter((c) => c && c.file) : [],
            video: Array.isArray(m.video) ? m.video.filter((c) => c && c.file) : [],
            pictures: Array.isArray(m.pictures) ? m.pictures.map((p) => p || {}) : [],
            shot: m.shot && Number(m.shot.frames) > 0 ? { frames: Number(m.shot.frames) } : null,
            settings: migrateOverlap(m.settings && typeof m.settings === "object" ? m.settings : {}, m.audio),
        };
    } catch { return { audio: [], video: [], pictures: [], shot: null, settings: {} }; }
}
// A save from before the node-wide overlap row: settings.overlap takes sound card 1's overlap (22 when the card has
// none) on read, so the next write keeps it whatever later happens to the song. Off the list is kept as it is; the
// run and the row snap it with a warning (asset_loader._overlap_setting reads the same).
function migrateOverlap(settings, audio) {
    if (settings.overlap != null) return settings;
    const card = (Array.isArray(audio) ? audio : []).find((c) => c && c.file);
    return card ? { ...settings, overlap: card.overlap ?? DEFAULT_OVERLAP } : settings;
}
// The editor's state -> media_state text (writeMedia); "" when there is nothing to keep.
function mediaJSON(m) {
    const out = {};
    if (m.audio?.length) out.audio = m.audio;
    if (m.video?.length) out.video = m.video;
    // Picture roles only matter once one of them is not a plain reference.
    if (m.pictures?.some((p) => p?.role === "key")) out.pictures = m.pictures;
    if (m.shot?.frames) out.shot = m.shot;
    // A "continuation" key from an older save is dropped here: the card is gone and the backend ignores it.
    if (m.settings && Object.keys(m.settings).length) out.settings = m.settings;
    return Object.keys(out).length ? JSON.stringify(out) : "";
}
// The length of a sound card's window in the editor, in seconds (asset_loader._song / _sounds). shot: shotLen()'s
// {seconds, rate, grid}. Sound card 1, the song, is always the shot; a reference or guide sound (cards 2-3) keeps its
// own Duration, at most the shot (0 = the shot).
function soundWindow(card, index, shot) {
    if (index === 0) return shot.seconds;
    const d = Number(card?.duration) || 0;
    return d > 0 ? Math.min(d, shot.seconds) : shot.seconds;
}
// A typed or dragged length for a sound card -> { shotFrames (the node's new shot length, or null), duration }.
// Only the song's length sets the shot length (rounded UP onto the grid); cards 2-3 change only their own Duration.
function soundLengthPatch(index, seconds, shot) {
    if (index === 0) {
        const frames = Math.max(1, framesFor(seconds, shot.rate, shot.grid));
        return { shotFrames: frames, duration: frames / shot.rate };
    }
    return { shotFrames: null, duration: Math.min(Number(seconds), shot.seconds) };
}
// --- frame rule: end ---

// --- naming rule: begin (pure; tests\test_asset_naming.mjs runs this block outside ComfyUI) ---
// Mirror of asset_timeline.h3_tags, which is luna_naming.assign / tag in the minimax row (the Asset
// Loader's row for every ⚙ Target). H3 numbers ONLY the references, in order, so a keyframe on card 1
// makes card 2 <Picture 1>. Sound order: each reference clip's kept soundtrack right before its clip,
// then the standalone reference sounds.
//   pics: "ref" | "key" per picture card; clips: { role: "ref" | "guide", sound: bool } per clip card;
//   sounds: "ref" | "lock" | "guide" per sound card.
// names: each card's @ name (its card number) with what H3 receives: its tag, or its part on the shot.
function cardTags(pics, clips, sounds) {
    const out = { pictures: {}, videos: {}, videoAudio: {}, sounds: {}, names: {} };
    let p = 0, v = 0, a = 0;
    pics.forEach((role, i) => { if (role === "ref") out.pictures[i] = `<Picture ${++p}>`; });
    clips.forEach((c, k) => {
        if (c.role !== "ref") return;
        if (c.sound) out.videoAudio[k] = `<Audio ${++a}>`;
        out.videos[k] = `<Video ${++v}>`;
    });
    sounds.forEach((role, j) => { if (role === "ref") out.sounds[j] = `<Audio ${++a}>`; });
    const onShot = (role, j) => (role === "lock" ? "song lock" : j === 0 ? "song guide" : "guide");
    out.names = {
        p: pics.map((_, i) => `@image${i + 1} · ${out.pictures[i] || "keyframe"}`),
        v: clips.map((_, k) => `@video${k + 1} · ${out.videos[k] || "guide clip"}`),
        a: sounds.map((role, j) => `@audio${j + 1} · ${out.sounds[j] || onShot(role, j)}`),
    };
    return out;
}
// --- naming rule: end ---

function h3Tags(node) {
    const m = readMedia(node), n = readPaths(node).length;
    return cardTags(
        Array.from({ length: n }, (_, i) => picRole(m, i)),
        // A kept soundtrack counts only when the file has sound (known once its info is in).
        m.video.map((c) => ({ role: clipRole(c), sound: c.sound !== "drop" && _hasSound.get(c.file) !== false })),
        m.audio.map((c, j) => soundRole(c, j)));
}
const escTag = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

/* -------------------------------------------------------------- ⚙ panel */

// The settings panel, after LLM Prompt's (llm_prompt_panel.mjs, itself after
// Pixaroma's node_panel.mjs, MIT): it sits against the node's right edge and is
// redrawn every frame at the canvas zoom, so it pans and zooms with the node; it
// does not close on an outside click (only ⚙, ✕, Done, Esc); a focused field is
// blurred before it goes, or Chrome drops the typed value.
let _sp = null;   // { el, node, stop }
function closeSettings() {
    if (!_sp) return;
    const p = _sp;
    _sp = null;
    if (p.el.contains(document.activeElement)) document.activeElement.blur();
    p.stop();
    p.el.remove();
}
const settingsOpenFor = (node) => !!_sp && _sp.node === node;

function openSettings(node) {
    if (settingsOpenFor(node)) { closeSettings(); return; }
    closeSettings();
    const el = document.createElement("div");
    el.className = "luna-al-sp";
    el.innerHTML = `<div class="hd"><span>⚙</span><span class="t">Luna-H3-AssetLoader settings</span><span class="x" title="Close">✕</span></div>
        <div class="bd"></div><div class="ft"><span class="rs" title="Back to the defaults">reset</span><span class="dn">Done</span></div>`;
    const body = el.querySelector(".bd");
    const set = (patch) => {
        const m = readMedia(node);
        const cur = m.settings || {};
        m.settings = { ...cur, ...patch,
            ...(patch.video ? { video: { ...(cur.video || {}), ...patch.video } } : {}),
            ...(patch.audio ? { audio: { ...(cur.audio || {}), ...patch.audio } } : {}),
            ...(patch.node ? { node: { ...(cur.node || {}), ...patch.node } } : {}) };
        writeMedia(node, m);
        node._lunaRender?.();
        draw();
    };
    const chips = (label, opts, cur, f, note) => {
        const r = document.createElement("div");
        r.className = "rw";
        r.innerHTML = `<span class="l"></span><span class="ch"></span>`;
        r.querySelector(".l").textContent = label;
        for (const [v, text] of opts) {
            const c = document.createElement("span");
            c.textContent = text;
            if (String(v) === String(cur)) c.className = "on";
            c.onclick = () => f(v);
            r.querySelector(".ch").appendChild(c);
        }
        body.appendChild(r);
        if (note) { const n = document.createElement("div"); n.className = "nt"; n.textContent = note; body.appendChild(n); }
    };
    const head = (icon, text) => { const h = document.createElement("div"); h.className = "sh"; h.textContent = `${icon}  ${text}`; body.appendChild(h); };
    function draw() {
        const s = settingsOf(readMedia(node));
        body.replaceChildren();
        head("◎", "Target");
        chips("Workflow", [["h3", "MiniMax H3"], ["ltx", "LTX"], ["general", "General"]], s.target,
              (v) => set({ target: v }), (TARGETS[s.target] || TARGETS.h3).note);
        head("▶", "New clips");
        chips("Frame rate", [["target", "Target"], ["source", "Source"]], s.video.fps, (v) => set({ video: { fps: v } }));
        chips("Size", [["canvas", "Match canvas"], ["edge", "768 edge"], ["source", "Source"]], s.video.size, (v) => set({ video: { size: v } }));
        chips("Precision", [["fp16", "fp16"], ["fp32", "fp32"]], s.video.precision, (v) => set({ video: { precision: v } }));
        const r = document.createElement("div");
        r.className = "rw";
        r.innerHTML = `<span class="l">Memory cap</span><span class="ch"><input class="nb" value="${Number(s.memory_gb)}"> GB</span>`;
        const nb = r.querySelector(".nb");
        nb.onchange = () => { const v = Number(nb.value); if (v >= 1 && v <= 64) set({ memory_gb: v }); else nb.value = s.memory_gb; };
        nb.addEventListener("keydown", (e) => { if (e.key === "Enter") nb.blur(); });
        body.appendChild(r);
        const nt = document.createElement("div"); nt.className = "nt";
        nt.textContent = "A clip that would need more than this is refused before decoding. Applies to every clip on the node.";
        body.appendChild(nt);
        head("♪", "Sound");
        chips("Nudge step", [[0.1, "0.1 s"], [1 / 24, "1 frame"], [0.5, "0.5 s"]], s.audio.nudge, (v) => set({ audio: { nudge: v } }));
        chips("New songs end in", [["silence", "Silence"], ["loop", "Loop"]], s.audio.fill, (v) => set({ audio: { fill: v } }));
        head("▦", "Node");
        chips("Preview height", [["S", "S"], ["M", "M"], ["L", "L"]], s.node.preview, (v) => set({ node: { preview: v } }));
    }
    draw();
    el.querySelector(".x").onclick = closeSettings;
    el.querySelector(".dn").onclick = closeSettings;
    el.querySelector(".rs").onclick = () => { const m = readMedia(node); m.settings = {}; writeMedia(node, m); node._lunaRender?.(); draw(); };
    el.addEventListener("keydown", (e) => { if (e.key === "Escape") { e.stopPropagation(); closeSettings(); } else e.stopPropagation(); });
    el.addEventListener("pointerdown", (e) => e.stopPropagation());
    el.addEventListener("wheel", (e) => e.stopPropagation(), { passive: true });
    document.body.appendChild(el);

    let raf = 0, alive = true;
    const follow = () => {
        if (!alive) return;
        if (!node.graph || app.canvas?.graph !== node.graph) { closeSettings(); return; }
        const ds = app.canvas.ds, r = app.canvas.canvas.getBoundingClientRect();
        const th = window.LiteGraph?.NODE_TITLE_HEIGHT || 30;
        el.style.left = `${r.left + (node.pos[0] + node.size[0] + 10 + ds.offset[0]) * ds.scale}px`;
        el.style.top = `${r.top + (node.pos[1] - th + ds.offset[1]) * ds.scale}px`;
        el.style.transform = `scale(${ds.scale})`;
        raf = requestAnimationFrame(follow);
    };
    follow();
    const onEsc = (e) => { if (e.key === "Escape" && !/^(INPUT|TEXTAREA|SELECT)$/.test(e.target?.tagName || "")) closeSettings(); };
    document.addEventListener("keydown", onEsc, true);
    _sp = { el, node, stop() { alive = false; cancelAnimationFrame(raf); document.removeEventListener("keydown", onEsc, true); } };
}

// Luna-H3-Canvas's seconds from its duration: the shared rule, whole frames at 24 then UP onto 17k+5.
const canvasSeconds = (d) => framesFor(Number(d), H3_FPS, "h3") / H3_FPS;

// What an output on the node at the other end of a wire carries, when the frontend
// can know it before a run: a node that says so itself (Luna-H3-ShotList's
// lunaValueOf), the H3 Canvas's seconds (worked out from its duration widget), or a
// plain number widget named like the output (a Float primitive's "value").
function sourceValue(src, outName) {
    if (!src) return null;
    try {
        if (typeof src.lunaValueOf === "function") {
            const v = src.lunaValueOf(outName);
            if (v !== undefined) return v;
        }
        if (src.type === "Luna-H3-Canvas" && outName === "seconds") {
            const d = Number(widget(src, "duration_seconds")?.value);
            return d > 0 ? canvasSeconds(d) : null;
        }
    } catch { return null; }
    const w = src.widgets?.find((x) => x.name === outName) || src.widgets?.find((x) => x.name === "value");
    return w && Number.isFinite(Number(w.value)) ? Number(w.value) : null;
}

function wiredSource(node, name) {
    const inp = node.inputs?.find((x) => x.name === name);
    if (!inp || inp.link == null) return null;
    const g = node.graph;
    const l = g?.getLink?.(inp.link) ?? g?.links?.get?.(inp.link) ?? g?.links?.[inp.link];
    const src = l && g.getNodeById(l.origin_id);
    return src ? { src, out: src.outputs?.[l.origin_slot]?.name } : { src: null, out: null };
}

// The numbers on the nodes feeding the wired inputs, as one string. Checked on
// every draw, so editing the Float/Int (or the Canvas, or the shot list) upstream
// moves the window straight away.
function wiredSignature(node) {
    return WIRED_INPUTS.map((name) => {
        const w = wiredSource(node, name);
        if (!w) return "-";
        return String(sourceValue(w.src, w.out) ?? "?");
    }).join("|");
}
const MAX_SOUNDS = 3;
const AUDIO_EXT = /\.(wav|mp3|flac|ogg|opus|m4a|aac|wma|aiff?)$/i;
const isAudioFile = (f) => (f.type || "").startsWith("audio/") || AUDIO_EXT.test(f.name || "");

const readMedia = (node) => parseMedia(widget(node, MEDIA_WIDGET)?.value);

/* ---------------------------------------------------------------- settings */

// Node-wide settings (⚙). Defaults when nothing is set: MiniMax H3, 6 GB.
const SETTINGS = {
    target: "h3", memory_gb: 6,
    video: { fps: "target", size: "canvas", precision: "fp16" },
    audio: { nudge: 0.1, fill: "silence" },
    node: { preview: "M" },
};
const settingsOf = (m) => {
    const s = m.settings || {};
    return { ...SETTINGS, ...s, video: { ...SETTINGS.video, ...(s.video || {}) },
             audio: { ...SETTINGS.audio, ...(s.audio || {}) }, node: { ...SETTINGS.node, ...(s.node || {}) } };
};
// What each target means — mirrors asset_timeline.target_fps / target_grid.
const TARGETS = {
    h3: { label: "MiniMax H3", fps: 24, grid: "h3", step: 17, gridText: "H3 grid (17k+5)",
          note: "24 fps. Frame counts round up onto 17k+5; past 362 a warning, nothing cut. Pictures and clips anchor at any frame; a sound runs from its frame to the end." },
    ltx: { label: "LTX", fps: 25, grid: "ltx", step: 8, gridText: "LTX grid (8k+1)",
           note: "25 fps. Frame counts round up onto 8k+1; clips anchored at frame 0 or 8k+1 (LTXVAddGuide moves anything else)." },
    general: { label: "General", fps: null, grid: "free", step: 24, gridText: "Free",
               note: "Files as they are: clip V1's frame rate (24 with no clip) and any length." },
};
const targetOf = (m) => TARGETS[settingsOf(m).target] || TARGETS.h3;
const PREVIEW_SIZES = { S: 120, M: PREVIEW_H, L: 260 };
const previewH = (node) => PREVIEW_SIZES[settingsOf(readMedia(node)).node.preview] || PREVIEW_H;
// A frame count on a grid, rounded UP (media_io.grid_frames; no 362 cut), and one grid step up or down.
const gridFrames = (n, grid) => framesUp(Math.max(1, Math.trunc(Number(n) || 1)), grid);
function stepGrid(n, dir, grid) {
    const f = Math.trunc(Number(n) || 1);
    return dir > 0 ? framesUp(f + 1, grid) : Math.max(firstFrames(grid), framesDown(f - 1, grid));
}
// The shot's frame rate (asset_loader: luna_units.fps_for of clip card V1's delivered rate): H3 24, LTX 25,
// General clip V1's rate after its fps choice and every-Nth, 24 with no clip or before its facts are in.
function shotRate(m) {
    const target = settingsOf(m).target, c = m.video?.[0];
    if (!c) return fpsFor(target, null);
    const src = _mediaFps.get(c.file);
    const decode = (c.fps ?? "target") === "target" ? fpsFor(target, src) : c.fps === "source" ? src : Number(c.fps);
    return fpsFor(target, decode ? decode / Math.max(1, Number(c.nth) || 1) : null);
}

function writeMedia(node, m) {
    const w = widget(node, MEDIA_WIDGET);
    if (!w) return;
    w.value = mediaJSON(m);
    w.callback?.(w.value);
    node.graph?.setDirtyCanvas(true, true);
}

const clock = (t) => {
    t = Math.max(0, t || 0);
    const m = Math.floor(t / 60), s = t - m * 60;
    return `${m}:${s.toFixed(2).padStart(5, "0")}`;
};
const parseClock = (txt) => {
    const s = String(txt).trim().replace(",", ".");
    const m = s.match(/^(\d+):(\d+(?:\.\d+)?)$/);
    const v = m ? Number(m[1]) * 60 + Number(m[2]) : Number(s.replace(/\s*s$/, ""));
    return Number.isFinite(v) ? v : null;
};

// Waveform peaks, 100 per second, decoded once per file in the browser — the
// file is already in the input folder, so nothing is decoded on the server just
// to draw a picture of it (the same split Pixaroma's Load Audio makes).
const PEAKS_PER_S = 100;
const MIN_VIEW_S = 2;           // the song zoom stops at 2 s across the waveform
const _songView = new Map();     // song file -> the zoomed part {v0, v1}, kept across re-renders
const _peaks = new Map();     // file -> Promise<{peaks, duration, sampleRate, channels}>
function loadPeaks(file) {
    if (_peaks.has(file)) return _peaks.get(file);
    const p = (async () => {
        const buf = await (await fetch(soundURL(file))).arrayBuffer();
        const ctx = new OfflineAudioContext(1, 1, 44100);
        const ab = await ctx.decodeAudioData(buf);
        const n = Math.max(1, Math.ceil(ab.duration * PEAKS_PER_S));
        const per = ab.length / n, peaks = new Float32Array(n);
        for (let c = 0; c < ab.numberOfChannels; c++) {
            const d = ab.getChannelData(c);
            for (let i = 0; i < n; i++) {
                let mx = peaks[i];
                const a = Math.floor(i * per), b = Math.min(d.length, Math.floor((i + 1) * per));
                for (let j = a; j < b; j++) { const v = d[j] < 0 ? -d[j] : d[j]; if (v > mx) mx = v; }
                peaks[i] = mx;
            }
        }
        return { peaks, duration: ab.duration, sampleRate: ab.sampleRate, channels: ab.numberOfChannels };
    })();
    p.catch(() => _peaks.delete(file));
    _peaks.set(file, p);
    return p;
}

// Draw [t0, t1] of a song; samples inside [a, b] in the selection colour.
function drawWave(cv, info, t0, t1, a, b, opts = {}) {
    if (!cv?.isConnected || !info) return;
    const r = cv.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
    if (!r.width || !r.height) return;
    cv.width = Math.round(r.width * dpr); cv.height = Math.round(r.height * dpr);
    const g = cv.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    const W = r.width, H = r.height, mid = H / 2, span = Math.max(1e-6, t1 - t0);
    if (opts.grid) {
        g.strokeStyle = "#2a2c35";
        for (let s = Math.ceil(t0 * 2) / 2; s < t1; s += 0.5) {
            const x = (s - t0) / span * W;
            g.beginPath(); g.moveTo(x, 0); g.lineTo(x, H); g.stroke();
        }
    }
    const { peaks } = info, step = opts.step || 2;
    for (let x = 0; x < W; x += step) {
        const ta = t0 + x / W * span, tb = t0 + (x + step) / W * span;
        let mx = 0;
        for (let i = Math.max(0, Math.floor(ta * PEAKS_PER_S)); i <= Math.min(peaks.length - 1, Math.floor(tb * PEAKS_PER_S)); i++)
            if (peaks[i] > mx) mx = peaks[i];
        const tm = (ta + tb) / 2, h = Math.max(1, mx * (H / 2 - 2));
        g.fillStyle = tm >= a && tm <= b ? (opts.color || "#e07bb4") : "#5d616e";
        g.fillRect(x, mid - h, Math.max(1, step - 0.6), h * 2);
    }
    if (opts.mark != null) {   // the cut: where the window begins
        const x = (opts.mark - t0) / span * W;
        g.fillStyle = "#e0a458";
        g.fillRect(Math.round(x) - 1, 0, 2, H);
    }
}

// The last run's song report, per node. Module memory rather than node.properties:
// properties are compared by the change tracker, and writing them on every run
// would mark the workflow modified (the lesson LLM Prompt's face learned).
const _lastSong = new Map();

/* ------------------------------------------------------- per-image resize state */

const STATE_WIDGET = "asset_state";
const DEFAULT_ITEM = {
    mode: "off", max_mp: 2.0, longest_side: 1024, scale_factor: 1.0,
    ratio: "", ratio_action: "crop", crop_anchor: "center",
    snap: 0, allow_upscale: false,
};

function readState(node) {
    const w = widget(node, STATE_WIDGET);
    try {
        const p = JSON.parse(String(w?.value || "") || "{}");
        return { all: p.all || null, items: Array.isArray(p.items) ? p.items : [] };
    } catch { return { all: null, items: [] }; }
}

function itemAt(node, i) {
    const st = readState(node);
    return { ...DEFAULT_ITEM, ...(st.all || {}), ...(st.items[i] || {}) };
}

function writeItem(node, i, patch, toAll = false) {
    const w = widget(node, STATE_WIDGET);
    if (!w) return;
    const st = readState(node);
    if (toAll) {
        st.all = { ...(st.all || {}), ...patch };
        st.items = [];           // an explicit "apply to all" clears per-card overrides
    } else {
        while (st.items.length <= i) st.items.push({});
        st.items[i] = { ...st.items[i], ...patch };
    }
    const out = {};
    if (st.all) out.all = st.all;
    if (st.items.length) out.items = st.items;
    // Empty state means "untouched", which is what REF2VA wants — keep the field
    // literally empty rather than writing a no-op object.
    w.value = Object.keys(out).length ? JSON.stringify(out) : "";
    w.callback?.(w.value);
}

// What the Python side will produce, mirrored here so the panel can show
// INPUT -> OUTPUT before the graph runs. Kept in step with asset_images.apply_state.
function predictSize(w, h, st) {
    if (!w || !h) return [0, 0];
    let W = w, H = h;
    if (st.ratio && st.ratio.includes(":")) {
        const [rw, rh] = st.ratio.split(":").map(Number);
        const t = rw / rh;
        if (t && Math.abs(W / H - t) > 0.001) {
            if (st.ratio_action === "pad") {
                if (W / H > t) H = Math.round(W / t); else W = Math.round(H * t);
            } else {
                if (W / H > t) W = Math.round(H * t); else H = Math.round(W / t);
            }
        }
    }
    let s = 1;
    if (st.mode === "max_mp") s = Math.sqrt((Number(st.max_mp) * 1e6) / (W * H));
    else if (st.mode === "longest_side") s = Number(st.longest_side) / Math.max(W, H);
    else if (st.mode === "scale_factor") s = Number(st.scale_factor);
    if (!st.allow_upscale) s = Math.min(1, s);
    const snap = Number(st.snap) || 0;
    // Snap goes DOWN to the grid and crops the rest (asset_images.snap_size).
    const fix = (v) => (snap > 1 ? Math.max(snap, Math.floor(v / snap + 1e-6) * snap) : Math.max(1, Math.round(v)));
    return [fix(W * s), fix(H * s)];
}

// Aspect wording kept deliberately close to the Python side's _describe_ar, so the
// card and the asset_brief never disagree about what a picture is.
const COMMON_AR = [[1,1],[2,3],[3,2],[3,4],[4,3],[4,5],[5,4],[9,16],[16,9],[1,2],[2,1],[21,9]];

// Two forms on purpose. The card is ~120px wide, so anything longer than about
// "1672×941 ≈16:9" truncates — and a truncated resolution defeats the point of
// showing it. The long form goes in the tooltip, where there is room.
function describeAR(w, h) {
    if (!w || !h) return { short: "", long: "" };
    const r = w / h;
    let best = COMMON_AR[0], bd = Infinity;
    for (const [a, b] of COMMON_AR) {
        const d = Math.abs(a / b - r);
        if (d < bd) { bd = d; best = [a, b]; }
    }
    const gcd = (a, b) => (b ? gcd(b, a % b) : a);
    const g = gcd(w, h) || 1;
    const ew = w / g, eh = h / g;
    const nearest = `${best[0]}:${best[1]}`;
    if (ew === best[0] && eh === best[1]) return { short: nearest, long: nearest };
    if (Math.max(ew, eh) <= 32)
        return { short: `${ew}:${eh}`, long: `exactly ${ew}:${eh}, nearest ${nearest}` };
    return { short: `≈${nearest}`, long: `${r.toFixed(2)}:1, nearest ${nearest}` };
}

async function uploadFiles(files) {
    const saved = [];
    for (const file of files) {
        const form = new FormData();
        form.append("image", file);
        form.append("type", "input");
        form.append("overwrite", "false");
        const resp = await fetch("/upload/image", { method: "POST", body: form });
        if (!resp.ok) {
            alert(`Upload failed for ${file.name}: ${resp.status} ${resp.statusText}`);
            continue;
        }
        const data = await resp.json();
        saved.push(data.subfolder ? `${data.subfolder}/${data.name}` : data.name);
    }
    return saved;
}

/* --------------------------------------------------------------- outputs */

// --- output migration: begin (pure; tests\test_asset_outputs.mjs runs this block outside ComfyUI) ---
// Every output, always (Luna-MinimaxH3 SPEC § 2): the ten fixed outputs at slots 0-9, image_1..image_9 at 10-18
// whatever the number of picture cards, then fps (19) and vocals (20). ComfyUI sends each link's frontend slot
// number with the prompt (graphToPrompt: [origin_id, origin_slot]) and the backend reads that index of the returned
// tuple, so the slots must be the Python order exactly and none may come or go.
const MIGRATED_NODE = "Luna-H3-AssetLoader";
const FIXED_OUTS = [["images", "IMAGE"], ["width", "INT"], ["height", "INT"], ["asset_brief", "STRING"],
                    ["audio", "AUDIO"], ["refs", "LUNA_H3_REFS"], ["timeline", "LUNA_TIMELINE"],
                    ["video", "IMAGE"], ["frames", "INT"], ["overlap", "INT"]];
const FIXED = FIXED_OUTS.length;
const IMAGE_OUTS = Array.from({ length: 9 }, (_, i) => [`image_${i + 1}`, "IMAGE"]);
const TAIL_OUTS = [["fps", "FLOAT"], ["vocals", "AUDIO"]];
const ALL_OUTS = [...FIXED_OUTS, ...IMAGE_OUTS, ...TAIL_OUTS];

// The fixed-prefix rule, one rule, not one step per release: a saved node keeps the
// fixed outputs it was saved with (everything before its image_1) and gets every
// fixed output it lacks inserted before image_1, at that output's own index in
// FIXED_OUTS. image_1 then sits at FIXED whatever the save: old saves (4 fixed,
// image_1 at slot 4), v3 saves (7 fixed) and v4 saves (9 fixed) all end with
// image_1 at slot 10; a node that already has all ten is left alone. Every link
// leaving image_1 or higher is moved by the number of slots inserted. The saved
// fixed outputs are a prefix of FIXED_OUTS, so they keep their numbers.
const savedFixedCount = (outputs) => {
    const k = Array.isArray(outputs) ? outputs.findIndex((o) => o?.name === "image_1") : -1;
    return k >= 0 && k < FIXED ? k : -1;
};
const outSlot = ([name, type]) => ({ name, localized_name: name, type, links: null });

// Frontend 1.55.17 overwrites a saved node's output names BY POSITION from the
// current definition and keeps every link on its slot number, so left alone an
// old image_1 wire lands on `audio` (or `video`) without a word (seen in the S0
// spike, 2026-10-16). onConfigure is too late for that; this rewrites the raw
// workflow JSON before the graph is configured, root graph and subgraph definitions alike:
//   1. the fixed-prefix rule above (an old save's links leaving image_1 or higher move up);
//   2. then every slot the save lacks is APPENDED: the image_N it did not show (a save made while the node shrank
//      to image_1..3 gets image_4..9), then fps and vocals. Appending moves no link.
// A migrated save has all 21 slots, so it is never changed twice. Returns what it did, one line per change.
function migrateWorkflowJSON(data) {
    const moved = [];
    const fixGraph = (g, where) => {
        if (!Array.isArray(g?.nodes)) return;
        const old = new Map();   // node id -> [first moved slot, slots inserted]
        for (const n of g.nodes) {
            if (n?.type !== MIGRATED_NODE) continue;
            if (!Array.isArray(n.outputs)) n.outputs = [];
            const k = savedFixedCount(n.outputs);
            if (k >= 0) {
                const missing = FIXED_OUTS.slice(k);
                old.set(n.id, [k, missing.length]);
                n.outputs.splice(k, 0, ...missing.map(outSlot));
            }
            const prefix = n.outputs.every((o, i) => o?.name === ALL_OUTS[i]?.[0]);
            if (!prefix) { moved.push(`${where} node ${n.id}: outputs not in the known order, left as saved`); continue; }
            if (n.outputs.length < ALL_OUTS.length) {
                const tail = ALL_OUTS.slice(n.outputs.length);
                n.outputs.push(...tail.map(outSlot));
                moved.push(`${where} node ${n.id}: appended ${tail.map(([name]) => name).join(", ")}`);
            }
        }
        if (!old.size) return;
        // Floating links (a wire dragged to a reroute with nothing after it) carry an
        // origin too; they move with the rest.
        for (const l of [...(g.links || []), ...(g.floatingLinks || [])]) {
            // [id, origin_id, origin_slot, target_id, target_slot, type] in the root
            // graph; {id, origin_id, origin_slot, ...} inside subgraph definitions.
            const arr = Array.isArray(l);
            const oid = arr ? l[1] : l?.origin_id, os = arr ? l[2] : l?.origin_slot;
            const mv = old.get(oid);
            if (!mv || !(os >= mv[0])) continue;
            if (arr) l[2] = os + mv[1]; else l.origin_slot = os + mv[1];
            moved.push(`${where} link ${arr ? l[0] : l.id}: slot ${os} -> ${os + mv[1]}`);
        }
    };
    fixGraph(data, "root");
    for (const sg of data?.definitions?.subgraphs || []) fixGraph(sg, `subgraph ${sg.id}`);
    // clipboard / blueprint data keeps its subgraphs at the top level
    for (const sg of Array.isArray(data?.subgraphs) ? data.subgraphs : []) fixGraph(sg, `subgraph ${sg.id}`);
    return moved;
}
// --- output migration: end ---

// Paste and subgraph-blueprint insertion go through LGraphCanvas._deserializeItems,
// never loadGraphData, so beforeConfigureGraph never sees them (review,
// 2026-10-16). Tested: an old pasted node lost its image_1/image_3 wires (AUDIO and
// LUNA_TIMELINE slots refuse an IMAGE link) and image_4+ would land on image_1+.
// The same JSON migration runs on that data before it becomes nodes.
function patchDeserialize() {
    const proto = app.canvas && Object.getPrototypeOf(app.canvas);
    if (!proto || typeof proto._deserializeItems !== "function" || proto._lunaAssetMigrates) return;
    const orig = proto._deserializeItems;
    proto._deserializeItems = function (items, options) {
        try {
            const moved = migrateWorkflowJSON(items);
            if (moved.length) console.info("[Luna-H3-AssetLoader] updated pasted Luna-H3-AssetLoader outputs:", moved);
        } catch (e) { console.warn("[Luna-H3-AssetLoader] paste migration skipped:", e); }
        return orig.call(this, items, options);
    };
    proto._lunaAssetMigrates = true;
}

// The live node always ends with every output: slots it lacks (a node configured from an older form) are APPENDED
// in ALL_OUTS order. Never removes a slot or moves a link.
function ensureOutputs(node) {
    if (!node.outputs) return;
    while (node.outputs.length < ALL_OUTS.length) {
        const [name, type] = ALL_OUTS[node.outputs.length];
        node.addOutput(name, type);
    }
}

// Paste and subgraph-blueprint insertion never call loadGraphData, so the JSON
// migration above never sees them (review, 2026-10-16). Such a node is configured
// from its old serialized form — `from` is where image_1 sat in `info` — and its
// links are made right after. So: once they exist, move every link leaving slot
// `from` or higher up by FIXED - from, on the live graph. A node from a migrated
// save has image_1 at FIXED in `info` and is left alone.
function migrateLiveNode(node, from) {
    const g = node.graph;
    if (!g || !node.outputs || !(from >= 0 && from < FIXED)) return 0;
    const shift = FIXED - from;
    const getLink = (id) => g.getLink?.(id) ?? g.links?.get?.(id) ?? g.links?.[id];
    const moves = [];
    node.outputs.forEach((o, i) => {
        if (i < from) return;
        for (const id of o.links || []) { const l = getLink(id); if (l && l.origin_id === node.id) moves.push([i, id, l]); }
    });
    ensureOutputs(node);   // every slot exists before the links move up
    if (!moves.length) return 0;
    for (const [i] of moves) node.outputs[i].links = [];
    for (const [i, id, l] of moves) {
        const to = i + shift;
        l.origin_slot = to;
        const out = node.outputs[to];
        if (out) (out.links = out.links || []).push(id);
    }
    node.setDirtyCanvas(true, true);
    return moves.length;
}

// image_count no longer adds or removes sockets (all nine are always there, SPEC § 2): it says how many picture
// cards the node expects, and the image sockets past it are labelled as spare. Returns { target, shown, blocked }
// for the callers; nothing can block any more.
function applyCount(node) {
    if (!node.outputs) return null;
    const w = widget(node, COUNT_WIDGET);
    if (!w) return null;
    const target = Math.max(1, Math.min(MAX, Number(w.value) || 1));
    ensureOutputs(node);
    labelOutputs(node, target);
    node.setDirtyCanvas(true, true);
    return { target, shown: target, blocked: 0 };
}

// Output names as shown: the image sockets read as the H3 inputs they feed, counted from 0
// like them; one past image_count reads "(spare)". Display only: migration and saved wires go by the real names.
const OUTPUT_LABELS = { images: "llm_images", video: "llm_video" };
function labelOutputs(node, count = MAX) {
    for (const slot of node.outputs || []) {
        const n = /^image_(\d+)$/.exec(slot.name || "");
        const label = n ? `ref_image_${Number(n[1]) - 1}${Number(n[1]) > count ? " (spare)" : ""}` : OUTPUT_LABELS[slot.name];
        if (label) slot.label = label;
    }
}

// The one way the count is ever set — stepper, arrows, card list, hand-typed
// path, all land here. Clamped to 1..MAX.
function setCount(node, value) {
    const cw = widget(node, COUNT_WIDGET);
    if (!cw) return null;
    const next = Math.max(1, Math.min(MAX, Number(value) || 1));
    if (Number(cw.value) !== next) {
        cw.value = next;
        cw.callback?.(next);      // the wrapped callback re-labels the sockets
    }
    const res = applyCount(node);
    node.setDirtyCanvas(true, true);
    return res;
}

// image_count follows the number of loaded images in BOTH directions; only the labels change.
const syncCount = (node, count) => setCount(node, count);

// Clicking the number types one. LiteGraph's own prompt, so it looks like every
// other numeric entry on the canvas.
function promptCount(node, cur, e) {
    const apply = (v) => {
        const n = parseInt(v, 10);
        if (Number.isFinite(n)) setCount(node, n);
    };
    const canvas = app.canvas || node.graph?.list_of_graphcanvas?.[0];
    if (canvas?.prompt) canvas.prompt("Outputs", cur, apply, e);
    else apply(window.prompt(`Numbered outputs (1-${MAX})`, String(cur)));
}

/* ------------------------------------------------------- the Image slots stepper */

// The count control sits in the node BODY, top-left, alongside the four fixed
// output rows. Two constraints put it exactly there, and both were measured:
//
//   1. It must not move. Widgets are drawn below the slot rows, so growing 2
//      outputs to 9 pushed the old image_count widget 78px down the node and the
//      arrow slid out from under the cursor mid-click. The four fixed outputs sit
//      at y 14/34/54/74 and image_1 starts at y 94 — everything above y≈84 is
//      stationary at every count. The card widget starts at y≈126 at the node's
//      smallest, so nothing collides.
//   2. It must not be on the TITLE BAR. That was the previous attempt: stationary,
//      but the title bar is where LiteGraph opens the rename editor on
//      double-click, so a quick double-press on + renamed the node. Do not move
//      it back up there.
//
// The node's three inputs are widget-sockets pinned to their own widget rows far
// below (y 276+), so the top-left region really is free.
const CTL = {
    left: 12, top: 8, boxH: 28, stepW: 30, numW: 40, gap: 5,
    label: "Image slots", labelGap: 10, minWidthForLabel: 300,
};

function countCtlRects(node, ctx) {
    // Pure arithmetic on node.size — never measured from what is rendered, which
    // is the rule that keeps a layout from feeding back into itself.
    let x = CTL.left;
    let label = null;
    if (node.size[0] >= CTL.minWidthForLabel) {
        // Measuring a STRING is fine; measuring the LAYOUT is what is forbidden.
        const w = ctx ? ctx.measureText(CTL.label).width : CTL.label.length * 6.2;
        label = [x, 0, w, CTL.boxH];   // y is set below, with the rest of the row
        x += w + CTL.labelGap;
    }
    // Input sockets (`seconds`, v3) occupy the left column's top rows; the stepper
    // sits below them. Counted from the inputs, never measured.
    const sockets = (node.inputs || []).filter((i) => !i.widget).length;
    const y = CTL.top + (sockets ? sockets * (window.LiteGraph?.NODE_SLOT_HEIGHT || 20) + 4 : 0);
    const r = (rx, rw) => [rx, y, rw, CTL.boxH];
    if (label) label[1] = y;
    return {
        label,
        minus: r(x, CTL.stepW),
        num:   r(x + CTL.stepW + CTL.gap, CTL.numW),
        plus:  r(x + CTL.stepW + CTL.gap + CTL.numW + CTL.gap, CTL.stepW),
    };
}

const inRect = (pos, r) =>
    pos[0] >= r[0] && pos[0] <= r[0] + r[2] && pos[1] >= r[1] && pos[1] <= r[1] + r[3];

// Which part is under the cursor, or "" — one function for draw, hover and click
// so they can never disagree about where the boxes are.
function ctlHit(node, pos) {
    const rects = node._lunaCtlRects || countCtlRects(node, null);
    if (inRect(pos, rects.minus)) return "minus";
    if (inRect(pos, rects.num)) return "num";
    if (inRect(pos, rects.plus)) return "plus";
    return "";
}

function drawCtl(node, ctx) {
    const FONT = "'Segoe UI', system-ui, sans-serif";
    ctx.save();
    ctx.font = `12px ${FONT}`;                 // set BEFORE measuring the label
    const rects = countCtlRects(node, ctx);
    node._lunaCtlRects = rects;                // hit tests reuse exactly what was drawn
    const value = Math.max(1, Math.min(MAX, Number(widget(node, COUNT_WIDGET)?.value) || 1));
    const hot = node._lunaCtlHover || "";

    if (rects.label) {
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";
        ctx.fillStyle = LUNA.muted;
        ctx.fillText(CTL.label, rects.label[0], rects.label[1] + rects.label[3] / 2 + 0.5);
    }

    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    const box = (r, glyph, key) => {
        const live = hot === key;
        // A disabled end-stop reads as broken unless it looks disabled.
        const dead = (key === "minus" && value <= 1) || (key === "plus" && value >= MAX);
        ctx.beginPath();
        if (ctx.roundRect) ctx.roundRect(r[0], r[1], r[2], r[3], 5);
        else ctx.rect(r[0], r[1], r[2], r[3]);
        ctx.fillStyle = live && !dead ? LUNA.accent : LUNA.panel;
        ctx.fill();
        ctx.strokeStyle = dead ? LUNA.border : live ? LUNA.accent : LUNA.accentSoft;
        ctx.lineWidth = live ? 1.6 : 1.2;
        ctx.stroke();
        // Amber glyphs on the steppers so the control reads as a control at a
        // glance, so nobody has to hunt for it.
        ctx.fillStyle = dead ? LUNA.muted : live ? LUNA.bg : (key === "num" ? LUNA.text : LUNA.accent);
        ctx.font = `bold ${key === "num" ? 14 : 17}px ${FONT}`;
        ctx.fillText(glyph, r[0] + r[2] / 2, r[1] + r[3] / 2 + (key === "num" ? 0.5 : 0));
    };
    box(rects.minus, "−", "minus");
    box(rects.num, String(value), "num");
    box(rects.plus, "+", "plus");
    ctx.restore();
}

/* ------------------------------------------------------------------- cards */

function buildUI(node) {
    injectCSS();
    const root = document.createElement("div");
    root.className = "luna-al";

    const bar = document.createElement("div");
    bar.className = "luna-al-bar";
    const cards = document.createElement("div");
    cards.className = "luna-al-cards";
    // A larger look at whichever card is selected. The grid shows what you have;
    // this shows what you are about to act on — the resize panel underneath edits
    // exactly this picture.
    const preview = document.createElement("div");
    preview.className = "luna-al-preview";
    const previewImg = document.createElement("img");
    const previewLbl = document.createElement("div");
    previewLbl.className = "luna-al-pvlbl";
    preview.append(previewImg, previewLbl);

    const panel = document.createElement("div");
    panel.className = "luna-al-panel";
    // The song editor takes the preview's place while a sound card is selected.
    const aed = document.createElement("div");
    aed.className = "luna-al-aed";
    aed.style.display = "none";
    // Drags and typing in here belong to the editor, never to the canvas.
    for (const ev of ["pointerdown", "keydown", "wheel"]) aed.addEventListener(ev, (e) => e.stopPropagation());
    // The clip editor, same idea, while a video card is selected.
    const ved = document.createElement("div");
    ved.className = "luna-al-ved";
    ved.style.display = "none";
    for (const ev of ["pointerdown", "keydown", "wheel"]) ved.addEventListener(ev, (e) => e.stopPropagation());
    // The shot timeline: what is anchored where in the video being generated.
    const tl = document.createElement("div");
    tl.className = "luna-al-tl";
    tl.style.display = "none";
    for (const ev of ["pointerdown", "keydown", "wheel"]) tl.addEventListener(ev, (e) => e.stopPropagation());
    // Three rows, one per kind (2026-10-16: separate timelines for image,
    // video and audio): each with its own + add, its own
    // cards, and its editor opening right under it. The shot timeline sits below.
    const vcards = document.createElement("div");
    vcards.className = "luna-al-cards";
    const acards = document.createElement("div");
    acards.className = "luna-al-cards";
    const ACCEPT = {
        p: "image/*",
        v: "video/*,.mp4,.mov,.mkv,.webm,.avi,.m4v",
        a: "audio/*,.mp3,.wav,.flac,.ogg,.opus,.m4a,.aac",
    };
    const heads = {};
    const section = (kind, title, grid, editors) => {
        const sec = document.createElement("div");
        sec.className = `luna-al-sec k-${kind}`;
        const h = document.createElement("div");
        h.className = "luna-al-sech";
        h.innerHTML = `<span class="tt"></span><span class="n"></span><span class="sp"></span>`;
        h.querySelector(".tt").textContent = title;
        const add = document.createElement("div");
        add.className = "luna-al-btn add";
        add.textContent = "+ add";
        add.title = `Add ${title.toLowerCase()}`;
        add.onclick = () => { picker.accept = ACCEPT[kind]; picker.click(); };
        h.appendChild(add);
        heads[kind] = h.querySelector(".n");
        sec.append(h, grid, ...editors);
        return sec;
    };
    // The node's overlap, a row of its own, with or without a song (SPEC § 2): frames each shot shares with the one
    // before, for the song, the vocal stem and the clip cards by shot number; out on `overlap`.
    const ovRow = document.createElement("div");
    ovRow.className = "luna-al-row luna-al-ovn";
    ovRow.title = "Frames each shot shares with the one before: the song, the vocal stem and the clip cards by shot number move by the shot length minus this. Goes out on the overlap output.";
    root.append(bar, ovRow,
        section("p", "Pictures", cards, [preview, panel]),
        section("v", "Clips", vcards, [ved]),
        section("a", "Sounds", acards, [aed]),
        tl);

    let selected = 0;                 // which card the resize panel edits
    let selAudio = -1;                // which sound card the song editor edits; -1 = none
    let selVideo = -1;                // which video card the clip editor edits; -1 = none
    // One editor at a time: picking a card of one kind clears the others.
    const pick = (kind, i) => {
        selAudio = kind === "a" ? i : -1;
        selVideo = kind === "v" ? i : -1;
        if (kind === "p") selected = i;
        render();
    };
    const natural = new Map();        // filename -> [w, h], filled as thumbs decode

    const picker = document.createElement("input");
    picker.type = "file";
    const ACCEPT_ALL = "image/*,audio/*,video/*,.mp3,.wav,.flac,.ogg,.opus,.m4a,.aac,.mp4,.mov,.mkv,.webm,.avi,.m4v";
    picker.accept = ACCEPT_ALL;
    picker.multiple = true;
    picker.style.display = "none";
    root.appendChild(picker);

    const mkBtn = (label, cls, fn) => {
        const b = document.createElement("div");
        b.className = `luna-al-btn${cls ? " " + cls : ""}`;
        b.textContent = label;
        b.onclick = fn;
        bar.appendChild(b);
        return b;
    };

    mkBtn("Upload", "", () => picker.click());
    mkBtn("Clear", "danger", () => {
        writePaths(node, []); syncCount(node, 0);
        // The cards go; the node's settings (⚙, the Overlap row, an old save's migrated overlap) stay.
        const kept = readMedia(node);
        writeMedia(node, { audio: [], video: [], pictures: [], shot: kept.shot, settings: kept.settings }); selAudio = -1; selVideo = -1;
        render();
    });

    // image_paths and asset_state are hidden once the cards own them — two
    // multiline boxes were taking most of the node. They stay reachable, because
    // image_paths remains the source of truth and typing into it must still work
    // if the frontend ever misbehaves.
    // image_count joins them: the title-bar stepper owns it now, and leaving a
    // second copy in the body would be the control that slides around. Fields
    // brings all three back, so there is still a way in if the stepper misbehaves.
    let fieldsShown = false;
    const setFieldsVisible = (show) => {
        for (const name of [PATHS_WIDGET, COUNT_WIDGET, "asset_state", MEDIA_WIDGET]) {
            const w = widget(node, name);
            if (!w) continue;
            // `w.hidden` is the flag that actually works. Measured in 1.48.x by
            // zeroing last_y and redrawing: with type="hidden" alone the widget is
            // still drawn, with hidden=true it is skipped. The text widgets only
            // looked hidden before because their DOM elements were hidden below —
            // image_count has no element, so it kept drawing its row.
            w.hidden = !show;
            if (show) {
                w.type = w._lunaType ?? w.type;
                delete w.computeSize;
            } else {
                w._lunaType = w._lunaType ?? w.type;
                w.type = "hidden";
                w.computeSize = () => [0, -4];
            }
            // Multiline STRING widgets are DOM textareas in this frontend, so
            // neither flag moves them — the element has to go too.
            if (w.element) {
                w.element.hidden = !show;
                w.element.style.display = show ? "" : "none";
            }
        }
        node.setDirtyCanvas(true, true);
    };
    // ⚙ — node-wide settings: target workflow, new-clip defaults, memory cap, sound, node.
    const gearBtn = mkBtn("⚙", "", () => openSettings(node));
    gearBtn.title = "Settings: target workflow (H3 / LTX / General), new-clip defaults, memory cap, nudge step, preview height";
    const fieldsBtn = mkBtn("Fields", "", () => {
        fieldsShown = !fieldsShown;
        setFieldsVisible(fieldsShown);
        fieldsBtn.textContent = fieldsShown ? "Hide fields" : "Fields";
    });
    fieldsBtn.title = "Show the raw image_paths / image_count / asset_state widgets";
    requestAnimationFrame(() => setFieldsVisible(false));

    // Says what the last click did, then clears itself. Node-local on purpose:
    // a toast for something this small would fire across the whole canvas.
    const note = document.createElement("div");
    note.className = "luna-al-note";
    bar.appendChild(note);
    let noteTimer = 0;
    const notice = (msg) => {
        note.textContent = msg || "";
        clearTimeout(noteTimer);
        if (msg) noteTimer = setTimeout(() => { note.textContent = ""; }, 3000);
    };
    node._lunaNotice = notice;

    const counter = document.createElement("div");
    counter.className = "luna-al-count";
    bar.appendChild(counter);

    picker.onchange = async () => {
        const files = Array.from(picker.files || []);
        picker.value = "";
        picker.accept = ACCEPT_ALL;      // a row's + add narrowed it for one pick
        if (!files.length) return;
        // One button for everything: sound files become sound cards, clips video
        // cards, the rest pictures.
        const sounds = files.filter(isAudioFile), clips = files.filter((f) => !isAudioFile(f) && isVideoFile(f));
        const pictures = files.filter((f) => !isAudioFile(f) && !isVideoFile(f));
        if (sounds.length) {
            const m = readMedia(node);
            const room = MAX_SOUNDS - m.audio.length;
            const names = await uploadFiles(sounds.slice(0, Math.max(0, room)));
            for (const file of names) m.audio.push({ file, start: 0, duration: 0, fill: settingsOf(m).audio.fill });
            if (names.length) { writeMedia(node, m); selAudio = m.audio.length - 1; selVideo = -1; }
            if (sounds.length > room) notice(`Room for ${MAX_SOUNDS} sounds; ${sounds.length - Math.max(0, room)} not added.`);
        }
        if (clips.length) {
            const m = readMedia(node);
            const room = MAX_VIDEOS - m.video.length;
            const names = await uploadFiles(clips.slice(0, Math.max(0, room)));
            for (const file of names) m.video.push({ file, ...NEW_VIDEO, ...settingsOf(m).video });   // ⚙ defaults
            if (names.length) { writeMedia(node, m); selVideo = m.video.length - 1; selAudio = -1; }
            if (clips.length > room) notice(`Room for ${MAX_VIDEOS} clips; ${clips.length - Math.max(0, room)} not added.`);
        }
        if (pictures.length) {
            const names = await uploadFiles(pictures);
            if (names.length) {
                const next = readPaths(node).concat(names).slice(0, MAX);
                writePaths(node, next);
                // The card list drives the socket count, both ways. The arrows still
                // override it by hand; the override holds until the list changes again.
                syncCount(node, next.length);
                if (!sounds.length && !clips.length) { selAudio = -1; selVideo = -1; }
            }
        }
        render();
    };

    let dragFrom = -1;

    function render() {
        const paths = readPaths(node);
        const media = readMedia(node);
        if (selAudio >= media.audio.length) selAudio = media.audio.length - 1;
        if (selVideo >= media.video.length) selVideo = media.video.length - 1;
        cards.replaceChildren();
        vcards.replaceChildren();
        acards.replaceChildren();
        counter.textContent = "";
        heads.p.textContent = `${paths.length} / ${MAX}`;
        heads.v.textContent = `${media.video.length} / ${MAX_VIDEOS}`;
        heads.a.textContent = `${media.audio.length} / ${MAX_SOUNDS}`;

        // No early return on an empty row: renderPreview/renderPanel handle "nothing
        // loaded" themselves, and skipping them once left a removed picture on screen.
        const empty = (grid, text) => {
            const e = document.createElement("div");
            e.className = "luna-al-empty";
            e.textContent = text;
            grid.appendChild(e);
        };
        if (!paths.length) empty(cards, "No pictures. + add, or type file names in image_paths (Fields).");
        if (!media.video.length) empty(vcards, "No clips. + add a video.");
        if (!media.audio.length) empty(acards, "No sounds. + add a song — sound 1 drives the shot.");
        if (paths.length > MAX) {
            const warn = document.createElement("div");
            warn.className = "luna-al-warn";
            warn.textContent = `MiniMax H3 takes at most ${MAX} references.`;
            cards.appendChild(warn);
        }
        // Each card's @ name and the tag H3 receives (cardTags), for the card titles.
        const tags = h3Tags(node), names = tags.names;

        paths.forEach((name, i) => {
            const card = document.createElement("div");
            card.className = "luna-al-card" + (i === selected && selAudio < 0 && selVideo < 0 ? " sel" : "");
            card.draggable = true;
            // Selecting is what makes a nine-image node editable: the resize panel
            // below edits whichever card is selected, so REF2VA references can stay
            // untouched while slots 1 and 2 get conformed.
            card.onclick = () => pick("p", i);

            const img = document.createElement("img");
            img.className = "luna-al-thumb";
            img.src = viewURL(name);

            const pic = document.createElement("div");
            pic.className = "luna-al-pic";
            pic.textContent = i + 1;
            // The real tag: a keyframe on an earlier card moves this one's <Picture n> down.
            const label = names.p[i];
            pic.title = label;
            card.title = `${label}\n${name}`;

            const cap = document.createElement("div");
            cap.className = "luna-al-cap";
            const dim = document.createElement("div");
            dim.className = "luna-al-dim";
            dim.textContent = "…";
            const nm = document.createElement("div");
            nm.className = "luna-al-name";
            nm.textContent = name;
            nm.title = name;
            cap.append(dim, nm);

            // Original size and aspect, read off the decoded image — the whole
            // point is to see what you actually loaded before choosing the
            // generation resolution.
            img.onload = () => {
                const w = img.naturalWidth, h = img.naturalHeight;
                const ar = describeAR(w, h);
                dim.textContent = `${w}×${h} ${ar.short}`;
                card.title = `${label}  —  ${w} × ${h} (${ar.long})\n${name}`;
                // The panel's INPUT → OUTPUT preview needs the decoded size, which
                // only exists once the thumbnail has loaded.
                natural.set(name, [w, h]);
                if (i === selected) renderPanel();
            };
            img.onerror = () => { dim.textContent = "missing from input folder"; };

            const x = document.createElement("div");
            x.className = "luna-al-x";
            x.textContent = "✕";
            x.title = "Remove";
            x.onclick = (ev) => {
                ev.stopPropagation();
                const next = readPaths(node);
                next.splice(i, 1);
                // A picture's role travels with it.
                const m = readMedia(node);
                if (m.pictures.length > i) { m.pictures.splice(i, 1); writeMedia(node, m); }
                writePaths(node, next);
                syncCount(node, next.length);
                render();
            };

            card.addEventListener("dragstart", () => { dragFrom = i; card.classList.add("drag"); });
            card.addEventListener("dragend", () => { dragFrom = -1; card.classList.remove("drag"); render(); });
            card.addEventListener("dragover", (ev) => { ev.preventDefault(); card.classList.add("over"); });
            card.addEventListener("dragleave", () => card.classList.remove("over"));
            card.addEventListener("drop", (ev) => {
                ev.preventDefault();
                card.classList.remove("over");
                if (dragFrom < 0 || dragFrom === i) return;
                const next = readPaths(node);
                const [moved] = next.splice(dragFrom, 1);
                next.splice(i, 0, moved);
                // Keep each picture's role with the picture, not with the slot.
                const m = readMedia(node);
                if (m.pictures.some((p) => p?.role === "key")) {
                    while (m.pictures.length < Math.max(next.length, dragFrom + 1)) m.pictures.push({});
                    const [mr] = m.pictures.splice(dragFrom, 1);
                    m.pictures.splice(i, 0, mr);
                    writeMedia(node, m);
                }
                writePaths(node, next);   // reorder IS the renumbering
                render();
            });

            card.append(img, pic, x, cap);
            cards.appendChild(card);
        });

        // Video cards after the pictures: V1-V3, MiniMax H3's <Video k> in this order.
        media.video.forEach((c, i) => {
            const card = document.createElement("div");
            card.className = "luna-al-card" + (i === selVideo ? " sel" : "");
            card.style.cursor = "pointer";
            // Read at hover: whether the clip has sound (its <Audio n>) is only known once its info is in.
            card.title = `${names.v[i]}\n${c.file}`;
            card.addEventListener("mouseenter", () => {
                const t = h3Tags(node);
                card.title = `${t.names.v[i]}${t.videoAudio[i] ? `, its sound ${t.videoAudio[i]}` : ""}\n${c.file}`;
            });
            card.onclick = () => pick("v", i);
            const img = document.createElement("img");
            img.className = "luna-al-vthumb";
            img.src = stripURL(c.file, 1);
            const pic = document.createElement("div");
            pic.className = "luna-al-pic vid";
            pic.textContent = `V${i + 1}`;
            const cap = document.createElement("div");
            cap.className = "luna-al-cap";
            const dim = document.createElement("div");
            dim.className = "luna-al-dim";
            dim.textContent = "reading…";
            const nm = document.createElement("div");
            nm.className = "luna-al-name";
            nm.textContent = c.file;
            cap.append(dim, nm);
            const x = document.createElement("div");
            x.className = "luna-al-x";
            x.textContent = "✕";
            x.title = "Remove";
            x.onclick = (ev) => {
                ev.stopPropagation();
                const m = readMedia(node);
                m.video.splice(i, 1);
                writeMedia(node, m);
                if (selVideo >= m.video.length) selVideo = m.video.length - 1;
                render();
            };
            card.append(img, pic, x, cap);
            vcards.appendChild(card);
            mediaInfo(c.file).then((mi) => {
                dim.textContent = `${(mi.duration || 0).toFixed(1)} s · ${+(mi.fps || 0).toFixed(2)} fps${mi.audio ? "" : " · no sound"}`;
                // A silent clip shifts the <Audio n> numbers, and clip V1's rate is the General shot's rate: redraw
                // the timeline now that they are known.
                if ((!mi.audio && c.sound !== "drop") || i === 0) renderTimeline();
            }).catch(() => { dim.textContent = "cannot be read"; });
        });

        // Sound cards after the pictures. Not draggable: they are numbered on their
        // own (A1 is the song), and pictures never trade places with them.
        media.audio.forEach((c, i) => {
            const card = document.createElement("div");
            card.className = "luna-al-card snd" + (i === selAudio ? " sel" : "");
            // Read at hover: a silent clip moves the <Audio n> numbers once its info is in.
            const soundTitle = (t) => `${t.names.a[i]}${i === 0 ? " — the song on the audio output" : ""}\n${c.file}`;
            card.title = soundTitle(tags);
            card.addEventListener("mouseenter", () => { card.title = soundTitle(h3Tags(node)); });
            card.onclick = () => pick("a", i);
            const cv = document.createElement("canvas");
            cv.className = "luna-al-snd";
            const pic = document.createElement("div");
            pic.className = "luna-al-pic snd";
            pic.textContent = `A${i + 1}`;
            const cap = document.createElement("div");
            cap.className = "luna-al-cap";
            const dim = document.createElement("div");
            dim.className = "luna-al-dim";
            dim.textContent = "reading…";
            const nm = document.createElement("div");
            nm.className = "luna-al-name";
            nm.textContent = c.file;
            cap.append(dim, nm);
            const x = document.createElement("div");
            x.className = "luna-al-x";
            x.textContent = "✕";
            x.title = "Remove";
            x.onclick = (ev) => {
                ev.stopPropagation();
                const m = readMedia(node);
                m.audio.splice(i, 1);
                writeMedia(node, m);
                if (selAudio >= m.audio.length) selAudio = m.audio.length - 1;
                if (!m.audio.length) selAudio = -1;
                render();
            };
            card.append(cv, pic, x, cap);
            acards.appendChild(card);
            loadPeaks(c.file).then((info) => {
                dim.textContent = `${clock(info.duration).replace(/\.\d+$/, "")} · ${(info.sampleRate / 1000).toFixed(1)} kHz`;
                const dur = soundWindow(c, i, shotLen());
                setTimeout(() => drawWave(cv, info, 0, info.duration, c.start, c.start + dur, { step: 2 }), 0);
            }).catch(() => { dim.textContent = "cannot be read"; });
        });

        syncAccent(node, root);
        renderOverlap();
        renderTimeline();
        renderPreview();
        renderPanel();
        renderAudio();
        renderVideo();

        // The widget's getHeight depends on how many cards there are, so a redraw
        // has to follow any change to the list or the node keeps its old height.
        node.setDirtyCanvas(true, true);
    }

    function renderOverlap() {
        const m = readMedia(node), [ov, warn] = nodeOverlap(m);
        ovRow.innerHTML = `<div class="luna-al-lbl">Overlap</div><div class="luna-al-seg"></div><span class="luna-al-io"></span>`;
        const seg = ovRow.querySelector(".luna-al-seg");
        for (const v of OVERLAP_ROW) {
            const o = document.createElement("div");
            o.className = "luna-al-opt" + (v === ov ? " on" : "");
            o.textContent = v === OVERLAP_ROW[OVERLAP_ROW.length - 1] ? `${v} frames` : String(v);
            o.onclick = () => {
                const mm = readMedia(node);
                mm.settings = { ...(mm.settings || {}), overlap: v };
                writeMedia(node, mm);
                render();
            };
            seg.appendChild(o);
        }
        ovRow.querySelector(".luna-al-io").textContent = warn || (OVERLAP_ROW.includes(ov) ? "" : `${ov} (the first clip)`);
    }

    function renderPreview() {
        const paths = readPaths(node);
        if (selAudio >= 0 || selVideo >= 0) { preview.style.display = "none"; return; }
        if (!paths.length) {
            // Drop the picture as well as hiding the box: leaving the src set kept
            // the removed image decoded, and the dataset guard below would then
            // skip re-setting it if the same file were uploaded again.
            preview.style.display = "none";
            previewImg.removeAttribute("src");
            previewImg.dataset.name = "";
            previewLbl.textContent = "";
            selected = 0;
            return;
        }
        if (selected >= paths.length) selected = 0;
        preview.style.display = "";
        preview.style.height = previewH(node) + "px";
        const name = paths[selected];
        // Reuse the card's already-decoded copy: same URL, so the browser serves
        // it from cache instead of fetching the picture a second time.
        if (previewImg.dataset.name !== name) {
            previewImg.dataset.name = name;
            previewImg.src = viewURL(name);
        }
        const nat = natural.get(name);
        previewLbl.textContent = nat
            ? `${picLabel(selected)}  ${nat[0]}×${nat[1]} ${describeAR(nat[0], nat[1]).short}`
            : picLabel(selected);
    }

    /* ------------------------------------------------------------- resize panel */

    function renderPanel() {
        panel.replaceChildren();
        panel.style.display = selAudio >= 0 || selVideo >= 0 ? "none" : "";
        if (selAudio >= 0 || selVideo >= 0) return;
        const paths = readPaths(node);
        if (!paths.length) return;
        if (selected >= paths.length) selected = 0;

        const st = itemAt(node, selected);
        const name = paths[selected];
        const nat = natural.get(name);

        const row = (label) => {
            const r = document.createElement("div");
            r.className = "luna-al-row";
            if (label !== null) {
                const l = document.createElement("div");
                l.className = "luna-al-lbl";
                l.textContent = label;
                r.appendChild(l);
            }
            panel.appendChild(r);
            return r;
        };
        const seg = (parent, opts, current, onPick) => {
            const s = document.createElement("div");
            s.className = "luna-al-seg";
            for (const [val, text] of opts) {
                const o = document.createElement("div");
                o.className = "luna-al-opt" + (val === current ? " on" : "");
                o.textContent = text;
                o.onclick = () => onPick(val);
                s.appendChild(o);
            }
            parent.appendChild(s);
            return s;
        };
        const set = (patch) => { writeItem(node, selected, patch); render(); };

        // which picture the panel is editing
        const head = row(null);
        const who = document.createElement("div");
        who.className = "luna-al-io";
        const out = nat ? predictSize(nat[0], nat[1], st) : null;
        who.innerHTML = nat
            ? `<b>${picLabel(selected).replace(/</g, "&lt;").replace(/>/g, "&gt;")}</b> &nbsp; ${nat[0]}×${nat[1]} → <b>${out[0]}×${out[1]}</b>`
            : `<b>${picLabel(selected).replace(/</g, "&lt;").replace(/>/g, "&gt;")}</b>`;
        head.appendChild(who);

        // Reference for H3, or a keyframe anchored on the shot.
        const mm = readMedia(node);
        roleRow(panel, "p", selected, [["ref", "Reference"], ["key", "Keyframe on the shot"]],
                picRole(mm, selected), Number(mm.pictures[selected]?.at) || 0, 1);

        const r1 = row("Size");
        seg(r1, [["off", "Off"], ["max_mp", "Max MP"], ["longest_side", "Longest"],
                 ["scale_factor", "Scale ×"]], st.mode, (v) => set({ mode: v }));

        if (st.mode !== "off") {
            const r2 = row("");
            const inp = document.createElement("input");
            inp.className = "luna-al-num";
            inp.type = "number";
            inp.step = st.mode === "max_mp" ? "0.1" : (st.mode === "scale_factor" ? "0.05" : "32");
            inp.value = st.mode === "max_mp" ? st.max_mp
                      : st.mode === "longest_side" ? st.longest_side : st.scale_factor;
            inp.onchange = () => {
                const k = st.mode === "max_mp" ? "max_mp"
                        : st.mode === "longest_side" ? "longest_side" : "scale_factor";
                set({ [k]: Number(inp.value) });
            };
            r2.appendChild(inp);
            const up = document.createElement("div");
            up.className = "luna-al-opt" + (st.allow_upscale ? " on" : "");
            up.textContent = "Upscale";
            up.title = "H3 never upscales a reference itself — this is the only place it can happen";
            up.onclick = () => set({ allow_upscale: !st.allow_upscale });
            r2.appendChild(up);
        }

        const r3 = row("Ratio");
        seg(r3, [["", "Source"], ["1:1", "1:1"], ["2:3", "2:3"], ["3:2", "3:2"],
                 ["9:16", "9:16"], ["16:9", "16:9"]], st.ratio, (v) => set({ ratio: v }));

        if (st.ratio) {
            const r4 = row("");
            seg(r4, [["crop", "Crop to fill"], ["pad", "Pad"]], st.ratio_action,
                (v) => set({ ratio_action: v }));
        }

        const r5 = row("Snap");
        seg(r5, [[0, "Off"], [8, "8"], [16, "16"], [32, "32"], [64, "64"]],
            Number(st.snap) || 0, (v) => set({ snap: v }));

        const r6 = row(null);
        const all = document.createElement("div");
        all.className = "luna-al-btn";
        all.textContent = "Apply to all";
        all.title = "Use these settings for every image and drop per-image overrides";
        all.onclick = () => {
            const { mode, max_mp, longest_side, scale_factor, ratio, ratio_action, snap, allow_upscale } = st;
            writeItem(node, 0, { mode, max_mp, longest_side, scale_factor, ratio, ratio_action, snap, allow_upscale }, true);
            render();
        };
        const reset = document.createElement("div");
        reset.className = "luna-al-btn danger";
        reset.textContent = "Reset";
        reset.onclick = () => { const w = widget(node, STATE_WIDGET); if (w) { w.value = ""; w.callback?.(""); } render(); };
        r6.append(all, reset);
    }

    // Picture card i's @ name and what H3 calls it: its real tag when it is a reference, or that it is a keyframe.
    const picLabel = (i) => h3Tags(node).names.p[i] || `@image${i + 1}`;

    /* ------------------------------------------------------- shot timeline */

    // The shot length in frames, the same one count as asset_loader (asset_timeline.shot_frames): the wired
    // seconds at the shot's rate when there are any, else the length kept on the node (124 frames = MiniMax H3's
    // 5.17 s default), rounded UP onto the target's grid; past 362 on H3 a warning, never a cut.
    function shotLen() {
        const m = readMedia(node), grid = targetOf(m).grid, rate = shotRate(m), ws = wiredSeconds();
        const wired = ws.wired && ws.value > 0;
        const frames = wired ? framesFor(ws.value, rate, grid) : framesUp(m.shot?.frames || 124, grid);
        return { frames, from: wired ? ws.from : "", rate, grid, seconds: frames / rate, warning: framesWarning(frames, grid) };
    }

    // Patch one card's role/placement. kind: "p" (picture index), "v", "a".
    function setPlacement(kind, i, patch) {
        const m = readMedia(node);
        if (kind === "p") {
            while (m.pictures.length <= i) m.pictures.push({});
            m.pictures[i] = { ...m.pictures[i], ...patch };
        } else {
            const list = kind === "v" ? m.video : m.audio;
            if (!list[i]) return;
            list[i] = { ...list[i], ...patch };
        }
        writeMedia(node, m);
        render();
    }

    // A clip's length on the shot, in output frames — what the guide will hold.
    const clipLen = (c) => {
        const f = Math.max(1, Number(c.frames) || 124);
        const n = (c.count || "target") === "target" ? gridFrames(f, targetOf(readMedia(node)).grid) : f;
        return Math.ceil(n / Math.max(1, Number(c.nth) || 1));
    };

    // The role row every editor shows: which role, and the frame when it has one.
    function roleRow(host, kind, i, roles, cur, at, len) {
        const S = shotLen().frames;
        const row = document.createElement("div");
        row.className = "luna-al-row";
        row.innerHTML = `<div class="luna-al-lbl">Role</div><div class="luna-al-seg"></div>`;
        const segEl = row.querySelector(".luna-al-seg");
        for (const [val, text] of roles) {
            const o = document.createElement("div");
            o.className = "luna-al-opt" + (val === cur ? " on" : "");
            o.textContent = text;
            o.onclick = () => setPlacement(kind, i, { role: val, at: Math.min(at || 0, Math.max(0, S - len)) });
            segEl.appendChild(o);
        }
        host.appendChild(row);
        if (cur === "ref") return;
        const r2 = document.createElement("div");
        r2.className = "luna-al-row";
        r2.innerHTML = `<div class="luna-al-lbl">At frame</div><div class="luna-al-nf"><span class="luna-al-nud" data-a="first" title="The first frame">First</span><input class="at" style="width:52px"><span class="luna-al-nud" data-a="last" title="As late as it fits">Last</span><span class="luna-al-lock"></span></div>`;
        const inp = r2.querySelector(".at");
        inp.value = String(at || 0);
        r2.querySelector(".luna-al-lock").textContent = `of ${S} · ${((at || 0) / shotLen().rate).toFixed(2)} s`;
        const set = (v) => setPlacement(kind, i, { at: Math.max(0, Math.min(Math.round(v), Math.max(0, S - len))) });
        inp.onchange = () => { const v = Number(inp.value); if (Number.isFinite(v)) set(v); };
        inp.addEventListener("keydown", (e) => { if (e.key === "Enter") inp.blur(); e.stopPropagation(); });
        r2.querySelector('[data-a="first"]').onclick = () => set(0);
        r2.querySelector('[data-a="last"]').onclick = () => set(S - len);
        host.appendChild(r2);
    }

    function renderTimeline() {
        const m = readMedia(node), paths = readPaths(node);
        if (!paths.length && !m.video.length && !m.audio.length) { tl.style.display = "none"; tl.replaceChildren(); return; }
        tl.style.display = "";
        const { frames: S, from, rate, warning } = shotLen();
        const tags = h3Tags(node);
        tl.innerHTML = `
            <div class="luna-al-tlh"><span>Shot</span><b>${S} frames · ${(S / rate).toFixed(2)} s</b>
              <span>${from ? `from ${from.replace(/[<>&]/g, "")}` : "set here"}</span><span class="sp"></span>
              ${from ? "" : `<span class="luna-al-nud" data-s="-1" title="17 frames shorter">−</span><span class="luna-al-nud" data-s="1" title="17 frames longer">+</span>`}</div>
            <div class="luna-al-lane"><span></span><div class="luna-al-truler"></div></div>
            <div class="luna-al-lane"><span class="ln">Pictures</span><div class="luna-al-track" data-k="p"></div></div>
            <div class="luna-al-lane"><span class="ln">Clips</span><div class="luna-al-track" data-k="v"></div></div>
            <div class="luna-al-lane"><span class="ln">Sounds</span><div class="luna-al-track" data-k="a"></div></div>
            <div class="luna-al-lane"><span class="ln">References</span><div class="luna-al-refs"></div></div>
            <div class="luna-al-tlnote"></div>`;
        const q = (s) => tl.querySelector(s);
        const ruler = q(".luna-al-truler"), stepR = S > 200 ? 48 : 24;
        for (let f = 0; f <= S; f += stepR) {
            const sp = document.createElement("span");
            sp.style.left = `${f / S * 100}%`;
            sp.textContent = String(f);
            ruler.appendChild(sp);
        }
        tl.querySelectorAll("[data-s]").forEach((b) => {
            b.onclick = () => {
                const mm = readMedia(node);
                mm.shot = { frames: Math.max(1, Math.min(1000, stepGrid(S, Number(b.dataset.s), targetOf(mm).grid))) };
                writeMedia(node, mm); render();
            };
        });

        // One draggable item on a lane; dragging changes its frame, a click selects its card.
        const place = (track, cls, kind, i, at, len, label, title, isSel) => {
            const el = document.createElement("div");
            el.className = `luna-al-blk ${cls}` + (isSel ? " sel" : "");
            el.textContent = label;
            el.title = title;
            const pos = (a) => {
                if (cls === "key") el.style.left = `${(a + 0.5) / S * 100}%`;
                else { el.style.left = `${a / S * 100}%`; el.style.width = `${Math.max(0.8, Math.min(len, S - a) / S * 100)}%`; }
            };
            pos(at);
            let d = null;
            el.addEventListener("pointerdown", (e) => { d = { x: e.clientX, at, moved: false }; el.setPointerCapture(e.pointerId); e.preventDefault(); });
            el.addEventListener("pointermove", (e) => {
                if (!d) return;
                const df = (e.clientX - d.x) / track.getBoundingClientRect().width * S;
                const a = Math.max(0, Math.min(Math.round(d.at + df), Math.max(0, S - (cls === "key" ? 1 : len))));
                if (a !== d.cur) { d.cur = a; d.moved = true; pos(a); el.title = `${title} — frame ${a}`; }
            });
            const up = () => {
                if (!d) return;
                const moved = d.moved, a = d.cur;
                d = null;
                // LTX anchors a clip only at frame 0 or 8k+1: the drop lands on the nearest.
                const at2 = moved && cls === "vid" && targetOf(readMedia(node)).grid === "ltx" && a > 0
                    ? Math.min(Math.max(1, Math.round((a - 1) / 8) * 8 + 1), Math.max(0, S - len)) : a;
                if (moved) setPlacement(kind, i, { at: at2 }); else pick(kind, i);
            };
            el.addEventListener("pointerup", up);
            el.addEventListener("pointercancel", up);
            track.appendChild(el);
        };

        const tp = q('[data-k="p"]'), tv = q('[data-k="v"]'), ta = q('[data-k="a"]'), refs = q(".luna-al-refs");
        const chip = (cls, text, kind, i, title) => {
            const c = document.createElement("span");
            c.className = `luna-al-chip ${cls}`;
            c.textContent = text;
            c.title = title;
            c.onclick = () => pick(kind, i);
            refs.appendChild(c);
        };
        paths.forEach((name, i) => {
            if (picRole(m, i) === "key") {
                const at = Number(m.pictures[i]?.at) || 0;
                place(tp, "key", "p", i, at, 1, String(i + 1), `${tags.names.p[i]}: picture card ${i + 1} at frame ${at}, not sent as a reference`, selAudio < 0 && selVideo < 0 && selected === i);
            } else chip("p", tags.names.p[i], "p", i, `@image${i + 1} in the prompt becomes ${tags.pictures[i]}: picture card ${i + 1} (${name})`);
        });
        m.video.forEach((c, k) => {
            if (clipRole(c) === "guide") {
                const len = clipLen(c), at = Number(c.at) || 0;
                place(tv, "vid", "v", k, at, len, `V${k + 1} · ${at}–${at + len - 1}`, `${tags.names.v[k]}: clip V${k + 1} on the shot, not sent as a reference`, selVideo === k);
            } else chip("v", tags.names.v[k] + (tags.videoAudio[k] ? ` + ${tags.videoAudio[k]}` : ""), "v", k,
                `@video${k + 1} in the prompt becomes ${tags.videos[k]}: clip card V${k + 1}` + (tags.videoAudio[k] ? `; its sound is ${tags.videoAudio[k]}` : ""));
        });
        m.audio.forEach((c, j) => {
            const role = soundRole(c, j);
            if (role === "ref") { chip("a", tags.names.a[j], "a", j, `@audio${j + 1} in the prompt becomes ${tags.sounds[j]}: sound card A${j + 1}`); return; }
            const at = Number(c.at) || 0;
            place(ta, "snd", "a", j, at, S - at, `A${j + 1} · ${role === "lock" ? "locked" : "guide"} from ${at}`,
                `${tags.names.a[j]}: sound A${j + 1} ${role === "lock" ? "locked — the picture follows it" : "as a guide"}, from frame ${at}, not sent as a reference`, selAudio === j);
        });
        if (!refs.children.length) refs.innerHTML = `<span class="luna-al-tlnote">none — everything is on the shot</span>`;
        q(".luna-al-tlnote").textContent = (warning ? `WARNING: ${warning} ` : "")
            + `${targetOf(m).label}, ${+rate.toFixed(3)} fps: ${targetOf(m).note}`
            + (targetOf(m).grid === "h3" ? " H3 numbers only the references, in this order." : "");
    }

    /* -------------------------------------------------------- song editor */

    // What the `seconds` input says, when it is wired. The frontend can read a
    // plain number widget on the node at the other end (a Float primitive, a
    // canvas node's seconds); anything computed is only known after a run, so the
    // last run's value stands in, and the editor says which it is showing.
    function wiredSeconds() { return wiredNumber("seconds", (last) => last?.wired && last.seconds > 0 ? last.seconds : null); }
    function wiredShot() { return wiredNumber("shot", (last) => last?.shot > 0 ? last.shot : null); }
    // song_start may be 0 (a scene at the very top of the song).
    function wiredSongStart() { return wiredNumber("song_start", (last) => last?.start_wired ? last.start : null, 0); }
    function wiredNumber(name, fromLast, min = Number.MIN_VALUE) {
        let value = null, from = "the wire";
        const ws = wiredSource(node, name);
        if (!ws) return { wired: false, value: null, from: "" };
        try {
            from = ws.src?.title || from;
            const v = sourceValue(ws.src, ws.out);
            if (v != null && Number.isFinite(Number(v)) && Number(v) >= min) value = Number(v);
        } catch { /* graph mid-change: fall back to the last run */ }
        if (value == null) {
            const v = fromLast(_lastSong.get(node.id));
            if (v != null) { value = v; from += ", last run"; }
        }
        return { wired: true, value, from };
    }

    // Shot mode: shot n's window starts at song_start + (n-1) x (shot length - overlap / the shot's rate)
    // (media_io.song_shot_start); the overlap is the node's (nodeOverlap).
    const shotStep = (len, overlap, rate = H3_FPS) => len - overlap / rate;
    const shotCount = (T, songStart, len, overlap, rate = H3_FPS) => {
        const step = shotStep(len, overlap, rate);
        return step > 0 && T > songStart ? Math.ceil((T - songStart) / step) : 0;
    };
    // Rows of the editor beyond the base, for the node's arithmetic height. The vocal stem row (sound card 1)
    // takes the place of the old Snap row, which is in the base.
    const audioExtraRows = () => {
        const c = readMedia(node).audio[selAudio];
        const role = c ? soundRole(c, selAudio) : "ref";
        return 1 + (c?.position === "shot" ? 1 : 0) + 1 + (role !== "ref" ? 1 : 0);   // Position, Shot, Role, At frame
    };

    let player = null, playRaf = 0;
    function stopPlay() {
        cancelAnimationFrame(playRaf);
        player?.pause();
        aed.querySelectorAll(".luna-al-ph").forEach((p) => { p.style.display = "none"; });
        const b = aed.querySelector(".luna-al-play");
        if (b) b.textContent = "▶";
    }

    function renderAudio() {
        stopPlay();
        const media = readMedia(node);
        if (selAudio < 0 || !media.audio[selAudio]) {
            aed.style.display = "none";
            aed.replaceChildren();
            return;
        }
        aed.style.display = "";
        const idx = selAudio;
        const card = { fill: "silence", position: "free", song_start: 0, shot: 1, ...media.audio[idx] };
        // The window is the shot length (one count per run); the overlap is the node's row.
        const shot = shotLen(), rate = shot.rate, grid = targetOf(media).grid, overlap = nodeOverlap(media)[0];
        const nudge = () => Number(settingsOf(media).audio.nudge) || 0.1;
        // A wired song_start (Luna-H3-ShotList) places the window; the card's start and
        // its shot-number mode stand aside, exactly as in Python.
        const wss = wiredSongStart(), placed = wss.wired;
        const byShot = card.position === "shot" && !placed;
        let start = Number(card.start) || 0, songStart = Number(card.song_start) || 0;
        let dur = soundWindow(card, idx, shot), info = null;
        // A wired seconds input owns the song's length (the shot); cards 2-3 keep their own, at most the shot.
        const isSong = idx === 0;
        const ws = wiredSeconds(), wsh = wiredShot();
        const lockLen = ws.wired && idx === 0;
        const placedNote = () => notice(`The start comes from the song_start input (${wss.from}).`);
        const shotNo = () => Math.max(1, Math.round(wsh.wired && wsh.value > 0 ? wsh.value : Number(card.shot) || 1));

        aed.innerHTML = `
            <div class="luna-al-ahead"><b>${escTag(h3Tags(node).names.a[idx])}</b><span class="nm"></span><span class="sp"></span><span class="meta">reading the song…</span>
              <div class="luna-al-play" title="Play the window">▶</div></div>
            <div class="luna-al-ztools"><span class="l">Zoom</span><span class="luna-al-nud" data-z="out" title="Zoom out">−</span><span class="luna-al-nud" data-z="in" title="Zoom in (or the mouse wheel over the song)">+</span><span class="luna-al-nud zfit" data-z="fit" title="The whole song">Fit</span><span class="zx"></span><span class="sp"></span><span class="zr"></span></div>
            <div class="luna-al-wave"><canvas></canvas><div class="luna-al-win"><div class="h l"></div><div class="h r"></div><div class="tag"></div></div><div class="luna-al-ph"></div></div>
            <div class="luna-al-ruler"></div>
            <div class="luna-al-ovw" title="The whole song; the light box is what the waveform shows — drag it to scroll"><div class="vw"></div><div class="wn"></div></div>
            <div class="luna-al-nums">
              <span class="l stl">Start</span><div class="luna-al-nf"><span class="luna-al-nud" data-n="-1" title="0.1 s earlier">◀</span><input class="st"><span class="luna-al-nud" data-n="1" title="0.1 s later">▶</span></div>
              <span class="l">Duration</span><div class="luna-al-nf"><input class="du"><span class="luna-al-lock dl"></span></div>
              <span class="l">Window</span><div class="luna-al-nf"><input class="en" readonly tabindex="-1" style="width:120px"></div><span></span><span></span>
            </div>
            <div class="rolehost" style="display:flex;flex-direction:column;gap:5px"></div>
            <div class="luna-al-row"><div class="luna-al-lbl">Position</div><div class="luna-al-seg pos"></div></div>
            <div class="luna-al-row shotrow"><div class="luna-al-lbl">Shot</div>
              <div class="luna-al-nf"><span class="luna-al-nud" data-s="-1" title="Previous shot">◀</span><input class="sn" style="width:46px"><span class="luna-al-nud" data-s="1" title="Next shot">▶</span><span class="luna-al-lock sl"></span></div></div>
            <div class="luna-al-row"><div class="luna-al-lbl">Song ends</div><div class="luna-al-seg fill"></div></div>
            <div class="luna-al-row vocrow"><div class="luna-al-lbl">Vocal stem</div><div class="luna-al-nf"><span class="luna-al-lock vn"></span><span class="luna-al-opt vadd" title="The vocal stem of this song (for H3's lip-sync lock): cut to the same window, out on vocals">+ stem</span><span class="luna-al-opt vx" title="Remove the vocal stem">Remove</span></div></div>
            <div class="luna-al-astat"></div>`;
        const $ = (s) => aed.querySelector(s);
        $(".nm").textContent = card.file;
        const wave = $(".luna-al-wave"), win = $(".luna-al-win"), ovw = $(".luna-al-ovw");
        const phW = aed.querySelector(".luna-al-ph");
        // The visible part of the song: [v0, v1] seconds. Kept per song across
        // re-renders, so a zoom survives every edit. Whole song until zoomed.
        let view = _songView.get(card.file) || null;
        const T0 = () => info?.duration || 0;
        const vspan = () => (view ? view.v1 - view.v0 : T0());
        const setView = (v0, span) => {
            if (!Number.isFinite(v0) || !Number.isFinite(span)) return;
            const T = T0(), sp = Math.max(Math.min(MIN_VIEW_S, T), Math.min(T, span));
            const a0 = Math.max(0, Math.min(v0, T - sp));
            view = sp >= T - 1e-6 ? null : { v0: a0, v1: a0 + sp };
            if (view) _songView.set(card.file, view); else _songView.delete(card.file);
        };
        const V0 = () => (view ? view.v0 : 0), V1 = () => (view ? view.v1 : T0());
        const iSt = $(".st"), iDu = $(".du"), iEn = $(".en"), iSn = $(".sn");
        roleRow($(".rolehost"), "a", idx,
                idx === 0 ? [["lock", "Lock to shot"], ["guide", "Guide at frame"], ["ref", "Reference"]]
                          : [["guide", "Guide at frame"], ["ref", "Reference"]],   // the shot has one locked sound: the song
                soundRole(card, idx), Number(card.at) || 0, 1);

        const total = () => info?.duration || 0;
        // dur 0 means "to the end of the file"; a wired seconds input wins over both.
        // Always the shot length; an edge drag changes it, and on release the node's shot length follows.
        const eff = () => dur;
        const step = () => shotStep(eff(), overlap, rate);
        // The window this run uses. In shot mode it is derived from where shot 1 begins.
        const winStart = () => placed ? (wss.value ?? start) : byShot ? songStart + (shotNo() - 1) * step() : start;
        const clampWin = (t) => Math.max(byShot ? (shotNo() - 1) * Math.max(0, step()) : 0,
            Math.min(t, Math.max(0, total() - Math.min(eff(), total()))));
        // Moving the window: free mode moves it; shot mode moves every shot together.
        const setWin = (t) => {
            const c = clampWin(t);
            if (byShot) songStart = Math.max(0, c - (shotNo() - 1) * step()); else start = c;
        };

        const seg = (host, opts, cur, pick) => {
            host.replaceChildren();
            for (const [val, text] of opts) {
                const o = document.createElement("div");
                o.className = "luna-al-opt" + (val === cur ? " on" : "");
                o.textContent = text;
                o.onclick = () => pick(val);
                host.appendChild(o);
            }
        };
        seg($(".pos"), [["free", "Free"], ["shot", "By shot number"]], card.position, (v) => {
            if (v === card.position) return;
            if (v === "shot") {
                // Start from what is on screen: this window becomes shot 1.
                commit({ position: "shot", song_start: +start.toFixed(6), shot: 1, duration: +dur.toFixed(6) });
            } else commit({ position: "free", start: +winStart().toFixed(6) });
        });
        seg($(".fill"), [["silence", "Silence"], ["loop", "Loop the song"]], card.fill, (v) => commit({ fill: v }));
        // The vocal stem rides sound card 1 only: the same start, length and fill as the song.
        $(".vocrow").style.display = idx === 0 ? "" : "none";
        $(".vn").textContent = card.vocals ? card.vocals : "none";
        $(".vx").style.display = card.vocals ? "" : "none";
        $(".vadd").textContent = card.vocals ? "Change" : "+ stem";
        $(".vadd").onclick = () => {
            const pickStem = document.createElement("input");
            pickStem.type = "file";
            pickStem.accept = "audio/*,.mp3,.wav,.flac,.ogg,.opus,.m4a,.aac";
            pickStem.onchange = async () => {
                const f = Array.from(pickStem.files || []).filter(isAudioFile);
                if (!f.length) return;
                const [name] = await uploadFiles(f.slice(0, 1));
                if (name) commit({ vocals: name });
            };
            pickStem.click();
        };
        $(".vx").onclick = () => commit({ vocals: "" });
        $(".shotrow").style.display = byShot ? "" : "none";
        $(".pos").parentElement.style.display = placed ? "none" : "";
        $(".stl").textContent = byShot ? "Song start" : "Start";

        function commit(patch = {}) {
            const m = readMedia(node);
            if (!m.audio[idx]) return;
            // A new length (an edge, the Duration field): the song's is the node's shot length, rounded UP onto the
            // grid (a wired seconds input owns it instead); a card 2-3's is its own Duration, at most the shot. An
            // unchanged length keeps what is stored, so a card on 0 (= the shot) keeps following the shot.
            const len = soundLengthPatch(idx, dur, shot);
            const changed = Math.abs(dur - soundWindow(card, idx, shot)) > 1e-6;
            if (!ws.wired && len.shotFrames && changed) m.shot = { frames: len.shotFrames };
            const duration = isSong || changed ? +len.duration.toFixed(6) : Number(m.audio[idx].duration) || 0;
            m.audio[idx] = { ...m.audio[idx], start: +start.toFixed(6), song_start: +songStart.toFixed(6),
                             duration, ...patch };
            writeMedia(node, m);
            render();
        }

        function draw() {
            const e = eff(), T = total(), a = winStart(), n = shotNo();
            const N = byShot && T ? shotCount(T, songStart, e, overlap, rate) : 0;
            iSt.value = clock(byShot ? songStart : placed ? a : start);
            iSt.readOnly = placed;
            iSt.classList.toggle("lockd", placed);
            iDu.value = `${e.toFixed(2)} s`;
            iEn.value = `${clock(a)} – ${clock(a + e)}`;
            iDu.readOnly = lockLen;
            iDu.classList.toggle("lockd", lockLen);
            $(".dl").textContent = !isSong ? `at most the shot, ${shot.seconds.toFixed(2)} s` : (ws.wired
                ? (ws.value > 0 ? `🔒 ${shot.frames} frames · ${ws.from}` : `🔒 set by ${ws.from} — known after a run`)
                : `${framesFor(e, rate, grid)} frames at ${+rate.toFixed(3)} fps · the shot length`)
                + (shot.warning ? ` · ${shot.frames} is past 362: kept, not cut` : "");
            iSn.value = String(n);
            iSn.readOnly = wsh.wired;
            iSn.classList.toggle("lockd", wsh.wired);
            $(".sl").textContent = wsh.wired
                ? (wsh.value > 0 ? `🔒 from ${wsh.from}` : `🔒 set by ${wsh.from} — known after a run`)
                : (N ? `of ${N}` : "");
            win.classList.toggle("locked", lockLen || placed);
            if (T) {
                // Percent, never measured pixels: ComfyUI scales DOM widgets with the canvas
                // zoom, so getBoundingClientRect() is SCREEN pixels while style.left is the
                // node's own pixels — at any zoom but 100% the window drifted off the pink
                // selection it marks (screenshots, 2026-10-16).
                // Everything is placed relative to the visible part [v0, v1]; the
                // waveform clips whatever lies outside it.
                const v0 = V0(), v1 = V1(), sp = v1 - v0;
                const wEnd = Math.min(a + e, T);
                win.style.left = `${(a - v0) / sp * 100}%`;
                win.style.width = `${Math.max(0.4, (wEnd - a) / sp * 100)}%`;
                $(".tag").textContent = byShot ? `shot ${n} · ${e.toFixed(2)} s` : `${e.toFixed(2)} s`;
                const cv = wave.querySelector("canvas");
                drawWave(cv, info, v0, v1, a, a + e, { grid: sp <= 20, step: sp <= 20 ? 1 : 2 });
                if (byShot && step() > 0) drawShotStrip(cv, v0, sp, N, n, e);
                ruler();
                $(".zr").textContent = `${clock(v0)} – ${clock(v1)}`;
                $(".zx").textContent = view ? `×${(T / sp).toFixed(1)}` : "whole song";
                ovw.querySelector(".vw").style.cssText = `left:${v0 / T * 100}%;width:${sp / T * 100}%`;
                ovw.querySelector(".wn").style.cssText = `left:${Math.min(a, T) / T * 100}%;width:${Math.max(0.3, (wEnd - a) / T * 100)}%`;
                ovw.style.display = view ? "" : "none";
            }
            const last = _lastSong.get(node.id);
            let msg;
            if (byShot && step() <= 0)
                msg = `<span class="w">The overlap is as long as the shot, so the song would never move on. Choose a smaller overlap.</span>`;
            else if (byShot && T && a >= T)
                msg = `<span class="w">Shot ${n} would start at ${clock(a)}, after the song ends (${clock(T)}). This song has ${N} shot(s); the run will stop with that message.</span>`;
            else if (byShot)
                msg = `Shot <b>${n}</b>${N ? ` of ${N}` : ""} uses the song from <b>${clock(a)}</b> to <b>${clock(a + e)}</b>`
                    + (n > 1 && overlap ? `, sharing its first ${(overlap / rate).toFixed(2)} s with shot ${n - 1}` : "")
                    + `. Each shot starts ${step().toFixed(2)} s after the one before. Change the shot number${wsh.wired ? "" : " (or wire <b>shot</b>)"} and the window follows; dragging moves every shot together.`;
            else if (placed)
                msg = wss.value != null
                    ? `🔒 The start comes from the <b>song_start</b> input (${wss.from}): ${clock(a)}–${clock(a + e)}. Change the scene there and the window follows.`
                    : `🔒 The start comes from the <b>song_start</b> input (${wss.from}) — known after a run.`;
            else if (!isSong)
                msg = `A reference or guide sound keeps its own window, at most the shot length (${shot.seconds.toFixed(2)} s). Drag it or type Start and Duration; the shot length does not change.`;
            else msg = ws.wired
                ? `The duration comes from the <b>seconds</b> input, so the song always matches the shot. Move the window or type a start.`
                : `The window is the shot length. Drag it to choose the part; drag its right edge or type a Duration and the shot length follows (rounded up onto the grid).`;
            const short = T && a < T ? a + e - T : 0;
            if (short > 0.005)
                msg += `<br><span class="w">The song ends ${short.toFixed(2)} s before the window does; that part will be ${card.fill === "loop" ? "the song again from its start" : "silence"}.</span>`;
            if (last && last.name === card.file)
                msg += `<br>Last run: ${last.shot ? `shot ${last.shot}, ` : ""}${clock(last.start)}–${clock(last.start + last.seconds)}, ${last.seconds.toFixed(2)} s${last.filled > 0.0005 ? `, ${last.filled.toFixed(2)} s filled` : ""}.`;
            $(".luna-al-astat").innerHTML = msg;
        }

        // Every shot's window as a thin strip along the top of the song; the current one in pink.
        function drawShotStrip(cv, v0, sp, N, n, e) {
            if (!cv?.isConnected) return;
            const g = cv.getContext("2d"), W = cv.getBoundingClientRect().width, T = T0();
            g.font = "9px Segoe UI, system-ui, sans-serif";
            for (let k = 1; k <= N; k++) {
                const s = songStart + (k - 1) * step();
                const x0 = (s - v0) / sp * W, x1 = (Math.min(T, s + e) - v0) / sp * W, cur = k === n;
                g.fillStyle = cur ? "#e07bb4" : (k % 2 ? "#3a3d48" : "#4f5361");
                g.fillRect(x0, k % 2 ? 1 : 5, Math.max(1, x1 - x0), 3);
                if (cur || k === 1 || k === N) { g.fillStyle = cur ? "#e07bb4" : "#8b8e99"; g.fillText(String(k), x0 + 1, 17); }
            }
        }

        const ruler = () => {
            const r = $(".luna-al-ruler");
            r.replaceChildren();
            if (!T0()) return;
            const v0 = V0(), v1 = V1(), sp = v1 - v0;
            const stp = sp > 240 ? 60 : sp > 90 ? 30 : sp > 30 ? 10 : sp > 12 ? 5 : sp > 5 ? 1 : 0.5;
            for (let t = Math.ceil(v0 / stp) * stp; t <= v1 + 1e-6; t += stp) {
                const s = document.createElement("span");
                s.style.left = `${(t - v0) / sp * 100}%`;
                s.textContent = stp < 1 ? clock(t).replace(/(\.\d)\d$/, "$1") : clock(t).replace(/\.\d+$/, "");
                r.appendChild(s);
            }
        };

        loadPeaks(card.file).then((p) => {
            if (!aed.isConnected || aed.contains(win) === false) return;
            info = p;
            // The stored start is NOT clamped here: Python uses it as stored, so the
            // window is drawn where it really is and the status says if the song runs
            // out. Only a user's move is kept inside the song.
            $(".meta").textContent = `${clock(p.duration)} · ${(p.sampleRate / 1000).toFixed(1)} kHz · ${p.channels === 1 ? "mono" : p.channels === 2 ? "stereo" : p.channels + " ch"}`;
            // Keep the window in sight: if an edit (shot number, nudge) moved it out of
            // the zoomed view, the view follows it at the same zoom.
            if (view) { const a = winStart(); if (a < view.v0 || a > view.v1) setView(a - vspan() * 0.25, vspan()); }
            setTimeout(draw, 0);     // not rAF: a background tab never runs it
        }).catch(() => { $(".meta").textContent = "this file cannot be read by the browser"; });
        draw();

        // Window: the body moves it, the edges resize it (not while seconds is wired;
        // in shot mode only the right edge, which is the shot length).
        let drag = null;
        win.addEventListener("pointerdown", (e) => {
            if (!info) return;
            const edge = e.target.classList.contains("h") ? (e.target.classList.contains("l") ? "l" : "r") : "m";
            if (placed && edge !== "r") { placedNote(); return; }
            if (edge !== "m" && lockLen) { notice("The duration comes from the seconds input."); return; }
            if (edge === "l" && byShot) { notice("In shot mode the window starts where the shot number puts it."); return; }
            drag = { edge, x: e.clientX, a: winStart(), d: eff() };
            win.setPointerCapture(e.pointerId);
            e.preventDefault();
        });
        win.addEventListener("pointermove", (e) => {
            if (!drag) return;
            const T = total(), dt = (e.clientX - drag.x) / wave.getBoundingClientRect().width * vspan();
            if (drag.edge === "m") setWin(drag.a + dt);
            else if (drag.edge === "l") {
                start = Math.max(0, Math.min(drag.a + drag.d - 0.1, drag.a + dt));
                dur = drag.a + drag.d - start;
            } else dur = Math.max(1 / rate, Math.min(T - drag.a, drag.d + dt, isSong ? Infinity : shot.seconds));
            draw();
        });
        const end = () => {
            if (!drag) return;
            const resized = drag.edge !== "m";
            drag = null;
            if (resized) dur = soundLengthPatch(idx, dur, shot).duration;   // the song onto the grid; others capped
            commit();
        };
        win.addEventListener("pointerup", end);
        win.addEventListener("pointercancel", end);
        // On the song outside the window: a click centres the window there; a drag
        // (when zoomed in) scrolls the song instead.
        let pan = null;
        wave.addEventListener("pointerdown", (e) => {
            if (!info || win.contains(e.target)) return;
            pan = { x: e.clientX, v0: V0(), moved: false };
            wave.setPointerCapture(e.pointerId);
        });
        wave.addEventListener("pointermove", (e) => {
            if (!pan || !view || !wave.getBoundingClientRect().width) return;
            const dx = e.clientX - pan.x;
            if (!pan.moved && Math.abs(dx) < 4) return;
            pan.moved = true;
            setView(pan.v0 - dx / wave.getBoundingClientRect().width * vspan(), vspan());
            draw();
        });
        wave.addEventListener("pointerup", (e) => {
            if (!pan) return;
            const p = pan;
            pan = null;
            if (p.moved) return;
            if (placed) { placedNote(); return; }
            const r = wave.getBoundingClientRect();
            setWin(V0() + (e.clientX - r.left) / r.width * vspan() - eff() / 2);
            commit();
        });
        wave.addEventListener("pointercancel", () => { pan = null; });
        // Zoom: the buttons around the window's start (the cut), the wheel around the pointer.
        const zoomAt = (focus, factor) => {
            const sp = vspan(), ns = sp * factor;
            if (!Number.isFinite(focus) || !(sp > 0)) return;   // not laid out yet: nothing to zoom
            setView(focus - (focus - V0()) * (ns / sp), ns);
            draw();
        };
        aed.querySelectorAll("[data-z]").forEach((b) => {
            b.onclick = () => {
                if (!info) return;
                const a = winStart(), focus = a >= V0() && a <= V1() ? a : (V0() + V1()) / 2;
                if (b.dataset.z === "fit") { setView(0, T0()); draw(); }
                else zoomAt(focus, b.dataset.z === "in" ? 0.5 : 2);
            };
        });
        wave.addEventListener("wheel", (e) => {
            if (!info) return;
            e.preventDefault();
            const r = wave.getBoundingClientRect();
            if (!r.width) return;
            zoomAt(V0() + (e.clientX - r.left) / r.width * vspan(), e.deltaY < 0 ? 0.6 : 1 / 0.6);
        }, { passive: false });
        // The overview under a zoomed song: drag the light box to scroll, click to jump.
        let od = null;
        ovw.addEventListener("pointerdown", (e) => {
            if (!info || !view) return;
            const r = ovw.getBoundingClientRect(), t = (e.clientX - r.left) / r.width * T0();
            if (t < V0() || t > V1()) { setView(t - vspan() / 2, vspan()); draw(); }
            od = { x: e.clientX, v0: V0() };
            ovw.setPointerCapture(e.pointerId);
        });
        ovw.addEventListener("pointermove", (e) => {
            if (!od) return;
            setView(od.v0 + (e.clientX - od.x) / ovw.getBoundingClientRect().width * T0(), vspan());
            draw();
        });
        ovw.addEventListener("pointerup", () => { od = null; });
        ovw.addEventListener("pointercancel", () => { od = null; });

        aed.querySelectorAll("[data-n]").forEach((b) => {
            b.onclick = () => { if (placed) { placedNote(); return; } setWin(winStart() + nudge() * Number(b.dataset.n)); commit(); };
        });
        aed.querySelectorAll("[data-s]").forEach((b) => {
            b.onclick = () => {
                if (wsh.wired) { notice("The shot number comes from the shot input."); return; }
                commit({ shot: Math.max(1, (Number(card.shot) || 1) + Number(b.dataset.s)) });
            };
        });
        iSt.onchange = () => {
            if (placed) { draw(); return; }
            const v = parseClock(iSt.value);
            if (v != null) { if (byShot) songStart = Math.max(0, v); else start = clampWin(v); }
            commit();
        };
        iDu.onchange = () => {
            if (lockLen) return;
            const v = parseClock(iDu.value);
            if (v != null && v > 0) dur = soundLengthPatch(idx, v, shot).duration;
            commit();
        };
        iSn.onchange = () => {
            if (wsh.wired) return;
            const v = Math.round(Number(iSn.value));
            commit(Number.isFinite(v) && v >= 1 ? { shot: v } : {});
        };
        for (const inp of [iSt, iDu, iSn]) inp.addEventListener("keydown", (e) => { if (e.key === "Enter") inp.blur(); });

        $(".luna-al-play").onclick = () => {
            if (!info) return;
            if (player && !player.paused) { stopPlay(); return; }
            player = player || new Audio();
            const src = soundURL(card.file);
            if (!player.src.endsWith(src)) player.src = src;
            const a = winStart(), b = a + Math.min(eff(), total() - a);
            player.currentTime = a;
            player.play().catch(() => notice("The browser would not start playback."));
            $(".luna-al-play").textContent = "■";
            const tick = () => {
                const t = player.currentTime;
                if (t >= b || player.paused) { stopPlay(); return; }
                const inView = t >= V0() && t <= V1();
                phW.style.display = inView ? "block" : "none";
                phW.style.left = `${(t - V0()) / vspan() * 100}%`;
                playRaf = requestAnimationFrame(tick);
            };
            playRaf = requestAnimationFrame(tick);
        };
    }

    /* --------------------------------------------------------- clip editor */

    function renderVideo() {
        const vplayer = ved.querySelector("video");
        if (vplayer) { vplayer.pause(); vplayer.removeAttribute("src"); vplayer.load(); }
        const media = readMedia(node);
        if (selVideo < 0 || !media.video[selVideo]) {
            ved.style.display = "none";
            ved.replaceChildren();
            return;
        }
        ved.style.display = "";
        const idx = selVideo, card = { ...NEW_VIDEO, ...media.video[idx] };
        let start = Number(card.start) || 0, frames = Math.max(1, Number(card.frames) || 124), mi = null;
        const ww = wiredNumber("width", () => null), wh = wiredNumber("height", () => null);

        ved.innerHTML = `
            <div class="luna-al-ahead"><b>${escTag(h3Tags(node).names.v[idx])}</b><span class="nm"></span><span class="sp"></span><span class="meta">reading the clip…</span>
              <div class="luna-al-play" title="Play the clip window">▶</div></div>
            <video class="luna-al-video" playsinline preload="metadata"></video>
            <div class="luna-al-zlbl"><span>The whole clip · drag the window to choose the part</span><span class="zr"></span></div>
            <div class="luna-al-strip"><div class="luna-al-win"><div class="h l"></div><div class="h r"></div><div class="tag"></div></div></div>
            <div class="luna-al-ruler"></div>
            <div class="luna-al-nums">
              <span class="l">Start</span><div class="luna-al-nf"><span class="luna-al-nud" data-n="-1" title="One source frame earlier">◀</span><input class="st"><span class="luna-al-nud" data-n="1" title="One source frame later">▶</span></div>
              <span class="l">Length</span><div class="luna-al-nf"><span class="luna-al-nud" data-f="-1" title="Shorter">−</span><input class="fr" style="width:56px"><span class="luna-al-nud" data-f="1" title="Longer">+</span><span class="luna-al-lock fl"></span></div>
            </div>
            <div class="rolehost" style="display:flex;flex-direction:column;gap:5px"></div>
            <div class="luna-al-row"><div class="luna-al-lbl">Position</div><div class="luna-al-seg vpos"></div></div>
            <div class="luna-al-row vshotrow"><div class="luna-al-lbl">Shot</div>
              <div class="luna-al-nf"><span class="luna-al-nud" data-vs="-1" title="Previous shot">◀</span><input class="vsn" style="width:46px"><span class="luna-al-nud" data-vs="1" title="Next shot">▶</span><span class="luna-al-lock vsl"></span></div></div>
            <div class="luna-al-sect">Load settings</div>
            <div class="luna-al-row"><div class="luna-al-lbl">Frame rate</div><div class="luna-al-seg fps"></div></div>
            <div class="luna-al-row"><div class="luna-al-lbl">Frames</div><div class="luna-al-seg cnt"></div></div>
            <div class="luna-al-row"><div class="luna-al-lbl">Size</div><div class="luna-al-seg size"></div></div>
            <div class="luna-al-row fitrow"><div class="luna-al-lbl">Fit</div><div class="luna-al-seg fit"></div></div>
            <div class="luna-al-row"><div class="luna-al-lbl">Every Nth</div><div class="luna-al-seg nth"></div></div>
            <div class="luna-al-row"><div class="luna-al-lbl">Soundtrack</div><div class="luna-al-seg snd"></div></div>
            <div class="luna-al-row"><div class="luna-al-lbl">Precision</div><div class="luna-al-seg prec"></div></div>
            <div class="luna-al-astat"></div>`;
        const $ = (s) => ved.querySelector(s);
        $(".nm").textContent = card.file;
        const video = $("video"), strip = $(".luna-al-strip"), win = $(".luna-al-win");
        const iSt = $(".st"), iFr = $(".fr");
        roleRow($(".rolehost"), "v", idx, [["ref", "Reference"], ["guide", "On the shot"]],
                clipRole(card), Number(card.at) || 0, clipLen(card));
        video.src = soundURL(card.file);
        strip.style.backgroundImage = `url("${stripURL(card.file, 16)}")`;

        const T = targetOf(media), grid = () => (card.count === "target" ? T.grid : "free");
        const outFps = () => card.fps === "target" ? (T.fps || mi?.fps || H3_FPS) : card.fps === "source" ? (mi?.fps || H3_FPS) : Number(card.fps);
        const dur = () => mi?.duration || 0;
        const clipS = () => frames / outFps();
        const outSize = () => {
            if (!mi) return [0, 0];
            if (card.size === "canvas" && ww.value > 0 && wh.value > 0) return [ww.value, wh.value];
            if (card.size === "source") return [mi.width, mi.height];
            // Same rule as media_io._target_size: scale the short edge to 768, never
            // up, then even dimensions.
            const s = Math.min(1, 768 / Math.min(mi.width, mi.height));
            const ev = (v) => Math.max(2, v - (v % 2));
            return [ev(Math.round(mi.width * s)), ev(Math.round(mi.height * s))];
        };
        const clampStart = (t) => Math.max(0, Math.min(t, Math.max(0, dur() - Math.min(clipS(), dur()))));
        const snapCount = (n) => gridFrames(n, grid());

        const seg = (host, opts, cur, pickOpt) => {
            host.replaceChildren();
            for (const [val, text] of opts) {
                const o = document.createElement("div");
                o.className = "luna-al-opt" + (String(val) === String(cur) ? " on" : "");
                o.textContent = text;
                o.onclick = () => pickOpt(val);
                host.appendChild(o);
            }
        };
        const opts = () => {
            seg($(".fps"), [["target", T.fps ? `${T.label} (${T.fps})` : "Target = source"], ["source", `Source${mi?.fps ? ` (${+mi.fps.toFixed(2)})` : ""}`], [24, "24"], [25, "25"], [30, "30"]], card.fps, (v) => commit({ fps: v }));
            seg($(".cnt"), [["target", T.gridText], ["free", "Free"]], card.count, (v) => {
                card.count = v;
                commit({ count: v, frames: snapCount(frames) });
            });
            seg($(".size"), [["canvas", ww.wired && wh.wired ? "Match canvas" : "Match canvas (not wired)"], ["edge", "768 short edge"], ["source", "Source"]], card.size, (v) => commit({ size: v }));
            seg($(".fit"), [["crop", "Crop to fill"], ["pad", "Pad"], ["stretch", "Stretch"]], card.fit, (v) => commit({ fit: v }));
            seg($(".nth"), [[1, "All"], [2, "2nd"], [3, "3rd"], [4, "4th"]], card.nth, (v) => commit({ nth: v }));
            seg($(".snd"), [["keep", mi && !mi.audio ? "Keep (none in file)" : "Keep"], ["drop", "Drop"]], card.sound, (v) => commit({ sound: v }));
            seg($(".prec"), [["fp16", "fp16 (half the memory)"], ["fp32", "fp32"]], card.precision, (v) => commit({ precision: v }));
            // Like the song card: Free opens at Start; By shot number makes Start shot 1's, and shot n opens
            // (n - 1) x (shot length - overlap) later, n being Shot (+ the chain's clip when Luna-H3-Chain is wired).
            seg($(".vpos"), [["free", "Free"], ["shot", "By shot number"]], card.position, (v) => commit({ position: v }));
            $(".vshotrow").style.display = card.position === "shot" ? "" : "none";
            $(".fitrow").style.display = card.size === "canvas" ? "" : "none";
        };
        // The shot this clip card opens at, when it moves by shot number (asset_loader._clip_starts).
        const vShotNo = () => {
            const wsh = wiredNumber("shot", (last) => last?.shot > 0 ? last.shot : null);
            return Math.max(1, Math.round(wsh.wired && wsh.value > 0 ? wsh.value : Number(card.shot) || 1));
        };

        function commit(patch = {}) {
            const m = readMedia(node);
            if (!m.video[idx]) return;
            m.video[idx] = { ...m.video[idx], start: +start.toFixed(6), frames, ...patch };
            writeMedia(node, m);
            render();
        }

        function draw() {
            const fps = outFps(), T = dur(), s = clipS();
            iSt.value = clock(start);
            iFr.value = String(frames);
            $(".vsn").value = String(Number(card.shot) || 1);
            $(".fl").textContent = `${s.toFixed(2)} s at ${+fps.toFixed(3)} fps`;
            if (T) {
                win.style.left = `${Math.min(start, T) / T * 100}%`;
                win.style.width = `${Math.max(0.6, Math.min(s, Math.max(0, T - start)) / T * 100)}%`;
                $(".tag").textContent = `${frames} f · ${s.toFixed(2)} s`;
                $(".zr").textContent = `${clock(start)} – ${clock(start + s)}`;
            }
            const [w, h] = outSize();
            const kept = Math.ceil(frames / Math.max(1, card.nth));
            const gb = w && h ? kept * w * h * 3 * (card.precision === "fp16" ? 2 : 4) / 1024 ** 3 : 0;
            let msg = w ? `Will decode <b>${kept} frames at ${w}×${h}</b> = <b>${gb.toFixed(2)} GB</b> as ${card.precision}.` : "";
            const capGB = Number(settingsOf(media).memory_gb) || MEMORY_CAP_GB;
            if (gb > capGB) msg += ` <span class="w">That is more than the ${capGB} GB this node allows (⚙ Memory cap); the run will refuse it.</span>`;
            if (grid() !== "free" && Number(card.nth) > 1)
                msg += `<br><span class="w">Every Nth on the H3 grid makes H3 play the clip faster (it assumes 24 fps) and its sound no longer matches — the run refuses it. Keep every frame, or set Frames to Free.</span>`;
            if (card.size === "canvas" && !(ww.value > 0 && wh.value > 0))
                msg += `<br><span class="w">Match canvas needs <b>width</b> and <b>height</b> wired; until then the clip keeps its shape at a 768 short edge.</span>`;
            if (T && start + s > T + 1e-3)
                msg += `<br><span class="w">The clip ends ${(start + s - T).toFixed(2)} s before the window does; the run takes what is there${card.count === "target" ? ", cut down to the H3 grid" : ""}.</span>`;
            const sh = shotLen(), delivered = fps / Math.max(1, Number(card.nth) || 1);
            if (card.position === "shot") {
                const n = vShotNo(), opens = start + (n - 1) * (sh.seconds - nodeOverlap(media)[0] / sh.rate);
                $(".vsl").textContent = `opens at ${clock(opens)}`;
                msg += `<br>By shot number: shot <b>${n}</b> opens this clip at <b>${clock(opens)}</b>; each shot moves it ${(sh.seconds - nodeOverlap(media)[0] / sh.rate).toFixed(2)} s on (the shot length minus the overlap). With Luna-H3-Chain wired, the chain's clip number is added to Shot.`;
            }
            if (Math.abs(delivered - sh.rate) > 1e-3 && (mi || card.fps !== "source"))
                msg += `<br><span class="w">Delivered at ${+delivered.toFixed(3)} fps, not the shot's ${+sh.rate.toFixed(3)} fps${targetOf(media).grid !== "free" ? `: ${targetOf(media).label} plays it at ${sh.rate} fps, so it does not fit the shot` : ""}. The run says so in asset_brief.</span>`;
            if (mi) {
                const how = mi.fps && Math.abs(fps - mi.fps) > 0.01 ? (fps < mi.fps ? "drops frames" : "repeats frames") : "unchanged";
                msg += `<br>${+mi.fps.toFixed(2)} → ${+fps.toFixed(2)} fps (${how}). MiniMax H3 sees it as <b>&lt;Video ${idx + 1}&gt;</b>${card.sound === "keep" && mi.audio ? ", its sound as the &lt;Audio&gt; right before it" : ""}.`;
            }
            $(".luna-al-astat").innerHTML = msg;
        }

        const ruler = () => {
            const T = dur(), r = $(".luna-al-ruler");
            r.replaceChildren();
            if (!T) return;
            const stp = T > 120 ? 30 : T > 40 ? 10 : T > 15 ? 5 : T > 6 ? 2 : 1;
            for (let t = 0; t <= T + 1e-6; t += stp) {
                const sp = document.createElement("span");
                sp.style.left = `${t / T * 100}%`;
                sp.textContent = `${t}s`;
                r.appendChild(sp);
            }
        };

        opts();
        draw();
        mediaInfo(card.file).then((info) => {
            if (!ved.contains(win)) return;
            mi = info;
            $(".meta").textContent = `${info.width}×${info.height} · ${+(info.fps || 0).toFixed(2)} fps · ${(info.duration || 0).toFixed(2)} s · ${info.audio ? "sound" : "no sound"}`;
            opts(); ruler();
            setTimeout(draw, 0);
        }).catch(() => { $(".meta").textContent = "this clip cannot be read"; });

        let drag = null;
        win.addEventListener("pointerdown", (e) => {
            if (!mi) return;
            const edge = e.target.classList.contains("h") ? (e.target.classList.contains("l") ? "l" : "r") : "m";
            drag = { edge, x: e.clientX, s: start, f: frames };
            win.setPointerCapture(e.pointerId);
            e.preventDefault();
        });
        win.addEventListener("pointermove", (e) => {
            if (!drag) return;
            const dt = (e.clientX - drag.x) / strip.getBoundingClientRect().width * dur();
            if (drag.edge === "m") start = clampStart(drag.s + dt);
            else if (drag.edge === "r") frames = Math.max(1, Math.round(drag.f + dt * outFps()));
            else {
                const s0 = Math.max(0, Math.min(drag.s + drag.f / outFps() - 0.05, drag.s + dt));
                frames = Math.max(1, Math.round(drag.f - (s0 - drag.s) * outFps()));
                start = s0;
            }
            draw();
        });
        const end = () => {
            if (!drag) return;
            const resized = drag.edge !== "m";
            drag = null;
            if (resized) frames = snapCount(frames);
            commit();
        };
        win.addEventListener("pointerup", end);
        win.addEventListener("pointercancel", end);
        strip.addEventListener("pointerdown", (e) => {
            if (!mi || win.contains(e.target)) return;
            const r = strip.getBoundingClientRect();
            start = clampStart((e.clientX - r.left) / r.width * dur() - clipS() / 2);
            commit();
        });
        ved.querySelectorAll("[data-n]").forEach((b) => {
            b.onclick = () => { start = clampStart(start + Number(b.dataset.n) / (mi?.fps || H3_FPS)); commit(); };
        });
        ved.querySelectorAll("[data-f]").forEach((b) => {
            b.onclick = () => {
                frames = stepGrid(frames, Number(b.dataset.f), grid());
                commit();
            };
        });
        iSt.onchange = () => { const v = parseClock(iSt.value); if (v != null) start = clampStart(v); commit(); };
        iFr.onchange = () => { const v = Math.round(Number(iFr.value)); if (v >= 1) frames = snapCount(v); commit(); };
        const iVs = $(".vsn");
        iVs.onchange = () => { const v = Math.round(Number(iVs.value)); commit(Number.isFinite(v) && v >= 1 ? { shot: v } : {}); };
        ved.querySelectorAll("[data-vs]").forEach((b) => {
            b.onclick = () => commit({ shot: Math.max(1, (Number(card.shot) || 1) + Number(b.dataset.vs)) });
        });
        for (const inp of [iSt, iFr, iVs]) inp.addEventListener("keydown", (e) => { if (e.key === "Enter") inp.blur(); });

        let vt = 0;
        $(".luna-al-play").onclick = () => {
            if (!mi) return;
            if (!video.paused) { video.pause(); clearTimeout(vt); $(".luna-al-play").textContent = "▶"; return; }
            video.currentTime = start;
            video.play().catch(() => notice("The browser would not start playback."));
            $(".luna-al-play").textContent = "■";
            clearTimeout(vt);
            vt = setTimeout(() => { video.pause(); $(".luna-al-play").textContent = "▶"; }, Math.min(clipS(), dur() - start) * 1000);
        };
        video.addEventListener("loadedmetadata", () => { try { video.currentTime = start; } catch { /* not seekable yet */ } });
    }

    // Added up from its rows, like AUDIO_H; never measured. + Role, + At frame, + Position, + Shot.
    const videoHeight = () => {
        const c = readMedia(node).video[selVideo];
        return 572 + 28 + (c && clipRole(c) === "guide" ? 28 : 0) + 28 + (c?.position === "shot" ? 28 : 0);
    };
    const timelineHeight = () => {
        const m = readMedia(node);
        return readPaths(node).length || m.video.length || m.audio.length ? 222 : 0;
    };

    return { root, render, selectCard: (i) => { selected = i; renderPanel(); }, natural,
             audioOpen: () => selAudio >= 0, audioExtraRows,
             videoOpen: () => selVideo >= 0, videoHeight, timelineHeight };
}

/* ---------------------------------------------------------------- register */

app.registerExtension({
    name: "LunaMinimaxH3.AssetLoader",
    setup() { patchDeserialize(); },
    beforeConfigureGraph(graphData) {
        const moved = migrateWorkflowJSON(graphData);
        if (moved.length) console.info("[Luna-H3-AssetLoader] updated a workflow saved before the current outputs:", moved);
    },
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== NODE) return;

        const onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onCreated?.apply(this, arguments);

            const ui = buildUI(this);
            // Size the card area to its contents, the same way this pack's video
            // preview widget does. Without it the two multiline text fields take
            // the node's height and the cards get a single visible row.
            // Grid: work out how many rows the cards will wrap into at the node's
            // current width, so the node grows by row rather than by image.
            const CELL = 120, BAR = 34, PAD = 16, MINH = 150, MAXH = 720;
            // The song editor's height, added up from its rows (head 20, song 64,
            // ruler 12, zoom label 14, zoom 40, two number rows 52, two option
            // rows 50, status 42, gaps and padding 66). Arithmetic, never measured.
            const AUDIO_H = 360;
            // Three rows (heading + cards, or a one-line note when empty), the one open
            // editor, and the shot timeline. Arithmetic on node.size, never measured
            // from the rendered layout. The part heights below were measured ONCE in
            // frontend 1.55.17 (2026-10-16) and are constants from then on.
            const GRID_SIDE = 35;          // DOM widget margins + root padding, horizontally
            const CARD_MIN = 112, CARD_GAP = 8;
            const ROOT = 12 + 27 + 22 + 5 * 6;  // root padding, toolbar, the Overlap row, gaps between bar/rows/timeline
            const SEC = 7 + 21 + 6;        // row border + padding, heading, gap to the cards
            // AUDIO_BASE: song editor without its Position/Role/At-frame rows, measured 276 after the zoom
            // tools replaced the strip (2026-10-16), + 16 kept for the overview bar shown when zoomed.
            const EMPTY = 26, PANEL = 125, AUDIO_BASE = 292, VIDEO_BASE = 598, TIMELINE = 194;
            const gridH = (n) => {
                if (!n) return EMPTY;
                const gw = Math.max(CARD_MIN, this.size[0] - GRID_SIDE);
                const cols = Math.max(1, Math.floor((gw + CARD_GAP) / (CARD_MIN + CARD_GAP)));
                const card = (gw - (cols - 1) * CARD_GAP) / cols + 2;   // square cards + border
                const rows = Math.ceil(n / cols);
                return rows * card + (rows - 1) * CARD_GAP + 4;
            };
            const cardsHeight = () => {
                const list = readPaths(this);
                const md = readMedia(this);
                const rowsH = 3 * SEC + gridH(list.length) + gridH(md.video.length)
                            + gridH(md.audio.length);
                const tlh = ui.timelineHeight() ? TIMELINE : 0;
                let editor = 0;
                if (ui.videoOpen()) editor = 6 + VIDEO_BASE + (ui.videoHeight() - 572);   // + Role, + At frame
                else if (ui.audioOpen()) editor = 6 + AUDIO_BASE + ui.audioExtraRows() * 28;   // Position, Shot + Overlap, Role, At frame
                else if (list.length) {
                    // The picture preview and its resize panel, which grows a row when a
                    // mode or a ratio is chosen, and with the At frame row of a keyframe.
                    const st = itemAt(this, 0);
                    const key = md.pictures.some((p) => p?.role === "key");
                    editor = 6 + previewH(this) + 6 + PANEL + (st.mode !== "off" ? 26 : 0) + (st.ratio ? 26 : 0) + (key ? 28 : 0);
                }
                // + 16: the calibration came within ±16 px; spare space beats clipping.
                return Math.max(MINH, ROOT + rowsH + editor + tlh + 16);
            };            const w = this.addDOMWidget("luna_asset_cards", "div", ui.root, {
                serialize: false,
                hideOnZoom: false,
                getHeight: cardsHeight,
            });
            w.computeSize = (width) => [width, cardsHeight()];
            keepNodeWidth(w);   // the side panel must not shrink the cards (luna_widget_width.mjs)
            this._lunaRender = () => { ui.render(); this.setDirtyCanvas(true, true); };
            this._lunaRoot = ui.root;

            // There is no "Update outputs" button any more, deliberately. Every
            // route into the node now applies itself — the arrows, upload, remove,
            // Clear, and typing into image_paths (hooked below). A button whose
            // work is always already done cannot be told apart from a broken one,
            // and it read as broken. Do not add it back to "be safe"; add the
            // missing automatic path instead.
            const cw = widget(this, COUNT_WIDGET);
            if (cw) {
                const orig = cw.callback;
                const self = this;
                cw.callback = function (value, canvas) {
                    const out = orig ? orig.apply(this, arguments) : undefined;
                    // Apply on EVERY change, interactive or not. This used to fire
                    // only on workflow reload, which left the count and the sockets
                    // disagreeing until you found the button. Changing the number
                    // IS the action.
                    applyCount(self);   // the labels only: every socket always exists
                    return out;
                };
            }

            // Typing into image_paths was the last route the cards did not own —
            // it was the button's one real job. Now the field syncs the count and
            // rebuilds the cards itself. `internalWrite` keeps the cards' own
            // writes from coming back through here and rendering twice.
            const pw = widget(this, PATHS_WIDGET);
            if (pw) {
                const origPaths = pw.callback;
                const self = this;
                pw.callback = function (value) {
                    const out = origPaths ? origPaths.apply(this, arguments) : undefined;
                    if (!internalWrite) {
                        syncCount(self, readPaths(self).length);
                        self._lunaRender?.();
                    }
                    return out;
                };
            }

            if (this.size[1] < 320) this.size[1] = 320;
            requestAnimationFrame(() => { applyCount(this); ui.render(); });
            return r;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            const r = onConfigure?.apply(this, arguments);
            const savedFixed = savedFixedCount(info?.outputs);
            if (savedFixed >= 0) {
                setTimeout(() => {
                    const n = migrateLiveNode(this, savedFixed);
                    if (n) console.info(`[Luna-H3-AssetLoader] moved ${n} wire(s) of a pasted older node up by ${FIXED - savedFixed} slots.`);
                }, 0);
            }
            // A saved node keeps its saved input list, so the removed `previous_frames` socket
            // (the Continuation card, gone 2026-10-29) would still show. Drop it, wire and all.
            const stale = this.inputs?.findIndex((i) => i.name === "previous_frames") ?? -1;
            if (stale >= 0) this.removeInput(stale);
            requestAnimationFrame(() => { applyCount(this); this._lunaRender?.(); });
            return r;
        };

        // The run reports what the song window really delivered (and whether the
        // file ran out); the editor shows it as "Last run".
        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            const r = onExecuted?.apply(this, arguments);
            const s = message?.luna_asset_audio?.[0];
            if (s) _lastSong.set(this.id, s);
            this._lunaRender?.();
            return r;
        };

        // Wiring or unwiring `seconds` / `shot` locks or frees the song's duration
        // and shot number at once.
        const onConn = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function (type, slot, connected, link, ioSlot) {
            const r = onConn?.apply(this, arguments);
            // setTimeout, not requestAnimationFrame: a background tab never runs the
            // latter, and the lock must be right whenever the node is next seen.
            if (WIRED_INPUTS.includes(ioSlot?.name)) setTimeout(() => this._lunaRender?.(), 0);
            return r;
        };

        // Right-click > Colors fires no event, so pick the change up on the next
        // draw. syncAccent early-outs unless the colour actually differs.
        const onDraw = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function (ctx) {
            const r = onDraw?.apply(this, arguments);
            if (this._lunaRoot) syncAccent(this, this._lunaRoot);
            const sig = wiredSignature(this);
            if (sig !== this._lunaWiredSig) {
                const first = this._lunaWiredSig === undefined;
                this._lunaWiredSig = sig;
                if (!first) setTimeout(() => this._lunaRender?.(), 0);
            }
            if (!this.flags?.collapsed && ctx) drawCtl(this, ctx);
            return r;
        };

        // The stepper's click. Returning true consumes the event so LiteGraph does
        // not start dragging the node out from under a button press.
        const onDown = nodeType.prototype.onMouseDown;
        nodeType.prototype.onMouseDown = function (e, pos) {
            if (!this.flags?.collapsed) {
                const part = ctlHit(this, pos);
                if (part) {
                    const cur = Math.max(1, Math.min(MAX, Number(widget(this, COUNT_WIDGET)?.value) || 1));
                    if (part === "minus") setCount(this, cur - 1);
                    else if (part === "plus") setCount(this, cur + 1);
                    else promptCount(this, cur, e);   // click the number to type one
                    return true;
                }
            }
            return onDown?.apply(this, arguments);
        };

        // Swallow double-clicks that land on the stepper. This is the bug that
        // moved the control off the title bar: two quick presses on + there opened
        // LiteGraph's rename editor. Down here the title cannot be hit at all, and
        // this stops any other double-click behaviour the frontend may attach to
        // the node body from firing between two increments.
        const onDbl = nodeType.prototype.onDblClick;
        nodeType.prototype.onDblClick = function (e, pos) {
            if (!this.flags?.collapsed && ctlHit(this, pos)) return true;
            return onDbl?.apply(this, arguments);
        };

        const onMove = nodeType.prototype.onMouseMove;
        nodeType.prototype.onMouseMove = function (e, pos) {
            const part = this.flags?.collapsed ? "" : ctlHit(this, pos);
            if (part !== (this._lunaCtlHover || "")) {
                this._lunaCtlHover = part;
                this.setDirtyCanvas(true, false);
            }
            return onMove?.apply(this, arguments);
        };
    },
});
