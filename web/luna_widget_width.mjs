// Keep a DOM widget as wide as its node. Copy of Luna-Media-Tools web/luna_widget_width.mjs (no cross-pack import in
// ComfyUI's frontend); change both together.
//
// ComfyUI_frontend issue #12443: when the right side panel (Parameters / Nodes) draws a node's widgets, its legacy
// widget renderer writes the panel's width (about 200 px) into widget.width. The canvas then sizes a DOM widget from
// widget.width before node.width, so a player or editor shrinks to the panel's width on a wide node and stays there.
// Seen 2026-11-09 on Luna-SaveVideo: node 1298 wide, player 183 wide, widget.width 203.
//
// A DOM widget of ours never needs a width of its own, so widget.width reads as unset and writes are dropped; the canvas
// falls back to the node's width. Same idea as pekkAi-dev/ComfyUI-LegacyWidgetWidthFix, scoped to our widgets only.

export function keepNodeWidth(widget) {
    if (!widget) return widget;
    Object.defineProperty(widget, "width", {
        configurable: true,
        enumerable: true,
        get() {
            return undefined;
        },
        set() {},
    });
    return widget;
}
