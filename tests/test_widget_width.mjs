// keepNodeWidth: a width written by ComfyUI's side panel (frontend #12443) must not reach the canvas, which sizes a DOM
// widget from widget.width before node.width.   Run: node --test tests\test_widget_width.mjs
import { test } from "node:test";
import assert from "node:assert/strict";
import { keepNodeWidth } from "../web/luna_widget_width.mjs";

// The canvas rule from ComfyUI's frontend (DomWidgets: `(n.width ?? i.width) - t*2`).
const domWidth = (widget, node) => (widget.width ?? node.width) - widget.margin * 2;

test("a panel width written into the widget is dropped", () => {
    const node = { width: 1298 };
    const widget = keepNodeWidth({ margin: 10 });
    widget.width = 203;                       // what the side panel's legacy renderer does
    assert.equal(widget.width, undefined);
    assert.equal(domWidth(widget, node), 1278);
});

test("an older width already on the widget is cleared too", () => {
    const widget = keepNodeWidth({ margin: 10, width: 203 });
    assert.equal(domWidth(widget, { width: 900 }), 880);
});

test("no widget, no error", () => {
    assert.equal(keepNodeWidth(undefined), undefined);
});
