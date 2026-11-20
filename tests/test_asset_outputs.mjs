// Luna-H3-AssetLoader's output migration (web/luna_asset_loader.js, the "output migration" block): every node ends
// with the ten fixed outputs, image_1..image_9 and then fps (19) and vocals (20), the Python order
// (asset_loader.RETURN_NAMES), and every saved link keeps the output it was wired to (Luna-MinimaxH3 SPEC § 2).
//     node --test tests/test_asset_outputs.mjs
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");

// The block is pure (no app, no DOM), so it runs here as it is.
function block() {
    const src = readFileSync(join(root, "web", "luna_asset_loader.js"), "utf8");
    const m = src.match(/\/\/ --- output migration: begin[^\n]*\n([\s\S]*?)\/\/ --- output migration: end ---/);
    assert.ok(m, "output migration block not found");
    return new Function(`${m[1]}\nreturn { migrateWorkflowJSON, ALL_OUTS, FIXED_OUTS };`)();
}
const { migrateWorkflowJSON, ALL_OUTS, FIXED_OUTS } = block();

const NAMES = ALL_OUTS.map(([name]) => name);
const FIXED_NAMES = FIXED_OUTS.map(([name]) => name);
const images = (n) => Array.from({ length: n }, (_, i) => `image_${i + 1}`);
const out = (name) => ({ name, localized_name: name, type: "X", links: [] });

// A workflow with one Asset Loader (id 7) whose outputs are `names`, a link from each slot in `wired` to node 99,
// and the same node again inside a subgraph definition (links as objects there).
function workflow(names, wired) {
    const links = wired.map((slot, k) => [100 + k, 7, slot, 99, k, "X"]);
    const node = () => ({ id: 7, type: "Luna-H3-AssetLoader", outputs: names.map(out) });
    return {
        nodes: [node(), { id: 99, type: "Other" }],
        links,
        definitions: { subgraphs: [{ id: "sg", nodes: [node()],
            links: wired.map((slot, k) => ({ id: 200 + k, origin_id: 7, origin_slot: slot, target_id: 99, target_slot: k })) }] },
    };
}

// For every link from the Asset Loader: [link id, the output name it leaves from].
function wiredNames(g) {
    const node = g.nodes.find((n) => n.type === "Luna-H3-AssetLoader");
    return g.links.map((l) => (Array.isArray(l) ? [l[0], node.outputs[l[2]]?.name, l[2]]
                                                 : [l.id, node.outputs[l.origin_slot]?.name, l.origin_slot]));
}

function check(names, wired, expectSlots) {
    const wf = workflow(names, wired);
    const before = [wiredNames(wf), wiredNames(wf.definitions.subgraphs[0])];
    migrateWorkflowJSON(wf);
    for (const [g, was] of [[wf, before[0]], [wf.definitions.subgraphs[0], before[1]]]) {
        const node = g.nodes.find((n) => n.type === "Luna-H3-AssetLoader");
        assert.deepEqual(node.outputs.map((o) => o.name), NAMES);          // every output, in the Python order
        assert.equal(node.outputs[19].name, "fps");
        assert.equal(node.outputs[20].name, "vocals");
        const now = wiredNames(g);
        assert.deepEqual(now.map(([id, name]) => [id, name]), was.map(([id, name]) => [id, name]));   // names kept
        assert.deepEqual(now.map(([, , slot]) => slot), expectSlots);
    }
    // A second run changes nothing.
    const again = JSON.stringify(wf);
    assert.deepEqual(migrateWorkflowJSON(wf), []);
    assert.equal(JSON.stringify(wf), again);
}

test("the order is the Python order: 10 fixed, image_1..9, fps, vocals", () => {
    assert.deepEqual(NAMES, [...FIXED_NAMES, ...images(9), "fps", "vocals"]);
    assert.equal(NAMES.length, 21);
});

test("a save with image_1..3 linked keeps every slot and gains image_4..9, fps, vocals", () => {
    check([...FIXED_NAMES, ...images(3)], [5, 10, 12], [5, 10, 12]);
});

test("a save with all nine linked keeps every slot and gains fps, vocals", () => {
    check([...FIXED_NAMES, ...images(9)], [3, 10, 18], [3, 10, 18]);
});

test("an old 4-fixed save: the fixed outputs go in before image_1, the image links move with their names", () => {
    check(["images", "width", "height", "asset_brief", ...images(2)], [3, 4, 5], [3, 10, 11]);
});

test("an old 10-fixed save (before fps and vocals) only gains the missing tail", () => {
    check([...FIXED_NAMES, ...images(2)], [8, 9, 11], [8, 9, 11]);
});

test("v3 (7 fixed) and v4 (9 fixed) saves still migrate first", () => {
    check([...FIXED_NAMES.slice(0, 7), ...images(4)], [6, 7, 10], [6, 10, 13]);
    check([...FIXED_NAMES.slice(0, 9), ...images(1)], [8, 9], [8, 10]);
});

test("a node already in the new order is left alone", () => {
    const wf = workflow(NAMES, [19, 20, 18]);
    const was = JSON.stringify(wf);
    assert.deepEqual(migrateWorkflowJSON(wf), []);
    assert.equal(JSON.stringify(wf), was);
});

test("outputs in an unknown order are left as saved and reported", () => {
    const wf = workflow(["images", "something_else"], [1]);
    const moved = migrateWorkflowJSON(wf);
    assert.ok(moved.some((m) => m.includes("not in the known order")));
    assert.deepEqual(wf.nodes[0].outputs.map((o) => o.name), ["images", "something_else"]);
});
