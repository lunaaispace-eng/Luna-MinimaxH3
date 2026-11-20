// Luna-H3-AssetLoader's frame rule in JS (web/luna_asset_loader.js, the "frame rule" block) gives the same counts as
// luna_units.py on one table of inputs, and reads an old media_state's overlap the way asset_loader.py does.
//     node --test tests/test_asset_frames.mjs
// The Python side runs ComfyUI's Python (env LUNA_PYTHON, default E:\ComfyUI-Easy-Install\python_embeded\python.exe);
// without it that test is skipped.
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const PY = process.env.LUNA_PYTHON || "E:\\ComfyUI-Easy-Install\\python_embeded\\python.exe";

// The block is pure (no app, no DOM), so it runs here as it is.
function frameRule() {
    const src = readFileSync(join(root, "web", "luna_asset_loader.js"), "utf8");
    const m = src.match(/\/\/ --- frame rule: begin[^\n]*\n([\s\S]*?)\/\/ --- frame rule: end ---/);
    assert.ok(m, "frame rule block not found");
    return new Function(`${m[1]}\nreturn { framesUp, framesDown, framesFor, framesWarning, snapOverlap, nodeOverlap,
        gridFor, fpsFor, OVERLAPS, OVERLAP_ROW, parseMedia, mediaJSON, soundWindow, soundLengthPatch };`)();
}
const R = frameRule();

const SECONDS = [0.9375, 5.0, 5.17, 16, 124 / 24];
const FRAMES = [120, 124, 125, 380];
const TARGETS = ["h3", "ltx"];

function jsTable() {
    const rows = [];
    for (const t of TARGETS) {
        const grid = R.gridFor(t), fps = R.fpsFor(t, null);
        for (const s of SECONDS) rows.push(["seconds", t, s, R.framesFor(s, fps, grid), R.framesWarning(R.framesFor(s, fps, grid), grid)]);
        for (const f of FRAMES) rows.push(["frames", t, f, R.framesUp(f, grid), R.framesWarning(R.framesUp(f, grid), grid)]);
    }
    return rows;
}

const PY_TABLE = `
import json, sys
sys.path.insert(0, sys.argv[1])
import luna_units as u
rows = []
for t in ("h3", "ltx"):
    grid, fps = u.grid_for(t), u.fps_for(t, None)
    for s in (0.9375, 5.0, 5.17, 16, 124 / 24):
        f = u.frames_for(s, fps, grid)
        rows.append(["seconds", t, s, f, u.frames_warning(f, grid)])
    for n in (120, 124, 125, 380):
        f = u.frames_up(n, grid)
        rows.append(["frames", t, n, f, u.frames_warning(f, grid)])
print(json.dumps(rows))
`;

test("the JS frame rule and luna_units.py give the same counts and warnings", { skip: !existsSync(PY) && `no Python at ${PY}` }, () => {
    const r = spawnSync(PY, ["-I", "-c", PY_TABLE, root], { encoding: "utf8" });
    assert.equal(r.status, 0, r.stderr);
    assert.deepEqual(jsTable(), JSON.parse(r.stdout));
});

test("the table's expected counts", () => {
    const got = Object.fromEntries(jsTable().map(([k, t, v, f]) => [`${k} ${t} ${v}`, f]));
    assert.equal(got["seconds h3 5"], 124);
    assert.equal(got["seconds h3 16"], 396);
    assert.equal(got["seconds h3 0.9375"], 39);           // 22.5 frames, half up to 23, up onto 39
    assert.equal(got[`seconds h3 ${124 / 24}`], 124);
    assert.equal(got["seconds ltx 5"], 129);
    assert.equal(got["frames h3 380"], 396);
    assert.equal(got["frames h3 125"], 141);
    assert.equal(got["frames ltx 120"], 121);
});

test("an old save with shot frames 380 shows 396 and the warning", () => {
    assert.equal(R.framesUp(380, "h3"), 396);
    assert.match(R.framesWarning(396, "h3"), /396 frames is past the 362 frames H3 was trained on/);
    assert.equal(R.framesWarning(362, "h3"), "");
    assert.equal(R.framesWarning(500, "ltx"), "");
});

test("the overlap: an old media_state takes sound card 1's, the setting wins, off the list snaps", () => {
    const old = { audio: [{ file: "s.wav", overlap: 39 }], settings: {} };
    assert.deepEqual(R.nodeOverlap(old), [39, ""]);
    assert.deepEqual(R.nodeOverlap({ audio: [{ file: "s.wav", overlap: 39 }], settings: { overlap: 141 } }), [141, ""]);
    assert.deepEqual(R.nodeOverlap({ audio: [], settings: {} }), [22, ""]);
    const [v, warn] = R.nodeOverlap({ audio: [], settings: { overlap: 50 } });
    assert.equal(v, 39);
    assert.match(warn, /^Overlap 50 is not on the list/);
    assert.deepEqual(R.OVERLAP_ROW, [22, 39, 90, 141, 192]);
});

test("the overlap list and the snap match luna_units.py", { skip: !existsSync(PY) && `no Python at ${PY}` }, () => {
    const code = `import json, sys\nsys.path.insert(0, sys.argv[1])\nimport luna_units as u\n`
        + `print(json.dumps([list(u.OVERLAPS)] + [u.snap_overlap(v)[0] for v in (0, 10, 11, 12, 30, 31, 50, 64.5, 65, 300)]))`;
    const r = spawnSync(PY, ["-I", "-c", code, root], { encoding: "utf8" });
    assert.equal(r.status, 0, r.stderr);
    const [list, ...snaps] = JSON.parse(r.stdout);
    assert.deepEqual(R.OVERLAPS, list);
    assert.deepEqual([0, 10, 11, 12, 30, 31, 50, 64.5, 65, 300].map((v) => R.snapOverlap(v)[0]), snaps);
});

// A 124-frame H3 shot, as shotLen() gives it.
const SHOT = { frames: 124, seconds: 124 / 24, rate: 24, grid: "h3" };

test("a reference sound keeps its own window, capped by the shot (asset_loader._sounds)", () => {
    assert.equal(R.soundWindow({ file: "r.wav", duration: 2 }, 1, SHOT), 2);
    assert.equal(R.soundWindow({ file: "r.wav", duration: 0 }, 1, SHOT), SHOT.seconds);    // 0 = the shot
    assert.equal(R.soundWindow({ file: "r.wav", duration: 30 }, 2, SHOT), SHOT.seconds);   // capped
    assert.equal(R.soundWindow({ file: "s.wav", duration: 2 }, 0, SHOT), SHOT.seconds);    // the song: the shot
});

test("only the song's length sets the node's shot length", () => {
    assert.deepEqual(R.soundLengthPatch(1, 2, SHOT), { shotFrames: null, duration: 2 });
    assert.deepEqual(R.soundLengthPatch(2, 30, SHOT), { shotFrames: null, duration: SHOT.seconds });
    assert.deepEqual(R.soundLengthPatch(0, 10, SHOT), { shotFrames: 243, duration: 243 / 24 });   // 240 up onto 17k+5
});

// The text the JS writes after the song is removed from an old save: asset_loader.py reads the same literal
// (test_loader_connect.py OverlapTests.test_the_migrated_overlap_outlives_the_song).
const OLD_SAVE = JSON.stringify({ audio: [{ file: "song.wav", overlap: 39 }], settings: {} });
const AFTER_REMOVAL = JSON.stringify({ settings: { overlap: 39 } });

test("an old save's overlap is migrated into settings.overlap on read and outlives the song", () => {
    const m = R.parseMedia(OLD_SAVE);
    assert.equal(m.settings.overlap, 39);
    m.audio.splice(0, 1);                                    // the song card's remove button
    const text = R.mediaJSON(m);
    assert.equal(text, AFTER_REMOVAL);
    assert.deepEqual(R.nodeOverlap(R.parseMedia(text)), [39, ""]);
    // A save that already has the setting is left as it is; one with no song has nothing to migrate.
    assert.equal(R.parseMedia(JSON.stringify({ audio: [{ file: "s.wav", overlap: 39 }], settings: { overlap: 90 } })).settings.overlap, 90);
    assert.equal(R.parseMedia(JSON.stringify({ settings: {} })).settings.overlap, undefined);
    assert.equal(R.parseMedia(JSON.stringify({ audio: [{ file: "s.wav" }] })).settings.overlap, 22);
});
