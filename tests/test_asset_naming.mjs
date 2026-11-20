// Luna-H3-AssetLoader's card naming rule in JS (web/luna_asset_loader.js, the "naming rule" block, cardTags)
// gives the same tags as asset_timeline.h3_tags (tests/test_asset_timeline.py TagTests and NamingTests use the
// same cases), and each card's @ name label.
//     node --test tests/test_asset_naming.mjs
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");

// The block is pure (no app, no DOM), so it runs here as it is.
function naming() {
    const src = readFileSync(join(root, "web", "luna_asset_loader.js"), "utf8");
    const m = src.match(/\/\/ --- naming rule: begin[^\n]*\n([\s\S]*?)\/\/ --- naming rule: end ---/);
    assert.ok(m, "naming rule block not found");
    return new Function(`${m[1]}\nreturn { cardTags };`)();
}

const { cardTags } = naming();
const tagsOnly = (t) => ({ pictures: t.pictures, videos: t.videos, videoAudio: t.videoAudio, sounds: t.sounds });

test("acceptance A: keyframe first, a referenced clip with its sound, the song locked, a reference sound", () => {
    const t = cardTags(["key", "ref", "ref"], [{ role: "ref", sound: true }], ["lock", "ref"]);
    assert.deepEqual(tagsOnly(t), {
        pictures: { 1: "<Picture 1>", 2: "<Picture 2>" },
        videos: { 0: "<Video 1>" },
        videoAudio: { 0: "<Audio 1>" },
        sounds: { 1: "<Audio 2>" },
    });
    assert.deepEqual(t.names, {
        p: ["@image1 · keyframe", "@image2 · <Picture 1>", "@image3 · <Picture 2>"],
        v: ["@video1 · <Video 1>"],
        a: ["@audio1 · song lock", "@audio2 · <Audio 2>"],
    });
});

// The three cases of test_asset_timeline.py TagTests, with h3_tags' results.
test("only references are numbered, in card order", () => {
    const t = cardTags(["key", "ref", "key", "ref"], [], []);
    assert.deepEqual(t.pictures, { 1: "<Picture 1>", 3: "<Picture 2>" });
});

test("clip soundtracks come before the standalone sounds", () => {
    const t = cardTags([], [{ role: "ref", sound: true }, { role: "ref", sound: false }, { role: "guide", sound: true }],
        ["lock", "ref", "ref"]);
    assert.deepEqual(t.videos, { 0: "<Video 1>", 1: "<Video 2>" });
    assert.deepEqual(t.videoAudio, { 0: "<Audio 1>" });
    assert.deepEqual(t.sounds, { 1: "<Audio 2>", 2: "<Audio 3>" });
    assert.deepEqual(t.names.v, ["@video1 · <Video 1>", "@video2 · <Video 2>", "@video3 · guide clip"]);
    assert.deepEqual(t.names.a, ["@audio1 · song lock", "@audio2 · <Audio 2>", "@audio3 · <Audio 3>"]);
});

test("nothing referenced gives empty maps and on-shot labels", () => {
    const t = cardTags(["key"], [{ role: "guide", sound: true }], ["guide", "guide"]);
    assert.deepEqual(tagsOnly(t), { pictures: {}, videos: {}, videoAudio: {}, sounds: {} });
    assert.deepEqual(t.names, { p: ["@image1 · keyframe"], v: ["@video1 · guide clip"],
        a: ["@audio1 · song guide", "@audio2 · guide"] });
});

test("the song as a reference sound takes its <Audio n> like any other", () => {
    const t = cardTags(["ref"], [], ["ref", "ref"]);
    assert.deepEqual(t.sounds, { 0: "<Audio 1>", 1: "<Audio 2>" });
    assert.deepEqual(t.names.a, ["@audio1 · <Audio 1>", "@audio2 · <Audio 2>"]);
});
