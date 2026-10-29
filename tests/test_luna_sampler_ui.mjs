// Luna-H3-Sampler's frontend: the readout file and its place in the pack's help and collapse lists.
// The kit files' identity with Luna-Media-Tools is checked in test_registration.py.
//     node --test tests/test_luna_sampler_ui.mjs
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");

function code(name) {
    return readFileSync(join(root, "web", name), "utf8").split(/\r?\n/)
        .map((line) => line.replace(/\/\/.*$/, "")).join("\n");
}

test("pack theme lists Luna-H3-Sampler for help and collapse, tint stays off", () => {
    const src = code("luna_pack_theme.js");
    const calls = [...src.matchAll(/registerLuna(?:Help|Collapse)\(\s*app\s*,\s*\[([^\]]*)\]/g)];
    assert.equal(calls.length, 2);
    for (const call of calls) {
        assert.ok(call[1].includes('"Luna-H3-Sampler"'));
    }
    assert.ok(!src.includes("registerLunaTheme("));
});

test("readout file registers no second help, collapse or tint", () => {
    const src = code("luna_sampler.js");
    assert.ok(!/registerLuna(?:Help|Collapse|Theme)\(/.test(src));
    assert.match(src, /fetch\("\/luna_sampler\/recipes"\)/);
    assert.match(
        src,
        /pass 1: steps 1-\$\{transition\} \(low\) \/ pass 2: steps \$\{transition \+ 1\}-\$\{total\} \(high\)/,
    );
    assert.match(src, /recipe\.strength/);
    assert.match(src, /recipe\.total_steps/);
    assert.match(src, /recipe\.shift/);
    assert.match(src, /recipe\.audio_shift/);
});
