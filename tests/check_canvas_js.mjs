// Checks that web/luna_h3_canvas.js lists the same preset sizes as h3_canvas.py.
// Runs the JS maths outside the browser (imports stubbed) and compares it with the
// lists Python writes to stdout.
//
//   E:\ComfyUI-Easy-Install\python_embeded\python.exe tests\dump_canvas_presets.py > presets.json
//   node tests\check_canvas_js.mjs presets.json
import { readFileSync } from "node:fs";

const here = new URL("../web/luna_h3_canvas.js", import.meta.url);
const source = readFileSync(here, "utf8")
    .replace(/^import .*$/gm, "")
    .replace("app.registerExtension(", "(() => {})(");
const LUNA = {};
const api = new Function("LUNA", `${source}\nreturn { presetLabels, ASPECT_RATIOS };`)(LUNA);

const python = JSON.parse(readFileSync(process.argv[2], "utf8").replace(/^\uFEFF/, ""));
let bad = 0;
let count = 0;
for (const [key, labels] of Object.entries(python)) {
    const [ratio, multiple] = key.split("|");
    const js = api.presetLabels(ratio, multiple);
    count += labels.length;
    if (JSON.stringify(js) !== JSON.stringify(labels)) {
        bad += 1;
        console.log(`MISMATCH ${key}\n  py ${labels.join(", ")}\n  js ${js.join(", ")}`);
    }
}
console.log(bad ? `${bad} list(s) differ` : `all ${Object.keys(python).length} lists match (${count} labels)`);
process.exit(bad ? 1 : 0);
