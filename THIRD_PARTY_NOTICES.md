# Third-party notices

Luna-MinimaxH3 is licensed under GPL-3.0 (see `LICENSE`). This file lists code adapted from other projects,
with its source, licence and where it lives here.

| Source | Licence | Adapted into | What |
|---|---|---|---|
| ComfyUI-MiniMaxH3-Context-Loop (ethanfel), `chain_nodes.py` MiniMaxH3ChainLoopEnd; its recursion adapted from Ethanfel's SxCP loop nodes | GPL-3.0, Copyright (C) 2026 NikoDemon80 | `h3_loop.py`: `_explore_dependencies`, `_collect_contained`, `find_body` / `expand_body` (from `_recurse`), `release_between_scenes` (shape of `_release_loop_boundary_resources`) | Loop body detection, GraphBuilder clone for the next scene, memory release between scenes |
| ComfyUI core (comfy_extras/nodes_minimax_h3.py: MiniMaxH3AddGuide, MiniMaxH3ImageToVideo, MiniMaxH3ReferenceToVideo) | GPL-3.0 | `h3_loop_guide.py`: `make_keyframe`, `valid_clip_length`, `_resize`, `vlm_items` / `insert_guide_items` | The keyframe-row construction of Add Guide, the FL2VA keyframe-image pairing, and core's 2 fps video item layout for the VLM |
| ComfyUI-Pixaroma (c) 2026 pixaroma, `LoadImageMini` state and resize helpers (`nodes/_resize_helpers.py`), `node_panel.mjs`, and Load Audio / Load Video (frame-rate handling, audio silence/loop choice) | MIT | Luna-H3-AssetLoader, Luna-H3-ReferenceLoader and their media code: the per-image state schema, the resize model, the settings panel idea, the audio loading rules; the rest is this pack's own | `asset_images.py`, `asset_loader.py`, `media_io.py`, `web/luna_asset_loader.js` |
| LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler, `nodes/minimax_h3_latent_upscaler_3d.py` (retrieved 2026-10-28) | MIT, Copyright (c) 2026 LBH-123-AI | `luna_upscaler.py` (Luna-H3-Sampler): `AttnBlock3D`, `ResBlockEmb3D`, `TemporalConv`, `LatentResizer3D`, normalization helpers, H3 latent statistics | The H3 latent upscaler network; full notice below |

## Licence of the shared frontend kit

`web/luna_theme.mjs`, `web/luna_help.mjs` and `web/luna_collapse.mjs` (and `web/luna_pack_theme.js`,
which uses them) are Apache-2.0, Copyright 2026 Luna AI Space. They stay Apache-2.0 inside this
GPL-3.0 pack; Apache-2.0 code may be combined into a GPL-3.0 work. Copy them into another pack under
Apache-2.0.

## What each licence asks

- GPL-3.0 (Context Loop, ComfyUI core): the combined work is GPL-3.0 and its source stays available.
- MIT (Pixaroma): keep the copyright notice and permission notice. The headers of `asset_images.py`,
  `asset_loader.py` and `media_io.py` say what was taken and what is this pack's own.
- MIT (LBH-123-AI): keep the copyright notice and permission notice, given in full below.
- No DaSiWa code is in this pack.

## MiniMax H3 latent upscaler (LBH-123-AI, MIT)

The network classes (`AttnBlock3D`, `ResBlockEmb3D`, `TemporalConv`, `LatentResizer3D`),
normalization helpers and H3 latent statistics in `luna_upscaler.py` are adapted from
[LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler),
[`nodes/minimax_h3_latent_upscaler_3d.py`](https://raw.githubusercontent.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler/main/nodes/minimax_h3_latent_upscaler_3d.py),
retrieved 2026-10-28. [Upstream licence](https://raw.githubusercontent.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler/main/LICENSE).

Adaptations: preserve temporal length in scale-only calls; ComfyUI folder discovery,
safe CPU checkpoint loading and architecture inference; cache on CPU; exact target size,
original dtype/device restoration, and unconditional offload/cleanup after inference.
The 2D source is not used. Model weights are not bundled or relicensed by this notice.
The upstream LICENSE has a literal `/n/n` between paragraphs; it is rendered below as a blank line.

MIT License

Copyright (c) 2026 LBH-123-AI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
