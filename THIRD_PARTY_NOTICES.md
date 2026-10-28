# Third-party notices

Luna-MinimaxH3 is licensed under GPL-3.0 (see `LICENSE`). This file lists code adapted from other projects,
with its source, licence and where it lives here.

| Source | Licence | Adapted into | What |
|---|---|---|---|
| ComfyUI-MiniMaxH3-Context-Loop (ethanfel), `chain_nodes.py` MiniMaxH3ChainLoopEnd; its recursion adapted from Ethanfel's SxCP loop nodes | GPL-3.0, Copyright (C) 2026 NikoDemon80 | `h3_loop.py`: `_explore_dependencies`, `_collect_contained`, `find_body` / `expand_body` (from `_recurse`), `release_between_scenes` (shape of `_release_loop_boundary_resources`) | Loop body detection, GraphBuilder clone for the next scene, memory release between scenes |
| ComfyUI core (comfy_extras/nodes_minimax_h3.py: MiniMaxH3AddGuide, MiniMaxH3ImageToVideo, MiniMaxH3ReferenceToVideo) | GPL-3.0 | `h3_loop_guide.py`: `make_keyframe`, `valid_clip_length`, `_resize`, `vlm_items` / `insert_guide_items` | The keyframe-row construction of Add Guide, the FL2VA keyframe-image pairing, and core's 2 fps video item layout for the VLM |
| ComfyUI-Pixaroma (c) 2026 pixaroma, `LoadImageMini` state and resize helpers (`nodes/_resize_helpers.py`), `node_panel.mjs`, and Load Audio / Load Video (frame-rate handling, audio silence/loop choice) | MIT | Luna-H3-AssetLoader, Luna-H3-ReferenceLoader and their media code: the per-image state schema, the resize model, the settings panel idea, the audio loading rules; the rest is this pack's own | `asset_images.py`, `asset_loader.py`, `media_io.py`, `web/luna_asset_loader.js` |

## Licence of the shared frontend kit

`web/luna_theme.mjs`, `web/luna_help.mjs` and `web/luna_collapse.mjs` (and `web/luna_pack_theme.js`,
which uses them) are Apache-2.0, Copyright 2026 Luna AI Space. They stay Apache-2.0 inside this
GPL-3.0 pack; Apache-2.0 code may be combined into a GPL-3.0 work. Copy them into another pack under
Apache-2.0.

## What each licence asks

- GPL-3.0 (Context Loop, ComfyUI core): the combined work is GPL-3.0 and its source stays available.
- MIT (Pixaroma): keep the copyright notice and permission notice. The headers of `asset_images.py`,
  `asset_loader.py` and `media_io.py` say what was taken and what is this pack's own.
- No DaSiWa code is in this pack.
