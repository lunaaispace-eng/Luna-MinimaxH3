# Luna-MinimaxH3

The nodes for **MiniMax H3** in ComfyUI, thirteen in one pack: the canvas and AV-latent nodes H3 needs and does not
ship, one loader for every H3 reference (and a lighter loader for pictures and a song), a shot list that plans scenes
over one song, a scene loop that renders a whole music video in one queue, with guide keyframes, and a two-resolution
sampler (low-resolution pass, lift, full-resolution pass) in one node.

Every node's id is its display name, and every node sits in the category `Luna-MinimaxH3`.
Each node has an (i) help button on its title bar, built from the node's own description and tooltips, and a
chevron that folds the settings away.

## Install

Clone or copy this folder into `ComfyUI/custom_nodes/`, then restart ComfyUI:

```
cd ComfyUI/custom_nodes
git clone https://github.com/lunaaispace-eng/Luna-MinimaxH3
```

Dependencies are in `requirements.txt` (Pillow, numpy; everything else, including PyAV, OpenCV and einops, comes with
ComfyUI). The loop's scene builder also uses the Beta 2.0 workflow nodes (comfyui-vrgamedevgirl); the other
nodes need nothing else.

## Node tree

- **Luna-H3-Canvas**: H3 resolution, frame count and fps from ratio and seconds.
- **Luna-H3-ConcatAVLatent**: Merges a video and an audio latent into H3's AV latent.
- **Luna-H3-ReferenceLoader**: Every H3 reference in one node: 9 images, 3 clips, 3 audio.
- **Luna-H3-AssetLoader**: Up to 9 ordered reference images and a song.
- **Luna-H3-RefsOut**: Unpacks the Asset Loader's refs into every H3 reference socket.
- **Luna-H3-ShotList**: Scenes of different lengths over one song, one scene per run.
- **Luna-H3-LoopStart**: Starts a scene loop from a JSON scene list.
- **Luna-H3-LoopScene**: The current scene's settings, audio window, seeds and refs.
- **Luna-H3-LoopProfile**: Model, LoRAs, steps and shifts for the scene's profile.
- **Luna-H3-GuideApply**: Adds the scene's guide keyframes to the positive conditioning.
- **Luna-H3-GuideClip**: A CLIP that also shows the guide frames to the VLM.
- **Luna-H3-LoopEnd**: Saves the scene's frames, audio and latent; runs the next scene.
- **Luna-H3-Sampler**: Two-resolution H3 audio+video sampling in one node (see below).

## Replaces

Luna-MinimaxH3 replaces ComfyUI-LunaMiniMaxH3 and the H3 nodes of ComfyUI-SaveSimple (Luna Asset Loader,
Luna H3 Reference Loader, Luna H3 Refs Out, Luna Shot List). Node ids changed with the rename, so workflows saved
with the old nodes need those nodes swapped for the new ones. Remove the old folders from `custom_nodes` when you
install this pack.

It also replaces the separate [Luna-H3-Sampler](https://github.com/lunaaispace-eng/Luna-H3-Sampler) pack. That node
keeps its id `Luna-H3-Sampler`, so saved workflows load unchanged. Remove the `Luna-H3-Sampler` folder from
`custom_nodes`: two packs with the same node id collide.

## Luna-H3-Sampler

One node for MiniMax H3 audio+video: a low-resolution pass, a lift of the video to full size, and a full-resolution
pass. Each pass has its own LoRA, shift, sampler and CFG. The audio runs as one trajectory through both passes.
Protected regions (a video prefix with mask 0, a locked song) come out exactly as they went in, so scenes can be
chained. Own code, written from the SelfLift paper (arXiv:2609.02036). The learned latent upscaler
`minimax_h3_latent_upscaler_3d_fp16.safetensors` is optional (`upscaler_model: none` lifts with nearest).

### How it works

1. One schedule of `total_steps` with `scheduler`. Pass 1 runs the first `transition` steps on a grid of
   `lowres_scale` (0.5 = half size). Pass 2 runs the rest at full size.
2. Each pass clones the incoming model, adds its own LoRA and its own video and audio sigma shift, and builds its
   sigmas on that shift.
3. Between the passes the video is lifted to full size (learned upscaler, optionally blended with the pixel route by
   `rho`, `w_min`, `w_max`) and re-noised to pass 2's first sigma. The audio is handed over, not re-noised.
4. Mask-0 regions of `latent_image` are restored exactly after each pass.

What each pass decides (MEASURED on DOJO2, 12 steps / transition 9, 2026-10-29): pass 1 sets the motion and also the
face and skin; pass 2 (starting at sigma 0.8) refines detail. A different model in pass 2 changed nothing visible.

### Inputs

| Input | What it is |
|---|---|
| `model` | MiniMax H3 model, with any memory patches and extra LoRAs already on it. Pass 1 uses it, and pass 2 too unless `high_res_model` is connected. |
| `positive`, `negative`, `vae` | Conditioning (shared by both passes) and the video VAE (used by the pixel route of the lift). |
| `latent_image` | H3 AV latent at the final size. Mask 0 keeps that region; an audio mask of 0 locks the song. |
| `lora_low`, `strength_low`, `shift_low`, `audio_shift_low`, `sampler_low`, `cfg_low` | Pass 1. |
| `lora_high`, `strength_high`, `shift_high`, `audio_shift_high`, `sampler_high`, `cfg_high` | Pass 2. |
| `scheduler`, `total_steps`, `transition` | The one shared schedule and where it splits. |
| `highres_denoise` | 1.0 starts pass 2 at the transition sigma; lower changes the lifted video less. |
| `noise_seed` | Pass 1 seed; pass 2's fresh video noise uses the next seed. |
| `lowres_scale`, `upscaler_model`, `rho`, `w_min`, `w_max` | The lift. |
| `low_res_head` (optional, lazy) | The previous scene's `low_res_latent`: written into the low-grid prefix. Needs a mask-0 prefix in `latent_image`. |
| `sigmas_low`, `sigmas_high` (optional) | Replace the computed sigmas of that pass. |
| `high_res_model` (optional) | Pass 2's model. See below. |

Samplers come in as `SAMPLER` sockets (`KSamplerSelect`, `SamplerERSDE`, Detail Daemon and the like), so each pass
can use any sampler with its own settings.

Outputs: `latent` (full size, pass-2 video and the finished audio) and `low_res_latent` (pass-1 video on the low grid
with the same audio; connect it to the next scene's `low_res_head`). The node shows a readout: both passes' LoRA,
shifts and sigmas, the hand-over sigma, the pass-2 model and the upscaler.

### Recipes

`luna_recipes.json` holds tested values per LoRA. Picking a LoRA that has a recipe fills that pass's strength,
steps and shifts in the UI, and the node warns when a value differs. DMAD
(`minimax_h3_DMAD_4step_full_lora_avg_rank_39_bf16`) uses shift 12 / audio 2, the value on its card.

Measured 2 MP recipe (Singularity v1.3 pruned int8): ref2v lightx2v turbo rank-20 LoRA 0.6 in pass 1 and 0.8 in
pass 2, Euler both passes, `simple`, 12 steps, transition 9, shift 12 / 3. It matches the original SelfLift node on
the same scenes and runs about 5 % faster.

### `high_res_model` (optional, added 2026-10-29)

Not connected: pass 2 uses `model`, as before. Connected: pass 2 clones it and adds `lora_high`, `strength_high`,
`shift_high` and `audio_shift_high`; pass 1 still uses `model`. Both must be MiniMax H3 (one latent space and audio
scale).

Use it for a LoRA stack that only one pass should get: chain the LoRA on `model` and connect the plain model to
`high_res_model`. A LoRA chained before `model` with nothing on `high_res_model` acts in both passes. Also for memory
patches that only pass 2 needs.

Tested 2026-10-29 (DOJO2, 2 MP, same seed):
- Singularity V2 unet in pass 1 + v1.3 unet in pass 2: motion and look both followed V2; +9 s for the second unet.
  A second unet in pass 2 is not worth it with this recipe.
- v1.3 unet + Kijai's Singularity V2 LoRA (`minimax_h3_singularity_v2_lora_avg_rank_351_bf16`, 1.0) on `model`,
  plain v1.3 on `high_res_model`: about 70 % of V2's extra motion, the steadiest motion of all takes, v1.3's skin
  colour, one unet, no extra time. Its strength works as a dial between normal and fast motion.

### Example workflows

`example_workflows/` holds five small workflows (reference to video, image to video, LoRA on pass 1 only, and 1 MP or
2 MP to 2K) built from ComfyUI core nodes and KJNodes only. ComfyUI lists them under Templates › Custom Nodes.

The two 2K workflows add Kijai's 2K refine stage (learned latent upscale to 2560x1440, the Open-H3-Regenerate-2K
refiner LoRA, 2 steps) after this node. Tested 2026-10-29 on an RTX 4090: from 1 MP in 166 s, better than Kijai's own
chain at the same size; from 2 MP in 259 s, the best quality. They need ComfyUI 0.43 nightly (2026-10-28) or newer.

**Limit (24 GB card):** the 2K stage is proven only for 5 s clips (124 frames). At 14.4 s (345 frames) it did not
finish its first step in 12.5 minutes on an RTX 4090 with 64 GB RAM, while this node's 2 MP stage ran the same scene
fine. Lengths between are untested; a windowed 2K pass is planned. Table: [example_workflows/README.md](example_workflows/README.md) › Limit.
Details, credits and model links: `example_workflows/README.md`.

### Tools

`tools/` holds the scripts used to build and test the sampler's workflows: `make_example_workflows.py` (the graphs
behind `example_workflows/`), `make_scene_rerun.py`, `make_luna_2k_chain.py`, `make_r2v_2k.py`, `ui_to_api.py`,
`stitch_compare.py`, `measure_detail.py`, `measure_audio.py`, `repair_luna_workflow.py`, `make_test_workflow.py`,
`make_luna_test_workflow.py`. They are not loaded by ComfyUI.

## Tests

```
python -m pytest tests
node --test tests/test_luna_sampler_ui.mjs
```

The Luna-H3-Sampler tests run on a tiny CPU H3 harness (`tests/h3_harness.py`); no GPU needed.

A ComfyUI folder must be importable; the registration test uses `COMFYUI_DIR` (default
`E:\ComfyUI-Easy-Install\ComfyUI`). Some loop-builder tests call an external graph builder script
(`music-video-from-audio` skill, `build_loop_graph.py`) that is not part of this pack.

## Licence

GPL-3.0, see [LICENSE](LICENSE). Derived code and its licences: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
The three `luna_*.mjs` frontend kit files are Apache-2.0.
