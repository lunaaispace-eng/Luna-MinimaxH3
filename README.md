# Luna-MinimaxH3

The nodes for **MiniMax H3** in ComfyUI, thirteen in one pack: the canvas and AV-latent nodes H3 needs and does not
ship, one loader for every H3 reference (and a lighter loader for pictures and a song), a shot list that plans scenes
over one song, a scene loop that renders a whole music video in one queue with guide keyframes, and a
two-resolution sampler (low-resolution pass, lift, full-resolution pass) in one node.

Every node's id is its display name, and every node sits in the category `Luna-MinimaxH3`.
Each node has an (i) help button and a chevron that folds the settings away.

## Install

Clone into `ComfyUI/custom_nodes/` (`git clone https://github.com/lunaaispace-eng/Luna-MinimaxH3`), then restart ComfyUI.

Dependencies are in `requirements.txt` (Pillow, numpy; PyAV, OpenCV and einops come with ComfyUI). The loop's scene
builder also uses the Beta 2.0 workflow nodes (comfyui-vrgamedevgirl); the other nodes need nothing else.

## Node tree

- **Luna-H3-Canvas**: H3 resolution, frame count (`frames`) and fps from ratio and seconds; an optional `image` sets the aspect.
- **Luna-H3-ConcatAVLatent**: Merges a video and an audio latent into H3's AV latent.
- **Luna-H3-ReferenceLoader**: Every H3 reference in one node: 9 images, 3 clips, 3 audio.
- **Luna-H3-AssetLoader**: Up to 9 ordered reference images, clips and a song; outputs include `video` and `frames`, and it runs empty.
- **Luna-H3-RefsOut**: Unpacks the loader refs into every H3 reference socket; per-category on/off switches, a rebuilt `asset_brief`, and the song as `audio`.
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

One schedule of `total_steps`. Pass 1 runs the first `transition` steps on a grid of `lowres_scale` (0.5 = half
size); pass 2 runs the rest at full size. Each pass clones the model, adds its own LoRA and video and audio sigma
shift. Between the passes the video is lifted to full size (learned upscaler, optionally blended with the pixel route
by `rho`, `w_min`, `w_max`) and re-noised; the audio is handed over, not re-noised. Mask-0 regions of `latent_image`
are restored exactly after each pass. Measured on DOJO2 (12 steps, transition 9): pass 1 sets motion, face and skin;
pass 2 refines detail.

### Main inputs and outputs

- `model`, `positive`, `negative`, `vae`, `latent_image` (H3 AV latent at final size; mask 0 keeps a region, an audio
  mask of 0 locks the song).
- Per pass: `lora_*`, `strength_*`, `shift_*`, `audio_shift_*`, `sampler_*`, `cfg_*` (`low` = pass 1, `high` = pass 2).
  Samplers come in as `SAMPLER` sockets, so each pass can use any sampler.
- Shared: `scheduler`, `total_steps`, `transition`, `highres_denoise`, `noise_seed`, `lowres_scale`, `upscaler_model`,
  `rho`, `w_min`, `w_max`.
- Optional: `low_res_head` (previous scene's `low_res_latent`), `sigmas_low`, `sigmas_high`, `high_res_model`,
  `positive_low`, `negative_low`.
- Outputs: `latent` (full size) and `low_res_latent` (pass 1 on the low grid, for the next scene's `low_res_head`).

`luna_recipes.json` holds tested values per LoRA; picking a LoRA fills that pass's strength, steps and shifts.

`high_res_model` (optional): not connected, pass 2 uses `model`. Connected, pass 2 clones it and adds its own LoRA and
shifts; pass 1 still uses `model`. Both must be MiniMax H3. Use it for a LoRA that only one pass should get. Tested
2026-10-29 (DOJO2, 2 MP): v1.3 unet plus Kijai's Singularity V2 LoRA on `model` and plain v1.3 on `high_res_model`
gave about 70 % of V2's extra motion at no extra time.

`positive_low`, `negative_low` (optional): conditioning for pass 1 only. Not connected, pass 1 uses `positive` and
`negative`, with their keyframe latents shrunk to the low grid. Connected, build it with a second `MiniMaxH3ImageToVideo`
at pass 1's size, as a native two-stage graph does; `negative_low` is used only when `cfg_low` is above 1.

`sigmas_low` ending at 0.01 or below means pass 1 has finished: pass 2 then draws fresh noise for the audio too, from
the same seed as the video, as a native second pass does. This works with any sampler in pass 1.

### Example workflows and tools

`example_workflows/` holds five small workflows (reference to video, image to video, LoRA on pass 1 only, 1 MP and
2 MP to 2K), built from ComfyUI core nodes and KJNodes. ComfyUI lists them under Templates › Custom Nodes. The 2K
ones need ComfyUI 0.43 nightly (2026-10-28) or newer. Tested 2026-10-29 on an RTX 4090: 166 s from 1 MP, 259 s from
2 MP. **Limit (24 GB card):** the 2K stage is proven only for 5 s clips (124 frames); at 14.4 s it did not finish its
first step in 12.5 minutes. Details and credits: `example_workflows/README.md`. `tools/` holds the scripts that build
and test these workflows; ComfyUI does not load them.

## Tests

With ComfyUI's Python: `python -m unittest discover -s tests` and `node --test tests/test_luna_sampler_ui.mjs`. The Sampler tests run on a tiny CPU H3 harness; no GPU needed. The registration test needs
a ComfyUI folder (`COMFYUI_DIR`).

## Licence

GPL-3.0, see [LICENSE](LICENSE). Derived code and its licences: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
The three `luna_*.mjs` frontend kit files are Apache-2.0.
