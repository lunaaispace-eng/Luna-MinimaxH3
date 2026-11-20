# Luna-MinimaxH3

The nodes for **MiniMax H3** in ComfyUI, seventeen in one pack:
- the canvas and AV-latent nodes H3 needs and does not ship;
- one loader for every H3 reference (and a lighter loader for pictures and a song);
- a shot list that plans scenes over one song;
- a scene loop that renders a whole music video in one queue with guide keyframes;
- a two-resolution sampler (low-resolution pass, lift, full-resolution pass) in one node;
- a clip chain where every clip continues the one before over 22 to 192 locked frames.

Every node:
- has its display name as its id;
- sits in the category `Luna-MinimaxH3`;
- has an (i) help button and a chevron that folds the settings away.

## Install

- Clone into `ComfyUI/custom_nodes/` (`git clone https://github.com/lunaaispace-eng/Luna-MinimaxH3`), then restart ComfyUI.
- Dependencies are in `requirements.txt`: Pillow, numpy. PyAV, OpenCV and einops come with ComfyUI.
- The loop's scene builder also uses the Beta 2.0 workflow nodes (comfyui-vrgamedevgirl); the other nodes need nothing else.

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
- **Luna-H3-Chain**: Many clips in one queue on a Luna-MetaBatch (Luna-Media-Tools) or VHS Meta Batch Manager; clip number, seed and song start per clip.
- **Luna-H3-ChainContinue**: Puts the end of the clip before at the start of this clip's latent, locked.
- **Luna-H3-ChainRemember**: Saves the end of this clip, full size and pass 1, for the next clip.
- **Luna-H3-ChainJoin**: Cuts the copied frames off and gathers the sound into one video. `clip_audio` is each clip's own
  sound, for a per-clip Luna-SaveVideo that keeps every clip on disk, so a stopped run can be continued and joined later.

## Replaces

- Luna-MinimaxH3 replaces ComfyUI-LunaMiniMaxH3 and the H3 nodes of ComfyUI-SaveSimple (Luna Asset Loader, Luna H3 Reference
  Loader, Luna H3 Refs Out, Luna Shot List).
- Node ids changed with the rename: in workflows saved with the old nodes, swap those nodes for the new ones.
- Remove the old folders from `custom_nodes` when you install this pack.
- It also replaces the separate Luna-H3-Sampler pack. That node keeps its id `Luna-H3-Sampler`, so saved workflows load
  unchanged.
- Remove the `Luna-H3-Sampler` folder from `custom_nodes`: two packs with the same node id collide.

## Luna-H3-Sampler

- One node for MiniMax H3 audio+video: a low-resolution pass, a lift of the video to full size, and a full-resolution pass.
- Each pass has its own LoRA, shift, sampler and CFG.
- The audio runs as one trajectory through both passes.
- Protected regions (a video prefix with mask 0, a locked song) come out exactly as they went in, so scenes can be chained.
- Own code, written from the SelfLift paper (arXiv:2609.02036).
- The learned latent upscaler `minimax_h3_latent_upscaler_3d_fp16.safetensors` is optional (`upscaler_model: none` lifts with
  nearest).

### Set it up, step by step

**1. What you need**
- ComfyUI 0.43 nightly (2026-10-28) or newer: MiniMax H3 is in ComfyUI core from then on.
- This node pack, [Luna-MinimaxH3](https://github.com/lunaaispace-eng/Luna-MinimaxH3): in ComfyUI's `custom_nodes` folder run
  `git clone https://github.com/lunaaispace-eng/Luna-MinimaxH3`, then restart ComfyUI. The Sampler needs no other node pack.
- The model files, in these `ComfyUI/models` folders:

| File | Folder | Where |
|---|---|---|
| `minimax_h3_ref2va_pruned_int8_convrot` (or a fine-tune) | `diffusion_models` | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/diffusion_models) |
| `qwen3vl_32b_minimax_h3_int8_convrot` | `text_encoders` | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/text_encoders) |
| `minimax_h3_video_vae_int8_convrot` | `vae` | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/vae) |
| `minimax_h3_audio_vae_fp32` | `vae` | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/vae) |
| A speed LoRA pair (see step 4) | `loras` | DMAD: [ZhengmingYu/DMAD](https://huggingface.co/ZhengmingYu/DMAD); Pika: [Pika-Labs/Pika-H3-Flex](https://huggingface.co/Pika-Labs/Pika-H3-Flex); lightx: [lightx2v/Minimax-h3-Turbo](https://huggingface.co/lightx2v/Minimax-h3-Turbo) |
| `minimax_h3_latent_upscaler_3d_fp16`, optional | `latent_upscale_models` | [Open-H3-Regenerate-2K](https://huggingface.co/Efficient-Large-Model/Open-H3-Regenerate-2K) |

**2. Start from an example**
- In ComfyUI open Templates › Custom Nodes › Luna-MinimaxH3 › `Luna-H3-Sampler - Reference to Video`.
- In each loader pick your files from step 1. A red node means a file or a node is missing.

**3. Or wire it yourself** (the same graph, ComfyUI core nodes only)
1. `Load Diffusion Model` (UNETLoader): the H3 model.
2. `Load CLIP` (CLIPLoader), type `minimax`: the text encoder.
3. Two `Load VAE`: the video VAE and the audio VAE.
4. The H3 conditioning node, for example `MiniMaxH3ReferenceToVideo`: prompt, `width`, `height`, `length`, reference
   pictures. Its `positive` and `latent` go to the Sampler. Size and length are the final video's.
5. `ConditioningZeroOut` on that `positive` gives the `negative` (CFG 1 with a speed LoRA).
6. Two `KSamplerSelect` nodes, one into `sampler_low`, one into `sampler_high`. The ⚙ menu sets their sampler names.
7. **Luna-H3-Sampler**: `model`, `positive`, `negative`, `vae` (the video VAE), `latent_image` (from the H3 node, never an
   Empty Latent Image: it must be H3's audio+video latent).
8. Its `latent` output into `VAE Decode` (video VAE) for the frames and `VAE Decode Audio` (audio VAE) for the sound, both
   into `Create Video` (24 fps), then `Save Video`.

**4. Set the values with the ⚙ menu**
1. Click the ⚙ on the Sampler's title bar.
2. Tab **Recipes**, pick one:
   - **Natural skin · 2 MP**: DMAD 8-step LoRA in pass 1, DMAD 4-step in pass 2. Truest faces in our tests.
   - **Smooth · 2 MP**: the two Pika LoRAs. Smoother motion, a little washed out.
   - **Fast · lightx 12 / 9**: the lightx turbo LoRA in both passes.
   - A recipe whose LoRA file is not in `models/loras` shows "missing" and cannot be applied.
3. **Validate**, then **Apply**. Apply sets the LoRAs, strengths, shifts, steps, `transition`, `handover_sigma`,
   `lowres_scale` and the two KSamplerSelect sampler names.
4. The readout under the node now names the recipe. Change a widget and it says "custom (changed from …)".
- Other LoRAs: tab **Custom**. Pick the two LoRAs and the final size; the node recommends the rest; **Apply**.

**5. The settings of each recipe** (what Apply writes; set them by hand if you do not use the ⚙ menu)

| Setting | Natural skin | Smooth | Fast |
|---|---|---|---|
| `lora_low`, strength | DMAD 8-step rank 58, 1.0 | Pika ref2va rank 55, 1.0 | lightx fl2v turbo rank 21, 0.6 |
| `lora_high`, strength | DMAD 4-step rank 39, 1.0 | Pika fl2va, 1.0 | lightx fl2v turbo rank 21, 0.8 |
| sampler pass 1 / pass 2 (KSamplerSelect) | `lcm` / `er_sde` | `lcm` / `lcm` | `euler` / `euler` |
| `scheduler_low`, `scheduler_high` | `simple` | `linear_quadratic` | `simple` |
| `shift_low`, `shift_high` | 12 | 12 | 12 |
| `audio_shift_low`, `audio_shift_high` | 2 | 3 | 3 |
| `cfg_low`, `cfg_high` | 1 | 1 | 1 |
| `total_steps` / `transition` | 6 / 4 | 8 / 5 | 12 / 9 |
| `handover_sigma` | 0.923077 | 0 | 0 |
| `highres_denoise` | 1 | 1 | 1 |
| `lowres_scale` | by size, step 6 | by size, step 6 | by size, step 6 |
| `upscaler_model` | `minimax_h3_latent_upscaler_3d_fp16`, or `none` | same | same |
| `rho`, `w_min`, `w_max` | 0, 0.5, 1 | 0, 0.5, 1 | 0, 0.5, 1 |

- These values do not change with the resolution: only `lowres_scale` does (step 6).
- All three are tested at 1920x1088 (2 MP), 5 s, once each; Smooth also at 1344x768 (1 MP) with `lowres_scale` 0.5.

**6. Size and length**
- Only `lowres_scale` changes with the size; the recipe's other settings (step 5) stay the same.

| Final size | MP | `lowres_scale` | Pass 1 | Tested |
|---|---|---|---|---|
| 1344x768 | 1.0 | 0.5 | 672x384 | yes: clean, with the Smooth settings (Pika, `lcm`, `linear_quadratic`, 8 / 5) and the upscaler |
| 1920x1088 | 2.0 | 0.7 | about 1 MP | yes: clean, all three recipes. 0.5 is too small here: it draws letters |
| 1280x720, 960x544 | 0.9, 0.5 | 0.5 | a quarter of the area | no; 0.5 as at 1 MP |
| 1536x864 | 1.3 | 0.85 | about 1 MP | no |
| 2560x1440 | 3.5 | 0.5 | about 1 MP | no |
| 3840x2176 | 8.0 | 0.35 | about 1 MP | no |

- Up to 1 MP: `lowres_scale` 0.5. Above 1 MP: pass 1 at about 1 MP, `lowres_scale` = √(1 MP ÷ final MP) in steps of 0.05.
  The ⚙ **Custom** tab computes that second rule from the final size.
- For a 2K or 4K video: render at 1344x768 with 0.5, then upscale the video (tested: Luna-RelayVSR x2 to 2688x1536,
  178 s for the whole graph). Far faces are smaller at 1 MP, so they keep less detail than a 2 MP render.
- Memory grows with the final size: 2 MP, 5 s is tested on an RTX 4090 (24 GB) with 64 GB RAM.
- Width and height are multiples of 32. H3 lengths are 17·k + 5 frames: 124 frames = 5 s at 24 fps.
- Close other GPU programs first.

**7. Run**
- Queue. The readout shows which steps run small and which at full size, and the hand-over sigma.

**8. If it stops with an error**

| Message | Fix |
|---|---|
| `latent_image must be an H3 audio+video latent` | Wire `latent_image` from the H3 conditioning node, not an Empty Latent Image. |
| `low_res_model must be a rectified-flow model (MiniMax H3)` | `model` is not an H3 model. Load the H3 diffusion model. |
| `transition N is not in [1, M]` | `transition` must be at least 1 and below `total_steps`. |
| A recipe shows "missing" | Put that LoRA file in `models/loras` and reopen the ⚙ menu. |
| Apply says `sampler_low is not connected` | Wire a `KSamplerSelect` into `sampler_low` and `sampler_high`. |
| A red node, "missing node type" | Update ComfyUI to a nightly with H3, install this pack, restart. |
| `upscaler_model` lists only `none` | Optional: put the upscaler file in `models/latent_upscale_models` and restart. |

### How it works

- One schedule of `total_steps`.
- Pass 1 runs the first `transition` steps on a grid of `lowres_scale` (0.5 = half size); pass 2 runs the rest at full size.
- Each pass clones the model, adds its own LoRA and video and audio sigma shift.
- Between the passes the video is lifted to full size (learned upscaler, optionally blended with the pixel route by `rho`,
  `w_min`, `w_max`) and re-noised; the audio is handed over, not re-noised.
- Mask-0 regions of `latent_image` are restored exactly after each pass.
- Measured on DOJO2 (12 steps, transition 9): pass 1 sets motion, face and skin; pass 2 refines detail.

### Main inputs and outputs

- `model`, `positive`, `negative`, `vae`, `latent_image` (H3 AV latent at final size; mask 0 keeps a region, an audio
  mask of 0 locks the song).
- Per pass: `lora_*`, `strength_*`, `shift_*`, `audio_shift_*`, `sampler_*`, `cfg_*`, `scheduler_*` (`low` = pass 1,
  `high` = pass 2). Samplers come in as `SAMPLER` sockets, so each pass can use any sampler. Pass 1 takes its first
  `transition` steps from a schedule built with `scheduler_low`, pass 2 the rest from one built with `scheduler_high`.
  A workflow saved with the older single `scheduler` opens with that scheduler in both.
- Shared: `total_steps`, `transition`, `highres_denoise`, `noise_seed`, `lowres_scale`, `upscaler_model`, `rho`, `w_min`,
  `w_max`.
- Optional: `low_res_head` (previous scene's `low_res_latent`), `sigmas_low`, `sigmas_high`, `high_res_model`,
  `positive_low`, `negative_low`, `handover_sigma` (a widget, default 0).
- Outputs: `latent` (full size) and `low_res_latent` (pass 1 on the low grid, for the next scene's `low_res_head`).

### Settings in detail

**Where pass 2 starts**
- `handover_sigma` above 0 (default 0, the step split; the ⚙ menu sets it per LoRA pair): the hand-over is a noise level. Pass 1 runs its `transition` steps from 1 down to it,
  pass 2 runs the remaining steps from it to 0, each with its own scheduler's shape on its own shift. `highres_denoise` is then
  ignored. 0 splits the schedule by step number, as below. A workflow saved before this widget opens with 0.
- `handover_sigma` 0, `sigmas_high` empty and `highres_denoise` 1: pass 2 continues the same schedule from the transition.
- `highres_denoise` below 1: pass 2 restarts lower and keeps more of pass 1's picture.
- `sigmas_high` connected (ManualSigmas): pass 2 runs exactly that list; `highres_denoise` and `scheduler_high` are not used,
  although the readout still shows the denoise. `sigmas_low` does the same for pass 1 and must end above 0 (for example 0.001).
- With lists connected, keep `total_steps` and `transition` equal to their step counts: the node still checks them.
- `linear_quadratic` is a fixed curve that ignores shift. With many steps its last steps are all low, so it suits pass 2 only with
  few steps (8 steps, transition 5: pass 2 starts at 0.909).

**LoRAs**
- One speed LoRA per pass, in `lora_low` / `lora_high`. Pika at 1.0.
- `lora_high` loads on top of `high_res_model`: a LoRA in both loads twice.
- Pass 1's LoRA sets the motion. Without a speed LoRA, a few steps in pass 1 break the picture.

**Custom sigmas**

| List | Use |
|---|---|
| `1.0, 0.972973, 0.923077, 0.8, 0.571429, 0.333333, 0.001` | Pika's 6 trained steps in pass 1 |
| `0.8, 0.571429, 0.333333, 0.0` | Pika's last 3 steps in pass 2 |
| `0.7, 0.494118, 0.0` | refiner pass (Kijai, Trent) |
| `0.909375, 0.857143, 0.666667, 0.0` | refiner pass (NVlabs) |

**Song recipe** (tested on one scene, 1920x1088, 8.7 s, about 200 s on an RTX 4090)
- Pass 1 `pika_h3_flex_ref2va_v1_avg_rank_55` at 1.0, pass 2 `pika_h3_flex_fl2va_v1` at 1.0.
- `lcm` and `linear_quadratic` in both, `total_steps` 8, `transition` 5, `lowres_scale` 0.5, Sol-Attn tau 1.0.
- For generated sound, `lowres_scale` 0.7 at 1920x1088.

**Recipes per LoRA**
- `luna_recipes.json` holds tested values per LoRA; picking a LoRA fills that pass's strength, steps and shifts.
- A LoRA is matched by its file name, in whatever folder ComfyUI finds it.

**Recipe menu (⚙ on the title bar)**
- **Recipes**: the tested pairs (`luna_pair_recipes.json`) and your own. Pick one to see every value it sets; **Validate**
  checks it, **Apply** writes the widgets and the `sampler_name` of the KSamplerSelect nodes wired into `sampler_low` and
  `sampler_high`. A recipe whose LoRA is not installed says "missing" and cannot be applied.
- **Custom**: pick the two LoRAs, strengths, samplers and the final size. The node recommends the hand-over sigma (the
  lowest sigma point both LoRAs' schedulers make that leaves pass 2 two of its own steps), the steps per pass and a
  `lowres_scale` that puts pass 1 at about 1 MP (the size is read from Luna-H3-Canvas, or typed). A LoRA with no known
  sigma points, or with a custom list (Pika), gets the step split (`handover_sigma` 0). Every value can be changed; a
  changed one is marked "yours ↺".
- **Save as recipe**: after a clip you judged, save what the node runs now, with what it was tested on and your verdict.
  Saved recipes live in ComfyUI's `user\default\luna_sampler_recipes.json`, so a pack update never touches them.
- The readout names the recipe while the widgets still match it.

**`high_res_model`** (optional)
- Not connected: pass 2 uses `model`.
- Connected: pass 2 clones it and adds its own LoRA and shifts; pass 1 still uses `model`.
- Both must be MiniMax H3. Use it for a LoRA that only one pass should get.
- Tested 2026-10-29 (DOJO2, 2 MP): v1.3 unet plus Kijai's Singularity V2 LoRA on `model` and plain v1.3 on `high_res_model`
  gave about 70 % of V2's extra motion at no extra time.

**`positive_low`, `negative_low`** (optional)
- Conditioning for pass 1 only.
- Not connected: pass 1 uses `positive` and `negative`, with their keyframe latents shrunk to the low grid.
- Connected: build it with a second `MiniMaxH3ImageToVideo` at pass 1's size, as a native two-stage graph does.
- `negative_low` is used only when `cfg_low` is above 1.

**Finished pass 1**
- `sigmas_low` ending at 0.01 or below means pass 1 has finished.
- Pass 2 then draws fresh noise for the audio too, from the same seed as the video, as a native second pass does.
- This works with any sampler in pass 1.

### Example workflows and tools

- `example_workflows/` holds five small workflows (reference to video, image to video, LoRA on pass 1 only, 1 MP and 2 MP to
  2K), built from ComfyUI core nodes and KJNodes.
- ComfyUI lists them under Templates › Custom Nodes.
- The 2K ones need ComfyUI 0.43 nightly (2026-10-28) or newer.
- Tested 2026-10-29 on an RTX 4090: 166 s from 1 MP, 259 s from 2 MP.
- **Limit (24 GB card):** the 2K stage is proven only for 5 s clips (124 frames); at 14.4 s it did not finish its first step in
  12.5 minutes.
- Details and credits: `example_workflows/README.md`.
- `tools/` holds the scripts that build and test these workflows; ComfyUI does not load them.

## Tests

- With ComfyUI's Python: `python -m unittest discover -s tests` and `node --test tests/test_luna_sampler_ui.mjs`.
- The Sampler tests run on a tiny CPU H3 harness; no GPU needed.
- The registration test needs a ComfyUI folder (`COMFYUI_DIR`).

## Licence

- GPL-3.0, see [LICENSE](LICENSE).
- Derived code and its licences: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
- The three `luna_*.mjs` frontend kit files are Apache-2.0.
