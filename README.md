# Luna-MinimaxH3

The nodes for **MiniMax H3** in ComfyUI, twelve in one pack: the canvas and AV-latent nodes H3 needs and does not
ship, one loader for every H3 reference (and a lighter loader for pictures and a song), a shot list that plans scenes
over one song, and a scene loop that renders a whole music video in one queue, with guide keyframes.

Every node's id is its display name, and every node sits in the category `Luna-MinimaxH3`.
Each node has an (i) help button on its title bar, built from the node's own description and tooltips, and a
chevron that folds the settings away.

## Install

Clone or copy this folder into `ComfyUI/custom_nodes/`, then restart ComfyUI:

```
cd ComfyUI/custom_nodes
git clone https://github.com/lunaaispace-eng/Luna-MinimaxH3
```

Dependencies are in `requirements.txt` (Pillow, numpy; everything else, including PyAV and OpenCV, comes with
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

## Replaces

Luna-MinimaxH3 replaces ComfyUI-LunaMiniMaxH3 and the H3 nodes of ComfyUI-SaveSimple (Luna Asset Loader,
Luna H3 Reference Loader, Luna H3 Refs Out, Luna Shot List). Node ids changed with the rename, so workflows saved
with the old nodes need those nodes swapped for the new ones. Remove the old folders from `custom_nodes` when you
install this pack.

## Tests

```
python -m pytest tests
```

A ComfyUI folder must be importable; the registration test uses `COMFYUI_DIR` (default
`E:\ComfyUI-Easy-Install\ComfyUI`). Some loop-builder tests call an external graph builder script
(`music-video-from-audio` skill, `build_loop_graph.py`) that is not part of this pack.

## Licence

GPL-3.0, see [LICENSE](LICENSE). Derived code and its licences: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
The three `luna_*.mjs` frontend kit files are Apache-2.0.
