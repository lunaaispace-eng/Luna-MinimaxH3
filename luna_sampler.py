"""Luna-H3-Sampler — one node for both passes.

The base model arrives with its memory patches already applied. This node clones
it twice. Each clone gets that pass's LoRA and that pass's video/audio shift,
then `two_pass_sample` runs the shared schedule: the first `transition` steps at
the low grid, the rest at the full grid.
"""

from __future__ import annotations

import logging

import comfy.model_sampling
import comfy.samplers
import comfy.sd
import comfy.utils
import folder_paths
from comfy_api.latest import io, ui

if __package__:
    from .luna_lift import lift_video
    from .luna_recipes import check_values
    from .luna_two_pass import arm_sigmas, two_pass_sample
    from .luna_upscaler import learned_latent_lift, list_upscaler_models
else:
    from luna_lift import lift_video
    from luna_recipes import check_values
    from luna_two_pass import arm_sigmas, two_pass_sample
    from luna_upscaler import learned_latent_lift, list_upscaler_models


DESCRIPTION = """\
Pass 1 runs the first transition steps of one schedule at the low grid. Pass 2 runs the remaining steps at the full grid. Each pass clones the base model (pass 2 clones high_res_model when it is connected), loads its own LoRA and sets its own video and audio shift, then samples with its own sampler and CFG.

Pass 1 inputs are lora_low, strength_low, shift_low, audio_shift_low, sampler_low and cfg_low. Pass 2 inputs are lora_high, strength_high, shift_high, audio_shift_high, sampler_high and cfg_high. scheduler, total_steps, transition, highres_denoise and noise_seed belong to the one shared schedule. lowres_scale, upscaler_model, rho, w_min and w_max belong to the lift. upscaler_model selects the learned H3 latent upscaler; none uses nearest. low_res_head is the previous scene's low-grid tail. sigmas_low and sigmas_high replace the computed schedules when they are connected.

Audio is one trajectory across both passes and finishes in pass 2, next to the full-size video. A locked song (audio mask 0) stays exact in both outputs.

positive_low and negative_low are optional: conditioning built at pass 1's size, used by pass 1 only.

latent is the full-grid result. Connect it to a save node, or to the next scene as latent_image. low_res_latent is the low-grid video with that same finished audio. Connect it to the next scene's low_res_head.
"""


def _lora_names():
    try:
        names = [name.replace("\\", "/") for name in folder_paths.get_filename_list("loras")]
    except Exception:
        names = []
    if "none" in names:
        names = [name for name in names if name != "none"]
    return ["none", *names]


def _sampler_name(sampler):
    function = getattr(sampler, "sampler_function", None)
    name = getattr(function, "__name__", "")
    if isinstance(name, str) and name.startswith("sample_"):
        return name[len("sample_"):]
    return None


def _sigma_text(sigmas):
    return ", ".join(f"{float(item):.4f}" for item in sigmas)


def _apply_shift(model, shift, audio_shift):
    """Core MiniMaxH3SigmaShift: clone, ModelSamplingAV + CONST, both transformer keys."""
    shifted = model.clone()

    class ModelSamplingAdvanced(comfy.model_sampling.ModelSamplingAV, comfy.model_sampling.CONST):
        pass

    original = shifted.get_model_object("model_sampling")
    model_sampling = ModelSamplingAdvanced(shifted.model.model_config)
    model_sampling.set_parameters(shift=shift, audio_shift=audio_shift)
    if hasattr(original, "noise_scale"):
        model_sampling.set_noise_scale(original.noise_scale)
    shifted.add_object_patch("model_sampling", model_sampling)
    options = shifted.model_options["transformer_options"] = shifted.model_options.get("transformer_options", {}).copy()
    options["minimax_h3_sigma_shift_video"] = shift
    options["minimax_h3_sigma_shift_audio"] = audio_shift
    return shifted


def _load_lora(model, lora_name, strength, loaded):
    """Core LoraLoaderModelOnly. `none` and strength 0 load nothing."""
    if not lora_name or lora_name == "none" or float(strength) == 0.0:
        return model
    if lora_name not in loaded:
        path = folder_paths.get_full_path_or_raise("loras", lora_name)
        loaded[lora_name] = comfy.utils.load_torch_file(path, safe_load=True)
    patched, _clip = comfy.sd.load_lora_for_models(model, None, loaded[lora_name], float(strength), 0)
    return patched


def _arm_model(model, lora_name, strength, shift, audio_shift, loaded):
    cloned = model.clone()
    cloned = _load_lora(cloned, lora_name, strength, loaded)
    return _apply_shift(cloned, float(shift), float(audio_shift))


def _readout(low, high, sigmas_low, sigmas_high, total_steps, transition, highres_denoise, warnings):
    lines = [f"pass 1: steps 1-{transition} (low) / pass 2: steps {transition + 1}-{total_steps} (high)"]
    if highres_denoise is not None and float(highres_denoise) < 1.0:
        lines.append(f"highres_denoise {float(highres_denoise):.2f}")
    for label, arm, sigmas in (("pass 1", low, sigmas_low), ("pass 2", high, sigmas_high)):
        lines.append(
            f"{label}: {arm['lora']} × {float(arm['strength']):.2f}, "
            f"shift {float(arm['shift']):.1f} / {float(arm['audio_shift']):.1f}, "
            f"sigmas {_sigma_text(sigmas)}"
        )
    lines.append(f"hand-over sigma {float(sigmas_low[-1]):.4f} -> {float(sigmas_high[0]):.4f}")
    lines.extend(warnings)
    return "\n".join(lines)


class LunaH3Sampler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        upscalers = list_upscaler_models()
        default_upscaler = next(
            (name for name in upscalers if name.rsplit("/", 1)[-1] == "minimax_h3_latent_upscaler_3d_fp16.safetensors"),
            next((name for name in upscalers if name != "none"), "none"),
        )
        return io.Schema(
            node_id="Luna-H3-Sampler",
            display_name="Luna-H3-Sampler",
            category="Luna-MinimaxH3",
            description=DESCRIPTION,
            inputs=[
                io.Model.Input(
                    "model",
                    tooltip="Base MiniMax H3 model. Memory patches are already on this chain. Each pass clones it.",
                ),
                io.Conditioning.Input("positive", tooltip="Positive conditioning, shared by both passes."),
                io.Conditioning.Input("negative", tooltip="Negative conditioning, shared by both passes."),
                io.Vae.Input(
                    "vae",
                    tooltip="Video VAE. The lift uses it for the pixel route when rho is above 0.",
                ),
                io.Latent.Input(
                    "latent_image",
                    tooltip="H3 audio+video latent at the final size. Mask 0 keeps that region. An audio mask of 0 locks the song.",
                ),
                io.Combo.Input(
                    "lora_low",
                    options=_lora_names(),
                    default="none",
                    tooltip="Pass 1 LoRA, or none. Picking a LoRA that has a recipe fills this pass's strength, steps and shifts.",
                ),
                io.Float.Input(
                    "strength_low",
                    default=0.8,
                    min=0.0,
                    max=2.0,
                    step=0.01,
                    tooltip="Pass 1 LoRA strength. 0 loads no LoRA. 2 is the maximum.",
                ),
                io.Float.Input(
                    "shift_low",
                    default=12.0,
                    min=0.01,
                    max=100.0,
                    step=0.01,
                    tooltip="Pass 1 video sigma shift. This pass's sigmas are built on it.",
                ),
                io.Float.Input(
                    "audio_shift_low",
                    default=3.0,
                    min=0.01,
                    max=100.0,
                    step=0.01,
                    tooltip="Pass 1 audio sigma shift. The audio level follows this and the video shift.",
                ),
                io.Sampler.Input("sampler_low", tooltip="Pass 1 sampler. Any sampler, including er_sde."),
                io.Float.Input(
                    "cfg_low",
                    default=1.0,
                    min=0.0,
                    max=100.0,
                    step=0.1,
                    tooltip="Pass 1 CFG. 1.0 is the distilled H3 setting.",
                ),
                io.Combo.Input(
                    "lora_high",
                    options=_lora_names(),
                    default="none",
                    tooltip="Pass 2 LoRA, or none. It may be a different file from pass 1.",
                ),
                io.Float.Input(
                    "strength_high",
                    default=0.8,
                    min=0.0,
                    max=2.0,
                    step=0.01,
                    tooltip="Pass 2 LoRA strength. 0 loads no LoRA. 2 is the maximum.",
                ),
                io.Float.Input(
                    "shift_high",
                    default=12.0,
                    min=0.01,
                    max=100.0,
                    step=0.01,
                    tooltip="Pass 2 video sigma shift. Pass 2's sigmas are built on this shift, not pass 1's.",
                ),
                io.Float.Input(
                    "audio_shift_high",
                    default=3.0,
                    min=0.01,
                    max=100.0,
                    step=0.01,
                    tooltip="Pass 2 audio sigma shift. The hand-over moves the audio onto this shift.",
                ),
                io.Sampler.Input("sampler_high", tooltip="Pass 2 sampler. It can differ from pass 1."),
                io.Float.Input(
                    "cfg_high",
                    default=1.0,
                    min=0.0,
                    max=100.0,
                    step=0.1,
                    tooltip="Pass 2 CFG. 1.0 is the distilled H3 setting.",
                ),
                io.Combo.Input(
                    "scheduler",
                    options=list(comfy.samplers.SCHEDULER_NAMES),
                    default="simple",
                    tooltip="Scheduler for the one shared schedule. Each pass runs it on its own shift.",
                ),
                io.Int.Input(
                    "total_steps",
                    default=12,
                    min=1,
                    max=200,
                    step=1,
                    tooltip="Length of the one schedule. Pass 1 runs the first transition steps; pass 2 runs the rest.",
                ),
                io.Int.Input(
                    "transition",
                    default=8,
                    min=1,
                    max=199,
                    step=1,
                    tooltip="Number of low-resolution steps. It has to be at least 1 and below total_steps.",
                ),
                io.Float.Input(
                    "highres_denoise",
                    default=1.0,
                    min=0.01,
                    max=1.0,
                    step=0.01,
                    tooltip="Where pass 2 starts. 1.0 starts at the transition sigma. Lower starts pass 2 at a lower sigma: it changes the lifted video less.",
                ),
                io.Int.Input(
                    "noise_seed",
                    default=0,
                    min=0,
                    max=0xffffffffffffffff,
                    step=1,
                    control_after_generate=True,
                    tooltip="Pass 1 noise seed. Pass 2's fresh video noise uses the next seed.",
                ),
                io.Float.Input(
                    "lowres_scale",
                    default=0.5,
                    min=0.25,
                    max=1.0,
                    step=0.01,
                    tooltip="Pass 1 grid relative to the final grid. 0.5 is half size. 1.0 keeps the full grid.",
                ),
                io.Combo.Input(
                    "upscaler_model",
                    options=upscalers,
                    default=default_upscaler,
                    tooltip="Lift video with the learned H3 3D latent upscaler. none uses nearest-neighbor resizing. Audio is unchanged by the lift.",
                ),
                io.Float.Input(
                    "rho",
                    default=0.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="How much of the lift may come from the pixel route. 0 keeps the latent route only.",
                ),
                io.Float.Input(
                    "w_min",
                    default=0.5,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Lowest blend toward the pixel route inside the region the lift corrects.",
                ),
                io.Float.Input(
                    "w_max",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Highest blend toward the pixel route inside the region the lift corrects.",
                ),
                io.Latent.Input(
                    "low_res_head",
                    optional=True,
                    lazy=True,
                    tooltip="Previous scene's low_res_latent tail, video only, already on the low grid. Evaluated only when connected.",
                ),
                io.Sigmas.Input(
                    "sigmas_low",
                    optional=True,
                    tooltip="Replaces pass 1's computed sigmas. Leave it disconnected to use the schedule.",
                ),
                io.Sigmas.Input(
                    "sigmas_high",
                    optional=True,
                    tooltip="Replaces pass 2's computed sigmas. Leave it disconnected to use the schedule.",
                ),
                io.Model.Input(
                    "high_res_model",
                    optional=True,
                    tooltip="Pass 2 model, for example another fine-tune or its own LoRA stack. Pass 2 adds lora_high and its shifts to it. Leave it disconnected to use model.",
                ),
                io.Conditioning.Input(
                    "positive_low",
                    optional=True,
                    tooltip="Pass 1 positive, from a second MiniMaxH3ImageToVideo.positive at pass 1's size, as a native two-stage graph does. Leave it disconnected to use positive.",
                ),
                io.Conditioning.Input(
                    "negative_low",
                    optional=True,
                    tooltip="Pass 1 negative, built from positive_low. Used only when cfg_low is above 1. Leave it disconnected to use negative.",
                ),
            ],
            outputs=[
                io.Latent.Output(
                    "latent",
                    tooltip="Full-grid AV latent: pass 2 video and the audio finished in pass 2. Connect it to a save node, or to the next scene as latent_image.",
                ),
                io.Latent.Output(
                    "low_res_latent",
                    tooltip="Low-grid video from pass 1 with that same finished audio. Connect it to the next scene's low_res_head.",
                ),
            ],
        )

    @classmethod
    def check_lazy_status(cls, **kwargs):
        # Unconnected optional inputs are absent. None means connected but not evaluated yet.
        if "low_res_head" in kwargs and kwargs["low_res_head"] is None:
            return ["low_res_head"]
        return []

    @classmethod
    def execute(
        cls,
        model,
        positive,
        negative,
        vae,
        latent_image,
        lora_low,
        strength_low,
        shift_low,
        audio_shift_low,
        sampler_low,
        cfg_low,
        lora_high,
        strength_high,
        shift_high,
        audio_shift_high,
        sampler_high,
        cfg_high,
        scheduler,
        total_steps,
        transition,
        highres_denoise,
        noise_seed,
        lowres_scale,
        rho,
        w_min,
        w_max,
        low_res_head=None,
        sigmas_low=None,
        sigmas_high=None,
        high_res_model=None,
        upscaler_model="none",
        positive_low=None,
        negative_low=None,
    ):
        low_arm = {
            "lora": lora_low,
            "strength": strength_low,
            "shift": shift_low,
            "audio_shift": audio_shift_low,
            "sampler": _sampler_name(sampler_low),
            "cfg": cfg_low,
        }
        high_arm = {
            "lora": lora_high,
            "strength": strength_high,
            "shift": shift_high,
            "audio_shift": audio_shift_high,
            "sampler": _sampler_name(sampler_high),
            "cfg": cfg_high,
        }
        errors, warnings = check_values({
            "total_steps": total_steps,
            "transition": transition,
            "scheduler": scheduler,
            "passes": [low_arm, high_arm],
        })
        if errors:
            raise ValueError("Luna-H3-Sampler: " + "; ".join(errors))
        for warning in warnings:
            logging.warning("[Luna-H3-Sampler] %s", warning)

        total_steps = int(total_steps)
        transition = int(transition)
        denoise = None if float(highres_denoise) >= 1.0 else float(highres_denoise)
        loaded = {}
        low_model = _arm_model(model, lora_low, strength_low, shift_low, audio_shift_low, loaded)
        high_base = model if high_res_model is None else high_res_model
        high_model = _arm_model(high_base, lora_high, strength_high, shift_high, audio_shift_high, loaded)
        used_low = sigmas_low if sigmas_low is not None else arm_sigmas(
            low_model, scheduler, total_steps, transition, low=True)
        used_high = sigmas_high if sigmas_high is not None else arm_sigmas(
            high_model, scheduler, total_steps, transition, low=False, denoise=denoise)
        latent, low_latent = two_pass_sample(
            low_model,
            positive,
            negative,
            vae,
            latent_image,
            sampler_low,
            sampler_high,
            scheduler,
            total_steps,
            transition,
            int(noise_seed),
            float(cfg_low),
            float(cfg_high),
            float(lowres_scale),
            float(rho),
            float(w_min),
            float(w_max),
            high_res_model=high_model,
            low_res_head=low_res_head,
            highres_denoise=denoise,
            sigmas_low=sigmas_low,
            sigmas_high=sigmas_high,
            lift=lift_video,
            latent_lifter=(None if upscaler_model == "none" else
                           lambda z, hw: learned_latent_lift(z, hw, upscaler_model)),
            positive_low=positive_low,
            negative_low=negative_low,
        )
        text = _readout(low_arm, high_arm, used_low, used_high, total_steps, transition, denoise, warnings)
        text += "\npass 1 conditioning: " + ("positive" if positive_low is None else "positive_low")
        text += "\npass 2 model: " + ("model" if high_res_model is None else "high_res_model")
        text += f"\nupscaler: {upscaler_model}" + (" (nearest)" if upscaler_model == "none" else "")
        return io.NodeOutput(latent, low_latent, ui=ui.PreviewText(text))
