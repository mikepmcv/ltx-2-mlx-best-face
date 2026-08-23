# Best Face ID on LTX-2 MLX

This fork adds a native-MLX inference path for `Alissonerdx/LTX-Best-Face-ID` on Apple Silicon.

The implementation uses the existing LTX-2.3 MLX model, VAE, distilled sampler, upsampler and LoRA loader. It adds the identity-specific inference mechanism that Best Face was trained with:

- reference image encoded with the LTX video VAE;
- reference latent appended as separate tokens (not used as generated frame 0);
- reference tokens kept clean at timestep 0;
- overlap T/H/W positions;
- source-phase/TASS-RoPE tagging (`source_id=2`, `phase_scale=1` by default);
- actual Best Face LoRA weights fused into the LTX transformer;
- reference tokens removed before upscaling/decoding.

The optional ArcFace projector is not part of v1. The Best Face model author describes it as marginal; the main identity signal comes from the overlap reference latent, source-phase RoPE and LoRA.

## First test

Use a clean, frontal, well-lit close-up/bust reference with one person and a clearly visible face.

```bash
git checkout feature/best-face-mlx
uv sync --all-extras

uv run python -m ltx_pipelines_mlx.best_face \
  --reference /absolute/path/to/host.png \
  --prompt "A person sits at a podcast desk speaking naturally toward the camera. Medium close-up, locked camera, subtle head motion and blinking, soft studio lighting, realistic skin texture." \
  --frames 49 \
  --frame-rate 24 \
  -H 576 -W 768 \
  --seed 42 \
  -o best-face-test.mp4
```

`ref_t2v:` is automatically prefixed if it is not already present.

The default generation path is LTX-2.3 distilled two-stage generation: normally 8 denoising steps at half resolution plus 3 full-resolution refinement steps.

## Important prompt detail

Best Face identity is strongly prompt-driven. For the closest match, include visible attributes from the reference near the beginning of the prompt: hair color/style, glasses, facial hair, face shape, eye color when clearly visible, and similar non-sensitive visual details.

Example:

```text
ref_t2v: A light-skinned adult person with short dark hair, brown eyes, rectangular glasses and light stubble sits at a podcast desk speaking naturally toward the camera. Medium close-up, locked camera, subtle head movement and blinking, soft studio lighting, realistic skin texture.
```

## Character-sheet mode

The fork also supports the Best Face character-sheet continuation checkpoint:

```bash
uv run python -m ltx_pipelines_mlx.best_face \
  --character-sheet \
  --reference /absolute/path/to/character-sheet.png \
  --prompt "A person sits at a podcast desk speaking naturally toward the camera. Medium close-up, locked camera." \
  --frames 49 \
  -H 576 -W 768 \
  --seed 42 \
  -o best-face-character-sheet.mp4
```

Character-sheet mode preserves the reference at its native resolution instead of shrinking it to the target video bucket. For Best Face's published character-sheet checkpoint, use the reference-sheet dimensions/layout recommended by its model card.

## Advanced controls

```text
--best-face-strength FLOAT
--source-id FLOAT          # default 2
--phase-scale FLOAT        # default 1
--resize-mode match_target|native_resolution
--reference-crf INT        # default 0; direct reference-to-VAE path
--stage1-steps INT
--stage2-steps INT
--extra-lora PATH STRENGTH # repeatable
```

## v1 scope

Included:

- Best Face LoRA loading;
- overlap reference conditioning;
- source-phase/TASS-RoPE;
- clean reference timesteps;
- two-stage distilled generation;
- close-up and character-sheet adapters;
- unit tests for source-phase math.

Not yet included:

- optional ArcFace projector;
- Best Face-specific multimodal prompt enhancer;
- external-audio A2V combined with Best Face;
- previous-video continuation for long-form generation;
- low-RAM block streaming with identity source-phase;
- multi-reference/strata/sidecar reference layouts.

The next milestone after visual identity parity is external-audio A2V + Best Face, followed by rolling video-prefix continuation for long-form podcast generation.
