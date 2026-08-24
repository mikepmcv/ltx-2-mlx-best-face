# Best Face ID on LTX-2 MLX

This fork adds native-MLX inference for `Alissonerdx/LTX-Best-Face-ID` on Apple Silicon.

The port adds the identity-specific inference mechanism Best Face was trained with:

- reference image encoded with the normal LTX video VAE;
- reference latent appended as separate context tokens, not rendered frame-0 tokens;
- reference tokens kept clean at timestep 0;
- overlap T/H/W positions;
- source-phase/TASS-RoPE tagging (`source_id=2`, `phase_scale=1` by default);
- the actual Best Face LoRA weights fused into the LTX transformer;
- reference tokens removed before upscaling/decoding.

The optional ArcFace projector is not part of v1. Best Face's model card says it adds little on top of the overlap reference latent + LoRA. The ArcFace identity loss used during training is already reflected in the trained LoRA weights.

## Two test modes

There are two pipelines on the branch.

### 1. `best_face_exact` — run this first

This is the parity-first recipe. It mirrors the Best Face author's published fast demo as closely as the current MLX stack allows:

- LTX-2.3 dev transformer;
- official LTX-2.3 distilled-1.1 LoRA at strength 1.0;
- Best Face LoRA at strength 1.0;
- the distilled two-stage 8-step + 3-step schedule;
- native MLX overlap/source-phase reference conditioning.

Use a clean, frontal, well-lit close-up/bust reference with one person and a clearly visible face.

```bash
git checkout feature/best-face-mlx
uv sync --all-extras

uv run python -m ltx_pipelines_mlx.best_face_exact \
  --reference /absolute/path/to/host.png \
  --prompt "A person sits at a podcast desk speaking naturally toward the camera. Medium close-up, locked camera, subtle head motion and blinking, soft studio lighting, realistic skin texture." \
  --frames 49 \
  --frame-rate 24 \
  -H 576 -W 768 \
  --seed 42 \
  -o best-face-exact.mp4
```

`ref_t2v:` is automatically prefixed when absent.

### 2. `best_face` — compare speed after parity works

This uses the standalone LTX-2.3 distilled checkpoint plus the Best Face LoRA. It should be a convenient/faster path, but it is intentionally the second test because the author's published demo instead uses dev + the official distilled LoRA.

```bash
uv run python -m ltx_pipelines_mlx.best_face \
  --reference /absolute/path/to/host.png \
  --prompt "A person sits at a podcast desk speaking naturally toward the camera. Medium close-up, locked camera, subtle head motion and blinking, soft studio lighting, realistic skin texture." \
  --frames 49 \
  --frame-rate 24 \
  -H 576 -W 768 \
  --seed 42 \
  -o best-face-fast.mp4
```

Run both with the same reference, prompt, dimensions and seed. The useful comparison is identity quality versus wall-clock time.

## Important prompt detail

Best Face identity is strongly prompt-driven. For the closest match, put visible attributes from the reference near the beginning of the prompt: hair color/style, glasses, facial hair, face shape, eye color when clearly visible, and similar visual attributes.

Example:

```text
ref_t2v: An adult person with short dark hair, brown eyes, rectangular glasses and light stubble sits at a podcast desk speaking naturally toward the camera. Medium close-up, locked camera, subtle head movement and blinking, soft studio lighting, realistic skin texture.
```

## Character-sheet mode

The character-sheet continuation checkpoint is also supported. Its reference should be the four-panel Best Face sheet and should stay at native resolution.

```bash
uv run python -m ltx_pipelines_mlx.best_face_exact \
  --character-sheet \
  --reference /absolute/path/to/character-sheet.png \
  --reference-scale 0.5 \
  --prompt "A person sits at a podcast desk speaking naturally toward the camera. Medium close-up, locked camera." \
  --frames 49 \
  -H 576 -W 768 \
  --seed 42 \
  -o best-face-character-sheet.mp4
```

The published character-sheet checkpoint was trained with a wide four-panel sheet at its own fixed resolution, commonly 1536×1024. `--character-sheet` selects `native_resolution` automatically unless you explicitly override it. `--reference-scale 0.5` encodes that sheet at 768×512, reducing its reference token count by roughly 75%, while scaling its H/W positions back to preserve the original 1536×1024 positional span. The default scale is `1.0` for backward compatibility.

If character-sheet identity is weaker than desired, the Best Face model card suggests additionally mixing the base face LoRA at a low strength (around 0.2 or higher). That is available through `--extra-lora` and can be automated after the first parity test.

## Advanced controls

Common controls:

```text
--best-face-strength FLOAT
--source-id FLOAT          # default 2
--phase-scale FLOAT        # default 1
--resize-mode match_target|native_resolution
--reference-scale FLOAT    # (0, 1], default 1.0; lower uses fewer reference tokens
--reference-crf INT        # default 0; direct reference-to-VAE path
--stage1-steps INT
--stage2-steps INT
--extra-lora PATH STRENGTH # repeatable
```

`best_face_exact` also exposes:

```text
--distilled-lora PATH
--distilled-lora-strength FLOAT  # default 1.0
```

### First/last-frame guidance

Both Best Face commands can anchor the opening and closing composition with
images. The images are VAE-encoded at each generation stage and appended as
clean keyframe tokens alongside the identity reference:

```bash
--first-frame /absolute/path/to/opening.png \
--last-frame /absolute/path/to/ending.png \
--first-frame-strength 1.0 \
--last-frame-strength 0.9
```

Strengths must be between `0` and `1`. The first image is placed at pixel frame
`0`; the last image is placed at pixel frame `--frames - 1`. These controls
guide the endpoint compositions but do not copy pixels directly into the output.

When an endpoint should guide only subject placement and background layout—not
the person's face, texture, or color grade—use layout mode:

```bash
--first-frame /absolute/path/to/opening.png \
--first-frame-mode layout \
--first-frame-strength 0.3 \
--keyframe-layout-blur 32
```

Layout mode desaturates and spatially blurs the keyframe before VAE encoding.
This retains coarse silhouette, scale, screen position, and room geometry while
leaving character identity and facial features to the Best Face reference.

After a successful generation, a JSON sidecar is written beside the video as
`<output>.json`. It records the resolved seed, prompt, model and LoRAs, reference
and keyframe paths, strengths, dimensions, frame count, frame rate, and pipeline
settings needed to reproduce the run.

## v1 scope

Included:

- Best Face LoRA loading;
- overlap reference conditioning;
- source-phase/TASS-RoPE;
- clean reference timesteps;
- parity-first dev + official distilled LoRA recipe;
- standalone-distilled comparison path;
- close-up and character-sheet adapters;
- identity reference injection in both generation stages;
- unit tests for source-phase math.

Not yet included:

- optional ArcFace projector;
- Best Face-specific multimodal prompt enhancer;
- reference-CFG extra pass;
- external-audio A2V combined with Best Face;
- previous-video continuation for long-form generation;
- low-RAM block streaming with identity source-phase;
- multi-reference/strata/sidecar layouts.

The next milestone after visual identity parity is external-audio A2V + Best Face, followed by rolling video-prefix continuation for long-form podcast generation.
