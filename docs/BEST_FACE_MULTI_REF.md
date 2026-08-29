# Best Face multi-reference experiment

Branch: `feature/best-face-multi-ref`

This branch adds an experimental Best Face pipeline that accepts more than one identity image while leaving `feature/best-face-mlx` unchanged.

## Why

The current character-sheet workflow is good at overall identity, but talking-head generations can still lose fine mouth/teeth detail. Multi-reference conditioning lets us test whether a dedicated close-up and expression/teeth image can reinforce the character sheet.

Recommended first test set:

1. **Sharp face close-up** — primary reference; neutral expression and maximum eye/skin detail.
2. **Character sheet** — front/full-body/profile/back for overall identity, hair and proportions.
3. **Smile / visible-teeth reference** — clean frontal expression for mouth and dental detail.

Use 2–3 references initially. More reference tokens increase attention work and may reduce speed.

## CLI

```bash
uv run python -m ltx_pipelines_mlx.best_face_multi_ref \
  --character-sheet \
  --reference face-closeup.png \
  --reference character-sheet.png \
  --reference smile-teeth.png \
  --prompt "ref_t2v: A woman speaks naturally to the camera. Static locked camera. Natural subtle speech." \
  --ugc-fast \
  --frames 145 --frame-rate 24 \
  -H 1024 -W 576 \
  --seed 42 \
  -o best-face-multi-ref.mp4
```

`--reference` is repeatable. The first image is treated as the primary reference for backwards-compatible metadata, but all supplied images are independently VAE-encoded for identity conditioning.

## Implementation

For a single reference, `BestFaceMultiRefExactPipeline` delegates directly to the existing `BestFaceExactPipeline` identity-conditioning implementation. This keeps the existing single-image path unchanged.

For multiple references:

- each image is resized according to the existing Best Face reference rules;
- each image is independently encoded by the LTX VAE;
- each clean reference latent is converted to reference tokens;
- each token set keeps its normal overlapping frame-0 RoPE coordinates;
- the token and position sets are concatenated;
- the combined set is appended as one identity-conditioning segment;
- all references share the same source-phase identity because they represent the same character.

The denoising schedule, Best Face LoRAs, distilled LoRA, Stage 2 refinement, spatial upscaler and `--ugc-fast`/`--ugc-ultrafast` presets are inherited unchanged.

## Metadata

The normal sidecar JSON still includes `reference` for compatibility. This variant also adds:

- `references`
- `reference_count`
- `reference_strategy`

## What to compare

Use the same prompt, seed, duration and resolution for:

1. existing one-image character-sheet `--ugc-fast`;
2. multi-ref close-up + sheet;
3. multi-ref close-up + sheet + teeth/smile.

Score each result for:

- identity similarity;
- eye sharpness;
- teeth stability during speech;
- lip/mouth geometry;
- skin/hair micro-detail;
- temporal stability;
- total generation time.

The experiment should only be merged back into the main Best Face branch if the extra reference tokens give a visible identity/mouth-detail improvement that justifies their runtime cost.
