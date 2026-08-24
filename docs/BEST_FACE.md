# Best Face ID on MLX

The Best Face pipelines add character-reference conditioning to LTX-2.3. Use
`best_face_exact` for the closest match to the published character-sheet
workflow. Existing commands retain the official 8+3-step, native-reference,
CFG++ refinement defaults.

## Quality/parity workflow

```bash
uv run python -m ltx_pipelines_mlx.best_face_exact \
  --character-sheet \
  --reference character-sheet.png \
  --prompt "ref_t2v: A woman speaks naturally to the camera." \
  --frames 145 --frame-rate 24 \
  -H 1024 -W 576 \
  --seed 42 \
  -o best-face-quality.mp4
```

## Opt-in UGC fast workflow

Add `--ugc-fast` to target close and waist-up UGC shots where reduced runtime
is more important than exact sampler parity:

```bash
uv run python -m ltx_pipelines_mlx.best_face_exact \
  --character-sheet \
  --reference character-sheet.png \
  --prompt "ref_t2v: A woman speaks naturally to the camera. Static locked camera." \
  --ugc-fast \
  --frames 145 --frame-rate 24 \
  -H 1024 -W 576 \
  --seed 42 \
  -o best-face-ugc-fast.mp4
```

The preset applies:

- 6 Stage 1 steps instead of 8;
- 2 Stage 2 steps instead of 3;
- `0.5` reference scale in half-resolution Stage 1;
- native `1.0` reference scale in full-resolution Stage 2;
- single conditioned Stage 2 refinement instead of two-pass CFG++.

It is deliberately opt-in. Benchmark it against the quality workflow with the
same prompt, seed, reference, dimensions, and frame count before choosing it
for production. The intended target is roughly half the generation time while
retaining most perceived quality; actual speed and quality depend on framing,
clip length, and hardware.

Each part can also be selected independently:

```text
--stage1-steps 6
--stage2-steps 2
--stage1-reference-scale 0.5
--stage2-reference-scale 1.0
--fast-refine
```

`--reference-scale` remains supported and continues to set both stages when
neither stage-specific scale is supplied.
