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

Add `--ugc-fast` for the tested balance of identity, quality, and speed on
close and waist-up UGC shots:

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
- native `1.0` reference scale in both stages;
- single conditioned Stage 2 refinement instead of two-pass CFG++.

It is deliberately opt-in. In the initial 9:16 waist-up UGC tests it reduced
generation from roughly 50x real time to 20x while retaining good identity and
quality. Actual speed and quality depend on framing, clip length, and hardware.

For maximum speed, `--ugc-ultrafast` changes only the Stage 1 reference scale
to `0.5`. Initial testing reached roughly 15x real time but showed noticeably
more identity drift, so this preset is intended for previews and experiments:

```bash
uv run python -m ltx_pipelines_mlx.best_face_exact \
  --character-sheet \
  --reference character-sheet.png \
  --prompt "ref_t2v: A woman speaks naturally to the camera." \
  --ugc-ultrafast \
  --frames 145 --frame-rate 24 \
  -H 1024 -W 576 \
  --seed 42 \
  -o best-face-ugc-ultrafast.mp4
```

Each part can also be selected independently:

```text
--stage1-steps 6
--stage2-steps 2
--stage1-reference-scale 1.0
--stage2-reference-scale 1.0
--fast-refine
```

`--reference-scale` remains supported and continues to set both stages when
neither stage-specific scale is supplied.
