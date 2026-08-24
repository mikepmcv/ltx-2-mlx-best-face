# LTX-2.5 on MLX

This branch adds the core LTX-2.5 distilled two-stage inference path while preserving LTX-2.3 behavior.

## Model

Use an MLX conversion that contains the 2.5 transformer metadata and the tuned text encoder directory:

```text
mlx-community/ltx-2.5-mlx
```

The loader detects 2.5 from `use_keyframes_abs_pos_embedding` in the transformer checkpoint config. LTX-2.5 automatically uses the in-checkpoint `gemma4-12b-ltx-v1` text encoder; LTX-2.3 continues to use the configured Gemma 3 model.

## Example

```bash
ltx-2-mlx generate \
  --model mlx-community/ltx-2.5-mlx \
  --distilled \
  -H 720 -W 1280 \
  -p "A cinematic close-up of a presenter speaking naturally to camera" \
  -o ltx25.mp4
```

Image-to-video uses the same pipeline with `--image`.

## What is implemented

- LTX-2.5 checkpoint-driven transformer architecture deltas
- bias-free 2.5 video feed-forward layers
- keyframe absolute-position embedding
- Gemma 4 unified text encoding, including BOS/final-norm behavior
- fp32 diffusion timestep embedding for parity
- 2.5 first-frame keyframe mask
- ancestral Euler sampling for distilled stage 1
- existing convolutional VAE decode path
- existing `--low-ram` block streaming path

## Deferred

The first compatibility PR intentionally does not enable DiffVAE/DFR or LTX-2.5 modality tiling. These can land independently after the core 2.5 path is smoke-tested on Apple Silicon. LTX-2.5 currently raises a clear error if `--tile-frames` or `--tile-spatial` is requested.
