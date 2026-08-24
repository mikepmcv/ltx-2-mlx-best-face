# LTX-2.5 on MLX

This branch adds the core LTX-2.5 T2V/I2V/A2V inference paths while preserving LTX-2.3 behavior.

## Model

Use an MLX conversion that contains the 2.5 transformer metadata and the tuned text encoder directory:

```text
mlx-community/ltx-2.5-mlx
```

The loader detects 2.5 from `use_keyframes_abs_pos_embedding` in the transformer checkpoint config. LTX-2.5 automatically uses the in-checkpoint `gemma4-12b-ltx-v1` text encoder; LTX-2.3 continues to use the configured Gemma 3 model.

## T2V / I2V example

```bash
ltx-2-mlx generate \
  --model mlx-community/ltx-2.5-mlx \
  --distilled \
  -H 720 -W 1280 \
  -p "A cinematic close-up of a presenter speaking naturally to camera" \
  -o ltx25.mp4
```

Image-to-video uses the same pipeline with `--image`.

## A2V example

```bash
ltx-2-mlx a2v \
  --model mlx-community/ltx-2.5-mlx \
  --audio speech.wav \
  -H 480 -W 704 -f 97 --frame-rate 24 \
  -p "A presenter speaking naturally to camera" \
  -o a2v.mp4
```

LTX-2.5 A2V uses `transformer-dev.safetensors` for guided stage 1 and swaps to `transformer-distilled.safetensors` for stage 2. The source audio is frozen through both 2.5 stages and the original input audio is muxed into the final video. LTX-2.3 keeps its existing dev + distilled-LoRA behavior.

## What is implemented

- LTX-2.5 checkpoint-driven transformer architecture deltas
- bias-free 2.5 video feed-forward layers
- keyframe absolute-position embedding
- Gemma 4 unified text encoding, including BOS/final-norm behavior
- fp32 diffusion timestep embedding for parity
- 2.5 first-frame keyframe mask
- ancestral Euler sampling for distilled T2V/I2V stage 1
- guided dev-model A2V stage 1 with CFG/STG and fp32 sigma
- full distilled-checkpoint A2V stage 2 (no legacy 2.3 distilled LoRA required)
- source-audio freezing across both 2.5 A2V stages
- existing convolutional VAE decode path
- existing `--low-ram` block streaming path

## Deferred

The first compatibility PR intentionally does not enable DiffVAE/DFR, generated multishot/keyframe-slot controls, or LTX-2.5 modality tiling. These can land independently after the core 2.5 paths are smoke-tested on Apple Silicon. LTX-2.5 currently raises a clear error if `--tile-frames` or `--tile-spatial` is requested.