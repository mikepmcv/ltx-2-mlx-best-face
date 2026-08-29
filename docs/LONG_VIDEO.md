# Best Face long videos with locked TTS

This workflow creates long talking-head videos as resumable short shots. Every
shot reuses the same Best Face identity reference and master scene frame. The
supplied TTS is frozen in the audio latent to drive facial motion. Model audio
is stripped from every shot; the final file contains only PCM copied from the
original TTS recording.

## Ten-minute portrait example

Prepare:

- `narration.wav`: the complete TTS recording (approximately 10 minutes);
- `face.png`: a clean frontal Best Face identity reference;
- `studio.png`: the exact presenter, framing, room and lighting for frame 0;
- optionally `presenter-mask.png`: white over the presenter/motion region and
  black over pixels that must remain identical to `studio.png`.

```bash
uv run python -m ltx_pipelines_mlx.long_video \
  --audio narration.wav \
  --reference face.png \
  --first-frame studio.png \
  --foreground-mask presenter-mask.png \
  --prompt "A presenter speaks naturally to camera with subtle expressions and restrained hand gestures." \
  --segment-seconds 8 \
  --quality fast \
  -H 1024 -W 576 \
  -o ten-minute-presenter.mp4
```

At 24 fps, the default eight-second plan uses 193-frame shots. A ten-minute
recording becomes about 75 independent generations. The final shot is padded
with silence only when required to reach an `8k + 1` frame count.

The default work directory is `ten-minute-presenter.mp4.work`. Completed audio,
raw, and visual segments are content-addressed and reused automatically when
the command is restarted. Changing an input file, prompt, seed, model setting,
or shot override produces new segment names instead of silently reusing stale
work.

## Background guarantees

Without `--foreground-mask`, the master first frame strongly anchors each
shot's opening composition, but LTX may still move or relight the background
during the shot.

With `--foreground-mask`, the generated presenter is composited over the master
frame after every shot. Black mask pixels therefore remain exactly static. Use
a soft mask covering the head, hair, shoulders, torso, and intended hand-motion
area. `--mask-feather 6` softens its edge; increase or decrease it as needed.

## Audio guarantees

For each shot the orchestrator:

1. copies the correct TTS interval to a 48 kHz PCM WAV;
2. freezes its encoded audio tokens through both Best Face denoising stages;
3. uses those tokens for joint audio/video lip movement;
4. strips the complete audio stream from the model output;
5. concatenates the original PCM intervals and muxes that track into the final
   MP4.

Consequently generated dialogue, ambience, music, or LTX audio-VAE output
cannot reach the finished file.

## One supplied-audio shot

The normal Best Face command also accepts a TTS recording:

```bash
uv run python -m ltx_pipelines_mlx.best_face \
  --reference face.png \
  --first-frame studio.png \
  --audio line.wav \
  --prompt "The presenter speaks naturally to camera." \
  --frames 193 --ugc-fast \
  -H 1024 -W 576 \
  -o line.mp4
```

Use `--audio-start` and `--audio-max-duration` to select a section of a longer
recording. Short input is padded with silence to the requested video duration.

## Shot overrides

An optional JSON manifest can change the prompt, master frame, mask, or seed for
specific generated segments. This supports intentional cuts between several
approved camera setups while keeping the TTS timeline automatic.

```json
{
  "segments": [
    {
      "index": 0,
      "prompt": "Medium shot. The presenter introduces the topic.",
      "first_frame": "studio-medium.png",
      "foreground_mask": "studio-medium-mask.png"
    },
    {
      "index": 4,
      "prompt": "Close-up. The presenter emphasizes the key point.",
      "first_frame": "studio-close.png",
      "foreground_mask": "studio-close-mask.png",
      "seed": 24004
    }
  ]
}
```

Pass it with `--shot-manifest shots.json`. Unlisted segments use the global
prompt, frame, mask, and incrementing seed.

## Memory and throughput

The long-video command keeps model components resident by default, which is
appropriate for a 128 GB Mac and avoids reloading the model for every shot. Use
`--low-memory` only when necessary; it reloads large components between shots
and can substantially increase total render time.

`--quality fast` uses the existing UGC-fast 6+2 preset. `standard` uses the
full Best Face schedule, while `ultrafast` reduces the Stage 1 reference scale
and may weaken identity.
