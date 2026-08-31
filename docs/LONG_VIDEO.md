# Best Face long videos with locked TTS

This workflow creates long talking-head videos as resumable short shots. Every
shot reuses the same Best Face identity reference and master scene frame. The
supplied TTS is frozen in the audio latent to drive facial motion. Model audio
is stripped from every shot; the original continuous TTS recording is muxed
directly into the final video so segment processing cannot add audio gaps.

By default every segment restarts from the master scene frame, receives 16
frames of hidden audio/video preroll, discards that preroll, and joins the
previous segment with a hard creator-style cut. This avoids cumulative
softening from repeatedly feeding generated end frames back into the model,
while letting the mouth begin moving before each visible cut.

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
  --segment-handoff master \
  --segment-preroll-frames 16 \
  --transition hard \
  --preencode-audio \
  --quality fast \
  -H 1024 -W 576 \
  -o ten-minute-presenter.mp4
```

At 24 fps, the default eight-second plan uses 193-frame shots. A ten-minute
recording becomes about 75 independent generations. Later shots include a
hidden 16-frame preroll. Each generated conditioning window is padded only
when required to reach an `8k + 1` frame count, then trimmed back to the exact
source timeline before assembly.

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

1. copies the TTS interval plus any hidden preroll to a 48 kHz PCM WAV;
2. freezes its encoded audio tokens through both Best Face denoising stages;
3. uses those tokens for joint audio/video lip movement;
4. strips the complete audio stream from the model output;
5. muxes the original, unsegmented source recording directly into the final
   MP4 timeline.

Consequently generated dialogue, ambience, music, or LTX audio-VAE output
cannot reach the finished file.

## Segment continuity

The default `--segment-handoff master --transition hard` mode is intended for
long UGC and YouTube-style speech. It trades tiny pose changes at cuts for
stable long-term sharpness: every segment starts from the same clean master
frame rather than a progressively softer generated frame.

`--segment-preroll-frames 16` begins conditioning roughly 0.67 seconds before
each cut at 24 fps, then removes those 16 frames exactly. The visible segment
therefore starts at the correct audio timestamp with speech already in motion,
instead of repeatedly revealing the master frame's closed mouth.

For the older continuous-handoff behavior, use `--segment-handoff previous`.
Add `--transition fade` if a short visual blend is preferred, although that
mode can accumulate softness over a long video.

## Warm generation and latent upscaling

The normal long-video mode keeps the transformer, VAE encoder, official LTX
spatial latent upscaler, decoders, and reusable conditioning in one pipeline
instance across segments. Do not add `--low-memory` on a 128 GB machine: that
flag deliberately unloads large components between shots and is slower.

`--preencode-audio` is enabled by default. Before the first missing segment is
generated, all TTS conditioning windows are encoded and retained in memory.
The character sheet is also VAE-encoded only once at each required stage
resolution and reused across the run. Both caches are in-memory performance
optimizations; the existing content-addressed WAV/video cache remains the
restart mechanism.

Best Face is already a two-stage latent-upscale workflow: Stage 1 generates at
half resolution, the official LTX spatial upscaler enlarges the latent, and
Stage 2 refines it at output resolution. There is therefore no separate final
MP4 latent-upscale switch. Use `--quality balanced` for three Stage 2 detail
passes when eyes and teeth matter; `fast` uses the quicker two-pass refine.

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
