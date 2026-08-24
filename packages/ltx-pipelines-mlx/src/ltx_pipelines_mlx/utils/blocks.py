"""Composable pipeline blocks.

Mirrors upstream ``ltx_pipelines.utils.blocks`` (composition over
inheritance). Each block owns the lifecycle of one model component
(load, use, free) and exposes a small ``__call__`` API.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

import mlx.core as mx

from ltx_core_mlx.model.audio_vae.audio_vae import AudioVAEDecoder
from ltx_core_mlx.model.audio_vae.bwe import VocoderWithBWE
from ltx_core_mlx.model.upsampler.model import LatentUpsampler
from ltx_core_mlx.model.video_vae.video_vae import VideoDecoder as _VideoVAEDecoder
from ltx_core_mlx.model.video_vae.video_vae import VideoEncoder as _VideoVAEEncoder
from ltx_core_mlx.model.video_vae.video_vae import _compute_decode_tiling
from ltx_core_mlx.text_encoders.gemma.encoders.base_encoder import GemmaLanguageModel
from ltx_core_mlx.text_encoders.gemma.feature_extractor import GemmaFeaturesExtractorV2
from ltx_core_mlx.utils.memory import aggressive_cleanup
from ltx_core_mlx.utils.weights import load_split_safetensors, remap_audio_vae_keys

_materialize = getattr(mx, "eval")  # noqa: B009


def _resolve_model_dir(model_dir: str | Path) -> Path:
    path = Path(model_dir)
    if path.exists():
        return path
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(str(model_dir)))


class PromptEncoder:
    """Owns Gemma + connector lifecycle and selects Gemma 3/4 from the checkpoint."""

    def __init__(
        self,
        model_dir: str | Path,
        gemma_model_id: str = "mlx-community/gemma-3-12b-it-4bit",
    ) -> None:
        self.model_dir = _resolve_model_dir(model_dir)
        self.gemma_model_id = gemma_model_id
        self._text_encoder: GemmaLanguageModel | None = None
        self._feature_extractor: GemmaFeaturesExtractorV2 | None = None

    def load(self) -> None:
        if self._text_encoder is None:
            from ltx_core_mlx.text_encoders.gemma.encoders.gemma4_encoder import resolve_text_encoder

            self._text_encoder = resolve_text_encoder(self.model_dir, self.gemma_model_id)
            self._text_encoder.load()
            aggressive_cleanup()

        if self._feature_extractor is None:
            self._feature_extractor = GemmaFeaturesExtractorV2()
            connector_weights = load_split_safetensors(self.model_dir / "connector.safetensors", prefix="connector.")
            self._feature_extractor.connector.load_weights(list(connector_weights.items()))
            aggressive_cleanup()

    def free(self) -> None:
        self._text_encoder = None
        self._feature_extractor = None
        aggressive_cleanup()

    def encode(self, prompt: str) -> tuple[mx.array, mx.array]:
        import os

        self.load()
        assert self._text_encoder is not None
        assert self._feature_extractor is not None
        max_length = int(os.environ.get("LTX2_GEMMA_MAX_LENGTH", "1024"))
        all_hidden_states, attention_mask = self._text_encoder.encode_all_layers(prompt, max_length=max_length)
        return self._feature_extractor(all_hidden_states, attention_mask=attention_mask)

    def __call__(
        self,
        prompts: str | list[str],
        *,
        free_after: bool = True,
    ) -> tuple[mx.array, mx.array] | list[tuple[mx.array, mx.array]]:
        if isinstance(prompts, str):
            video, audio = self.encode(prompts)
            _materialize(video, audio)
            if free_after:
                self.free()
            return video, audio

        outputs: list[tuple[mx.array, mx.array]] = []
        for p in prompts:
            video, audio = self.encode(p)
            _materialize(video, audio)
            outputs.append((video, audio))
        if free_after:
            self.free()
        return outputs


class ImageConditioner:
    def __init__(self, model_dir: str | Path) -> None:
        self.model_dir = _resolve_model_dir(model_dir)
        self._encoder: _VideoVAEEncoder | None = None

    def load(self) -> _VideoVAEEncoder:
        if self._encoder is not None:
            return self._encoder
        self._encoder = _VideoVAEEncoder()
        weights = load_split_safetensors(self.model_dir / "vae_encoder.safetensors", prefix="vae_encoder.")
        weights = {
            k.replace("._mean_of_means", ".mean_of_means").replace("._std_of_means", ".std_of_means"): v
            for k, v in weights.items()
        }
        self._encoder.load_weights(list(weights.items()))
        aggressive_cleanup()
        return self._encoder

    def free(self) -> None:
        self._encoder = None
        aggressive_cleanup()

    def __call__(self, fn: Callable[[_VideoVAEEncoder], object], *, free_after: bool = True) -> object:
        encoder = self.load()
        result = fn(encoder)
        if free_after:
            self.free()
        return result


class VideoDecoder:
    def __init__(self, model_dir: str | Path, verbose: bool = True) -> None:
        self.model_dir = _resolve_model_dir(model_dir)
        self.verbose = verbose
        self._decoder: _VideoVAEDecoder | None = None

    def load(self) -> _VideoVAEDecoder:
        if self._decoder is not None:
            return self._decoder
        self._decoder = _VideoVAEDecoder()
        weights = load_split_safetensors(self.model_dir / "vae_decoder.safetensors", prefix="vae_decoder.")
        self._decoder.load_weights(list(weights.items()))
        aggressive_cleanup()
        return self._decoder

    def free(self) -> None:
        self._decoder = None
        aggressive_cleanup()

    def decode_and_stream(
        self,
        video_latent: mx.array,
        output_path: str,
        frame_rate: float = 24.0,
        audio_path: str | None = None,
    ) -> str:
        if self.verbose:
            tiling = _compute_decode_tiling(video_latent.shape, frame_rate=frame_rate)
            if tiling is not None and tiling.temporal_config is not None:
                tc = tiling.temporal_config
                print(
                    f"[vae-decode tiled: tile_frames={tc.tile_size_in_frames} overlap={tc.tile_overlap_in_frames}]",
                    file=sys.stderr,
                    flush=True,
                )
        decoder = self.load()
        decoder.decode_and_stream(video_latent, output_path, frame_rate=frame_rate, audio_path=audio_path)
        return output_path


class AudioDecoder:
    def __init__(self, model_dir: str | Path) -> None:
        self.model_dir = _resolve_model_dir(model_dir)
        self._audio_decoder: AudioVAEDecoder | None = None
        self._vocoder: VocoderWithBWE | None = None

    def load(self) -> tuple[AudioVAEDecoder, VocoderWithBWE]:
        if self._audio_decoder is None:
            self._audio_decoder = AudioVAEDecoder()
            decoder_weights = load_split_safetensors(
                self.model_dir / "audio_vae.safetensors", prefix="audio_vae.decoder."
            )
            all_audio = load_split_safetensors(self.model_dir / "audio_vae.safetensors", prefix="audio_vae.")
            for k, v in all_audio.items():
                if k.startswith("per_channel_statistics."):
                    decoder_weights[k] = v
            decoder_weights = remap_audio_vae_keys(decoder_weights)
            self._audio_decoder.load_weights(list(decoder_weights.items()))
            aggressive_cleanup()

        if self._vocoder is None:
            self._vocoder = VocoderWithBWE()
            vocoder_weights = load_split_safetensors(self.model_dir / "vocoder.safetensors", prefix="vocoder.")
            self._vocoder.load_weights(list(vocoder_weights.items()))
            self._vocoder.upcast_weights_to_fp32()
            aggressive_cleanup()

        return self._audio_decoder, self._vocoder

    def free(self) -> None:
        self._audio_decoder = None
        self._vocoder = None
        aggressive_cleanup()

    def __call__(self, audio_latent: mx.array) -> mx.array:
        decoder, vocoder = self.load()
        mel = decoder.decode(audio_latent)
        return vocoder(mel)


class AudioConditioner:
    def __init__(self, model_dir: str | Path) -> None:
        self.model_dir = _resolve_model_dir(model_dir)
        self._encoder: object | None = None
        self._processor: object | None = None

    def load(self) -> tuple[object, object]:
        if self._encoder is not None and self._processor is not None:
            return self._encoder, self._processor
        from ltx_core_mlx.model.audio_vae import AudioProcessor, AudioVAEEncoder

        self._encoder = AudioVAEEncoder()
        encoder_weights = load_split_safetensors(self.model_dir / "audio_vae.safetensors", prefix="audio_vae.encoder.")
        all_audio = load_split_safetensors(self.model_dir / "audio_vae.safetensors", prefix="audio_vae.")
        for k, v in all_audio.items():
            if k.startswith("per_channel_statistics."):
                encoder_weights[k] = v
        encoder_weights = remap_audio_vae_keys(encoder_weights)
        self._encoder.load_weights(list(encoder_weights.items()))
        self._processor = AudioProcessor()
        aggressive_cleanup()
        return self._encoder, self._processor

    def free(self) -> None:
        self._encoder = None
        self._processor = None
        aggressive_cleanup()

    def __call__(self, fn: Callable[[object, object], object], *, free_after: bool = True) -> object:
        encoder, processor = self.load()
        result = fn(encoder, processor)
        if free_after:
            self.free()
        return result


class VideoUpsampler:
    def __init__(
        self,
        model_dir: str | Path,
        name: str = "spatial_upscaler_x2_v1_1",
    ) -> None:
        self.model_dir = _resolve_model_dir(model_dir)
        self.name = name
        self._upsampler: LatentUpsampler | None = None

    def load(self) -> LatentUpsampler:
        if self._upsampler is not None:
            return self._upsampler
        import json

        config_path = self.model_dir / f"{self.name}_config.json"
        weights_path = self.model_dir / f"{self.name}.safetensors"
        if config_path.exists():
            config = json.loads(config_path.read_text()).get("config", {})
            self._upsampler = LatentUpsampler.from_config(config)
        else:
            self._upsampler = LatentUpsampler()
        if weights_path.exists():
            weights = load_split_safetensors(weights_path, prefix=f"{self.name}.")
            self._upsampler.load_weights(list(weights.items()))
        aggressive_cleanup()
        return self._upsampler

    def free(self) -> None:
        self._upsampler = None
        aggressive_cleanup()

    def __call__(self, latent: mx.array) -> mx.array:
        return self.load()(latent)


__all__ = [
    "AudioConditioner",
    "AudioDecoder",
    "ImageConditioner",
    "PromptEncoder",
    "VideoDecoder",
    "VideoUpsampler",
]
