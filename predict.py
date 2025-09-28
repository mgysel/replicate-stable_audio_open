# --- predict.py --------------------------------------------------------------
# Cog predictor for Stable Audio Open on Replicate

import os
from pathlib import Path

import torch
import torchaudio
from cog import BasePredictor, Input
from dotenv import load_dotenv
from huggingface_hub import login

from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond


def _safe_peak_normalize(x: torch.Tensor) -> torch.Tensor:
    """Peak-normalize to [-1, 1] with zero-guard in float32 on CPU."""
    x = x.to(torch.float32)
    peak = torch.max(torch.abs(x))
    if torch.isfinite(peak) and peak > 0:
        x = x / peak
    return x.clamp(-1.0, 1.0).cpu()


def _to_channels_first(audio: torch.Tensor) -> torch.Tensor:
    """
    Ensure tensor is (channels, samples) for torchaudio.save.
    Accepts shapes:
      (B, C, T) -> take first batch -> (C, T)
      (C, T)    -> as is
      (T,)      -> mono -> (1, T)
    """
    if audio.ndim == 3:
        # (batch, channels, samples)
        audio = audio[0]
    elif audio.ndim == 1:
        audio = audio.unsqueeze(0)
    # now (C, T)
    return audio


def _trim_to_seconds(audio_ct: torch.Tensor, sr: int, seconds: float) -> torch.Tensor:
    """Trim channels-first audio to exact number of samples for given seconds."""
    max_samples = int(round(seconds * sr))
    return audio_ct[..., :max_samples]


class Predictor(BasePredictor):
    def setup(self):
        load_dotenv(override=False)

        token = os.getenv("HUGGING_FACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
        if not token:
            raise RuntimeError(
                "Set HUGGING_FACE_HUB_TOKEN (or HF_TOKEN) in the environment or .env."
            )

        login(token=token)

        # Load model + config
        self.model, self.model_config = get_pretrained_model(
            "stabilityai/stable-audio-open-1.0"
        )
        self.sample_rate: int = int(self.model_config["sample_rate"])
        # IMPORTANT: sample_size is the model's latent/audio window size (in samples)
        self.sample_size: int = int(self.model_config["sample_size"])

        # Device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)

        # Precompute max seconds the model window supports
        self.max_seconds = self.sample_size / float(self.sample_rate)

    def predict(
        self,
        description: str = Input(description="Text prompt for the audio."),
        duration: int = Input(
            default=8, ge=1, le=120,
            description="Requested length of the generated audio in seconds "
                        "(will be trimmed if beyond the model window).",
        ),
        seed: int = Input(default=0, description="Random seed (0 = random)."),
        steps: int = Input(default=100, ge=10, le=200, description="Diffusion steps."),
        cfg_scale: float = Input(default=7.0, ge=1.0, le=20.0, description="CFG scale."),
        sigma_min: float = Input(default=0.3, ge=0.01, le=10.0, description="Min sigma."),
        sigma_max: float = Input(default=500.0, ge=1.0, le=1000.0, description="Max sigma."),
        sampler_type: str = Input(
            default="dpmpp-3m-sde",
            description="Sampler type (e.g., 'dpmpp-3m-sde').",
        ),
        normalize: bool = Input(
            default=True,
            description="Peak-normalize before writing WAV."
        ),
        float32_wav: bool = Input(
            default=True,
            description="Write WAV as 32-bit float (True) or 16-bit PCM (False)."
        ),
    ) -> Path:
        """
        Correct usage for Stable Audio Open:
          - Keep model_config['sample_size'] fixed.
          - Control timing via 'conditioning' seconds and then trim.
        """

        # Build conditioning over the requested duration.
        conditioning = [{
            "prompt": description,
            "seconds_start": 0,
            "seconds_total": float(duration),
        }]

        # NOTE: Do NOT override sample_size with duration-derived value.
        # The generator returns up to `self.sample_size` samples; we trim afterward.
        gen_kwargs = dict(
            model=self.model,
            steps=int(steps),
            cfg_scale=float(cfg_scale),
            conditioning=conditioning,
            sample_size=self.sample_size,      # CRITICAL: use model window size
            sigma_min=float(sigma_min),
            sigma_max=float(sigma_max),
            sampler_type=sampler_type,
            device=self.device,
            seed=None if seed == 0 else int(seed),
        )

        with torch.inference_mode():
            audio = generate_diffusion_cond(**gen_kwargs)
            # audio expected shape: (B, C, T) or (C, T)

        # Ensure (C, T)
        audio_ct = _to_channels_first(audio)

        # Optional peak normalize with guard
        if normalize:
            audio_ct = _safe_peak_normalize(audio_ct)

        # Trim to requested duration (never pad; users usually prefer exact or shorter)
        audio_ct = _trim_to_seconds(audio_ct, self.sample_rate, float(duration))

        # Choose dtype / subtype
        out = Path("/tmp/output.wav")  # Replicate/Cog safe path

        if float32_wav:
            # torchaudio.save will write PCM_F32 when given float32 tensor
            torchaudio.save(str(out), audio_ct, sample_rate=self.sample_rate, format="wav")
        else:
            # 16-bit PCM path
            int16 = (audio_ct * 32767.0).round().clamp(-32768, 32767).to(torch.int16)
            torchaudio.save(str(out), int16, sample_rate=self.sample_rate, format="wav")

        return out
