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


def _coerce_to_channels_first(audio: torch.Tensor) -> torch.Tensor:
    """
    Return tensor as (channels, frames).
    Accepts (B,C,T), (C,T), (T,), (B,T). Picks first batch if present.
    """
    if audio.ndim == 3:
        # (B, C, T) -> take first batch -> (C, T)
        return audio[0]
    elif audio.ndim == 2:
        # Could be (C, T) or (B, T). If first dim is small (<= 8), assume channels.
        d0, d1 = audio.shape
        if d0 <= 8 and d1 >= 1024:
            return audio
        # Else treat as (B, T) -> first batch -> (1, T)
        return audio[:1, :]
    elif audio.ndim == 1:
        # (T,) -> (1, T)
        return audio.unsqueeze(0)
    else:
        raise ValueError(f"Unexpected audio tensor shape: {audio.shape}")


def _peak_normalize(audio: torch.Tensor) -> torch.Tensor:
    """
    Peak-normalize to [-1, 1] in float32 with finite guard.
    Expects (C, T); returns float32.
    """
    audio = torch.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    audio = audio.to(torch.float32)
    peak = torch.max(torch.abs(audio))
    if torch.isfinite(peak) and peak > 0:
        audio = (audio / peak).clamp(-1.0, 1.0)
    else:
        audio = audio.clamp(-1.0, 1.0)
    return audio


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

        # Safe model config handling with fallbacks
        sr = self.model_config.get("sample_rate")
        ss = self.model_config.get("sample_size")
        if sr is None or ss is None:
            sr = sr if sr is not None else getattr(self.model, "sample_rate", None)
            ss = ss if ss is not None else getattr(self.model, "sample_size", None)
        if sr is None or ss is None:
            raise RuntimeError("Model config missing sample_rate/sample_size.")
        self.sample_rate = int(sr)
        self.sample_size = int(ss)

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
            description=(
                "Requested length of the generated audio in seconds "
                "(will be trimmed if beyond the model window)."
            ),
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
            default=False,
            description="Write WAV as 32-bit float (True) or 16-bit PCM (False)."
        ),
    ) -> Path:
        """
        Correct usage for Stable Audio Open:
          - Keep model_config['sample_size'] fixed.
          - Control timing via 'conditioning' seconds and then trim/pad as desired.
        """

        # Coerce Nones coming from Replicate UI/API
        duration = int(duration or 8)
        steps = int(steps or 100)
        cfg_scale = float(cfg_scale or 7.0)
        sigma_min = float(sigma_min or 0.3)
        sigma_max = float(sigma_max or 500.0)
        sampler_type = (sampler_type or "dpmpp-3m-sde")

        # Handle seed - coerce None to 0 (which means random)
        try:
            seed_int = int(seed) if seed is not None else 0
        except (TypeError, ValueError):
            seed_int = 0

        print({"resolved_seed": seed_int, "requested_seconds": duration})

        # 1) Generate (keep model window; control time via conditioning)
        with torch.inference_mode():
            audio = generate_diffusion_cond(
                model=self.model,
                steps=steps,
                cfg_scale=cfg_scale,
                conditioning=[{
                    "prompt": description,
                    "seconds_start": 0,
                    "seconds_total": float(duration),
                }],
                sample_size=self.sample_size,  # fixed window
                sigma_min=sigma_min,
                sigma_max=sigma_max,
                sampler_type=sampler_type,
                device=self.device,
                seed=seed_int,  # never None
            )

        # 2) Shape -> (C, T)
        audio = _coerce_to_channels_first(audio)

        # 3) Normalize or at least sanitize
        if normalize:
            audio = _peak_normalize(audio)
        else:
            audio = torch.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
            audio = audio.to(torch.float32).clamp(-1.0, 1.0)

        # 4) Trim to requested seconds
        max_samples = int(self.sample_rate * float(duration))
        if audio.size(-1) > max_samples:
            audio = audio[..., :max_samples]
        # If you prefer exact-length files even when shorter, uncomment to pad:
        # else:
        #     pad = max_samples - audio.size(-1)
        #     if pad > 0:
        #         audio = torch.nn.functional.pad(audio, (0, pad))

        # 5) Move to CPU, make contiguous
        audio = audio.contiguous().cpu()

        # Debug log for sanity
        print({
            "shape": tuple(audio.shape),
            "dtype": str(audio.dtype),
            "min": float(audio.min()) if audio.numel() > 0 else 0.0,
            "max": float(audio.max()) if audio.numel() > 0 else 0.0,
            "sample_rate": self.sample_rate,
            "seconds": (audio.size(-1) / float(self.sample_rate)) if audio.numel() > 0 else 0.0,
            "encoding": "PCM_F32LE" if float32_wav else "PCM_S16LE",
        })

        # 6) Save WAV (explicit encoding)
        out = Path("/tmp/output.wav")
        if float32_wav:
            # Lossless float; some previewers may be picky
            torchaudio.save(
                str(out),
                audio,
                sample_rate=self.sample_rate,
                format="wav",
                encoding="PCM_F32LE",
            )
        else:
            # Safest cross-player choice
            torchaudio.save(
                str(out),
                audio,
                sample_rate=self.sample_rate,
                format="wav",
                encoding="PCM_S16LE",
            )

        return out
