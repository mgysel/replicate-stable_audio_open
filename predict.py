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
            # fallback to model attrs if available
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

        # Coerce Nones coming from Replicate UI/API
        duration = int(duration or 8)
        steps = int(steps or 150)
        cfg_scale = float(cfg_scale or 7.0)
        sigma_min = float(sigma_min or 0.3)
        sigma_max = float(sigma_max or 500.0)
        sampler_type = (sampler_type or "dpmpp-3m-sde")
        seed = 0 if seed is None else int(seed)

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
                sample_size=self.sample_size,      # <- fixed window
                sigma_min=sigma_min,
                sigma_max=sigma_max,
                sampler_type=sampler_type,
                device=self.device,
                seed=None if seed == 0 else seed,  # None=random; same logic on both
            )

        # 2) Sanitize & shape
        audio = torch.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        # (B,C,T)->(C,T); (T,)->(1,T)
        if audio.ndim == 3:
            audio = audio[0]
        elif audio.ndim == 1:
            audio = audio.unsqueeze(0)

        # 3) Peak normalize with guard (do this or do nothing—just be consistent)
        if normalize:
            peak = torch.max(torch.abs(audio)).to(torch.float32)
            if torch.isfinite(peak) and peak > 0:
                audio = (audio.to(torch.float32) / peak).clamp(-1.0, 1.0)
            else:
                audio = audio.to(torch.float32).clamp(-1.0, 1.0)
        else:
            audio = audio.to(torch.float32)

        # 4) Trim (use the same rounding everywhere; I suggest floor)
        max_samples = int(self.sample_rate * float(duration))
        audio = audio[..., :max_samples]

        # 5) CPU + contiguous
        audio = audio.contiguous().cpu()

        # 6) Save with the same encoding on both sides (choose ONE)
        out = Path("/tmp/output.wav")
        if float32_wav:
            # Float32 (recommended)
            torchaudio.save(str(out), audio, sample_rate=self.sample_rate, format="wav")
        else:
            # If you prefer 16-bit, do this in BOTH places instead:
            int16 = (audio * 32767.0).round().clamp(-32768, 32767).to(torch.int16)
            torchaudio.save(str(out), int16, sample_rate=self.sample_rate, format="wav")

        return out
