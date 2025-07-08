# --- predict.py --------------------------------------------------------------
# removed: from __future__ import annotations

import os
import torch
import torchaudio
from pathlib import Path
from einops import rearrange
from cog import BasePredictor, Input
from dotenv import load_dotenv
from huggingface_hub import login
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond


class Predictor(BasePredictor):
    def setup(self):
        load_dotenv(override=False)               # local .env convenience
        token = (os.getenv("HUGGING_FACE_HUB_TOKEN")
                 or os.getenv("HF_TOKEN"))
        if not token:
            raise RuntimeError(
                "Set HUGGING_FACE_HUB_TOKEN (or HF_TOKEN) "
                "in the env or in a .env file."
            )

        login(token=token)
        self.model, self.model_config = get_pretrained_model(
            "stabilityai/stable-audio-open-1.0"
        )
        self.sample_rate = self.model_config["sample_rate"]
        self.sample_size = self.model_config["sample_size"]
        
        # Set device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)

    def predict(
        self,
        description: str = Input(
            description="Text prompt for the audio"),
        duration: int = Input(
            default=8, ge=1, le=120,
            description="Length of the generated audio in seconds"),
    ) -> Path:
        # Set up text and timing conditioning
        conditioning = [{
            "prompt": description,
            "seconds_start": 0, 
            "seconds_total": duration
        }]

        # Calculate sample size based on requested duration
        target_sample_size = int(duration * self.sample_rate)
        
        # Generate stereo audio
        output = generate_diffusion_cond(
            self.model,
            steps=100,
            cfg_scale=7,
            conditioning=conditioning,
            sample_size=target_sample_size,
            sigma_min=0.3,
            sigma_max=500,
            sampler_type="dpmpp-3m-sde",
            device=self.device
        )

        # Rearrange audio batch to a single sequence
        output = rearrange(output, "b d n -> d (b n)")

        # Peak normalize, clip, convert to int16, and save to file
        output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
        
        out = Path("output.wav")
        torchaudio.save(str(out), output, self.sample_rate)
        return out
