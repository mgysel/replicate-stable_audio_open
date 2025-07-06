import cog
import os
from stable_audio_tools import get_pretrained_model
from pathlib import Path
from huggingface_hub import login

class Predictor:
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        # Try to authenticate with Hugging Face if token is available
        hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
        if hf_token:
            try:
                login(token=hf_token)
                print("Successfully authenticated with Hugging Face")
            except Exception as e:
                print(f"Warning: Could not authenticate with Hugging Face: {e}")
        
        self.model, self.cfg = get_pretrained_model("stabilityai/stable-audio-open-1.0")

    def predict(self, description: str, duration: int = 8) -> Path:
        """Generate audio from text description"""
        audio = self.model.sample(
            description=description,
            seconds_total=duration,
            cfg=self.cfg,
        )
        output_path = Path("output.wav")
        audio.write_wav(str(output_path))
        return output_path 