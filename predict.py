import cog
import os
from stable_audio_tools import get_pretrained_model
from pathlib import Path
from huggingface_hub import login

class Predictor:
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        # Get Hugging Face token from environment
        hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
        
        if not hf_token:
            print("Warning: HUGGING_FACE_HUB_TOKEN not found in environment variables")
            print("This may cause issues with accessing the gated model")
        else:
            try:
                # Log into Hugging Face
                login(token=hf_token)
                print(f"Successfully authenticated with Hugging Face (token: {hf_token[:10]}...)")
            except Exception as e:
                print(f"Warning: Could not authenticate with Hugging Face: {e}")
        
        print("Loading stable-audio-open model...")
        # The get_pretrained_model function should now work with the authenticated session
        self.model, self.cfg = get_pretrained_model("stabilityai/stable-audio-open-1.0")
        print("Model loaded successfully!")

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