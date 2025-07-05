import cog
from stable_audio_tools import get_pretrained_model
from pathlib import Path

class Predictor(cog.Predictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.model, self.cfg = get_pretrained_model("stabilityai/stable-audio-open-1.0")

    @cog.input("description", type=str, help="Description of the audio to generate")
    @cog.input("duration", type=int, default=8, help="Duration of the audio in seconds")
    def predict(self, description: str, duration: int) -> Path:
        """Generate audio from text description"""
        
        audio = self.model.sample(
            description=description,
            seconds_total=duration,
            cfg=self.cfg,
        )
        
        output_path = Path("output.wav")
        audio.write_wav(str(output_path))
        
        return output_path 