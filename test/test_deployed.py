import replicate
import os
import requests
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file in root directory
load_dotenv()

def generate_audio(description: str, duration: int = 1):
    """
    Generate audio using the deployed Replicate model
    
    Args:
        description (str): Text description of the audio to generate
        duration (int): Duration in seconds (default: 1)
    
    Returns:
        str: Path to the downloaded audio file
    """
    
    # Check if API token is set
    api_token = os.getenv("REPLICATE_API_TOKEN")
    if not api_token:
        print("Error: REPLICATE_API_TOKEN environment variable not set")
        print("Please create a .env file in the root directory with:")
        print("REPLICATE_API_TOKEN=your_token_here")
        print("Or set it with: export REPLICATE_API_TOKEN='your_token_here'")
        return None
    else:
        print(f"API token loaded successfully (first 10 chars: {api_token[:10]}...)")
    
    try:
        print(f"Generating {duration}-second audio for: '{description}'")
        print(f"Using model: mgysel/stable-audio-open:d90a0c38317e1c4316732753a632dbc9757f4bcae7c16b0128e85e457014da71")
        print("🔄 Starting audio generation (this may take a few minutes for the first run)...")
        
        # Use the model version directly
        prediction = replicate.predictions.create(
            version="mgysel/stable-audio-open:d90a0c38317e1c4316732753a632dbc9757f4bcae7c16b0128e85e457014da71",
            input={
                "description": description,
                "duration": duration
            }
        )
        
        print("⏳ Waiting for prediction to complete...")
        prediction.wait()
        output = prediction.output
        
        print("✅ Audio generation completed!")
        
        print(f"Model output: {output}")
        
        # The output is a URL to the generated audio file
        if output and isinstance(output, str):
            # Download the audio file
            response = requests.get(output)
            if response.status_code == 200:
                # Create output directory if it doesn't exist
                output_dir = Path("generated_audio")
                output_dir.mkdir(exist_ok=True)
                
                # Save the audio file
                filename = f"generated_audio_{duration}s.wav"
                output_path = output_dir / filename
                
                with open(output_path, "wb") as f:
                    f.write(response.content)
                
                print(f"Audio file saved to: {output_path.absolute()}")
                return str(output_path)
            else:
                print(f"Failed to download audio file: {response.status_code}")
                return None
        else:
            print("No output received from model")
            return None
            
    except Exception as e:
        print(f"Error generating audio: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Example usage
    description = "heavanly flowing pad"
    audio_file = generate_audio(description, duration=2)
    
    if audio_file:
        print(f"\n✅ Success! Audio file created at: {audio_file}")
    else:
        print("\n❌ Failed to generate audio file") 