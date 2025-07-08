import replicate
import os
import requests
import time
import signal
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file in root directory
load_dotenv()

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Request timed out")

def generate_audio_with_timeout(description: str, duration: int = 1, timeout_seconds: int = 600):
    """
    Generate audio using the deployed Replicate model with timeout
    
    Args:
        description (str): Text description of the audio to generate
        duration (int): Duration in seconds (default: 1)
        timeout_seconds (int): Timeout in seconds (default: 600)
    
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
        print(f"Using model: mgysel/stable-audio-open:8465bfb2f7a77991f33e26db02083f0ca21799e8325a124901549c5effb1945d")
        print(f"⏱️  Timeout set to {timeout_seconds} seconds")
        print("🔄 Starting audio generation (this may take a few minutes for the first run)...")
        
        # Set up timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_seconds)
        
        try:
            # Use the model version directly
            prediction = replicate.predictions.create(
                version="mgysel/stable-audio-open:8465bfb2f7a77991f33e26db02083f0ca21799e8325a124901549c5effb1945d",
                input={
                    "description": description,
                    "duration": duration
                }
            )
            
            print("⏳ Waiting for prediction to complete...")
            prediction.wait()
            output = prediction.output
            
            # Cancel the alarm
            signal.alarm(0)
            
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
                
        except TimeoutError:
            print(f"❌ Request timed out after {timeout_seconds} seconds")
            return None
        finally:
            # Make sure to cancel the alarm
            signal.alarm(0)
            
    except Exception as e:
        print(f"Error generating audio: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Example usage with 30-second timeout
    description = "heavenly flowing pad"
    audio_file = generate_audio_with_timeout(description, duration=1)
    
    if audio_file:
        print(f"\n✅ Success! Audio file created at: {audio_file}")
    else:
        print("\n❌ Failed to generate audio file") 