import replicate
import os
import requests
from pathlib import Path

def generate_audio_with_debug(description: str, duration: int = 1):
    """
    Generate audio using the deployed Replicate model with debug info
    """
    
    # Check if API token is set
    api_token = os.getenv("REPLICATE_API_TOKEN")
    if not api_token:
        print("Error: REPLICATE_API_TOKEN environment variable not set")
        return None
    
    print("=== API CALL DEBUG INFO ===")
    print(f"Model: mgysel/stable-audio-open")
    print(f"Description: {description}")
    print(f"Duration: {duration} seconds")
    print(f"API Token: {api_token[:10]}...{api_token[-4:] if len(api_token) > 14 else '***'}")
    
    try:
        # The replicate.run() function internally makes a POST request to:
        # https://api.replicate.com/v1/predictions
        
        print("\n=== INTERNAL API CALL ===")
        print("URL: https://api.replicate.com/v1/predictions")
        print("Method: POST")
        print("Headers: Authorization: Token <your_token>, Content-Type: application/json")
        
        # Make the API call
        output = replicate.run(
            "mgysel/stable-audio-open",
            input={
                "description": description,
                "duration": duration
            }
        )
        
        print(f"\n=== RESPONSE ===")
        print(f"Model output URL: {output}")
        
        # The output is a URL to the generated audio file
        if output and isinstance(output, str):
            print(f"\n=== DOWNLOADING AUDIO ===")
            print(f"Download URL: {output}")
            
            # Download the audio file
            response = requests.get(output)
            print(f"Download status code: {response.status_code}")
            
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
        return None

if __name__ == "__main__":
    # Example usage
    description = "A quick drum beat"
    audio_file = generate_audio_with_debug(description, duration=1)
    
    if audio_file:
        print(f"\n✅ Success! Audio file created at: {audio_file}")
    else:
        print("\n❌ Failed to generate audio file") 