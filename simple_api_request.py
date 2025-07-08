#!/usr/bin/env python3
"""
Simple API request to Replicate for stable-audio-open
"""

import replicate
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def make_api_request(description: str, duration: int = 8):
    """
    Make a simple API request to Replicate
    
    Args:
        description (str): Text description of the audio to generate
        duration (int): Duration in seconds (default: 8)
    
    Returns:
        str: URL to the generated audio file
    """
    
    # Check if API token is set
    api_token = os.getenv("REPLICATE_API_TOKEN")
    if not api_token:
        print("❌ REPLICATE_API_TOKEN not found")
        print("Please set it with: export REPLICATE_API_TOKEN='your_token_here'")
        return None
    
    print(f"🔑 Using API token: {api_token[:10]}...")
    print(f"🎵 Generating {duration}s audio for: '{description}'")
    
    try:
        # Use the model version directly
        print("🚀 Sending request to Replicate model...")
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
        
        print(f"✅ Success! Audio URL: {output}")
        return output
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def download_audio(url: str, filename: str = "generated_audio.wav"):
    """
    Download audio from URL and save to file
    
    Args:
        url (str): URL to the audio file
        filename (str): Local filename to save as
    
    Returns:
        str: Path to saved file, or None if failed
    """
    import requests
    
    try:
        print(f"📥 Downloading audio from: {url}")
        response = requests.get(url)
        
        if response.status_code == 200:
            # Create output directory
            output_dir = Path("generated_audio")
            output_dir.mkdir(exist_ok=True)
            
            # Save file
            output_path = output_dir / filename
            with open(output_path, "wb") as f:
                f.write(response.content)
            
            print(f"💾 Audio saved to: {output_path}")
            return str(output_path)
        else:
            print(f"❌ Download failed: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ Download error: {e}")
        return None

if __name__ == "__main__":
    # Example usage
    description = "Noisia style neuro dnb bass at 170bpm"
    duration = 10
    
    print("🎼 Stable Audio Open - API Request")
    print("=" * 40)
    
    # Make the API request
    audio_url = make_api_request(description, duration)
    
    if audio_url:
        # Download the audio
        filename = f"{description}_{duration}s.wav"
        local_file = download_audio(audio_url, filename)
        
        if local_file:
            print(f"\n🎉 Complete! Audio file: {local_file}")
        else:
            print("\n⚠️  API request succeeded but download failed")
    else:
        print("\n❌ API request failed") 