import replicate
import os
from dotenv import load_dotenv

# Load environment variables from .env file in root directory
load_dotenv()

def test_api_connection():
    """Test basic Replicate API connection"""
    
    # Check if API token is set
    api_token = os.getenv("REPLICATE_API_TOKEN")
    if not api_token:
        print("Error: REPLICATE_API_TOKEN environment variable not set")
        return False
    
    print(f"API token loaded (first 10 chars: {api_token[:10]}...)")
    
    try:
        # Test a simple API call to list models
        print("Testing API connection...")
        
        # Try to get model info
        model_info = replicate.models.get("mgysel/stable-audio-open:d90a0c38317e1c4316732753a632dbc9757f4bcae7c16b0128e85e457014da71")
        print(f"✅ Model found: {model_info}")
        
        # Model info shows the latest version is available
        print(f"✅ Model is accessible and ready to use")
        
        return True
        
    except Exception as e:
        print(f"❌ API connection failed: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_api_connection() 