#!/usr/bin/env python3
"""
Test script to verify that the stable-audio model can be loaded locally
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add the parent directory to the path so we can import predict
sys.path.append(str(Path(__file__).parent.parent))

# Load environment variables
load_dotenv()

def test_model_loading():
    """Test loading the stable-audio model locally"""
    print("🧪 Testing Stable Audio Model Loading...")
    print("=" * 50)
    
    # Check if token is available
    hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
    if not hf_token:
        print("❌ HUGGING_FACE_HUB_TOKEN not found in environment")
        return False
    
    print(f"✅ HUGGING_FACE_HUB_TOKEN found (first 10 chars: {hf_token[:10]}...)")
    
    try:
        # Import the predictor
        from predict import Predictor
        
        print("🔄 Creating Predictor instance...")
        predictor = Predictor()
        
        print("🔄 Running setup (this will download the model if not cached)...")
        predictor.setup()
        
        print("✅ Model loaded successfully!")
        print(f"✅ Model config: {type(predictor.cfg)}")
        print(f"✅ Model instance: {type(predictor.model)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False

def test_quick_generation():
    """Test a quick audio generation"""
    print("\n🎵 Testing Quick Audio Generation...")
    print("=" * 50)
    
    try:
        from predict import Predictor
        
        predictor = Predictor()
        predictor.setup()
        
        print("🔄 Generating 1-second audio sample...")
        output_path = predictor.predict("soft piano melody", duration=1)
        
        if output_path and output_path.exists():
            print(f"✅ Audio generated successfully: {output_path}")
            print(f"✅ File size: {output_path.stat().st_size} bytes")
            return True
        else:
            print("❌ No output file generated")
            return False
            
    except Exception as e:
        print(f"❌ Error generating audio: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Stable Audio Model Test Suite")
    print("=" * 50)
    
    # Test model loading
    loading_success = test_model_loading()
    
    if loading_success:
        print("\n" + "=" * 50)
        # Test quick generation
        generation_success = test_quick_generation()
        
        if generation_success:
            print("\n🎉 All tests passed! Your model is ready for deployment.")
        else:
            print("\n⚠️  Model loads but generation failed. Check the logs above.")
    else:
        print("\n❌ Model loading failed. Please check your authentication and access.")
    
    print("\n" + "=" * 50)
    print("💡 Next steps:")
    if loading_success:
        print("1. Your model is working locally!")
        print("2. Deploy to Replicate with: cog push")
        print("3. Make sure to set HUGGING_FACE_HUB_TOKEN in your Replicate secrets")
    else:
        print("1. Check that you have access to stabilityai/stable-audio-open-1.0")
        print("2. Verify your HUGGING_FACE_HUB_TOKEN is correct")
        print("3. Try running: python test/test_hf_auth.py") 