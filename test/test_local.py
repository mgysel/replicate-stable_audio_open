#!/usr/bin/env python3
"""
Test script to verify the Predictor class works locally
"""

import sys
import os
from pathlib import Path

# Add the parent directory to the path so we can import the predictor
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from predict import Predictor

def test_local_model():
    """
    Test the local Cog model
    """
    try:
        print("Setting up the model...")
        predictor = Predictor()
        predictor.setup()
        
        print("Generating audio...")
        description = "heavenly flowing pad"
        duration = 2
        
        output_path = predictor.predict(description, duration)
        
        print(f"✅ Success! Audio file created at: {output_path.absolute()}")
        return str(output_path)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == "__main__":
    test_local_model() 