#!/usr/bin/env python3
"""
Test script to verify the Predictor class works locally
"""

import sys
import os

# Add current directory to path so we can import predict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from predict import Predictor
    print("✅ Successfully imported Predictor class")
    
    # Test instantiation
    predictor = Predictor()
    print("✅ Successfully created Predictor instance")
    
    # Test setup (this will download the model)
    print("🔄 Setting up model (this may take a while)...")
    predictor.setup()
    print("✅ Model setup completed successfully")
    
    # Test prediction with a simple example
    print("🔄 Testing prediction...")
    result = predictor.predict("A gentle piano melody", 5)
    print(f"✅ Prediction completed! Output: {result}")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc() 