#!/usr/bin/env python3
"""
Test script to check Hugging Face authentication and access to the gated model
"""

import os
from huggingface_hub import HfApi, login, whoami
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_hf_authentication():
    """Test Hugging Face authentication"""
    print("🔐 Testing Hugging Face Authentication...")
    
    # Check for multiple possible token environment variable names
    hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN") or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    
    if not hf_token:
        print("❌ Hugging Face token not found in environment variables")
        print("Please add one of these to your .env file:")
        print("HUGGING_FACE_HUB_TOKEN=your_huggingface_token_here")
        print("HF_TOKEN=your_huggingface_token_here")
        print("HUGGINGFACE_HUB_TOKEN=your_huggingface_token_here")
        return False
    
    # Determine which token was found
    token_var = "HUGGING_FACE_HUB_TOKEN" if os.getenv("HUGGING_FACE_HUB_TOKEN") else "HF_TOKEN" if os.getenv("HF_TOKEN") else "HUGGINGFACE_HUB_TOKEN"
    print(f"✅ {token_var} found (first 10 chars: {hf_token[:10]}...)")
    
    try:
        # Try to authenticate
        login(token=hf_token, add_to_git_credential=False)
        print("✅ Successfully logged in to Hugging Face")
        
        # Get user info
        user_info = whoami()
        print(f"✅ Authenticated as: {user_info['name']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        return False

def test_model_access():
    """Test access to the gated model"""
    print("\n🔒 Testing access to stabilityai/stable-audio-open-1.0...")
    
    api = HfApi()
    model_id = "stabilityai/stable-audio-open-1.0"
    
    try:
        # Try to get model info
        model_info = api.model_info(model_id)
        print(f"✅ Model found: {model_info.id}")
        print(f"✅ Model is {'gated' if model_info.gated else 'public'}")
        
        if model_info.gated:
            print("⚠️  This model is gated - you need to:")
            print("   1. Visit https://huggingface.co/stabilityai/stable-audio-open-1.0")
            print("   2. Click 'Request access' and fill out the form")
            print("   3. Wait for approval from Stability AI")
        
        return True
        
    except GatedRepoError as e:
        print("❌ Access denied to gated repository")
        print("📝 You need to request access:")
        print("   1. Visit https://huggingface.co/stabilityai/stable-audio-open-1.0")
        print("   2. Click 'Request access' and fill out the form")
        print("   3. Wait for approval from Stability AI")
        print("   4. Make sure your HF_TOKEN has the required permissions")
        return False
        
    except RepositoryNotFoundError:
        print("❌ Repository not found")
        return False
        
    except Exception as e:
        print(f"❌ Error accessing model: {e}")
        return False

def test_file_download():
    """Test downloading a specific file from the model"""
    print("\n📥 Testing file download from the model...")
    
    try:
        from huggingface_hub import hf_hub_download
        
        # Try to download the model config file
        config_path = hf_hub_download(
            repo_id="stabilityai/stable-audio-open-1.0",
            filename="model_config.json",
            repo_type="model"
        )
        print(f"✅ Successfully downloaded config to: {config_path}")
        return True
        
    except GatedRepoError:
        print("❌ Cannot download - access denied to gated repository")
        return False
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False

def check_alternative_models():
    """Check for alternative non-gated models"""
    print("\n🔍 Checking for alternative models...")
    
    alternatives = [
        "facebook/musicgen-small",
        "facebook/musicgen-medium", 
        "facebook/musicgen-large",
        "microsoft/speecht5_tts",
    ]
    
    api = HfApi()
    
    for model_id in alternatives:
        try:
            model_info = api.model_info(model_id)
            gated_status = "🔒 Gated" if model_info.gated else "🔓 Public"
            print(f"   {model_id}: {gated_status}")
        except Exception as e:
            print(f"   {model_id}: ❌ Error - {e}")

if __name__ == "__main__":
    print("🧪 Hugging Face Authentication & Model Access Test")
    print("=" * 50)
    
    # Test authentication
    auth_success = test_hf_authentication()
    
    if auth_success:
        # Test model access
        test_model_access()
        
        # Test file download
        test_file_download()
    
    # Check alternatives
    check_alternative_models()
    
    print("\n" + "=" * 50)
    print("💡 Next steps:")
    print("1. If authentication failed: Get a HF token from https://huggingface.co/settings/tokens")
    print("2. If model access denied: Request access at https://huggingface.co/stabilityai/stable-audio-open-1.0")
    print("3. Consider using alternative models if access is not granted") 