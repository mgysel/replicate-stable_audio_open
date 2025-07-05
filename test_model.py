#!/usr/bin/env python3

import argparse
from stable_audio_tools import get_pretrained_model

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--description", default="Gentle piano melody", help="Audio description")
    p.add_argument("--duration", type=int, default=5, help="Duration in seconds")
    p.add_argument("--output-path", default="test_output.wav", help="Output file path")
    args = p.parse_args()

    print(f"Loading Stable Audio Open model...")
    model, cfg = get_pretrained_model("stabilityai/stable-audio-open-1.0")
    
    print(f"Generating audio: '{args.description}' for {args.duration} seconds...")
    audio = model.sample(
        description=args.description,
        seconds_total=args.duration,
        cfg=cfg,
    )
    
    print(f"Saving to {args.output_path}...")
    audio.write_wav(args.output_path)
    print(f"Audio generated successfully!")

if __name__ == "__main__":
    main() 