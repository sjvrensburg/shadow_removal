import argparse
from pathlib import Path

from shadow_removal.pipeline import ShadowRemovalPipeline

def main():
    parser = argparse.ArgumentParser(description="Shadow Removal Pipeline")
    parser.add_argument("--input", type=str, required=True, help="Input image or directory")
    parser.add_argument("--output", type=str, required=True, help="Output path (file or directory)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    pipeline = ShadowRemovalPipeline(device=args.device)
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if input_path.is_file():
        pipeline.process_image(input_path, output_path)
    else:
        pipeline.process_directory(input_path, output_path)

if __name__ == "__main__":
    main()