# inference_pipeline.py
import argparse
from pathlib import Path
from craft_text_detector import Craft

def run_inference(input_dir, output_dir):
    craft = Craft(output_dir=output_dir, crop_type="poly", cuda=False)
    image_paths = list(Path(input_dir).glob("*.jpg")) + list(Path(input_dir).glob("*.png"))
    for img in image_paths:
        print(f"Processing {img}")
        craft.detect_text(str(img))
    craft.unload_craftnet_model()
    craft.unload_refinenet_model()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    run_inference(args.input_dir, args.output_dir)
