from datasets import load_dataset
import os
from PIL import Image
from tqdm import tqdm

# Define styles you want
STYLES = ["Realism", "Cubism", "Impressionism", "Abstract Art"]
OUTPUT_DIR = "data/wikiart_hf"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the dataset
dataset = load_dataset("asahi417/wikiart-all", split="train")

# Filter and save
for style in STYLES:
    style_dir = os.path.join(OUTPUT_DIR, style.lower().replace(" ", "_"))
    os.makedirs(style_dir, exist_ok=True)
    
    style_subset = dataset.filter(lambda x: x["style"] == style)
    print(f"{style}: {len(style_subset)} images")

    for idx, sample in enumerate(tqdm(style_subset, desc=f"Saving {style}")):
        try:
            image = sample["image"]
            image.save(os.path.join(style_dir, f"{idx+1}.jpg"))
        except Exception as e:
            print(f"Error saving image: {e}")
