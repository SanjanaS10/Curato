from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

# Load model + processor once
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Candidate tags (can expand later)
CANDIDATE_TAGS = [
    "portrait", "landscape", "abstract", "surreal", "dark", "bright", 
    "melancholy", "joyful", "blue tones", "warm colors", "minimalist", "detailed"
]

def generate_tags(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=CANDIDATE_TAGS, images=image, return_tensors="pt", padding=True)
    outputs = model(**inputs)

    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)

    top_probs, indices = probs.topk(5)
    tags = [CANDIDATE_TAGS[i] for i in indices[0]]

    return tags
