# search.py
import torch
import pickle
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

# Load model once
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load saved gallery
with open("gallery_embeddings.pkl", "rb") as f:
    GALLERY = pickle.load(f)

def find_similar_images(query_image_path, top_k=5):
    image = Image.open(query_image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        query_emb = model.get_image_features(**inputs)
    query_emb = query_emb / query_emb.norm(p=2, dim=-1)

    similarities = []
    for item in GALLERY:
        gallery_emb = item["embedding"]
        score = torch.nn.functional.cosine_similarity(query_emb, gallery_emb.unsqueeze(0)).item()
        similarities.append((item["filename"], score))

    top_matches = sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]
    return top_matches
