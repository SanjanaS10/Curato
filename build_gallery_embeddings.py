# build_gallery_embeddings.py
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import os
import pickle

gallery_dir = "gallery"
embedding_file = "gallery_embeddings.pkl"

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

embeddings = []

for fname in os.listdir(gallery_dir):
    if fname.endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(gallery_dir, fname)
        image = Image.open(img_path).convert("RGB")

        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            image_emb = model.get_image_features(**inputs)
        image_emb = image_emb / image_emb.norm(p=2, dim=-1)  # normalize

        embeddings.append({
            "filename": fname,
            "embedding": image_emb.squeeze().cpu()
        })

# Save embeddings
with open(embedding_file, "wb") as f:
    pickle.dump(embeddings, f)

print(f"Saved {len(embeddings)} image embeddings.")
