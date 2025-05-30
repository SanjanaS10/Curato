from datasets import load_dataset
from torchvision import transforms, models
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import json

# Define target styles
STYLES = ["Realism", "Cubism", "Impressionism", "Abstract Art"]

# Load dataset
dataset = load_dataset("asahi417/wikiart-all", split="test")

# Filter dataset for only target styles
dataset = dataset.filter(lambda x: any(style in STYLES for style in x["styles"]))

# Map styles to numeric labels
label_map = {style: idx for idx, style in enumerate(STYLES)}

def get_label(x):
    for style in x["styles"]:
        if style in STYLES:
            return label_map[style]
    return -1  # Should never hit this after filtering



from torchvision import transforms

# Define your transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # Converts image to tensor and scales to [0, 1]
])

# Function to apply to batches
def transform_images(batch):
    batch["pixel_values"] = [transform(image) for image in batch["image"]]
    return batch



dataset = dataset.map(transform_images, batched=True, batch_size=4, keep_in_memory=False)


# Define image transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Apply transform to images
 

def transform_images(batch):
    batch["image"] = [transform(image.convert("RGB")) for image in batch["image"]]
    return batch


dataset = dataset.map(transform_images, batched=True)

# Set dataset format for PyTorch
dataset.set_format(type="torch", columns=["image", "label"])

# Create DataLoader
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, len(STYLES))
model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
model.train()
for epoch in range(20):  # Increase epochs for better training
    running_loss = 0.0
    for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f"Epoch {epoch+1} Loss: {running_loss:.4f}")

# Save model and classes
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/style_model_hf.pth")
with open("models/style_classes.json", "w") as f:
    json.dump(STYLES, f)

print("✅ Model saved to models/style_model_hf.pth")
print("✅ Style classes saved to models/style_classes.json")
