from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image

# Define transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load HF dataset (use 'test' split because 'train' doesn't exist)
hf_dataset = load_dataset("asahi417/wikiart-all", split="test")

# Your custom Dataset class
class WikiArtDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]["image"]
        label = self.dataset[idx]["style"]

        if self.transform:
            image = self.transform(image)

        return image, label

# Create PyTorch dataset
dataset = WikiArtDataset(hf_dataset, transform=transform)

'''from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import json

# Styles (should match folder names in data/wikiart_hf_small)
STYLES = ["Realism", "Cubism", "Impressionism", "Abstract Art"]
DATA_DIR = "data/wikiart"

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load dataset from folders
dataset = datasets.ImageFolder(root="data/wikiart", transform=transform)

dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pretrained model
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, len(STYLES))
model.to(device)

# Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Train
model.train()
for epoch in range(20):
    running_loss = 0.0
    for images, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f"Epoch {epoch+1} Loss: {running_loss:.4f}")

# Save model and label map
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/style_model_hf.pth")

# Save class names
with open("models/style_classes.json", "w") as f:
    json.dump(dataset.classes, f)

print("✅ Model saved to models/style_model_hf.pth")
print("✅ Style classes saved to models/style_classes.json")
'''