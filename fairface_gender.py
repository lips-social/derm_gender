import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import classification_report, f1_score, accuracy_score

# ========== Config ==========
IMAGE_DIR = "D:/NVIDIA Research/FairFace/Train"
LABEL_FILE = "fairface_label_train.csv"
BATCH_SIZE = 32
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Filter CSV to only rows where filename matches one of the 1000 images
available_images = set(f.lower().strip() for f in os.listdir(IMAGE_DIR)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png')))

df = pd.read_csv(LABEL_FILE)
df['file'] = df['file'].astype(str).str.strip()
df['filename'] = df['file'].apply(lambda x: os.path.basename(x).lower().strip())
df['gender'] = df['gender'].astype(str).str.lower().str.strip()

# Keep only filenames that exist in image folder
df = df[df['filename'].isin(available_images)]
df = df[df['gender'].isin(['male', 'female'])]

# Check result
print(f"\n Filtered dataset contains {len(df)} valid image records")
print(df[['filename', 'gender']].head())



# ========== Dataset ==========
class FairFaceGenderDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.data = dataframe
        self.image_dir = image_dir
        self.transform = transform
        self.gender_map = {'male': 0, 'female': 1}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_name = row['filename']
        gender = self.gender_map[str(row['gender']).lower().strip()]
        img_path = os.path.join(self.image_dir, img_name)

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, gender

# ========== Transforms ==========
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

dataset = FairFaceGenderDataset(df, IMAGE_DIR, transform)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ========== Model ==========
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(DEVICE)

# ========== Training Setup ==========
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ========== Training Loop ==========
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss:.4f}")

# ========== Evaluation ==========
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in train_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

# ========== Metrics ==========
f1 = f1_score(all_labels, all_preds, average='macro')
accuracy = accuracy_score(all_labels, all_preds)

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=["Male", "Female"]))
print(f"\n F1 Score (Macro Avg): {f1:.4f}")
print(f" Prediction Accuracy: {accuracy:.4f}")
