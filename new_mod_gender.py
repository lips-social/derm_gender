import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# === Config ===
IMAGE_FOLDER = "D:/NVIDIA Research/processed_images1"
CSV_FILE = "output1.csv"
EPOCHS = 30
BATCH_SIZE = 32
NUM_WORKERS = 0

# === Data Prep ===
df = pd.read_csv(CSV_FILE)


df['gender'] = df['gender'].astype(int)
print(df['gender'].value_counts())


df['roughness'] = df['roughness'].apply(lambda x: 2 if x in [3, 4] else x)
df['pores'] = df['pores'].apply(lambda x: 2 if x in [3, 4] else x)
df['blackheads'] = df['blackheads'].apply(lambda x: 2 if x in [3] else x)
df = df[df['roughness'] != -1].reset_index(drop=True)
df = df[df['pores'] != -1].reset_index(drop=True)
df = df[df['blackheads'] != -1].reset_index(drop=True)
print(df["blackheads"].value_counts())
print(df['roughness'].value_counts())
print(df['pores'].value_counts())

"""transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),"""

# === Transforms ===
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# === Dataset ===
class SkinDataset(Dataset):
    def __init__(self, df, label_col, image_folder, transform=None):
        self.df = df.reset_index(drop=True)
        self.label_col = label_col
        self.image_folder = image_folder
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.image_folder, row['file'])
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
        label = int(row[self.label_col])
        return image, label

def cross_validate_skin_model(df, label_col, num_classes, image_folder, n_splits=5):
    from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
    from sklearn.utils.class_weight import compute_class_weight

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    all_f1_scores = []
    all_accuracies = []
    best_model = None
    best_f1 = -1

    # Compute class weights
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(df[label_col]), y=df[label_col])
    class_weights = torch.tensor(class_weights, dtype=torch.float)

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df[label_col])):
        print(f"\nFold {fold + 1}/{n_splits}")
        train_df = df.loc[train_idx].reset_index(drop=True)
        val_df = df.loc[val_idx].reset_index(drop=True)

        # Datasets & Loaders
        train_dataset = SkinDataset(train_df, label_col, image_folder, transform=train_transform)
        val_dataset = SkinDataset(val_df, label_col, image_folder, transform=val_transform)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
        val_loader = DataLoader(val_dataset, batch_size=1)

        # Model
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        # Training loop
        model.train()
        for epoch in range(EPOCHS):
            total_loss = 0
            for images, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            scheduler.step()
            print(f"  Epoch {epoch + 1}/{EPOCHS}, Loss: {total_loss:.4f}")

        # Evaluation
        model.eval()
        all_preds, all_labels = [], []
        confidences = []
        correct_flags = []

        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                conf, preds = torch.max(probs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                confidences.extend(conf.cpu().numpy())
                correct_flags.extend((preds == labels).cpu().numpy())

        # Metrics
        f1 = f1_score(all_labels, all_preds, average='macro')
        accuracy = accuracy_score(all_labels, all_preds)
        all_f1_scores.append(f1)
        all_accuracies.append(accuracy)

        report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
        cm = confusion_matrix(all_labels, all_preds)
        class_accuracy = cm.diagonal() / cm.sum(axis=1)

        conf_correct = np.array(confidences)[np.array(correct_flags) == 1]
        conf_incorrect = np.array(confidences)[np.array(correct_flags) == 0]
        avg_conf_correct = np.mean(conf_correct) if len(conf_correct) > 0 else 0
        avg_conf_incorrect = np.mean(conf_incorrect) if len(conf_incorrect) > 0 else 0

        print(f"  Fold Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")
        print(f"  Avg Confidence (Correct): {avg_conf_correct:.4f}, (Incorrect): {avg_conf_incorrect:.4f}")

        # Track best model
        if f1 > best_f1:
            best_f1 = f1
            best_model = model

    print(f"\n=== Final Cross-Validation Results ===")
    print(f"Average Accuracy across {n_splits} folds: {np.mean(all_accuracies):.4f}")
    print(f"Average F1 Score across {n_splits} folds: {np.mean(all_f1_scores):.4f}")
    print(f"Best F1 Score: {best_f1:.4f}")

    return best_model

# Extract features
def extract_skin_features(model, df, label_col, feature_name, image_folder):
    model.eval()
    dataset = SkinDataset(df, label_col, image_folder=image_folder, transform=train_transform)
    loader = DataLoader(dataset, batch_size=1)
    features = []
    with torch.no_grad():
        for images, _ in loader:
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            features.append(probs.cpu().numpy().flatten())
    return pd.DataFrame(features, columns=[f"{feature_name}_{i}" for i in range(probs.shape[1])])

from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score

from sklearn.metrics import make_scorer, accuracy_score, f1_score
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
import pandas as pd

def classify_gender_using_skin_features(df, image_folder, rough_model, pores_model, blackheads_model):
    # Extract features
    f_rough = extract_skin_features(rough_model, df, 'roughness', 'rough', image_folder)
    f_pores = extract_skin_features(pores_model, df, 'pores', 'pores', image_folder)
    f_black = extract_skin_features(blackheads_model, df, 'blackheads', 'black', image_folder)

    # Combine feature vectors
    features = pd.concat([f_rough, f_pores, f_black], axis=1).reset_index(drop=True)
    all_features = features.copy()
    all_features['file'] = df['file'].values  # Add identifier to merge later
    all_features.to_csv("skin_features.csv", index=False)

    targets = df['gender'].reset_index(drop=True)
    combined_df = pd.concat([features, targets], axis=1).dropna()

    X = combined_df.drop(columns=['gender'])
    y = combined_df['gender']

    # Train gender classifier using XGBoost
    clf_gender = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)

    # Accuracy and F1 scorers
    f1 = cross_val_score(clf_gender, X, y, cv=5, scoring='f1_macro')
    acc = cross_val_score(clf_gender, X, y, cv=5, scoring='accuracy')

    print("\n=== Gender Prediction ===")
    print("F1 Score (Macro Avg):", f1.mean())
    print("Accuracy:", acc.mean())

# === Run ===
#cross_validate_skin_model(df, 'pores', num_classes=3, image_folder=IMAGE_FOLDER)
if __name__ == "__main__":
    print("\n--- Training Skin Feature Models ---")
    model_rough = cross_validate_skin_model(df, 'roughness', num_classes=3, image_folder=IMAGE_FOLDER)
    model_pores = cross_validate_skin_model(df, 'pores', num_classes=3, image_folder=IMAGE_FOLDER)
    model_black = cross_validate_skin_model(df, 'blackheads', num_classes=3, image_folder=IMAGE_FOLDER)

    print("\n--- Gender Classification using Skin Features ---")
    classify_gender_using_skin_features(df, IMAGE_FOLDER, model_rough, model_pores, model_black)


