import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Dataset directories (UPDATED FOR GITHUB PROJECT STRUCTURE)
train_dir = 'data/train'
val_dir = 'data/val'
test_dir = 'data/test'

# Load datasets
train_dataset = datasets.ImageFolder(train_dir, transform=transform)
val_dataset = datasets.ImageFolder(val_dir, transform=transform)
test_dataset = datasets.ImageFolder(test_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Model: DenseNet201
class DenseNet201Model(nn.Module):
    def __init__(self, num_classes):
        super(DenseNet201Model, self).__init__()
        self.densenet = models.densenet201(pretrained=False)
        self.densenet.classifier = nn.Sequential(
            nn.Linear(1920, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.densenet(x)

num_classes = len(train_dataset.classes)
model = DenseNet201Model(num_classes=num_classes).to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training
epochs = 20

for epoch in range(epochs):
    model.train()
    train_loss, correct_train = 0.0, 0
    total_train = 0

    print(f'Epoch {epoch+1}/{epochs}')
    train_bar = tqdm(train_loader, total=len(train_loader))

    for images, labels in train_bar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        correct_train += (preds == labels).sum().item()
        total_train += labels.size(0)
        train_loss += loss.item() * images.size(0)

        train_bar.set_description(f"Train Loss: {loss.item():.4f}")

    avg_train_loss = train_loss / len(train_dataset)
    train_accuracy = correct_train / total_train * 100

    # Validation
    model.eval()
    val_loss, correct_val = 0.0, 0
    total_val = 0

    with torch.no_grad():
        val_bar = tqdm(val_loader, total=len(val_loader))
        for images, labels in val_bar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            correct_val += (preds == labels).sum().item()
            total_val += labels.size(0)
            val_loss += loss.item() * images.size(0)

            val_bar.set_description(f"Val Loss: {loss.item():.4f}")

    avg_val_loss = val_loss / len(val_dataset)
    val_accuracy = correct_val / total_val * 100

    print(f'Epoch {epoch+1}/{epochs} "
          f"Train Loss: {avg_train_loss:.4f} Train Acc: {train_accuracy:.2f}% "
          f"Val Loss: {avg_val_loss:.4f} Val Acc: {val_accuracy:.2f}%")

# Save trained model
torch.save(model.state_dict(), "colon_model.pth")


# Testing function
def test_model(model, test_loader):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("Classification Report:")
    print(classification_report(all_labels, all_preds,
                                target_names=test_dataset.classes))


# Run testing
test_model(model, test_loader)
