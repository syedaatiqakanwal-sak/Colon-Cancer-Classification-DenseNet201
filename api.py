from flask import Flask, request, jsonify
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# SAME MODEL STRUCTURE AS TRAINING
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

# ⚠ IMPORTANT: Set 8 classes for Kvasir
num_classes = 8

model = DenseNet201Model(num_classes=num_classes)
model.load_state_dict(torch.load("colon_model.pth", map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    image = Image.open(file).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    return jsonify({"predicted_class_index": int(predicted.item())})

if __name__ == "__main__":
    app.run(debug=True)
