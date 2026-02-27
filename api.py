from flask import Flask, request, jsonify
import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.densenet201(pretrained=False)
num_features = model.classifier.in_features
model.classifier = nn.Sequential(
    nn.Linear(num_features, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 4)
)

model.load_state_dict(torch.load("colon_model.pth", map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
    image = Image.open(file).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    return jsonify({"prediction": int(predicted.item())})

if __name__ == "__main__":
    app.run(debug=True)
