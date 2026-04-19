from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

# -------------------------
# CLASS NAMES — alphabetical (ImageFolder order, 8 classes)
# -------------------------
classes = [
    "Pepper__bell__Bacterial_spot",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy"
]

# -------------------------
# DISEASE INFO (matching 8 trained classes)
# -------------------------
disease_info = {
    "Pepper__bell__Bacterial_spot": {
        "display": "Pepper Bell — Bacterial Spot",
        "plant": "Pepper",
        "type": "Bacterial",
        "severity": "Moderate",
        "emoji": "🫑",
        "treatment": "Apply copper-based bactericides. Remove and destroy infected leaves. Avoid overhead irrigation. Use disease-free seeds."
    },
    "Tomato_Leaf_Mold": {
        "display": "Tomato — Leaf Mold",
        "plant": "Tomato",
        "type": "Fungal",
        "severity": "Low-Moderate",
        "emoji": "🍅",
        "treatment": "Improve greenhouse ventilation. Reduce humidity below 85%. Apply copper-based fungicides. Remove heavily infected leaves."
    },
    "Tomato_Septoria_leaf_spot": {
        "display": "Tomato — Septoria Leaf Spot",
        "plant": "Tomato",
        "type": "Fungal",
        "severity": "Moderate",
        "emoji": "🍅",
        "treatment": "Apply fungicides at first sign. Remove infected lower leaves promptly. Avoid wetting foliage. Mulch to prevent soil splash."
    },
    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "display": "Tomato — Spider Mites",
        "plant": "Tomato",
        "type": "Pest",
        "severity": "Moderate",
        "emoji": "🕷️",
        "treatment": "Use miticides or insecticidal soap spray. Increase humidity around plants. Introduce predatory mites. Avoid dusty conditions."
    },
    "Tomato__Target_Spot": {
        "display": "Tomato — Target Spot",
        "plant": "Tomato",
        "type": "Fungal",
        "severity": "Moderate",
        "emoji": "🍅",
        "treatment": "Apply fungicides (azoxystrobin). Improve air circulation. Avoid overhead irrigation. Rotate crops annually."
    },
    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "display": "Tomato — Yellow Leaf Curl Virus",
        "plant": "Tomato",
        "type": "Viral",
        "severity": "High",
        "emoji": "🍅",
        "treatment": "Control whitefly vectors with insecticides. Use reflective mulches. Remove and destroy infected plants. Plant resistant varieties."
    },
    "Tomato__Tomato_mosaic_virus": {
        "display": "Tomato — Mosaic Virus",
        "plant": "Tomato",
        "type": "Viral",
        "severity": "High",
        "emoji": "🍅",
        "treatment": "Remove and destroy infected plants immediately. Disinfect tools. Control aphid vectors. Use virus-free certified seeds."
    },
    "Tomato_healthy": {
        "display": "Tomato — Healthy",
        "plant": "Tomato",
        "type": "None",
        "severity": "None",
        "emoji": "✅",
        "treatment": "Your tomato plant looks healthy! Continue regular care, monitor for pests, and ensure consistent watering."
    }
}

# -------------------------
# LOAD MODEL
# -------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "crop_model.pth")

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(classes))

if os.path.exists(MODEL_PATH):
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            elif "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"])
            else:
                model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)
        print(f"✅ Model loaded! ({len(classes)} classes)")
    except Exception as e:
        print(f"⚠️ Model load error: {e}")
else:
    print("⚠️ crop_model.pth not found. Place it in the same folder as app.py.")

model.eval()

# -------------------------
# IMAGE TRANSFORM
# -------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
    # No Normalize — matches your original training pipeline
])

# -------------------------
# ROUTES
# -------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        image = Image.open(file).convert("RGB")
        tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            confidence, predicted_idx = torch.max(probabilities, 0)

        predicted_key = classes[predicted_idx.item()]
        info = disease_info.get(predicted_key, {})

        top3_probs, top3_idx = torch.topk(probabilities, 3)
        top3 = [
            {
                "label": disease_info.get(classes[idx.item()], {}).get("display", classes[idx.item()]),
                "confidence": round(prob.item() * 100, 2)
            }
            for prob, idx in zip(top3_probs, top3_idx)
        ]

        return jsonify({
            "prediction": info.get("display", predicted_key),
            "confidence": round(confidence.item() * 100, 2),
            "plant": info.get("plant", "Unknown"),
            "type": info.get("type", "Unknown"),
            "severity": info.get("severity", "Unknown"),
            "treatment": info.get("treatment", "Consult an agricultural expert."),
            "emoji": info.get("emoji", "🌿"),
            "top3": top3
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/classes", methods=["GET"])
def get_classes():
    return jsonify({
        "count": len(classes),
        "classes": [disease_info[c]["display"] for c in classes]
    })

if __name__ == "__main__":
    app.run(port=5000, debug=True)
