from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
import os
import cv2
import uuid
import gdown

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"
MODEL_PATH = "weed_model_v1.pt"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# ===== DOWNLOAD MODEL IF NOT PRESENT =====
MODEL_URL = "PASTE_YOUR_GOOGLE_DRIVE_LINK_HERE"

if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# ===== LOAD MODEL =====
model = YOLO(MODEL_PATH)

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]

    if file.filename == "" or not file.content_type.startswith("image/"):
        return jsonify({"error": "Invalid file"}), 400

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".jpg", ".jpeg", ".png", ".webp"]:
        ext = ".jpg"

    filename = f"{uuid.uuid4().hex}{ext}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    results = model(filepath)

    out_filename = f"result_{filename}"
    result_path = os.path.join(STATIC_FOLDER, out_filename)

    annotated = results[0].plot()
    cv2.imwrite(result_path, annotated)

    detections = []
    boxes = results[0].boxes
    names = results[0].names

    if boxes is not None:
        for box in boxes:
            detections.append({
                "class_id": int(box.cls[0]),
                "class_name": names[int(box.cls[0])],
                "confidence": float(box.conf[0])
            })

    return jsonify({
        "detections": detections,
        "result_image": result_path.replace("\\", "/")
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)