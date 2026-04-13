from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
import os
import cv2
import uuid

app = Flask(__name__)

# ===== CONFIG =====
UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# ===== MODEL PATH =====
MODEL_PATH = os.path.join(os.getcwd(), "weed_model_v1.pt")

if not os.path.exists(MODEL_PATH):
    raise Exception("❌ Model file not found! Upload weed_model_v1.pt")

# ===== LOAD MODEL LAZY (IMPORTANT FIX) =====
model = None

def load_model():
    global model
    if model is None:
        print("🚀 Loading YOLO model...")
        model = YOLO(MODEL_PATH)


# ===== ROUTES =====
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/detect", methods=["POST"])
def detect():
    try:
        # 🔥 Load model only when needed
        load_model()

        if "image" not in request.files:
            return jsonify({"error": "No image provided"}), 400

        file = request.files["image"]

        if file.filename == "" or not file.content_type.startswith("image/"):
            return jsonify({"error": "Invalid file"}), 400

        # ===== SAVE FILE =====
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in [".jpg", ".jpeg", ".png", ".webp"]:
            ext = ".jpg"

        filename = f"{uuid.uuid4().hex}{ext}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # ===== RUN MODEL =====
        results = model(filepath)

        # ===== SAVE OUTPUT =====
        out_filename = f"result_{filename}"
        result_path = os.path.join(STATIC_FOLDER, out_filename)

        annotated = results[0].plot()
        cv2.imwrite(result_path, annotated)

        # ===== EXTRACT DETECTIONS =====
        detections = []
        boxes = results[0].boxes
        names = results[0].names

        if boxes is not None:
            for box in boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = names.get(class_id, f"class_{class_id}")

                detections.append({
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": round(confidence, 4)
                })

        return jsonify({
            "detections": detections,
            "total": len(detections),
            "result_image": result_path.replace("\\", "/")
        })

    except Exception as e:
        print("❌ ERROR:", str(e))
        return jsonify({"error": str(e)}), 500


# ===== RUN APP =====
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)