from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
import os
import cv2
import uuid
import gdown

app = Flask(__name__)

# ===== CONFIG =====
UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"
MODEL_PATH = "weed_model_v1.pt"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# ===== GOOGLE DRIVE MODEL DOWNLOAD =====
MODEL_URL = "https://drive.google.com/uc?id=1CgthuUyWPBzPpw-G0TkxH2SKzxUJRWzE"

if not os.path.exists(MODEL_PATH):
    print("⬇️ Downloading model from Google Drive...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# ===== LOAD MODEL =====
model = YOLO(MODEL_PATH)

# ===== ROUTES =====
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/detect", methods=["POST"])
def detect():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image provided"}), 400

        file = request.files["image"]

        if file.filename == "" or not file.content_type.startswith("image/"):
            return jsonify({"error": "Invalid file"}), 400

        # ===== SAVE IMAGE =====
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in [".jpg", ".jpeg", ".png", ".webp"]:
            ext = ".jpg"

        filename = f"{uuid.uuid4().hex}{ext}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # ===== RUN MODEL (OPTIMIZED) =====
        results = model(filepath, imgsz=320, device="cpu")

        # ===== SAVE RESULT =====
        out_filename = f"result_{filename}"
        result_path = os.path.join(os.getcwd(), STATIC_FOLDER, out_filename)

        annotated = results[0].plot()

        success = cv2.imwrite(result_path, annotated)
        if not success:
            raise Exception("Failed to save output image")

        # ===== DETECTIONS =====
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
            "result_image": f"/static/{out_filename}"
        })

    except Exception as e:
        print("❌ ERROR:", str(e))
        return jsonify({"error": str(e)}), 500


# ===== RUN APP =====
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)