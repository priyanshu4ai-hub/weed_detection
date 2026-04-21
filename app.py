import logging
import os
import uuid
from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
import cv2
import gdown

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ===== CONFIG =====
UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"
MODEL_PATH = "weed_model_v1.pt"
MODEL_URL = "https://drive.google.com/uc?id=1CgthuUyWPBzPpw-G0TkxH2SKzxUJRWzE"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# ===== DOWNLOAD MODEL =====
if not os.path.exists(MODEL_PATH):
    logger.info("Downloading model from Google Drive...")
    try:
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    except Exception as e:
        logger.error(f"Failed to download model: {e}")

# ===== LOAD MODEL AT STARTUP =====
logger.info("Loading YOLO model...")
try:
    model = YOLO(MODEL_PATH)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None

# ===== ROUTES =====
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/health")
def health():
    return jsonify({"status": "healthy", "model_loaded": model is not None})

@app.route("/detect", methods=["POST"])
def detect():
    if model is None:
        return jsonify({"error": "Model not loaded. Check server logs."}), 500

    try:
        if "image" not in request.files:
            return jsonify({"error": "No image provided"}), 400

        file = request.files["image"]

        if file.filename == "" or not file.content_type.startswith("image/"):
            return jsonify({"error": "Invalid file type"}), 400

        # ===== SAVE IMAGE =====
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in [".jpg", ".jpeg", ".png", ".webp"]:
            ext = ".jpg"

        filename = f"{uuid.uuid4().hex}{ext}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # ===== RUN MODEL =====
        logger.info(f"Running inference on {filepath}...")
        # Use a small imgsz to save memory on Render
        results = model(filepath, imgsz=320, device="cpu")
        logger.info("Inference complete")

        # ===== SAVE RESULT IMAGE =====
        out_filename = f"result_{filename}"
        result_path = os.path.join(STATIC_FOLDER, out_filename)

        annotated = results[0].plot()

        logger.info(f"Saving annotated image to {result_path}...")
        success = cv2.imwrite(result_path, annotated)
        if not success:
            logger.error(f"Failed to save image to {result_path}")
            return jsonify({"error": "Failed to save output image"}), 500

        # ===== DETECTIONS =====
        detections = []
        boxes = results[0].boxes
        names = results[0].names

        if boxes is not None:
            for box in boxes:
                detections.append({
                    "class_id": int(box.cls[0]),
                    "class_name": names[int(box.cls[0])],
                    "confidence": round(float(box.conf[0]), 4)
                })

        # Cleanup original upload to save space
        try:
            os.remove(filepath)
        except Exception as e:
            logger.warning(f"Failed to delete original file {filepath}: {e}")

        return jsonify({
            "detections": detections,
            "total": len(detections),
            "result_image": f"/static/{out_filename}"
        })

    except Exception as e:
        logger.exception("An error occurred during detection")
        return jsonify({"error": str(e)}), 500

# ===== RUN APP =====
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
