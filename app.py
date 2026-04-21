import logging
import os
import uuid
import gc
import numpy as np
from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
import cv2
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Reduce resource overhead
torch.set_num_threads(1)
cv2.setNumThreads(1)

app = Flask(__name__)

# ===== CONFIG =====
UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"
MODEL_PATH = "weed_model_v1.pt"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# ===== LOAD MODEL AT STARTUP =====
logger.info("Loading YOLO model...")
try:
    # Load model and move to CPU explicitly
    model = YOLO(MODEL_PATH)
    # Fuse model for faster and lighter inference
    model.fuse()
    logger.info("Model loaded and fused successfully")
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

        # ===== READ & PREPROCESS IMAGE IN MEMORY =====
        # Read file into numpy array directly to avoid double disk I/O
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"error": "Failed to decode image"}), 400

        h, w = img.shape[:2]
        # Aggressive resizing to save memory (max 480px for visualization, 320 for inference)
        max_dim = 480
        if max(h, w) > max_dim:
            logger.info(f"Resizing image from {w}x{h} to max dimension {max_dim}")
            scale = max_dim / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)))

        # Free up the raw bytes
        del file_bytes
        gc.collect()

        # ===== RUN MODEL =====
        logger.info("Running inference...")
        with torch.inference_mode():
            # imgsz=320 is very memory efficient. Pass the image array directly.
            results = model.predict(source=img, imgsz=320, device="cpu", verbose=False)

        logger.info("Inference complete")

        # ===== SAVE RESULT IMAGE =====
        out_filename = f"result_{uuid.uuid4().hex}.jpg"
        result_path = os.path.join(STATIC_FOLDER, out_filename)

        result = results[0]
        annotated = result.plot()

        logger.info(f"Saving annotated image to {result_path}...")
        # Use slightly higher compression for JPG to save space/RAM
        success = cv2.imwrite(result_path, annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not success:
            logger.error(f"Failed to save image to {result_path}")
            return jsonify({"error": "Failed to save output image"}), 500

        # ===== DETECTIONS =====
        detections = []
        boxes = result.boxes
        names = result.names

        if boxes is not None:
            for box in boxes:
                detections.append({
                    "class_id": int(box.cls[0]),
                    "class_name": names[int(box.cls[0])],
                    "confidence": round(float(box.conf[0]), 4)
                })

        # ===== CLEANUP =====
        del results
        del result
        del annotated
        del img
        gc.collect()

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
