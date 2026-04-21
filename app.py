import logging
import os
import uuid
import gc
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

# Reduce torch threads to save memory/CPU overhead
torch.set_num_threads(1)

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

        # ===== RESIZE IMAGE TO SAVE MEMORY =====
        # Large images cause OOM during results.plot() and cv2.imwrite()
        logger.info(f"Checking image size for {filepath}...")
        img = cv2.imread(filepath)
        if img is not None:
            h, w = img.shape[:2]
            max_dim = 640
            if max(h, w) > max_dim:
                logger.info(f"Resizing image from {w}x{h} to max dimension {max_dim}")
                scale = max_dim / max(h, w)
                img = cv2.resize(img, (int(w * scale), int(h * scale)))
                cv2.imwrite(filepath, img)
            del img
            gc.collect()

        # ===== RUN MODEL =====
        logger.info(f"Running inference on {filepath}...")

        with torch.no_grad():
            # imgsz=320 is good for performance/memory on small models
            results = model.predict(source=filepath, imgsz=320, device="cpu", verbose=False)

        logger.info("Inference complete")

        # ===== SAVE RESULT IMAGE =====
        out_filename = f"result_{filename}"
        result_path = os.path.join(STATIC_FOLDER, out_filename)

        result = results[0]
        annotated = result.plot()

        logger.info(f"Saving annotated image to {result_path}...")
        success = cv2.imwrite(result_path, annotated)
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
        gc.collect()

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
