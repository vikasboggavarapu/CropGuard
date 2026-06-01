import os
import uuid
import base64
from io import BytesIO
from pathlib import Path

from flask import Flask, request, jsonify, render_template, abort
from PIL import Image

from predict import predictor

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def image_to_base64(image: Image.Image, fmt: str = "JPEG") -> str:
    buf = BytesIO()
    if image.mode != "RGB":
        image = image.convert("RGB")
    image.save(buf, format=fmt, quality=85)
    return base64.b64encode(buf.getvalue()).decode()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if not file or not file.filename:
        return jsonify({"error": "Empty file"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

    try:
        image = Image.open(file.stream)
        # Resize for display (keep thumbnail ≤ 600px on longest side)
        display = image.copy()
        display.thumbnail((600, 600))
        img_b64 = image_to_base64(display)

        result = predictor.predict(image)
        result["image_b64"] = img_b64
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    print("Loading model …")
    try:
        predictor.load()
        print("Model loaded successfully.")
    except FileNotFoundError as e:
        print(f"WARNING: {e}")
        print("The app will start but predictions will fail until the model is trained.")

    app.run(debug=False, host="0.0.0.0", port=5000)
