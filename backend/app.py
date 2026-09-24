from __future__ import annotations

import hashlib
from io import BytesIO
import os
from pathlib import Path
import uuid

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request, send_from_directory, url_for
from PIL import Image, UnidentifiedImageError
from werkzeug.utils import secure_filename

from backend.database import init_db, log_prediction
from models.pipeline import DeepfakePipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

UPLOAD_FOLDER = Path(os.environ.get("UPLOAD_FOLDER", "backend/static/uploads"))
if not UPLOAD_FOLDER.is_absolute():
    UPLOAD_FOLDER = PROJECT_ROOT / UPLOAD_FOLDER

app = Flask(
    __name__,
    template_folder=str(PROJECT_ROOT / "frontend" / "templates"),
    static_folder=str(PROJECT_ROOT / "frontend" / "static"),
)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "development-secret-key")
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
init_db()
pipeline = DeepfakePipeline()


@app.get("/")
def index() -> str:
    return render_template("index.html")


@app.get("/static/uploads/<path:filename>")
def uploaded_heatmap(filename: str):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.get("/api/health")
def health() -> tuple[object, int]:
    return jsonify({"status": "ok"}), 200


@app.post("/api/analyze")
def analyze() -> tuple[object, int]:
    uploaded = request.files.get("file")
    if uploaded is None or not uploaded.filename:
        return jsonify({"error": "missing image file"}), 400

    image_bytes = uploaded.read()
    if not image_bytes:
        return jsonify({"error": "uploaded file is empty"}), 400
    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image.verify()
    except (UnidentifiedImageError, OSError):
        return jsonify({"error": "uploaded file is not a valid image"}), 400

    suffix = Path(secure_filename(uploaded.filename)).suffix.lower() or ".img"
    input_path = UPLOAD_FOLDER / f"{uuid.uuid4().hex}{suffix}"
    input_path.write_bytes(image_bytes)
    image_hash = hashlib.sha256(image_bytes).hexdigest()

    try:
        result = pipeline.run_pipeline(input_path)
    finally:
        input_path.unlink(missing_ok=True)

    if result.get("error") == "no_face_detected":
        return jsonify({"error": "no_face_detected"}), 422

    heatmap = result["heatmap"]
    forensic_heatmap = result["forensic_heatmap"]
    heatmap_name = f"{uuid.uuid4().hex}.png"
    forensic_heatmap_name = f"{uuid.uuid4().hex}.png"
    heatmap_path = UPLOAD_FOLDER / heatmap_name
    forensic_heatmap_path = UPLOAD_FOLDER / forensic_heatmap_name
    Image.fromarray(heatmap).save(heatmap_path, format="PNG")
    Image.fromarray(forensic_heatmap).save(forensic_heatmap_path, format="PNG")
    log_prediction(image_hash, result["label"], result["confidence"])
    return jsonify(
        {
            "label": result["label"],
            "confidence": result["confidence"],
            "heatmap_url": url_for("static", filename=f"uploads/{heatmap_name}"),
            "forensic_heatmap_url": url_for(
                "static", filename=f"uploads/{forensic_heatmap_name}"
            ),
        }
    ), 200
