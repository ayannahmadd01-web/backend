
import io
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import tensorflow as tf

APP_TITLE = "Image Model API"
MODEL_PATH = "best_model.keras"
IMG_SIZE = (224, 224)
CLASS_NAMES = ["Normal", "Tuberculosis"]

app = FastAPI(title=APP_TITLE)
model = None

@app.on_event("startup")
def load_model():
    global model
    model = tf.keras.models.load_model(MODEL_PATH)

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    img = img.resize((IMG_SIZE[1], IMG_SIZE[0]))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload an image")

    image_bytes = await file.read()
    x = preprocess_image(image_bytes)
    preds = model.predict(x)
    preds = np.asarray(preds)

    scores = preds[0].tolist()
    best_idx = int(np.argmax(preds[0]))
    return {
        "scores": scores,
        "predicted_index": best_idx,
        "predicted_label": CLASS_NAMES[best_idx],
        "confidence": float(preds[0][best_idx]),
    }

