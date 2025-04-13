

from fastapi import FastAPI, File, UploadFile
import uvicorn
import logging
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    
    "http://localhost:5173", 
    
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Optional: TS serving endpoint (not used in this local model prediction)
# endpoint = "http://localhost:8501/v1/models/potatoes_disease:predict"

# Load model and class names
MODEL = tf.keras.models.load_model("saudidattt.h5")
CLASS_NAMES = ["Ajwa", "Galaxy", "Mejdool", "Meneifi", "NabtatAli", "Rutab", "Shaishe","Sokari","Sugaey"]

logger = logging.getLogger('uvicorn.error')
logger.setLevel(logging.DEBUG)

def read_file_as_image(data) -> np.ndarray:
    # Open the image from bytes data
    image = Image.open(BytesIO(data)).convert("RGB")
    # Resize the image to (256, 256)
    image = image.resize((224, 224), Image.Resampling.LANCZOS)
    # Convert the image to a NumPy array
    image = np.array(image)
    return image

@app.post("/predicts")
async def predict(file: UploadFile = File(...)):
    image_array = read_file_as_image(await file.read())
    image_batch = np.expand_dims(image_array, 0)  # (1, 256, 256, 3)
    
    predictions = MODEL.predict(image_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence_level = np.max(predictions[0])
    
    logger.debug(f"Prediction: {predicted_class}, Confidence: {confidence_level}")
    
    return {
        "class": predicted_class,
        "confidence": float(confidence_level)
    }


if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8000)
    



