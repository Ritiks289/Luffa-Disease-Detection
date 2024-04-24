from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

# Load your TensorFlow model and define class names
MODEL = tf.keras.models.load_model('/Users/ritiksingh/Downloads/codabasicLuffa(8classes).h5')
CLASS_NAMES = ['Alternaria', 'Angular_Spot', 'Faulty', 'Flower', 'Freash', 'Fresh', 'Holed', 'Mosaic']

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Set this to the domain of your frontend app (e.g., "http://localhost:3000")
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    try:
        image = np.array(Image.open(BytesIO(data)))
        return image
    except Exception as e:
        print(f"Error reading image: {e}")
        return None

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())

    if image is None:
        return {"error": "Failed to read or process the image."}

    print(f"Image shape: {image.shape}, dtype: {image.dtype}")

    image_batch = np.expand_dims(image, 0)

    print(f"Image batch shape: {image_batch.shape}, dtype: {image_batch.dtype}")

    prediction = MODEL.predict(image_batch)

    class_index = np.argmax(prediction)
    predicted_class = CLASS_NAMES[class_index]
    prediction_probabilities = prediction.tolist()[0]
    confidence = prediction[0][class_index].item()  # Extract confidence and convert to float
  # Get confidence score for predicted class

    # Include predicted_class and confidence in the response dictionary
    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "prediction_probabilities": prediction_probabilities
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
