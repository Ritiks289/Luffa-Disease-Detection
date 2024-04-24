# from fastapi import FastAPI, File, UploadFile
# from fastapi.middleware.cors import CORSMiddleware
# import uvicorn
# import numpy as np
# from io import BytesIO
# from PIL import Image
# import tensorflow as tf
# import requests

# app = FastAPI()

# #
# app = FastAPI()
# #/Users/ritiksingh/Downloads/codabasicLuffa.h5
# MODEL = tf.keras.models.load_model('/Users/ritiksingh/Downloads/model(codebasics)reduction2.h5')
# CLASS_NAMES =['Alternaria_224',
#  'Angular_Spot_224',
#  'Faulty_224',
#  'Flower_224',
#  'Freash_224',
#  'Fresh_224',
#  'Holed_224',
#  'Mosaic_224']


# @app.get("/ping")
# async def ping():
#     return "Hello, I am alive"

# def read_file_as_image(data) -> np.ndarray:
#     image = np.array(Image.open(BytesIO(data)))
#     return image

# @app.post("/predict")
# async def predict(
#     file: UploadFile = File(...)
# ):
#     image = read_file_as_image(await file.read())
#     img_batch = np.expand_dims(image, 0)

#     json_data = {
#         "instances": img_batch.tolist()
#     }

#     response = requests.post(endpoint, json=json_data)
#     prediction = np.array(response.json()["predictions"][0])

#     predicted_class = CLASS_NAMES[np.argmax(prediction)]
#     confidence = np.max(prediction)

#     return {
#         "class": predicted_class,
#         "confidence": float(confidence)
#     }

# if __name__ == "__main__":
#     uvicorn.run(app, host='localhost', port=8000)

from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()
#/Users/ritiksingh/Downloads/codabasicLuffa.h5
MODEL = tf.keras.models.load_model('/Users/ritiksingh/Downloads/model(codebasics)reduction2.h5')
CLASS_NAMES =['Alternaria_224',
 'Angular_Spot_224',
 'Faulty_224',
 'Flower_224',
 'Freash_224',
 'Fresh_224',
 'Holed_224',
 'Mosaic_224']

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
    confidence = prediction[0][class_index].item()

    return {
        "predicted_class": predicted_class,
        "prediction_probabilities": prediction_probabilities
    }



if __name__ == "__main__":
    uvicorn.run(app,host='localhost', port=8000)

