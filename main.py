import uvicorn
from fastapi import FastAPI, UploadFile, File
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image
import shutil
import time
# from tensorflow_hub.version import __version__

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# __version__ = "0.13.0"

port = int(os.getenv("PORT"))
app = FastAPI()  # create a new FastAPI app instance

t = time.time()
export_path = "saved_model/1685796311".format(int(t))
model = tf.keras.models.load_model(
    export_path, custom_objects={"KerasLayer": hub.KerasLayer}
)

Labels = ["Atopic-Dermatitis", "Poison-Ivy", "Scabies-Lyme"]

def predict(img):
    img = tf.io.read_file(img)  # Read the image file
    img = tf.image.decode_image(img, channels=3)  # Decode the image
    img = tf.image.resize(img, (224, 224))  # Resize the image to (224, 224)
    img = img / 255.0  # Normalize
    probabilities = model.predict(np.asarray([img]))[0]
    class_idx = np.argmax(probabilities)

    return [Labels[class_idx], probabilities[class_idx]]


@app.get("/")
def hello_world():
    return "hello world"


@app.post("/predict")
def classify(input: UploadFile = File(...)):
    savefile = input.filename
    with open(savefile, "wb") as buffer:
        shutil.copyfileobj(input.file, buffer)
    result = predict(savefile)
    os.remove(savefile)
    return str(result)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=port, timeout_keep_alive=1200)