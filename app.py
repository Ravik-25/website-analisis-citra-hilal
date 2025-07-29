import gradio as gr
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps

# Load model dan label
model = load_model("keras_model.h5", compile=False)
class_names = [line.strip() for line in open("labels.txt", "r").readlines()]

def predict(image):
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    img = np.asarray(image)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    index = np.argmax(prediction)
    confidence = prediction[0][index]

    return {class_names[i]: float(prediction[0][i]) for i in range(len(class_names))}

gr.Interface(fn=predict,
             inputs=gr.Image(type="pil"),
             outputs=gr.Label(num_top_classes=1),
             title="Prediksi Hilal Teachable Machine").launch()