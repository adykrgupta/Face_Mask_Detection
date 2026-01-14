import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

IMG_SIZE = (224, 224)

# Load model
model = tf.keras.models.load_model("mask_model.keras")

class_names = ["with_mask", "without_mask"]

def preprocess_image(img):
    img = img.resize(IMG_SIZE)
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict(image):
    img = preprocess_image(image)
    pred = model.predict(img)[0][0]

    if pred > 0.5:
        return {"without_mask": float(pred), "with_mask": float(1 - pred)}
    else:
        return {"with_mask": float(1 - pred), "without_mask": float(pred)}

interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=2),
    title="Face Mask Detection",
    description="Detect whether a person is wearing a face mask"
)

interface.launch()
