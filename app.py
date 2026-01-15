import gradio as gr
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model("mask_model.keras")
class_names = ['with_mask', 'without_mask']

def predict(img):

    img = tf.image.resize(img, (224, 224))
    img = tf.expand_dims(img, axis=0) 
    
    prediction = model.predict(img)[0][0]
    if prediction > 0.5:
        return {"Without Mask": float(prediction), "With Mask": float(1-prediction)}
    else:
        return {"With Mask": float(1-prediction), "Without Mask": float(prediction)}

interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(),
    outputs=gr.Label(num_top_classes=2),
    title="Face Mask Detector",
    description="Upload an image to check if the person is wearing a mask."
)

interface.launch()