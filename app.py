import gradio as gr
import tensorflow as tf
import numpy as np

# 1. Load your model
model = tf.keras.models.load_model("mask_detector.keras")
class_names = ['with_mask', 'without_mask']

def predict(img):
    # Preprocess image to match EfficientNetV2B0 input (224x224)
    img = tf.image.resize(img, (224, 224))
    img = tf.expand_dims(img, axis=0) # Add batch dimension
    
    # Get prediction
    prediction = model.predict(img)[0][0]
    
    # Logic for Sigmoid output
    if prediction > 0.5:
        return {"Without Mask": float(prediction), "With Mask": float(1-prediction)}
    else:
        return {"With Mask": float(1-prediction), "Without Mask": float(prediction)}

# 2. Create Gradio Interface
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(),
    outputs=gr.Label(num_top_classes=2),
    title="Face Mask Detector",
    description="Upload an image to check if the person is wearing a mask."
)

interface.launch()