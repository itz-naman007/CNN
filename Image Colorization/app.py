import gradio as gr
import numpy as np
from PIL import Image
import tensorflow as tf

# Load your trained model
model = tf.keras.models.load_model(r"C:\Users\naman\Desktop\Image Project\autoenc_model010.keras")

def preprocess_image(img):
    if img is None:
        raise ValueError("Please upload an image before proceeding.")
    if not isinstance(img, Image.Image):
        img = Image.open(img)
    img = img.convert("L")
    img = img.resize((128,128))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=-1)
    img = np.repeat(img, 3, axis=-1)
    img = np.expand_dims(img, axis=0)
    return img

def colorize_image(input_image):
    try:
        input_tensor = preprocess_image(input_image)
        predicted = model.predict(input_tensor)
        output = (predicted[0] * 255).astype(np.uint8)
        return Image.fromarray(output)
    except Exception as e:
        return f"Error: {e}"

demo = gr.Interface(
    fn=colorize_image,
    inputs=gr.Image(type="pil", label="Upload Grayscale Image"),
    outputs=gr.Image(type="pil", label="Colorized Image"),
    title="Image Colorization",
    description="Upload a grayscale image to colorize it!"
)

demo.launch(share=True)
