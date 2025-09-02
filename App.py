import gradio as gr
import tensorflow as tf
import numpy as np

model=tf.keras.models.load_model('model.keras')

def recognize_image(image):
    if image is not None:
        image = image.convert("L").resize((28, 28))
        image = np.array(image).reshape(1, 28, 28, 1).astype("float32") / 255.0
        predictions = model.predict(image)
        return {str(i): float(predictions[0][i]) for i in range(10)}
    else :
       return None
    

iface = gr.Interface(
    fn=recognize_image,
    inputs=gr.Sketchpad(
        canvas_size=(28, 28),   # canvas size
        type="pil"          # return PIL image
    ),
    outputs=gr.Label(num_top_classes=5),
    live=True,
    title="Digit Recognizer"
)

iface.launch(share=True)





