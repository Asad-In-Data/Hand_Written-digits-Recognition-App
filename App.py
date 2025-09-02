import gradio as gr
import tensorflow as tf
import numpy as np

model=tf.keras.models.load_model('model.keras')

def recognize_image(image):
    if image is not None:
        image= image.reshape(1,28,28,1).astype("float")/255
        image = tf.expand_dims(image, axis=0)
        image = tf.cast(image, tf.float32) / 255.0
        predictions = model.predict(image)
        return {str(i): float(predictions[0][i]) for i in range(10)}
    else :
       return None
    

iface= gr.Interface(fn=recognize_image, 
        inputs=gr.Image(width=28, height=28,
        image_mode='L',
        invert_colors=True,
        source='canvas'),
        outputs=gr.Label(num_top_classes=5),
        live=True,
        title="Digit Recognizer",


)

iface.launch(share=True)


