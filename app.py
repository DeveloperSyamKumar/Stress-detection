import gradio as gr
import cv2
import pandas as pd
from camera import VideoCamera, music_rec
from tensorflow.keras.models import load_model
import numpy as np

# Initialize camera and model
cam = VideoCamera()
model = load_model("model.h5")

# Table headings
headings = ("Name", "Album", "Artist")

# Generator function for streaming frames
def stream_video():
    while True:
        frame, df = cam.get_frame()
        df_top = df.head(15)  # keep top 15
        # Convert frame BGR -> RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Example: if using frame to predict stress
        # frame_input = cv2.resize(frame, (224,224)) / 255.0
        # frame_input = np.expand_dims(frame_input, axis=0)
        # prediction = model.predict(frame_input)[0]
        # stress_status = "Stressed" if prediction > 0.5 else "Not Stressed"

        yield frame_rgb, df_top.to_dict(orient="records")

# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("## Stress Detection with Live Webcam Feed & Music Recommendations")

    with gr.Row():
        video_output = gr.Image(label="Live Webcam Feed")
        table_output = gr.Dataframe(headers=headings, label="Top Recommendations")

    # Start streaming when the app loads
    demo.load(fn=stream_video, inputs=None, outputs=[video_output, table_output])

if __name__ == "__main__":
    demo.launch()
