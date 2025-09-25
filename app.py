import gradio as gr
import cv2
import numpy as np
import pandas as pd
from camera import VideoCamera, music_rec  # keep your camera.py imports
from tensorflow.keras.models import load_model

# Load your trained model
model = load_model("model.h5")

# Initial table data (top 15 music recommendations)
headings = ("Name", "Album", "Artist")
df1 = music_rec().head(15)

# Function to capture frame from webcam and predict stress
def capture_frame():
    cam = VideoCamera()
    frame, global_df = cam.get_frame()
    df = global_df.head(15)

    # Convert frame from BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Example: if your model uses frame to predict stress
    # Preprocess frame (resize / normalize) if needed
    # frame_input = cv2.resize(frame, (224,224)) / 255.0
    # frame_input = np.expand_dims(frame_input, axis=0)
    # prediction = model.predict(frame_input)[0]
    # stress_status = "Stressed" if prediction > 0.5 else "Not Stressed"

    # For now, just return table + frame
    return frame_rgb, df.to_dict(orient="records")

# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("## Stress Detection with Live Webcam Feed")

    with gr.Row():
        video_output = gr.Image(label="Live Webcam Feed")
        table_output = gr.Dataframe(headers=headings, label="Top Recommendations")

    capture_btn = gr.Button("Capture Frame")
    capture_btn.click(
        fn=capture_frame,
        inputs=[],
        outputs=[video_output, table_output]
    )

if __name__ == "__main__":
    demo.launch()
