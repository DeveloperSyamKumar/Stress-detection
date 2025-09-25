import gradio as gr
import cv2
import pandas as pd
from camera import VideoCamera, music_rec

# Initialize camera and table
cam = VideoCamera()
headings = ("Name", "Album", "Artist")
df1 = music_rec().head(15)

# Generator function to stream webcam frames
def stream_video():
    while True:
        frame, global_df = cam.get_frame()
        df = global_df.head(15)  # keep top 15
        # Convert BGR to RGB for Gradio
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Yield frame and table
        yield frame_rgb, df.to_dict(orient="records")

# Build Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("## 🎵 Stress Detection with Music Recommendation")

    with gr.Row():
        video_output = gr.Video(label="Live Webcam Feed", streaming=True)
        table_output = gr.Dataframe(headers=headings, label="Top Recommendations")

    # Start streaming
    demo.load(fn=stream_video, inputs=None, outputs=[video_output, table_output])

# Launch app
if __name__ == "__main__":
    demo.launch()
