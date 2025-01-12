import streamlit as st
import torch
import torch.nn as nn
import cv2
import numpy as np
from huggingface_hub import hf_hub_download
import tempfile
import os

# 1. First define your model classes
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    # [Your existing UNet class implementation]
    # Copy the entire UNet class from your code here

# 2. Add model loading function with caching
@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet().to(device)
    weights_path = hf_hub_download(
        repo_id="vinster619/UNet_USC_TIMIT",
        filename="best_unet_model.pth"
    )
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, device

# 3. Video processing function
def process_video(video_file):
    model, device = load_model()
    
    # Create temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') 
    tfile.write(video_file.read())
    
    # Process video
    cap = cv2.VideoCapture(tfile.name)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create output video writer
    output_path = "processed_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (64, 64))
    
    # Add progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    frame_counter = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Update progress
        progress = frame_counter/total_frames
        progress_bar.progress(progress)
        status_text.text(f'Processing frame {frame_counter}/{total_frames}')
        
        # Process frame
        frame_resized = cv2.resize(frame, (64, 64))
        frame_normalized = frame_resized / 255.0
        frame_tensor = torch.from_numpy(frame_normalized).permute(2, 0, 1).unsqueeze(0).float().to(device)

        with torch.no_grad():
            predictions = model(frame_tensor)

        # Create visualization
        vis_frame = frame_resized.copy()
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]
        for pred, color in zip(predictions, colors):
            mask = torch.argmax(pred, dim=1).squeeze().cpu().numpy()
            contours, _ = cv2.findContours(
                mask.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(vis_frame, contours, -1, color, 1)

        out.write(vis_frame)
        frame_counter += 1

    cap.release()
    out.release()
    os.unlink(tfile.name)
    
    return output_path

# 4. Streamlit UI
def main():
    st.set_page_config(page_title="rtMRI Video Segmentation", layout="wide")
    
    st.title("rtMRI Video Segmentation")
    st.write("Upload your rtMRI video to generate anatomical segmentations")

    uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'avi'])

    if uploaded_file is not None:
        st.video(uploaded_file)
        
        if st.button('Process Video'):
            with st.spinner('Processing video...'):
                try:
                    output_path = process_video(uploaded_file)
                    
                    # Display and allow download of processed video
                    with open(output_path, 'rb') as f:
                        st.download_button(
                            label="Download processed video",
                            data=f,
                            file_name="processed_video.mp4",
                            mime="video/mp4"
                        )
                    
                    # Display the processed video
                    st.video(output_path)
                    
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

    # Add color legend
    st.markdown("""
    ### Color Legend:
    - 🔴 Red: Contour 1 (Upper airway boundary)
    - 🔵 Blue: Contour 2 (Lower airway boundary)
    - 🟢 Green: Contour 3 (Pharyngeal wall)
    """)

if __name__ == "__main__":
    main()
