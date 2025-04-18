import os
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import json
from tqdm import tqdm

def load_sample_videos(data_dir, num_samples=5):
    """Load a small sample of videos from the dataset."""
    video_dir = Path(data_dir) / "PanAf500" / "videos"
    videos = list(video_dir.glob("*.MP4"))[:num_samples]
    return videos

def extract_frames(video_path, frame_interval=30):
    """Extract frames from video at specified intervals."""
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_interval == 0:
            frames.append(frame)
        frame_count += 1
    
    cap.release()
    return frames

def run_yolo_detection(frames, model):
    """Run YOLO detection on frames."""
    results = []
    for frame in tqdm(frames, desc="Processing frames"):
        result = model(frame, conf=0.5)  # Added confidence threshold
        results.append(result)
    return results

def visualize_results(frame, result, output_path):
    """Visualize detection results on frame."""
    annotated_frame = result[0].plot()
    cv2.imwrite(str(output_path), annotated_frame)

def main():
    # Initialize YOLO model - using YOLOv12 nano model
    model = YOLO('yolov12n.yaml')  # Using YOLOv12 nano model
    
    # Setup paths
    data_dir = Path("../data/raw")
    output_dir = Path("../data/processed/yolo_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load sample videos
    sample_videos = load_sample_videos(data_dir)
    print(f"Found {len(sample_videos)} sample videos")
    
    # Process each video
    for video_path in sample_videos:
        print(f"\nProcessing video: {video_path.name}")
        
        # Extract frames
        frames = extract_frames(video_path)
        print(f"Extracted {len(frames)} frames")
        
        # Run detection
        results = run_yolo_detection(frames, model)
        
        # Save results
        video_output_dir = output_dir / video_path.stem
        video_output_dir.mkdir(exist_ok=True)
        
        for i, (frame, result) in enumerate(zip(frames, results)):
            output_path = video_output_dir / f"frame_{i:04d}.jpg"
            visualize_results(frame, result, output_path)
        
        print(f"Results saved to: {video_output_dir}")

if __name__ == "__main__":
    main() 