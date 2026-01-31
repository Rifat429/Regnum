from ultralytics import YOLO
import cv2
import psutil
import torch
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def test_tracking():
    model_path = r"runs\detect\traffic_monitoring\yolo26_traffic\weights\best.pt"
    video_path = r"Dataset\Video\Supporting video for Dataset-3.mp4"
    
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    
    ret, frame = cap.read()
    if not ret:
        print("Failed to read video")
        return

    # Test tracking on 5 frames
    for i in range(5):
        ret, frame = cap.read()
        if not ret: break
        
        results = model.track(frame, persist=True, verbose=False)[0]
        
        print(f"Frame {i}: Detected {len(results.boxes)} objects")
        if results.boxes.id is not None:
            ids = results.boxes.id.cpu().numpy().astype(int)
            print(f"Tracking IDs: {ids}")
        else:
            print("No tracking IDs assigned yet")

    cap.release()
    
    print(f"CPU usage: {psutil.cpu_percent()}%")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU not available")

if __name__ == "__main__":
    test_tracking()
