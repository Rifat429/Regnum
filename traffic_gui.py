import sys
import os
import time
import psutil
import torch
import cv2

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QFrame, QGridLayout,
                             QScrollArea)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor, QPalette
from ultralytics import YOLO

# Paths to models and sample video
MODEL_PATH = r"runs\detect\traffic_monitoring\yolo26_traffic\weights\best.pt"
VIDEO_PATH = r"Dataset\Video\Supporting video for Dataset-3.mp4"

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    metrics_signal = pyqtSignal(dict)

    def __init__(self, model_path, video_path):
        super().__init__()
        self.model_path = model_path
        self.video_path = video_path
        self._run_flag = True
        self.paused = False

    def run(self):
        # Load model
        model = YOLO(self.model_path)
        cap = cv2.VideoCapture(self.video_path)
        
        prev_time = 0
        
        while self._run_flag:
            if self.paused:
                time.sleep(0.1)
                continue
                
            ret, frame = cap.read()
            if not ret:
                # Loop video
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            # Run YOLO Tracking
            #persist=True ensures IDs are maintained across frames
            results = model.track(frame, persist=True, verbose=False)[0]
            
            # Annotated frame
            annotated_frame = results.plot()
            
            # Calculate FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
            prev_time = curr_time
            
            # Get metrics
            cpu_usage = psutil.cpu_percent()
            gpu_usage = 0
            if torch.cuda.is_available():
                # Note: Rough estimate of GPU utilization if possible, or just memory
                # For simplicity, we use torch.cuda.utilization() if available in newer torch or just 0
                try:
                    gpu_usage = torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 0
                except:
                    gpu_usage = 0
            
            # Object counts
            counts = {}
            if results.boxes.id is not None:
                classes = results.boxes.cls.cpu().numpy()
                for cls_idx in classes:
                    cls_name = model.names[int(cls_idx)]
                    counts[cls_name] = counts.get(cls_name, 0) + 1

            metrics = {
                "fps": fps,
                "cpu": cpu_usage,
                "gpu": gpu_usage,
                "counts": counts
            }
            
            self.change_pixmap_signal.emit(annotated_frame)
            self.metrics_signal.emit(metrics)
            
        cap.release()

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()

    def toggle_pause(self):
        self.paused = not self.paused

class TrafficMonitorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Traffic Monitoring System")
        self.setMinimumSize(1200, 800)
        
        # UI Styling
        self.setStyleSheet("""
            QMainWindow {
                background-color: #121212;
            }
            QLabel {
                color: #e0e0e0;
                font-family: 'Segoe UI', Arial;
            }
            QPushButton {
                background-color: #333333;
                color: white;
                border-radius: 5px;
                padding: 10px;
                font-size: 14px;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #444444;
            }
            QPushButton#startButton {
                background-color: #2e7d32;
            }
            QPushButton#startButton:hover {
                background-color: #388e3c;
            }
            QPushButton#stopButton {
                background-color: #c62828;
            }
            QPushButton#stopButton:hover {
                background-color: #d32f2f;
            }
            QFrame#card {
                background-color: #1e1e1e;
                border-radius: 10px;
                padding: 15px;
            }
        """)

        # Main Widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Left Side: Video Feed
        self.video_container = QVBoxLayout()
        self.video_label = QLabel("Click 'Start' to Load Video Feed")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black; border-radius: 10px;")
        self.video_container.addWidget(self.video_label)
        
        # Controls below video
        control_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start Feed")
        self.start_btn.setObjectName("startButton")
        self.start_btn.clicked.connect(self.start_feed)
        
        self.stop_btn = QPushButton("Stop Feed")
        self.stop_btn.setObjectName("stopButton")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_feed)
        
        control_layout.addStretch()
        control_layout.addWidget(self.start_btn)
        control_layout.addWidget(self.stop_btn)
        control_layout.addStretch()
        
        self.video_container.addLayout(control_layout)
        main_layout.addLayout(self.video_container, stretch=3)

        # Right Side: Dashboard
        self.dashboard = QVBoxLayout()
        
        # Header
        header = QLabel("DASHBOARD")
        header.setFont(QFont("Segoe UI", 18, QFont.Bold))
        self.dashboard.addWidget(header)
        
        # System Metrics Card
        sys_card = QFrame()
        sys_card.setObjectName("card")
        sys_layout = QVBoxLayout(sys_card)
        
        self.fps_label = QLabel("FPS: 0.0")
        self.cpu_label = QLabel("CPU Utilization: 0%")
        self.gpu_label = QLabel("GPU Utilization: 0%")
        
        for lbl in [self.fps_label, self.cpu_label, self.gpu_label]:
            lbl.setFont(QFont("Segoe UI", 12))
            sys_layout.addWidget(lbl)
            
        self.dashboard.addWidget(sys_card)
        
        # Object Counts Card with Scroll Area
        count_card = QFrame()
        count_card.setObjectName("card")
        count_layout = QVBoxLayout(count_card)
        count_title = QLabel("OBJECT COUNTS")
        count_title.setFont(QFont("Segoe UI", 12, QFont.Bold))
        count_layout.addWidget(count_title)
        
        # Scroll Area for counts
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("background-color: transparent; border: none;")
        
        scroll_content = QWidget()
        scroll_content.setStyleSheet("background-color: transparent;")
        self.count_labels_container = QVBoxLayout(scroll_content)
        self.count_labels_container.setAlignment(Qt.AlignTop)
        self.count_labels_container.setSpacing(5)
        
        scroll.setWidget(scroll_content)
        count_layout.addWidget(scroll)
        
        self.dashboard.addWidget(count_card)
        self.dashboard.addStretch()
        
        main_layout.addLayout(self.dashboard, stretch=1)

        # State
        self.thread = None
        self.count_labels = {}

    def start_feed(self):
        if self.thread is None:
            self.thread = VideoThread(MODEL_PATH, VIDEO_PATH)
            self.thread.change_pixmap_signal.connect(self.update_image)
            self.thread.metrics_signal.connect(self.update_metrics)
            self.thread.start()
            
            self.start_btn.setText("Pause")
            self.start_btn.setStyleSheet("background-color: #f57c00;")
            self.stop_btn.setEnabled(True)
        else:
            self.thread.toggle_pause()
            if self.thread.paused:
                self.start_btn.setText("Resume")
            else:
                self.start_btn.setText("Pause")

    def stop_feed(self):
        if self.thread:
            self.thread.stop()
            self.thread = None
            self.video_label.setPixmap(QPixmap())
            self.video_label.setText("Feed Stopped")
            self.start_btn.setText("Start Feed")
            self.start_btn.setStyleSheet("")
            self.stop_btn.setEnabled(False)

    def update_image(self, cv_img):
        """Updates the video_label with a new opencv image"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.video_label.width(), self.video_label.height(), Qt.KeepAspectRatio)
        self.video_label.setPixmap(QPixmap.fromImage(p))

    def update_metrics(self, metrics):
        self.fps_label.setText(f"FPS: {metrics['fps']:.1f}")
        self.cpu_label.setText(f"CPU Utilization: {metrics['cpu']:.1f}%")
        self.gpu_label.setText(f"GPU Utilization: {metrics['gpu']:.1f}%")
        
        # Update object counts
        counts = metrics['counts']
        
        # Update object counts
        counts = metrics['counts']
        
        for name, count in counts.items():
            if name not in self.count_labels:
                label = QLabel(f"{name}: {count}")
                label.setFont(QFont("Segoe UI", 10))
                label.setStyleSheet("padding: 2px;")
                self.count_labels[name] = label
                self.count_labels_container.addWidget(label)
            else:
                self.count_labels[name].setText(f"{name}: {count}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TrafficMonitorApp()
    window.show()
    sys.exit(app.exec())
