from PyQt6.QtWidgets import (QMainWindow, QLabel, QVBoxLayout,
                            QHBoxLayout, QWidget, QPushButton, QProgressBar)
from PyQt6.QtGui import QImage, QPixmap, QFont
from PyQt6.QtCore import Qt
import cv2
from video_processor import VideoProcessor

class DrowsinessDetectionUI(QMainWindow):
    def __init__(self, config):
        super().__init__()
        self.video_label = None
        self.overlay_label = None
        self.config = config
        self.video_thread = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Driver Drowsiness Detection")
        self.setGeometry(100, 100, 1000, 600)
        self.setStyleSheet("background-color: #2c3e50;")

        # Main layout
        main_layout = QHBoxLayout()

        # Left panel (video feed) - now a container widget with stacked labels
        self.video_container = QWidget()
        self.video_container.setMinimumSize(640, 480)
        self.video_container.setStyleSheet("border: 2px solid #3498db; background-color: #34495e;")
        video_container_layout = QVBoxLayout(self.video_container)
        video_container_layout.setContentsMargins(0, 0, 0, 0)  # Remove margins

        # Video label
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        video_container_layout.addWidget(self.video_label)

        # Overlay label (initially hidden)
        self.overlay_label = QLabel("Video Stopped")
        self.overlay_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.overlay_label.setStyleSheet(
            "color: #ecf0f1; font-size: 18pt; font-weight: bold; background-color: transparent;")
        self.overlay_label.setVisible(True)  # Initially visible with "Video Stopped" message
        video_container_layout.addWidget(self.overlay_label)

        # Right panel (controls and status)
        right_panel = QVBoxLayout()

        # App title
        title_label = QLabel("Driver Drowsiness\nDetection")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        title_label.setStyleSheet("color: #ecf0f1; margin: 10px;")

        # Status section
        status_box = QWidget()
        status_box.setStyleSheet("background-color: #34495e; border-radius: 10px; padding: 15px;")
        status_layout = QVBoxLayout(status_box)

        status_title = QLabel("Status")
        status_title.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        status_title.setStyleSheet("color: #ecf0f1;")

        self.status_label = QLabel("Video thread not started")
        self.status_label.setFont(QFont("Arial", 20, QFont.Weight.Bold))
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("color: #2ecc71; margin: 10px;")

        # Alertness bar
        alert_label = QLabel("Alertness Level:")
        alert_label.setStyleSheet("color: #ecf0f1;")
        self.alertness_bar = QProgressBar()
        self.alertness_bar.setRange(0, 100)
        self.alertness_bar.setValue(0)
        self.alertness_bar.setStyleSheet("""
            QProgressBar {
                background-color: #34495e;
                border: 1px solid #7f8c8d;
                border-radius: 5px;
                text-align: center;
                height: 25px;
            }
            QProgressBar::chunk {
                background-color: #2ecc71;
                border-radius: 5px;
            }
        """)

        # Eye status
        eye_label = QLabel("Eye Status:")
        eye_label.setStyleSheet("color: #ecf0f1;")
        self.eye_status = QLabel("Invalid")
        self.eye_status.setFont(QFont("Arial", 12))
        self.eye_status.setStyleSheet("color: #2ecc71;")

        # Yawn status
        yawn_label = QLabel("Yawn Status:")
        yawn_label.setStyleSheet("color: #ecf0f1;")
        self.yawn_status = QLabel("Invalid")
        self.yawn_status.setFont(QFont("Arial", 12))
        self.yawn_status.setStyleSheet("color: #2ecc71;")

        # Add widgets to status layout
        status_layout.addWidget(status_title)
        status_layout.addWidget(self.status_label)
        status_layout.addWidget(alert_label)
        status_layout.addWidget(self.alertness_bar)
        status_layout.addWidget(eye_label)
        status_layout.addWidget(self.eye_status)
        status_layout.addWidget(yawn_label)
        status_layout.addWidget(self.yawn_status)

        # Buttons
        button_layout = QHBoxLayout()

        self.start_button = QPushButton("Start")
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #2ecc71;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #27ae60;
            }
        """)

        self.stop_button = QPushButton("Stop")
        self.stop_button.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
        """)

        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)

        # Add all components to the right panel
        right_panel.addWidget(title_label)
        right_panel.addWidget(status_box)
        right_panel.addStretch(1)
        right_panel.addLayout(button_layout)

        # Add panels to main layout
        main_layout.addWidget(self.video_label, 2)
        main_layout.addLayout(right_panel, 1)

        # Set central widget
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # Connect buttons
        self.start_button.clicked.connect(self.start_video)
        self.stop_button.clicked.connect(self.stop_video)
        self.stop_button.setEnabled(False)

    def start_video(self):
        if self.video_thread is None:
            self.video_thread = VideoProcessor(
                self.config['predictor_path'],
                {
                    'left_eye': self.config['left_eye_path'],
                    'right_eye': self.config['right_eye_path'],
                    'alert': self.config['alert_sound'],
                    'focus': self.config['focus_sound'],
                    'break': self.config['break_sound']
                }
            )

        self.video_thread.update_frame.connect(self.update_frame)
        self.video_thread.update_status.connect(self.update_status)
        self.video_thread.start()

        # Hide the overlay label when video starts ??????
        if self.overlay_label:
            self.overlay_label.setVisible(False)

        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

    def stop_video(self):
        if self.video_thread is not None:
            self.video_thread.stop()  # custom stop method — make sure it sets a flag to break the loop
            self.video_thread.wait()  # <--- this is critical: block until thread exits!
            self.video_thread = None

        # Clear the video label ?????
        if self.video_label:
            self.video_label.clear()

        # Show the overlay label with "Video Stopped" message
        if self.overlay_label:
            self.overlay_label.setText("Video Stopped")
            self.overlay_label.setVisible(True)

        self.status_label.setText("Video thread stopped")
        self.alertness_bar.setValue(0)
        self.eye_status.setText("Invalid")
        self.yawn_status.setText("Invalid")

        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def update_frame(self, frame):
        # Hide overlay label when frames are being updated
        if self.overlay_label:
            self.overlay_label.setVisible(False)

        if self.video_label is None:
            return

        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(q_image).scaled(
            self.video_container.width(), self.video_container.height(),
            Qt.AspectRatioMode.KeepAspectRatio))

    def update_status(self, status):
        if self.video_thread is None:
            return
        # Update alertness bar based on EAR
        ear_value = status["ear"]
        alertness = min(100, max(0, int((ear_value - 0.15) * 200)))
        self.alertness_bar.setValue(alertness)

        # Update color based on alert level
        if status["alert_level"] == 0:
            self.status_label.setText("Normal")
            self.status_label.setStyleSheet("color: #2ecc71; font-weight: bold; font-size: 20pt;")
            self.alertness_bar.setStyleSheet("""
                QProgressBar {
                    background-color: #34495e;
                    border: 1px solid #7f8c8d;
                    border-radius: 5px;
                    text-align: center;
                }
                QProgressBar::chunk {
                    background-color: #2ecc71;
                    border-radius: 5px;
                }
            """)
        elif status["alert_level"] == 1:
            self.status_label.setText("Drowsy")
            self.status_label.setStyleSheet("color: #f39c12; font-weight: bold; font-size: 20pt;")
            self.alertness_bar.setStyleSheet("""
                QProgressBar {
                    background-color: #34495e;
                    border: 1px solid #7f8c8d;
                    border-radius: 5px;
                    text-align: center;
                }
                QProgressBar::chunk {
                    background-color: #f39c12;
                    border-radius: 5px;
                }
            """)
        elif status["alert_level"] >= 2:
            self.status_label.setText(status["message"])
            self.status_label.setStyleSheet("color: #e74c3c; font-weight: bold; font-size: 20pt;")
            self.alertness_bar.setStyleSheet("""
                QProgressBar {
                    background-color: #34495e;
                    border: 1px solid #7f8c8d;
                    border-radius: 5px;
                    text-align: center;
                }
                QProgressBar::chunk {
                    background-color: #e74c3c;
                    border-radius: 5px;
                }
            """)

        # Update eye status
        if ear_value < self.video_thread.close_thresh:
            self.eye_status.setText("Closed")
            self.eye_status.setStyleSheet("color: #e74c3c; font-weight: bold;")
        else:
            self.eye_status.setText("Open")
            self.eye_status.setStyleSheet("color: #2ecc71; font-weight: bold;")

        # Update yawn status
        if status["yawning"]:
            self.yawn_status.setText("Yawning")
            self.yawn_status.setStyleSheet("color: #f39c12; font-weight: bold;")
        else:
            self.yawn_status.setText("Not Yawning")
            self.yawn_status.setStyleSheet("color: #2ecc71; font-weight: bold;")

    def closeEvent(self, event):
        try:
            self.stop_video()
            event.accept()
        except:
            pass