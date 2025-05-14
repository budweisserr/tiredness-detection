import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Add the parent directory to path so we can import the modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest
from src import ui as DrowsinessDetectionUI
from src.config import APP_CONFIG

# Create QApplication instance for testing
app = QApplication(sys.argv)


class TestDrowsinessDetectionUI(unittest.TestCase):
    def setUp(self):
        """Setup runs before each test method"""
        self.ui = DrowsinessDetectionUI(APP_CONFIG)

    def test_initial_state(self):
        # Check initial state of UI elements
        self.assertEqual(self.ui.status_label.text(), "Normal")
        self.assertEqual(self.ui.alertness_bar.value(), 100)
        self.assertEqual(self.ui.eye_status.text(), "Open")
        self.assertEqual(self.ui.yawn_status.text(), "Not Yawning")

        # Check initial button states
        self.assertTrue(self.ui.start_button.isEnabled())
        self.assertFalse(self.ui.stop_button.isEnabled())

    @patch('video_processor.VideoProcessor')
    def test_start_stop_buttons(self, mock_video_processor):
        # Create mock for video thread
        mock_thread = MagicMock()
        mock_video_processor.return_value = mock_thread

        # Click start button
        QTest.mouseClick(self.ui.start_button, Qt.MouseButton.LeftButton)

        # Check button states after clicking start
        self.assertFalse(self.ui.start_button.isEnabled())
        self.assertTrue(self.ui.stop_button.isEnabled())

        # Click stop button
        QTest.mouseClick(self.ui.stop_button, Qt.MouseButton.LeftButton)

        # Check button states after clicking stop
        self.assertTrue(self.ui.start_button.isEnabled())
        self.assertFalse(self.ui.stop_button.isEnabled())

        # Verify video processor was stopped
        self.assertTrue(mock_thread.stop.called)

    def test_update_status(self):
        # Create mock video thread with properties
        self.ui.video_thread = MagicMock()
        self.ui.video_thread.close_thresh = 0.3

        # Test normal status
        normal_status = {
            "alert_level": 0,
            "ear": 0.4,
            "yawning": False,
            "message": "Normal"
        }
        self.ui.update_status(normal_status)
        self.assertEqual(self.ui.status_label.text(), "Normal")
        self.assertEqual(self.ui.eye_status.text(), "Open")
        self.assertEqual(self.ui.yawn_status.text(), "Not Yawning")

        # Test drowsy status
        drowsy_status = {
            "alert_level": 1,
            "ear": 0.25,
            "yawning": False,
            "message": "Drowsy"
        }
        self.ui.update_status(drowsy_status)
        self.assertEqual(self.ui.status_label.text(), "Drowsy")
        self.assertEqual(self.ui.eye_status.text(), "Closed")

        # Test yawning status
        yawning_status = {
            "alert_level": 2,
            "ear": 0.4,
            "yawning": True,
            "message": "Sleepy (Body Posture)"
        }
        self.ui.update_status(yawning_status)
        self.assertEqual(self.ui.status_label.text(), "Sleepy (Body Posture)")
        self.assertEqual(self.ui.yawn_status.text(), "Yawning")


if __name__ == '__main__':
    unittest.main()