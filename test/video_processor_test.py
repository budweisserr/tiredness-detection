import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import cv2
import dlib
from src.video_processor import VideoProcessor


class TestVideoProcessor(unittest.TestCase):
    def setUp(self):
        # Common test setup
        self.sound_paths = {
            'left_eye': '../image/left-eye.jpg',
            'right_eye': '../image/right-eye.jpg',
            'alert': '../sound/alert.mp3',
            'focus': '../sound/focus.mp3',
            'break': '../sound/break.mp3'
        }
        self.predictor_path = '../predictor/shape_predictor_68_face_landmarks.dat'

    @patch('cv2.VideoCapture')
    @patch('dlib.get_frontal_face_detector')
    @patch('dlib.shape_predictor')
    def test_processes_frame_when_camera_returns_valid_frame(self, mock_predictor, mock_detector, mock_video_capture):
        # Setup
        mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_capture = MagicMock()
        mock_capture.read.return_value = (True, mock_frame)
        mock_video_capture.return_value = mock_capture

        # Mock the detector to return no faces
        mock_detector_instance = MagicMock()
        mock_detector_instance.return_value = []
        mock_detector.return_value = mock_detector_instance

        # Create processor with mocked signals
        video_processor = VideoProcessor(self.predictor_path, self.sound_paths)
        video_processor.update_frame = MagicMock()
        video_processor.update_status = MagicMock()

        # Set running to False so it exits after one iteration
        video_processor.running = False

        # Run the processor
        video_processor.run()

        # Assertions
        video_processor.update_frame.emit.assert_called_once()
        video_processor.update_status.emit.assert_called_once()

    def test_calculates_ear_correctly_for_valid_eye_points(self):
        # Setup
        video_processor = VideoProcessor(self.predictor_path, self.sound_paths)
        eye = np.array([
            [0, 0], [1, 1], [2, 1], [3, 0], [2, -1], [1, -1]
        ])

        # Calculate expected EAR using the formula
        # EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
        expected_ear = (2 + 2) / (2 * 3)

        # Test
        actual_ear = video_processor.ear(eye)

        # Assert
        self.assertAlmostEqual(actual_ear, expected_ear, places=4)

    def test_detects_yawn_correctly_for_open_and_closed_mouth(self):
        # Setup
        video_processor = VideoProcessor(self.predictor_path, self.sound_paths)

        # Create a closed mouth shape
        closed_mouth = np.array([
            [0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0],
            [6, 0], [5, 0], [4, 0], [3, 0], [2, 0], [1, 0]
        ])

        # Create an open mouth shape by moving points vertically
        open_mouth = closed_mouth.copy()
        open_mouth[2] = [2, -2]  # Move up
        open_mouth[3] = [3, -2]  # Move up
        open_mouth[8] = [4, 2]  # Move down
        open_mouth[9] = [3, 2]  # Move down

        # Test
        closed_yawn_ratio = video_processor.yawn(closed_mouth)
        open_yawn_ratio = video_processor.yawn(open_mouth)

        # Assert
        self.assertLess(closed_yawn_ratio, 0.6, "Closed mouth should not be detected as yawn")
        self.assertGreater(open_yawn_ratio, 0.6, "Open mouth should be detected as yawn")

    @patch('cv2.VideoCapture')
    def test_stops_processing_when_running_is_set_to_false(self, mock_video_capture):
        # Setup
        mock_capture = MagicMock()
        mock_capture.read.return_value = (True, MagicMock())
        mock_video_capture.return_value = mock_capture

        video_processor = VideoProcessor(self.predictor_path, self.sound_paths)
        video_processor.running = False
        video_processor.alert = MagicMock()

        # Run
        video_processor.run()

        # Assert
        self.assertFalse(video_processor.alert.play.called)

    def test_euclidean_distance_calculation(self):
        # Setup
        video_processor = VideoProcessor(self.predictor_path, self.sound_paths)
        point_a = np.array([0, 0])
        point_b = np.array([3, 4])

        # Expected distance using Pythagorean theorem
        expected_distance = 5.0

        # Test
        actual_distance = video_processor.euclideanDist(point_a, point_b)

        # Assert
        self.assertEqual(actual_distance, expected_distance)

    @patch('vlc.MediaPlayer')
    def test_stop_method_stops_alert_and_thread(self, mock_media_player):
        # Setup
        video_processor = VideoProcessor(self.predictor_path, self.sound_paths)
        video_processor.alert = MagicMock()
        video_processor.wait = MagicMock()

        # Run
        video_processor.stop()

        # Assert
        self.assertFalse(video_processor.running)
        video_processor.alert.stop.assert_called_once()
        video_processor.wait.assert_called_once()

    @patch('cv2.solvePnP')
    def test_get_face_direction(self, mock_solve_pnp):
        # Setup
        video_processor = VideoProcessor(self.predictor_path, self.sound_paths)

        # Mock points for a face
        shape = np.zeros((68, 2))
        for i in range(68):
            shape[i] = [i, i]

        # Mock return value for solvePnP
        mock_solve_pnp.return_value = (True, None, np.array([[0], [5], [0]]))

        # Test
        result = video_processor.getFaceDirection(shape, (480, 640))

        # Assert
        self.assertEqual(result, 5)
        mock_solve_pnp.assert_called_once()

    @patch('cv2.imwrite')
    def test_write_eyes_function(self, mock_imwrite):
        # Setup
        video_processor = VideoProcessor(self.predictor_path, self.sound_paths)

        left_eye = np.array([
            [10, 20], [12, 18], [14, 18], [16, 20], [14, 22], [12, 22]
        ])

        right_eye = np.array([
            [30, 20], [32, 18], [34, 18], [36, 20], [34, 22], [32, 22]
        ])

        img = np.zeros((100, 100, 3), dtype=np.uint8)

        # Test
        video_processor.writeEyes(left_eye, right_eye, img)

        # Assert
        self.assertEqual(mock_imwrite.call_count, 2)
        mock_imwrite.assert_any_call(self.sound_paths['left_eye'], img[18:22, 10:16])
        mock_imwrite.assert_any_call(self.sound_paths['right_eye'], img[18:22, 30:36])


if __name__ == '__main__':
    unittest.main()