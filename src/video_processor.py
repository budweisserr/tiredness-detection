import cv2
import math
import numpy as np
import dlib
from imutils import face_utils
import vlc
from PyQt6.QtCore import QThread, pyqtSignal


class VideoProcessor(QThread):
    update_frame = pyqtSignal(np.ndarray)
    update_status = pyqtSignal(dict)

    def __init__(self, predictor_path, sound_paths):
        super().__init__()
        self.running = True

        # Paths
        self.eye_images = {
            'left': sound_paths['left_eye'],
            'right': sound_paths['right_eye']
        }

        # Sounds
        self.sounds = {
            'alert': vlc.MediaPlayer(sound_paths['alert']),
            'focus': vlc.MediaPlayer(sound_paths['focus']),
            'break': vlc.MediaPlayer(sound_paths['break'])
        }

        # Initialize detection variables
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)
        (self.leStart, self.leEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (self.reStart, self.reEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        (self.mStart, self.mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

        # Drowsiness detection parameters
        self.alert = self.sounds['focus']
        self.frame_thresh_1 = 15
        self.frame_thresh_2 = 10
        self.frame_thresh_3 = 5
        self.close_thresh = 0.3
        self.flag = 0
        self.yawn_countdown = 0
        self.map_counter = 0
        self.map_flag = 1
        self.avgEAR = 0

    def run(self):
        capture = cv2.VideoCapture(0)
        self.running = True

        while self.running:
            ret, frame = capture.read()
            if not ret:
                continue

            size = frame.shape
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            color_frame = frame.copy()

            rects = self.detector(gray, 0)
            status = {
                "alert_level": 0,  # 0: normal, 1: mild, 2: moderate, 3: severe
                "ear": 0,
                "yawning": False,
                "message": "Normal"
            }

            if len(rects):
                shape = face_utils.shape_to_np(self.predictor(gray, rects[0]))
                leftEye = shape[self.leStart:self.leEnd]
                rightEye = shape[self.reStart:self.reEnd]
                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                mouth = shape[self.mStart:self.mEnd]
                mouthHull = cv2.convexHull(mouth)

                leftEAR = self.ear(leftEye)
                rightEAR = self.ear(rightEye)
                self.avgEAR = (leftEAR + rightEAR) / 2.0
                status["ear"] = self.avgEAR

                eyeContourColor = (0, 255, 0)  # Default: green

                yawn_ratio = self.yawn(mouth)
                if yawn_ratio > 0.6:
                    status["yawning"] = True
                    self.yawn_countdown = 1
                    cv2.drawContours(color_frame, [mouthHull], -1, (0, 0, 255), 2)
                else:
                    cv2.drawContours(color_frame, [mouthHull], -1, (0, 255, 0), 1)

                if self.avgEAR < self.close_thresh:
                    self.flag += 1

                    if self.yawn_countdown and self.flag >= self.frame_thresh_3:
                        eyeContourColor = (147, 20, 255)  # Purple
                        status["alert_level"] = 3
                        status["message"] = "Сонність (позіхання)"
                        self.alert.play()
                        if self.map_flag:
                            self.map_flag = 0
                            self.map_counter += 1
                    elif self.flag >= self.frame_thresh_2 and self.getFaceDirection(shape, size) < 0:
                        eyeContourColor = (255, 0, 0)  # Blue
                        status["alert_level"] = 2
                        status["message"] = "Сонність"
                        self.alert.play()
                        if self.map_flag:
                            self.map_flag = 0
                            self.map_counter += 1
                    elif self.flag >= self.frame_thresh_1:
                        eyeContourColor = (0, 0, 255)  # Red
                        status["alert_level"] = 1
                        status["message"] = "Сонність (закриті очі)"
                        self.alert.play()
                        if self.map_flag:
                            self.map_flag = 0
                            self.map_counter += 1
                elif self.avgEAR > self.close_thresh and self.flag:
                    self.alert.stop()
                    self.yawn_countdown = 0
                    self.map_flag = 1
                    self.flag = 0

                if self.map_counter >= 3:
                    self.map_flag = 1
                    self.map_counter = 0
                    self.sounds['break'].play()
                    status["message"] = "TAKE A BREAK NOW"

                cv2.drawContours(color_frame, [leftEyeHull], -1, eyeContourColor, 2)
                cv2.drawContours(color_frame, [rightEyeHull], -1, eyeContourColor, 2)
                self.writeEyes(leftEye, rightEye, frame)

            if self.avgEAR > self.close_thresh:
                self.alert.stop()

            self.update_frame.emit(color_frame)
            self.update_status.emit(status)

    def stop(self):
        self.running = False
        self.alert.stop()
        self.quit()
        self.wait()

    def ear(self, eye):
        return (self.euclideanDist(eye[1], eye[5]) + self.euclideanDist(eye[2], eye[4])) / (
                2 * self.euclideanDist(eye[0], eye[3]))

    def yawn(self, mouth):
        return ((self.euclideanDist(mouth[2], mouth[10]) + self.euclideanDist(mouth[4], mouth[8])) /
                (2 * self.euclideanDist(mouth[0], mouth[6])))

    def euclideanDist(self, a, b):
        return math.sqrt(math.pow(a[0] - b[0], 2) + math.pow(a[1] - b[1], 2))

    def getFaceDirection(self, _shape, _size):
        image_points = np.array([
            _shape[33],  # Nose tip
            _shape[8],  # Chin
            _shape[45],  # Left eye left corner
            _shape[36],  # Right eye right corner
            _shape[54],  # Left Mouth corner
            _shape[48]  # Right mouth corner
        ], dtype="double")

        # 3D model points
        model_points = np.array([
            (0.0, 0.0, 0.0),  # Nose tip
            (0.0, -330.0, -65.0),  # Chin
            (-225.0, 170.0, -135.0),  # Left eye left corner
            (225.0, 170.0, -135.0),  # Right eye right corner
            (-150.0, -150.0, -125.0),  # Left Mouth corner
            (150.0, -150.0, -125.0)  # Right mouth corner
        ])

        # Camera internals
        focal_length = _size[1]
        center = (_size[1] / 2, _size[0] / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )
        dist_coefs = np.zeros((4, 1))  # Assuming no lens distortion
        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                      dist_coefs, flags=cv2.SOLVEPNP_ITERATIVE)
        return translation_vector[1][0]

    def writeEyes(self, left_eye, right_eye, img):
        def save_eye_image(eye, file_name):
            y1, y2 = max(eye[1][1], eye[2][1]), min(eye[4][1], eye[5][1])
            x1, x2 = eye[0][0], eye[3][0]
            if y2 > y1 and x2 > x1:
                cv2.imwrite(file_name, img[y1:y2, x1:x2])

        save_eye_image(left_eye, self.eye_images['left'])
        save_eye_image(right_eye, self.eye_images['right'])