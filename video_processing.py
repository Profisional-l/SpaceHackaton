import os.path

import cv2
import numpy as np
from camera_data import CameraData
import concurrent.futures

class VideoProcessor:
    """
    Класс для обработки видео
    """
    def __init__(self, video_path: str, camera_data: CameraData, video_type: str = 'rgb'):
        """
        :param video_path: Путь к видеофайлу
        :param camera_data: Данные о камере
        :param video_type: Тип видео (rgb или ir)
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f'No such file "{video_path}"')
        self.video_path = video_path
        self.camera_data = camera_data
        self.video_type = video_type
        self.video = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
        self.frame_size = (
            int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        )
        self.total_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.center_x = self.frame_size[0] // 2
        self.center_y = self.frame_size[1] // 2
        self.kalman = self._initialize_kalman_filter()
        self.object_hidden = False
        self.prev_gray_frame = None

    def _initialize_kalman_filter(self):
        """
        Инициализация фильтра Калмана
        """
        kalman = cv2.KalmanFilter(4, 2)  # 4 состояния, 2 измерения
        kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        kalman.transitionMatrix = np.eye(4, dtype=np.float32)
        kalman.statePost = np.zeros((4, 1), np.float32)  # Начальное состояние
        kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2  # Процессный шум
        kalman.measurementNoiseCov = np.array([[1, 0], [0, 1]], dtype=np.float32) * 1e-1  # Шум измерений
        return kalman

    def _kalman_filter(self, x, y):
        """
        Применение фильтра Калмана для предсказания координат
        """
        self.kalman.correct(np.array([[np.float32(x)], [np.float32(y)]]))
        predicted = self.kalman.predict()
        return predicted[0], predicted[1]

    def _preprocess_ir_frame(self, gray_ir):
        """
        Функция для улучшения изображения ИК-камеры
        """
        equalized = cv2.equalizeHist(gray_ir)  # Увеличение контраста
        denoised = cv2.medianBlur(equalized, 5)  # Уменьшение шума
        blurred = cv2.GaussianBlur(denoised, (5, 5), 0)  # Размытие
        return blurred

    def _find_ball_parameters(self, contour):
        """
        Функция для нахождения параметров шара и его диаметра
        """
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        (x, y), radius = cv2.minEnclosingCircle(approx)
        center = (int(x), int(y))
        diameter = 2 * radius
        return center, diameter

    def _process_rgb_frame(self, frame_rgb):
        """
        Функция для обработки RGB кадра
        """
        if self.prev_gray_frame is None:
            self.prev_gray_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2GRAY)
            return None, self.prev_gray_frame

        gray_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2GRAY)
        diff_rgb = cv2.absdiff(self.prev_gray_frame, gray_rgb)
        return diff_rgb, gray_rgb

    def _process_ir_frame(self, frame_ir):
        """
        Функция для обработки IR кадра
        """
        if self.prev_gray_frame is None:
            self.prev_gray_frame = cv2.cvtColor(frame_ir, cv2.COLOR_BGR2GRAY)
            processed_ir = self._preprocess_ir_frame(self.prev_gray_frame)
            return None, processed_ir

        gray_ir = cv2.cvtColor(frame_ir, cv2.COLOR_BGR2GRAY)
        processed_ir = self._preprocess_ir_frame(gray_ir)
        diff_ir = cv2.absdiff(self.prev_gray_frame, processed_ir)
        return diff_ir, processed_ir

    def get_all_coords(self) -> list[dict]:
        """
        Получение координат объекта на всех кадрах
        :return: Список координат и диаметров объекта
        """
        frame_count = 0
        coords_list = []

        with concurrent.futures.ThreadPoolExecutor() as executor:
            while True:
                ret, frame = self.video.read()
                if not ret:
                    break

                frame_count += 1
                if self.video_type == 'rgb':
                    diff, gray = self._process_rgb_frame(frame)
                else:
                    diff, gray = self._process_ir_frame(frame)

                diff_combined = diff if diff is not None else np.zeros_like(gray)
                _, mask = cv2.threshold(diff_combined, 30, 255, cv2.THRESH_BINARY)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if contours:
                    contours = [c for c in contours if 100 < cv2.contourArea(c) < 10000]
                    if not contours:
                        continue

                    largest_contour = max(contours, key=cv2.contourArea)
                    center, diameter = self._find_ball_parameters(largest_contour)

                    if diameter < 20:
                        continue

                    predicted_x, predicted_y = self._kalman_filter(center[0], center[1])
                    relative_x = predicted_x[0] - self.center_x
                    relative_y = predicted_y[0] - self.center_y

                    coords_list.append({'x': relative_x, 'y': relative_y, 'diameter': diameter})
                    self.object_hidden = False
                else:
                    coords_list.append(None)
                    self.object_hidden = True

                self.prev_gray_frame = gray.copy()

        self.video.release()
        cv2.destroyAllWindows()
        return coords_list