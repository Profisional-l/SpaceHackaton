from typing import List, Tuple
import cv2
import numpy as np
from camera_data import CameraData


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
        self.video_path = video_path
        self.camera_data = camera_data
        self.video_type = video_type

        # Открытие видеофайла
        self.video_capture = cv2.VideoCapture(video_path)
        if not self.video_capture.isOpened():
            raise ValueError(f"Ошибка при открытии видео: {video_path}")

        self.frame_count = 0
        self.prev_gray_frame = None

    def __next__(self) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Получение следующего кадра и координат объекта
        """
        ret, frame = self.video_capture.read()
        if not ret:
            raise StopIteration
        
        self.frame_count += 1
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.medianBlur(gray_frame, 5)

        # Получаем координаты объекта
        center_x, center_y = self.__get_coords(gray_frame)
        return frame, (center_x, center_y)

    def get_frame_size(self) -> Tuple[int, int]:
        """
        Получение размера кадра
        """
        return int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def __get_coords(self, gray_frame: np.ndarray) -> Tuple[int, int]:
        """
        Приватная функция для получения координат объекта на кадре
        """
        if self.prev_gray_frame is None:
            self.prev_gray_frame = gray_frame
            return None, None

        diff_frame = cv2.absdiff(self.prev_gray_frame, gray_frame)
        _, mask = cv2.threshold(diff_frame, 30, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            self.prev_gray_frame = gray_frame
            return x + w // 2, y + h // 2
        return None, None

    def get_all_coords(self) -> List[Tuple[float, float, float]]:
        """
        Получение координат объекта на всех кадрах
        """
        coords = []
        while True:
            try:
                frame, (x, y) = next(self)
                if x is not None and y is not None:
                    distance = self.calculate_distance(x)
                    coords.append((x, y, distance))
            except StopIteration:
                break
        return coords

    def calculate_distance(self, pixel_width: float) -> float:
        """
        Вычисление расстояния до объекта по ширине в пикселях
        :param pixel_width: Ширина объекта в пикселях
        :return: Расстояние до объекта
        """
        if pixel_width > 0:
            return (self.camera_data.focal_length * 0.15) / pixel_width  # Примерная формула для расчета расстояния
        return 0

    def release(self):
        """
        Освобождение ресурсов
        """
        self.video_capture.release()
