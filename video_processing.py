from typing import List, Tuple
import cv2
import numpy as np
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

    def _initialize_kalman_filter(self):
        """
        Инициализация фильтра Калмана
        """
        kalman = cv2.KalmanFilter(4, 2)  # 4 состояния, 2 измерения
        kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        kalman.transitionMatrix = np.eye(4, dtype=np.float32)
        kalman.statePost = np.zeros((4, 1), np.float32)  # Начальное состояние
        return kalman

    def _kalman_filter(self, x, y):
        """
        Применение фильтра Калмана для предсказания координат
        """
        self.kalman.correct(np.array([[np.float32(x)], [np.float32(y)]]))
        predicted = self.kalman.predict()
        return predicted[0], predicted[1]

    def __get_coords_and_diameter(self, frame):
        """
        Приватная функция для получения координат объекта и диаметра на кадре
        """
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.medianBlur(gray_frame, 5)

        # Вычитание для обнаружения движения (различие между кадрами)
        if not hasattr(self, 'prev_gray_frame'):
            self.prev_gray_frame = gray_frame
            return None, None

        diff_frame = cv2.absdiff(self.prev_gray_frame, gray_frame)
        _, mask = cv2.threshold(diff_frame, 30, 255, cv2.THRESH_BINARY)

        # Находим контуры движущихся объектов
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Выбираем самый большой контур, считая, что это наш объект (шар)
            largest_contour = max(contours, key=cv2.contourArea)
            (x, y), radius = cv2.minEnclosingCircle(largest_contour)

            # Если радиус слишком мал, игнорируем объект
            if radius < 5:
                return None, None

            # Применяем фильтр Калмана
            predicted_x, predicted_y = self._kalman_filter(int(x), int(y))

            # Преобразуем координаты в систему координат относительно центра фрейма
            relative_x = predicted_x[0] - self.center_x
            relative_y = predicted_y[0] - self.center_y

            # Возвращаем координаты и диаметр (2 * радиус)
            self.prev_gray_frame = gray_frame.copy()
            # print(f"Координаты объекта: ({relative_x}, {relative_y})")
            return (relative_x, relative_y), 2 * radius
        else:
            if not self.object_hidden:
                # print("Объект скрыт из кадра")
                self.object_hidden = True
            return None, None

    def get_all_coords(self):
        """
        Получение координат объекта на всех кадрах
        :return: Список координат и диаметров объекта
        """
        frame_count = 0
        coords_list = []

        while True:
            ret, frame = self.video.read()
            if not ret:
                break

            frame_count += 1
            coords, diameter = self.__get_coords_and_diameter(frame)

            if coords:
                coords_list.append({'x': coords[0], 'y': coords[1], 'diameter': diameter})
                self.object_hidden = False
            else:
                coords_list.append(None)
                self.object_hidden = True

            # Вывод прогресса в консоль
            # progress = (frame_count / self.total_frames) * 100
            # elapsed_time = frame_count / self.video.get(cv2.CAP_PROP_FPS)
            # print(f"\nКадр: {frame_count}/{self.total_frames} ({progress:.2f}%) - Время: {elapsed_time:.2f}с", end="\r")

            # Выход по нажатию ESC
            # if cv2.waitKey(1) & 0xFF == 27:
            #     break

        self.video.release()
        cv2.destroyAllWindows()
        return coords_list
