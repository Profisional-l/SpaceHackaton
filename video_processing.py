from os import PathLike
from typing import AnyStr

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
        pass

    def __next__(self):
        """
        Получение следующего кадра (для обработки внутри функции)
        """
        pass

    def get_frame_size(self):
        """
        Получение размера кадра
        """
        pass

    def __get_coords(self):
        """
        Получение координат объекта на кадре
        Приватная функция
        """
        pass

    def get_all_coords(self):
        """
        Получение координат объекта на всех кадрах
        """
        # for i in enumerate(self):
        #     ...
        pass
