import os
from camera_data import CameraData
from video_processing import VideoProcessor


def get_camera_info():
    """
    Получение информации о камере по индексу
    """
    # TODO: Парсинг данных из таблицы
    #! Пока что ввод данных вручную
    # Общие параметры
    focal_length = float(input("Введите фокусное расстояние камеры (мм): "))
    matrix_width = float(input("Введите ширину матрицы камеры (мм): "))
    matrix_height = float(input("Введите высоту матрицы камеры (мм): "))

    # Камера 1
    x1, y1, z1 = float(input("Введите координату X камеры 1: ")), \
                 float(input("Введите координату Y камеры 1: ")), \
                 float(input("Введите координату Z камеры 1: "))

    az1 = float(input("Введите угол азимута камеры 1: "))

    # Камера 2
    x2, y2, z2 = float(input("Введите координату X камеры 2: ")), \
                 float(input("Введите координату Y камеры 2: ")), \
                 float(input("Введите координату Z камеры 2: "))

    az2 = float(input("Введите угол азимута камеры 2: "))

    # Камера 3
    x3, y3, z3 = float(input("Введите координату X камеры 3: ")), \
                 float(input("Введите координату Y камеры 3: ")), \
                 float(input("Введите координату Z камеры 3: "))

    az3 = float(input("Введите угол азимута камеры 3: "))

    return [
        CameraData(
            focal_length=focal_length,
            matrix_width=matrix_width,
            matrix_height=matrix_height,
            x=x1,
            y=y1,
            z=z1,
            az=az1
        ),
        CameraData(
            focal_length=focal_length,
            matrix_width=matrix_width,
            matrix_height=matrix_height,
            x=x2,
            y=y2,
            z=z2,
            az=az2
        ),
        CameraData(
            focal_length=focal_length,
            matrix_width=matrix_width,
            matrix_height=matrix_height,
            x=x3,
            y=y3,
            z=z3,
            az=az3
        )
    ]


def main():
    camera_info = get_camera_info()

    videoset = int(input("Введите номер видеосета (1-8): "))
    video_processors = []

    for i in range(3):
       # Генерация правильных путей в зависимости от видеосета
        path = os.path.abspath(f'tests\\step1\\videoset{videoset}\\Seq{videoset}_camera{i + 1}.mov')
        path_t = os.path.abspath(f'tests\\step1\\videoset{videoset}\\Seq{videoset}_camera{i + 1}T.mov')

        # Создаем два VideoProcessor для каждой камеры: один для RGB, другой для IR
        video_processors.append(VideoProcessor(path, camera_info[i], 'rgb'))
        video_processors.append(VideoProcessor(path_t, camera_info[i], 'ir'))

    # Обработка видео и вывод координат в терминал
    for video_processor in video_processors:
        print(f"Обработка видео: {video_processor.video_path}")
        coords = video_processor.get_all_coords()
        for frame_coords in coords:
            x, y, distance = frame_coords
            print(f"Координаты: X={x:.2f}, Y={y:.2f}, Расстояние: {distance:.2f}м")


if __name__ == "__main__":
    main()
