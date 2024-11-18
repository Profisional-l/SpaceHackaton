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
    focal_length = input("Введите фокусное расстояние камеры (мм): ")
    matrix_width = input("Введите ширину матрицы камеры (мм): ")
    matrix_height = input("Введите высоту матрицы камеры (мм): ")
    sphere_diameter = input("Введите диаметр сферы (м): ")

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
            az=az1,
            sphere_diameter=sphere_diameter
        ),
        CameraData(
            focal_length=focal_length,
            matrix_width=matrix_width,
            matrix_height=matrix_height,
            x=x2,
            y=y2,
            z=z2,
            az=az2,
            sphere_diameter=sphere_diameter
        ),
        CameraData(
            focal_length=focal_length,
            matrix_width=matrix_width,
            matrix_height=matrix_height,
            x=x3,
            y=y3,
            z=z3,
            az=az3,
            sphere_diameter=sphere_diameter
        )
    ]


def main():
    camera_info = get_camera_info()

    videoset = int(input("Введите номер видеосета (1-8): "))
    video_processors = []

    for i in range(3):
        path = os.path.abspath('tests\\step1\\videoset' + str(videoset) + f'\\Seq{i+1}_camera{i + 1}.mov')
        path_t = os.path.abspath('tests\\step1\\videoset' + str(videoset) + f'\\Seq{i+1}_camera{i + 1}T.mov')
        video_processors.append(VideoProcessor(path, camera_info[i], 'rgb'))
        video_processors.append(VideoProcessor(path_t, camera_info[i], 'ir'))

    a = video_processors[0].get_all_coords()

    print(a)

    # TODO: Обработка видео
    # Это нужно распараллелить

    # for video_processor in video_processors:
    #     smth like 'video_processor.get_all_coords()'
    #     save_somewhere(coords)
    # for coords in coords_list:
    #     get_coords_from projection(coords)
    #     save_in_excel(coords)



if __name__ == "__main__":
    main()
