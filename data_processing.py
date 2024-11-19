import numpy as np

from camera_data import CameraData


def get_coords_from_projection(coords: tuple[int, int], frame_size: tuple[int, int], sphere_diameter, camera_data: CameraData) -> (tuple[float, float, float], float):
    """
    Получение координат объекта по его проекции на кадре
    :param coords: Координаты объекта на кадре
    :param frame_size: Размер кадра
    :param sphere_diameter: Диаметр сферы
    :param camera_data: Данные о камере
    :return: Координаты объекта в пространстве и предел погрешности
    """
    # TODO: Реализовать функцию
    f = camera_data.focal_length * 1e-3
    w, h = camera_data.matrix_width * 1e-3, camera_data.matrix_height * 1e-3

    alpha = np.sqrt(w * h / frame_size[0] / frame_size[1])
    d = alpha * sphere_diameter

    z_c = (f * camera_data.sphere_diameter) / d
    x_c = coords[0] * z_c / f * alpha
    y_c = coords[1] * z_c / f * alpha

    x = x_c * np.cos(np.deg2rad(camera_data.az)) - z_c * np.sin(np.deg2rad(camera_data.az)) + camera_data.x
    z = x_c * np.sin(np.deg2rad(camera_data.az)) + z_c * np.cos(np.deg2rad(camera_data.az)) + camera_data.z
    y = y_c + camera_data.y

    dz_c = np.max([np.abs(f * camera_data.sphere_diameter / (d * (1-(a/2)/sphere_diameter)) - f * camera_data.sphere_diameter / (d * (1+(a/2)/sphere_diameter))) for a in range(1, 5)])
    dx_c = coords[0] / f * dz_c * alpha
    dy_c = coords[1] / f * dz_c * alpha

    delta = np.sqrt(dx_c**2 + dy_c**2 + dz_c**2)

    return (x, y, z), delta

def calculate_sphere_coordinates(
    # camera_coords,        # Координаты камеры (x_c, y_c, z_c)
    # azimuth_angle,        # Угол азимута камеры (phi_c в радианах)
    # frame_resolution,     # Разрешение фрейма (w, h)
    # sphere_projection,    # Координаты и диаметр шара в пикселях (x_s', y_s', d_s')
    # sphere_diameter,      # Реальный диаметр шара (D_s в метрах)
    # focal_length,         # Фокусное расстояние камеры (f в метрах)
    # matrix_size           # Размеры матрицы камеры (W_m, H_m в метрах)
    camera_data: CameraData,
    frame_resolution: tuple[int, int],
    sphere_projection: tuple[int, int, int]
) -> (tuple[float, float, float], float):
    """
    Расчет координат объекта в пространстве по его проекции на кадре
    :param camera_data: Данные о камере
    :param frame_resolution: Разрешение фрейма (w, h)
    :param sphere_projection: Координаты и диаметр шара в пикселях (x_s', y_s', d_s')
    """

    # Распаковка входных данных
    x_c, y_c, z_c = camera_data.x, camera_data.y, camera_data.z
    w, h = frame_resolution
    x_s_prime, y_s_prime, d_s_prime = sphere_projection
    D_s = camera_data.sphere_diameter
    W_m, H_m = camera_data.matrix_width * 1e-3, camera_data.matrix_height * 1e-3
    phi_c = np.radians(camera_data.az)
    focal_length = camera_data.focal_length * 1e-3

    # 1. Перевод координат из пикселей в метры
    p_x = W_m / w
    p_y = H_m / h
    x_s_double_prime = x_s_prime * p_x
    y_s_double_prime = y_s_prime * p_y

    # 2. Расчет расстояния до шара
    z_s = (focal_length * D_s) / d_s_prime

    # 3. Определение координат шара в системе координат камеры
    x_s = x_s_double_prime * z_s / focal_length
    y_s = y_s_double_prime * z_s / focal_length

    # 4. Переход в абсолютную систему координат
    x_s_abs = x_c + x_s * np.cos(phi_c) - z_s * np.sin(phi_c)
    y_s_abs = y_c + y_s
    z_s_abs = z_c + x_s * np.sin(phi_c) + z_s * np.cos(phi_c)

    # 5. Оценка погрешности
    delta_x = p_x * z_s / focal_length
    delta_y = p_y * z_s / focal_length
    delta_max = np.sqrt(delta_x**2 + delta_y**2)

    # Возврат результата
    return ((x_s_abs, y_s_abs, z_s_abs), delta_max)

def merge_data_sets(data_set1, data_set2):
    """
    Объединение двух наборов данных с координатами и диаметрами объектов.
    :param data_set1: Первый набор данных (список координат и диаметров)
    :param data_set2: Второй набор данных (список координат и диаметров)
    :return: Объединенный набор данных
    """
    # Длина объединенного набора данных будет равна длине самого длинного из двух
    max_length = max(len(data_set1), len(data_set2))
    merged_data = []

    for i in range(max_length):
        # Получаем данные из первого и второго набора (если они существуют)
        data1 = data_set1[i] if i < len(data_set1) else None
        data2 = data_set2[i] if i < len(data_set2) else None

        if data1 and data2:
            # Если данные есть в обоих наборах, можно объединить их как угодно
            # В данном случае вычислим среднее взвешенное значение координат и диаметров
            x = (data1['x'] + data2['x']) / 2
            y = (data1['y'] + data2['y']) / 2
            diameter = (data1['diameter'] + data2['diameter']) / 2
            merged_data.append({'x': x, 'y': y, 'diameter': diameter})
        elif data1:
            # Если данные есть только в первом наборе
            merged_data.append(data1)
        elif data2:
            # Если данные есть только во втором наборе
            merged_data.append(data2)
        else:
            # Если данных нет ни в одном из наборов
            merged_data.append(None)

    return merged_data
