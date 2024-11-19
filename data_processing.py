import numpy as np
from scipy.interpolate import interp1d

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
    az = np.radians(camera_data.az)
    d = np.sqrt(camera_data.matrix_width * camera_data.matrix_height / (frame_size[0] * frame_size[1])) * 1e-3 * sphere_diameter

    z_sm = camera_data.sphere_diameter * camera_data.focal_length / d * 1e-3
    x_sm = camera_data.sphere_diameter * coords[0] / sphere_diameter
    y_sm = camera_data.sphere_diameter * coords[1] / sphere_diameter

    x = -z_sm * np.sin(np.radians(az)) - x_sm * np.cos(az) + camera_data.x
    y = -y_sm + camera_data.y
    z = -z_sm * np.cos(np.radians(az)) + x_sm * np.sin(az) + camera_data.z

    delta_z = lambda beta: 1e-3 * (camera_data.sphere_diameter * camera_data.focal_length / d / (1-beta) + camera_data.sphere_diameter * camera_data.focal_length / d / (1+beta))
    delta_x = lambda beta, g_1: coords[0] * camera_data.sphere_diameter / sphere_diameter * ((1+g_1) / (1-beta) - (1-g_1) / (1+beta))
    delta_y = lambda beta, g_2: coords[1] * camera_data.sphere_diameter / sphere_diameter * ((1+g_2) / (1-beta) - (1-g_2) / (1+beta))

    delta = max([np.sqrt(delta_x(a/sphere_diameter, a_1/coords[0])**2 + delta_y(a/sphere_diameter, a_2/coords[1])**2 + delta_z(a/sphere_diameter)) for a in [0.5, 1, 1.5, 2] for a_1 in [0.5, 1, 1.5, 2] for a_2 in [0.5, 1, 1.5, 2]])
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

def combine_arrays(arr1, arr2, arr3):
    combined = []
    for a, b, c in zip(arr1, arr2, arr3):
        candidates = [a, b, c]
        candidates = [x for x in candidates if x is not None]
        if candidates:
            best_candidate = min(candidates, key=lambda x: x[1])
            combined.append(best_candidate)
        else:
            combined.append((None, None))
    return combined

def interpolate_missing_values(combined):
    """
    Интерполяция пропущенных значений в наборе данных.
    Формат данных: список кортежей (координаты, погрешность)
    :param combined: Набор данных
    :return: Набор данных с интерполированными значениями
    """
    coords = np.array([x[0] if x[0] is not None else (np.nan, np.nan, np.nan) for x in combined])
    errors = np.array([x[1] if x[1] is not None else np.nan for x in combined])

    # Find the first and last valid indices
    first_valid_index = next((i for i, x in enumerate(coords) if x is not None), None)
    last_valid_index = next((i for i, x in enumerate(reversed(coords)) if x is not None), None)
    if last_valid_index is not None:
        last_valid_index = len(coords) - 1 - last_valid_index

    if first_valid_index is None or last_valid_index is None:
        return combined  # No valid data to interpolate

    # Interpolate x, y, z coordinates separately using cubic interpolation
    for i in range(3):
        valid = ~np.isnan(coords[first_valid_index:last_valid_index + 1, i])
        if np.sum(valid) > 1:  # Ensure there are enough points to interpolate
            interp_func = interp1d(np.where(valid)[0], coords[first_valid_index:last_valid_index + 1][valid, i], kind='cubic', fill_value='extrapolate')
            coords[first_valid_index:last_valid_index + 1, i] = interp_func(np.arange(last_valid_index - first_valid_index + 1))

    # Replace None with interpolated values, excluding start and end None values
    for i in range(first_valid_index, last_valid_index + 1):
        if combined[i][0] is None:
            combined[i] = (tuple(coords[i]), errors[i])

    return combined
