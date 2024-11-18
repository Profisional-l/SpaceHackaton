class CameraData:
    def __init__(self, focal_length: float, matrix_width: float, matrix_height: float, x: float, y: float, z: float, az: float):
        self.focal_length = focal_length  # Фокусное расстояние камеры (мм)
        self.matrix_width = matrix_width  # Ширина матрицы камеры (мм)
        self.matrix_height = matrix_height  # Высота матрицы камеры (мм)
        self.x = x  # Координата X камеры
        self.y = y  # Координата Y камеры
        self.z = z  # Координата Z камеры
        self.az = az  # Азимут камеры (угол ориентации)
