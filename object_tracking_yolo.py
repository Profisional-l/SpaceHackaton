import cv2
import numpy as np
from video_processing import VideoProcessor
from camera_data import CameraData

# Известные параметры
KNOWN_WIDTH = 3  # Известный диаметр объекта (в метрах)
FOCAL_LENGTH = 800  # Фокусное расстояние камеры (в пикселях)

# Функция для вычисления расстояния до объекта
def calculate_distance(pixel_width, focal_length=FOCAL_LENGTH):
    try:
        if pixel_width > 0:
            return (KNOWN_WIDTH * focal_length) / pixel_width
        else:
            raise ValueError("Ширина объекта в пикселях должна быть больше нуля.")
    except ZeroDivisionError:
        print("Ошибка: деление на ноль в calculate_distance.")
        return None


# Функция для фильтра Калмана
def kalman_filter(x, y, kalman, smooth_factor=0.3):
    """Фильтр Калмана для предсказания координат с динамическим сглаживанием"""
    kalman.correct(np.array([[np.float32(x)], [np.float32(y)]]))
    predicted = kalman.predict()
    predicted_x, predicted_y = predicted[0, 0], predicted[1, 0]
    return predicted_x, predicted_y

def preprocess_ir_frame(gray_ir):
    """Функция для улучшения изображения ИК-камеры"""
    equalized = cv2.equalizeHist(gray_ir)  # Увеличение контраста
    denoised = cv2.medianBlur(equalized, 5)  # Уменьшение шума
    blurred = cv2.GaussianBlur(denoised, (5, 5), 0)  # Размытие
    return blurred

def process_video(video_rgb_path, video_ir_path, camera_data):
    video_rgb = cv2.VideoCapture(video_rgb_path)
    video_ir = cv2.VideoCapture(video_ir_path)

    if not video_rgb.isOpened() or not video_ir.isOpened():
        print("Ошибка при открытии видео")
        return

    # Инициализация фильтра Калмана
    kalman = cv2.KalmanFilter(4, 2)
    kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kalman.transitionMatrix = np.eye(4, dtype=np.float32)
    kalman.statePost = np.zeros((4, 1), np.float32)

    # Создаем объект VideoProcessor
    video_processor = VideoProcessor(video_rgb, video_ir, camera_data)
    
    frame_count = 0
    object_hidden = False
    predicted_x, predicted_y = None, None

    while True:
        ret_rgb, frame_rgb = video_rgb.read()
        ret_ir, frame_ir = video_ir.read()

        if not ret_rgb or not ret_ir:
            break

        frame_count += 1

        # Преобразуем кадры в оттенки серого
        gray_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2GRAY)
        gray_ir = cv2.cvtColor(frame_ir, cv2.COLOR_BGR2GRAY)

        # Обработка ИК-кадра
        processed_ir = preprocess_ir_frame(gray_ir)
        diff_ir = cv2.absdiff(gray_ir, processed_ir)

        # Обработка RGB-кадра
        diff_rgb = cv2.absdiff(prev_gray_rgb, gray_rgb)

        # Объединяем различия
        diff_combined = cv2.add(diff_rgb, diff_ir)
        _, mask = cv2.threshold(diff_combined, 30, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        # Поиск контуров для определения размеров объекта
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)

            # Применяем фильтр Калмана для предсказания координат
            center_x = x + w // 2
            center_y = y + h // 2
            predicted_x, predicted_y = kalman_filter(center_x, center_y, kalman)

            # Расчет расстояния до объекта
            distance = calculate_distance(w)
            print(f"Координаты объекта: X={predicted_x:.2f}, Y={predicted_y:.2f}, Расстояние: {distance:.2f}м")

            # Отображаем область интереса
            cv2.rectangle(frame_rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame_rgb, f"Dist: {distance:.2f}m", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            object_hidden = False  # Объект найден, сбрасываем флаг
        else:
            # Если контуры не найдены, объект скрыт
            if not object_hidden and predicted_x is not None and predicted_y is not None:
                print("Объект скрыт, продолжаем следить за последним местоположением")
                # Применяем последнюю позицию, чтобы предсказать следующую координату
                cv2.putText(frame_rgb, "Object Hidden", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                # Показать старые предсказания
                cv2.circle(frame_rgb, (int(predicted_x), int(predicted_y)), 5, (0, 0, 255), -1)
                object_hidden = True  # Устанавливаем флаг скрытого объекта

        # Отображаем финальный результат
        cv2.imshow("Motion Detection", frame_rgb)

        prev_gray_rgb = gray_rgb.copy()
        prev_gray_ir = processed_ir.copy()

        if cv2.waitKey(1) & 0xFF == 27:  # Выход по нажатию клавиши 'Esc'
            break

    video_rgb.release()
    video_ir.release()
    cv2.destroyAllWindows()


# Пример использования:
camera_data = CameraData(focal_length=FOCAL_LENGTH, matrix_width=1920, matrix_height=1080, x=0, y=0, z=0, az=0)
process_video('../Seq1_camera1.mov', '../Seq1_camera1T.mov', camera_data)
