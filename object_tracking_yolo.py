import cv2
import numpy as np
from video_processing import VideoProcessor
from camera_data import CameraData

# Известные параметры
KNOWN_WIDTH = 3  # Известный диаметр шара (в метрах)
FOCAL_LENGTH = 800  # Фокусное расстояние камеры (в пикселях)

# Функция для вычисления расстояния до объекта
def calculate_distance(pixel_width, focal_length=FOCAL_LENGTH):
    if pixel_width > 0:
        return (KNOWN_WIDTH * focal_length) / pixel_width
    return None

# Функция для фильтра Калмана
def kalman_filter(x, y, kalman, smooth_factor=0.3):
    """Фильтр Калмана для предсказания координат с динамическим сглаживанием"""
    kalman.correct(np.array([[np.float32(x)], [np.float32(y)]]))
    predicted = kalman.predict()
    predicted_x, predicted_y = predicted[0, 0], predicted[1, 0]
    return predicted_x, predicted_y

def process_video(video_path, camera_data):
    video_processor = VideoProcessor(video_path, camera_data)
    
    kalman = cv2.KalmanFilter(4, 2)
    kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kalman.transitionMatrix = np.eye(4, dtype=np.float32)
    kalman.statePost = np.zeros((4, 1), np.float32)
    
    frame_count = 0
    object_hidden = False
    predicted_x, predicted_y = None, None

    while True:
        frame, (x, y) = next(video_processor)
        if x is None or y is None:
            if not object_hidden:
                predicted_x, predicted_y = kalman_filter(predicted_x, predicted_y, kalman)
                print(f"Объект скрыт из кадра, предсказанное положение: X={predicted_x:.2f}, Y={predicted_y:.2f}")
                object_hidden = True
            cv2.circle(frame, (int(predicted_x), int(predicted_y)), 10, (0, 0, 255), -1)
        else:
            predicted_x, predicted_y = kalman_filter(x, y, kalman)
            distance = calculate_distance(x)
            print(f"Кадр: {frame_count}, Центр: X={predicted_x:.2f}, Y={predicted_y:.2f}, Расстояние: {distance:.2f}м")
            cv2.rectangle(frame, (x-10, y-10), (x+10, y+10), (0, 255, 0), 2)

        frame_count += 1
        cv2.imshow("Tracking", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    video_processor.release()
    cv2.destroyAllWindows()

# Пример использования:
camera_data = CameraData(focal_length=FOCAL_LENGTH, matrix_width=1920, matrix_height=1080, x=0, y=0, z=0, az=0)
process_video('../Seq1_camera1.mov', camera_data)
