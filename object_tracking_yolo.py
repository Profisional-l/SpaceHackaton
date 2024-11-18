import cv2
import numpy as np
import time

# Известные параметры
KNOWN_WIDTH = 0.15  # Известный диаметр шара (в метрах)
FOCAL_LENGTH = 800  # Фокусное расстояние камеры (в пикселях)

# Функция для вычисления расстояния до объекта
def calculate_distance(pixel_width):
    if pixel_width > 0:
        return (KNOWN_WIDTH * FOCAL_LENGTH) / pixel_width
    return None

# Функция для фильтра Калмана
def kalman_filter(x, y, kalman):
    """Фильтр Калмана для предсказания координат"""
    kalman.correct(np.array([[np.float32(x)], [np.float32(y)]]))
    predicted = kalman.predict()
    return predicted[0, 0], predicted[1, 0]

# Захват видео
video_rgb_path = '../Seq1_camera1.mov'
video_ir_path = '../Seq1_camera1T.mov'

video_rgb = cv2.VideoCapture(video_rgb_path)
video_ir = cv2.VideoCapture(video_ir_path)

if not video_rgb.isOpened() or not video_ir.isOpened():
    print("Ошибка при открытии видео")
    exit()

# Получаем количество кадров в видео
total_frames_rgb = int(video_rgb.get(cv2.CAP_PROP_FRAME_COUNT))
total_frames_ir = int(video_ir.get(cv2.CAP_PROP_FRAME_COUNT))

# Считываем первый кадр
ret_rgb, prev_frame_rgb = video_rgb.read()
ret_ir, prev_frame_ir = video_ir.read()
if not ret_rgb or not ret_ir:
    print("Не удалось прочитать кадр")
    exit()

prev_gray_rgb = cv2.cvtColor(prev_frame_rgb, cv2.COLOR_BGR2GRAY)
prev_gray_ir = cv2.cvtColor(prev_frame_ir, cv2.COLOR_BGR2GRAY)

# Инициализация фильтра Калмана
kalman = cv2.KalmanFilter(4, 2)
kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
kalman.transitionMatrix = np.eye(4, dtype=np.float32)
kalman.statePost = np.zeros((4, 1), np.float32)

frame_count = 0
object_hidden = False

cv2.namedWindow("Motion Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Motion Detection", 640, 480)

while True:
    ret_rgb, frame_rgb = video_rgb.read()
    ret_ir, frame_ir = video_ir.read()

    if not ret_rgb or not ret_ir:
        break

    frame_count += 1
    gray_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2GRAY)
    gray_ir = cv2.cvtColor(frame_ir, cv2.COLOR_BGR2GRAY)

    gray_rgb = cv2.medianBlur(gray_rgb, 5)
    gray_ir = cv2.medianBlur(gray_ir, 5)

    diff_rgb = cv2.absdiff(prev_gray_rgb, gray_rgb)
    diff_ir = cv2.absdiff(prev_gray_ir, gray_ir)

    diff_combined = cv2.add(diff_rgb, diff_ir)

    _, mask = cv2.threshold(diff_combined, 30, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    # Поиск контуров для определения размеров объекта
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Определяем центр и предсказываем координаты
        center_x = x + w // 2
        center_y = y + h // 2
        predicted_x, predicted_y = kalman_filter(center_x, center_y, kalman)

        # Расчет расстояния до объекта
        distance = calculate_distance(w)

        # Выводим информацию
        print(f"Кадр: {frame_count}, Центр: X={predicted_x:.2f}, Y={predicted_y:.2f}, Ширина: {w}px, Расстояние: {distance:.2f}м")
        object_hidden = False

        # Отображаем область интереса
        cv2.rectangle(frame_rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame_rgb, f"Dist: {distance:.2f}m", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    else:
        if not object_hidden:
            print(f"Кадр: {frame_count}, Объект скрыт из кадра")
            object_hidden = True

    cv2.imshow("Motion Detection", frame_rgb)
    prev_gray_rgb = gray_rgb.copy()
    prev_gray_ir = gray_ir.copy()

    progress_rgb = (frame_count / total_frames_rgb) * 100
    elapsed_time = frame_count / video_rgb.get(cv2.CAP_PROP_FPS)
    print(f"Прогресс: {progress_rgb:.2f}%, Время: {elapsed_time:.2f}с", end="\r")

    if cv2.waitKey(1) & 0xFF == 27:
        break

video_rgb.release()
video_ir.release()
cv2.destroyAllWindows()
