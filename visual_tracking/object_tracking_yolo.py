import cv2
import numpy as np
import concurrent.futures

if __name__ != '__main__':
    raise ImportError("This module is not intended to be imported")

# Известные параметры
KNOWN_WIDTH = 3  # Известный диаметр объекта (в метрах)
FOCAL_LENGTH = 800  # Фокусное расстояние камеры (в пикселях)


# Функция для улучшения изображения ИК-камеры
def preprocess_ir_frame(gray_ir):
    equalized = cv2.equalizeHist(gray_ir)  # Увеличение контраста
    denoised = cv2.medianBlur(equalized, 5)  # Уменьшение шума
    blurred = cv2.GaussianBlur(denoised, (5, 5), 0)  # Размытие
    return blurred


# Функция для вычисления расстояния до объекта
def calculate_distance(pixel_width):
    if pixel_width > 0:
        return (KNOWN_WIDTH * FOCAL_LENGTH) / pixel_width
    return None


# Функция для фильтра Калмана
def kalman_filter(x, y, kalman):
    kalman.correct(np.array([[np.float32(x)], [np.float32(y)]]))  # Коррекция с помощью измерения
    predicted = kalman.predict()  # Предсказание следующего состояния
    predicted_x, predicted_y = predicted[0, 0], predicted[1, 0]
    return predicted_x, predicted_y


# Захват видео
video_rgb_path = '../tests/step1/videoset1/Seq1_camera1.mov'
video_ir_path = '../tests/step1/videoset1/Seq1_camera1T.mov'

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
kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2  # Процессный шум
kalman.measurementNoiseCov = np.array([[1, 0], [0, 1]], dtype=np.float32) * 1e-1  # Шум измерений

frame_count = 0
object_hidden = False  # Флаг, если объект скрыт
predicted_x, predicted_y = None, None


# Функция для вычисления оптического потока
def calculate_optical_flow(prev_gray, gray):
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return flow


# Функция для обработки RGB кадра
def process_rgb_frame(frame_rgb):
    gray_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2GRAY)
    diff_rgb = cv2.absdiff(prev_gray_rgb, gray_rgb)
    return diff_rgb, gray_rgb


# Функция для обработки IR кадра
def process_ir_frame(frame_ir):
    gray_ir = cv2.cvtColor(frame_ir, cv2.COLOR_BGR2GRAY)
    processed_ir = preprocess_ir_frame(gray_ir)
    diff_ir = cv2.absdiff(prev_gray_ir, processed_ir)
    return diff_ir, processed_ir


cv2.namedWindow("Motion Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Motion Detection", 960, 540)
# Функция для нахождения параметров шара и его диаметра
# Функция для нахождения параметров шара и его диаметра
def find_ball_parameters(contour):
    # Аппроксимация контура для уменьшения количества точек
    epsilon = 0.01 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # Находим минимальную окружность, охватывающую аппроксимированный контур
    (x, y), radius = cv2.minEnclosingCircle(approx)
    center = (int(x), int(y))
    diameter = 2 * radius
    return center, diameter


# Многозадачная обработка
with concurrent.futures.ThreadPoolExecutor() as executor:
    while True:
        ret_rgb, frame_rgb = video_rgb.read()
        ret_ir, frame_ir = video_ir.read()

        if not ret_rgb or not ret_ir:
            break

        frame_count += 1

        # Параллельная обработка кадров RGB и IR
        rgb_future = executor.submit(process_rgb_frame, frame_rgb)
        ir_future = executor.submit(process_ir_frame, frame_ir)

        diff_rgb, gray_rgb = rgb_future.result()
        diff_ir, processed_ir = ir_future.result()

        # Объединяем различия
        diff_combined = cv2.add(diff_rgb, diff_ir)
        _, mask = cv2.threshold(diff_combined, 30, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        # Поиск контуров для определения размеров объекта
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Фильтрация контуров по площади
            contours = [c for c in contours if 100 < cv2.contourArea(c) < 10000]
            if not contours:
                continue

            # Если контуры найдены, продолжаем обработку
            largest_contour = max(contours, key=cv2.contourArea)
            center, diameter = find_ball_parameters(largest_contour)

            if diameter < 20:  # Порог для минимального диаметра объекта
                continue

            # Применение фильтра Калмана для предсказания координат
            predicted_x, predicted_y = kalman_filter(center[0], center[1], kalman)

            # Расчет расстояния до объекта
            distance = calculate_distance(diameter)

            # Вывод координат в консоль
            print(f"Координаты объекта: X={predicted_x:.2f}, Y={predicted_y:.2f}, Диаметр={diameter:.2f}")

            # Отображаем область интереса
            cv2.circle(frame_rgb, center, int(diameter // 2), (0, 255, 0), 2)
            cv2.putText(frame_rgb, f"Dist: {distance:.2f}m", (center[0], center[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 0, 0), 2)

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

        progress_rgb = (frame_count / total_frames_rgb) * 100
        elapsed_time = frame_count / video_rgb.get(cv2.CAP_PROP_FPS)
        print(f"Прогресс: {progress_rgb:.2f}%, Время: {elapsed_time:.2f}с", end="\r")

        if cv2.waitKey(1) & 0xFF == 27:  # Выход по нажатию клавиши 'Esc'
            break

video_rgb.release()
video_ir.release()
cv2.destroyAllWindows()
