import cv2
import numpy as np
import time

# Функция для вычисления оптического потока и движения
def kalman_filter(x, y, kalman):
    """Фильтр Калмана для предсказания координат"""
    # Обновляем измерения (X, Y координаты)
    kalman.correct(np.array([[np.float32(x)], [np.float32(y)]]))  # correct the filter
    
    # Получаем предсказанные координаты
    predicted = kalman.predict()
    return predicted[0], predicted[1]

# Захват видео
video_rgb_path = '../Seq1_camera1.mov'  # Обычное видео
video_ir_path = '../Seq1_camera1T.mov'   # Инфракрасное видео

# Открытие видеопотоков
video_rgb = cv2.VideoCapture(video_rgb_path)
video_ir = cv2.VideoCapture(video_ir_path)

# Проверим, что видео было загружено
if not video_rgb.isOpened() or not video_ir.isOpened():
    print("Ошибка при открытии видео")
    exit()

# Получаем количество кадров в видео для отображения прогресса
total_frames_rgb = int(video_rgb.get(cv2.CAP_PROP_FRAME_COUNT))
total_frames_ir = int(video_ir.get(cv2.CAP_PROP_FRAME_COUNT))

# Считываем первый кадр для обоих видео
ret_rgb, prev_frame_rgb = video_rgb.read()
ret_ir, prev_frame_ir = video_ir.read()
if not ret_rgb or not ret_ir:
    print("Не удалось прочитать кадр")
    exit()

# Преобразуем кадры в серый цвет для анализа
prev_gray_rgb = cv2.cvtColor(prev_frame_rgb, cv2.COLOR_BGR2GRAY)
prev_gray_ir = cv2.cvtColor(prev_frame_ir, cv2.COLOR_BGR2GRAY)

# Инициализация фильтра Калмана
kalman = cv2.KalmanFilter(4, 2)  # 4 состояния, 2 измерения
kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
kalman.transitionMatrix = np.eye(4, dtype=np.float32)
kalman.statePost = np.zeros((4, 1), np.float32)  # Начальное состояние

# Инициализация счётчика кадров
frame_count = 0
object_hidden = False  # Флаг, отслеживающий, был ли объект скрыт

# Создание окна для отображения с возможностью изменения размера
cv2.namedWindow("Motion Detection", cv2.WINDOW_NORMAL)  # Окно с изменяемым размером
cv2.resizeWindow("Motion Detection", 920, 540)  # Устанавливаем размер окна (например 640x480)

while True:
    ret_rgb, frame_rgb = video_rgb.read()
    ret_ir, frame_ir = video_ir.read()
    
    if not ret_rgb or not ret_ir:
        break
    
    frame_count += 1  # Увеличиваем счётчик кадров
    
    # Преобразуем кадры в серый для анализа
    gray_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2GRAY)
    gray_ir = cv2.cvtColor(frame_ir, cv2.COLOR_BGR2GRAY)
    
    # Применяем медианный фильтр для уменьшения шума
    gray_rgb = cv2.medianBlur(gray_rgb, 5)
    gray_ir = cv2.medianBlur(gray_ir, 5)
    
    # Вычитание кадров для выявления изменений
    diff_rgb = cv2.absdiff(prev_gray_rgb, gray_rgb)
    diff_ir = cv2.absdiff(prev_gray_ir, gray_ir)
    
    # Суммируем изменения по обоим видео
    diff_combined = cv2.add(diff_rgb, diff_ir)
    
    # Применяем порог для выделения движущихся объектов
    _, mask = cv2.threshold(diff_combined, 30, 255, cv2.THRESH_BINARY)
    
    # Применяем маску к исходному кадру
    result_frame = cv2.bitwise_and(frame_rgb, frame_rgb, mask=mask)  # Маска для одноцветного канала
    
    # Получаем координаты (например, X, Y центра движения)
    non_zero_points = np.nonzero(mask)  # Индексы ненулевых пикселей
    if len(non_zero_points[0]) > 0:
        # Если есть движущиеся пиксели, выводим их средние координаты
        center_x = int(np.mean(non_zero_points[1]))  # Среднее по оси X
        center_y = int(np.mean(non_zero_points[0]))  # Среднее по оси Y
        
        # Применяем фильтр Калмана для сглаживания координат
        predicted_x, predicted_y = kalman_filter(center_x, center_y, kalman)
        
        # Теперь извлекаем отдельные значения для predicted_x и predicted_y
        predicted_x_value = predicted_x[0] if isinstance(predicted_x, np.ndarray) else predicted_x
        predicted_y_value = predicted_y[0] if isinstance(predicted_y, np.ndarray) else predicted_y
        
        # Выводим координаты
        print(f"Координаты объекта: X={predicted_x_value:.2f}, Y={predicted_y_value:.2f}")
        
        # Обновляем флаг скрытия объекта
        object_hidden = False
    else:
        if not object_hidden:
            print("Объект скрыт из кадра")
            object_hidden = True
    
    # Отображаем результат
    cv2.imshow("Motion Detection", result_frame)
    
    # Обновляем предыдущие кадры
    prev_gray_rgb = gray_rgb.copy()
    prev_gray_ir = gray_ir.copy()
    
    # Вывод прогресса в консоль
    progress_rgb = (frame_count / total_frames_rgb) * 100
    elapsed_time = frame_count / video_rgb.get(cv2.CAP_PROP_FPS)  # Время в секундах
    print(f"Кадр: {frame_count}/{total_frames_rgb} ({progress_rgb:.2f}%) - Время: {elapsed_time:.2f}с", end="\r")
    
    # Выход из цикла по нажатию клавиши ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

video_rgb.release()
video_ir.release()
cv2.destroyAllWindows()
