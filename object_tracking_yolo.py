import cv2
import numpy as np

# Функция для вычисления оптического потока и движения
def compute_motion(prev_gray, gray):
    # Вычисляем оптический поток
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    # Вычисляем изменения в каждом пикселе
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    return magnitude

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

# Частота кадров
fps_rgb = video_rgb.get(cv2.CAP_PROP_FPS)

# Считываем первый кадр для обоих видео
ret_rgb, prev_frame_rgb = video_rgb.read()
ret_ir, prev_frame_ir = video_ir.read()
if not ret_rgb or not ret_ir:
    print("Не удалось прочитать кадр")
    exit()

# Преобразуем кадры в серый цвет для оптического потока
prev_gray_rgb = cv2.cvtColor(prev_frame_rgb, cv2.COLOR_BGR2GRAY)
prev_gray_ir = cv2.cvtColor(prev_frame_ir, cv2.COLOR_BGR2GRAY)

# Инициализация счётчика кадров
frame_count = 0
object_was_visible = False  # Переменная для отслеживания видимости объекта

while True:
    ret_rgb, frame_rgb = video_rgb.read()
    ret_ir, frame_ir = video_ir.read()
    
    if not ret_rgb or not ret_ir:
        break
    
    frame_count += 1  # Увеличиваем счётчик кадров
    
    # Преобразуем кадры в серый для расчета оптического потока
    gray_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2GRAY)
    gray_ir = cv2.cvtColor(frame_ir, cv2.COLOR_BGR2GRAY)
    
    # Вычисление оптического потока для RGB и IR каналов
    magnitude_rgb = compute_motion(prev_gray_rgb, gray_rgb)
    magnitude_ir = compute_motion(prev_gray_ir, gray_ir)
    
    # Суммируем величины изменения в обоих каналах
    combined_magnitude = cv2.add(magnitude_rgb, magnitude_ir)
    
    # Устанавливаем порог для выделения движущихся объектов
    motion_threshold = 2.0
    mask = np.zeros_like(frame_rgb, dtype=np.uint8)
    mask[combined_magnitude > motion_threshold] = 255
    
    # Применяем маску к исходному кадру
    result_frame = cv2.bitwise_and(frame_rgb, frame_rgb, mask=mask[:, :, 0])  # Маска для одноцветного канала
    
    # Получаем координаты (например, X, Y центра движения)
    non_zero_points = np.nonzero(mask[:, :, 0])  # Индексы ненулевых пикселей
    if len(non_zero_points[0]) > 0:
        # Если есть движущиеся пиксели, выводим их средние координаты
        center_x = int(np.mean(non_zero_points[1]))  # Среднее по оси X
        center_y = int(np.mean(non_zero_points[0]))  # Среднее по оси Y
        object_was_visible = True
        print(f"Координаты объекта: X={center_x}, Y={center_y} (Кадр {frame_count}/{total_frames_rgb} - Время: {frame_count / fps_rgb:.2f} сек)")
    else:
        if object_was_visible:
            print(f"Кадр {frame_count}/{total_frames_rgb} - Время: {frame_count / fps_rgb:.2f} сек")
            print("Объект скрыт из кадра.")
            object_was_visible = False
    
    # Уменьшаем размер окна для отображения результата
    result_frame_resized = cv2.resize(result_frame, (640, 360))  # Новый размер окна
    
    # Отображаем результат
    cv2.imshow("Motion Detection", result_frame_resized)
    
    # Обновляем предыдущие кадры
    prev_gray_rgb = gray_rgb.copy()
    prev_gray_ir = gray_ir.copy()
    
    # Вывод прогресса в консоль
    progress_rgb = (frame_count / total_frames_rgb) * 100
    print(f"Обработано {frame_count}/{total_frames_rgb} кадров ({progress_rgb:.2f}%)", end="\r")
    
    # Выход из цикла по нажатию клавиши ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

video_rgb.release()
video_ir.release()
cv2.destroyAllWindows()
