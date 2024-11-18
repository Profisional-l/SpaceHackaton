import cv2
import numpy as np
from ultralytics import YOLO

# Загрузка модели YOLO (используем предобученную YOLOv8)
model = YOLO('yolov8n.pt')  # Можно выбрать yolov8n.pt, yolov8s.pt и другие

# Параметры для примитивного вычисления глубины
FOCAL_LENGTH = 800  # Фокусное расстояние (примерное значение)
KNOWN_WIDTH = 0.5   # Известная ширина объекта (в метрах)

def calculate_3d_coordinates(bbox, depth_map):
    """
    Вычисление 3D координат объекта на основе его 2D позиции и глубины.
    bbox: [x1, y1, x2, y2]
    depth_map: карта глубины кадра
    """
    x1, y1, x2, y2 = map(int, bbox)
    x_center = (x1 + x2) // 2
    y_center = (y1 + y2) // 2
    
    # Пример получения глубины в центре объекта
    z = depth_map[y_center, x_center]  # Глубина (z-координата)
    
    # Примерные 3D координаты (должны быть уточнены)
    x = (x_center - depth_map.shape[1] // 2) * z / FOCAL_LENGTH
    y = (y_center - depth_map.shape[0] // 2) * z / FOCAL_LENGTH
    return x, y, z

# Захват видео
video = cv2.VideoCapture('test02.mp4')
  # 0 - вебкамера, или путь к видеофайлу

while True:
    ret, frame = video.read()
    if not ret:
        break

    # Детекция объектов
    results = model(frame, stream=True)  # Результаты работы YOLO
    depth_map = np.full(frame.shape[:2], 2.0)  # Простая фейковая карта глубины
    
    for result in results:
        for box in result.boxes:
            # Координаты ограничивающего прямоугольника
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0]
            cls = int(box.cls[0])
            
            # Условие фильтрации по порогу уверенности
            if confidence > 0.5:
                # Рисуем прямоугольник вокруг объекта
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Добавление метки класса и вероятности
                label = f"{model.names[cls]} {confidence:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Пример 3D координат
                x, y, z = calculate_3d_coordinates((x1, y1, x2, y2), depth_map)
                cv2.putText(frame, f"3D: ({x:.2f}, {y:.2f}, {z:.2f})", (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Показ видео
    cv2.imshow('Object Detection', frame)

    # Выход из цикла по нажатию ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

video.release()
cv2.destroyAllWindows()
