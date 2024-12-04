import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

keys = [
    ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O'],
    ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L'],
    ['Z', 'X', 'C', 'V', 'B', 'N', 'M'],
    ['SPACE', 'Bsp', 'QUIT']
]

# Размеры клавиш и пробела
key_width, key_height = 60, 60
key_gap = 10
space_width = 7 * (key_width + key_gap) - key_gap

# Цветовые параметры клавиш
default_color = (0, 0, 0, 128)
hover_color = (50, 50, 50, 128)
press_color = (100, 100, 100, 128)
text_color = (200, 200, 200)

output = []  # Список для вывода текста

# Преобразование координат точек ладони из относительных в пиксели
def get_points(landmark, shape):
    return np.array([[int(mark.x * shape[1]), int(mark.y * shape[0])] for mark in landmark], dtype=np.int32)

# Вычисление центра ладони по точкам запястья и основания большого пальца
def palm_center(landmark, shape):
    x = int((landmark[0].x + landmark[5].x) * shape[1] / 2)
    y = int((landmark[0].y + landmark[5].y) * shape[0] / 2)
    return x, y

# Расстояние между запястьем и основанием большого пальца (размер ладони)
def palm_size(landmark, shape):
    x1, y1 = landmark[0].x * shape[1], landmark[0].y * shape[0]
    x2, y2 = landmark[5].x * shape[1], landmark[5].y * shape[0]
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

# Проверка, сжата ли ладонь в кулак
def is_fist(landmark, shape):
    fingertips = [8, 12, 16, 20]
    joints = [6, 10, 14, 18]
    fingers_curled = all(landmark[tip].y > landmark[joint].y for tip, joint in zip(fingertips, joints))

    points = get_points(landmark, shape)
    (x, y), r = cv2.minEnclosingCircle(points)
    ws = palm_size(landmark, shape)
    circle_ratio = 2 * r / ws

    return fingers_curled and circle_ratio <= 1.4

# Отрисовка клавиатуры на экране
def draw_keyboard(frame, keys, hover_key=None, pressed_keys=None):
    y_offset = 10
    for row in keys:
        x_offset = 10
        for key in row:
            width = space_width if key == "SPACE" else key_width
            x1, y1 = x_offset, y_offset
            x2, y2 = x1 + width, y1 + key_height

            color = (
                press_color if key in pressed_keys
                else hover_color if hover_key == key
                else default_color
            )

            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
            cv2.putText(frame, key, (x1 + 10, y1 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
            x_offset += width + key_gap
        y_offset += key_height + key_gap
    return frame

# Рисует зеленую точку в центре ладони
def draw_pointer(frame, x, y):
    cv2.circle(frame, (x, y), 20, (0, 255, 0), -1)

# Определение клавиши, над которой находится указатель
def detect_hover_key(x, y, keys):
    y_offset = 10
    for row in keys:
        x_offset = 10
        for key in row:
            width = space_width if key == "SPACE" else key_width
            x1, y1 = x_offset, y_offset
            x2, y2 = x1 + width, y1 + key_height
            if x1 < x < x2 and y1 < y < y2:
                return key
            x_offset += width + key_gap
        y_offset += key_height + key_gap
    return None

cap = cv2.VideoCapture(1)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        hover_key = None
        pressed_keys = set()

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                x, y = palm_center(hand_landmarks.landmark, frame.shape)
                hover_key = detect_hover_key(x, y, keys)
                draw_pointer(frame, x, y)

                if is_fist(hand_landmarks.landmark, frame.shape) and hover_key:
                    if hover_key == 'QUIT':
                        print("Output:", "".join(output))
                        cap.release()
                        cv2.destroyAllWindows()
                        exit()
                    elif hover_key == 'Bsp':
                        if output:
                            output.pop()
                    elif hover_key not in output:
                        output.append(hover_key)
                        pressed_keys.add(hover_key)

                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        frame = draw_keyboard(frame, keys, hover_key, pressed_keys)
        cv2.imshow('poplach_ob_etom', frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

except StopIteration:
    pass

cap.release()
cv2.destroyAllWindows()