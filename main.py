import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

keys = [
    ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
    ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L'],
    ['Z', 'X', 'C', 'V', 'B', 'N', 'M'],
    ['SPACE']
]

key_width, key_height = 70, 70
key_gap = 10
spacebar_width = 7 * (key_width + key_gap) - key_gap  
default_color = (200, 0, 0)
hover_color = (0, 0, 255)
press_color = (0, 255, 0)
text_color = (255, 255, 255)
output = []

def get_points(landmark, shape):
    points = []
    for mark in landmark:{
        points.append([mark.x * shape[1], mark.y * shape[0]])
    }        
    
    return np.array(points, dtype=np.int32)

def palm_size(landmark, shape):
    x1, y1 = landmark[0].x * shape[1], landmark[0].y * shape[0]
    x2, y2 = landmark[5].x * shape[1], landmark[5].y * shape[0]
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

def is_fist(landmark, shape):
    fingertips = [8, 12, 16, 20]
    joints = [6, 10, 14, 18]
    fingers_curled = all(
        landmark[tip].y > landmark[joint].y for tip, joint in zip(fingertips, joints)
    )
    points = get_points(landmark, shape)
    (x, y), r = cv2.minEnclosingCircle(points)
    ws = palm_size(landmark, shape)
    circle_ratio = 2 * r / ws
    return fingers_curled and circle_ratio <= 1.4

def draw_keyboard(frame, keys, hover_key=None, pressed_keys=None):
    y_offset = 10
    for row_idx, row in enumerate(keys):
        x_offset = 10
        for col_idx, key in enumerate(row):
            if key == "SPACE":
                width = spacebar_width
            else:
                width = key_width
            x1, y1 = x_offset, y_offset
            x2, y2 = x1 + width, y1 + key_height
            color = press_color if key in pressed_keys else hover_color if hover_key == key else default_color
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
            cv2.putText(frame, key, (x1 + 15, y1 + 45), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
            x_offset += width + key_gap
        y_offset += key_height + key_gap
    return frame

def detect_hover_key(x, y, keys):
    y_offset = 10
    for row_idx, row in enumerate(keys):
        x_offset = 10
        for col_idx, key in enumerate(row):
            if key == "SPACE":
                width = spacebar_width
            else:
                width = key_width
            x1, y1 = x_offset, y_offset
            x2, y2 = x1 + width, y1 + key_height
            if x1 < x < x2 and y1 < y < y2:
                return key
            x_offset += width + key_gap
        y_offset += key_height + key_gap
    return None

cap = cv2.VideoCapture(1)

#      sudo rm -rf / --no-preserve-root
#⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⣤⣤⣤⣤⣤⣶⣦⣤⣄⡀⠀⠀⠀⠀⠀⠀⠀⠀ 
#⠀⠀⠀⠀⠀⠀⠀⠀⢀⣴⣿⡿⠛⠉⠙⠛⠛⠛⠛⠻⢿⣿⣷⣤⡀⠀⠀⠀⠀⠀ 
#⠀⠀⠀⠀⠀⠀⠀⠀⣼⣿⠋⠀⠀⠀⠀⠀ ⠀⠀     ⠈⢻⣿⣿⡄⠀⠀⠀⠀ 
#⠀⠀⠀⠀⠀⠀⠀⣸⣿⡏⠀⠀ ⠀⣠⣶⣾⣿⣿⣿⠿⠿⠿⢿⣿⣿⣿⣄⠀⠀⠀ 
#⠀⠀⠀⠀⠀⠀⠀⣿⣿⠁⠀⠀⢰⣿⣿⣯⠁⠀⠀⠀⠀⠀⠀  ⠀⠈⠙⢿⣷⡄⠀ 
#⠀⠀⣀⣤⣴⣶⣶⣿⡟⠀⠀⠀⢸⣿⣿⣿⣆⠀⠀⠀⠀⠀⠀⠀  ⠀⠀⠀⣿⣷⠀ 
#⠀⢰⣿⡟⠋⠉⣹⣿⡇⠀⠀⠀⠘⣿⣿⣿⣿⣷⣦⣤⣤⣤⣶⣶⣶⣶⣿⣿⣿⠀ 
#⠀⢸⣿⡇⠀⠀⣿⣿⡇⠀⠀⠀ ⠀⠹⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠃⠀ 
#⠀⣸⣿⡇⠀⠀⣿⣿⡇⠀⠀⠀⠀ ⠀⠉⠻⠿⣿⣿⣿⣿⡿⠿⠿⠛⢻⣿⡇⠀⠀ 
#⠀⣿⣿⠁⠀⠀⣿⣿⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀   ⠀⢸⣿⣧⠀⠀ 
#⠀⣿⣿⠀⠀⠀⣿⣿⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀  ⠀ ⢸⣿⣿⠀⠀ 
#⠀⣿⣿⠀⠀⠀⣿⣿⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀  ⠀ ⢸⣿⣿⠀⠀ 
#⠀⢿⣿⡆⠀⠀⣿⣿⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀  ⠀ ⢸⣿⡇⠀⠀ 
#⠀⠸⣿⣧⡀⠀⣿⣿⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀   ⣿⣿⠃⠀⠀ 
#⠀⠀⠛⢿⣿⣿⣿⣿⣇⠀⠀⠀⠀⠀⣰⣿⣿⣷⣶⣶⣶⣶⠶⠀ ⢠⣿⣿⠀⠀⠀ 
#⠀⠀⠀⠀⠀⠀⠀⣿⣿⠀⠀⠀⠀⠀ ⣿⣿⡇⠀⣽⣿⡏⠁⠀ ⠀⢸⣿⡇⠀⠀⠀ 
#⠀⠀⠀⠀⠀⠀⠀⣿⣿⠀⠀⠀⠀⠀ ⣿⣿⡇⠀⢹⣿⡆⠀⠀ ⠀⣸⣿⠇⠀⠀⠀ 
#⠀⠀⠀⠀⠀⠀⠀⢿⣿⣦⣄⣀⣠⣴⣿⣿⠁⠀⠈⠻⣿⣿⣿⣿⡿⠏⠀⠀⠀⠀ 
#⠀⠀⠀⠀⠀⠀⠀⠈⠛⠻⠿⠿⠿⠿⠋⠁⠀⠀

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        hover_key = None
        pressed_keys = set()

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                x = int(hand_landmarks.landmark[8].x * w)
                y = int(hand_landmarks.landmark[8].y * h)
                hover_key = detect_hover_key(x, y, keys)
                if is_fist(hand_landmarks.landmark, frame.shape) and hover_key:
                    if hover_key == 'Q':
                        print("Output:", output)
                        raise StopIteration
                    elif hover_key not in output:
                        output.append(hover_key)
                        pressed_keys.add(hover_key)
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        frame = draw_keyboard(frame, keys, hover_key, pressed_keys)
        cv2.imshow('Virtual Keyboard', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

except StopIteration:
    pass

cap.release()
cv2.destroyAllWindows()