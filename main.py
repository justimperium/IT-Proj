import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

pose = mp_pose.Pose()
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)

cap = cv2.VideoCapture(1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    flipped = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)
    pose_results = pose.process(rgb_frame)
    hands_results = hands.process(rgb_frame)
    h, w, _ = flipped.shape
    bg = np.zeros((h, w, 3), dtype=np.uint8)
    bg[:] = (102, 51, 0)  

    if pose_results.pose_landmarks:
        mp_drawing.draw_landmarks(
            bg,
            pose_results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
        )

        landmarks = pose_results.pose_landmarks.landmark
        neck_start = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        neck_end = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        neck_center_x = int((neck_start.x + neck_end.x) / 2 * w)
        neck_center_y = int((neck_start.y + neck_end.y) / 2 * h)
        head = landmarks[mp_pose.PoseLandmark.NOSE]
        head_x = int(head.x * w)
        head_y = int(head.y * h)
        cv2.line(bg, (neck_center_x, neck_center_y), (head_x, head_y), (255, 255, 255), 2)

    if hands_results.multi_hand_landmarks:
        for hand_landmarks in hands_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                bg,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )

    cv2.imshow("skeleton", bg)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    if cv2.waitKey(1) & 0xFF == ord("й"):
        break

cap.release()
pose.close()
hands.close()
cv2.destroyAllWindows()