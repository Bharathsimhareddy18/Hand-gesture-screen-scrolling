import cv2
import mediapipe as mp
import autopy
import numpy as np
import time

# Screen size
screen_w, screen_h = autopy.screen.size()

# MediaPipe hands setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

prev_click_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            lm = hand_landmarks.landmark

            # Index fingertip
            index_x = int(lm[8].x * w)
            index_y = int(lm[8].y * h)

            # Move mouse
            autopy.mouse.move(screen_w - (index_x * screen_w / w), index_y * screen_h / h)

            # Thumb tip
            thumb_x = int(lm[4].x * w)
            thumb_y = int(lm[4].y * h)

            # Distance between thumb and index
            dist = np.hypot(index_x - thumb_x, index_y - thumb_y)

            # If close enough, do a click (pinch gesture)
            if dist < 40:
                if time.time() - prev_click_time > 1:  # 1 sec cooldown
                    autopy.mouse.click()
                    print("🖱️ Clicked")
                    prev_click_time = time.time()

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("🖐️ Hand Mouse Control", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
