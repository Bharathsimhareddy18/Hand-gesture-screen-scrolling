import cv2
import mediapipe as mp
import pyautogui
import time

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)

# Variables to track scrolling
prev_y = None
scroll_cooldown = time.time()

print("🖐️ Raise your hand and move up/down to scroll. Press 'q' to quit.")

while True:
    success, img = cap.read()
    if not success:
        continue

    # Flip the image horizontally for natural (mirror) viewing
    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Get y position of index finger tip (landmark 8)
            y = hand_landmarks.landmark[8].y

            # Compare with previous y to determine direction
            if prev_y is not None:
                diff = y - prev_y

                if time.time() - scroll_cooldown > 0.2:  # Cooldown to prevent spamming
                    if diff > 0.02:
                        pyautogui.scroll(-100)  # Scroll down
                        print("📜 Scrolling down")
                        scroll_cooldown = time.time()
                    elif diff < -0.02:
                        pyautogui.scroll(100)   # Scroll up
                        print("📜 Scrolling up")
                        scroll_cooldown = time.time()

            prev_y = y

            # Draw landmarks
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Show camera feed
    cv2.imshow("Hand Scroll", img)

    # Break on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
 
