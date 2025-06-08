import cv2
import mediapipe as mp
import pyautogui
import streamlit as st
import time
import numpy as np

# Setup Streamlit page
st.set_page_config(page_title="🖐️ AI Hand Scroll", layout="wide")
st.title("🖐️ Hand Gesture Scrolling with AI")
st.markdown("Move your **index finger** up/down to scroll the page. Press `Stop` to exit.")

run = st.checkbox("Start Camera")

# Display video feed
FRAME_WINDOW = st.image([])

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
prev_y = None
scroll_cooldown = time.time()

while run:
    ret, frame = cap.read()
    if not ret:
        st.error("❌ Camera not found!")
        break

    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            y = hand_landmarks.landmark[8].y  # Index finger tip
            if prev_y is not None:
                diff = y - prev_y
                if time.time() - scroll_cooldown > 0.2:
                    if diff > 0.02:
                        pyautogui.scroll(-20)
                        st.caption("📜 Scrolling down")
                        scroll_cooldown = time.time()
                    elif diff < -0.02:
                        pyautogui.scroll(20)
                        st.caption("📜 Scrolling up")
                        scroll_cooldown = time.time()
            prev_y = y
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    else:
        st.caption("✋ No hand detected")

    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

# Stop camera when unchecked
cap.release()
 
