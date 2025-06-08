import cv2
import mediapipe as mp
import pyautogui
import streamlit as st
import time
import numpy as np

# Page config
st.set_page_config(page_title="Solo Leveling - Hand Scroll", layout="wide")

# Inject dark theme with custom fonts & styling
st.markdown("""
    <style>
    html, body, [class*="css"] {
        background-color: #0d0d0d;
        color: #00ffff;
        font-family: 'Consolas', monospace;
    }
    .big-font {
        font-size: 24px !important;
        color: #00ffff;
    }
    .scroll-box {
        background-color: rgba(0,255,255,0.05);
        border: 1px solid #00ffff;
        border-radius: 10px;
        padding: 10px;
        margin-top: 10px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='big-font'>🖐️ Solo Leveling: Gesture Scroll HUD</h1>", unsafe_allow_html=True)
st.markdown("Move your **index finger** ⬆️⬇️ to scroll like a hunter. Tap `Start Camera` to activate your skill.")

run = st.checkbox("🔛 Start Camera")

FRAME_WINDOW = st.image([])
status_box = st.empty()

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Camera
cap = cv2.VideoCapture(0)
prev_y = None
scroll_cooldown = time.time()

while run:
    ret, frame = cap.read()
    if not ret:
        status_box.error("❌ Camera not found!")
        break

    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    status_text = "🕶️ Standby Mode"

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            y = hand_landmarks.landmark[8].y
            if prev_y is not None:
                diff = y - prev_y
                if time.time() - scroll_cooldown > 0.2:
                    if diff > 0.02:
                        pyautogui.scroll(-20)
                        status_text = "⬇️ Scroll Down"
                        scroll_cooldown = time.time()
                    elif diff < -0.02:
                        pyautogui.scroll(20)
                        status_text = "⬆️ Scroll Up"
                        scroll_cooldown = time.time()
            prev_y = y
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    else:
        status_text = "❌ No hand detected"

    # Display
    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    status_box.markdown(f"<div class='scroll-box'>{status_text}</div>", unsafe_allow_html=True)

cap.release()
 
