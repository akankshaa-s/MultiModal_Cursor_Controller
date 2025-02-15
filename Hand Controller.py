import cv2
import mediapipe as mp
import pyautogui
import math
import os
import time

# Define gesture directories and counters
gesture_types = ['left_click', 'draw', 'scroll_up', 'scroll_down', 'right_click']
gesture_counters = {}

for gesture in gesture_types:
    gesture_folder = os.path.join("gesture_datasets", gesture)
    if not os.path.exists(gesture_folder):
        os.makedirs(gesture_folder)
    gesture_counters[gesture] = len(os.listdir(gesture_folder))

# Calculate Euclidean distance between two points
def calculate_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

# Smoothing for cursor movement using velocity
def smooth_movement_with_velocity(current_x, current_y, previous_x, previous_y, v_smoothing=0.5, max_step=50, alpha=0.2):
    velocity_x = current_x - previous_x
    velocity_y = current_y - previous_y
    smoothed_x = previous_x + min(max(velocity_x, -max_step), max_step) * v_smoothing
    smoothed_y = previous_y + min(max(velocity_y, -max_step), max_step) * v_smoothing
    smoothed_x = alpha * current_x + (1 - alpha) * smoothed_x
    smoothed_y = alpha * current_y + (1 - alpha) * smoothed_y
    return smoothed_x, smoothed_y

# Save hand gesture images
def save_gesture_image(frame, gesture_name, hand_bbox):
    x_min, y_min, x_max, y_max = hand_bbox
    cropped_hand = frame[y_min:y_max, x_min:x_max]
    gesture_counters[gesture_name] += 1
    file_name = f"gesture_datasets/{gesture_name}/{gesture_name}_{gesture_counters[gesture_name]}.png"
    cv2.imwrite(file_name, cropped_hand)

# Detect hand gestures and control cursor actions
def detect_hand():
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )

    cap = cv2.VideoCapture(0)
    screen_width, screen_height = pyautogui.size()
    smoothed_x, smoothed_y = 0, 0
    previous_x, previous_y = 0, 0
    last_left_click_time = 0
    click_threshold = 0.3  # Adjust debounce time for left click
    frame_counter = 0
    save_interval = 5
    prev_pos = None  # Track previous drawing position
    drawing_mode = False  # Declare drawing_mode as False initially
    is_dragging = False  # Track if the user is holding and dragging

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            thumb_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP]
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            ring_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
            pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

            # Calculate cursor position based on the index fingertip
            cursor_x = int(index_finger_tip.x * screen_width)
            cursor_y = int(index_finger_tip.y * screen_height)

            # Apply velocity-based smoothing
            smoothed_x, smoothed_y = smooth_movement_with_velocity(cursor_x, cursor_y, previous_x, previous_y)
            pyautogui.moveTo(int(smoothed_x), int(smoothed_y))
            previous_x, previous_y = smoothed_x, smoothed_y

            # Get bounding box for saving images
            x_coords = [int(lm.x * frame.shape[1]) for lm in hand_landmarks.landmark]
            y_coords = [int(lm.y * frame.shape[0]) for lm in hand_landmarks.landmark]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            hand_bbox = (max(0, x_min - 20), max(0, y_min - 20), min(frame.shape[1], x_max + 20), min(frame.shape[0], y_max + 20))

            # Left click: thumb and index fingers close
            if calculate_distance((thumb_tip.x, thumb_tip.y), (index_finger_tip.x, index_finger_tip.y)) < 0.05:
                if time.time() - last_left_click_time > click_threshold:
                    pyautogui.click()
                    last_left_click_time = time.time()
                    save_gesture_image(frame, 'left_click', hand_bbox)

            # Drawing mode: keep thumb and index finger together
            if calculate_distance((thumb_tip.x, thumb_tip.y), (index_finger_tip.x, index_finger_tip.y)) < 0.05:
                if not drawing_mode:
                    pyautogui.mouseDown()
                    drawing_mode = True
                    if frame_counter % save_interval == 0:
                        save_gesture_image(frame, 'draw', hand_bbox)
            else:
                if drawing_mode:
                    pyautogui.mouseUp()
                    drawing_mode = False

            # Smooth drawing: click-hold and drag
            if drawing_mode and not is_dragging:
                if calculate_distance((thumb_tip.x, thumb_tip.y), (index_finger_tip.x, index_finger_tip.y)) < 0.05:
                    is_dragging = True
                    prev_pos = (smoothed_x, smoothed_y)  # Save the previous position to start drawing

            if is_dragging:
                pyautogui.mouseDown()
                # Draw a smooth line if the user is dragging
                if prev_pos:
                    pyautogui.moveTo(smoothed_x, smoothed_y, duration=0.1)  # Duration makes the movement smooth
                    prev_pos = (smoothed_x, smoothed_y)

            if not drawing_mode and is_dragging:
                pyautogui.mouseUp()
                is_dragging = False
                prev_pos = None

            # Right click: index finger above thumb, rest of the fingers closed
            if index_finger_tip.y < thumb_tip.y and all(
                finger.y > thumb_tip.y for finger in [middle_finger_tip, ring_finger_tip, pinky_tip]):
                pyautogui.rightClick()
                save_gesture_image(frame, 'right_click', hand_bbox)

            # Scroll up: index and middle fingers above thumb, rest of the fingers closed
            if all(finger.y < thumb_tip.y for finger in [index_finger_tip, middle_finger_tip]) and all(
                finger.y > thumb_tip.y for finger in [ring_finger_tip, pinky_tip]):
                pyautogui.scroll(20)
                save_gesture_image(frame, 'scroll_up', hand_bbox)

            # Scroll down: thumb above index and middle fingers
            if thumb_tip.y < index_finger_tip.y < middle_finger_tip.y:
                pyautogui.scroll(-20)
                save_gesture_image(frame, 'scroll_down', hand_bbox)

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        frame_counter += 1
        cv2.imshow('Hand Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_hand()