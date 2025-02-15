import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import os
import time

# Directory to store eye images
dataset_directory = 'eye_img'
left_blink_folder = os.path.join(dataset_directory, 'left_blink')
right_blink_folder = os.path.join(dataset_directory, 'right_blink')
os.makedirs(left_blink_folder, exist_ok=True)
os.makedirs(right_blink_folder, exist_ok=True)

# Initialize Mediapipe and OpenCV
cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
screen_width, screen_height = pyautogui.size()

# Parameters for smoothing cursor movement
prev_x, prev_y = screen_width / 2, screen_height / 2
smooth_factor = 0.2  # Increased for smoother movement

# Blink detection thresholds
EYE_AR_THRESH = 0.22
EYE_AR_CONSEC_FRAMES = 3
blink_counter_left, blink_counter_right = 0, 0
click_triggered_left, click_triggered_right = False, False

# Function for Eye Aspect Ratio (EAR) for blink detection
def eye_aspect_ratio(eye_landmarks):
    if len(eye_landmarks) < 5:
        return 0
    vertical_dist = abs(eye_landmarks[1].y - eye_landmarks[4].y)
    horizontal_dist = abs(eye_landmarks[0].x - eye_landmarks[3].x)
    return vertical_dist / (horizontal_dist + 1e-6)

# Smoothing function for cursor movement
def smooth_cursor(prev_x, prev_y, target_x, target_y, smooth_factor):
    smooth_x = prev_x + (target_x - prev_x) * smooth_factor
    smooth_y = prev_y + (target_y - prev_y) * smooth_factor
    return smooth_x, smooth_y

while True:
    _, frame = cam.read()
    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks

    if landmark_points:
        landmarks = landmark_points[0].landmark

        if len(landmarks) >= 388:
            left_eye_landmarks = [landmarks[33], landmarks[133], landmarks[160], landmarks[159], landmarks[158]]
            right_eye_landmarks = [landmarks[362], landmarks[263], landmarks[387], landmarks[386], landmarks[385]]

            # Calculate eye aspect ratios
            left_eye_ratio = eye_aspect_ratio(left_eye_landmarks)
            right_eye_ratio = eye_aspect_ratio(right_eye_landmarks)

            # Smooth cursor based on eye positions
            midpoint_x = int((landmarks[33].x + landmarks[362].x) / 2 * screen_width)
            midpoint_y = int((landmarks[159].y + landmarks[386].y) / 2 * screen_height)
            smooth_x, smooth_y = smooth_cursor(prev_x, prev_y, midpoint_x, midpoint_y, smooth_factor)

            # Move the cursor to the smoothed coordinates
            pyautogui.moveTo(smooth_x, smooth_y)
            prev_x, prev_y = smooth_x, smooth_y

            # Detect left eye blink for left click
            if left_eye_ratio < EYE_AR_THRESH:
                blink_counter_left += 1
                if blink_counter_left >= EYE_AR_CONSEC_FRAMES and not click_triggered_left:
                    pyautogui.mouseDown(button='left')
                    click_triggered_left = True
            else:
                if click_triggered_left:
                    pyautogui.mouseUp(button='left')
                    click_triggered_left = False
                blink_counter_left = 0

            # Detect right eye blink for right click
            if right_eye_ratio < EYE_AR_THRESH:
                blink_counter_right += 1
                if blink_counter_right >= EYE_AR_CONSEC_FRAMES and not click_triggered_right:
                    pyautogui.mouseDown(button='right')
                    click_triggered_right = True
            else:
                if click_triggered_right:
                    pyautogui.mouseUp(button='right')
                    click_triggered_right = False
                blink_counter_right = 0

            # Capture eye images during blinks
            min_x = max(0, int(min([landmark.x for landmark in left_eye_landmarks + right_eye_landmarks]) * frame_width) - 10)
            min_y = max(0, int(min([landmark.y for landmark in left_eye_landmarks + right_eye_landmarks]) * frame_height) - 10)
            max_x = min(frame_width, int(max([landmark.x for landmark in left_eye_landmarks + right_eye_landmarks]) * frame_width) + 10)
            max_y = min(frame_height, int(max([landmark.y for landmark in left_eye_landmarks + right_eye_landmarks]) * frame_height) + 10)
            eye_frame = frame[min_y:max_y, min_x:max_x]

            # Save eye image in the respective folder based on which eye is blinking
            if left_eye_ratio < EYE_AR_THRESH:
                eye_image = cv2.resize(eye_frame, (100, 100))
                image_filename = os.path.join(left_blink_folder, f"left_blink_{int(time.time())}.png")
                cv2.imwrite(image_filename, eye_image)
            elif right_eye_ratio < EYE_AR_THRESH:
                eye_image = cv2.resize(eye_frame, (100, 100))
                image_filename = os.path.join(right_blink_folder, f"right_blink_{int(time.time())}.png")
                cv2.imwrite(image_filename, eye_image)

    # Display camera feed
    cv2.imshow('Eye Control', frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Exit on 'ESC' key
        break

cam.release()
cv2.destroyAllWindows()
