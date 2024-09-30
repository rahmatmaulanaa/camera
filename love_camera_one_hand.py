import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

# Initialize Mediapipe Hands and drawing utilities
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

# Load heart images
heart_img = Image.open("heart.png")  # Whole heart image
broken_heart_img = Image.open("broken_heart.png")  # Broken heart image

# Threshold for determining hand size
distance_break_threshold = 150  # Adjust this value to determine when the heart breaks

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a selfie-view
    frame = cv2.flip(frame, 1)
    
    # Convert the frame to RGB for Mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame for hand landmarks
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw hand landmarks on the frame
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get the position of the wrist, thumb tip, and index finger tip
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            # Convert the landmarks to pixel coordinates
            h, w, c = frame.shape
            wrist_x, wrist_y = int(wrist.x * w), int(wrist.y * h)
            thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
            index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)

            # Calculate the distance between the thumb and index finger tips
            distance_between_fingers = np.linalg.norm([thumb_x - index_x, thumb_y - index_y])

            # Set the heart size based on the distance between fingers
            heart_size = int(distance_between_fingers * 2)  # Scale the heart size based on finger distance
            heart_size = max(50, min(heart_size, 400))  # Constrain the heart size between 50 and 400 pixels

            # Determine whether to show the whole heart or broken heart based on distance threshold
            if distance_between_fingers > distance_break_threshold:  # Hand is open wide, break the heart
                img_to_display = broken_heart_img.resize((heart_size, heart_size))
            else:  # Hand is not too wide, show whole heart
                img_to_display = heart_img.resize((heart_size, heart_size))

            # Convert the frame to PIL for easier manipulation
            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Place the heart at the wrist position
            heart_position = (wrist_x - heart_size // 2, wrist_y - heart_size // 2)
            img_pil.paste(img_to_display, heart_position, img_to_display)

            # Convert back to OpenCV format
            frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    # Display the resulting frame
    cv2.imshow('Love Camera - One Hand Control', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
