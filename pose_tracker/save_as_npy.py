import cv2
import mediapipe as mp
import numpy as np
import os
from tqdm import tqdm

# ==== Setup ==== #
input_video_path = "/home/pghosh/Downloads/valentina_talking.mp4"  # <-- Replace this
output_dir = "../assets/halfbody_demo/from_vids/02"  # Output directory for frame-wise npy files
os.makedirs(output_dir, exist_ok=True)

# ==== MediaPipe Setup ==== #
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(static_image_mode=False,
                                max_num_hands=2,
                                min_detection_confidence=0.5,
                                min_tracking_confidence=0.5)

# ==== Video Setup ==== #
cap = cv2.VideoCapture(input_video_path)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# ==== Process Each Frame ==== #
for frame_index in tqdm(range(frame_count), desc="Extracting hand pose"):
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(frame_rgb)

    # Prepare hand keypoints and scores
    hands_xy = np.zeros((2, 21, 2), dtype=np.float32)
    hands_score = np.ones((2, 21), dtype=np.float32)  # Default to full confidence

    if results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            if i >= 2:
                break  # Only two hands
            for j, lm in enumerate(hand_landmarks.landmark):
                hands_xy[i, j] = [lm.x, lm.y]
                hands_score[i, j] = 1.0  # MediaPipe Hands does not return confidence, so we default to max

    # Compose the Echo-Mimic v2 format
    pose_dict = {
        'bodies': {
            'candidate': np.zeros((18, 2), dtype=np.float32),
            'subset': np.full((1, 18), -1.0, dtype=np.float32),
            'score': np.zeros((1, 18), dtype=np.float32),
        },
        'hands': hands_xy,
        'hands_score': hands_score,
        'faces': np.zeros((1, 68, 2), dtype=np.float32),
        'faces_score': np.zeros((1, 68), dtype=np.float32),
        'num': 0,
        'draw_pose_params': [768, 768, 0, 768, 0, 768],
    }

    # Save as .npy file
    np.save(os.path.join(output_dir, f"{frame_index}.npy"), pose_dict)

cap.release()
hands_detector.close()

print(f"✅ All frames processed and saved to {output_dir}")
