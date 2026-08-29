import os

import cv2
import mediapipe as mp
import numpy as np

ACTION_NAMES = ("eat", "idle")
DATA_PATH = os.path.join("TSL_Data")
CLIPS_PER_ACTION = 150
FRAMES_PER_CLIP = 40
FEATURES_PER_FRAME = 63


def normalize_landmarks(landmarks):
    """Wrist-relative, scale-normalized 21-landmark hand features.

    Match the same preprocessing used by the browser app for real-time inference.
    """
    if landmarks.shape != (21, 3):
        return np.zeros((21, 3), dtype=np.float32)

    wrist = landmarks[0]
    relative = landmarks - wrist
    scale = max(np.max(np.linalg.norm(relative[:, :2], axis=1)), 1e-6)
    return (relative / scale).astype(np.float32)


for action in ACTION_NAMES:
    for seq in range(CLIPS_PER_ACTION):
        os.makedirs(os.path.join(DATA_PATH, action, str(seq)), exist_ok=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
cap = cv2.VideoCapture(0)

print("Opening camera...")
cv2.waitKey(2000)

for action in ACTION_NAMES:
    for seq in range(CLIPS_PER_ACTION):
        print(f"Get ready for: {action.upper()} (Clip {seq + 1}/{CLIPS_PER_ACTION})")

        for countdown in range(3, 0, -1):
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError("Camera failed while collecting data.")
            cv2.putText(
                frame,
                f"Get ready for {action.upper()} in {countdown}...",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )
            cv2.imshow("Recorder", frame)
            cv2.waitKey(500)

        for frame_num in range(FRAMES_PER_CLIP):
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError("Camera frame capture failed during recording.")

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)

            keypoints = np.zeros((21, 3), dtype=np.float32)
            if res.multi_hand_landmarks:
                keypoints = np.asarray(
                    [[lm.x, lm.y, lm.z] for lm in res.multi_hand_landmarks[0].landmark],
                    dtype=np.float32,
                )
                keypoints = normalize_landmarks(keypoints)

            np.save(os.path.join(DATA_PATH, action, str(seq), str(frame_num)), keypoints)

            cv2.putText(
                frame,
                f"RECORDING: {action.upper()} | Clip #{seq + 1}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
            cv2.imshow("Recorder", frame)
            cv2.waitKey(30)

cap.release()
cv2.destroyAllWindows()
print("Data collection complete!")