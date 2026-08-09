import cv2
import numpy as np
import os
import mediapipe as mp

ACTIONS = np.array(['eat', 'walk'])
DATA_PATH = os.path.join('TSL_Data') 

for action in ACTIONS:
    for seq in range(30):
        os.makedirs(os.path.join(DATA_PATH, action, str(seq)), exist_ok=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
cap = cv2.VideoCapture(0)

print("Opening camera...")
cv2.waitKey(2000)

for action in ACTIONS:
    for seq in range(30):
        print(f"Get ready for: {action.upper()} (Clip {seq+1}/30)")
        
        for countdown in range(3, 0, -1):
            ret, frame = cap.read()
            cv2.putText(frame, f'Get ready for {action.upper()} in {countdown}...', 
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow('Recorder', frame)
            cv2.waitKey(500)

        for frame_num in range(30):
            ret, frame = cap.read()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)

            keypoints = []
            if res.multi_hand_landmarks:
                for lm in res.multi_hand_landmarks[0].landmark:
                    keypoints.append([lm.x, lm.y, lm.z])
            else:
                keypoints = np.zeros((21, 3))

            np.save(os.path.join(DATA_PATH, action, str(seq), str(frame_num)), keypoints)

            cv2.putText(frame, f'RECORDING: {action.upper()} | Clip #{seq+1}', (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow('Recorder', frame)
            cv2.waitKey(30)

cap.release()
cv2.destroyAllWindows()
print("Data collection complete!")