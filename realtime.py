import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import time

# Load model
model = tf.keras.models.load_model('sign_language_model.h5')

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Class names (must match training)
sentences = sorted(os.listdir('Dataset/train'))
print("Classes:", sentences)

RECORD_SECONDS = 4
TARGET_FRAMES = 120

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() \
        if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() \
        if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() \
        if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, lh, rh])

def resize_sequence(sequence, target_len):
    T, D = sequence.shape
    if T == target_len:
        return sequence
    idx = np.linspace(0, T-1, target_len)
    out = np.zeros((target_len, D))
    for d in range(D):
        out[:, d] = np.interp(idx, np.arange(T), sequence[:, d])
    return out

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

recording = False
frames_recorded = []
start_time = None
prediction_text = ""

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        image = frame.copy()

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = holistic.process(image_rgb)
        image_rgb.flags.writeable = True
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # Landmarks (optional)
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        if recording:
            elapsed = time.time() - start_time
            remaining = max(0, RECORD_SECONDS - elapsed)
            cv2.putText(image, f"Recording... {remaining:.1f}s", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            keypoints = extract_keypoints(results)
            frames_recorded.append(keypoints)

            if elapsed >= RECORD_SECONDS:
                recording = False
                if len(frames_recorded) > 0:
                    sequence = np.array(frames_recorded)
                    sequence = resize_sequence(sequence, TARGET_FRAMES)
                    input_data = np.expand_dims(sequence, axis=0)
                    predictions = model.predict(input_data, verbose=0)[0]

                    # ---------- DEBUG PRINTS ----------
                    print("\n--- Prediction ---")
                    # Show top-3 predictions
                    top_idx = np.argsort(predictions)[::-1][:3]
                    for i, idx in enumerate(top_idx):
                        print(f"{i+1}. {sentences[idx]}: {predictions[idx]:.4f}")
                    # Print entire probability vector
                    # print(predictions)   # Uncomment if needed
                    # Check hand detection ratio
                    hand_frames = sum(1 for f in frames_recorded if np.sum(f[132:258]) > 0)  # non-zero hand parts
                    print(f"Frames with hand data: {hand_frames}/{len(frames_recorded)}")
                    # -----------------------------------

                    predicted_idx = np.argmax(predictions)
                    confidence = predictions[predicted_idx]
                    if confidence > 0.5:
                        prediction_text = f"{sentences[predicted_idx]} ({confidence:.2f})"
                    else:
                        prediction_text = f"Low conf ({confidence:.2f}): {sentences[predicted_idx]}"
                frames_recorded = []
        else:
            if prediction_text:
                cv2.putText(image, prediction_text, (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(image, "Idle State... Press SPACE to record", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow('Sign Language Recognition', image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == 32:  # space
            if not recording:
                recording = True
                start_time = time.time()
                frames_recorded = []
                prediction_text = ""

cap.release()
cv2.destroyAllWindows()