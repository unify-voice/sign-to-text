import os
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


def mediapipe_detection(image, model):
    """Run MediaPipe Holistic on an image."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def extract_keypoints(results):
    """
    Extract pose, left hand, right hand landmarks.
    Returns: numpy array of shape (258,)
    pose: 33*4=132, lh: 21*3=63, rh: 21*3=63, total=258
    """
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() \
        if results.pose_landmarks else np.zeros(33 * 4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() \
        if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() \
        if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, lh, rh])


def resize_sequence(sequence, target_len):
    """
    Interpolate a sequence of keypoints to a fixed number of frames.
    sequence shape: (T, 258) -> (target_len, 258)
    """
    T, D = sequence.shape
    if T == target_len:
        return sequence
    idx = np.linspace(0, T - 1, target_len)
    out = np.zeros((target_len, D))
    for d in range(D):
        out[:, d] = np.interp(idx, np.arange(T), sequence[:, d])
    return out


# Folder paths
DATA_PATH = os.path.join('Dataset')
save_path = os.path.join('processed_data')
os.makedirs(save_path, exist_ok=True)

# Labels mapping
sentences = sorted(os.listdir(os.path.join(DATA_PATH, 'train')))  # sentence_1 .. sentence_10
label_map = {name: idx for idx, name in enumerate(sentences)}
print("Label map:", label_map)

# Target frames (4 seconds at 30 fps = 120)
TARGET_FRAMES = 120

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for split in ['train', 'val', 'test']:
        sequences, labels = [], []
        split_path = os.path.join(DATA_PATH, split)
        for sentence in sentences:
            class_idx = label_map[sentence]
            video_folder = os.path.join(split_path, sentence)
            if not os.path.isdir(video_folder):
                continue
            videos = [v for v in os.listdir(video_folder) if v.endswith('.mp4')]
            for vid in tqdm(videos, desc=f"Processing {split}/{sentence}"):
                cap = cv2.VideoCapture(os.path.join(video_folder, vid))
                frames_keypoints = []
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    # Optionally resize to 640x480 for consistency
                    frame = cv2.resize(frame, (640, 480))
                    image, results = mediapipe_detection(frame, holistic)
                    keypoints = extract_keypoints(results)
                    frames_keypoints.append(keypoints)
                cap.release()

                if len(frames_keypoints) == 0:
                    continue
                sequence = np.array(frames_keypoints)  # shape (T, 258)
                # Resize to TARGET_FRAMES
                sequence = resize_sequence(sequence, TARGET_FRAMES)
                sequences.append(sequence)
                labels.append(class_idx)

        # Save as numpy arrays
        X = np.array(sequences)  # (num_samples, 120, 258)
        y = np.array(labels)  # (num_samples,)
        np.save(os.path.join(save_path, f'{split}_X.npy'), X)
        np.save(os.path.join(save_path, f'{split}_y.npy'), y)
        print(f"{split}: X shape {X.shape}, y shape {y.shape}")