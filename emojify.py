import cv2
import numpy as np
from classify_emojis import classify
from processing import load_model

model = load_model()
cap = cv2.VideoCapture("video.mp4")

prev_emoji_index = None  # Initialize with None for the first frame
frames = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    emoji_index = classify(frame_rgb, model)

    if emoji_index == prev_emoji_index and emoji_index is not None:
        emoji_img = cv2.imread(f"output/emoji_{emoji_index}.jpg")
        emoji_position = (10, 10)
        emoji_height, emoji_width = 300, 300

        try:
            emoji_resized = cv2.resize(emoji_img, (emoji_width, emoji_height))
            combined_frame = frame.copy()
            combined_frame[emoji_position[1]:emoji_position[1] + emoji_height, emoji_position[0]:emoji_position[0] + emoji_width] = emoji_resized
            frames.append(combined_frame)
        except:
            frames.append(frame)
    else:
        frames.append(frame)

    prev_emoji_index = emoji_index

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

numpy_frames = np.array(frames)
np.save("frames.npy", numpy_frames)