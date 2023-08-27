import cv2
from classify_emojis import classify

# Capturing webcam feed
cap = cv2.VideoCapture("video.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Converting the frame to RGB as Mediapipe processes RGB images
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    emoji_index = classify(frame_rgb)

    if emoji_index is not None:
        emoji_img = cv2.imread(f"output/emoji_{emoji_index}.jpg")
        emoji_position = (10, 10)
        emoji_height, emoji_width = 300, 300
        emoji_resized = cv2.resize(emoji_img, (emoji_width, emoji_height))

        combined_frame = frame.copy()
        combined_frame[emoji_position[1]:emoji_position[1] + emoji_height, emoji_position[0]:emoji_position[0] + emoji_width] = emoji_resized
        cv2.imshow('Emojify', combined_frame)

    else:
        cv2.imshow('Emojify', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()