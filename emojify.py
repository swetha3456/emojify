import cv2
import mediapipe as mp
from emoji import emojize
import joblib

model = joblib.load("/mnt/f/emojify/model.pkl")

# Initializing Mediapipe face detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Creating a mapping of emojis to indices
emoji_mapping = ["😡", "🤢", "😨", "😄", "😐", "😢", "😮"]

# Capturing webcam feed
cap = cv2.VideoCapture("/mnt/f/emojify/video.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Converting the frame to RGB as Mediapipe processes RGB images
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detecting faces in the frame
    results = face_detection.process(frame_rgb)

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            bbox = x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                   int(bboxC.width * iw), int(bboxC.height * ih)
            
            cv2.rectangle(frame, bbox, (0, 255, 0), 2)
            '''
            # Cropping out the face and preprocessing to use the model
            face_region = frame[y:y+h, x:x+w]
            resized_face = cv2.resize(face_region, (48, 48))
            grayscale_face = cv2.cvtColor(resized_face, cv2.COLOR_RGB2GRAY)
            flattened_face = grayscale_face.flatten()
            predicted_label = model.predict([flattened_face])[0]
            predicted_emoji = emoji_mapping[predicted_label]
            print(predicted_emoji)
            '''
            # Display the emoji on the frame
            cv2.putText(frame, "", (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Emojify', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()