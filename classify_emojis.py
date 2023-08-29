import cv2
import mediapipe as mp
import numpy as np
import joblib

model = joblib.load("/mnt/f/emojify/model.pkl")

# Initializing Mediapipe face detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh()

# Initializing Mediapipe hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils
detect = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Preparing for Mediapipe gesture recognition
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path='gesture_recognizer.task'),
    running_mode=VisionRunningMode.IMAGE)

def num_closed_eyes(face_landmarks):
    left_eye = face_landmarks.landmark[mp_face.FACE_CONNECTIONS[2][0]]
    right_eye = face_landmarks.landmark[mp_face.FACE_CONNECTIONS[1][0]]
    return int(left_eye.y > 0.5) + int(right_eye.y > 0.5)

def classify(frame):
    hand_results = hands.process(frame)
    face_detection_results = detect.process(frame)
    face_results = face_mesh.process(frame)

    if hand_results.multi_hand_landmarks:
        return classify_hand(frame, hand_results)
    elif face_results.multi_face_landmarks:
        return classify_face(frame, face_detection_results)
    else:
        return None
    
def classify_face(frame, face_detection_results):
    bboxC = face_detection_results.detections[0].location_data.relative_bounding_box
    ih, iw, _ = frame.shape
    bbox = x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
    cv2.rectangle(frame, bbox, (0, 255, 0), 2)
        
    # Cropping out the face and preprocessing to use the model
    face_region = frame[y:y+h, x:x+w]
    resized_face = cv2.resize(face_region, (48, 48))
    grayscale_face = cv2.cvtColor(resized_face, cv2.COLOR_RGB2GRAY)
    flattened_face = grayscale_face.flatten()
    return model.predict([flattened_face])[0]

def classify_hand(frame, face_results):
    img_data = np.array(frame, dtype=np.uint8)
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_data)

    with GestureRecognizer.create_from_options(options) as recognizer:
        recognition_result = recognizer.recognize(img)
        
        # Check if gestures were recognized
        if recognition_result.gestures:
            top_gesture = recognition_result.gestures[0][0].category_name
            
            if top_gesture != "Unknown":
                return top_gesture

    return None


