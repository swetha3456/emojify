import cv2
import mediapipe as mp

# Initializing Mediapipe face detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh()

# Initializing Mediapipe hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

def num_closed_eyes(face_landmarks):
    left_eye = face_landmarks.landmark[mp_face.FACE_CONNECTIONS[2][0]]
    right_eye = face_landmarks.landmark[mp_face.FACE_CONNECTIONS[1][0]]
    return int(left_eye.y > 0.5) + int(right_eye.y > 0.5)

def classify(frame):
    hand_results = hands.process(frame)
    face_results = face_mesh.process(frame)

    if not hand_results.multi_face_landmarks:
        return classify_face(face_results)
    else:
        return classify_hand_and_face(hand_results, face_results)
    
def classify_face():
    

def classify_hand_and_face(hand_results, face_results):
    num_hands = len(hand_results.multi_hand_landmarks)
    
    if num_hands == 2:
        if 
    for hand_landmarks in hand_results.multi_hand_landmarks:
        # Detecting hand gestures
        thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
        pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
        wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

        # Check for specific hand gestures
        if all(tip.y > wrist.y for tip in [thumb_tip, index_finger_tip]):
            gesture = "Hugging Face"
        elif all(tip.y > wrist.y for tip in [thumb_tip, index_finger_tip, middle_finger_tip]):
            gesture = "Quiet Emoji"
        elif thumb_tip.y < index_finger_tip.y and middle_finger_tip.y < ring_finger_tip.y:
            gesture = "Face Cover"
        elif thumb_tip.y > index_finger_tip.y and all(tip.y > wrist.y for tip in [middle_finger_tip, ring_finger_tip, pinky_tip]):
            gesture = "Face Palm"
    

def classify_hand_gesture(frame):

    '''
    if hand_results.detections:
        for detection in hand_results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            bbox = x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                   int(bboxC.width * iw), int(bboxC.height * ih)
            
            cv2.rectangle(frame, bbox, (0, 255, 0), 2)
        
            # Cropping out the face and preprocessing to use the model
            face_region = frame[y:y+h, x:x+w]
            resized_face = cv2.resize(face_region, (48, 48))
            grayscale_face = cv2.cvtColor(resized_face, cv2.COLOR_RGB2GRAY)
            flattened_face = grayscale_face.flatten()
            predicted_label = model.predict([flattened_face])[0]
            predicted_emoji = emojis[predicted_label]
            print(predicted_emoji)
    '''
