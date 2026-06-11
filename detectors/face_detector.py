import mediapipe as mp

mp_face = mp.solutions.face_detection

face_detector = mp_face.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.5
)