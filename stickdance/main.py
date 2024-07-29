import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp.holistic=mp.solutions.holistic

cap = cv2.VideoCapture('dance.mp4')
bg= cv2.VideoCapture('Black Screen (5 Minutes).mp4')

with mp_holistic.Holistic(
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5) as holistic:
    while cap.isOpened():
        _,image=cap.read()
        _,bg=bg.read()

        image= cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        result = holistic.process(image)
        image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(
            bg,
            result.face_landmarks,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())

        mp_drawing.draw_landmarks(
            bg,
            result.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        cv2.imshow("dance", cv2.flip(bg,1))
        key_pressed = cv2.waitKey(1) & 0xFF
        if key_pressed == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

