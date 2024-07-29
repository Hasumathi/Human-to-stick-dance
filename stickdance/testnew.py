import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

cap = cv2.VideoCapture(0)  # Use the webcam
bg = cv2.VideoCapture('Black Screen (5 Minutes).mp4')

with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret_image, image = cap.read()
        ret_bg, bg_image = bg.read()

        if not ret_image or not ret_bg:
            break

        # Resize the background image to match the webcam frame size
        bg_image = cv2.resize(bg_image, (image.shape[1], image.shape[0]))

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = holistic.process(image_rgb)
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        if result.face_landmarks:
            mp_drawing.draw_landmarks(
                bg_image,
                result.face_landmarks,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())

        if result.pose_landmarks:
            mp_drawing.draw_landmarks(
                bg_image,
                result.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        combined_image = np.hstack((image_bgr, bg_image))

        cv2.imshow("Dance and Stick Video", cv2.flip(combined_image, 1))
        key_pressed = cv2.waitKey(1) & 0xFF
        if key_pressed == ord('q'):
            break

cap.release()
bg.release()
cv2.destroyAllWindows()
