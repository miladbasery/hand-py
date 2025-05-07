import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

capture = cv2.VideoCapture(0)

with mp_hands.Hands(
    min_detection_confidence=0.8,
    min_tracking_confidence=0.5
) as hands:
    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            print("دوربین پیدا نشد.")
            break

        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(
                        color=(255, 0, 255),  
                        thickness=4,
                        circle_radius=2
                    ),
                    mp_drawing.DrawingSpec(
                        color=(20, 180, 90),
                        thickness=2,
                        circle_radius=2
                    )
                )

        cv2.imshow('milad webcam', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

capture.release()
cv2.destroyAllWindows()