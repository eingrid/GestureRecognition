import cv2
import mediapipe as mp
import numpy as np



mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
landmark_dataset = None
# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    gesture_landmark = np.zeros(126)
    
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      for hand_no, hand_landmarks in enumerate(results.multi_hand_landmarks):
        print(f'HAND NUMBER: {hand_no+1}')
        print('-----------------------')
        
        for i in range(21):
            n = i*3 + hand_no*63
            gesture_landmark[n] = hand_landmarks.landmark[mp_hands.HandLandmark(i).value].x
            gesture_landmark[n+1] = hand_landmarks.landmark[mp_hands.HandLandmark(i).value].y
            gesture_landmark[n+2] = hand_landmarks.landmark[mp_hands.HandLandmark(i).value].z
            # print(f'{mp_hands.HandLandmark(i).name}:')
            # print('1')
            # print(f'{hand_landmarks.landmark[mp_hands.HandLandmark(i).value]}')
            # for j in hand_landmarks.landmark[mp_hands.HandLandmark(i).value]:
                # print(j)
            # print('2')
          
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))

    if landmark_dataset is None:
          landmark_dataset = gesture_landmark
    else:
          landmark_dataset = np.vstack((landmark_dataset,gesture_landmark))
    if cv2.waitKey(5) & 0xFF == 27:
      break
    
print(landmark_dataset.shape)
cap.release()
