import pandas as pd
import cv2
import numpy as np
import mediapipe as mp
import os.path

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# For webcam input:
video_name = './DATASET/VALIDATION_DATASET_VIDEOS/MORE.mkv'
dataset_folder = '/DATASET/VALIDATION_DATASET'
dataframe_name = 'MORE'

landmark_dataset = None
landmark_dataset_flipped = None
landmark_dataset_swap_hands = None

# Help - 1, Yes - 2, Hello - 3, HEY - 4, NAME - 5 
tag_label = {'HELP':1,'YES':2,'HELLO':3,'HEY':4,'NAME':5,'MILK':6,'REPEAT':7, 'MORE':8}
label_tag = {v:k for k,v in tag_label.items()}

label = None
for tag in tag_label:
    if tag in video_name:
      label = tag_label[tag]

if label is None:
    raise Exception('WRONG_VIDEOFILE_NAME')
sample_num = 0
write_dataset = True
labeling = False
cap = cv2.VideoCapture(video_name)
with mp_hands.Hands(
    model_complexity=1,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    
    
    gesture_landmark = np.zeros(128)
    if write_dataset:
      success, image = cap.read()
      
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      break

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    res2 = hands.process(cv2.flip(image,1))
    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
        for hand_no, hand_landmarks in enumerate(results.multi_hand_landmarks):
          for i in range(21):
            n = i*3 + hand_no*63
            if n <= 125: 
              gesture_landmark[n] = hand_landmarks.landmark[mp_hands.HandLandmark(i).value].x
              gesture_landmark[n+1] = hand_landmarks.landmark[mp_hands.HandLandmark(i).value].y
              gesture_landmark[n+2] = hand_landmarks.landmark[mp_hands.HandLandmark(i).value].z
          
          
          mp_drawing.draw_landmarks(
              image,
              hand_landmarks,
              mp_hands.HAND_CONNECTIONS,
              mp_drawing_styles.get_default_hand_landmarks_style(),
              mp_drawing_styles.get_default_hand_connections_style())
          
        gesture_landmark[126] = sample_num
        gesture_landmark[127] = label if labeling else 0
        if landmark_dataset is None and write_dataset:
          landmark_dataset = gesture_landmark
        else:
          landmark_dataset = np.vstack((landmark_dataset,gesture_landmark))
          
        if landmark_dataset_swap_hands is None and write_dataset:
              landmark_dataset_swap_hands = np.concatenate([gesture_landmark[:63],gesture_landmark[63:126],gesture_landmark[126:]])
        else:
          landmark_dataset_swap_hands = np.vstack((landmark_dataset_swap_hands,np.concatenate([gesture_landmark[:63],gesture_landmark[63:126],gesture_landmark[126:]])))
          
        

        
          
    if res2.multi_hand_landmarks:
        for hand_no, hand_landmarks in enumerate(res2.multi_hand_landmarks):
          for i in range(21):
            n = i*3 + hand_no*63
            if n <= 125: 
              gesture_landmark[n] = hand_landmarks.landmark[mp_hands.HandLandmark(i).value].x
              gesture_landmark[n+1] = hand_landmarks.landmark[mp_hands.HandLandmark(i).value].y
              gesture_landmark[n+2] = hand_landmarks.landmark[mp_hands.HandLandmark(i).value].z
              
        gesture_landmark[126] = sample_num
        gesture_landmark[127] = label if labeling else 0
        if landmark_dataset_flipped is None and write_dataset:
          landmark_dataset_flipped = gesture_landmark
        else:
          landmark_dataset_flipped = np.vstack((landmark_dataset_flipped,gesture_landmark))
          
          
          
    # Flip the image horizontally for a selfie-view display.
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, f'Label {label if labeling else 0}', (10,450), font, 3, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('MediaPipe Hands', image)
    key = cv2.waitKey(1)
    if key & 0xFF == 27:
      break
    elif key == ord(' '):
        labeling = not labeling
    ##stop/play
    elif key == ord('s'):
          write_dataset = not write_dataset
      

cap.release()


df = pd.DataFrame(landmark_dataset)
df2 = pd.DataFrame(landmark_dataset_flipped)
df3 = pd.DataFrame(landmark_dataset_swap_hands)


df.to_csv(f'.{dataset_folder}/{dataframe_name}_BASE.csv',index=False)
df2.to_csv(f'.{dataset_folder}/{dataframe_name}_FLIPPED.csv',index=False)
df3.to_csv(f'.{dataset_folder}/{dataframe_name}_SWAP_HANDS.csv',index=False)

