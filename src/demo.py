import cv2
import torch
import torch.nn.functional as F
import time 
import torch.nn as nn
import mediapipe as mp
import numpy as np
from src.config import mp_hands, mp_drawing_styles, mp_drawing
from src.models.net import load_model
import argparse



def main(args):
    model_path = args.model_path
    
    input_size = 126
    num_layers = 2
    hidden_size = 96
    num_classes = 9
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = load_model(input_size,hidden_size,num_layers,num_classes,device,model_path)
    window_size = 30
    window = np.zeros((window_size,126))
    confidance = 0.5
    
    ##Labels 
    tag_label = {'NO GESTURE':0,'HELP':1,'YES':2,'HELLO':3,'HEY':4,'NAME':5,'MILK':6,'REPEAT':7, 'MORE':8}
    number_to_label = {v:k for k,v in tag_label.items()}

    ## For input:
    # Here path to video might be inserted
    video_source = 0  
    cap = cv2.VideoCapture(video_source)

    ## Other 
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = ''
    start_time = 0
    total=0
    last_recorded_frame_idx = 0
        


    with mp_hands.Hands(
        model_complexity=1,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
        
            fps = cap.get(cv2.CAP_PROP_FPS)
            last_recorded_frame_idx += 1
        
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

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if last_recorded_frame_idx % 30 == 0:
                total = time.time() - start_time 
                start_time = time.time()
                last_recorded_frame_idx = 15
            
                with torch.no_grad():
                    batch = np.repeat(window.reshape((-1,window_size,126)),16,axis=0)
                    prediction = net(torch.tensor(batch,dtype=torch.float32).to(device))
                    prob = torch.exp(prediction[0].max())
                    pred_idx = prediction[0].argmax().cpu().numpy()
                if prob >= confidance:
                    if len(text) > 17 :
                        text = ''
                    text += number_to_label[pred_idx[()]] + f' {prob}'
                window[:15] = window[15:]
                window[15:] = np.zeros(window[15:].shape) 
        
        
        
            if results.multi_hand_landmarks:
                for hand_no, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    if hand_no <= 1:
                        # Record keypoint coords to window
                        for i in range(21):
                            n = i*3 + hand_no*63
                            window[last_recorded_frame_idx][n] = hand_landmarks.landmark[mp_hands.HandLandmark(i).value].x
                            window[last_recorded_frame_idx][n+1] = hand_landmarks.landmark[mp_hands.HandLandmark(i).value].y
                            window[last_recorded_frame_idx][n+2] = hand_landmarks.landmark[mp_hands.HandLandmark(i).value].z
                    
                        # Draw the hand annotations on the image.
                        mp_drawing.draw_landmarks(
                            image,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style())
                    
                
            # Flip the image horizontally for a selfie-view display.
            image=cv2.flip(image,1)
            cv2.putText(image, f'{text}' , (10,90), font, 3, (0, 255, 0), 2, cv2.LINE_AA)
            # cv2.putText(image, f' FPS : {30/(2*total+1e-5)} seconds' , (10,140), font, 3, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow(f'Dynamic Gesture Recognition', image)
            
            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',type = str, required=True, help='Path to model weights')
    args = parser.parse_args()
    main(args)