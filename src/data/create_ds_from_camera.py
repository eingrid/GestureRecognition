import pandas as pd
import cv2
import numpy as np
import mediapipe as mp
import os.path
import argparse
import sys
sys.path.append(os.path.abspath('/mnt/Dev/WORK/hand_gesture'))
from src.runner import Runner 



def main(args):
	video_name = args.video_path
	dataset_folder = args.save_folder
	dataframe_name = args.save_name

	runner = Runner(video_source=video_name,net = None, device = None)
	runner.create_ds(video_name,dataset_folder,dataframe_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path',type = str, required=True, help='Video input which  going to be labeled')
    parser.add_argument('--save_folder',type = str, required=True, help='Path where to save csv file')
    parser.add_argument('--save_name',type = str, required=True, help='Name of saved csv file')
    args = parser.parse_args()
    main(args)