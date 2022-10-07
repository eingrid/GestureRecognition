import cv2
import torch
import torch.nn.functional as F
import time 
import torch.nn as nn
import mediapipe as mp
import numpy as np
import argparse
from models.net import load_model
from runner import Runner


def main(args):
    model_path = args.model_path
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = load_model(126,96,2,9,device,model_path)
    runner = Runner(net,device)
    runner.run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',type = str, required=True, help='Path to model weights')
    args = parser.parse_args()
    main(args)