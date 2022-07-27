# Dynamic Gesture Recognition
## Description
The purpose of the project is to create real-time solution to recognize gestures with oppurtunity for futher improvements to make translator from sign language to text.
Sliding-window approach with window size 30 frames was used to make gesture recognition dynamic and mediapipe to collect hands' keypoints.

Environment:
- OS. Manjaro Linux x86_64.
- Python version 3.10.5.
- CUDA release 11.7, V11.7.64.

## Metrics 
Currently model was trained and tested on 8 gestures with following metrics.

Train accuracy 98%, f1 score 97%

Confusion Matrix Train
| NO GESTURE | HELP | YES | HELLO | HEY | NAME | MILK | REPEAT | MORE |
| --- | --- |  --- | --- |  --- | --- |  --- | --- |  --- |
| 330 | 5   | 3   | 22  | 2   | 4 | 8 | 1   | 3   |
| 2   | 366 | 0   | 0   | 0   | 0 | 0 | 0   | 0   |
| 1   | 0   | 323 | 0   | 0   | 0 | 0 | 0   | 0   |
| 4   | 0   | 0   | 330 | 0   | 0 | 0 | 0   | 0   |
| 1   | 0   | 0   | 0   | 343 | 0 | 0 | 0   | 0   |
| 0   | 0   | 0   | 0   | 0   |351| 0 | 0   | 0   |
| 0   | 0   | 0   | 0   | 0   | 0 |348| 0   | 0   |
| 0   | 0   | 0   | 0   | 0   | 0 | 0 | 333 | 0   |
| 0   | 0   | 0   | 0   | 0   | 0 | 0 | 0   | 356 |

Test accuracy 77%, f1 score 55%

Confusion Matrix Test
| NO GESTURE | HELP | YES | HELLO | HEY | NAME | MILK | REPEAT | MORE |
| --- | --- |  --- | --- |  --- | --- |  --- | --- |  --- |
| 125 | 7   | 32  | 22  | 7   | 18| 11| 40  | 10  |
| 1   | 72  | 0   | 0   | 0   | 1 | 1 | 0   | 0   |
| 7   | 0   | 128 | 0   | 0   | 0 | 0 | 0   | 0   |
| 7   | 0   | 0   | 147 | 0   | 0 | 0 | 0   | 0   |
| 11  | 0   | 0   | 11  | 83  | 0 | 0 | 0   | 0   |
| 7   | 2   | 0   | 0   | 0   |141| 2 | 1   | 0   |
| 5   | 24  | 0   | 0   | 0   | 3 |122| 0   | 0   |
| 29  | 0   | 5   | 0   | 0   | 1 | 0 | 90  | 2   |
| 16  | 0   | 0   | 0   | 0   | 0 | 0 | 0   | 74  |


So as we see model has some troubles with distinguishing no gesture with gestures what might be a drawback of a proposed labeling method.
## Example

## Dataset
Dataset used to train/evaluate model was created through recording videos for each of 8 gesture and then labeled using create_ds_from_camera.py script.
For each gesture there was recorded 3 videos for train, validation and test datasets.
For train dataset lenght of video is roughly 3 mins and both for validation and test 1 min.

Dataset is saved as csv files for each video, also data augmentation (horizontal flip, change of hands order detection) was used.

## Usage
To train model run :

```shell
python train.py -train_dataset path_to_folder_with_train_dataset -valid_dataset path_to_folder_with_valid_dataset -epoch_number epoch_number -unbalanced_dataset True
```
Params explanation:

train_dataset - path to folder with train dataset.

valid_dataset (optional) - path to folder with validation dataset.

epoch_number (optional) - number of epochs, default to 100.

unbalanced_dataset (optional) - if dataset is balanced then use False, else use True to undersample train dataset using WeightedRandomSampler. 

To run demo :

```shell
python demo.py -model_path path_to_model_weights
```