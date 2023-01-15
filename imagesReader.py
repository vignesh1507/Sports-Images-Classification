import cv2
import random
import numpy as np
import os
import csv


# Generate Excel file
def createcsv(out):
    with open('Predictions.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(out)


Train_data_dir="Dataset/Train/"

all_images = os.listdir(Train_data_dir)
all_images = [str(Train_data_dir+path) for path in all_images]

random.seed(10)
random.shuffle(all_images)

data_size = len(all_images)

channels = 3
IMG_SIZE = 224

# Dataset/Train/basketball_1.jpg

def GetSportName(imageName):
    x = imageName.split("/")
    y = x[-1].split("_") # basketball_1.jpg
    return y[0] # basketball


def createTraindata():
    dataset = np.ndarray(shape=(len(all_images), IMG_SIZE, IMG_SIZE, channels), dtype=np.uint8)
    labels=[]
    i = 0
    for _file in all_images:
        image = cv2.imread(_file, 1) # 1 Colored , 0 Gray, -1 UNCHANGED
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        dataset[i] = image
        name = GetSportName(_file)
        labels.append(name)
        i += 1
    labels = changelabels(labels)
    return dataset, labels

# Basketball --> 0
# Football-->1
# Rowing-->2
# Swimming-->3
# Tennis-->4
# Yoga-->5
def changelabels(label):
    for i in range(len(label)):
        if label[i].lower() == "basketball":
            label[i]=np.array([1,0,0,0,0,0])
        elif label[i].lower() == "football":
            label[i]=np.array([0,1,0,0,0,0])
        elif label[i].lower() == "rowing":
            label[i]=np.array([0,0,1,0,0,0])
        elif label[i].lower() == "swimming":
            label[i]=np.array([0,0,0,1,0,0])
        elif label[i].lower() == "tennis":
           label[i]=np.array([0,0,0,0,1,0])
        elif label[i].lower() == "yoga":
           label[i]=np.array([0,0,0,0,0,1])
    label = np.array(label)
    return label


Test_data_dir="Dataset/TestData/"
Test_images = os.listdir(Test_data_dir)
Test_images = [str(Test_data_dir+path) for path in Test_images]

def createTestdata():
    DS = np.ndarray(shape=(len(Test_images), IMG_SIZE, IMG_SIZE, channels),dtype=np.uint8)
    ImageNames_ext = []
    i = 0
    for _file in Test_images:
        image = cv2.imread(_file, 1)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        DS[i] = image
        name_ext = GetSportName(_file)
        ImageNames_ext.append(name_ext)
        i += 1
    return DS, ImageNames_ext
