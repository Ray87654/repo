import os
import cv2
import random
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool

matplotlib.rc('font', family='Microsoft JhengHei')

x_train = []
y_train = []
x_test  = []
y_test  = []

def plot_acc_loss (result):
    acc = result.history["accuracy"]
    val_acc = result.history["val_accuracy"]
    loss = result.history["loss"]
    val_loss = result.history["val_loss"]
    
    plt.figure(figsize = (20 , 20))
    plt.subplot(1, 2, 1)
    plt.title("Training and Validation Accuracy")
    plt.plot(acc, color = "green", label = "Training Accuracy")
    plt.plot(val_acc, color = "red", label = "Validation Accuracy")
    plt.legend(loc = "lower right")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    
    plt.subplot(1, 2, 2)
    plt.title("Training and Validation Loss")
    plt.plot(loss, color = "blue", label = "Training Loss")
    plt.plot(val_loss, color = "purple", label = "Validation Loss")
    plt.legend(loc = "upper right")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.show()

def load_image(n, path):
    list = os.listdir(path)
    N = len(list)
    for times in range (n * 220, (n + 1) * 220, 1):
        Word_path = path + "\\" + list[times]
        img_list = os.listdir(Word_path)
        for num in range (0, len(img_list), 1):
            img_path = Word_path + "\\" + img_list[num]
            image = cv_imread(img_path)
            if num < 10:
                x_test.append(image)
                A = [times]
                y_test.append(A)
            else :
                x_train.append(image)
                A = [times]
                y_train.append(A)
    
    return x_test,y_test,x_train,y_train

def visualizing_data(predict_test, list, test_image, test_label):
    predicition = np.argmax(predict_test, axis = 1)
    plt.figure(figsize=(15 , 15))
    for i in range (0, 9 ,1):
        plt.subplot(3 , 3 , i+1)
        n = random.randint(0 ,len(test_label))
        pred = list[predicition[n]]
        Truth = test_label[n]
        GroundTruth = list[Truth[0]]
        plt.imshow(test_image[n], cmap = plt.cm.binary)
        plt.xticks([])
        plt.yticks([])
        plt.xlabel(f"predict :{pred}\nTrue :{GroundTruth}", fontsize = 10)
    plt.show()

def check(A,B):
    n = 0
    for i in range(0,len(B),1):
        if A == B[i]: return True
        else: n += 1
    return False

def visualizing_data_forTest(predict_test, list, test_image, test_label):
    predicition = np.argmax(predict_test, axis = 1)
    # N = []
    # n = 0
    plt.figure(figsize=(15 , 15))
    # for i in range (0, 4 ,1):
        # plt.subplot(2 , 2 , i+1)
        # while check(n,N):
        #     n = random.randint(0 ,len(test_label) - 1)
    pred = list[predicition[0]]
    Truth = test_label[0]
    GroundTruth = list[Truth[0]]
    plt.imshow(test_image[0], cmap = plt.cm.binary)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel(f"predict :{pred}\nTrue :{GroundTruth}", fontsize = 10)
    # N.append(n)
    plt.show()

def cv_imread(filePath):
    cv_img = cv2.imdecode(np.fromfile(filePath, dtype = np.uint8), -1)
    return cv_img

def load(path):
    x_train = []
    y_train = []
    x_test  = []
    y_test  = []
    
    pool = Pool(6)
    
    result = pool.starmap(load_image, [(0,path), (1,path), (2,path), (3,path), (4,path),
                                       (5,path), (6,path), (7,path), (8,path), (9,path),
                                       (10,path), (11,path), (12,path), (13,path), (14,path),
                                       (15,path), (16,path) ,(17,path), (18,path), (19,path)])
    
    for i in range(0, 2, 1):
        x_test = x_test + result[i][0]
        y_test = y_test + result[i][1]
        x_train = x_train + result[i][2]
        y_train = y_train + result[i][3]
    
    num = len(x_test)
    
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    
    list = os.listdir(path)
    
    return (x_test,y_test) , (x_train,y_train), list[:num]

def Test_Dataload(path, list):
    x_test  = []
    y_test  = []
    img_list = os.listdir(path)
    for i in range(0, len(img_list), 1) :
        image_name = img_list[i]
        image_path = path + "\\" + image_name
        image = cv_imread(image_path)
        x_test.append(image)
        Locat = list.index(image_name[:1])
        A = [Locat]
        y_test.append(A)
    
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    
    return (x_test, y_test)    
