from keras import callbacks
from keras.engine import training
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import tensorflow as tf
import cv2
import random
import argparse

from tensorflow import keras
from keras.layers import Dense, Input, Dropout, Flatten, BatchNormalization, RandomRotation, RandomZoom, RandomFlip, RandomTranslation
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import applications

def convert_bad_image(img):
    stream = open(img, "rb")
    bytes = bytearray(stream.read())
    nparray = np.asarray(bytes, dtype=np.uint8)
    fixed_img = cv2.imdecode(nparray, cv2.IMREAD_UNCHANGED)
    return fixed_img 

def split(dict_items, label_dict):
    X_list, y_list = [], []
    for cond, images in dict_items.items():
        for image in images:
            try:
                img = cv2.imread(str(image))
                resized_img = cv2.resize(img, (img_size, img_size))
                X_list.append(resized_img)
                y_list.append(label_dict[cond])
            except Exception as e:
                img = convert_bad_image(str(image))
                resized_img = cv2.resize(img, (img_size, img_size))
                X_list.append(resized_img)
                y_list.append(label_dict[cond])

    return np.array(X_list), np.array(y_list)

def preprocessing(model, img_size):
    print("Preparing Data... \n")
    train_dict = {
        "clean": ["rubbish_detection/train/clean/" + str(filename) for filename in os.listdir("rubbish_detection/train/clean")],
        "dirty": ["rubbish_detection/train/dirty/" + str(filename) for filename in os.listdir("rubbish_detection/train/dirty")]
    }

    test_dict = {
        "clean": ["rubbish_detection/test/clean/" + str(filename) for filename in os.listdir("rubbish_detection/test/clean")],
        "dirty": ["rubbish_detection/test/dirty/" + str(filename) for filename in os.listdir("rubbish_detection/test/dirty")]
    }

    label_dict = {
        "clean": 0,
        "dirty": 1
    }

    img_size = img_size
    X_train, y_train = split(train_dict, label_dict)
    X_test, y_test = split(test_dict, label_dict)

    print("X_train Size:", str(len(X_train)))
    print("X_test Size:", str(len(X_test)))
    print("y_train Size:", str(len(y_train)))
    print("y_test Size:", str(len(y_test)))

    X_train_scaled = X_train / 255
    X_test_scaled = X_test / 255

    print("Initialising Callbacks... \n")
    # Callbacks
    checkpoint = ModelCheckpoint("./" + model + "model.h5", monitor = 'loss', verbose = 1, save_best_only = True, mode = 'max') # check your model at each point and save the model to the specified path.

    # if the loss is not decreasing then it will stop early so that it does not overfit the model
    early_stopping = EarlyStopping(
        monitor = 'loss',
        min_delta = 0,
        patience = 3,
        verbose = 1,
        restore_best_weights = True
    )

    # reduce the learning rate when a metric has stopped improving
    reduce_learningrate = ReduceLROnPlateau(
        monitor = 'loss',
        factor = 0.2,
        patience = 3,
        verbose = 1,
        min_delta = 0.0001
    )

    callbacks_list = [early_stopping, checkpoint, reduce_learningrate]

    # Augmentation
    data_augmentation = Sequential([
        RandomRotation(factor = 0.2), 
        RandomZoom(height_factor = 0.2, width_factor = 0.2), 
        RandomFlip(mode = "horizontal_and_vertical", seed = 2021), 
        RandomTranslation(height_factor = 0.2, width_factor = 0.2),
    ])

    return X_train_scaled, y_train, X_test_scaled, y_test, callbacks_list, data_augmentation

class Experiment:
    def __init__(self, callbacks_list):
        self.num_classes = 1
        self.batch_size = 32
        self.lr = 0.0001
        self.epochs = 30
        self.callbacks = callbacks_list

    def run_pretrained_model(self, model, X_train, y_train, X_test, y_test):
        model.compile(optimizer = keras.optimizers.Adam(learning_rate = self.lr),
                    loss = 'binary_crossentropy',
                    metrics = ['accuracy'])

        model.fit(X_train, y_train, epochs=self.epochs, batch_size = self.batch_size, callbacks = self.callbacks)
        print("Evaluating Model...")
        print(model.evaluate(X_test, y_test))
        
        return model.predict(X_test)

    def get_incorrect_items(self, predictions, X_test, y_test):
        pred = list(predictions)
        true = list(y_test)

        diff = []
        for i in range(len(pred)):
            if pred[i] != true[i]:
                diff.append(i)
                
        wrong_items = random.sample(diff, 25)

        fig = plt.figure(figsize=(16, 16))
        columns = 5
        rows = 5

        count = 0
        for i in range(1, columns*rows +1):
            img = X_test[wrong_items[i - 1]]
            fig.add_subplot(rows, columns, i)
            plt.imshow(img)
            
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Running Experiment")
    parser.add_argument("--model", help="model to use", choices=["vgg16", "resnet50", "resnet50v2", "densenet201", "xception"], default = "vgg16")
    parser.add_argument("--imgsize", help="size of rescaled image", choices=[32, 64, 96, 128, 160, 192, 224], type = int, default = 192)
    parser.add_argument("--unfreeze", help="unfreeze layers for fine tuning", choices=[True, False], type = bool, default = False)
    parser.add_argument("--layers", help="number of layers to unfreeze", type = int ,default = 0)
    args = parser.parse_args()

    model_sel = args.model
    img_size = int(args.imgsize)
    unfreeze = args.unfreeze
    layer_lim = args.layers
    print("Current Model:", model_sel)
    print("Image Size:", str(img_size))
    print("Unfreeze:", str(unfreeze))
    print("Layers:", str(layer_lim))

    X_train, y_train, X_test, y_test, callbacks_list, data_augmentation = preprocessing(model_sel, img_size)

    new_input = Input(shape=(img_size, img_size, 3))

    if model_sel == "vgg16":
        model = applications.VGG16(weights = 'imagenet', include_top = False, input_tensor = new_input)
    elif model_sel == "resnet50":
        model = applications.ResNet50(weights = 'imagenet', include_top = False, input_tensor = new_input)
    elif model_sel == "resnet50v2":
        model = applications.ResNet50V2(weights = 'imagenet', include_top = False, input_tensor = new_input)
    elif model_sel == "densenet201":
        model = applications.DenseNet121(weights = 'imagenet', include_top = False, input_tensor = new_input)
    elif model_sel == "xception":
        model = applications.Xception(weights = 'imagenet', include_top = False, input_tensor = new_input)
    else:
        model = None
        print("Model not defined...")

    if model:
        model.trainable = False
        experiment = Experiment(callbacks_list)

        exp_model = Sequential([ 
            data_augmentation,
            model,

            Flatten(),
            Dense(512, activation = 'relu'),
            Dense(256, activation = 'relu'),
            BatchNormalization(),
            Dropout(0.25),

            Dense(experiment.num_classes, activation = 'sigmoid')
        ])

        prediction = experiment.run_pretrained_model(exp_model, X_train, y_train, X_test, y_test)

        if unfreeze:
            print("Unfreezing Layers... \n")
            model.trainable = True
            for layer in model.layers[:layer_lim]:
                layer.trainable = False
            
            prediction = experiment.run_pretrained_model(exp_model, X_train, y_train, X_test, y_test)

        experiment.get_incorrect_items(prediction, X_test, y_test)