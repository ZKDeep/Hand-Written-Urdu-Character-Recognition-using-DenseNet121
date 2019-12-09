# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 01:46:43 2019

@author: zubair
"""




import numpy as np
import keras
from matplotlib import pyplot as plt
from keras import Model
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
import os
from keras.layers import Dense
from keras.layers import Activation, Flatten, GlobalAveragePooling2D
from keras.models import Sequential
from sklearn.metrics import classification_report, confusion_matrix
import numpy
import arguments
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

print("type 'train' or 'test' for training or testing")
check = input()



train_path = arguments.training_path
valid_path = arguments.validation_path
test_path = arguments.test_path


labels_reading = arguments.training_path # This will generate labels as per folders name
class_lables = os.listdir(labels_reading)


train_batches = ImageDataGenerator().flow_from_directory(train_path, target_size=(224,224), classes=class_lables, batch_size= arguments.batch_size, shuffle = True)
valid_batches = ImageDataGenerator().flow_from_directory(valid_path, target_size=(224,224), classes=class_lables, batch_size= arguments.batch_size, shuffle = True)
test_batches = ImageDataGenerator().flow_from_directory(test_path, target_size=(224,224), classes=class_lables, batch_size= arguments.batch_size, shuffle = False)

classes = len(np.unique(train_batches.classes))



dense121 = keras.applications.DenseNet121(include_top=False, weights='imagenet')

new_model=dense121.output
new_model=GlobalAveragePooling2D()(new_model)

new_model=Dense(512,activation='relu')(new_model) #dense layer 3
preds=Dense(classes,activation='softmax')(new_model) #final layer with softmax activation

model=Model(inputs=dense121.input,outputs=preds)

for i,layer in enumerate(model.layers):
  print(i,layer.name)

for layer in model.layers:
    layer.trainable=True
model.compile(optimizer='SGD',loss='categorical_crossentropy',metrics=['accuracy'])


def training():
    print("training the model")
    try:
        model.load_weights("results/weights.h5")
    except:
        print("No weights found training from scratch.....")
    
    step_size_train = train_batches.n//train_batches.batch_size
    hist = model.fit_generator(generator=train_batches, validation_data=valid_batches, 
                        validation_steps= valid_batches.n//valid_batches.batch_size,
                       steps_per_epoch=step_size_train,
                       epochs=arguments.no_epochs)
    model.save_weights("results/weights.h5")
    
    
    print("Please training results............")
    plt.plot(hist.history['acc'])
    plt.plot(hist.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('results/Acc.png')
    plt.show()
    
    
    
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('results/loss.png')
    plt.show()
    
    testing()


def testing():
    print("testing the model")
    try:
        model.load_weights("results/weights.h5")
    except:
        print("No weights found test on random weights")
    step_size_test = test_batches.n//test_batches.batch_size 
    evl = model.evaluate_generator(generator=test_batches, steps = step_size_test, verbose=1)
    Y_pred = model.predict_generator(test_batches, steps=step_size_test, verbose=1)
    
    y_pred = np.argmax(Y_pred, axis=1)
    
    dif = abs(len(y_pred) - len(test_batches.classes))
    if dif > 0:
        y_true = test_batches.classes[:-dif]
    else:
        y_true = test_batches.classes
        
    print('Confusion Matrix')
    print(confusion_matrix(y_true, y_pred))
    
    matrix = confusion_matrix(y_true, y_pred)
    sns.heatmap(matrix,annot=True,cbar=False)
    
    
    y_true = pd.Series(y_true, name="Actual")
    y_pred = pd.Series(y_pred, name="Predicted")
    df_confusion = pd.crosstab(y_true, y_pred)
    
    df_confusion.to_csv('results/confusion_matrix.csv')

    print('Classification Report')
    target_names = list((np.unique(y_true)))
    for i in range(len(target_names)):
        target_names[i] = str(target_names[i])
    print(classification_report(y_true, y_pred, target_names=target_names))
    return(evl)


if check == "train":
    training()
elif check == "test":
    print("testing")
    testing()

