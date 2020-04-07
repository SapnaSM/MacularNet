from macularnet import MacularNet
from keras.optimizers import *
import numpy as np
import cv2
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import recall_score, precision_score, confusion_matrix, classification_report
from keras.preprocessing.image import ImageDataGenerator
from Get_Available_Gpus import get_available_gpus
import tensorflow as tf

gpus = get_available_gpus(1)

with tf.device(gpus[0]):

    macularnet = MacularNet()
    att_vgg_model = macularnet.model()
    opt = SGD(lr=0.001)
    att_vgg_model.compile(loss="categorical_crossentropy",
              optimizer=opt,
              metrics=['accuracy'])


    def data_from_folder(folder):
        images = []
        for index, name in enumerate(os.listdir(folder)):
            print(name)
            train_folder=os.path.join(folder, name)          
            for im in os.listdir(train_folder):
                img=cv2.imread(os.path.join(train_folder,im))
                img=cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                img=cv2.resize(img,(512,496))
                img=np.array(img).reshape(496,512)
                img_1=np.dstack((img,img,img))
                img=np.array(img_1).reshape(496,512,3)
                img= cv2.resize(img,(224,224))
                img=np.array(img).reshape(224,224,3)
    
                if img is not None:
                    images.append((np.array(img),index))
                                                   
        return images
    
    
    train_data = data_from_folder('/home/bappaditya/Sapna/Duke_dataset/train')
    train_labels= np.array([i[1] for i in train_data])
    one_hot_encoder = OneHotEncoder()
    train_labels_one_hot = one_hot_encoder.fit_transform(train_labels.reshape(-1,1)).toarray()
    
    train_images = list([i[0] for i in train_data])
    train_images_array = np.array(train_images)
    '''
    val_data = data_from_folder('/home/bappaditya/Sapna/OCT2017/val')
    val_labels= np.array([i[1] for i in val_data])
    val_labels_one_hot = one_hot_encoder.fit_transform(val_labels.reshape(-1,1)).toarray()
    
    val_images = list([i[0] for i in val_data])
    val_images_array = np.array(val_images)
    '''
    
    test_data = data_from_folder('/home/bappaditya/Sapna/Duke_dataset/test')
    test_labels = np.array([i[1] for i in test_data])
    test_labels_one_hot = one_hot_encoder.fit_transform(test_labels.reshape(-1,1)).toarray()
    
    test_images = list([i[0] for i in test_data])
    test_images_array = np.array(test_images)
    
    batch_size = 64
    
    X_train, X_val, y_train, y_val = train_test_split(train_images_array, train_labels_one_hot, test_size=0.2)
    
    train_datagen = ImageDataGenerator(
        width_shift_range = 40,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True)
    
    train_generator = train_datagen.flow(
        X_train, y_train,
        batch_size = batch_size)
    
    def get_callbacks(name_weights, patience_lr):
        mcp_save = ModelCheckpoint(name_weights, save_best_only=True, monitor='val_acc', mode='max')
        reduce_lr_loss = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=patience_lr, verbose=1, epsilon=1e-4, mode='min')
        return [mcp_save, reduce_lr_loss]
    
    name_weights = "final_weights.h5"
    callbacks = get_callbacks(name_weights = name_weights, patience_lr=5)
    
    att_vgg_model.fit_generator(
        train_generator,
        steps_per_epoch = len(X_train)//batch_size,
        epochs=100,
        validation_data= (X_val, y_val),
        validation_steps=len(X_val)//batch_size, 
        callbacks=callbacks, 
        shuffle=True)
    
    score = att_vgg_model.evaluate(test_images_array, test_labels_one_hot, batch_size = batch_size)
    y_predict = att_vgg_model.predict(test_images_array, batch_size = batch_size)
    y_pred = np.argmax(y_predict, axis=1)
    y_true = test_labels
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    cc = confusion_matrix(y_true, y_pred)
    
    print("Accuracy = " + format(score[1]*100, '.2f') + "%")
    print("Precision: ", precision*100)
    print("Recall: ", recall*100)
    target_names = ['class 0', 'class 1', 'class 2']
    print(classification_report(y_true, y_pred, target_names = target_names))