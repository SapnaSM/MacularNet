from macularnet import MacularNet
from keras.optimizers import *
import numpy as np
import cv2
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import recall_score, precision_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from scipy import interp
from Get_Available_Gpus import get_available_gpus
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

gpus = get_available_gpus(1)

with tf.device(gpus[0]):
    
    nb_class = 3
    macularnet = MacularNet()
    att_vgg_model = macularnet.model()
    opt = SGD(lr=0.01)
    att_vgg_model.compile(loss="categorical_crossentropy",
                             optimizer=opt,
                             metrics=['accuracy'])

    ##########################################################################

    def data_from_folder(folder):
        data = []
        for index, name in enumerate(os.listdir(folder)):
            train_folder = os.path.join(folder, name)
            for ii, filename in enumerate(os.listdir(train_folder)):
                im_folder = os.path.join(train_folder, filename)

                for im in os.listdir(im_folder):
                    img = cv2.imread(os.path.join(im_folder, im))
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                    img = cv2.resize(img, (512, 496))
                    img = np.array(img).reshape(496, 512)
                    img_1 = np.dstack((img, img, img))
                    img = np.array(img_1).reshape(496, 512, 3)
                    img = cv2.resize(img, (224, 224))
                    img = np.array(img).reshape(224, 224, 3)
                    if img is not None:
                        data.append((np.array(img), index))
        return data

    train_data = data_from_folder('/home/bappaditya/Sapna/Full_data/Train_set')
    train_labels = np.array([i[1] for i in train_data])
    train_labels_encoder = OneHotEncoder()
    train_labels_encoded = train_labels_encoder.fit_transform(train_labels.reshape(-1, 1)).toarray()
   
    train_images = list([i[0] for i in train_data])
    train_images_array = np.array(train_images)

    min_max_scalar = MinMaxScaler()
    train_images_array = min_max_scalar.fit(train_images_array.reshape(len(train_images), 224 * 224 * 3))
    train_images_final = min_max_scalar.transform(train_images_array.reshape(len(train_images), 224 * 224 * 3))
    train_images_final = train_images_final.reshape(len(train_images), 224, 224, 3)

    batch_size = 32
    
    image_gen = ImageDataGenerator(
        width_shift_range=0.4,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)


    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=32).split(train_images_final, train_labels))

    def get_callbacks(name_weights, patience_lr):
        mcp_save = ModelCheckpoint(name_weights, save_best_only=True, monitor='val_acc', mode='max')
        reduce_lr_loss = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=patience_lr, verbose=1, epsilon=1e-4,
                                           mode='min')
        return [mcp_save, reduce_lr_loss]


    cvscores = []
    cvp = []
    cvr = []

    for j, (train_idx, val_idx) in enumerate(folds):
        print('\nFold ', j)
        X_train_cv = train_images_final[train_idx]
        y_train_cv = train_labels[train_idx]
        train_labels_encoder = OneHotEncoder()
        y_train_cv_enc = train_labels_encoder.fit_transform(y_train_cv.reshape(-1, 1)).toarray()

        X_valid_cv = train_images_final[val_idx]
        y_valid_cv = train_labels[val_idx]
        train_labels_encoder = OneHotEncoder()
        y_valid_cv_enc = train_labels_encoder.fit_transform(y_valid_cv.reshape(-1, 1)).toarray()

        name_weights = "att_vgg_model_fold" + str(j) + "_weights.h5"
        callbacks = get_callbacks(name_weights=name_weights, patience_lr=10)
        generator = image_gen.flow(X_train_cv, y_train_cv_enc, batch_size=batch_size)

        att_vgg_model.fit_generator(
            generator,
            steps_per_epoch=len(X_train_cv) // batch_size,
            epochs=25,
            shuffle=True,
            verbose=0,
            validation_data=(X_valid_cv, y_valid_cv),
            validation_steps=len(X_valid_cv) // batch_size,
            callbacks=callbacks
        )

        score = att_vgg_model.evaluate(X_valid_cv, y_valid_cv_enc, batch_size=batch_size)
        y_predict = att_vgg_model.predict(X_valid_cv, batch_size=batch_size)
        y_pred = np.argmax(y_predict, axis=1)
        y_true = y_valid_cv
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        cc = confusion_matrix(y_true, y_pred)
        cvscores.append(score[1] * 100)
        cvp.append(precision * 100)
        cvr.append(recall * 100)

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(3):
            fpr[i], tpr[i], _ = roc_curve(y_valid_cv_enc[:, i], y_predict[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        lw = 2
        # Compute macro-average ROC curve and ROC area
        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(nb_class)]))
        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(nb_class):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])
        # Finally average it and compute AUC
        mean_tpr /= nb_class

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        print("auc=", roc_auc["macro"])
        
        np.save('/home/bappaditya/Sapna/Codes/fold_{0}_rasti_fpr.npy'.format(j),all_fpr)
        np.save('/home/bappaditya/Sapna/Codes/fold_{0}_rasti_tpr.npy'.format(j),mean_tpr)
      #  att_vgg_model.save('/home/bappaditya/Sapna/Codes/fold_{0}_vgg_att_duke_cv.hdf5'.format(j))

    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvp), np.std(cvp)))
    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvr), np.std(cvr)))
    att_vgg_model.save('/home/bappaditya/Sapna/Codes/att_vgg_model_cv.hdf5')
