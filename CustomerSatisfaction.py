import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Flatten, Conv1D, MaxPool1D

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split

def model_train(epochs):
    from imblearn.over_sampling import SMOTE
    train=pd.read_csv("./dataset/train.csv")
    test=pd.read_csv("./dataset/test.csv")

    y_train_full=train['TARGET']
    x_train_full=train.drop(['ID', 'TARGET'], axis=1)
    x_test_final=test.drop(['ID'], axis=1)

    smt=SMOTE()
    x_train_full, y_train_full = smt.fit_resample(x_train_full, y_train_full)

    x_train, x_test, y_train, y_test = train_test_split(x_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full)

    quasi_filter=VarianceThreshold(0.01)
    x_train=quasi_filter.fit_transform(x_train)
    x_test=quasi_filter.transform(x_test)
    x_test_final=quasi_filter.transform(x_test_final)

    x_train_T=x_train.T
    x_test_T = x_test.T
    x_test_final_T=x_test_final.T
    x_train_T=pd.DataFrame(x_train_T)
    x_test_T=pd.DataFrame(x_test_T)
    x_test_final_T=pd.DataFrame(x_test_final_T)

    duplicated_features=x_train_T.duplicated()

    features_to_keep=[not index for index in duplicated_features]
    x_train=x_train_T[features_to_keep].T
    x_test=x_test_T[features_to_keep].T
    x_test_final=x_test_final_T[features_to_keep].T

    sc=StandardScaler()
    x_train_tx=sc.fit_transform(x_train)
    x_test_tx=sc.transform(x_test)
    x_test_final_tx=sc.transform(x_test_final)

    y_train=y_train.to_numpy()
    y_test=y_test.to_numpy()

    x_train_tx=x_train_tx.reshape(x_train_tx.shape[0], x_train_tx.shape[1], 1)
    x_test_tx=x_test_tx.reshape(x_test_tx.shape[0], x_test_tx.shape[1], 1)
    x_test_final_tx=x_test_final_tx.reshape(x_test_final_tx.shape[0], x_test_final_tx.shape[1], 1)
    model=Sequential()
    model.add(Conv1D(32, 3, activation='relu', input_shape=x_train_tx[0].shape))
    model.add(BatchNormalization())
    model.add(MaxPool1D(pool_size=2))
    model.add(Dropout(0.2))

    model.add(Conv1D(64, 3, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool1D(pool_size=2))
    model.add(Dropout(0.3))          

    model.add(Conv1D(128, 3, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool1D(pool_size=2))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))

    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.4))

    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(x_train_tx, y_train, validation_data=(x_test_tx, y_test), epochs=epochs, verbose=1)
    score = model.evaluate(x_test, y_test, verbose=0)
    a=score[1]*100
    model.save("CUSTOMER_SATISFACTION.h5")
    os.system("mv /CUSTOMER_SATISFACTION.h5 /mycode")
    return a

no_epoch=1
accuracy_train_model=model_train(no_epoch)
f = open("accuracy.txt","w+")
f.write(str(accuracy_train_model))
f.close()
os.system("mv /accuracy.txt /dataset")
