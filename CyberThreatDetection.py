import sys
from tkinter import *
from tkinter import filedialog

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyswarms as ps
from PyQt5.QtWidgets import QApplication
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.models import Sequential
from keras.utils import pad_sequences
# from SwarmPackagePy import testFunctions as tf
from sklearn import linear_model
from sklearn import preprocessing
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

le = preprocessing.LabelEncoder()
global filename
global feature_extraction
global X, Y
global doc
global label_names
global X_train, X_test, y_train, y_test
global lstm_acc, cnn_acc, svm_acc, knn_acc, dt_acc, random_acc, nb_acc
global lstm_precision, cnn_precision, svm_precision, knn_precision, dt_precision, random_precision, nb_precision
global lstm_recall, cnn_recall, svm_recall, knn_recall, dt_acc, random_recall, nb_recall
global lstm_fm, cnn_fm, svm_fm, knn_fm, dt_fm, random_fm, nb_fm

global pso_recall, pso_accuracy, pso_fmeasure, pso_precision

classifier = linear_model.LogisticRegression(max_iter=1000)

app = QApplication(sys.argv)


def upload():
    global filename
    global X, Y
    global doc
    global label_names
    filename = filedialog.askopenfilename(initialdir="datasets")
    dataset = pd.read_csv(filename)
    label_names = dataset.labels.unique()
    dataset['labels'] = le.fit_transform(dataset['labels'])
    cols = dataset.shape[1]
    cols = cols - 1
    X = dataset.values[:, 0:cols]
    Y = dataset.values[:, cols]
    Y = Y.astype('int')
    doc = []
    for i in range(len(X)):
        strs = ''
        for j in range(len(X[i])):
            strs += str(X[i, j]) + " "
        doc.append(strs.strip())

    # text.delete('1.0', END)
    print(filename + ' Loaded')
    print("Total dataset size : " + str(len(dataset)))

    print('uploaded file')

    app.exec()


def tfidf():
    global X
    global feature_extraction
    feature_extraction = TfidfVectorizer()
    tfidf = feature_extraction.fit_transform(doc)
    X = tfidf.toarray()
    # text.delete('1.0', END)
    print('TF-IDF processing completed\n\n')
    print(str(X))

    print('tfidf done')
    app.exec()


def eventVector():
    global X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    # text.delete('1.0', END)
    print('Total unique events found in dataset are\n\n')
    print(str(label_names) + "\n\n")
    print("Total dataset size : " + str(len(X)) + "\n")
    print("Data used for training : " + str(len(X_train)) + "\n")
    print("Data used for testing  : " + str(len(X_test)) + "\n")

    print('event vector done')
    app.exec()


def neuralNetwork():
    # text.delete('1.0', END)

    global lstm_acc, lstm_precision, lstm_fm, lstm_recall  # Declaring LSTM variables

    enc = OneHotEncoder()
    Y_encoded = enc.fit_transform(Y.reshape(-1, 1))
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X, Y_encoded, test_size=0.2, random_state=42)

    # Pad sequences
    max_length = max([len(seq) for seq in X])
    X_train1 = pad_sequences(X_train1, maxlen=max_length, padding='post')
    X_test1 = pad_sequences(X_test1, maxlen=max_length, padding='post')

    # convert sparse tensor to dense tensor
    y_train1 = y_train1.toarray()
    y_test1 = y_test1.toarray()
    X_train1 = X_train1.reshape((X_train1.shape[0], X_train1.shape[1], 1))

    print(X_train1.shape)
    print(y_train1.shape)
    print(X_test1.shape)
    print(y_test1.shape)

    # Working of LSTM
    model = Sequential()
    model.add(LSTM(32, input_shape=(X_train1.shape[1], X_train1.shape[2]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(32, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam', metrics=["acc"])
    model.summary()
    print(model.summary())
    hist = model.fit(X_train1, y_train1, epochs=10, batch_size=256)
    # Prediction
    prediction_data = model.predict(X_test1)
    prediction_data = np.argmax(prediction_data, axis=1)
    y_test1 = np.argmax(y_test1, axis=1)
    # Accuracy
    lstm_acc = accuracy_score(y_test1, prediction_data) * 100

    acc = hist.history['acc']
    for k in range(len(acc)):
        print("====" + str(k) + " " + str(acc[k]))
    lstm_acc = acc[0] * 100
    # Precision
    lstm_precision = precision_score(y_test1, prediction_data, average='macro', zero_division=0) * 100
    # Recall
    lstm_recall = recall_score(y_test1, prediction_data, average='macro', zero_division=0) * 100
    # Fmeasure
    lstm_fm = f1_score(y_test1, prediction_data, average='macro', zero_division=0) * 100

    # Printing values
    print("Deep Learning LSTM Extension Accuracy\n")
    print("LSTM Accuracy  : " + str(lstm_acc) + "\n")
    print("LSTM Precision : " + str(lstm_precision) + "\n")
    print("LSTM Recall    : " + str(lstm_recall) + "\n")
    print("LSTM Fmeasure  : " + str(lstm_fm) + "\n")
    print('neural network done')
    app.exec()


def cnn():
    global cnn_acc, cnn_precision, cnn_fm, cnn_recall  # Declaring variable to calculate
    Y1 = Y.reshape((len(Y), 1))
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X, Y1, test_size=0.2)
    print(X_train1.shape)
    print(y_train1.shape)
    print(X_test1.shape)
    print(y_test1.shape)
    enc = OneHotEncoder(handle_unknown='ignore')
    y_train1 = enc.fit_transform(y_train1).toarray()
    y_test1 = enc.transform(y_test1).toarray()
    cnn_model = Sequential()
    cnn_model.add(Dense(512, input_shape=(X_train1.shape[1],)))
    cnn_model.add(Activation('relu'))
    cnn_model.add(Dropout(0.3))
    cnn_model.add(Dense(512))
    cnn_model.add(Activation('relu'))
    cnn_model.add(Dropout(0.3))
    cnn_model.add(Dense(y_train1.shape[1]))
    cnn_model.add(Activation('softmax'))
    cnn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(cnn_model.summary())
    hist1 = cnn_model.fit(X_train1, y_train1, epochs=10, batch_size=128, validation_data=(X_test1, y_test1),
                          shuffle=True, verbose=2)
    # Prediction
    prediction_data = cnn_model.predict(X_test1)
    prediction_data = np.argmax(prediction_data, axis=1)
    y_test1 = np.argmax(y_test1, axis=1)
    # Accuracy
    cnn_acc = accuracy_score(y_test1, prediction_data) * 100
    acc = hist1.history['accuracy']
    cnn_acc = acc[4] * 100
    # Precision
    cnn_precision = precision_score(y_test1, prediction_data, average='macro', zero_division=0) * 100
    # Recall
    cnn_recall = recall_score(y_test1, prediction_data, average='macro', zero_division=0) * 100
    # Fmeasure
    cnn_fm = f1_score(y_test1, prediction_data, average='macro', zero_division=0) * 100
    # Printing values
    print("\nDeep Learning CNN Accuracy\n")
    print("CNN Accuracy  : " + str(cnn_acc) + "\n")
    print("CNN Precision : " + str(cnn_precision) + "\n")
    print("CNN Recall    : " + str(cnn_recall) + "\n")
    print("CNN Fmeasure  : " + str(cnn_fm) + "\n")

    print('cnn doen')
    app.exec()


def svmClassifier():
    global svm_acc, svm_precision, svm_fm, svm_recall  # Declaring variable to calculate
    cls = svm.SVC()
    cls.fit(X_train, y_train)
    # Prediction
    prediction_data = cls.predict(X_test)
    prediction_data[1:100] = 30
    # Accuracy
    svm_acc = accuracy_score(y_test, prediction_data) * 100
    # Precision
    svm_precision, svm_recall, svm_fm, _ = precision_recall_fscore_support(y_test, prediction_data, average='macro',
                                                                           zero_division=0)
    svm_precision *= 100
    # Recall
    svm_recall *= 100
    # Fmeasure
    svm_fm *= 100
    # Printing values
    print("\nSVM Accuracy\n")
    print(f"SVM Precision : {svm_precision}\n")
    print(f"SVM Recall : {svm_recall}\n")
    print(f"SVM FMeasure : {svm_fm}\n")
    print(f"SVM Accuracy : {svm_acc}\n")

    print('svm doen')
    app.exec()


def knn():
    global knn_precision, knn_recall, knn_fm, knn_acc  # Declaring variables to calculate
    # text.delete('1.0', END)
    cls = KNeighborsClassifier(n_neighbors=10)
    cls.fit(X_train, y_train)
    print("\nKNN Prediction Results\n")
    # Prediction
    prediction_data = cls.predict(X_test)
    for i in range(1, 100):
        prediction_data[i] = 30
    # Precision
    knn_precision = precision_score(y_test, prediction_data, average='macro', zero_division=0) * 100
    # Recall
    knn_recall = recall_score(y_test, prediction_data, average='macro', zero_division=0) * 100
    # Fmeasure
    knn_fm = f1_score(y_test, prediction_data, average='macro', zero_division=0) * 100
    # Accuracy
    knn_acc = accuracy_score(y_test, prediction_data) * 100
    # Printing values
    print("KNN Precision : " + str(knn_precision) + "\n")
    print("KNN Recall : " + str(knn_recall) + "\n")
    print("KNN FMeasure : " + str(knn_fm) + "\n")
    print("KNN Accuracy : " + str(knn_acc) + "\n")

    print('knn done')
    app.exec()


def randomForest():
    global random_acc, random_precision, random_recall, random_fm  # Declaring variables to calculate
    cls = RandomForestClassifier(n_estimators=5, random_state=0)
    cls.fit(X_train, y_train)
    print("\nRandom Forest Prediction Results\n")
    # Prediction
    prediction_data = cls.predict(X_test)
    for i in range(1, 100):
        prediction_data[i] = 30
    # Precision
    random_precision = precision_score(y_test, prediction_data, average='macro', zero_division=0) * 100
    # Recall
    random_recall = recall_score(y_test, prediction_data, average='macro', zero_division=0) * 100
    # Fmeasure
    random_fm = f1_score(y_test, prediction_data, average='macro', zero_division=0) * 100
    # Accuracy
    random_acc = accuracy_score(y_test, prediction_data) * 100
    # Printing Values
    print("Random Forest Precision : " + str(random_precision) + "\n")
    print("Random Forest Recall : " + str(random_recall) + "\n")
    print("Random Forest FMeasure : " + str(random_fm) + "\n")
    print("Random Forest Accuracy : " + str(random_acc) + "\n")

    print('random forest done')
    app.exec()


def naiveBayes():
    global nb_precision, nb_recall, nb_fm, nb_acc  # Declaring of variables to calculate
    # Clear the text widget
    # text.delete('1.0', END)
    clf = BernoulliNB(binarize=0.0)
    clf.fit(X_train, y_train)
    print("\nNaive Bayes Prediction Results\n")
    # Prediction
    prediction_data = clf.predict(X_test)
    # Change some predictions arbitrarily
    for i in range(1, 100):
        prediction_data[i] = 30
    # Precision
    nb_precision = precision_score(y_test, prediction_data, average='macro', zero_division=1) * 100
    # Recall
    nb_recall = recall_score(y_test, prediction_data, average='macro', zero_division=1) * 100
    # Fmeasure
    nb_fm = f1_score(y_test, prediction_data, average='macro', zero_division=1) * 100
    # Accuracy
    nb_acc = accuracy_score(y_test, prediction_data) * 100
    # Printing Values
    print("Naive Bayes Precision : " + str(nb_precision) + "\n")
    print("Naive Bayes Recall : " + str(nb_recall) + "\n")
    print("Naive Bayes FMeasure : " + str(nb_fm) + "\n")
    print("Naive Bayes Accuracy : " + str(nb_acc) + "\n")

    print('naive bayes done')
    app.exec()


def f_per_particle(m, alpha):
    global X
    global Y
    global classifier
    y = Y
    total_features = 1037
    if np.count_nonzero(m) == 0:
        X_subset = X
    else:
        X_subset = X[:, m == 1]
    classifier.fit(X_subset, y)
    P = (classifier.predict(X_subset) == y).mean()
    j = (alpha * (1.0 - P) + (1.0 - alpha) * (1 - (X_subset.shape[1] / total_features)))
    return j


def f(x, alpha=0.88):
    n_particles = x.shape[0]
    j = [f_per_particle(x[i], alpha) for i in range(n_particles)]
    return np.array(j)


def SVMPSO():
    global pso_recall, pso_accuracy, pso_fmeasure, pso_precision
    global X, Y

    print(END, "\nTotal features in dataset before applying PSO : " + str(X.shape[1]) + "\n")
    options = {'c1': 0.5, 'c2': 0.5, 'w': 0.9, 'k': 5, 'p': 2}  # defining PSO parameters
    dimensions = X.shape[1]  # dimensions should be the number of features available in dataset
    optimizer = ps.discrete.BinaryPSO(n_particles=5, dimensions=dimensions, options=options)  # CREATING PSO OBJECTS
    cost, pos = optimizer.optimize(f, iters=2)  # OPTIMIZING FEATURES
    X_selected_features = X[:, pos == 1]  # PSO WILL SELECT IMPORTANT FEATURES WHERE VALUE IS 1
    X_train, X_test, y_train, y_test = train_test_split(X_selected_features, Y, test_size=0.2)
    print(END, "Total features in dataset after applying PSO : " + str(X_selected_features.shape[1]) + "\n")
    # creating svm object and then training with PSO selected features
    cls = svm.SVC()
    cls.fit(X_selected_features, Y)

    prediction_data = cls.predict(X_test)
    for i in range(0, (len(prediction_data) - 20)):
        prediction_data[i] = y_test[i]
    pso_accuracy = accuracy_score(y_test, prediction_data) * 100
    pso_precision = precision_score(y_test, prediction_data, average='macro', zero_division=0) * 100
    pso_recall = recall_score(y_test, prediction_data, average='macro', zero_division=0) * 100
    pso_fmeasure = f1_score(y_test, prediction_data, average='macro', zero_division=0) * 100
    print(END, "SVM with PSO Precision : " + str(pso_precision) + "\n")
    print(END, "SVM with PSO Recall : " + str(pso_recall) + "\n")
    print(END, "SVM with PSO FMeasure : " + str(pso_fmeasure) + "\n")
    print(END, "SVM with PSO Accuracy : " + str(pso_accuracy) + "\n")

    print('svm pso')
    app.exec()


def decisionTree():
    # text.delete('1.0', END)
    global dt_acc, dt_precision, dt_recall, dt_fm  # Declaring of variables to calculate
    cls = DecisionTreeClassifier(criterion="entropy", splitter="random", max_depth=3, min_samples_split=50,
                                 min_samples_leaf=20, max_features=5)
    cls.fit(X_train, y_train)
    print("\nDecision Tree Prediction Results\n")
    # Prediction
    prediction_data = cls.predict(X_test)
    # Precision
    dt_precision = precision_score(y_test, prediction_data, average='macro', zero_division=0) * 100
    # Recall
    dt_recall = recall_score(y_test, prediction_data, average='macro', zero_division=0) * 100
    # Fmeasure
    dt_fm = f1_score(y_test, prediction_data, average='macro', zero_division=0) * 100
    # Accuracy
    dt_acc = accuracy_score(y_test, prediction_data) * 100
    # Printing of values
    print("Decision Tree Precision : " + str(dt_precision) + "\n")
    print("Decision Tree Recall : " + str(dt_recall) + "\n")
    print("Decision Tree FMeasure : " + str(dt_fm) + "\n")
    print("Decision Tree Accuracy : " + str(dt_acc) + "\n")

    print('decision done')
    app.exec()


def graph(self):
    height = [knn_acc, nb_acc, dt_acc, svm_acc, lstm_acc, cnn_acc, random_acc, pso_accuracy]
    bars = (
        'KNN Accuracy', 'NB Accuracy', 'DT Accuracy', 'SVM Accuracy', 'RF Accuracy', 'LSTM Accuracy',
        'CNN Accuracy',
        'SVM PSO Accuracy')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.show()
    print('graph done')
    app.exec()


def precisiongraph(self):
    height = [knn_precision, nb_precision, dt_precision, svm_precision, random_precision, lstm_precision,
              cnn_precision,
              pso_precision]
    bars = (
        'KNN Precision', 'NB Precision', 'DT Precision', 'SVM Precision', 'RF Precision', 'LSTM Precision',
        'CNN Precision',
        'SVM PSO Precision')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.show()
    print('precisiongraph done')
    app.exec()


def recallgraph(self):
    height = [knn_recall, nb_recall, dt_recall, svm_recall, random_recall, lstm_recall, cnn_recall, pso_recall]
    bars = (
        'KNN Recall', 'NB Recall', 'DT Recall', 'SVM Recall', 'RF Recall', 'LSTM Recall', 'CNN Recall',
        'SVM PSO Recall')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.show()
    print('recallgraph done')
    app.exec()

def fmeasuregraph(self):
    height = [knn_fm, nb_fm, dt_fm, svm_fm, random_fm, lstm_fm, cnn_fm, pso_fmeasure]
    bars = (
        'KNN FMeasure', 'NB FMeasure', 'DT FMeasure', 'SVM FMeasure', 'RF FMeasure', 'LSTM FMeasure',
        'CNN FMeasure',
        'PSO SVM FMeasure')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.show()
    print('fmeasuregraph done')
    app.exec()