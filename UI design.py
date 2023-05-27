import pandas as pd
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QApplication, QMainWindow, QSizePolicy
from PyQt5.QtWidgets import QLabel
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import sys

from CyberThreatDetection import upload, tfidf, eventVector, knn, randomForest, svmClassifier, naiveBayes, SVMPSO, \
    neuralNetwork, decisionTree, graph, precisiongraph, recallgraph, fmeasuregraph, cnn

app = QApplication(sys.argv)
main = QMainWindow()
main.setWindowTitle("Cyber Threat Detection Based on Artificial Neural Networks Using Event Profiles")
# main.setGeometry(5000, 5000, 5800, 2800)


font = QFont('times', 18, QFont.Bold)

button10 = QPushButton(main)
button10.setText("SVMPSO")
button10.clicked.connect(SVMPSO)
button10.setFont(font)
# button10.setGeometry(600, 1010, 620, 500)
button10.setFixedSize(350, 75)
button10.move(800, 500)
button10.setStyleSheet('QPushButton { text-align: center; }')
button10.setStyleSheet('QPushButton { background-color: black; color: white; }')

button11 = QPushButton(main)
button11.setText("NAIVE BAYES")
button11.clicked.connect(naiveBayes)
button11.setFont(font)
button11.setGeometry(60, 550, 260, 40)
button11.setFixedSize(350, 75)
button11.setStyleSheet('QPushButton { background-color: black; color: white; }')

button12 = QPushButton(main)
button12.setText("SVM CLASSIFIER")
button12.clicked.connect(svmClassifier)
button12.setFont(font)
button12.setGeometry(60, 590, 260, 40)
button12.setFixedSize(350, 75)
button12.setStyleSheet('QPushButton { background-color: black; color: white; }')

button13 = QPushButton(main)
button13.setText("RANDOM FOREST")
button13.clicked.connect(randomForest)
button13.setFont(font)
button13.setGeometry(310, 510, 260, 40)
button13.setFixedSize(350, 75)
button13.setStyleSheet('QPushButton { background-color: black; color: white; }')

button14 = QPushButton(main)
button14.setText("KNN")
button14.clicked.connect(knn)
button14.setFont(font)
button14.setGeometry(310, 550, 260, 40)
button14.setFixedSize(350, 75)
button14.setStyleSheet('QPushButton { background-color: black; color: white; }')

button15 = QPushButton(main)
button15.setText("EVENT VECTOR")
button15.clicked.connect(eventVector)
button15.setFont(font)
button15.setGeometry(310, 590, 260, 40)
button15.setFixedSize(350, 75)
button15.setStyleSheet('QPushButton { background-color: black; color: white; }')

button16 = QPushButton(main)
button16.setText("TFIDF")
button16.clicked.connect(tfidf)
button16.setFont(font)
button16.setGeometry(560, 470, 260, 40)
button16.setFixedSize(350, 75)
button16.setStyleSheet('QPushButton { background-color: black; color: white; }')

button17 = QPushButton(main)
button17.setText("UPLOAD DATASET")
button17.clicked.connect(upload)
button17.setFont(font)
button17.setGeometry(560, 510, 260, 40)
button17.setFixedSize(350, 75)
button10.move(800, 500)
button17.setStyleSheet('QPushButton { background-color: black; color: white; }')

button7 = QPushButton(main)
button7.setText("NEURAL NETWORK")
button7.clicked.connect(neuralNetwork)
button7.setFont(font)
button7.setGeometry(60, 470, 260, 40)
button7.setFixedSize(350, 75)
button7.setStyleSheet('QPushButton { background-color: black; color: white; }')

button6 = QPushButton(main)
button6.setText("DECISION TREE")
button6.clicked.connect(decisionTree)
button6.setFont(font)
button6.setGeometry(60, 430, 260, 40)
button6.setFixedSize(350, 75)
button6.setStyleSheet('QPushButton { background-color: black; color: white; }')

button4 = QPushButton(main)
button4.setText("ACCURACY GRAPH")
button4.clicked.connect(graph)
button4.setFont(font)
button4.setGeometry(60, 390, 260, 40)
button4.setFixedSize(350, 75)
button4.setStyleSheet('QPushButton { background-color: black; color: white; }')

button3 = QPushButton(main)
button3.setText("PRECISION GRAPH")
button3.clicked.connect(precisiongraph)
button3.setFont(font)
button3.setGeometry(60, 350, 260, 40)
button3.setFixedSize(350, 75)
button3.setStyleSheet('QPushButton { background-color: black; color: white; }')

button2 = QPushButton(main)
button2.setText("RECALL GRAPH")
button2.clicked.connect(recallgraph)
button2.setFont(font)
button2.setGeometry(60, 310, 260, 40)
button2.setFixedSize(350, 75)
button2.setStyleSheet('QPushButton { background-color: black; color: white; }')

button1 = QPushButton(main)
button1.setText("FMEASURE GRAPH")
button1.clicked.connect(fmeasuregraph)
button1.setFont(font)
button1.setGeometry(60, 310, 260, 40)
button1.setFixedSize(350, 75)
button1.setStyleSheet('QPushButton { background-color: black; color: white; }')

button5 = QPushButton(main)
button5.setText("CNN")
button5.clicked.connect(cnn)
button5.setFont(font)
button5.setGeometry(60, 270, 260, 40)
button5.setFixedSize(350, 75)
button5.setStyleSheet('QPushButton { background-color: black; color: white; }')


window = QWidget()
layout = QVBoxLayout()

layout.addWidget(button17)
layout.addWidget(button16)
layout.addWidget(button15)
layout.addWidget(button14)
layout.addWidget(button13)
layout.addWidget(button12)
layout.addWidget(button11)
layout.addWidget(button10)

layout.addWidget(button7)
layout.addWidget(button6)
layout.addWidget(button5)
layout.addWidget(button4)
layout.addWidget(button3)
layout.addWidget(button2)
layout.addWidget(button1)

window.setLayout(layout)
window.resize(5000, 5000)
window.setStyleSheet("background-color: grey")
window.show()
app.exec()
