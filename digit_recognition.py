from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np

import tensorflow as tf
import argparse
import sys

import time

import PIL.Image
from PIL import ImageOps

from scipy import misc

from tensorflow.examples.tutorials.mnist import input_data

data = input_data.read_data_sets("MNIST_data/", one_hot=True)
data.test.cls = np.argmax(data.test.labels, axis=1)

# Each image is 28x28 pixels
imageSize = 28
# 28x28 2D image flattened to 1D has size 784
inputSize = 784
# ten possible digits
numberOfClasses = 10

# First convolutional layer
filterSize1 = 5
numberOfFilters1 = 16

# Second convolutional layer
filterSize2 = 5
numberOfFilters2 = 36

# First fully connected layer
fullyConnectedSize = 128

numberOfTestsPerBatch = 256

currentIteration = 0


def optimize(numberOfIterations):
    startTime = time.time()
    global currentIteration

    for i in range(currentIteration, currentIteration + numberOfIterations):
        # Get a batch of training examples
        # xBatch holds a batch of images
        # yTrueBatch = true labels for each image in xBatch
        xBatch, yTrueBatch = data.train.next_batch(numberOfTestsPerBatch)

        # Puts each batch into a dict with our placeholders
        feedDictTrain = {x: xBatch, yTrue: yTrueBatch}

        #for i in range(784):
        #    print(xBatch[currentIteration][i], end="")
        #    if i % 28 == 0:
        #        print(" ")


        # Runs the optimizer function on the batch we just made
        # Each variable in feedDictTrain is assigned to a placeholder variable
        session.run(optimizer, feed_dict=feedDictTrain)

        # Displays the progress of our model
        if i % 50 == 0:

            currentAccuracy = session.run(accuracy, feed_dict=feedDictTrain)
            #prediction = session.run(yPredictionClass, feed_dict=feedDictTrain)
            #for i in range(10):
            #    print(prediction[i])
            print("Current iteration " + str(i) + " accuracy = " + str(currentAccuracy))

    currentIteration += numberOfIterations


def prediction():
    startTime = time.time()

    userPNG = PIL.Image.open(os.path.join('C:' + os.sep, 'Users', 'andes', 'Mumber.png')).convert('L')
    userPNG = ImageOps.invert(userPNG)
    # imageAsArray = misc.imread(userPNG)
    # np.reshape(imageAsArray, [-1, imageSize, imageSize, 1])
    from PIL import Image

    WIDTH, HEIGHT = userPNG.size

    data = list(userPNG.getdata())  # convert image data to a list of integers
    # convert that to 2D list (list of lists of integers)
    data = [data[offset:offset + WIDTH] for offset in range(0, WIDTH * HEIGHT, WIDTH)]

    for x in range(HEIGHT):
        for y in range(WIDTH):
            data[x][y] = data[x][y] / 255

    #for y in range(HEIGHT):
    #    row = (data[y][x] for x in range(WIDTH))
    #    print(' '.join('{:3}'.format(value) for value in row))
    #print(userPNG.getdata())

    xBatch = [np.reshape(data, [784])]
    for i in range(783):
        print(xBatch[0][i], end="")
        if i % 28 == 0:
            print(" ")
    yTempLabel = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    # Puts each batch into a dict with our placeholders
    feedDictTrain = {'X:0': xBatch, yTrue: yTempLabel}

    # Runs the optimizer function on the batch we just made
    # Each variable in feedDictTrain is assigned to a placeholder variable
    prediction = session.run(yPredictionClass, feed_dict=feedDictTrain)
    print(prediction[0])



def newWeights(shape):
    """
    This function is used to generate a new matrix of weights for our model
    Weights are values that affect the model's output
    Their effect is DEPENDENT of the input
    :param shape: matrix that determines the size of our weight matrix
    :return: a matrix of new weights
    """

    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


def newBiases(shape):
    """
    This function is used to return a vector of new biases for our model
    Biases are values that affect the model's output
    Their effect is INDEPENDENT of the input
    :param shape: matrix that determines the size of our bias vector
    :return: a vector of new biases
    """

    return tf.Variable(tf.constant(0.05, shape=[shape]))


def newLayer(input, numberOfInputChannels, filterSize, numberOfFilters, use2x2pooling=True):
    """
    Creates a new convolutional layer of neurons, conovlutional layers are 4 D
    Each each neuron in this new layer, the output of the last layer is taken in as input
    i.e. each neuron N can be written as a function of th previous layers output, N(o_1, o_2, . . . , o_n)
    :param input: previous convolutional layer
    :param numberOfInputChannels: number of neurons in the previous layer
    :param filterSize: width of height of our 2d filter matrix
    :param numberOfFilters:
    :param use2x2pooling: 2x2 pooling will downscale the image by set two adjacent pixels equal to their Max,
           helps reduce noise
    :return: a new layer of neurons, 4D matrix
    """

    # Filters == neurons
    # creates the shape of our filter
    shape = [filterSize, filterSize, numberOfInputChannels, numberOfFilters]

    # creates a set of weights for our filters
    weights = newWeights(shape=shape)

    # Creates a new set of biases for our filters
    biases = newBiases(shape=numberOfFilters)

    # strides = [1, y-axis movement pixels, x-axis movement pixels, 1]
    # padding='SAME' => input image is projected to the same size of the output image, i.e. padded with zeros
    layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1], padding='SAME')

    # Addes biases to the layer
    layer += biases

    if use2x2pooling:
        # 2 by 2 pooling takes a 2x2 window across the image and sets all 4 pixels equal to the max pixel value
        # Then moves 2 pixels on until the whole image has been down scaled
        layer = tf.nn.max_pool(value=layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Adds a rectified linear unit component to each neuron's output function
    # if a neuron is originally modeled by a function f(x1, x2, ... , xn) then once ReLU is added the neuron will be
    # N(x1, x2, ... , xn) = Max{f(x1, x2, ... , xn), 0}
    # This decreases the linearity of the neural network allowing for more complex modeling
    layer = tf.nn.relu(layer)

    return layer, weights


def flattenLayer(layer):
    """
    Out put of a convolutional layer is a 4D tensor
    This output needs to be flattened to 2D so that it can be taken as input to a by a fully connected layer
    :param layer: convolutional layer
    :return: convolutional layer output in 2D
    """
    # Get dimensions of the layer
    layerShape = layer.get_shape()

    numberOfFeatures = np.array(layerShape[1:4], dtype=int).prod()

    # setting first dimension to -1 allows it to be calculated to the size needed
    flatLayer = tf.reshape(layer, [-1, numberOfFeatures])

    return flatLayer, numberOfFeatures


def newFullyConnectedLayer(input, numberOfInputs, numberOfOutputs, useRelu=True):
    """
    Creates a new fully connected layer
    :param input: previous layer of the model
    :param numberOfInputs: number of inputs from the previous layer
    :param numberOfOutputs:
    :param useRelu: enables use of the ReLU activation function
    :return:
    """
    # Initalize new containers for weights and biases
    weights = newWeights(shape=[numberOfInputs, numberOfOutputs])
    biases = newBiases(shape=numberOfOutputs)

    # creates a layer that is the matrix multiplication of the input and the weights plus the biases
    layer = tf.matmul(input, weights) + biases

    # Adds ReLU activation to layer
    if useRelu:
        layer = tf.nn.relu(layer)

    return layer


def displayAccuracy():
    numberOfTests = len(data.test.images)
    classPredictions = np.zeros(shape=numberOfTests, dtype=np.int)

    i = 0
    while i < numberOfTests:
        # Index of the end of the last batch
        j = min(i + numberOfTestsPerBatch, numberOfTests)

        images = data.test.images[i:j, :]
        labels = data.test.labels[i:j, :]

        feedDict = {x: images, yTrue: labels}
        classPredictions[i:j] = session.run(yPredictionClass, feed_dict=feedDict)
        i = j

    classTrue = data.test.cls
    correct = (classTrue == classPredictions)

    correctSum = correct.sum()

    print(float(correctSum) / numberOfTests)


# Data that the model will be trained on
# data = input_data.read_data_sets('data/MNIST/', one_hot=True)

########### Module setup #############
# Our model can be thought of as graph where each neuron is a vertex and the connections are the edges
# The model consists of 5 main components
#     1.) Placeholders:      input data for the model
#     2.) Variables:         values that determine the model's output, these are updated in backpropigation
#     3.) Functions:         Mathematical functions of input values and Variables
#     4.) Cost:              Measurement of the model's output and the expected output, helps in updating Variables
#     5.) Backpropigation:   Method of updating Variables based on model's output, we will use gradient decent

# Place holders store values for later evaluation
# Helps cut down on cost by handing off large batches of evaluation rather than performing them one by one
# tf.placeholder( data type tf.DataType, matrix [ dimension size_1, dimension size_2 , . . ., dimension size_n])

# x will be the input into the model of size [Node, inputSize]
# [None, inputSize] : Node => arbitrary number of vectors, inputSize => all vectors are of length inputSize
x = tf.placeholder(tf.float32, [None, inputSize], name='X')


# Convolutional layers take in a 4D input
# xImage is x reshaped to be 4D
xImage = tf.reshape(x, [-1, imageSize, imageSize, 1])

# yTrue is a list of vectors representing the true label of the data in x
# We can compare the models output with yTrue to check accuracy of each prediction from the model
# matrix must but [None, numberOfClasses] since each digit is represented by a one hot 1x10 matrix
# i.e.  0 == [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]      7 == [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
yTrue = tf.placeholder(tf.float32, shape=[None, numberOfClasses], name='y_true')


# yTrueClass is a list of actual integers stored in our input matrix x
# We only need one dimension since we are storing integers and not vectors
yTrueClass = tf.argmax(yTrue, dimension=1)

# Creates the first convolutional layer of the network
# The network has numberOfFilters1 number of filters that all have dimension filterSize1 x filterSize1
convolutionalLayer1, convolutionalWeights1 = newLayer(input=xImage, numberOfInputChannels=1, filterSize=filterSize1,
                                                      numberOfFilters=numberOfFilters1, use2x2pooling=True)
print(convolutionalLayer1)

# Creates the second layer of the network
# This layer takes in the first layer as input
convolutionalLayer2, convolutionalWeights2 = newLayer(input=convolutionalLayer1, numberOfInputChannels=numberOfFilters1,
                                                      filterSize=filterSize2, numberOfFilters=numberOfFilters2,
                                                      use2x2pooling=True)
print(convolutionalLayer2)

# Next we flatten the the second layer so that it may be the input of the third layer, a fully connected layer
flattenedLayer, numberOfFeatures = flattenLayer(convolutionalLayer2)

# Creates the first fully connected layer, this is also the third layer of the model
fullyConnectedLayer1 = newFullyConnectedLayer(input=flattenedLayer, numberOfInputs=numberOfFeatures,
                                              numberOfOutputs=fullyConnectedSize, useRelu=True)

# Creates the second fully connected layer
# This will give us the final output of the model as a vector of length 10
fullyConnectedLayer2 = newFullyConnectedLayer(input=fullyConnectedLayer1, numberOfInputs=fullyConnectedSize,
                                              numberOfOutputs=numberOfClasses, useRelu=False)

########### END OF MODEL SETUP #############

# Gives the max value of the length 10 vector output by the model
yPrediction = tf.nn.softmax(fullyConnectedLayer2)

# Gives the predicted class
yPredictionClass = tf.argmax(yPrediction, dimension=1)

# crossEntropy is measurement of how far our model's predication is to the true classification of the input
crossEntropy = tf.nn.softmax_cross_entropy_with_logits(logits=fullyConnectedLayer2, labels=yTrue)

# cost is average error of our model's prediction
# Function that defines errors of the model
# To reduce this we calculate the gradient of the function and then follow it in the opposite direction
# This is gradient decent
cost = tf.reduce_mean(crossEntropy)

# Uses gradiant decent to update the weights and biases of the model based on the model's output compared to the true label
optimizer = tf.train.AdamOptimizer(learning_rate=1e-5).minimize(cost)

# Determines how many correct predictions were made by the model
correctPrediction = tf.equal(yPredictionClass, yTrueClass)

# Gives us the percent correct the model predicted
accuracy = tf.reduce_mean(tf.cast(correctPrediction, tf.float32))

# Creates a session with which to run our model
session = tf.Session()

# Fills our model with random values
session.run(tf.initialize_all_variables())

saver = tf.train.Saver()
# Runs the actual model

while True:
    selection = int(input("1.) train\n2.) test\n3.)exit\n"))
    if selection == 1:
        iterations = int(input("Number of training iterations: "))
        optimize(numberOfIterations=iterations)
        displayAccuracy()

    elif selection == 2:
        print("2 was selected")
        # Opens ms paint to a 28 by 28 canvas so that the user may input their own digits
        os.startfile(os.path.join('C:' + os.sep, 'Users', 'andes', 'Mumber.png'))
        x = input("enter anything once you have saved the image")
        prediction()
        print("klajlfdka")
    elif selection == 3:
        print('EXIT')
        break
