import requests
url = "https://github.com/mnielsen/neural-networks-and-deep-learning/raw/master/data/mnist.pkl.gz"
with open("mnist.pkl.gz", "wb") as fd:
    fd.write(requests.get(url).content)
import pickle, gzip
import numpy as np
with gzip.open("mnist.pkl.gz", "rb") as fd:
    train_set, valid_set, test_set = pickle.load(fd, encoding="latin")



train_x, train_y = train_set
valid_x, valid_y = valid_set
num_digits = 10
input_size = 784
learning_rate = 0.1

perceptrons = []

for _ in range(num_digits):
    perceptron = {
        'weights': np.random.rand(input_size),
        'bias': 0
    }
    perceptrons.append(perceptron)
def activate(weights, inputs, bias):
    preactivation = np.dot(weights, inputs) + bias
    return preactivation

def train_perceptron(perceptron, image, label):
    preactivation = activate(perceptron['weights'], image, perceptron['bias'])
    if preactivation >= 0:
        prediction = 1
    else:
        prediction = 0
    error = label - prediction
    perceptron['weights'] += learning_rate * error * image
    perceptron['bias'] += learning_rate * error

for epoch in range(10):
    for image, label in zip(train_x, train_y):
        for digit, perceptron in enumerate(perceptrons):
            if label == digit:
                target = 1
            else:
                target =0
            train_perceptron(perceptron, image, target)
def predict_digit(image):
    preactivations = [activate(perceptron['weights'], image, perceptron['bias']) for perceptron in perceptrons]
    return np.argmax(preactivations)

correct_predictions = 0
total_predictions = len(valid_x)

for image, label in zip(valid_x, valid_y):
    prediction = predict_digit(image)
    if prediction == label:
        correct_predictions += 1

accuracy = correct_predictions / total_predictions
print("Accuracy on the validation set:", accuracy)
