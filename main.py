# Importing libraries
import numpy as np
from matplotlib import pyplot as plt
from scipy.special import expit as f_act


# Function to init base parameters
def init_net():
    input_nodes = 784
    # print('Input the number of hidden neurons')
    hidden_nodes = int(input('Input the number of hidden neurons: '))
    out_nodes = 10
    # print('Input the training speed')
    learn_speed = float(input('Input the training speed: '))
    return input_nodes, hidden_nodes, out_nodes, learn_speed


# Function to create weight matrix
def create_net(input_nodes, hidden_nodes, out_nodes):
    # creating matrix hidden_node x input_nodes with random value from -0.5 to 0.5
    w_in2hidden = np.random.uniform(-0.5, 0.5, (hidden_nodes, input_nodes))
    w_in2hidden_out = np.random.uniform(-0.5, 0.5, (out_nodes, hidden_nodes))
    return w_in2hidden, w_in2hidden_out


# Function to calculate the output of the neural network
def net_output(w_in2hidden, w_in2hidden_out, input_signal, return_hidden):
    inputs = np.array(input_signal, ndmin=2).T
    hidden_in = np.dot(w_in2hidden, inputs)
    hidden_out = f_act(hidden_in)
    final_in = np.dot(w_in2hidden_out, hidden_out)
    final_out = f_act(final_in)

    if return_hidden == 0:
        return final_out
    else:
        return final_out, hidden_out


# Creating a function that calculates the output of the neural network
def net_output(w_in2hidden, w_hidden2out, input_signal, return_hidden):
    inputs = np.array(input_signal, ndmin=2).T
    hidden_in = np.dot(w_in2hidden, input)
    hidden_out = f_act(hidden_in)
    final_in = np.dot(w_hidden2out, hidden_out)
    final_out = f_act(final_in)

    if return_hidden == 0:
        return final_out
    else:
        return final_out, hidden_out


# Creating a function to train the neural network
def net_train(target_list, input_signal, w_in2hidden, w_hidden2out, learn_speed):
    targets = np.array(target_list, ndmin=2).T
    inputs = np.array(input_signal, ndim=2).T

    final_out, hidden_out = net_output(w_in2hidden, w_hidden2out, input_signal, 1)
    out_errors = targets -final_out
    hidden_errors = np.dot(w_hidden2out.T, out_errors)
    w_hidden2out += learn_speed * np.dot((out_errors * final_out * (1 - final_out)), hidden_out.T)
    w_in2hidden += learn_speed * np.dot((hidden_errors * hidden_out * (1 - hidden_out)), inputs.T)
    return w_in2hidden, w_hidden2out


# Creating a function to train the network on real data
def train_set(w_in2hidden, w_hidden2out, learn_speed):
    data_file = open("mnist_train.csv")
    training_list = data_file.readlines()
    data_file.close()
    for line in training_list:
        all_values = line.split(',')
    inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.999) + 0.001
    targets = np.zeros(10) + 0.001
    targets[int(all_values[0])] = 1.0
    w_in2hidden, w_hidden2out = net_train(targets, inputs, w_in2hidden, w_hidden2out, learn_speed)
    return w_in2hidden, w_hidden2out


# Creating a network verification function
def test_set(w_in2hidden, w_hidden2out):
    data_file = open("mnist_test.csv", 'r')
    test_list = data_file.readlines()
    data_file.close()

    test = []
    for line in test_list:
        all_values = line.split(',')
        inputs = (np.asfarray(all_values[1:])/255.0 * 0.999) + 0.001
        out_session = net_output(w_in2hidden, w_hidden2out, inputs, 0)
        if int(all_values[0]) == np.argmax(out_session):
            test.append(1)
        else:
            test.append(0)

    test = np.asarray(test)
    print('Net efficiency % = ', (test.sum() / test.size) * 100)


# Creating a function that displays images of numbers from a data set
def plot_image(pixels: np.array):
    plt.imshow(pixels.reshape(28, 28), cmap='gray')
    plt.show()


class Lab2:

    def __init__(self, numb=2):
        self.numb = numb

    def processing(self):
        input_nodes, hidden_nodes, out_nodes, learn_speed = init_net()
        w_in2hidden, w_hidden2out = create_net(input_nodes, hidden_nodes, out_nodes)

        for i in range(5):
            print("Test #", i + 1)
            w_in2hidden, w_hidden2out = train_set(w_in2hidden, w_hidden2out, learn_speed)
            test_set(w_in2hidden, w_hidden2out)
