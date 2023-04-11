# Importing libraries
import numpy as np
from matplotlib import pyplot as plt
from scipy.special import expit as f_act


class Perzeptron:

    def __init__(self):
        self.input_nodes = 784
        self.hidden_nodes = 100
        self.out_nodes = 10
        self.learning_speed = 0.5

    # Function to create weight matrix
    def create_net(self):
        # creating matrix hidden_node x input_nodes with random value from -0.5 to 0.5
        w_in2hidden = np.random.uniform(-0.5, 0.5, (self.hidden_nodes, self.input_nodes))
        w_in2hidden_out = np.random.uniform(-0.5, 0.5, (self.out_nodes, self.hidden_nodes))
        return w_in2hidden, w_in2hidden_out

    # Function to calculate the output of the neural network
    def net_output(self, w_in2hidden, w_hidden2out, input_signal, return_hidden):
        inputs = np.array(input_signal, ndmin=2).T
        hidden_in = np.dot(w_in2hidden, inputs)
        hidden_out = f_act(hidden_in)
        final_in = np.dot(w_hidden2out, hidden_out)
        final_out = f_act(final_in)

        if return_hidden == 0:
            return final_out
        else:
            return final_out, hidden_out

    # Creating a function to train the neural network
    def net_train(self, target_list, input_signal, w_in2hidden, w_hidden2out):
        targets = np.array(target_list, ndmin=2).T
        inputs = np.array(input_signal, ndmin=2).T

        final_out, hidden_out = self.net_output(w_in2hidden, w_hidden2out, input_signal, 1)
        out_errors = targets - final_out
        hidden_errors = np.dot(w_hidden2out.T, out_errors)
        w_hidden2out += self.learning_speed * np.dot((out_errors * final_out * (1 - final_out)), hidden_out.T)
        w_in2hidden += self.learning_speed * np.dot((hidden_errors * hidden_out * (1 - hidden_out)), inputs.T)
        return w_in2hidden, w_hidden2out

    # Creating a function to train the network on real data
    def train_set(self, w_in2hidden, w_hidden2out):
        data_file = open("mnist_train.csv", "r")
        training_list = data_file.readlines()
        data_file.close()

        all_values = []
        for line in training_list:
            all_values = line.split(',')
        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.999) + 0.001
        targets = np.zeros(10) + 0.001
        targets[int(all_values[0])] = 1.0
        w_in2hidden, w_hidden2out = self.net_train(targets, inputs, w_in2hidden, w_hidden2out)
        return w_in2hidden, w_hidden2out

    # Creating a network verification function
    def test_set(self, w_in2hidden, w_hidden2out):
        data_file = open("mnist_test.csv", 'r')
        test_list = data_file.readlines()
        data_file.close()

        test = []
        for line in test_list:
            all_values = line.split(',')
            inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.999) + 0.001
            out_session = self.net_output(w_in2hidden, w_hidden2out, inputs, 0)
            if int(all_values[0]) == np.argmax(out_session):
                test.append(1)
            else:
                test.append(0)

        test = np.asarray(test)
        print('Net efficiency % = ', (test.sum() / test.size) * 100)

    # Creating a function that displays images of numbers from a data set
    def plot_image(self, pixels: np.array):
        plt.imshow(pixels.reshape(28, 28), cmap='gray')
        plt.show()

    def processing(self):
        w_in2hidden, w_hidden2out = self.create_net()

        for i in range(10):
            print("Test #", i + 1)
            w_in2hidden, w_hidden2out = self.train_set(w_in2hidden, w_hidden2out)
            self.test_set(w_in2hidden, w_hidden2out)

        data_file = open("mnist_test.csv", 'r')
        test_list = data_file.readlines()
        data_file.close()
        all_values = test_list[int(np.random.uniform(0, 9999))].split(',')
        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.999) + 0.001
        out_session = self.net_output(w_in2hidden, w_hidden2out, inputs, 0)
        print(np.argmax(out_session))
        self.plot_image(np.asfarray(all_values[1:]))


if __name__ == "__main__":
    obj = Perzeptron()
    obj.processing()
